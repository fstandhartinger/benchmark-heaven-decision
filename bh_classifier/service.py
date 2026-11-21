from __future__ import annotations

import concurrent.futures
import hmac
import json
import os
import time
import threading
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer

from . import logkit
from .policy import FAST_USD_PER_1000, GEMINI_MODEL, estimated_gemini_cost, should_escalate, temper
from .upstreams import UpstreamError, classifier_fast, gemini_decision

REQUEST_SLOTS = threading.BoundedSemaphore(int(os.environ.get("MAX_CONCURRENT_REQUESTS", "16")))
MAX_INPUTS = int(os.environ.get("MAX_INPUTS_PER_REQUEST", "200"))


class BoundedHTTPServer(TCPServer):
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, address, handler, workers=16):
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self._slots = threading.BoundedSemaphore(workers)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        self._slots.acquire()
        self._pool.submit(self._run_request, request, client_address)

    def _run_request(self, request, client_address):
        try:
            self.finish_request(request, client_address)
            self.shutdown_request(request)
        except Exception:
            self.handle_error(request, client_address)
            self.shutdown_request(request)
        finally:
            self._slots.release()


def classify(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    inputs = payload.get("inputs")
    if inputs is None and "input" in payload:
        inputs = [payload["input"]]
    labels = payload.get("labels")
    instructions = str(payload.get("instructions") or "")
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= MAX_INPUTS or not all(isinstance(x, str) and len(x) <= 32000 for x in inputs):
        raise ValueError(f"inputs must contain 1..{MAX_INPUTS} strings of at most 32,000 characters")
    if not isinstance(labels, list) or not 2 <= len(labels) <= 100 or len(set(labels)) != len(labels) or not all(isinstance(x, str) and 0 < len(x) <= 200 for x in labels):
        raise ValueError("labels must contain 2..100 unique non-empty strings of at most 200 characters")
    if len(instructions) > 20000:
        raise ValueError("instructions must be at most 20,000 characters")
    started = time.monotonic()
    primary, primary_latency = classifier_fast(inputs, labels, instructions)
    final = []
    gated = []
    for index, row in enumerate(primary):
        raw = row["scores"]
        calibrated = temper(raw)
        item = {"label": max(calibrated, key=calibrated.get), "confidence": max(calibrated.values()),
                "scores": calibrated, "model": "classifier.dev/jev+temperature", "escalated": False,
                "ms": round(float(row.get("ms") or primary_latency * 1000 / len(inputs)), 2)}
        final.append(item)
        if should_escalate(raw):
            gated.append(index)
    failures = 0
    retry_attempts = 0
    failed_cost_unknown = 0
    gemini_cost = 0.0
    if gated:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(gated))) as pool:
            futures = {pool.submit(gemini_decision, inputs[i], labels, instructions): i for i in gated}
            for future, index in [(f, futures[f]) for f in futures]:
                try:
                    decision, latency = future.result()
                    gemini_cost += estimated_gemini_cost(decision["usage"])
                    retry_attempts += max(0, int(decision.get("attempts", 1)) - 1)
                    final[index] = {"label": decision["label"], "confidence": max(decision["scores"].values()),
                                    "scores": decision["scores"], "model": GEMINI_MODEL, "escalated": True,
                                    "ms": round((primary_latency + latency) * 1000, 2), "usage": decision["usage"]}
                except UpstreamError as exc:
                    failures += 1
                    retry_attempts += max(0, int(exc.attempts) - 1)
                    gemini_cost += estimated_gemini_cost(exc.usage)
                    if not exc.usage:
                        failed_cost_unknown += 1
                    final[index]["ms"] = round(final[index]["ms"] + exc.elapsed_s * 1000, 2)
    elapsed = time.monotonic() - started
    known_cost = len(inputs) / 1000 * FAST_USD_PER_1000 + gemini_cost
    return {"tier": "bh-smart-v1", "model": "bh-confidence-cascade-v1",
            "modelsUsed": sorted(set(x["model"] for x in final)), "results": final,
            "usage": {"classifications": len(inputs), "escalated": sum(x["escalated"] for x in final),
                      "escalation_failed": failures, "ms": round(elapsed * 1000, 2),
                      "gemini_retry_attempts": retry_attempts,
                      "failed_attempts_with_unknown_provider_cost": failed_cost_unknown,
                      "known_cost_usd": known_cost,
                      "estimated_cost_usd": None if failed_cost_unknown else known_cost}}


def systemone(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    state = payload.get("state")
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    if len(text) > 32000:
        raise ValueError("serialized state must be at most 32,000 characters")
    questions = payload.get("questions") or {}
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 8:
        raise ValueError("questions must be an object with 1..8 entries")
    answers = {}
    total_cost = 0.0
    cost_unknown = False
    for name, question in questions.items():
        if not isinstance(question, dict) or not isinstance(name, str) or len(name) > 200:
            raise ValueError("each question must have a short string name and object value")
        qtype = question.get("type")
        criteria = question.get("criteria")
        instructions = str(question.get("instructions") or "").strip()
        if qtype == "noul":
            labels = ["yes", "no"]
            criteria = criteria or {}
            instructions += ("\n\nOptions:\n- yes: " + str(criteria.get("true") or "yes") +
                             "\n- no: " + str(criteria.get("false") or "no"))
        elif qtype == "choice" and isinstance(criteria, dict):
            labels = list(criteria)
            if any(criteria.values()):
                instructions += "\n\nOptions:\n" + "\n".join(f"- {k}: {v or k}" for k, v in criteria.items())
        elif qtype == "score" and isinstance(criteria, list):
            labels = [str(i) for i in range(len(criteria))]
            instructions += "\n\nLevels, lowest to highest:\n" + "\n".join(f"- {i}: {v}" for i, v in enumerate(criteria))
        else:
            raise ValueError(f"unsupported question {name}")
        out = classify({"inputs": [text], "labels": labels, "instructions": instructions})
        result = out["results"][0]
        if out["usage"]["estimated_cost_usd"] is None:
            cost_unknown = True
            total_cost += out["usage"]["known_cost_usd"]
        else:
            total_cost += out["usage"]["estimated_cost_usd"]
        if qtype == "noul":
            answers[name] = {"type": "noul", "noul": result["scores"]["yes"]}
        elif qtype == "choice":
            answers[name] = {"type": "choice", "choice": result["label"], "probabilities": result["scores"]}
        else:
            answers[name] = {"type": "score", "score": int(result["label"]), "probabilities": result["scores"]}
    return {"model": "bh-confidence-cascade-v1", "answers": answers,
            "usage": {"known_cost_usd": total_cost, "estimated_cost_usd": None if cost_unknown else total_cost}}


class Handler(logkit.HTTPLogMixin, BaseHTTPRequestHandler):
    # Structured access lines (logging standard, AGENTS.md); request bodies and query strings are never logged, no IPs.
    logkit_ip = False
    logkit_ua = False

    server_version = "BHDecision/0.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(60)

    def send_json(self, status, data, headers=None):
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.send_header("access-control-allow-origin", "*")
        for key, value in (headers or {}).items():
            self.send_header(key, str(value))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            return self.send_json(200, {"ok": True, "version": "0.1.0"})
        if self.path == "/models":
            return self.send_json(200, {"data": [{"id": "bh-confidence-cascade-v1", "status": "live",
                                                   "primary": "classifier.dev fast (Jev)", "secondary": GEMINI_MODEL,
                                                   "gate": "confidence < 0.85", "calibrated_coverage": 1.0}]})
        return self.send_json(404, {"error": "not_found"})

    def do_POST(self):
        if not REQUEST_SLOTS.acquire(blocking=False):
            return self.send_json(503, {"error": "busy", "code": "capacity_exceeded"}, {"Retry-After": "1"})
        try:
            service_key = os.environ.get("SERVICE_API_KEY")
            supplied = self.headers.get("authorization", "")
            if service_key and not hmac.compare_digest(supplied, f"Bearer {service_key}"):
                return self.send_json(401, {"error": "unauthorized", "code": "unauthorized"})
            length = int(self.headers.get("content-length", "0"))
            if length > 8_000_000:
                raise ValueError("request too large")
            body = json.loads(self.rfile.read(length))
            started = time.monotonic()
            if self.path in ("/", "/v1/classify", "/v1/classify/batch"):
                out = classify(body)
            elif self.path == "/v1/systemone":
                out = systemone(body)
            else:
                return self.send_json(404, {"error": "not_found"})
            headers = {"X-Jev-Provider": "Benchmark Heaven", "X-Jev-Model": "bh-confidence-cascade-v1",
                       "X-Jev-Latency-Ms": round((time.monotonic() - started) * 1000, 2), "X-Jev-No-Markup": "true"}
            cost = out.get("usage", {}).get("estimated_cost_usd")
            if cost is not None:
                headers["X-Jev-Cost-Usd"] = cost
            return self.send_json(200, out, headers)
        except (ValueError, json.JSONDecodeError) as exc:
            return self.send_json(400, {"error": str(exc), "code": "invalid_request"})
        except UpstreamError as exc:
            return self.send_json(502, {"error": str(exc), "code": "upstream_error"})
        except (AttributeError, TypeError) as exc:
            return self.send_json(400, {"error": f"invalid request shape: {exc}", "code": "invalid_request"})
        finally:
            REQUEST_SLOTS.release()



def main():
    logkit.setup(app="benchmark-heaven-decision")
    port = int(os.environ.get("PORT", "8080"))
    workers = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "16"))
    BoundedHTTPServer(("0.0.0.0", port), Handler, workers).serve_forever()


if __name__ == "__main__":
    main()
