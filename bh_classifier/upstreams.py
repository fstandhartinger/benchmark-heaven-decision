from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .policy import GEMINI_MODEL, normalize


class UpstreamError(RuntimeError):
    def __init__(self, message, *, attempts=1, usage=None, elapsed_s=0.0):
        super().__init__(message)
        self.attempts = attempts
        self.usage = usage or {}
        self.elapsed_s = elapsed_s


def post_json(url: str, body: dict, headers: dict, timeout: float = 120) -> tuple[dict, float]:
    started = time.monotonic()
    request = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(512).decode(errors="replace")
        raise UpstreamError(f"upstream HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise UpstreamError(f"upstream transport error: {exc}") from exc
    return parsed, time.monotonic() - started


def classifier_fast(inputs: list[str], labels: list[str], instructions: str = "", timeout: float = 45) -> tuple[list[dict], float]:
    parsed, latency = post_json("https://classifier.dev/v1/classify",
                                {"inputs": inputs, "labels": labels, "instructions": instructions, "tier": "fast"},
                                {"accept": "application/json", "user-agent": "BH-Decision/0.1"}, timeout)
    if not isinstance(parsed, dict):
        raise UpstreamError("classifier.dev returned a non-object response")
    results = parsed.get("results")
    if not isinstance(results, list) or len(results) != len(inputs):
        raise UpstreamError("classifier.dev returned the wrong result count")
    for row in results:
        if not isinstance(row, dict):
            raise UpstreamError("classifier.dev returned a malformed result row")
        if not isinstance(row.get("scores"), dict) or set(row["scores"]) != set(labels):
            raise UpstreamError("classifier.dev returned an incomplete distribution")
        try:
            row["scores"] = normalize(row["scores"])
        except (TypeError, ValueError) as exc:
            raise UpstreamError(f"classifier.dev returned invalid scores: {exc}") from exc
    return results, latency


def gemini_decision(text: str, labels: list[str], instructions: str = "", timeout: float = 45) -> tuple[dict, float]:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise UpstreamError("GOOGLE_API_KEY is not configured")
    options = "\n".join(f"- {label}" for label in labels)
    prompt = ("Classify the input into exactly one supplied label. Return a probability for every label; "
              "probabilities must sum to 1. Do not invent labels.\n\nDECISION\n" + instructions +
              "\n\nOPTIONS\n" + options + "\n\nINPUT\n" + text)
    pair = {"type": "OBJECT", "properties": {"label": {"type": "STRING", "enum": labels},
                                                       "probability": {"type": "NUMBER", "minimum": 0, "maximum": 1}},
            "required": ["label", "probability"]}
    schema = {"type": "OBJECT", "properties": {"label": {"type": "STRING", "enum": labels},
                                                       "probabilities": {"type": "ARRAY", "minItems": len(labels),
                                                                         "maxItems": len(labels), "items": pair}},
              "required": ["label", "probabilities"]}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 2000, "responseMimeType": "application/json",
                                 "responseSchema": schema, "thinkingConfig": {"thinkingLevel": "LOW"}}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
    last = None
    accumulated = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}
    started = time.monotonic()
    for attempt in range(2):
        try:
            parsed, _ = post_json(url, body, {}, timeout)
            if not isinstance(parsed, dict):
                raise UpstreamError("Gemini returned a non-object response")
            metadata = parsed.get("usageMetadata") or {}
            current = {"input_tokens": metadata.get("promptTokenCount", 0),
                       "output_tokens": metadata.get("candidatesTokenCount", 0),
                       "thinking_tokens": metadata.get("thoughtsTokenCount", 0),
                       "total_tokens": metadata.get("totalTokenCount", 0)}
            for key_name, value in current.items():
                accumulated[key_name] += int(value or 0)
            answer = json.loads(parsed["candidates"][0]["content"]["parts"][-1]["text"])
            scores = normalize({p["label"]: p["probability"] for p in answer["probabilities"]})
            if set(scores) != set(labels):
                raise ValueError("Gemini distribution does not cover all labels")
            return {"label": max(scores, key=scores.get), "scores": scores,
                    "usage": accumulated, "attempts": attempt + 1}, time.monotonic() - started
        except (UpstreamError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < 1:
                time.sleep(2 ** attempt)
    raise UpstreamError(f"Gemini escalation failed after 2 attempts: {last}", attempts=2, usage=accumulated,
                        elapsed_s=time.monotonic() - started)
