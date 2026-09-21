#!/usr/bin/env python3
"""Collect Gemini candidate distributions on redistributable JevBench items.

This is a tuning-only collector. It refuses any task spec whose manifest flag is
not public and enforces a paid-cost ceiling from returned token accounting.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

INPUT_PER_M = .75
OUTPUT_PER_M = 3.75
_lock = threading.Lock()


def records(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def labels_and_prompt(task):
    q = task["question"]
    labels = task["labels"]
    criteria = q.get("criteria")
    if q["type"] == "noul":
        descriptions = {"yes": (criteria or {}).get("true") or "yes", "no": (criteria or {}).get("false") or "no"}
    elif q["type"] == "score":
        descriptions = {str(i): str(v) for i, v in enumerate(criteria)}
    else:
        descriptions = {str(k): str(v or k) for k, v in criteria.items()}
    state = task["state"] if isinstance(task["state"], str) else json.dumps(task["state"], ensure_ascii=False)
    options = "\n".join(f"- {lab}: {descriptions.get(lab, lab)}" for lab in labels)
    return labels, ("Classify the input into exactly one supplied label. Return a probability for every label; "
                    "probabilities must sum to 1. Do not invent labels.\n\nDECISION\n" + q["instructions"] +
                    "\n\nOPTIONS\n" + options + "\n\nINPUT\n" + state)


def call(task, key, model):
    labels, prompt = labels_and_prompt(task)
    schema = {"type": "OBJECT", "properties": {
        "label": {"type": "STRING", "enum": labels},
        "probabilities": {"type": "ARRAY", "minItems": len(labels), "maxItems": len(labels), "items": {
            "type": "OBJECT", "properties": {"label": {"type": "STRING", "enum": labels},
                                                   "probability": {"type": "NUMBER", "minimum": 0, "maximum": 1}},
            "required": ["label", "probability"]}}}, "required": ["label", "probabilities"]}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 2000, "responseMimeType": "application/json",
                                 "responseSchema": schema, "thinkingConfig": {"thinkingLevel": "LOW"}}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    started = time.monotonic()
    last_error = None
    for attempt in range(3):
        req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                parsed = json.load(response)
            text = parsed["candidates"][0]["content"]["parts"][-1]["text"]
            answer = json.loads(text)
            pairs = answer["probabilities"]
            probs = {x["label"]: float(x["probability"]) for x in pairs}
            if set(probs) != set(labels) or any(v < 0 or v > 1 for v in probs.values()) or not .98 <= sum(probs.values()) <= 1.02:
                raise ValueError("invalid probability distribution")
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}
            usage = parsed.get("usageMetadata") or {}
            return {"task_id": task["id"], "ok": True, "predicted": max(probs, key=probs.get), "probs": probs,
                    "latency_s": time.monotonic() - started, "model": model,
                    "usage": {"input_tokens": usage.get("promptTokenCount", 0),
                              "output_tokens": usage.get("candidatesTokenCount", 0),
                              "thinking_tokens": usage.get("thoughtsTokenCount", 0),
                              "total_tokens": usage.get("totalTokenCount", 0)}}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:180]}"
            time.sleep(2 ** attempt)
    return {"task_id": task["id"], "ok": False, "error": last_error, "latency_s": time.monotonic() - started, "model": model}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("output")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=8.0)
    args = ap.parse_args()
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GOOGLE_API_KEY is not set")
    manifest = json.loads(Path(args.manifest).read_text())
    tasks = []
    for spec in manifest["tasks"]:
        if spec["public"]:
            tasks.extend(records(spec["path"]))
    existing = {}
    output = Path(args.output)
    if output.exists():
        existing = {r["task_id"]: r for r in records(output)}
    pending = [t for t in tasks if not existing.get(t["id"], {}).get("ok")]
    results = dict(existing)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call, t, key, args.model): t["id"] for t in pending}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            results[row["task_id"]] = row
            ordered = [results[t["id"]] for t in tasks if t["id"] in results]
            output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in ordered))
            input_tokens = sum(r.get("usage", {}).get("input_tokens", 0) for r in results.values())
            output_tokens = sum(r.get("usage", {}).get("output_tokens", 0) + r.get("usage", {}).get("thinking_tokens", 0) for r in results.values())
            cost = input_tokens / 1e6 * INPUT_PER_M + output_tokens / 1e6 * OUTPUT_PER_M
            if cost > args.budget:
                for f in futures:
                    f.cancel()
                raise SystemExit(f"budget ceiling exceeded (${cost:.2f} > ${args.budget:.2f})")
    ok = sum(r.get("ok", False) for r in results.values())
    if ok != len(tasks):
        raise SystemExit(f"incomplete: {ok}/{len(tasks)} successful")


if __name__ == "__main__":
    main()
