#!/usr/bin/env python3
"""Collect classifier.dev Fast/Smart predictions on public JevBench only."""
from __future__ import annotations
import argparse, concurrent.futures, json, time, urllib.error, urllib.request
from pathlib import Path


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)


def request_for(task, tier):
    q = task["question"]; criteria = q.get("criteria"); labels = task["labels"]
    text = q["instructions"].strip()
    if q["type"] == "noul":
        criteria = criteria or {}; text += f"\n\nOptions:\n- yes: {criteria.get('true') or 'yes'}\n- no: {criteria.get('false') or 'no'}"
    elif q["type"] == "score":
        text += "\n\nLevels, lowest to highest:\n" + "\n".join(f"- {i}: {v}" for i, v in enumerate(criteria))
    elif any(criteria.values()):
        text += "\n\nOptions:\n" + "\n".join(f"- {k}: {v or k}" for k, v in criteria.items())
    state = task["state"] if isinstance(task["state"], str) else json.dumps(task["state"], ensure_ascii=False)
    return {"input": state, "labels": labels, "instructions": text, "tier": tier}


def call(task, tier):
    body = request_for(task, tier); started = time.monotonic(); last = None
    for attempt in range(4):
        req = urllib.request.Request("https://classifier.dev/v1/classify", json.dumps(body).encode(),
                                     {"content-type": "application/json", "user-agent": "BH-Decision-Eval/0.1"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as response: parsed = json.load(response)
            result = parsed["results"][0]; scores = result.get("scores")
            return {"task_id": task["id"], "ok": True, "predicted": result["label"], "probs": scores,
                    "latency_s": time.monotonic() - started, "model": result.get("model"),
                    "escalated": bool(result.get("escalated")), "confidence": result.get("confidence"),
                    "usage": parsed.get("usage") or {}}
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code == 429 and attempt < 3: time.sleep(20); continue
            break
        except Exception as exc:
            last = f"{type(exc).__name__}: {str(exc)[:120]}"; break
    return {"task_id": task["id"], "ok": False, "error": last, "latency_s": time.monotonic() - started}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("output"); ap.add_argument("--tier", choices=("fast", "smart"), required=True)
    args = ap.parse_args(); manifest = json.loads(Path(args.manifest).read_text()); tasks = []
    for spec in manifest["tasks"]:
        if spec["public"]: tasks.extend(rows(spec["path"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(lambda t: call(t, args.tier), tasks))
    Path(args.output).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in result))
    if sum(r["ok"] for r in result) != len(tasks): raise SystemExit("incomplete public collection")


if __name__ == "__main__": main()
