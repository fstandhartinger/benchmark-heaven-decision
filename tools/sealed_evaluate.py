#!/usr/bin/env python3
"""One-shot aggregate-only evaluator for the frozen private JevBench split.

This command intentionally has no tuning arguments. It writes a completion lock before
opening private tasks and refuses a second run. Output contains aggregates and hashes only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from bh_classifier.policy import FAST_USD_PER_1000, estimated_gemini_cost, should_escalate, temper
from bh_classifier.upstreams import gemini_decision

TIERS = {"easy": .14, "standard": .28, "judge": .28, "hard": .30}


def stream(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def percentile(values, q):
    xs = sorted(values); pos = (len(xs) - 1) * q; lo = math.floor(pos); hi = math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def ece(rows, bins=10):
    groups = [[] for _ in range(bins)]
    for row in rows:
        confidence = max(row["probs"].values()); prediction = max(row["probs"], key=row["probs"].get)
        groups[min(int(confidence * bins), bins - 1)].append((confidence, prediction == row["expected"]))
    return sum(len(g) / len(rows) * abs(sum(c for c, _ in g) / len(g) - sum(ok for _, ok in g) / len(g)) for g in groups if g)


def axes(rows, usd_per_1000):
    by = defaultdict(list)
    for row in rows: by[row["tier"]].append(row)
    tier_acc = {t: sum(max(r["probs"], key=r["probs"].get) == r["expected"] for r in rs) / len(rs) for t, rs in by.items()}
    intelligence = 100 * sum(TIERS[t] * tier_acc[t] for t in tier_acc) / sum(TIERS[t] for t in tier_acc)
    hard = by["hard"]; hard_ece = ece(hard)
    gold = [r for r in hard if r.get("gold_probs")]
    tvd = sum(.5 * sum(abs(r["probs"].get(k, 0) - v) for k, v in r["gold_probs"].items()) for r in gold) / len(gold)
    calibration = (100 * max(0, 1 - hard_ece / .5) + 100 * (1 - tvd)) / 2
    timed = by["standard"] + by["judge"]
    p50 = percentile([r["latency_s"] for r in timed], .5); p95 = percentile([r["latency_s"] for r in timed], .95)
    point = lambda s: max(0, min(100, 100 - 20 * math.log10(s / .1)))
    speed = (point(p50) + point(p95)) / 2
    cost = max(0, min(100, 100 - 30 * math.log10(usd_per_1000 / .001)))
    out_axes = {"intelligence": intelligence, "calibration": calibration, "speed": speed, "cost": cost}
    composite = math.exp(sum(.25 * math.log(max(v, 1)) for v in out_axes.values()))
    return {"n": len(rows), "accuracy": sum(max(r["probs"], key=r["probs"].get) == r["expected"] for r in rows) / len(rows),
            "tier_accuracy": tier_acc, "hard_ece": hard_ece, "hard_mean_tvd": tvd,
            "p50_s_standard_judge": p50, "p95_s_standard_judge": p95,
            "usd_per_1000": usd_per_1000, "axes": out_axes, "jevbench_score": composite}


def instructions(task):
    q = task["question"]; text = q["instructions"].strip(); criteria = q.get("criteria")
    if q["type"] == "noul":
        criteria = criteria or {}; return text + f"\n\nOptions:\n- yes: {criteria.get('true') or 'yes'}\n- no: {criteria.get('false') or 'no'}"
    if q["type"] == "score":
        return text + "\n\nLevels, lowest to highest:\n" + "\n".join(f"- {i}: {v}" for i, v in enumerate(criteria))
    if any(criteria.values()):
        return text + "\n\nOptions:\n" + "\n".join(f"- {k}: {v or k}" for k, v in criteria.items())
    return text


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("output"); ap.add_argument("--lock", required=True)
    args = ap.parse_args(); lock = Path(args.lock)
    if lock.exists(): raise SystemExit("sealed evaluation lock exists; rerun forbidden")
    lock.write_text("started\n")
    manifest_raw = Path(args.manifest).read_bytes(); manifest = json.loads(manifest_raw)
    tasks = {}
    for spec in manifest["tasks"]:
        if spec["public"]: continue
        for task in stream(spec["path"]):
            if task.get("question", {}).get("type") == "score": task["expected"] = str(task["expected"])
            tasks[task["id"]] = {**task, "tier": spec["tier"]}
    systems = {}
    for name in ("classifier-dev-fast", "jev-1.13.0"):
        rows = {}
        for path in manifest["systems"][name]["files"]:
            for row in stream(path):
                if row.get("task_id") in tasks and row.get("ok") and row.get("probs"): rows[row["task_id"]] = row
        if len(rows) != len(tasks): raise RuntimeError(f"incomplete sealed comparator {name}")
        systems[name] = rows
    candidate = []; gemini_cost = 0.0; escalated = 0; failures = 0
    for tid, task in tasks.items():
        base = systems["classifier-dev-fast"][tid]; probs = temper(base["probs"]); latency = base["latency_s"]
        if should_escalate(base["probs"]):
            state = task["state"] if isinstance(task["state"], str) else json.dumps(task["state"], ensure_ascii=False)
            try:
                decision, extra_latency = gemini_decision(state, task["labels"], instructions(task))
                probs = decision["scores"]; latency += extra_latency; gemini_cost += estimated_gemini_cost(decision["usage"]); escalated += 1
            except Exception:
                failures += 1
        candidate.append({"tier": task["tier"], "expected": task["expected"], "probs": probs, "latency_s": latency,
                          "gold_probs": task.get("provenance", {}).get("gold_probs")})
    output = {"schema": "bh-classifier-heldout/v1", "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
              "heldout_items": len(tasks), "policy": {"confidence_threshold": .85, "primary_temperature": 1.5,
                                                        "secondary": "gemini-3.8-flash", "secondary_weight": 1},
              "candidate": axes(candidate, FAST_USD_PER_1000 + gemini_cost / len(tasks) * 1000),
              "escalated": escalated, "escalation_rate": escalated / len(tasks), "escalation_failures": failures,
              "gemini_cost_usd": gemini_cost, "comparators": {}}
    for name, cost in (("classifier-dev-fast", manifest["published"]["classifier-dev-fast"]["cost"]),
                       ("jev-1.13.0", manifest["published"]["jev-1.13.0"]["cost"])):
        rows = [{"tier": tasks[tid]["tier"], "expected": tasks[tid]["expected"], "probs": row["probs"],
                 "latency_s": row["latency_s"], "gold_probs": tasks[tid].get("provenance", {}).get("gold_probs")}
                for tid, row in systems[name].items()]
        output["comparators"][name] = axes(rows, cost)
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    lock.write_text("completed\n")


if __name__ == "__main__": main()
