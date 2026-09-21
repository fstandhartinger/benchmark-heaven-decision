#!/usr/bin/env python3
"""Tune a bounded confidence cascade on PUBLIC JevBench rows only.

The input manifest may name private files, but this command constructs the public
task-id allow-list first and refuses to load result rows outside it.  A separate
sealed evaluator is used after the resulting policy is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

TIERS = {"easy": .14, "standard": .28, "judge": .28, "hard": .30}


def read_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def group_id(task_id: str) -> str:
    """Keep public paraphrase/source families together."""
    parts = task_id.rsplit("-", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else task_id


def assign_splits(tasks):
    """Deterministic grouped 70/30 split, balanced within tier/question type."""
    strata = defaultdict(lambda: defaultdict(list))
    for tid, task in tasks.items():
        strata[(task["tier"], task["question"]["type"])][group_id(tid)].append(tid)
    for groups in strata.values():
        target = round(sum(len(ids) for ids in groups.values()) * .30)
        chosen = 0
        ordered = sorted(groups.items(), key=lambda x: hashlib.sha256(("bh-classifier-v2:" + x[0]).encode()).hexdigest())
        for _, ids in ordered:
            use_validation = chosen < target and abs((chosen + len(ids)) - target) <= abs(chosen - target)
            for tid in ids:
                tasks[tid]["split"] = "validate" if use_validation else "tune"
            if use_validation:
                chosen += len(ids)


def normalize(p):
    values = {k: float(v) for k, v in p.items()}
    if not values or not all(math.isfinite(v) and v >= 0 for v in values.values()):
        raise ValueError("probabilities must be finite and nonnegative")
    total = sum(max(v, 1e-12) for v in values.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("invalid probability total")
    return {k: max(v, 1e-12) / total for k, v in values.items()}


def temper(p, temperature):
    p = normalize(p)
    q = {k: v ** (1.0 / temperature) for k, v in p.items()}
    return normalize(q)


def blend(a, b, weight_b):
    if set(a) != set(b):
        raise ValueError("cannot blend distributions with different support")
    return normalize({k: (1 - weight_b) * a.get(k, 0) + weight_b * b.get(k, 0) for k in a})


def ece(rows, bins=10):
    groups = [[] for _ in range(bins)]
    for r in rows:
        conf = max(r["probs"].values())
        pred = max(r["probs"], key=r["probs"].get)
        groups[min(int(conf * bins), bins - 1)].append((conf, pred == r["expected"]))
    n = len(rows)
    return sum(len(g) / n * abs(sum(c for c, _ in g) / len(g) - sum(ok for _, ok in g) / len(g)) for g in groups if g)


def metrics(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["tier"]].append(r)
    tier_accuracy = {tier: sum(max(x["probs"], key=x["probs"].get) == x["expected"] for x in xs) / len(xs) for tier, xs in by.items()}
    public_weighted = 100 * sum(TIERS[t] * v for t, v in tier_accuracy.items()) / sum(TIERS[t] for t in tier_accuracy)
    hard = by.get("hard", [])
    hard_ece = ece(hard) if hard else None
    gold = [r for r in hard if r.get("gold_probs")]
    tvd = (sum(.5 * sum(abs(r["probs"].get(k, 0) - v) for k, v in r["gold_probs"].items()) for r in gold) / len(gold)) if gold else None
    cal = None if hard_ece is None else (100 * max(0, 1 - hard_ece / .5) if tvd is None else (100 * max(0, 1 - hard_ece / .5) + 100 * (1 - tvd)) / 2)
    accuracy = sum(max(r["probs"], key=r["probs"].get) == r["expected"] for r in rows) / len(rows)
    return {"n": len(rows), "accuracy": accuracy, "tier_accuracy": tier_accuracy,
            "public_weighted_accuracy_not_four_tier_intelligence": public_weighted,
            "missing_tiers": sorted(set(TIERS) - set(tier_accuracy)), "overall_ece": ece(rows),
            "hard_ece": hard_ece, "hard_mean_tvd": tvd, "hard_gold_probability_n": len(gold),
            "jevbench_hard_calibration": cal}


def load_public(manifest):
    tasks = {}
    for spec in manifest["tasks"]:
        if not spec["public"]:
            continue
        for task in read_jsonl(spec["path"]):
            if task.get("question", {}).get("type") == "score":
                task["expected"] = str(task["expected"])
            tasks[task["id"]] = {**task, "tier": spec["tier"]}
    assign_splits(tasks)
    systems = {}
    for name, spec in manifest["systems"].items():
        rows = {}
        for path in spec["files"]:
            # Only retain explicitly public ids; private prediction rows are discarded
            # immediately and their task/label records are never opened.
            for row in read_jsonl(path):
                if row.get("task_id") in tasks and row.get("ok") and row.get("probs"):
                    task = tasks[row["task_id"]]
                    if task.get("expected") not in task.get("labels", []):
                        raise ValueError(f"expected label outside declared support: {row['task_id']}")
                    if not isinstance(row["probs"], dict) or set(row["probs"]) != set(task["labels"]):
                        raise ValueError(f"prediction support mismatch: {name}/{row['task_id']}")
                    normalize(row["probs"])
                    rows[row["task_id"]] = row
        if len(rows) == len(tasks):
            systems[name] = rows
    return tasks, systems


def routed_rows(tasks, systems, split, secondary, threshold, margin_threshold, weight, t_primary, t_secondary):
    out, escalated = [], 0
    for tid, task in tasks.items():
        if task["split"] != split:
            continue
        raw = normalize(systems["classifier-dev-fast"][tid]["probs"])
        p = temper(raw, t_primary)
        ordered = sorted(raw.values(), reverse=True)
        gate = ordered[0] < threshold or ordered[0] - ordered[1] < margin_threshold
        if gate:
            q = temper(systems[secondary][tid]["probs"], t_secondary)
            p = blend(p, q, weight)
            escalated += 1
        out.append({"tier": task["tier"], "expected": task["expected"], "probs": p,
                    "gold_probs": task.get("provenance", {}).get("gold_probs")})
    return out, escalated / len(out)


def objective(m, escalation):
    # Public-only objective. It is not named or compared as four-tier Intelligence.
    return 100 * m["accuracy"] - 8 * max(0, escalation - .20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("output")
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    tasks, systems = load_public(manifest)
    secondaries = [x for x in ("gemini-3.8-flash",) if x in systems]
    if not secondaries:
        raise SystemExit("gemini-3.8-flash public predictions are required")
    candidates = []
    for secondary, th, margin, weight, tp, ts in itertools.product(
            secondaries, [.70, .75, .80, .85, .90], [0], [1.0], [1.5], [1.0]):
        tune_rows, tune_esc = routed_rows(tasks, systems, "tune", secondary, th, margin, weight, tp, ts)
        tune = metrics(tune_rows)
        candidates.append({"secondary": secondary, "confidence_threshold": th, "margin_threshold": margin,
                           "secondary_weight": weight, "primary_temperature": tp, "secondary_temperature": ts,
                           "tune": tune, "tune_escalation": tune_esc, "tune_objective": objective(tune, tune_esc)})
    # Select solely on tune. Validation is evaluated once for the selected policy.
    chosen = max((c for c in candidates if c["tune_escalation"] <= .50), key=lambda x: x["tune_objective"])
    rows, esc = routed_rows(tasks, systems, "validate", chosen["secondary"], chosen["confidence_threshold"],
                            chosen["margin_threshold"], chosen["secondary_weight"], chosen["primary_temperature"], chosen["secondary_temperature"])
    chosen["validate"] = metrics(rows)
    chosen["validate_escalation"] = esc
    baseline = {}
    for split in ("tune", "validate"):
        rows, _ = routed_rows(tasks, systems, split, secondaries[0], 0, 0, 0, 1, 1)
        baseline[split] = metrics(rows)
    policy = {k: chosen[k] for k in ("secondary", "confidence_threshold", "margin_threshold", "secondary_weight", "primary_temperature", "secondary_temperature")}
    result = {"schema": "bh-classifier-tuning/v1", "public_items": len(tasks),
              "split_rule": "deterministic grouped 70/30, balanced within tier and question type; group hash seed bh-classifier-v2",
              "split_counts": {s: sum(t["split"] == s for t in tasks.values()) for s in ("tune", "validate")},
              "systems": sorted(systems), "candidate_count": len(candidates), "tune_candidates": candidates,
              "selection_rule": "highest tune accuracy objective with tune escalation <=50%; validation never selects policy",
              "chosen_policy": policy, "baseline": baseline,
              "chosen_metrics": {"tune": chosen["tune"], "validate": chosen["validate"],
                                 "tune_escalation": chosen["tune_escalation"], "validate_escalation": chosen["validate_escalation"]}}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
