#!/usr/bin/env python3
import hashlib, json
from pathlib import Path

FILES = [
    "/home/flori/jobs/jev-class-benchmark-20260919/repo/datasets/public/easy.jsonl",
    "/home/flori/jobs/jev-class-benchmark-20260919/repo/datasets/public/original.jsonl",
    "/home/flori/jobs/jev-class-benchmark-20260919/repo/datasets/public/hard.jsonl",
    "classifier-fast-public.jsonl", "classifier-smart-public.jsonl", "gemini-public.jsonl",
    "manifest-public-only.json", "tuning-v6.json", "tools/tune_policy.py",
]

out = {"schema": "bh-classifier-public-provenance/v1", "files": []}
for name in FILES:
    raw = Path(name).read_bytes()
    out["files"].append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
Path("PUBLIC_PROVENANCE.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
