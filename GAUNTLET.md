# Gauntlet record

Three independent review rounds were completed before the held-out run.

- Round 1: **revise**, 2 blockers and 5 majors. It found the raw-vs-tempered gate mismatch,
  invalid no-judge score naming, excessive validation search, weak calibration disclosure,
  retry accounting, missing bounds, and malformed-input handling.
- Round 2: **revise**, 1 blocker and 6 majors. It found that mixed result files were still
  opened before filtering, plus cost/latency, provenance, Smart-comparator, finite-score and
  malformed-upstream gaps.
- Round 3: **revise**, 4 majors and 1 minor, no blockers. It found missing support validation,
  retrospective collection receipts, comparator-basis caveats, unbounded connection threads,
  and an inconsistent unknown-cost header.

All deterministic code issues were repaired after round 3. The three-round cap forbids claiming
an independent clean pass. Residual audit limitations: the collection receipts are hashed and
time-ordered but not externally signed; Jev comparison uses the frozen published JevBench run,
not a contemporaneous rerun. Qualified Benchmark Heaven worker attempts also failed before a
verdict (transport failures, output-length failures, then OpenRouter HTTP 402); these failures are
not counted as passes. The fresh-context Codex reviewer never inspected held-out data.
