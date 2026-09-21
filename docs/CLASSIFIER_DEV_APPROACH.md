# classifier.dev Smart: source-backed description

Read 2026-09-21. Public implementation audited at commit
`a68b524f86e0ddd553eb1cd95c69be881657c578`.

- Fast sends each choice to Jev (`jev-latest`) and returns Jev's label and probabilities.
- Smart applies one fixed gate: if Jev confidence is strictly below 0.7, it asks
  `google/gemini-3.8-flash` once, at temperature 0 with low reasoning effort.
- A valid Gemini label replaces Jev unconditionally. It is not blended, voted or reviewed.
- Replaced answers have `confidence: null` and `scores: null`; thus Smart does not expose a
  calibrated distribution on escalated items. A failed escalation retains Jev's answer.
- It has no per-topic routing, option-order correction or classification abstention.
  Multi-label calls do not use the Smart cascade.
- Published measurements are two public 400-item sets. Smart improved AG News from 87.5%
  to 90.0% and emotion from 61.8% to 62.7%. The page warns that gaps below about five
  points at this size are not evidence. The backend costs about $0.70 per 1,000 escalations
  and about 2.3 seconds per escalated item.

Primary sources:

- https://classifier.dev/benchmark
- https://classifier.dev/about
- https://classifier.dev/pricing
- https://classifier.dev/docs
- https://github.com/mrmps/classifier-dev/blob/a68b524f86e0ddd553eb1cd95c69be881657c578/src/index.ts
- https://github.com/mrmps/classifier-dev/blob/a68b524f86e0ddd553eb1cd95c69be881657c578/src/jev.ts
