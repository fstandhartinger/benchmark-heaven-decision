# Benchmark Heaven Decision

An MIT-licensed, classifier.dev-compatible decision endpoint built by Benchmark Heaven.
It first calls classifier.dev Fast (Jev), recalibrates confident distributions with a
temperature learned only on public JevBench items, and asks Gemini 3.8 Flash only when the
raw Jev confidence is below 0.85. Unlike classifier.dev Smart, an escalation returns a full
probability distribution rather than null confidence.

This is an orchestration layer, not a new model. Its upstream dependencies are disclosed in
every response. It never ranks itself on Benchmark Heaven.

## API

- `POST /v1/classify` — classifier.dev-compatible `{inputs, labels, instructions?}`
- `POST /v1/systemone` — TypeSafe-compatible `{state, questions}`
- `GET /models`
- `GET /health`

Set `GOOGLE_API_KEY` and `SERVICE_API_KEY`, then run `python -m bh_classifier.service`.
When `SERVICE_API_KEY` is set, POST routes require `Authorization: Bearer ...`; health and
model discovery remain public. Request content is not logged. The service reports measured
upstream token cost with no markup.

## Method

The public tune/validation script is `tools/tune_policy.py`. The final private evaluation
is sealed and aggregate-only. See `docs/CLASSIFIER_DEV_APPROACH.md` for the audited baseline.

## Honest result

On the single frozen 303-item held-out run, the endpoint reached 96.04% accuracy, 97.03
JevBench intelligence, and 90.78 calibration. It beat Jev and classifier.dev Fast on those
predictive axes, but lost badly on economics: $0.566/1k classifications, 2.23 s p95, and a
59.30 composite versus 75.66 for Jev and 85.11 for classifier.dev Fast. We did not retune
after seeing this result and did not add the endpoint to the Benchmark Heaven board.

## License

MIT.
