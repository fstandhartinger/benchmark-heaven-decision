# Frozen design before held-out evaluation

Frozen 2026-09-21T16:43Z. No held-out task file or private per-item outcome was opened by
this job before this point.

Policy: classifier.dev Fast first; if its raw maximum probability is below **0.85**, call
Gemini 3.8 Flash (low reasoning, temperature 0) and use its complete distribution. Otherwise
retain the Fast label and apply temperature **1.5**. No abstention. Gemini failure retains the
primary answer, counts the failed latency, and makes total cost unknown if billing metadata is
unavailable.

Public-only selection: deterministic grouped 160/71 tune/validation split; five predeclared
thresholds; tune accuracy alone selected 0.85, with the lower-escalation threshold winning an
accuracy tie. Validation never selected the policy. Public validation accuracy was 98.59%, hard
calibration 88.52, escalation 29.58%. classifier.dev Smart was 97.40% on all 231 public items but
returned scored distributions for only 77.92%.

Frozen hashes:

- policy.py `db7de5f0510f8c602fb2d0d905eac3dfbeec433efcd2c70d03d739ea5c1bb3b6`
- upstreams.py `a7239a14c9610a70c0f470c5a5b6c6f3c3406d749e2b2c9b0ac95cd5029c7eb3`
- service.py `2fdd3be687aadb0ef700fd5af32a99eec086902e171a7b46cfe7bcba3a55bf74`
- sealed_evaluate.py `67701d854fcdb3a3f39cc0b39ac77cf01ec2b30c230a6b748f8db618fa470725`
- tune_policy.py `cd1316b8278cdd8781b69fe6b340f2a8c6314d72e1af31b45a37c52852a8555b`
- tuning-v6.json `8ba15f7e63e1700adf990a5e1fe2762dda71d87928579c6bfa64b9618aa6c49f`
- manifest-public-only.json `36f58b9ad9ea7df254eabded7e7ddcb2f9551b2fff289176f6848942294ec9b3`
- PUBLIC_PROVENANCE.json `8e4e78a5a6ad8a09998324eb0a459337b0628b82c3745201996838e8617ab0ec`

One-shot command:

`python3 tools/sealed_evaluate.py /home/flori/jobs/jev-ensemble-20260920/manifest.json heldout-result.json --lock heldout.lock`

After this command, held-out results may be reported but may not change the policy or service.
