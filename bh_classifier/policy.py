from __future__ import annotations
import math

PRIMARY_TEMPERATURE = 1.5
CONFIDENCE_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.0
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_INPUT_USD_PER_M = 0.75
GEMINI_OUTPUT_USD_PER_M = 3.75
FAST_USD_PER_1000 = 0.003333333333333333


def normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        raise ValueError("scores must not be empty")
    clean = {}
    for key, value in scores.items():
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("scores must be finite and nonnegative")
        clean[str(key)] = max(value, 1e-12)
    total = sum(clean.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("score total must be finite and positive")
    result = {k: v / total for k, v in clean.items()}
    if not all(math.isfinite(v) and v >= 0 for v in result.values()) or abs(sum(result.values()) - 1) > 1e-9:
        raise ValueError("normalized scores are invalid")
    return result


def temper(scores: dict[str, float], temperature: float = PRIMARY_TEMPERATURE) -> dict[str, float]:
    scores = normalize(scores)
    return normalize({k: v ** (1 / temperature) for k, v in scores.items()})


def should_escalate(scores: dict[str, float]) -> bool:
    ordered = sorted(normalize(scores).values(), reverse=True)
    return ordered[0] < CONFIDENCE_THRESHOLD or ordered[0] - ordered[1] < MARGIN_THRESHOLD


def estimated_gemini_cost(usage: dict) -> float:
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0) + int(usage.get("thinking_tokens") or 0)
    return input_tokens / 1e6 * GEMINI_INPUT_USD_PER_M + output_tokens / 1e6 * GEMINI_OUTPUT_USD_PER_M
