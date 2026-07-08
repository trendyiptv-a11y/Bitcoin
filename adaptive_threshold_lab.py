from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "btc-swing-strategy" / "coeziv_state.json"
RISK_PATH = ROOT / "btc-swing-strategy" / "risk_window.json"
OUT_PATH = ROOT / "btc-swing-strategy" / "adaptive_threshold_lab.json"

BACKTEST_START_YEAR = 2018
STEP = 0.05


def _num(value: Any) -> Optional[float]:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _first_short_start(state: Dict[str, Any]) -> Optional[float]:
    history = state.get("signal_history") or []
    for item in history:
        if item.get("signal") == "short":
            return _num(item.get("model_price_usd"))
    return _num(state.get("model_price_usd") or state.get("price_usd"))


def _current_context(state: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
    start = _first_short_start(state)
    old_threshold_pct = _num(risk.get("major_drawdown_threshold"))
    if old_threshold_pct is None:
        old_threshold_pct = -0.20
    old_threshold = start * (1.0 + old_threshold_pct) if start is not None else None
    miner_standard = _num((state.get("production_costs_usd") or {}).get("average"))
    p10 = _num((state.get("model_price_bands") or {}).get("p10"))
    price = _num(state.get("price_usd"))
    return {
        "start_price_usd": start,
        "old_threshold_pct": old_threshold_pct,
        "old_threshold_usd": old_threshold,
        "miner_standard_usd": miner_standard,
        "p10_cohesive_usd": p10,
        "price_usd": price,
        "model_price_usd": _num(state.get("model_price_usd")),
        "model_deviation_pct": (_num(state.get("model_price_deviation")) or 0.0) * 100.0,
        "ic_context": state.get("model_price_context") or {},
        "risk_context": risk,
    }


def _weights(step: float = STEP) -> Iterable[Tuple[float, float, float]]:
    n = int(round(1.0 / step))
    for a_i in range(0, n + 1):
        for b_i in range(0, n + 1 - a_i):
            c_i = n - a_i - b_i
            a = round(a_i * step, 10)
            b = round(b_i * step, 10)
            c = round(c_i * step, 10)
            yield a, b, c


def _adaptive_threshold(old_threshold: float, miner: float, p10: float, w: Tuple[float, float, float]) -> float:
    a, b, c = w
    return a * old_threshold + b * miner + c * p10


def _score_candidate(
    threshold: float,
    old_threshold: float,
    miner: float,
    p10: float,
    price: Optional[float],
    w: Tuple[float, float, float],
) -> Dict[str, Any]:
    """Current-state laboratory score.

    This is not the final historical validation. It ranks candidates for the
    current context by coherence:
    - keep threshold between old statistical drawdown and p10 band;
    - prefer not to fall below miner standard when miner context matters;
    - avoid sitting too close to current spot, to reduce noise;
    - preserve some memory of old statistical threshold.
    """
    a, b, c = w
    lo = min(old_threshold, miner, p10)
    hi = max(old_threshold, miner, p10)
    span = max(1.0, hi - lo)

    inside_score = 100.0 if lo <= threshold <= hi else max(0.0, 100.0 - abs(threshold - min(max(threshold, lo), hi)) / span * 100.0)
    miner_score = 100.0 - min(100.0, abs(threshold - miner) / max(1.0, miner) * 180.0)
    p10_score = 100.0 - min(100.0, abs(threshold - p10) / max(1.0, p10) * 150.0)
    memory_score = 100.0 - min(100.0, abs(threshold - old_threshold) / max(1.0, old_threshold) * 80.0)

    # Noise guard: if threshold is too close to current spot, it may trigger on normal volatility.
    if price and price > 0:
        distance_to_price_pct = abs(price - threshold) / price
        noise_score = min(100.0, distance_to_price_pct / 0.08 * 100.0)
    else:
        distance_to_price_pct = None
        noise_score = 50.0

    # Weight regularization: avoid one component completely dominating until historical backtest confirms it.
    dominance = max(a, b, c)
    balance_score = max(0.0, 100.0 - max(0.0, dominance - 0.65) / 0.35 * 100.0)

    final = (
        0.18 * inside_score
        + 0.22 * miner_score
        + 0.24 * p10_score
        + 0.12 * memory_score
        + 0.14 * noise_score
        + 0.10 * balance_score
    )
    return {
        "score": round(final, 4),
        "threshold_usd": round(threshold, 2),
        "weights": {
            "old_statistical": round(a, 2),
            "miner_standard": round(b, 2),
            "p10_cohesive": round(c, 2),
        },
        "components": {
            "inside_anchor_range": round(inside_score, 4),
            "miner_alignment": round(miner_score, 4),
            "p10_alignment": round(p10_score, 4),
            "statistical_memory": round(memory_score, 4),
            "noise_guard": round(noise_score, 4),
            "weight_balance": round(balance_score, 4),
            "distance_to_price_pct": _round(distance_to_price_pct, 6),
        },
    }


def run_lab(state: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _current_context(state, risk)
    old_threshold = ctx["old_threshold_usd"]
    miner = ctx["miner_standard_usd"]
    p10 = ctx["p10_cohesive_usd"]
    if old_threshold is None or miner is None or p10 is None:
        raise ValueError("Missing old_threshold, miner_standard or p10")

    candidates: List[Dict[str, Any]] = []
    for w in _weights(STEP):
        threshold = _adaptive_threshold(old_threshold, miner, p10, w)
        candidates.append(_score_candidate(threshold, old_threshold, miner, p10, ctx["price_usd"], w))

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    fixed_50_30_20 = _adaptive_threshold(old_threshold, miner, p10, (0.50, 0.30, 0.20))
    candidate_10_30_60 = _adaptive_threshold(old_threshold, miner, p10, (0.10, 0.30, 0.60))

    return {
        "schema": "cohesivx_adaptive_threshold_lab_v0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "branch_note": "Laboratory only. Does not change radar/main logic.",
        "backtest_rule": {
            "historical_validation_start_year": BACKTEST_START_YEAR,
            "excluded_period": "before 2018",
            "status": "current-context grid ranking only; full 2018+ backtest is the next validation step",
        },
        "current_context": {
            "start_price_usd": _round(ctx["start_price_usd"], 2),
            "old_threshold_pct": _round(ctx["old_threshold_pct"], 4),
            "old_threshold_usd": _round(old_threshold, 2),
            "miner_standard_usd": _round(miner, 2),
            "p10_cohesive_usd": _round(p10, 2),
            "spot_price_usd": _round(ctx["price_usd"], 2),
            "model_price_usd": _round(ctx["model_price_usd"], 2),
            "model_deviation_pct": _round(ctx["model_deviation_pct"], 2),
            "ic_context": ctx["ic_context"],
        },
        "benchmarks": {
            "old_fixed_threshold_usd": _round(old_threshold, 2),
            "manual_50_30_20_usd": _round(fixed_50_30_20, 2),
            "manual_10_30_60_usd": _round(candidate_10_30_60, 2),
        },
        "best_current_context_candidate": best,
        "top_10_candidates": candidates[:10],
        "method_warning": "These are current-context coherent candidates, not final validated weights. Final formula requires a 2018+ historical event backtest against realized drawdowns and false alarms.",
    }


def main() -> None:
    state = _load_json(STATE_PATH)
    risk = _load_json(RISK_PATH)
    result = run_lab(state, risk)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    best = result["best_current_context_candidate"]
    print("Adaptive threshold lab written:", OUT_PATH)
    print("Best threshold:", best["threshold_usd"])
    print("Best weights:", best["weights"])
    print("Score:", best["score"])


if __name__ == "__main__":
    main()
