from __future__ import annotations

import math
from typing import Any, Dict, Optional


def _num(value: Any) -> Optional[float]:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _score_from_abs_deviation(abs_pct: Optional[float]) -> Optional[float]:
    """100 near central, decays as deviation grows."""
    if abs_pct is None:
        return None
    return _clamp(100.0 - abs_pct * 2.0)


def _miner_health(price: Optional[float], standard_cost: Optional[float]) -> Optional[float]:
    if price is None or standard_cost is None or standard_cost <= 0:
        return None
    ratio = price / standard_cost
    if ratio >= 1.75:
        return 95.0
    if ratio >= 1.50:
        return 88.0
    if ratio >= 1.15:
        return 76.0
    if ratio >= 1.00:
        return 60.0
    if ratio >= 0.85:
        return 42.0
    return 22.0


def _fragility(abs_model_dev_pct: Optional[float], miner_health: Optional[float], confidence: Optional[float]) -> Optional[float]:
    if abs_model_dev_pct is None:
        return None
    miner_penalty = 0.0 if miner_health is None else (100.0 - miner_health) * 0.25
    confidence_penalty = 0.0 if confidence is None else (100.0 - confidence) * 0.25
    deviation_component = min(70.0, abs_model_dev_pct * 1.4)
    return _clamp(deviation_component + miner_penalty + confidence_penalty)


def _confidence(structure_score: Optional[float], miner_health: Optional[float], has_bands: bool, has_anchors: bool) -> Optional[float]:
    pieces = []
    if structure_score is not None:
        pieces.append(structure_score)
    if miner_health is not None:
        pieces.append(miner_health)
    if has_bands:
        pieces.append(85.0)
    if has_anchors:
        pieces.append(90.0)
    if not pieces:
        return None
    return _clamp(sum(pieces) / len(pieces))


def _market_state(model_dev_pct: Optional[float], miner_health: Optional[float], fragility: Optional[float]) -> str:
    if model_dev_pct is None:
        return "UNKNOWN"
    if model_dev_pct <= -15 and (miner_health or 0) >= 55 and (fragility or 100) < 55:
        return "STRUCTURAL_ACCUMULATION"
    if -15 < model_dev_pct < 15 and (fragility or 100) < 45:
        return "STRUCTURAL_BALANCE"
    if model_dev_pct >= 30 and (fragility or 0) >= 45:
        return "STRUCTURAL_OVEREXTENSION"
    if (miner_health or 100) < 45:
        return "MINER_PRESSURE"
    return "MIXED_STRUCTURE"


def build_terminal_scores(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build terminal-level scores from coeziv_state.json.

    These scores are explanatory structural diagnostics, not trading signals.
    """
    price = _num(state.get("price_usd"))
    central = _num(state.get("model_price_usd"))
    model_dev = _num(state.get("model_price_deviation"))
    model_dev_pct = model_dev * 100.0 if model_dev is not None else None
    abs_model_dev_pct = abs(model_dev_pct) if model_dev_pct is not None else None

    bands = state.get("model_price_bands") or {}
    p10 = _num(bands.get("p10"))
    p90 = _num(bands.get("p90"))
    has_bands = p10 is not None and p90 is not None

    costs = state.get("production_costs_usd") or {}
    standard_cost = _num(costs.get("average"))
    miner_health = _miner_health(price, standard_cost)
    structure_score = _score_from_abs_deviation(abs_model_dev_pct)
    has_anchors = bool((state.get("tradingview_anchors") or {}).get("yearly"))
    confidence = _confidence(structure_score, miner_health, has_bands, has_anchors)
    fragility = _fragility(abs_model_dev_pct, miner_health, confidence)

    production_ratio = price / standard_cost if price is not None and standard_cost and standard_cost > 0 else None

    return {
        "market_state": _market_state(model_dev_pct, miner_health, fragility),
        "structural_score": _round(structure_score),
        "miner_health": _round(miner_health),
        "fragility_index": _round(fragility),
        "confidence_score": _round(confidence),
        "model_deviation_pct": _round(model_dev_pct),
        "production_ratio": _round(production_ratio),
        "source_agreement": 100.0,
        "source_agreement_note": "Terminal state uses monitor spot as official source. TradingView can compute chart-vs-monitor source gap from the active chart symbol.",
        "inputs": {
            "price_usd": _round(price),
            "model_price_usd": _round(central),
            "p10": _round(p10),
            "p90": _round(p90),
            "standard_production_cost": _round(standard_cost),
            "has_tradingview_anchors": has_anchors,
        },
    }
