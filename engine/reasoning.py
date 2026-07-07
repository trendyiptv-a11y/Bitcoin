from __future__ import annotations

from typing import Any, Dict, List, Optional


def _num(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _round(value: Any, digits: int = 2) -> Optional[float]:
    v = _num(value)
    if v is None:
        return None
    return round(v, digits)


def _impact(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.0f}"


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else "−" if value < 0 else ""
    return f"{sign}{abs(value):.1f}%"


def _usd(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.0f} USD"


def _reason(factor: str, impact: float, description: str, value: Any = None, category: str = "structure") -> Dict[str, Any]:
    return {
        "factor": factor,
        "impact": _impact(impact),
        "impact_value": round(float(impact), 2),
        "category": category,
        "value": value,
        "description": description,
    }


def _summary(market_state: str, model_dev_pct: Optional[float], miner_health: Optional[float], fragility: Optional[float]) -> str:
    if market_state == "STRUCTURAL_ACCUMULATION":
        return "Bitcoin is below the CohesivX structural value while miner context remains acceptable. The terminal classifies this as structural accumulation, not as a short-term trading signal."
    if market_state == "STRUCTURAL_BALANCE":
        return "Bitcoin is close to the CohesivX structural value. The terminal reads the market as structurally balanced."
    if market_state == "STRUCTURAL_OVEREXTENSION":
        return "Bitcoin is materially above the CohesivX structural value and fragility is elevated. The terminal flags structural overextension."
    if market_state == "MINER_PRESSURE":
        return "Miner profitability is weak relative to the standard production-cost anchor. The terminal flags miner pressure."
    if model_dev_pct is None:
        return "The terminal cannot produce a strong structural summary because the model deviation is missing."
    if fragility is not None and fragility > 65:
        return "The terminal sees a mixed structure with elevated fragility. Interpretation should be cautious."
    if miner_health is not None and miner_health > 75:
        return "The terminal sees a mixed structure, but miner health remains supportive."
    return "The terminal sees a mixed structural state. No single dominant condition controls the interpretation."


def build_reasoning(state: Dict[str, Any], scores: Dict[str, Any]) -> Dict[str, Any]:
    """Build an auditable reasoning block for CohesivX Terminal.

    The output is intentionally simple and JSON-friendly so it can be used by
    website UI, Pine tables, Telegram summaries and future AI assistants.
    """
    price = _num(state.get("price_usd"))
    central = _num(state.get("model_price_usd"))
    model_dev_pct = _num(scores.get("model_deviation_pct"))
    structural_score = _num(scores.get("structural_score"))
    miner_health = _num(scores.get("miner_health"))
    fragility = _num(scores.get("fragility_index"))
    confidence = _num(scores.get("confidence_score"))
    production_ratio = _num(scores.get("production_ratio"))
    market_state = str(scores.get("market_state") or "UNKNOWN")

    costs = state.get("production_costs_usd") or {}
    standard_cost = _num(costs.get("average"))
    bands = state.get("model_price_bands") or {}
    p10 = _num(bands.get("p10"))
    p90 = _num(bands.get("p90"))
    has_anchors = bool((state.get("tradingview_anchors") or {}).get("yearly"))

    reasons: List[Dict[str, Any]] = []

    if model_dev_pct is not None:
        if model_dev_pct <= -15:
            reasons.append(_reason(
                "Price below structural value",
                22,
                f"Monitor spot is {_pct(model_dev_pct)} below the CohesivX central value. This supports an accumulation-style structural reading.",
                {"price_usd": _round(price), "model_price_usd": _round(central), "deviation_pct": _round(model_dev_pct)},
                "structure",
            ))
        elif model_dev_pct >= 30:
            reasons.append(_reason(
                "Price above structural value",
                -20,
                f"Monitor spot is {_pct(model_dev_pct)} above the CohesivX central value. This increases overextension risk.",
                {"price_usd": _round(price), "model_price_usd": _round(central), "deviation_pct": _round(model_dev_pct)},
                "structure",
            ))
        else:
            reasons.append(_reason(
                "Price near structural value",
                14,
                f"Monitor spot is {_pct(model_dev_pct)} from the CohesivX central value. This supports a balanced structural reading.",
                {"price_usd": _round(price), "model_price_usd": _round(central), "deviation_pct": _round(model_dev_pct)},
                "structure",
            ))

    if miner_health is not None:
        if miner_health >= 75:
            reasons.append(_reason(
                "Miner health supportive",
                18,
                f"Monitor spot is {production_ratio:.2f}× the standard production-cost anchor. Miner context is supportive.",
                {"production_ratio": _round(production_ratio), "standard_cost": _round(standard_cost)},
                "miners",
            ))
        elif miner_health >= 55:
            reasons.append(_reason(
                "Miner health neutral",
                8,
                f"Monitor spot is {production_ratio:.2f}× the standard production-cost anchor. Miner context is acceptable but not strong.",
                {"production_ratio": _round(production_ratio), "standard_cost": _round(standard_cost)},
                "miners",
            ))
        else:
            reasons.append(_reason(
                "Miner pressure",
                -18,
                f"Monitor spot is only {production_ratio:.2f}× the standard production-cost anchor. Miner pressure weakens the structure.",
                {"production_ratio": _round(production_ratio), "standard_cost": _round(standard_cost)},
                "miners",
            ))

    if p10 is not None and p90 is not None and price is not None:
        if price < p10:
            reasons.append(_reason(
                "Below statistical lower band",
                -10,
                f"Monitor spot is below p10 ({_usd(p10)}). This is statistically unusual and should be interpreted carefully.",
                {"p10": _round(p10), "p90": _round(p90)},
                "statistics",
            ))
        elif price > p90:
            reasons.append(_reason(
                "Above statistical upper band",
                -14,
                f"Monitor spot is above p90 ({_usd(p90)}). This indicates statistical overextension.",
                {"p10": _round(p10), "p90": _round(p90)},
                "statistics",
            ))
        else:
            reasons.append(_reason(
                "Inside statistical model band",
                12,
                f"Monitor spot remains inside the CohesivX p10/p90 statistical band ({_usd(p10)} – {_usd(p90)}).",
                {"p10": _round(p10), "p90": _round(p90)},
                "statistics",
            ))

    if has_anchors:
        reasons.append(_reason(
            "Historical anchors available",
            10,
            "TradingView yearly anchors are present, so the terminal can preserve the link between current structure and historical model context.",
            {"anchor_years": sorted((state.get("tradingview_anchors") or {}).get("yearly", {}).keys())},
            "history",
        ))
    else:
        reasons.append(_reason(
            "Historical anchors missing",
            -8,
            "TradingView yearly anchors are missing. The live terminal still works, but historical projection is weaker.",
            None,
            "history",
        ))

    if confidence is not None:
        if confidence >= 80:
            reasons.append(_reason(
                "High interpretation confidence",
                14,
                f"Confidence score is {confidence:.0f}/100. The available data supports the current structural interpretation.",
                {"confidence_score": _round(confidence)},
                "confidence",
            ))
        elif confidence >= 60:
            reasons.append(_reason(
                "Medium interpretation confidence",
                4,
                f"Confidence score is {confidence:.0f}/100. The interpretation is usable but should be checked against source context.",
                {"confidence_score": _round(confidence)},
                "confidence",
            ))
        else:
            reasons.append(_reason(
                "Low interpretation confidence",
                -14,
                f"Confidence score is {confidence:.0f}/100. The terminal should be read cautiously.",
                {"confidence_score": _round(confidence)},
                "confidence",
            ))

    if fragility is not None:
        if fragility >= 70:
            reasons.append(_reason(
                "High fragility",
                -18,
                f"Fragility index is {fragility:.0f}/100. The structure is sensitive to shocks or source disagreement.",
                {"fragility_index": _round(fragility)},
                "fragility",
            ))
        elif fragility <= 35:
            reasons.append(_reason(
                "Low fragility",
                12,
                f"Fragility index is {fragility:.0f}/100. Structural stress is limited under the current monitor assumptions.",
                {"fragility_index": _round(fragility)},
                "fragility",
            ))
        else:
            reasons.append(_reason(
                "Moderate fragility",
                2,
                f"Fragility index is {fragility:.0f}/100. The structure is neither clearly fragile nor fully relaxed.",
                {"fragility_index": _round(fragility)},
                "fragility",
            ))

    return {
        "market_state": market_state,
        "summary": _summary(market_state, model_dev_pct, miner_health, fragility),
        "headline": market_state.replace("_", " ").title(),
        "scores": {
            "structural_score": _round(structural_score),
            "miner_health": _round(miner_health),
            "fragility_index": _round(fragility),
            "confidence_score": _round(confidence),
        },
        "reasoning": reasons,
    }
