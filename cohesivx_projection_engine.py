from __future__ import annotations

import math
from statistics import median
from typing import Any, Dict, Optional

PROJECTION_START_YEAR = 2027
PROJECTION_END_YEAR = 2036


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _sf(value: Any) -> Optional[float]:
    if not _finite(value):
        return None
    return float(value)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clean_string(value: Any, fallback: str = "UNKNOWN") -> str:
    if value is None:
        return fallback
    text = str(value).replace('"', "'").strip()
    return text or fallback


def _standard_miner(data: Dict[str, Any]) -> Optional[float]:
    return _sf(data.get("standard_miner") or data.get("miner"))


def _projection_growth_rate(yearly: Dict[str, Any], active_year: str) -> float:
    years = []
    for year_str, data in yearly.items():
        try:
            year = int(year_str)
        except Exception:
            continue
        if str(year) > active_year:
            continue
        miner = _standard_miner(data or {})
        if miner and miner > 0:
            years.append((year, miner))
    years.sort()

    growth_rates = []
    for (_, prev), (_, cur) in zip(years, years[1:]):
        if prev > 0 and cur > 0:
            rate = (cur / prev) - 1.0
            if -0.35 <= rate <= 1.25:
                growth_rates.append(rate)

    if not growth_rates:
        return 0.16

    recent = growth_rates[-8:]
    return _clamp(float(median(recent)), 0.06, 0.35)


def _terminal_scores(state: Dict[str, Any]) -> Dict[str, Optional[float]]:
    scores = (((state.get("terminal_state") or {}).get("scores") or {}).get("scores") or {})
    if not scores:
        scores = ((state.get("terminal_state") or {}).get("scores") or {})
    fg = state.get("fg") or {}
    return {
        "structural_score": _sf(scores.get("structural_score")),
        "confidence_score": _sf(scores.get("confidence_score")),
        "fragility_index": _sf(scores.get("fragility_index")),
        "miner_health": _sf(scores.get("miner_health")),
        "flow_score": _sf(state.get("flow_score")),
        "fear_greed_combined": _sf(fg.get("combined")),
        "model_price_deviation": _sf(state.get("model_price_deviation")),
    }


def _regime_factor(state: Dict[str, Any]) -> float:
    regime = _clean_string(state.get("regime") or state.get("current_regime"), "neutral").lower()
    signal = _clean_string(state.get("signal"), "flat").lower()
    if "bear_struct" in regime:
        base = 0.92
    elif "bear_late" in regime:
        base = 0.96
    elif "accum" in regime:
        base = 1.01
    elif "bull" in regime or "expansion" in regime:
        base = 1.08
    else:
        base = 1.00
    if signal == "long":
        base *= 1.03
    elif signal == "short":
        base *= 0.97
    return _clamp(base, 0.85, 1.15)


def _cycle_halving_factor(year: int) -> float:
    # Structural supply-pressure factor, not a price promise.
    # It nudges the projection around expected post-halving windows.
    cycle_map = {
        2027: 1.02,
        2028: 1.05,
        2029: 1.10,
        2030: 1.06,
        2031: 1.02,
        2032: 1.05,
        2033: 1.10,
        2034: 1.06,
        2035: 1.02,
        2036: 1.05,
    }
    return cycle_map.get(year, 1.0)


def _cohesive_factor_pack(state: Dict[str, Any]) -> Dict[str, Any]:
    s = _terminal_scores(state)
    structural = s.get("structural_score")
    confidence = s.get("confidence_score")
    fragility = s.get("fragility_index")
    miner_health = s.get("miner_health")
    flow = s.get("flow_score")
    fg_combined = s.get("fear_greed_combined")
    model_deviation = s.get("model_price_deviation")

    structural_factor = 1.0 + (((structural if structural is not None else 50.0) - 50.0) / 100.0) * 0.16
    confidence_factor = 1.0 + (((confidence if confidence is not None else 50.0) - 50.0) / 100.0) * 0.10
    fragility_factor = 1.0 - (((fragility if fragility is not None else 50.0) - 50.0) / 100.0) * 0.14
    miner_health_factor = 1.0 + (((miner_health if miner_health is not None else 50.0) - 50.0) / 100.0) * 0.10
    flow_factor = 1.0 + _clamp(flow if flow is not None else 0.0, -1.0, 1.0) * 0.06
    fg_factor = 1.0 + (((fg_combined if fg_combined is not None else 50.0) - 50.0) / 100.0) * 0.06

    # Negative deviation means spot is below the central model. This is only mild mean-reversion room.
    deviation_factor = 1.0 + _clamp(-(model_deviation if model_deviation is not None else 0.0), -0.35, 0.55) * 0.08
    regime_factor = _regime_factor(state)

    combined = _clamp(
        structural_factor
        * confidence_factor
        * fragility_factor
        * miner_health_factor
        * flow_factor
        * fg_factor
        * deviation_factor
        * regime_factor,
        0.72,
        1.32,
    )

    downside_width = _clamp(1.0 - (((fragility if fragility is not None else 50.0) - 50.0) / 100.0) * 0.18, 0.78, 1.08)
    upside_width = _clamp(1.0 + (((confidence if confidence is not None else 50.0) - 50.0) / 100.0) * 0.14, 0.86, 1.18)

    return {
        "combined_factor": combined,
        "downside_width_factor": downside_width,
        "upside_width_factor": upside_width,
        "components": {
            "structural_factor": _clamp(structural_factor, 0.85, 1.15),
            "confidence_factor": _clamp(confidence_factor, 0.90, 1.10),
            "fragility_factor": _clamp(fragility_factor, 0.86, 1.14),
            "miner_health_factor": _clamp(miner_health_factor, 0.90, 1.10),
            "flow_factor": _clamp(flow_factor, 0.94, 1.06),
            "fear_greed_factor": _clamp(fg_factor, 0.94, 1.06),
            "mean_reversion_factor": _clamp(deviation_factor, 0.96, 1.05),
            "regime_factor": regime_factor,
        },
        "inputs": s,
    }


def build_10y_structural_projection(state: Dict[str, Any]) -> Dict[str, Any]:
    """Official CohesivX 10Y structural projection engine.

    This is the single source of truth for 10Y projection values.
    TradingView, website cards, reports, or future exports should consume this
    function instead of rebuilding projection logic locally.

    Output is a structural corridor, not a price target.
    """
    anchors = state.get("tradingview_anchors") or {}
    yearly = anchors.get("yearly") or {}
    active_year = str(anchors.get("active_year_override") or state.get("timestamp", "")[:4])
    active = yearly.get(active_year) or {}
    production_costs = state.get("production_costs_usd") or {}
    bands = state.get("model_price_bands") or {}

    try:
        base_year = int(active_year)
    except Exception:
        return {"active": False, "reason": "invalid_active_year"}

    base_miner = (
        _standard_miner(active)
        or _sf(production_costs.get("average"))
        or _sf(production_costs.get("standard"))
    )
    central = _sf(active.get("central") or active.get("p50") or state.get("model_price_usd"))
    low = _sf(active.get("p10") or bands.get("p10"))
    high = _sf(active.get("p90") or bands.get("p90"))

    if not base_miner or base_miner <= 0:
        return {"active": False, "reason": "missing_base_miner"}

    central_mult = _clamp((central or base_miner) / base_miner, 0.75, 8.0)
    low_mult = _clamp((low or base_miner * 0.9) / base_miner, 0.25, central_mult)
    high_mult = _clamp((high or base_miner * 4.0) / base_miner, central_mult, 12.0)
    annual_cost_growth = _projection_growth_rate(yearly, active_year)
    factor_pack = _cohesive_factor_pack(state)
    base_cohesive_factor = float(factor_pack["combined_factor"])
    downside_width = float(factor_pack["downside_width_factor"])
    upside_width = float(factor_pack["upside_width_factor"])

    out_years: Dict[str, Dict[str, float]] = {}
    for year in range(PROJECTION_START_YEAR, PROJECTION_END_YEAR + 1):
        n = year - base_year
        projected_miner = base_miner * ((1.0 + annual_cost_growth) ** n)
        structural_blend = min(1.0, max(0.25, n / 4.0))
        cohesive_factor = 1.0 + (base_cohesive_factor - 1.0) * structural_blend
        cycle_factor = _cycle_halving_factor(year)
        central_factor = _clamp(cohesive_factor * cycle_factor, 0.72, 1.45)

        out_years[str(year)] = {
            "miner": projected_miner,
            "low": projected_miner * low_mult * _clamp(min(central_factor, 1.0) * downside_width, 0.65, 1.10),
            "central": projected_miner * central_mult * central_factor,
            "high": projected_miner * high_mult * _clamp(max(central_factor, 1.0) * upside_width, 0.85, 1.55),
            "cohesive_factor": central_factor,
            "cycle_halving_factor": cycle_factor,
        }

    return {
        "active": True,
        "label": "10Y Structural Projection - not a price target",
        "engine": "cohesivx_projection_engine.build_10y_structural_projection",
        "base_year": base_year,
        "start_year": PROJECTION_START_YEAR,
        "end_year": PROJECTION_END_YEAR,
        "method": "projected production cost × structural price/cost multipliers × cohesive objective factors",
        "objective_factors_used": [
            "production_cost_trend",
            "p10_p50_p90_price_cost_multipliers",
            "terminal_structural_score",
            "terminal_confidence_score",
            "terminal_fragility_index",
            "miner_health",
            "flow_score",
            "fear_greed_combined",
            "model_price_deviation_mean_reversion",
            "current_regime_and_signal",
            "halving_cycle_supply_pressure",
        ],
        "annual_cost_growth_used": annual_cost_growth,
        "multipliers": {
            "low_price_cost": low_mult,
            "central_price_cost": central_mult,
            "high_price_cost": high_mult,
        },
        "cohesive_factor_pack": factor_pack,
        "years": out_years,
    }
