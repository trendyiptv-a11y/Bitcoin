from __future__ import annotations

import math
from statistics import median
from typing import Any, Dict, Optional

PROJECTION_START_YEAR = 2027
PROJECTION_END_YEAR = 2036
PRIMARY_CALIBRATION_START_YEAR = 2022
FALLBACK_CALIBRATION_START_YEAR = 2018
EXCLUDED_CALIBRATION_BEFORE_YEAR = 2018


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


def _actual_close(data: Dict[str, Any]) -> Optional[float]:
    return _sf(data.get("ic_close") or data.get("close") or data.get("spot_price_usd"))


def _miner_years(yearly: Dict[str, Any], active_year: str, start_year: int) -> list[tuple[int, float]]:
    years: list[tuple[int, float]] = []
    for year_str, data in yearly.items():
        try:
            year = int(year_str)
        except Exception:
            continue
        if year < start_year or str(year) > active_year:
            continue
        miner = _standard_miner(data or {})
        if miner and miner > 0:
            years.append((year, miner))
    years.sort()
    return years


def _rates_from_years(years: list[tuple[int, float]]) -> list[float]:
    growth_rates: list[float] = []
    for (_, prev), (_, cur) in zip(years, years[1:]):
        if prev > 0 and cur > 0:
            rate = (cur / prev) - 1.0
            if -0.35 <= rate <= 1.25:
                growth_rates.append(rate)
    return growth_rates


def _projection_growth_profile(yearly: Dict[str, Any], active_year: str) -> Dict[str, Any]:
    """Return the production-cost growth assumption under the official calibration policy.

    Policy:
    - 2022+ is the primary calibration window.
    - 2018+ is used only as fallback/robustness when 2022+ is too thin.
    - pre-2018 BTC is excluded from projection-weight calibration.
    """
    primary_years = _miner_years(yearly, active_year, PRIMARY_CALIBRATION_START_YEAR)
    fallback_years = _miner_years(yearly, active_year, FALLBACK_CALIBRATION_START_YEAR)
    primary_rates = _rates_from_years(primary_years)
    fallback_rates = _rates_from_years(fallback_years)

    if len(primary_rates) >= 2:
        raw_rate = float(median(primary_rates))
        source = "primary_2022_plus"
        used_years = [year for year, _ in primary_years]
        sample_count = len(primary_rates)
    elif fallback_rates:
        raw_rate = float(median(fallback_rates[-8:]))
        source = "fallback_2018_plus"
        used_years = [year for year, _ in fallback_years]
        sample_count = len(fallback_rates)
    else:
        raw_rate = 0.16
        source = "default_no_sufficient_history"
        used_years = []
        sample_count = 0

    capped_rate = _clamp(raw_rate, 0.06, 0.35)
    return {
        "annual_cost_growth_used": capped_rate,
        "raw_annual_cost_growth": raw_rate,
        "source": source,
        "primary_start_year": PRIMARY_CALIBRATION_START_YEAR,
        "fallback_start_year": FALLBACK_CALIBRATION_START_YEAR,
        "excluded_before_year": EXCLUDED_CALIBRATION_BEFORE_YEAR,
        "used_years": used_years,
        "sample_count": sample_count,
        "cap_range": [0.06, 0.35],
        "note": "2022+ is the official primary calibration window; 2018+ is fallback only; pre-2018 is excluded.",
    }


def _projection_growth_rate(yearly: Dict[str, Any], active_year: str) -> float:
    return float(_projection_growth_profile(yearly, active_year)["annual_cost_growth_used"])


def _empirical_output_calibration(yearly: Dict[str, Any], active_year: str) -> Dict[str, Any]:
    """Calibrate projection output from observed 2022+ anchors.

    This does not attempt direct 10Y calibration. It measures how yearly observed
    BTC closes related to the mechanism's central/band/cost anchors in the modern
    2022+ window and applies a controlled output-realistization factor.
    """
    rows: list[Dict[str, float]] = []
    for year_str, data in yearly.items():
        try:
            year = int(year_str)
        except Exception:
            continue
        if year < PRIMARY_CALIBRATION_START_YEAR or str(year) > active_year:
            continue
        data = data or {}
        actual = _actual_close(data)
        central = _sf(data.get("central") or data.get("p50"))
        miner = _standard_miner(data)
        low = _sf(data.get("p10"))
        high = _sf(data.get("p90"))
        if not actual or actual <= 0:
            continue
        row: Dict[str, float] = {"year": float(year), "actual": actual}
        if central and central > 0:
            row["actual_to_central"] = actual / central
        if miner and miner > 0:
            row["actual_to_miner"] = actual / miner
        if low and low > 0:
            row["actual_to_low"] = actual / low
        if high and high > 0:
            row["actual_to_high"] = actual / high
        rows.append(row)
    rows.sort(key=lambda item: item["year"])

    central_ratios = [r["actual_to_central"] for r in rows if "actual_to_central" in r and 0.05 <= r["actual_to_central"] <= 3.0]
    miner_ratios = [r["actual_to_miner"] for r in rows if "actual_to_miner" in r and 0.05 <= r["actual_to_miner"] <= 20.0]
    low_ratios = [r["actual_to_low"] for r in rows if "actual_to_low" in r and 0.05 <= r["actual_to_low"] <= 20.0]
    high_ratios = [r["actual_to_high"] for r in rows if "actual_to_high" in r and 0.0001 <= r["actual_to_high"] <= 3.0]

    if len(central_ratios) >= 3:
        raw_central = float(median(central_ratios))
        source = "primary_2022_plus_observed_output"
    elif len(central_ratios) >= 1:
        raw_central = float(median(central_ratios))
        source = "thin_2022_plus_observed_output"
    else:
        raw_central = 1.0
        source = "neutral_no_output_calibration"

    central_factor = _clamp(raw_central, 0.55, 1.25)

    # Apply the empirical correction asymmetrically:
    # - central receives the direct modern observed correction;
    # - low receives a gentler correction so the defense band is not overcompressed;
    # - high receives a stronger correction when recent history says the model was overextended.
    low_factor = _clamp(1.0 + (central_factor - 1.0) * 0.35, 0.78, 1.10)
    high_factor = _clamp(1.0 + (central_factor - 1.0) * 0.75, 0.58, 1.20)
    miner_factor = 1.0

    return {
        "active": source != "neutral_no_output_calibration",
        "source": source,
        "primary_start_year": PRIMARY_CALIBRATION_START_YEAR,
        "fallback_start_year": FALLBACK_CALIBRATION_START_YEAR,
        "excluded_before_year": EXCLUDED_CALIBRATION_BEFORE_YEAR,
        "sample_count": len(rows),
        "used_years": [int(r["year"]) for r in rows],
        "raw_central_output_factor": raw_central,
        "central_output_factor": central_factor,
        "low_output_factor": low_factor,
        "high_output_factor": high_factor,
        "miner_output_factor": miner_factor,
        "cap_ranges": {
            "central_output_factor": [0.55, 1.25],
            "low_output_factor": [0.78, 1.10],
            "high_output_factor": [0.58, 1.20],
        },
        "diagnostics": {
            "actual_to_central_median": float(median(central_ratios)) if central_ratios else None,
            "actual_to_miner_median": float(median(miner_ratios)) if miner_ratios else None,
            "actual_to_low_median": float(median(low_ratios)) if low_ratios else None,
            "actual_to_high_median": float(median(high_ratios)) if high_ratios else None,
        },
        "note": "Empirical 2022+ output calibration realistizes the structural corridor; it is not direct 10Y fitting.",
    }


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
        "weight_status": "beta_empirical_output_calibration_2022_plus_manual_component_weights",
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
    growth_profile = _projection_growth_profile(yearly, active_year)
    annual_cost_growth = float(growth_profile["annual_cost_growth_used"])
    output_calibration = _empirical_output_calibration(yearly, active_year)
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

        alpha_low = projected_miner * low_mult * _clamp(min(central_factor, 1.0) * downside_width, 0.65, 1.10)
        alpha_central = projected_miner * central_mult * central_factor
        alpha_high = projected_miner * high_mult * _clamp(max(central_factor, 1.0) * upside_width, 0.85, 1.55)

        beta_low = alpha_low * float(output_calibration["low_output_factor"])
        beta_central = alpha_central * float(output_calibration["central_output_factor"])
        beta_high = alpha_high * float(output_calibration["high_output_factor"])
        beta_miner = projected_miner * float(output_calibration["miner_output_factor"])

        out_years[str(year)] = {
            "miner": beta_miner,
            "low": beta_low,
            "central": beta_central,
            "high": beta_high,
            "alpha_unadjusted": {
                "miner": projected_miner,
                "low": alpha_low,
                "central": alpha_central,
                "high": alpha_high,
            },
            "cohesive_factor": central_factor,
            "cycle_halving_factor": cycle_factor,
            "empirical_output_factors": {
                "low": output_calibration["low_output_factor"],
                "central": output_calibration["central_output_factor"],
                "high": output_calibration["high_output_factor"],
                "miner": output_calibration["miner_output_factor"],
            },
        }

    return {
        "active": True,
        "label": "10Y Structural Projection - not a price target",
        "engine": "cohesivx_projection_engine.build_10y_structural_projection",
        "engine_status": "beta_empirical_output_calibration_2022_plus",
        "calibration_policy": {
            "primary_start_year": PRIMARY_CALIBRATION_START_YEAR,
            "fallback_start_year": FALLBACK_CALIBRATION_START_YEAR,
            "excluded_before_year": EXCLUDED_CALIBRATION_BEFORE_YEAR,
            "primary_window": "2022+",
            "fallback_window": "2018+ only if 2022+ is too thin",
            "pre_2018_policy": "excluded_from_projection_weights",
            "direct_10y_calibration": False,
            "note": "Production-cost growth uses 2022+ as primary calibration; output is realistized by empirical 2022+ observed close/model ratios.",
        },
        "base_year": base_year,
        "start_year": PROJECTION_START_YEAR,
        "end_year": PROJECTION_END_YEAR,
        "method": "alpha structural corridor × empirical 2022+ output calibration",
        "objective_factors_used": [
            "production_cost_trend_2022_plus_primary",
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
            "empirical_2022_plus_output_calibration",
        ],
        "annual_cost_growth_used": annual_cost_growth,
        "growth_profile": growth_profile,
        "empirical_output_calibration": output_calibration,
        "multipliers": {
            "low_price_cost": low_mult,
            "central_price_cost": central_mult,
            "high_price_cost": high_mult,
        },
        "cohesive_factor_pack": factor_pack,
        "years": out_years,
    }
