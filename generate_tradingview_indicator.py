from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any, Dict, Optional

from generate_terminal_state import build_terminal_state

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "btc-swing-strategy" / "coeziv_state.json"
PINE_PATH = ROOT / "Pine" / "indicator.txt"


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


def _fmt(value: Any, decimals: int = 2) -> Optional[str]:
    if not _finite(value):
        return None
    v = float(value)
    if abs(v) < 0.01 and v != 0:
        return f"{v:.5f}"
    if decimals <= 0:
        return f"{v:.0f}"
    return f"{v:.{decimals}f}"


def _clean_string(value: Any, fallback: str = "UNKNOWN") -> str:
    if value is None:
        return fallback
    text = str(value).replace('"', "'").strip()
    return text or fallback


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _replace_assignment_float(text: str, var_name: str, value: Any, decimals: int = 2) -> str:
    formatted = _fmt(value, decimals=decimals)
    if formatted is None:
        return text
    pattern = rf"({re.escape(var_name)}\s*=\s*input\.float\()[-+0-9.eE]+(\s*,)"
    new_text, count = re.subn(pattern, rf"\g<1>{formatted}\g<2>", text, count=1)
    if count == 0:
        print(f"WARN: nu am găsit input.float pentru {var_name}")
    return new_text


def _replace_assignment_string(text: str, var_name: str, value: Any) -> str:
    formatted = _clean_string(value)
    pattern = rf'({re.escape(var_name)}\s*=\s*input\.string\(")[^"]*("\s*,)'
    new_text, count = re.subn(pattern, rf'\g<1>{formatted}\g<2>', text, count=1)
    if count == 0:
        print(f"WARN: nu am găsit input.string pentru {var_name}")
    return new_text


def _year_decimals(year: int, value: Any) -> int:
    if year <= 2014 or (_finite(value) and abs(float(value)) < 1):
        return 5
    return 2


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
        "fg_combined": _sf(fg.get("combined")),
        "model_deviation": _sf(state.get("model_price_deviation")),
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
    # It nudges the multiplier around the projected post-halving windows.
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
    fg_combined = s.get("fg_combined")
    model_deviation = s.get("model_deviation")

    structural_factor = 1.0 + (((structural if structural is not None else 50.0) - 50.0) / 100.0) * 0.16
    confidence_factor = 1.0 + (((confidence if confidence is not None else 50.0) - 50.0) / 100.0) * 0.10
    fragility_factor = 1.0 - (((fragility if fragility is not None else 50.0) - 50.0) / 100.0) * 0.14
    miner_health_factor = 1.0 + (((miner_health if miner_health is not None else 50.0) - 50.0) / 100.0) * 0.10
    flow_factor = 1.0 + _clamp(flow if flow is not None else 0.0, -1.0, 1.0) * 0.06
    fg_factor = 1.0 + (((fg_combined if fg_combined is not None else 50.0) - 50.0) / 100.0) * 0.06

    # Negative model deviation means spot is below the central model. We treat it as mild mean-reversion room,
    # but clamp it so the projection does not become a price target.
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

    # Width logic: high fragility widens downside and restricts upside until the structure improves.
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


def build_projection_10y(state: Dict[str, Any], yearly: Dict[str, Any], active_year: str, active: Dict[str, Any]) -> Dict[str, Any]:
    production_costs = state.get("production_costs_usd") or {}
    bands = state.get("model_price_bands") or {}

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
        n = year - int(active_year)
        projected_miner = base_miner * ((1.0 + annual_cost_growth) ** n)

        # Blend from current structure into the long-term factor. This avoids overreacting to one snapshot.
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
        "base_year": int(active_year),
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


def _ensure_terminal_inputs(pine: str) -> str:
    if "terminalStructuralScore" in pine:
        return pine
    block = """

//━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// COHESIVX TERMINAL SCORES
// Updated automatically from coeziv_state.json / terminal_state
//━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

terminalMarketState = input.string("UNKNOWN", "Market state", group="Terminal scores")
terminalStructuralScore = input.float(0.0, "Structural Score", step=1.0, group="Terminal scores")
terminalConfidenceScore = input.float(0.0, "Confidence Score", step=1.0, group="Terminal scores")
terminalFragilityIndex = input.float(0.0, "Fragility Index", step=1.0, group="Terminal scores")
terminalMinerHealth = input.float(0.0, "Miner Health", step=1.0, group="Terminal scores")
"""
    needle = "currentInefficientMiner = input.float"
    idx = pine.find(needle)
    if idx == -1:
        return pine + block
    end = pine.find("\n", idx)
    return pine[: end + 1] + block + pine[end + 1 :]


def _ensure_terminal_colors(pine: str) -> str:
    if "terminalScoreColor" in pine:
        return pine
    block = """

terminalScoreColor = terminalStructuralScore >= 75 ? color.rgb(0, 220, 140) : terminalStructuralScore >= 55 ? color.rgb(255, 180, 0) : color.rgb(255, 80, 80)
confidenceScoreColor = terminalConfidenceScore >= 75 ? color.rgb(0, 220, 140) : terminalConfidenceScore >= 55 ? color.rgb(255, 180, 0) : color.rgb(255, 80, 80)
fragilityScoreColor = terminalFragilityIndex <= 35 ? color.rgb(0, 220, 140) : terminalFragilityIndex <= 65 ? color.rgb(255, 180, 0) : color.rgb(255, 80, 80)
minerHealthScoreColor = terminalMinerHealth >= 75 ? color.rgb(0, 220, 140) : terminalMinerHealth >= 55 ? color.rgb(255, 180, 0) : color.rgb(255, 80, 80)
"""
    needle = "labelColor = color.white"
    return pine.replace(needle, needle + block, 1)


def _projection_inputs_block() -> str:
    lines = [
        "",
        "//━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "// 10Y STRUCTURAL PROJECTION - NOT A PRICE TARGET",
        "// Generated from objective structural factors, not a fixed price forecast",
        "//━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
    ]
    for year in range(PROJECTION_START_YEAR, PROJECTION_END_YEAR + 1):
        lines.append(f'projC{year} = input.float(0.0, "Projection central {year}", step=1, group="10Y projection central")')
    lines.append("")
    for year in range(PROJECTION_START_YEAR, PROJECTION_END_YEAR + 1):
        lines.append(f'projLow{year} = input.float(0.0, "Projection low {year}", step=1, group="10Y projection low")')
    lines.append("")
    for year in range(PROJECTION_START_YEAR, PROJECTION_END_YEAR + 1):
        lines.append(f'projHigh{year} = input.float(0.0, "Projection high {year} - optional", step=1, group="10Y projection high")')
    lines.append("")
    for year in range(PROJECTION_START_YEAR, PROJECTION_END_YEAR + 1):
        lines.append(f'projMiner{year} = input.float(0.0, "Projection miner {year}", step=1, group="10Y projection miner")')
    return "\n".join(lines) + "\n"


def _ensure_projection_layer(pine: str) -> str:
    pine = pine.replace('indicator("CohesivX BTC Terminal v2.2 - Historical Monitor", overlay=true, max_labels_count=100)', 'indicator("CohesivX BTC Terminal v2.3 - 10Y Structural Projection", overlay=true, max_labels_count=100, max_lines_count=300)')
    pine = pine.replace("// CohesivX BTC Terminal v2.2", "// CohesivX BTC Terminal v2.3")
    pine = pine.replace('table.cell(t, 1, 0, "Terminal v2.2"', 'table.cell(t, 1, 0, "Terminal v2.3"')

    module_needle = 'showSignals = input.bool(false, "Show extreme signal markers", group="Modules")'
    if "showProjectionCentral" not in pine and module_needle in pine:
        projection_modules = '''showProjectionCentral = input.bool(true, "Show 10Y projection central", group="10Y projection")
showProjectionLow = input.bool(true, "Show 10Y projection low", group="10Y projection")
showProjectionHigh = input.bool(false, "Show 10Y projection high - optional", group="10Y projection")
showProjectionMiner = input.bool(true, "Show 10Y projected miner cost", group="10Y projection")
showProjectionLabels = input.bool(true, "Show projection labels", group="10Y projection")'''
        pine = pine.replace(module_needle, module_needle + "\n" + projection_modules, 1)

    input_needle = 'm2026 = input.float'
    if "projC2027" not in pine and input_needle in pine:
        idx = pine.find(input_needle)
        end = pine.find("\n", idx)
        pine = pine[: end + 1] + _projection_inputs_block() + pine[end + 1 :]

    if "t2037 = timestamp" not in pine:
        time_lines = "".join([f"t{year} = timestamp({year}, 1, 1, 0, 0)\n" for year in range(2028, 2038)])
        pine = pine.replace("t2027 = timestamp(2027, 1, 1, 0, 0)\n", "t2027 = timestamp(2027, 1, 1, 0, 0)\n" + time_lines, 1)

    if "projectionCentralColor" not in pine:
        color_block = """
projectionCentralColor = color.rgb(0, 220, 180)
projectionLowColor = color.rgb(80, 180, 255)
projectionHighColor = color.rgb(255, 120, 120)
projectionMinerColor = color.rgb(255, 210, 80)
projectionLabelColor = color.white
"""
        pine = pine.replace("inefficientMinerColor = color.rgb(255, 80, 80)\n", "inefficientMinerColor = color.rgb(255, 80, 80)\n" + color_block, 1)

    if "projectionLines = array.new_line" not in pine:
        draw_block = """
//━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 10Y STRUCTURAL PROJECTION DRAWING
// Projection layer only; not a price target.
//━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

var line[] projectionLines = array.new_line()
var label[] projectionLabels = array.new_label()

f_projection_line(_show, _x1, _y1, _x2, _y2, _clr, _width) =>
    if _show and not na(_y1) and not na(_y2) and _y1 > 0 and _y2 > 0
        array.push(projectionLines, line.new(_x1, _y1, _x2, _y2, xloc=xloc.bar_time, extend=extend.none, color=_clr, width=_width, style=line.style_dashed))

f_projection_label(_show, _x, _y, _txt, _clr) =>
    if _show and not na(_y) and _y > 0
        array.push(projectionLabels, label.new(_x, _y, _txt, xloc=xloc.bar_time, style=label.style_label_left, textcolor=projectionLabelColor, color=color.new(_clr, 20), size=size.tiny))

if barstate.islast
    if array.size(projectionLines) > 0
        for i = 0 to array.size(projectionLines) - 1
            line.delete(array.get(projectionLines, i))
    array.clear(projectionLines)
    if array.size(projectionLabels) > 0
        for i = 0 to array.size(projectionLabels) - 1
            label.delete(array.get(projectionLabels, i))
    array.clear(projectionLabels)

    f_projection_line(showProjectionCentral, t2026, c2026, t2027, projC2027, projectionCentralColor, 3)
    f_projection_line(showProjectionCentral, t2027, projC2027, t2028, projC2028, projectionCentralColor, 3)
    f_projection_line(showProjectionCentral, t2028, projC2028, t2029, projC2029, projectionCentralColor, 3)
    f_projection_line(showProjectionCentral, t2029, projC2029, t2030, projC2030, projectionCentralColor, 3)
    f_projection_line(showProjectionCentral, t2030, projC2030, t2031, projC2031, projectionCentralColor, 3)
    f_projection_line(showProjectionCentral, t2031, projC2031, t2032, projC2032, projectionCentralColor, 3)
    f_projection_line(showProjectionCentral, t2032, projC2032, t2033, projC2033, projectionCentralColor, 3)
    f_projection_line(showProjectionCentral, t2033, projC2033, t2034, projC2034, projectionCentralColor, 3)
    f_projection_line(showProjectionCentral, t2034, projC2034, t2035, projC2035, projectionCentralColor, 3)
    f_projection_line(showProjectionCentral, t2035, projC2035, t2036, projC2036, projectionCentralColor, 3)

    f_projection_line(showProjectionLow, t2026, p10_2026, t2027, projLow2027, projectionLowColor, 1)
    f_projection_line(showProjectionLow, t2027, projLow2027, t2028, projLow2028, projectionLowColor, 1)
    f_projection_line(showProjectionLow, t2028, projLow2028, t2029, projLow2029, projectionLowColor, 1)
    f_projection_line(showProjectionLow, t2029, projLow2029, t2030, projLow2030, projectionLowColor, 1)
    f_projection_line(showProjectionLow, t2030, projLow2030, t2031, projLow2031, projectionLowColor, 1)
    f_projection_line(showProjectionLow, t2031, projLow2031, t2032, projLow2032, projectionLowColor, 1)
    f_projection_line(showProjectionLow, t2032, projLow2032, t2033, projLow2033, projectionLowColor, 1)
    f_projection_line(showProjectionLow, t2033, projLow2033, t2034, projLow2034, projectionLowColor, 1)
    f_projection_line(showProjectionLow, t2034, projLow2034, t2035, projLow2035, projectionLowColor, 1)
    f_projection_line(showProjectionLow, t2035, projLow2035, t2036, projLow2036, projectionLowColor, 1)

    f_projection_line(showProjectionHigh, t2026, p90_2026, t2027, projHigh2027, projectionHighColor, 1)
    f_projection_line(showProjectionHigh, t2027, projHigh2027, t2028, projHigh2028, projectionHighColor, 1)
    f_projection_line(showProjectionHigh, t2028, projHigh2028, t2029, projHigh2029, projectionHighColor, 1)
    f_projection_line(showProjectionHigh, t2029, projHigh2029, t2030, projHigh2030, projectionHighColor, 1)
    f_projection_line(showProjectionHigh, t2030, projHigh2030, t2031, projHigh2031, projectionHighColor, 1)
    f_projection_line(showProjectionHigh, t2031, projHigh2031, t2032, projHigh2032, projectionHighColor, 1)
    f_projection_line(showProjectionHigh, t2032, projHigh2032, t2033, projHigh2033, projectionHighColor, 1)
    f_projection_line(showProjectionHigh, t2033, projHigh2033, t2034, projHigh2034, projectionHighColor, 1)
    f_projection_line(showProjectionHigh, t2034, projHigh2034, t2035, projHigh2035, projectionHighColor, 1)
    f_projection_line(showProjectionHigh, t2035, projHigh2035, t2036, projHigh2036, projectionHighColor, 1)

    f_projection_line(showProjectionMiner, t2026, m2026, t2027, projMiner2027, projectionMinerColor, 2)
    f_projection_line(showProjectionMiner, t2027, projMiner2027, t2028, projMiner2028, projectionMinerColor, 2)
    f_projection_line(showProjectionMiner, t2028, projMiner2028, t2029, projMiner2029, projectionMinerColor, 2)
    f_projection_line(showProjectionMiner, t2029, projMiner2029, t2030, projMiner2030, projectionMinerColor, 2)
    f_projection_line(showProjectionMiner, t2030, projMiner2030, t2031, projMiner2031, projectionMinerColor, 2)
    f_projection_line(showProjectionMiner, t2031, projMiner2031, t2032, projMiner2032, projectionMinerColor, 2)
    f_projection_line(showProjectionMiner, t2032, projMiner2032, t2033, projMiner2033, projectionMinerColor, 2)
    f_projection_line(showProjectionMiner, t2033, projMiner2033, t2034, projMiner2034, projectionMinerColor, 2)
    f_projection_line(showProjectionMiner, t2034, projMiner2034, t2035, projMiner2035, projectionMinerColor, 2)
    f_projection_line(showProjectionMiner, t2035, projMiner2035, t2036, projMiner2036, projectionMinerColor, 2)

    f_projection_label(showProjectionLabels and showProjectionCentral, t2036, projC2036, "10Y central projection\\nnot a price target", projectionCentralColor)
    f_projection_label(showProjectionLabels and showProjectionLow, t2036, projLow2036, "10Y structural low", projectionLowColor)
    f_projection_label(showProjectionLabels and showProjectionMiner, t2036, projMiner2036, "10Y projected miner cost", projectionMinerColor)

"""
        pine = pine.replace("//━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n// TABLE HELPERS", draw_block + "//━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n// TABLE HELPERS", 1)

    return pine


def _patch_table(pine: str) -> str:
    pine = pine.replace('indicator("CohesivX BTC Suite v2.1 - Historical Monitor"', 'indicator("CohesivX BTC Terminal v2.2 - Historical Monitor"')
    pine = pine.replace("// CohesivX BTC Suite v2.1", "// CohesivX BTC Terminal v2.2")
    pine = pine.replace('table.cell(t, 1, 0, "Suite v2.1"', 'table.cell(t, 1, 0, "Terminal v2.2"')
    pine = pine.replace("var table t = table.new(f_pos(tablePosition), 2, 18, border_width=1)", "var table t = table.new(f_pos(tablePosition), 2, 22, border_width=1)")
    pine = pine.replace("for i = 0 to 17", "for i = 0 to 21")
    pine = pine.replace("for i = 6 to 17", "for i = 8 to 21")
    pine = pine.replace("for i = 9 to 17", "for i = 13 to 21")

    minimal_old = """        if tableMode == "Minimal"
            f_row(t, 1, "Chart close", str.tostring(math.round(price)), color.white)
            f_row(t, 2, "Monitor spot", str.tostring(math.round(monitorSpot)), monitorDeviationColor)
            f_row(t, 3, "Model central", str.tostring(math.round(currentCentral)), currentCentralColor)
            f_row(t, 4, "Monitor dev", str.tostring(monitorDeviationPct, "#.##") + "%", monitorDeviationColor)
            f_row(t, 5, "Miner status", minerStatusCurrent + " / " + str.tostring(productionRatioCurrent, "#.##") + "x", minerStatusColor)
            for i = 8 to 21
                f_blank(t, i)
"""
    minimal_new = """        if tableMode == "Minimal"
            f_row(t, 1, "Market State", terminalMarketState, terminalScoreColor)
            f_row(t, 2, "Structural", str.tostring(terminalStructuralScore, "#.##") + "/100", terminalScoreColor)
            f_row(t, 3, "Confidence", str.tostring(terminalConfidenceScore, "#.##") + "/100", confidenceScoreColor)
            f_row(t, 4, "Fragility", str.tostring(terminalFragilityIndex, "#.##") + "/100", fragilityScoreColor)
            f_row(t, 5, "Monitor spot", str.tostring(math.round(monitorSpot)), monitorDeviationColor)
            f_row(t, 6, "Model central", str.tostring(math.round(currentCentral)), currentCentralColor)
            f_row(t, 7, "Miner health", str.tostring(terminalMinerHealth, "#.##") + "/100", minerHealthScoreColor)
            for i = 8 to 21
                f_blank(t, i)
"""
    pine = pine.replace(minimal_old, minimal_new)

    compact_old = """        else if tableMode == "Compact"
            f_row(t, 1, "Chart close", str.tostring(math.round(price)), color.white)
            f_row(t, 2, "Monitor spot", str.tostring(math.round(monitorSpot)), monitorDeviationColor)
            f_row(t, 3, "Source gap", str.tostring(sourceGapPct, "#.##") + "% / " + sourceStatus, sourceGapColor)
            f_row(t, 4, "Model central", str.tostring(math.round(currentCentral)), currentCentralColor)
            f_row(t, 5, "Chart dev", str.tostring(currentDeviationPct, "#.##") + "%", snapshotDeviationColor)
            f_row(t, 6, "Monitor dev", str.tostring(monitorDeviationPct, "#.##") + "%", monitorDeviationColor)
            f_row(t, 7, "Current p10/p90", str.tostring(math.round(currentP10)) + " / " + str.tostring(math.round(currentP90)), currentCentralColor)
            f_row(t, 8, "Miner status", minerStatusCurrent + " / " + str.tostring(productionRatioCurrent, "#.##") + "x", minerStatusColor)
            for i = 13 to 21
                f_blank(t, i)
"""
    compact_new = """        else if tableMode == "Compact"
            f_row(t, 1, "Market State", terminalMarketState, terminalScoreColor)
            f_row(t, 2, "Structural", str.tostring(terminalStructuralScore, "#.##") + "/100", terminalScoreColor)
            f_row(t, 3, "Confidence", str.tostring(terminalConfidenceScore, "#.##") + "/100", confidenceScoreColor)
            f_row(t, 4, "Fragility", str.tostring(terminalFragilityIndex, "#.##") + "/100", fragilityScoreColor)
            f_row(t, 5, "Chart close", str.tostring(math.round(price)), color.white)
            f_row(t, 6, "Monitor spot", str.tostring(math.round(monitorSpot)), monitorDeviationColor)
            f_row(t, 7, "Source gap", str.tostring(sourceGapPct, "#.##") + "% / " + sourceStatus, sourceGapColor)
            f_row(t, 8, "Model central", str.tostring(math.round(currentCentral)), currentCentralColor)
            f_row(t, 9, "Chart dev", str.tostring(currentDeviationPct, "#.##") + "%", snapshotDeviationColor)
            f_row(t, 10, "Monitor dev", str.tostring(monitorDeviationPct, "#.##") + "%", monitorDeviationColor)
            f_row(t, 11, "p10/p90", str.tostring(math.round(currentP10)) + " / " + str.tostring(math.round(currentP90)), currentCentralColor)
            f_row(t, 12, "Miner health", str.tostring(terminalMinerHealth, "#.##") + "/100", minerHealthScoreColor)
            for i = 13 to 21
                f_blank(t, i)
"""
    pine = pine.replace(compact_old, compact_new)
    return pine


def main() -> None:
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"Lipsește {STATE_PATH}")
    if not PINE_PATH.exists():
        raise FileNotFoundError(f"Lipsește {PINE_PATH}")

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))

    try:
        state["terminal_state"] = build_terminal_state(state)
        print("Terminal state embedded into coeziv_state.json")
    except Exception as exc:
        print("WARN: nu am putut genera terminal_state:", exc)

    pine = PINE_PATH.read_text(encoding="utf-8")
    pine = _ensure_terminal_inputs(pine)
    pine = _ensure_terminal_colors(pine)
    pine = _patch_table(pine)
    pine = _ensure_projection_layer(pine)

    anchors = state.get("tradingview_anchors") or {}
    yearly = anchors.get("yearly") or {}
    active_year = str(anchors.get("active_year_override") or state.get("timestamp", "")[:4])
    active = yearly.get(active_year) or {}

    projection = build_projection_10y(state, yearly, active_year, active)
    state["tradingview_projection_10y"] = projection
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    replacements = {
        "monitorSpot": state.get("price_usd"),
        "currentCentral": active.get("central") or state.get("model_price_usd"),
        "currentP10": active.get("p10") or (state.get("model_price_bands") or {}).get("p10"),
        "currentP90": active.get("p90") or (state.get("model_price_bands") or {}).get("p90"),
        "currentEfficientMiner": active.get("efficient_miner") or (state.get("production_costs_usd") or {}).get("cheap"),
        "currentStandardMiner": active.get("standard_miner") or active.get("miner") or (state.get("production_costs_usd") or {}).get("average"),
        "currentInefficientMiner": active.get("inefficient_miner") or (state.get("production_costs_usd") or {}).get("expensive"),
    }
    for var_name, value in replacements.items():
        pine = _replace_assignment_float(pine, var_name, value, decimals=2)

    terminal_scores = (((state.get("terminal_state") or {}).get("scores") or {}).get("scores") or {})
    if not terminal_scores:
        terminal_scores = ((state.get("terminal_state") or {}).get("scores") or {})
    terminal_summary = (state.get("terminal_state") or {}).get("summary") or {}
    pine = _replace_assignment_string(pine, "terminalMarketState", terminal_summary.get("market_state") or terminal_scores.get("market_state"))
    pine = _replace_assignment_float(pine, "terminalStructuralScore", terminal_scores.get("structural_score"), decimals=2)
    pine = _replace_assignment_float(pine, "terminalConfidenceScore", terminal_scores.get("confidence_score"), decimals=2)
    pine = _replace_assignment_float(pine, "terminalFragilityIndex", terminal_scores.get("fragility_index"), decimals=2)
    pine = _replace_assignment_float(pine, "terminalMinerHealth", terminal_scores.get("miner_health"), decimals=2)

    for year_str, data in sorted(yearly.items()):
        try:
            year = int(year_str)
        except Exception:
            continue
        mappings = {
            f"c{year}": data.get("central") or data.get("p50"),
            f"p10_{year}": data.get("p10"),
            f"p90_{year}": data.get("p90"),
            f"m{year}": data.get("standard_miner") or data.get("miner"),
        }
        for var_name, value in mappings.items():
            pine = _replace_assignment_float(pine, var_name, value, decimals=_year_decimals(year, value))

    projection_years = (projection or {}).get("years") or {}
    for year_str, data in sorted(projection_years.items()):
        try:
            year = int(year_str)
        except Exception:
            continue
        mappings = {
            f"projC{year}": data.get("central"),
            f"projLow{year}": data.get("low"),
            f"projHigh{year}": data.get("high"),
            f"projMiner{year}": data.get("miner"),
        }
        for var_name, value in mappings.items():
            pine = _replace_assignment_float(pine, var_name, value, decimals=2)

    PINE_PATH.write_text(pine, encoding="utf-8")
    print("TradingView Pine indicator actualizat:", PINE_PATH)
    print("Monitor spot:", state.get("price_usd"))
    print("Active year:", active_year)
    print("10Y projection:", projection.get("label") if isinstance(projection, dict) else None)


if __name__ == "__main__":
    main()
