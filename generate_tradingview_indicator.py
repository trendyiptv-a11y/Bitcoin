from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Optional

from generate_terminal_state import build_terminal_state

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "btc-swing-strategy" / "coeziv_state.json"
PINE_PATH = ROOT / "Pine" / "indicator.txt"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


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
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Terminal state embedded into coeziv_state.json")
    except Exception as exc:
        print("WARN: nu am putut genera terminal_state:", exc)

    pine = PINE_PATH.read_text(encoding="utf-8")
    pine = _ensure_terminal_inputs(pine)
    pine = _ensure_terminal_colors(pine)
    pine = _patch_table(pine)

    anchors = state.get("tradingview_anchors") or {}
    yearly = anchors.get("yearly") or {}
    active_year = str(anchors.get("active_year_override") or state.get("timestamp", "")[:4])
    active = yearly.get(active_year) or {}

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

    PINE_PATH.write_text(pine, encoding="utf-8")
    print("TradingView Pine indicator actualizat:", PINE_PATH)
    print("Monitor spot:", state.get("price_usd"))
    print("Active year:", active_year)


if __name__ == "__main__":
    main()
