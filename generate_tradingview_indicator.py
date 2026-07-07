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


def _replace_assignment_float(text: str, var_name: str, value: Any, decimals: int = 2) -> str:
    formatted = _fmt(value, decimals=decimals)
    if formatted is None:
        return text

    # Matches: varName = input.float(123.45, "Label", ...)
    pattern = rf"({re.escape(var_name)}\s*=\s*input\.float\()[-+0-9.eE]+(\s*,)"
    new_text, count = re.subn(pattern, rf"\g<1>{formatted}\g<2>", text, count=1)
    if count == 0:
        print(f"WARN: nu am găsit input.float pentru {var_name}")
    return new_text


def _year_decimals(var_prefix: str, year: int, value: Any) -> int:
    if year <= 2014 or (abs(float(value)) < 1 if _finite(value) else False):
        return 5 if var_prefix in ("c", "p10", "p90", "m") else 2
    if var_prefix in ("c", "p10", "p90", "m") and _finite(value) and abs(float(value)) < 100:
        return 2
    return 2


def main() -> None:
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"Lipsește {STATE_PATH}")
    if not PINE_PATH.exists():
        raise FileNotFoundError(f"Lipsește {PINE_PATH}")

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))

    # The workflow already auto-commits coeziv_state.json. Embedding the terminal
    # state here makes the CohesivX OS scores/reasoning update automatically
    # without requiring workflow YAML changes.
    try:
        state["terminal_state"] = build_terminal_state(state)
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Terminal state embedded into coeziv_state.json")
    except Exception as exc:
        print("WARN: nu am putut genera terminal_state:", exc)

    pine = PINE_PATH.read_text(encoding="utf-8")

    anchors = state.get("tradingview_anchors") or {}
    yearly = anchors.get("yearly") or {}
    active_year = str(anchors.get("active_year_override") or state.get("timestamp", "")[:4])
    active = yearly.get(active_year) or {}

    # Current monitor snapshot values used by the table and horizontal snapshot lines.
    pine = _replace_assignment_float(pine, "monitorSpot", state.get("price_usd"), decimals=2)
    pine = _replace_assignment_float(pine, "currentCentral", active.get("central") or state.get("model_price_usd"), decimals=2)
    pine = _replace_assignment_float(pine, "currentP10", active.get("p10") or (state.get("model_price_bands") or {}).get("p10"), decimals=2)
    pine = _replace_assignment_float(pine, "currentP90", active.get("p90") or (state.get("model_price_bands") or {}).get("p90"), decimals=2)
    pine = _replace_assignment_float(pine, "currentEfficientMiner", active.get("efficient_miner") or (state.get("production_costs_usd") or {}).get("cheap"), decimals=2)
    pine = _replace_assignment_float(pine, "currentStandardMiner", active.get("standard_miner") or active.get("miner") or (state.get("production_costs_usd") or {}).get("average"), decimals=2)
    pine = _replace_assignment_float(pine, "currentInefficientMiner", active.get("inefficient_miner") or (state.get("production_costs_usd") or {}).get("expensive"), decimals=2)

    # Historical yearly anchors. Missing years are left unchanged.
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
            prefix = "p10" if var_name.startswith("p10_") else "p90" if var_name.startswith("p90_") else var_name[0]
            pine = _replace_assignment_float(
                pine,
                var_name,
                value,
                decimals=_year_decimals(prefix, year, value),
            )

    PINE_PATH.write_text(pine, encoding="utf-8")

    print("TradingView Pine indicator actualizat:", PINE_PATH)
    print("Monitor spot:", state.get("price_usd"))
    print("Active year:", active_year)
    if active:
        print("Current central:", active.get("central"))
        print("Current p10/p90:", active.get("p10"), active.get("p90"))


if __name__ == "__main__":
    main()
