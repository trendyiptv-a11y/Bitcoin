from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

from cohesivx_projection_engine import PROJECTION_END_YEAR, PROJECTION_START_YEAR

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "btc-swing-strategy" / "coeziv_state.json"
SEED_DIR = ROOT / "TradingViewSeeds"
PINE_SEED_PATH = ROOT / "Pine" / "indicator_seed_request.txt"

SEED_SOURCE = "trendyiptv-a11y/Bitcoin"
CURRENT_SYMBOL = "COHESIVX_BTC_CURRENT"
PROJ_SYMBOL_PREFIX = "COHESIVX_BTC_PROJ_"


def _sf(value: Any) -> Optional[float]:
    try:
        v = float(value)
        if v == v and abs(v) != float("inf"):
            return v
    except Exception:
        return None
    return None


def _fmt(value: Any) -> str:
    v = _sf(value)
    if v is None:
        return ""
    return f"{v:.8f}".rstrip("0").rstrip(".")


def _date_from_state(state: Dict[str, Any]) -> str:
    # request.seed() reads EOD-style rows. We keep all projection values on the
    # latest monitor date so Pine can hold them with ta.valuewhen() and draw future lines.
    ts = str(state.get("timestamp") or state.get("generated_at") or "")
    if len(ts) >= 10:
        return ts[:10]
    return "2026-01-01"


def _write_seed_csv(path: Path, date: str, open_: Any, high: Any, low: Any, close: Any, volume: Any = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "open", "high", "low", "close", "volume"])
        writer.writerow([date, _fmt(open_), _fmt(high), _fmt(low), _fmt(close), _fmt(volume)])


def _pine_seed_template() -> str:
    years = list(range(PROJECTION_START_YEAR, PROJECTION_END_YEAR + 1))

    lines: list[str] = [
        "//@version=5",
        'indicator("CohesivX BTC Terminal v2.4-seed - Auto Projection", overlay=true, max_lines_count=300, max_labels_count=80)',
        "",
        "// Experimental auto-data version.",
        "// GitHub Actions updates CSV seed files; Pine reads them with request.seed().",
        "// Projection layer only; not a price target.",
        "",
        f'seedSource = "{SEED_SOURCE}"',
        'showProjectionCentral = input.bool(true, "Show 10Y projection central", group="10Y projection")',
        'showProjectionLow = input.bool(true, "Show 10Y projection low", group="10Y projection")',
        'showProjectionHigh = input.bool(false, "Show 10Y projection high - optional", group="10Y projection")',
        'showProjectionMiner = input.bool(true, "Show 10Y projected miner cost", group="10Y projection")',
        'showProjectionLabels = input.bool(true, "Show projection labels", group="10Y projection")',
        "",
        "f_hold(_x) =>",
        "    ta.valuewhen(not na(_x), _x, 0)",
        "",
        f'curCentral = f_hold(request.seed(seedSource, "{CURRENT_SYMBOL}", open, ignore_invalid_symbol=true))',
        f'curP90 = f_hold(request.seed(seedSource, "{CURRENT_SYMBOL}", high, ignore_invalid_symbol=true))',
        f'curP10 = f_hold(request.seed(seedSource, "{CURRENT_SYMBOL}", low, ignore_invalid_symbol=true))',
        f'curMiner = f_hold(request.seed(seedSource, "{CURRENT_SYMBOL}", close, ignore_invalid_symbol=true))',
        "",
    ]

    for year in years:
        sym = f"{PROJ_SYMBOL_PREFIX}{year}"
        lines.extend([
            f'projC{year} = f_hold(request.seed(seedSource, "{sym}", open, ignore_invalid_symbol=true))',
            f'projHigh{year} = f_hold(request.seed(seedSource, "{sym}", high, ignore_invalid_symbol=true))',
            f'projLow{year} = f_hold(request.seed(seedSource, "{sym}", low, ignore_invalid_symbol=true))',
            f'projMiner{year} = f_hold(request.seed(seedSource, "{sym}", close, ignore_invalid_symbol=true))',
            "",
        ])

    lines.extend([
        "t2026 = timestamp(2026, 1, 1, 0, 0)",
    ])
    for year in years:
        lines.append(f"t{year} = timestamp({year}, 1, 1, 0, 0)")

    lines.extend([
        "",
        "projectionCentralColor = color.rgb(0, 220, 180)",
        "projectionLowColor = color.rgb(80, 180, 255)",
        "projectionHighColor = color.rgb(255, 120, 120)",
        "projectionMinerColor = color.rgb(255, 210, 80)",
        "projectionLabelColor = color.white",
        "",
        "var line[] projectionLines = array.new_line()",
        "var label[] projectionLabels = array.new_label()",
        "",
        "f_projection_line(_show, _x1, _y1, _x2, _y2, _clr, _width) =>",
        "    if _show and not na(_y1) and not na(_y2) and _y1 > 0 and _y2 > 0",
        "        array.push(projectionLines, line.new(_x1, _y1, _x2, _y2, xloc=xloc.bar_time, extend=extend.none, color=_clr, width=_width, style=line.style_dashed))",
        "",
        "f_projection_label(_show, _x, _y, _txt, _clr) =>",
        "    if _show and not na(_y) and _y > 0",
        "        array.push(projectionLabels, label.new(_x, _y, _txt, xloc=xloc.bar_time, style=label.style_label_left, textcolor=projectionLabelColor, color=color.new(_clr, 20), size=size.tiny))",
        "",
        "if barstate.islast",
        "    if array.size(projectionLines) > 0",
        "        for i = 0 to array.size(projectionLines) - 1",
        "            line.delete(array.get(projectionLines, i))",
        "    array.clear(projectionLines)",
        "    if array.size(projectionLabels) > 0",
        "        for i = 0 to array.size(projectionLabels) - 1",
        "            label.delete(array.get(projectionLabels, i))",
        "    array.clear(projectionLabels)",
        "",
    ])

    prev = "2026"
    for year in years:
        start_c = "curCentral" if prev == "2026" else f"projC{prev}"
        start_low = "curP10" if prev == "2026" else f"projLow{prev}"
        start_high = "curP90" if prev == "2026" else f"projHigh{prev}"
        start_miner = "curMiner" if prev == "2026" else f"projMiner{prev}"
        lines.extend([
            f"    f_projection_line(showProjectionCentral, t{prev}, {start_c}, t{year}, projC{year}, projectionCentralColor, 3)",
            f"    f_projection_line(showProjectionLow, t{prev}, {start_low}, t{year}, projLow{year}, projectionLowColor, 1)",
            f"    f_projection_line(showProjectionHigh, t{prev}, {start_high}, t{year}, projHigh{year}, projectionHighColor, 1)",
            f"    f_projection_line(showProjectionMiner, t{prev}, {start_miner}, t{year}, projMiner{year}, projectionMinerColor, 2)",
        ])
        prev = str(year)

    lines.extend([
        "",
        "    f_projection_label(showProjectionLabels and showProjectionCentral, t2036, projC2036, \"10Y central projection\\nauto seed · not target\", projectionCentralColor)",
        "    f_projection_label(showProjectionLabels and showProjectionLow, t2036, projLow2036, \"10Y structural low\", projectionLowColor)",
        "    f_projection_label(showProjectionLabels and showProjectionMiner, t2036, projMiner2036, \"10Y projected miner cost\", projectionMinerColor)",
        "",
        "// Small status marker: if seed values are missing, this remains empty/na.",
        "plot(showProjectionCentral ? curCentral : na, title=\"Current central from seed\", display=display.none)",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"Missing {STATE_PATH}")

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    date = _date_from_state(state)
    projection = state.get("tradingview_projection_10y") or {}
    years = projection.get("years") or {}

    anchors = state.get("tradingview_anchors") or {}
    active_year = str(anchors.get("active_year_override") or state.get("timestamp", "")[:4])
    active = (anchors.get("yearly") or {}).get(active_year) or {}
    bands = state.get("model_price_bands") or {}
    production = state.get("production_costs_usd") or {}

    current_central = active.get("central") or active.get("p50") or state.get("model_price_usd")
    current_high = active.get("p90") or bands.get("p90")
    current_low = active.get("p10") or bands.get("p10")
    current_miner = active.get("standard_miner") or active.get("miner") or production.get("average")

    _write_seed_csv(ROOT / f"{CURRENT_SYMBOL}.csv", date, current_central, current_high, current_low, current_miner, state.get("price_usd"))

    manifest = {
        "seed_source": SEED_SOURCE,
        "generated_at": state.get("generated_at"),
        "state_timestamp": state.get("timestamp"),
        "engine_status": projection.get("engine_status"),
        "method": projection.get("method"),
        "symbols": {"current": CURRENT_SYMBOL, "projection_prefix": PROJ_SYMBOL_PREFIX},
        "projection_years": {},
        "note": "CSV OHLC mapping: open=central, high=high band, low=low band, close=projected miner cost, volume=spot/year marker.",
    }

    for year in range(PROJECTION_START_YEAR, PROJECTION_END_YEAR + 1):
        data = years.get(str(year)) or {}
        symbol = f"{PROJ_SYMBOL_PREFIX}{year}"
        _write_seed_csv(
            ROOT / f"{symbol}.csv",
            date,
            data.get("central"),
            data.get("high"),
            data.get("low"),
            data.get("miner"),
            year,
        )
        manifest["projection_years"][str(year)] = {
            "symbol": symbol,
            "central": data.get("central"),
            "low": data.get("low"),
            "high": data.get("high"),
            "miner": data.get("miner"),
        }

    SEED_DIR.mkdir(parents=True, exist_ok=True)
    (SEED_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (SEED_DIR / "README.md").write_text(
        "# CohesivX TradingView seed export\n\n"
        "GitHub Actions updates root CSV seed symbols for TradingView `request.seed()`.\n\n"
        "Mapping per CSV row:\n"
        "- `open` = central structural projection\n"
        "- `high` = high/statistical projection\n"
        "- `low` = low/defense projection\n"
        "- `close` = projected standard miner cost\n\n"
        "Use `Pine/indicator_seed_request.txt` as the experimental auto-data Pine script.\n"
        "The normal `Pine/indicator.txt` remains the stable hardcoded fallback.\n",
        encoding="utf-8",
    )

    PINE_SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    PINE_SEED_PATH.write_text(_pine_seed_template(), encoding="utf-8")

    print("TradingView seed CSV files updated.")
    print("Seed Pine template updated:", PINE_SEED_PATH)


if __name__ == "__main__":
    main()
