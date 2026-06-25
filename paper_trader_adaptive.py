#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import paper_trader

STATE_PATH = Path(__file__).resolve().parent / "btc-swing-strategy" / "coeziv_state.json"


def active_profit_rules(regime: str) -> dict[str, float | str]:
    """
    Regime-aware take-profit thresholds for the paper bot.
    This wrapper does not change the bot's decision architecture; it only sets
    the active accounting thresholds before paper_trader.main() runs.
    """
    r = (regime or "").lower()

    if r == "bear_late":
        return {
            "profile": "bear_late_defensive",
            "take_profit_small_pnl": 0.045,
            "take_profit_medium_pnl": 0.09,
            "take_profit_small_fraction": 0.20,
            "take_profit_medium_fraction": 0.35,
        }

    if r.startswith("bear"):
        return {
            "profile": "bear_defensive",
            "take_profit_small_pnl": 0.035,
            "take_profit_medium_pnl": 0.075,
            "take_profit_small_fraction": 0.20,
            "take_profit_medium_fraction": 0.35,
        }

    if r.startswith("bull"):
        return {
            "profile": "bull_patient",
            "take_profit_small_pnl": 0.08,
            "take_profit_medium_pnl": 0.16,
            "take_profit_small_fraction": 0.20,
            "take_profit_medium_fraction": 0.35,
        }

    return {
        "profile": "range_neutral_default",
        "take_profit_small_pnl": 0.06,
        "take_profit_medium_pnl": 0.12,
        "take_profit_small_fraction": 0.20,
        "take_profit_medium_fraction": 0.35,
    }


def current_regime() -> str:
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return str((state.get("model_price_context") or {}).get("regime") or "")
    except Exception:
        return ""


def apply_rules() -> dict[str, float | str]:
    regime = current_regime()
    rules = active_profit_rules(regime)

    paper_trader.TAKE_PROFIT_SMALL_PNL = float(rules["take_profit_small_pnl"])
    paper_trader.TAKE_PROFIT_MEDIUM_PNL = float(rules["take_profit_medium_pnl"])
    paper_trader.TAKE_PROFIT_SMALL_FRACTION = float(rules["take_profit_small_fraction"])
    paper_trader.TAKE_PROFIT_MEDIUM_FRACTION = float(rules["take_profit_medium_fraction"])

    print("Adaptive profit profile:", rules["profile"])
    print("Structural regime:", regime or "unknown")
    print("TP small:", paper_trader.TAKE_PROFIT_SMALL_PNL)
    print("TP medium:", paper_trader.TAKE_PROFIT_MEDIUM_PNL)
    return {"regime": regime, **rules}


if __name__ == "__main__":
    apply_rules()
    paper_trader.main()
