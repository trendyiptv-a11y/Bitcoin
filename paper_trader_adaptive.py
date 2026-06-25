#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import paper_trader

STATE_PATH = Path(__file__).resolve().parent / "btc-swing-strategy" / "coeziv_state.json"
MIN_USEFUL_ENTRY_USDT = 5.0


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


def active_cash_corridor_rules(regime: str, edge: dict[str, Any]) -> dict[str, float | str]:
    """
    Adaptive exposure corridor.

    The hard bot limit stays 40%, but normal buying should preserve a cash
    reserve, especially in bear regimes. Only a very strong memory signal is
    allowed to use the full corridor toward 40%.
    """
    r = (regime or "").lower()
    decision_edge = paper_trader.f(edge.get("decision_edge"))
    confidence = paper_trader.f(edge.get("memory_confidence"))
    drawdown_risk = paper_trader.f(edge.get("historical_drawdown_risk"))

    if r == "bear_late":
        profile = "bear_late_cash_corridor"
        base_max_exposure = 0.34
        min_cash_reserve = 0.24
    elif r.startswith("bear"):
        profile = "bear_cash_corridor"
        base_max_exposure = 0.30
        min_cash_reserve = 0.30
    elif r.startswith("bull"):
        profile = "bull_flexible_corridor"
        base_max_exposure = 0.40
        min_cash_reserve = 0.12
    else:
        profile = "range_cash_corridor"
        base_max_exposure = 0.35
        min_cash_reserve = 0.20

    strong_signal = decision_edge >= 0.18 and confidence >= 0.58 and drawdown_risk < 0.28
    moderate_signal = decision_edge >= 0.12 and confidence >= 0.55 and drawdown_risk < 0.32

    if strong_signal:
        max_exposure = 0.40
        min_cash_reserve = min(min_cash_reserve, 0.10)
        profile += "_strong_override"
    elif moderate_signal:
        max_exposure = min(0.37, base_max_exposure + 0.03)
        min_cash_reserve = max(0.15, min_cash_reserve - 0.05)
        profile += "_moderate"
    else:
        max_exposure = base_max_exposure

    # The cash reserve is an additional safety rail. It prevents the bot from
    # using the whole 40% exposure corridor too early in a falling market.
    reserve_cap = max(0.0, 1.0 - min_cash_reserve)
    max_exposure = min(max_exposure, reserve_cap, 0.40)
    max_exposure = max(0.15, max_exposure)

    return {
        "profile": profile,
        "max_btc_exposure_fraction": max_exposure,
        "min_cash_reserve_fraction": min_cash_reserve,
        "strong_signal": 1.0 if strong_signal else 0.0,
        "moderate_signal": 1.0 if moderate_signal else 0.0,
    }


def install_cash_corridor() -> None:
    if getattr(paper_trader, "_cash_corridor_installed", False):
        return

    original_estimate_memory_edge = paper_trader.estimate_memory_edge
    original_decide = paper_trader.decide

    def estimate_with_corridor(s: dict[str, Any]) -> dict[str, Any]:
        edge = original_estimate_memory_edge(s)
        if not edge.get("available"):
            return edge

        acct = s.get("paper_portfolio") or {}
        regime = (s.get("regime") or {}).get("structural_code") or ""
        corridor = active_cash_corridor_rules(str(regime), edge)
        max_exposure = float(corridor["max_btc_exposure_fraction"])
        current_exposure = paper_trader.f(acct.get("btc_exposure"))
        portfolio_value = paper_trader.f(acct.get("portfolio_value_usdt"), paper_trader.STARTING_BALANCE_USDT)
        btc = paper_trader.f(acct.get("btc_amount"))
        remaining_fraction = max(0.0, max_exposure - current_exposure)
        remaining_usdt = max(0.0, portfolio_value * remaining_fraction)

        paper_trader.MAX_BTC_EXPOSURE_FRACTION = max_exposure
        edge["cash_corridor"] = {
            **corridor,
            "current_btc_exposure_fraction": current_exposure,
            "remaining_corridor_fraction": remaining_fraction,
            "remaining_corridor_usdt": remaining_usdt,
            "min_useful_entry_usdt": MIN_USEFUL_ENTRY_USDT,
        }

        if edge.get("memory_action") in {"ACCUMULATE_SMALL", "OBSERVE_ACCUMULATE_SMALL"}:
            if remaining_usdt < MIN_USEFUL_ENTRY_USDT:
                edge["memory_action"] = "HOLD_CASH_CORRIDOR" if btc > 0 else "OBSERVE_CASH_CORRIDOR"
                edge["position_fraction"] = 0.0
                edge["confidence_label"] = "cash_corridor"
            else:
                edge["position_fraction"] = min(float(edge.get("position_fraction") or 0.0), remaining_fraction)

        edge["method"] = str(edge.get("method") or "memory_weighted_distribution") + "+cash_corridor_v0.1"
        return edge

    def decide_with_corridor(*args: Any, **kwargs: Any):
        action, confidence, reasons, fraction, snapshot, edge = original_decide(*args, **kwargs)
        corridor = edge.get("cash_corridor") or {}
        if corridor:
            reasons.append(
                "Cash corridor: "
                f"profile {corridor.get('profile')}, "
                f"max exposure {float(corridor.get('max_btc_exposure_fraction', 0.0)) * 100:.1f}%, "
                f"min cash reserve {float(corridor.get('min_cash_reserve_fraction', 0.0)) * 100:.1f}%, "
                f"current exposure {float(corridor.get('current_btc_exposure_fraction', 0.0)) * 100:.1f}%."
            )
            if action in {"HOLD_CASH_CORRIDOR", "OBSERVE_CASH_CORRIDOR"}:
                reasons.append("No new buy: remaining cash corridor is below useful minimum entry size.")
        return action, confidence, reasons, fraction, snapshot, edge

    paper_trader.estimate_memory_edge = estimate_with_corridor
    paper_trader.decide = decide_with_corridor
    paper_trader._cash_corridor_installed = True


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
    install_cash_corridor()

    print("Adaptive profit profile:", rules["profile"])
    print("Structural regime:", regime or "unknown")
    print("TP small:", paper_trader.TAKE_PROFIT_SMALL_PNL)
    print("TP medium:", paper_trader.TAKE_PROFIT_MEDIUM_PNL)
    print("Cash corridor: enabled")
    return {"regime": regime, **rules}


if __name__ == "__main__":
    apply_rules()
    paper_trader.main()
