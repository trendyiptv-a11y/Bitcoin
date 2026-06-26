#!/usr/bin/env python
from __future__ import annotations

"""
CohesivX Paper Trader v0.6.3 Adapter

This module does not execute real trades and does not replace paper_trader.py.
It installs a v0.6.3-style decision overlay on top of the existing paper trader
execution/accounting layer.

Design:
- keep live price safety from paper_trader.py;
- keep paper accounting and CSV/JSON output from paper_trader.py;
- keep adaptive cash corridor/take-profit wrapper when available;
- translate the validated v0.6.3 guarded-accumulation philosophy into
  paper-trader-compatible actions.

The validated historical engine remains backtest_cohesivx_paper.py v0.6.3.
This adapter is a live-paper bridge, not a new backtest engine.
"""

from dataclasses import dataclass
from typing import Any

import paper_trader

ENGINE_VERSION = "cohesivx_backtest_v0.6.3_guarded_accumulation"
ADAPTER_VERSION = "paper_trader_v063_adapter_v0.1"

# v0.6.3 guard constants, adapted to the paper-live execution layer.
DRAWDOWN_CAUTION = -0.54
ADAPTIVE_BUY_LOCK_DD = -0.58
DRAWDOWN_GUARD = -0.60
DRAWDOWN_GUARD_TARGET_CAP = 0.72
MAX_ENTRY_FRACTION_GUARD = 0.08
TARGET_UP_STEP_DEEP = 0.14

# Paper-live safety caps. These are intentionally more conservative than the
# historical backtest final exposure because live/paper usage must manage
# start-date risk.
PAPER_BASE_MAX_EXPOSURE = 0.40
PAPER_GUARD_MAX_EXPOSURE = 0.30
PAPER_CAUTION_MAX_EXPOSURE = 0.35
PAPER_SELL_FRACTION = 0.25
PAPER_TRIM_HYSTERESIS = 0.04


@dataclass(frozen=True)
class V063Decision:
    raw_action: str
    paper_action: str
    confidence: str
    position_fraction: float
    target_exposure: float
    reason: str


def _f(value: Any, default: float = 0.0) -> float:
    return paper_trader.f(value, default)


def _clamp(x: float, lo: float, hi: float) -> float:
    return paper_trader.clamp(x, lo, hi)


def _portfolio_drawdown(paper: dict[str, Any], acct: dict[str, float]) -> tuple[float, float]:
    """Return current paper drawdown and updated peak portfolio value."""
    current_value = _f(acct.get("portfolio_value_usdt"), paper_trader.STARTING_BALANCE_USDT)
    saved_peak = _f(paper.get("v063_peak_portfolio_value_usdt"), paper_trader.STARTING_BALANCE_USDT)
    peak = max(saved_peak, current_value, paper_trader.STARTING_BALANCE_USDT)
    drawdown = (current_value - peak) / peak if peak > 0 else 0.0
    paper["v063_peak_portfolio_value_usdt"] = round(peak, 8)
    paper["v063_current_drawdown_pct"] = round(drawdown * 100, 4)
    return drawdown, peak


def _base_target_from_edge(edge: dict[str, Any]) -> tuple[float, str]:
    decision_edge = _f(edge.get("decision_edge"))
    confidence = _f(edge.get("memory_confidence"))
    drawdown_risk = _f(edge.get("historical_drawdown_risk"))
    expected_30d = _f(edge.get("expected_30d_return"))

    if decision_edge >= 0.18 and confidence >= 0.58 and drawdown_risk < 0.28:
        return PAPER_BASE_MAX_EXPOSURE, "strong_memory_accumulation"
    if decision_edge >= 0.12 and confidence >= 0.55 and drawdown_risk < 0.32:
        return 0.34, "moderate_memory_accumulation"
    if decision_edge >= 0.07 and confidence >= 0.45 and drawdown_risk < 0.38:
        return 0.25, "small_memory_accumulation"
    if expected_30d < -0.08 or decision_edge < -0.06:
        return 0.0, "negative_edge_reduce_risk"
    return -1.0, "neutral_observe"


def compute_v063_decision(
    paper: dict[str, Any],
    snapshot: dict[str, Any],
    edge: dict[str, Any],
) -> V063Decision:
    """Translate v0.6.3 guarded-accumulation behaviour into paper actions."""
    acct = snapshot.get("paper_portfolio") or {}
    exposure = _f(acct.get("btc_exposure"))
    btc = _f(acct.get("btc_amount"))
    cash = _f(acct.get("cash_usdt"))
    drawdown, peak = _portfolio_drawdown(paper, acct)

    base_target, base_reason = _base_target_from_edge(edge)
    target = base_target if base_target >= 0 else exposure

    guard_state = "normal"
    if drawdown <= DRAWDOWN_GUARD:
        guard_state = "drawdown_guard"
        target = min(target, PAPER_GUARD_MAX_EXPOSURE, DRAWDOWN_GUARD_TARGET_CAP)
    elif drawdown <= ADAPTIVE_BUY_LOCK_DD:
        guard_state = "adaptive_buy_lock"
        target = min(target, exposure)
    elif drawdown <= DRAWDOWN_CAUTION:
        guard_state = "drawdown_caution"
        target = min(target, PAPER_CAUTION_MAX_EXPOSURE)

    decision_edge = _f(edge.get("decision_edge"))
    expected_30d = _f(edge.get("expected_30d_return"))
    drawdown_risk = _f(edge.get("historical_drawdown_risk"))
    unrealized_pct = _f(acct.get("unrealized_pnl_pct"))

    if btc > 0 and (expected_30d < -0.08 or decision_edge < -0.06):
        return V063Decision(
            raw_action="REDUCE_RISK_V063_GUARD",
            paper_action="REDUCE_RISK",
            confidence="v063_defensive",
            position_fraction=PAPER_SELL_FRACTION,
            target_exposure=max(0.0, target),
            reason="v0.6.3 defensive guard: expected return/edge turned negative while BTC is held.",
        )

    if btc > 0 and unrealized_pct >= 0.12 and (decision_edge < 0.20 or drawdown_risk > 0.30):
        return V063Decision(
            raw_action="TAKE_PROFIT_OVERVALUED_V063",
            paper_action="TAKE_PROFIT_MEDIUM",
            confidence="v063_profit_protection",
            position_fraction=0.35,
            target_exposure=max(0.0, target),
            reason="v0.6.3 profit protection: profit is available and edge/risk is no longer ideal.",
        )

    if btc > 0 and exposure > target + PAPER_TRIM_HYSTERESIS and base_target >= 0:
        excess_fraction_of_position = (exposure - target) / exposure if exposure > 0 else PAPER_SELL_FRACTION
        return V063Decision(
            raw_action="TRIM_ABOVE_SMOOTHED_TARGET_V063",
            paper_action="REDUCE_RISK",
            confidence="v063_trim",
            position_fraction=_clamp(excess_fraction_of_position, 0.05, PAPER_SELL_FRACTION),
            target_exposure=max(0.0, target),
            reason="v0.6.3 target smoothing: current BTC exposure is above the guarded target.",
        )

    if guard_state == "adaptive_buy_lock":
        return V063Decision(
            raw_action="HOLD_ADAPTIVE_BUY_LOCK_V063",
            paper_action="HOLD" if btc > 0 else "OBSERVE",
            confidence="v063_buy_lock",
            position_fraction=0.0,
            target_exposure=exposure,
            reason="v0.6.3 adaptive buy lock: paper drawdown is too deep for new accumulation.",
        )

    if base_target < 0:
        return V063Decision(
            raw_action="OBSERVE_V063",
            paper_action="HOLD" if btc > 0 else "OBSERVE",
            confidence="v063_observe",
            position_fraction=0.0,
            target_exposure=exposure,
            reason="v0.6.3 neutral state: no target increase and no forced reduction.",
        )

    if cash <= 10 or exposure >= target:
        return V063Decision(
            raw_action="HOLD_TARGET_REACHED_V063",
            paper_action="HOLD" if btc > 0 else "OBSERVE",
            confidence="v063_target_reached",
            position_fraction=0.0,
            target_exposure=target,
            reason="v0.6.3 guarded target reached or cash too low for useful accumulation.",
        )

    remaining = max(0.0, target - exposure)
    if remaining <= 0:
        return V063Decision(
            raw_action="OBSERVE_V063",
            paper_action="HOLD" if btc > 0 else "OBSERVE",
            confidence="v063_observe",
            position_fraction=0.0,
            target_exposure=target,
            reason="v0.6.3 no remaining guarded exposure corridor.",
        )

    max_entry = MAX_ENTRY_FRACTION_GUARD if guard_state in {"drawdown_guard", "drawdown_caution"} else paper_trader.MAX_ENTRY_FRACTION
    fraction = min(remaining, max_entry)
    if base_reason == "strong_memory_accumulation":
        raw_action = "ALLOCATE_DEEP_UNDERPRICED_V063"
        paper_action = "ACCUMULATE_SMALL"
        confidence = "v063_memory_moderate"
    else:
        raw_action = "ALLOCATE_TO_SMOOTHED_TARGET_V063"
        paper_action = "OBSERVE_ACCUMULATE_SMALL"
        confidence = "v063_memory_moderate_low"

    return V063Decision(
        raw_action=raw_action,
        paper_action=paper_action,
        confidence=confidence,
        position_fraction=_clamp(fraction, 0.0, max_entry),
        target_exposure=target,
        reason=f"v0.6.3 guarded accumulation: {base_reason}; guard_state={guard_state}.",
    )


def install_v063_adapter() -> None:
    """Patch paper_trader in memory. Safe to call multiple times."""
    if getattr(paper_trader, "_v063_adapter_installed", False):
        return

    original_estimate_memory_edge = paper_trader.estimate_memory_edge
    original_decide = paper_trader.decide

    def estimate_with_v063(snapshot: dict[str, Any]) -> dict[str, Any]:
        edge = original_estimate_memory_edge(snapshot)
        if not edge.get("available"):
            edge["v063_adapter"] = {"available": False, "version": ADAPTER_VERSION, "engine_version": ENGINE_VERSION}
            return edge
        # The actual paper state is needed for drawdown/peak tracking, so the
        # final v0.6.3 translation happens in decide_with_v063.
        edge["v063_adapter"] = {
            "available": True,
            "version": ADAPTER_VERSION,
            "engine_version": ENGINE_VERSION,
            "mode": "pending_decide_translation",
        }
        return edge

    def decide_with_v063(state: dict[str, Any], paper: dict[str, Any], live_price: float | None, live_source: str, live_status: dict[str, Any]):
        action, confidence, reasons, fraction, snapshot, edge = original_decide(state, paper, live_price, live_source, live_status)
        if not edge.get("available") or not live_price:
            return action, confidence, reasons, fraction, snapshot, edge

        v063 = compute_v063_decision(paper, snapshot, edge)
        edge["v063_adapter"] = {
            "available": True,
            "version": ADAPTER_VERSION,
            "engine_version": ENGINE_VERSION,
            "raw_action": v063.raw_action,
            "paper_action": v063.paper_action,
            "target_exposure_fraction": v063.target_exposure,
            "position_fraction": v063.position_fraction,
            "reason": v063.reason,
            "paper_drawdown_pct": paper.get("v063_current_drawdown_pct"),
            "peak_portfolio_value_usdt": paper.get("v063_peak_portfolio_value_usdt"),
            "constants": {
                "drawdown_caution": DRAWDOWN_CAUTION,
                "adaptive_buy_lock_dd": ADAPTIVE_BUY_LOCK_DD,
                "drawdown_guard": DRAWDOWN_GUARD,
                "max_entry_fraction_guard": MAX_ENTRY_FRACTION_GUARD,
                "paper_base_max_exposure": PAPER_BASE_MAX_EXPOSURE,
            },
        }
        edge["method"] = str(edge.get("method") or "memory_weighted_distribution") + "+v063_guarded_adapter"
        edge["memory_action_before_v063_adapter"] = edge.get("memory_action")
        edge["position_fraction_before_v063_adapter"] = edge.get("position_fraction")
        edge["memory_action"] = v063.paper_action
        edge["position_fraction"] = v063.position_fraction
        edge["confidence_label"] = v063.confidence
        edge["final_action"] = v063.paper_action
        edge["final_position_fraction"] = v063.position_fraction

        reasons.append(v063.reason)
        reasons.append(
            "v0.6.3 adapter: "
            f"raw={v063.raw_action}, paper={v063.paper_action}, "
            f"target exposure={v063.target_exposure * 100:.1f}%, "
            f"fraction={v063.position_fraction * 100:.2f}%."
        )
        return v063.paper_action, v063.confidence, reasons, v063.position_fraction, snapshot, edge

    paper_trader.estimate_memory_edge = estimate_with_v063
    paper_trader.decide = decide_with_v063
    paper_trader._v063_adapter_installed = True


def apply_rules() -> dict[str, Any]:
    install_v063_adapter()
    return {"adapter_version": ADAPTER_VERSION, "engine_version": ENGINE_VERSION, "installed": True}


def main() -> None:
    install_v063_adapter()
    # Keep the existing adaptive wrapper if it is present in the repository.
    # It adds regime-aware take-profit and cash corridor after the v0.6.3 signal.
    try:
        import paper_trader_adaptive

        paper_trader_adaptive.apply_rules()
    except Exception as exc:
        print(f"Adaptive wrapper not applied: {exc.__class__.__name__}: {exc}")
    paper_trader.main()


if __name__ == "__main__":
    main()
