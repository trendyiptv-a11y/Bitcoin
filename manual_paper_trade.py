#!/usr/bin/env python
from __future__ import annotations

import os
from typing import Any

import paper_trader

MIN_MANUAL_TRADE_USDT = 5.0


def env_float(name: str, default: float) -> float:
    return paper_trader.f(os.getenv(name), default)


def manual_inputs() -> dict[str, Any]:
    return {
        "manual_action": (os.getenv("MANUAL_ACTION") or "BUY_USDT").upper().strip(),
        "amount_usdt": env_float("MANUAL_AMOUNT_USDT", 25.0),
        "sell_fraction": env_float("MANUAL_SELL_FRACTION", 0.10),
        "note": os.getenv("MANUAL_NOTE") or "Manual paper action from GitHub workflow.",
    }


def account_after(cash: float, btc: float, cost_basis: float, realized: float, price: float) -> dict[str, float]:
    market_value = btc * price if price > 0 else 0.0
    portfolio_value = cash + market_value
    avg_entry = cost_basis / btc if btc > 0 and cost_basis > 0 else 0.0
    unrealized = market_value - cost_basis if btc > 0 else 0.0
    unrealized_pct = unrealized / cost_basis if cost_basis > 0 else 0.0
    total_pnl = realized + unrealized
    total_pnl_pct = total_pnl / paper_trader.STARTING_BALANCE_USDT
    exposure = market_value / portfolio_value if portfolio_value > 0 else 0.0
    return {
        "cash_usdt": cash,
        "btc_amount": btc,
        "cost_basis_usdt": cost_basis,
        "avg_entry_price_usd": avg_entry,
        "realized_pnl_usdt": realized,
        "unrealized_pnl_usdt": unrealized,
        "unrealized_pnl_pct": unrealized_pct,
        "total_pnl_usdt": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "portfolio_value_usdt": portfolio_value,
        "btc_exposure_pct": exposure * 100,
    }


def execute_manual_trade(paper: dict[str, Any], price: float, inputs: dict[str, Any]) -> tuple[str, str, list[str], dict[str, float]]:
    paper = paper_trader.normalize_paper_state(paper, price)
    cash = paper_trader.f(paper.get("cash_usdt"), paper_trader.STARTING_BALANCE_USDT)
    btc = paper_trader.f(paper.get("btc_amount"))
    cost_basis = paper_trader.f(paper.get("cost_basis_usdt"))
    realized = paper_trader.f(paper.get("realized_pnl_usdt"))
    before = cash + btc * price if price > 0 else cash
    action = str(inputs["manual_action"])
    requested_usdt = max(0.0, float(inputs["amount_usdt"]))
    sell_fraction = paper_trader.clamp(float(inputs["sell_fraction"]), 0.0, 1.0)
    executed_usdt = 0.0
    executed_btc = 0.0
    realized_delta = 0.0
    reasons = [str(inputs["note"]), "Manual paper mode: no real funds, no exchange order."]

    if price <= 0:
        reasons.append("Manual trade blocked: live BTC price is unavailable.")
        after = account_after(cash, btc, cost_basis, realized, price)
        after.update({"executed_usdt": 0.0, "executed_btc": 0.0, "realized_pnl_delta_usdt": 0.0})
        return "MANUAL_BLOCKED_LIVE_PRICE", "manual_blocked", reasons, after

    if action == "BUY_USDT":
        market_value = btc * price
        max_allowed_by_exposure = max(0.0, paper_trader.MAX_BTC_EXPOSURE_FRACTION * before - market_value)
        executed_usdt = min(cash, requested_usdt, max_allowed_by_exposure)
        if executed_usdt >= MIN_MANUAL_TRADE_USDT:
            executed_btc = executed_usdt / price
            cash -= executed_usdt
            btc += executed_btc
            cost_basis += executed_usdt
            reasons.append(f"Manual buy executed: {executed_usdt:.2f} USDT at {price:.2f} USD.")
            reasons.append(f"Hard exposure cap respected: {paper_trader.MAX_BTC_EXPOSURE_FRACTION * 100:.1f}%.")
            final_action = "MANUAL_BUY"
            confidence = "manual_buy"
        else:
            executed_usdt = 0.0
            reasons.append("Manual buy blocked: requested amount/cash/exposure corridor leaves less than minimum useful trade size.")
            final_action = "MANUAL_BUY_BLOCKED"
            confidence = "manual_blocked"
    elif action == "BUY_USDT_FORCE":
        executed_usdt = min(cash, requested_usdt)
        if executed_usdt >= MIN_MANUAL_TRADE_USDT:
            executed_btc = executed_usdt / price
            cash -= executed_usdt
            btc += executed_btc
            cost_basis += executed_usdt
            reasons.append(f"Manual FORCE buy executed: {executed_usdt:.2f} USDT at {price:.2f} USD.")
            reasons.append("Operator override: bot exposure corridor was intentionally ignored for this paper trade.")
            final_action = "MANUAL_BUY_FORCE"
            confidence = "manual_override"
        else:
            executed_usdt = 0.0
            reasons.append("Manual FORCE buy blocked: available cash/requested amount is below minimum useful trade size.")
            final_action = "MANUAL_BUY_FORCE_BLOCKED"
            confidence = "manual_blocked"
    elif action == "SELL_PERCENT":
        sell_btc = min(btc, btc * sell_fraction)
        executed_usdt = sell_btc * price
        if executed_usdt >= MIN_MANUAL_TRADE_USDT and sell_btc > 0:
            basis_removed = cost_basis * (sell_btc / btc) if btc > 0 and cost_basis > 0 else 0.0
            realized_delta = executed_usdt - basis_removed
            realized += realized_delta
            cost_basis = max(0.0, cost_basis - basis_removed)
            cash += executed_usdt
            btc -= sell_btc
            executed_btc = -sell_btc
            reasons.append(f"Manual sell executed: {sell_fraction * 100:.1f}% of BTC position at {price:.2f} USD.")
            final_action = "MANUAL_SELL"
            confidence = "manual_sell"
        else:
            executed_usdt = 0.0
            reasons.append("Manual sell blocked: BTC position or resulting USDT amount is below minimum useful trade size.")
            final_action = "MANUAL_SELL_BLOCKED"
            confidence = "manual_blocked"
    elif action == "SELL_ALL":
        sell_btc = btc
        executed_usdt = sell_btc * price
        if executed_usdt >= MIN_MANUAL_TRADE_USDT and sell_btc > 0:
            basis_removed = cost_basis
            realized_delta = executed_usdt - basis_removed
            realized += realized_delta
            cost_basis = 0.0
            cash += executed_usdt
            btc = 0.0
            executed_btc = -sell_btc
            reasons.append(f"Manual sell-all executed at {price:.2f} USD.")
            final_action = "MANUAL_SELL_ALL"
            confidence = "manual_sell_all"
        else:
            executed_usdt = 0.0
            reasons.append("Manual sell-all blocked: no meaningful BTC position to sell.")
            final_action = "MANUAL_SELL_BLOCKED"
            confidence = "manual_blocked"
    else:
        reasons.append(f"Manual action '{action}' is unknown. Allowed: BUY_USDT, BUY_USDT_FORCE, SELL_PERCENT, SELL_ALL.")
        final_action = "MANUAL_UNKNOWN_ACTION"
        confidence = "manual_blocked"

    after = account_after(cash, btc, cost_basis, realized, price)
    after.update({
        "executed_usdt": executed_usdt,
        "executed_btc": executed_btc,
        "realized_pnl_delta_usdt": realized_delta,
    })
    return final_action, confidence, reasons, after


def main() -> None:
    state = paper_trader.load_json(paper_trader.STATE_PATH, {})
    paper = paper_trader.load_json(paper_trader.PAPER_STATE_PATH, paper_trader.initial_paper_state()) or paper_trader.initial_paper_state()
    inputs = manual_inputs()
    live_price, live_source, live_status = paper_trader.fetch_live_btc_price()
    price = live_price or 0.0
    action, confidence, reasons, execution = execute_manual_trade(paper, price, inputs)
    run_at = paper_trader.now_iso()

    updated = {
        **paper_trader.normalize_paper_state(paper, price),
        "mode": "paper",
        "cash_usdt": round(execution["cash_usdt"], 8),
        "btc_amount": round(execution["btc_amount"], 12),
        "cost_basis_usdt": round(execution["cost_basis_usdt"], 8),
        "avg_entry_price_usd": round(execution["avg_entry_price_usd"], 8),
        "realized_pnl_usdt": round(execution["realized_pnl_usdt"], 8),
        "unrealized_pnl_usdt": round(execution["unrealized_pnl_usdt"], 8),
        "unrealized_pnl_pct": round(execution["unrealized_pnl_pct"] * 100, 4),
        "total_pnl_usdt": round(execution["total_pnl_usdt"] * 100, 4),
        "total_pnl_pct": round(execution["total_pnl_pct"] * 100, 4),
        "portfolio_value_usdt": round(execution["portfolio_value_usdt"], 8),
        "btc_exposure_pct": round(execution["btc_exposure_pct"], 4),
        "last_action": action,
        "last_confidence": confidence,
        "last_reason": reasons,
        "last_run_at": run_at,
        "last_execution_price_usd": price,
        "last_execution_price_source": live_source,
        "last_live_price_status": live_status,
        "last_manual_inputs": inputs,
        "not_trading_advice": True,
    }
    # Correct total PnL amount in USDT, not percentage-scaled.
    updated["total_pnl_usdt"] = round(execution["total_pnl_usdt"], 8)
    paper_trader.save_json(paper_trader.PAPER_STATE_PATH, updated)

    decision_doc = {
        "run_at": run_at,
        "action": action,
        "confidence": confidence,
        "manual_inputs": inputs,
        "execution": execution,
        "live_price_status": live_status,
        "reason": reasons,
        "not_trading_advice": True,
    }
    paper_trader.save_json(paper_trader.DECISION_PATH, decision_doc)

    row = {
        "run_at": run_at,
        "state_date": paper_trader.state_date(state),
        "execution_price_source": live_source,
        "execution_price_usd": round(price, 8),
        "snapshot_price_usd": state.get("price_usd", ""),
        "model_price_usd": state.get("model_price_usd", ""),
        "deviation_pct": "",
        "expected_7d_return_pct": "",
        "expected_30d_return_pct": "",
        "historical_drawdown_risk_pct": "",
        "memory_confidence_pct": "",
        "decision_edge_pct": "",
        "structural_regime": (state.get("model_price_context") or {}).get("regime", ""),
        "market_regime_code": (state.get("market_regime") or {}).get("code", ""),
        "action": action,
        "confidence": confidence,
        "executed_usdt": round(execution["executed_usdt"], 8),
        "executed_btc": round(execution["executed_btc"], 12),
        "cash_usdt": round(updated["cash_usdt"], 8),
        "btc_amount": round(updated["btc_amount"], 12),
        "cost_basis_usdt": round(updated["cost_basis_usdt"], 8),
        "avg_entry_price_usd": round(updated["avg_entry_price_usd"], 8),
        "unrealized_pnl_usdt": round(updated["unrealized_pnl_usdt"], 8),
        "unrealized_pnl_pct": round(execution["unrealized_pnl_pct"] * 100, 4),
        "realized_pnl_usdt": round(updated["realized_pnl_usdt"], 8),
        "total_pnl_usdt": round(updated["total_pnl_usdt"], 8),
        "total_pnl_pct": round(updated["total_pnl_pct"], 4),
        "btc_exposure_pct": round(updated["btc_exposure_pct"], 4),
        "portfolio_value_usdt": round(updated["portfolio_value_usdt"], 8),
        "reason": " | ".join(reasons),
    }
    paper_trader.append_log(row)

    print(f"Manual paper action: {action}")
    print(f"Confidence: {confidence}")
    print(f"Live source: {live_source}")
    print(f"Executed USDT: {row['executed_usdt']}")
    print(f"BTC amount: {row['btc_amount']}")
    print(f"Cash: {row['cash_usdt']} USDT")
    print(f"Portfolio value: {row['portfolio_value_usdt']} USDT")


if __name__ == "__main__":
    main()
