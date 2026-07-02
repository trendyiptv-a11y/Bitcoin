#!/usr/bin/env python
from __future__ import annotations

"""
CohesivX Paper Profit Overlay Experiment v0.1

Paper-only shadow layer. It does not modify the official paper trader state and
never sends real orders. It creates a separate shadow portfolio to test whether
faster partial profit harvesting would improve or hurt the main v0.6.3 paper
portfolio from this point forward.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import paper_trader

ROOT = Path(__file__).resolve().parent
BASE_STATE_PATH = ROOT / "btc-swing-strategy" / "paper_trading_state.json"
BASE_DECISION_PATH = ROOT / "btc-swing-strategy" / "paper_trader_decision.json"
WATCH_SIGNAL_PATH = ROOT / "btc-swing-strategy" / "profit_watch_signal.json"
STATE_PATH = ROOT / "btc-swing-strategy" / "paper_profit_overlay_experiment_state.json"
REPORT_PATH = ROOT / "btc-swing-strategy" / "paper_profit_overlay_experiment_report.json"
LOG_PATH = ROOT / "btc-swing-strategy" / "paper_profit_overlay_experiment_log.jsonl"

EXPERIMENT_VERSION = "paper_profit_overlay_experiment_v0.1"
STARTING_BALANCE_USDT = 1000.0

# Experimental rule: harvest earlier than the official adaptive watcher.
# Official bear_late small threshold is currently 4.5%; this shadow layer tests
# whether a 2.5% partial harvest would be better or worse over time.
EARLY_TAKE_PROFIT_PCT = 0.025
EARLY_TAKE_PROFIT_FRACTION = 0.25
REBUY_PULLBACK_PCT = 0.015
REBUY_FRACTION_OF_CASH = 0.25
MAX_SHADOW_EXPOSURE = 0.40
MIN_USEFUL_TRADE_USDT = 5.0
COOLDOWN_MINUTES = 120


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_log(row: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def minutes_since(value: str | None) -> float | None:
    dt = parse_dt(value)
    if not dt:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 60.0)


def account(cash: float, btc: float, cost_basis: float, realized: float, price: float) -> dict[str, float]:
    market_value = btc * price
    portfolio_value = cash + market_value
    unrealized = market_value - cost_basis if btc > 0 else 0.0
    unrealized_pct = unrealized / cost_basis if cost_basis > 0 else 0.0
    total_pnl = realized + unrealized
    exposure = market_value / portfolio_value if portfolio_value > 0 else 0.0
    avg_entry = cost_basis / btc if btc > 0 and cost_basis > 0 else 0.0
    return {
        "cash_usdt": cash,
        "btc_amount": btc,
        "cost_basis_usdt": cost_basis,
        "avg_entry_price_usd": avg_entry,
        "realized_pnl_usdt": realized,
        "market_value_usdt": market_value,
        "portfolio_value_usdt": portfolio_value,
        "unrealized_pnl_usdt": unrealized,
        "unrealized_pnl_pct": unrealized_pct,
        "total_pnl_usdt": total_pnl,
        "total_pnl_pct": total_pnl / STARTING_BALANCE_USDT,
        "btc_exposure_pct": exposure * 100.0,
        "btc_exposure_fraction": exposure,
    }


def initial_shadow_state(base: dict[str, Any], price: float) -> dict[str, Any]:
    cash = f(base.get("cash_usdt"), STARTING_BALANCE_USDT)
    btc = f(base.get("btc_amount"))
    cost_basis = f(base.get("cost_basis_usdt"))
    realized = f(base.get("realized_pnl_usdt"))
    acct = account(cash, btc, cost_basis, realized, price)
    return {
        "version": EXPERIMENT_VERSION,
        "mode": "paper_shadow_only",
        "created_at": now_iso(),
        "created_from": "paper_trading_state.json",
        "cash_usdt": round(cash, 8),
        "btc_amount": round(btc, 12),
        "cost_basis_usdt": round(cost_basis, 8),
        "realized_pnl_usdt": round(realized, 8),
        "last_action": "INIT_FROM_BASE",
        "last_action_at": None,
        "last_sell_price_usd": None,
        "last_buyback_price_usd": None,
        "trade_count": 0,
        "sell_count": 0,
        "rebuy_count": 0,
        "baseline_base_portfolio_value_usdt": f(base.get("portfolio_value_usdt"), acct["portfolio_value_usdt"]),
        "baseline_shadow_portfolio_value_usdt": acct["portfolio_value_usdt"],
        "rules": rules_doc(),
    }


def rules_doc() -> dict[str, Any]:
    return {
        "version": EXPERIMENT_VERSION,
        "paper_only": True,
        "modifies_official_paper_trader": False,
        "early_take_profit_pct": EARLY_TAKE_PROFIT_PCT,
        "early_take_profit_fraction": EARLY_TAKE_PROFIT_FRACTION,
        "rebuy_pullback_pct": REBUY_PULLBACK_PCT,
        "rebuy_fraction_of_cash": REBUY_FRACTION_OF_CASH,
        "max_shadow_exposure": MAX_SHADOW_EXPOSURE,
        "min_useful_trade_usdt": MIN_USEFUL_TRADE_USDT,
        "cooldown_minutes": COOLDOWN_MINUTES,
    }


def apply_sell(state: dict[str, Any], price: float, fraction: float) -> tuple[dict[str, Any], dict[str, Any]]:
    cash = f(state.get("cash_usdt"))
    btc = f(state.get("btc_amount"))
    cost_basis = f(state.get("cost_basis_usdt"))
    realized = f(state.get("realized_pnl_usdt"))
    sell_btc = min(btc, btc * fraction)
    executed_usdt = sell_btc * price
    if executed_usdt < MIN_USEFUL_TRADE_USDT or btc <= 0:
        return state, {"executed": False, "reason": "sell below useful minimum"}
    basis_removed = cost_basis * (sell_btc / btc) if cost_basis > 0 else 0.0
    realized_delta = executed_usdt - basis_removed
    cash += executed_usdt
    btc -= sell_btc
    cost_basis = max(0.0, cost_basis - basis_removed)
    state.update({
        "cash_usdt": round(cash, 8),
        "btc_amount": round(btc, 12),
        "cost_basis_usdt": round(cost_basis, 8),
        "realized_pnl_usdt": round(realized + realized_delta, 8),
        "last_action": "EXP_TAKE_PROFIT_25",
        "last_action_at": now_iso(),
        "last_sell_price_usd": price,
        "trade_count": int(state.get("trade_count") or 0) + 1,
        "sell_count": int(state.get("sell_count") or 0) + 1,
    })
    return state, {
        "executed": True,
        "side": "sell",
        "executed_usdt": executed_usdt,
        "executed_btc": -sell_btc,
        "realized_delta_usdt": realized_delta,
        "basis_removed_usdt": basis_removed,
    }


def apply_rebuy(state: dict[str, Any], price: float, fraction_of_cash: float) -> tuple[dict[str, Any], dict[str, Any]]:
    cash = f(state.get("cash_usdt"))
    btc = f(state.get("btc_amount"))
    cost_basis = f(state.get("cost_basis_usdt"))
    realized = f(state.get("realized_pnl_usdt"))
    acct = account(cash, btc, cost_basis, realized, price)
    max_buy_by_exposure = max(0.0, STARTING_BALANCE_USDT * MAX_SHADOW_EXPOSURE - acct["market_value_usdt"])
    spend = min(cash * fraction_of_cash, max_buy_by_exposure)
    if spend < MIN_USEFUL_TRADE_USDT:
        return state, {"executed": False, "reason": "rebuy below useful minimum or exposure cap"}
    buy_btc = spend / price
    cash -= spend
    btc += buy_btc
    cost_basis += spend
    state.update({
        "cash_usdt": round(cash, 8),
        "btc_amount": round(btc, 12),
        "cost_basis_usdt": round(cost_basis, 8),
        "realized_pnl_usdt": round(realized, 8),
        "last_action": "EXP_REBUY_PULLBACK_25",
        "last_action_at": now_iso(),
        "last_buyback_price_usd": price,
        "trade_count": int(state.get("trade_count") or 0) + 1,
        "rebuy_count": int(state.get("rebuy_count") or 0) + 1,
    })
    return state, {
        "executed": True,
        "side": "buy",
        "executed_usdt": spend,
        "executed_btc": buy_btc,
        "realized_delta_usdt": 0.0,
    }


def main() -> None:
    base = load_json(BASE_STATE_PATH, paper_trader.initial_paper_state()) or paper_trader.initial_paper_state()
    decision = load_json(BASE_DECISION_PATH, {})
    signal = load_json(WATCH_SIGNAL_PATH, {})
    live_price, live_source, live_status = paper_trader.fetch_live_btc_price()
    price = f(live_price) or f(base.get("last_execution_price_usd"))
    if price <= 0:
        report = {
            "checked_at": now_iso(),
            "version": EXPERIMENT_VERSION,
            "status": "NO_PRICE",
            "reason": "No valid live or fallback price.",
            "not_trading_advice": True,
        }
        save_json(REPORT_PATH, report)
        print("Experimental overlay: no price")
        return

    shadow = load_json(STATE_PATH, {})
    if not shadow or shadow.get("version") != EXPERIMENT_VERSION:
        shadow = initial_shadow_state(base, price)

    cash = f(shadow.get("cash_usdt"))
    btc = f(shadow.get("btc_amount"))
    cost_basis = f(shadow.get("cost_basis_usdt"))
    realized = f(shadow.get("realized_pnl_usdt"))
    before = account(cash, btc, cost_basis, realized, price)

    base_cash = f(base.get("cash_usdt"), STARTING_BALANCE_USDT)
    base_btc = f(base.get("btc_amount"))
    base_cost = f(base.get("cost_basis_usdt"))
    base_realized = f(base.get("realized_pnl_usdt"))
    base_acct = account(base_cash, base_btc, base_cost, base_realized, price)

    action = "EXP_HOLD"
    reason = "No experimental trigger."
    execution: dict[str, Any] = {"executed": False}

    cooldown = minutes_since(shadow.get("last_action_at"))
    cooldown_ok = cooldown is None or cooldown >= COOLDOWN_MINUTES
    last_sell_price = f(shadow.get("last_sell_price_usd"))
    expected_30d = f((decision.get("memory_weighted_decision") or {}).get("expected_30d_return"))

    if btc <= 0:
        reason = "No shadow BTC position."
    elif not cooldown_ok:
        reason = f"Cooldown active: last experimental action was {cooldown:.1f} minutes ago."
    elif before["unrealized_pnl_pct"] >= EARLY_TAKE_PROFIT_PCT:
        action = "EXP_TAKE_PROFIT_25"
        reason = f"Experimental early harvest: {before['unrealized_pnl_pct'] * 100:.2f}% >= {EARLY_TAKE_PROFIT_PCT * 100:.2f}%."
        shadow, execution = apply_sell(shadow, price, EARLY_TAKE_PROFIT_FRACTION)
    elif last_sell_price > 0 and price <= last_sell_price * (1.0 - REBUY_PULLBACK_PCT) and expected_30d > 0:
        action = "EXP_REBUY_PULLBACK_25"
        reason = f"Experimental rebuy: price pulled back {(1 - price / last_sell_price) * 100:.2f}% from last shadow sell."
        shadow, execution = apply_rebuy(shadow, price, REBUY_FRACTION_OF_CASH)

    after = account(
        f(shadow.get("cash_usdt")),
        f(shadow.get("btc_amount")),
        f(shadow.get("cost_basis_usdt")),
        f(shadow.get("realized_pnl_usdt")),
        price,
    )

    base_vs_shadow = after["portfolio_value_usdt"] - base_acct["portfolio_value_usdt"]
    base_vs_shadow_pct = base_vs_shadow / base_acct["portfolio_value_usdt"] if base_acct["portfolio_value_usdt"] else 0.0

    report = {
        "checked_at": now_iso(),
        "version": EXPERIMENT_VERSION,
        "status": "PASS",
        "mode": "paper_shadow_only",
        "action": action,
        "reason": reason,
        "execution": execution,
        "live_price_usd": price,
        "live_price_source": live_source,
        "live_price_status": live_status,
        "base_portfolio": base_acct,
        "shadow_portfolio": after,
        "shadow_before_action": before,
        "overlay_edge_usdt": base_vs_shadow,
        "overlay_edge_pct": base_vs_shadow_pct,
        "official_profit_watch_signal": signal,
        "official_last_action": base.get("last_action"),
        "official_last_confidence": base.get("last_confidence"),
        "rules": rules_doc(),
        "not_trading_advice": True,
    }

    shadow.update({
        "last_checked_at": report["checked_at"],
        "last_live_price_usd": price,
        "last_live_price_source": live_source,
        "last_report": {
            "action": action,
            "reason": reason,
            "shadow_portfolio_value_usdt": after["portfolio_value_usdt"],
            "base_portfolio_value_usdt": base_acct["portfolio_value_usdt"],
            "overlay_edge_usdt": base_vs_shadow,
        },
        "rules": rules_doc(),
    })

    save_json(STATE_PATH, shadow)
    save_json(REPORT_PATH, report)
    append_log({
        "checked_at": report["checked_at"],
        "action": action,
        "reason": reason,
        "live_price_usd": price,
        "base_value_usdt": base_acct["portfolio_value_usdt"],
        "shadow_value_usdt": after["portfolio_value_usdt"],
        "overlay_edge_usdt": base_vs_shadow,
        "shadow_unrealized_pnl_pct": after["unrealized_pnl_pct"],
        "shadow_realized_pnl_usdt": after["realized_pnl_usdt"],
        "executed": execution,
    })

    print("Experimental overlay action:", action)
    print("Reason:", reason)
    print("Shadow value:", round(after["portfolio_value_usdt"], 6))
    print("Base value:", round(base_acct["portfolio_value_usdt"], 6))
    print("Overlay edge USDT:", round(base_vs_shadow, 6))


if __name__ == "__main__":
    main()
