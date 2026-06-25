#!/usr/bin/env python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import paper_trader
from paper_trader_adaptive import active_profit_rules

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "btc-swing-strategy" / "coeziv_state.json"
PAPER_STATE_PATH = ROOT / "btc-swing-strategy" / "paper_trading_state.json"
DECISION_PATH = ROOT / "btc-swing-strategy" / "paper_trader_decision.json"
SIGNAL_PATH = ROOT / "btc-swing-strategy" / "profit_watch_signal.json"


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def state_date(state: dict[str, Any]) -> str:
    context = state.get("model_price_context") or {}
    return str(context.get("date") or str(state.get("timestamp") or "")[:10] or "")[:10]


def current_regime(state: dict[str, Any]) -> str:
    return str((state.get("model_price_context") or {}).get("regime") or "")


def main() -> None:
    state = load_json(STATE_PATH, {})
    paper = load_json(PAPER_STATE_PATH, paper_trader.initial_paper_state()) or paper_trader.initial_paper_state()
    signal = load_json(SIGNAL_PATH, {})

    live_price, live_source, live_status = paper_trader.fetch_live_btc_price()
    if not live_price or live_price <= 0:
        print("No live price. Profit exit executor does nothing.")
        return

    regime = current_regime(state)
    rules = active_profit_rules(regime)
    normalized = paper_trader.normalize_paper_state(paper, live_price)
    acct = paper_trader.accounting_snapshot(normalized, live_price)
    unrealized_pct = acct["unrealized_pnl_pct"]
    btc = acct["btc_amount"]

    action = "OBSERVE_PROFIT_ONLY"
    confidence = "profit_watch_only"
    fraction = 0.0
    trigger_level = "none"

    if btc <= 0:
        reason = ["Profit exit executor: no BTC position."]
    elif unrealized_pct >= float(rules["take_profit_medium_pnl"]):
        action = "TAKE_PROFIT_MEDIUM"
        confidence = "profit_protection_stale_safe"
        fraction = float(rules["take_profit_medium_fraction"])
        trigger_level = "medium"
        reason = [
            "Profit-only executor activated from watcher.",
            f"Medium adaptive threshold reached: {unrealized_pct * 100:.2f}% >= {float(rules['take_profit_medium_pnl']) * 100:.2f}%.",
            "This path is sell-only and does not use stale state for new buys.",
        ]
    elif unrealized_pct >= float(rules["take_profit_small_pnl"]):
        action = "TAKE_PROFIT_SMALL"
        confidence = "profit_protection_stale_safe"
        fraction = float(rules["take_profit_small_fraction"])
        trigger_level = "small"
        reason = [
            "Profit-only executor activated from watcher.",
            f"Small adaptive threshold reached: {unrealized_pct * 100:.2f}% >= {float(rules['take_profit_small_pnl']) * 100:.2f}%.",
            "This path is sell-only and does not use stale state for new buys.",
        ]
    else:
        reason = [
            "Profit exit executor checked live PnL, but no adaptive threshold is active anymore.",
            f"Current unrealized PnL: {unrealized_pct * 100:.2f}%.",
        ]

    execution = paper_trader.apply_action(normalized, live_price, action, fraction)
    run_at = now_iso()
    after = execution["portfolio_value_after"]
    exposure = execution["btc_amount"] * live_price / after if after > 0 and live_price > 0 else 0.0

    updated = {
        **paper_trader.normalize_paper_state(normalized, live_price),
        "mode": "paper",
        "cash_usdt": round(execution["cash_usdt"], 8),
        "btc_amount": round(execution["btc_amount"], 12),
        "cost_basis_usdt": round(execution["cost_basis_usdt"], 8),
        "avg_entry_price_usd": round(execution["avg_entry_price_usd"], 8),
        "realized_pnl_usdt": round(execution["realized_pnl_usdt"], 8),
        "unrealized_pnl_usdt": round(execution["unrealized_pnl_usdt"], 8),
        "unrealized_pnl_pct": round(execution["unrealized_pnl_pct"] * 100, 4),
        "total_pnl_usdt": round(execution["total_pnl_usdt"], 8),
        "total_pnl_pct": round(execution["total_pnl_pct"] * 100, 4),
        "portfolio_value_usdt": round(after, 8),
        "btc_exposure_pct": round(exposure * 100, 4),
        "last_action": action,
        "last_confidence": confidence,
        "last_reason": reason,
        "last_run_at": run_at,
        "last_execution_price_usd": live_price,
        "last_execution_price_source": live_source,
        "last_snapshot_price_usd": paper_trader.f(state.get("price_usd")),
        "last_model_price_usd": paper_trader.f(state.get("model_price_usd")),
        "last_live_price_status": live_status,
        "last_profit_watch_signal": signal,
        "not_trading_advice": True,
        "rules": {
            "logic": "CohesivX profit-only stale-safe executor. Sell-only. No new buys on stale state.",
            "profit_profile": rules.get("profile"),
            "take_profit_small_pnl": rules.get("take_profit_small_pnl"),
            "take_profit_medium_pnl": rules.get("take_profit_medium_pnl"),
            "take_profit_small_fraction": rules.get("take_profit_small_fraction"),
            "take_profit_medium_fraction": rules.get("take_profit_medium_fraction"),
            "sell_only": True,
        },
    }
    paper_trader.save_json(PAPER_STATE_PATH, updated)

    decision_doc = {
        "run_at": run_at,
        "action": action,
        "confidence": confidence,
        "position_fraction": fraction,
        "trigger_level": trigger_level,
        "execution": execution,
        "live_price_status": live_status,
        "profit_profile": rules,
        "state_date": state_date(state),
        "run_date_utc": today_utc(),
        "state_was_fresh": state_date(state) == today_utc(),
        "reason": reason,
        "not_trading_advice": True,
    }
    paper_trader.save_json(DECISION_PATH, decision_doc)

    row = {
        "run_at": run_at,
        "state_date": state_date(state),
        "execution_price_source": live_source,
        "execution_price_usd": round(live_price, 8),
        "snapshot_price_usd": paper_trader.f(state.get("price_usd")),
        "model_price_usd": paper_trader.f(state.get("model_price_usd")),
        "deviation_pct": "",
        "expected_7d_return_pct": "",
        "expected_30d_return_pct": "",
        "historical_drawdown_risk_pct": "",
        "memory_confidence_pct": "",
        "decision_edge_pct": "",
        "structural_regime": regime,
        "market_regime_code": "profit_only_executor",
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
        "reason": " | ".join(reason),
    }
    paper_trader.append_log(row)

    print(f"Profit exit executor action: {action}")
    print(f"Trigger level: {trigger_level}")
    print(f"Live source: {live_source}")
    print(f"Unrealized PnL before action: {unrealized_pct * 100:.2f}%")
    print(f"Executed USDT: {row['executed_usdt']}")


if __name__ == "__main__":
    main()
