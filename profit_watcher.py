#!/usr/bin/env python
from __future__ import annotations

import json
import os
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

COOLDOWN_MINUTES = 20


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def state_date(state: dict[str, Any]) -> str:
    context = state.get("model_price_context") or {}
    return str(context.get("date") or str(state.get("timestamp") or "")[:10] or "")[:10]


def current_regime(state: dict[str, Any]) -> str:
    return str((state.get("model_price_context") or {}).get("regime") or "")


def last_bot_run_minutes_ago() -> float | None:
    decision = load_json(DECISION_PATH, {})
    dt = parse_dt(str(decision.get("run_at") or ""))
    if not dt:
        return None
    return max(0.0, (now_utc() - dt).total_seconds() / 60.0)


def write_github_output(signal: dict[str, Any]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    trigger = "true" if signal.get("trigger_paper_trader") else "false"
    reason = str(signal.get("reason") or "")[:200].replace("\n", " ")
    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(f"trigger_paper_trader={trigger}\n")
        fh.write(f"trigger_reason={reason}\n")


def main() -> None:
    state = load_json(STATE_PATH, {})
    paper = load_json(PAPER_STATE_PATH, paper_trader.initial_paper_state()) or paper_trader.initial_paper_state()
    live_price, live_source, live_status = paper_trader.fetch_live_btc_price()

    regime = current_regime(state)
    rules = active_profit_rules(regime)
    today = now_utc().date().isoformat()
    sdate = state_date(state)
    minutes_ago = last_bot_run_minutes_ago()

    signal: dict[str, Any] = {
        "checked_at": now_utc().isoformat(),
        "state_date": sdate,
        "run_date_utc": today,
        "state_is_fresh": sdate == today,
        "regime": regime,
        "profit_profile": rules.get("profile"),
        "take_profit_small_pnl": rules.get("take_profit_small_pnl"),
        "take_profit_medium_pnl": rules.get("take_profit_medium_pnl"),
        "live_price_source": live_source,
        "live_price_usd": live_price,
        "live_price_status": live_status,
        "last_bot_run_minutes_ago": minutes_ago,
        "trigger_paper_trader": False,
        "reason": "No profit trigger.",
    }

    if not live_price or live_price <= 0:
        signal["reason"] = "No live price; watcher will not trigger paper trader."
    else:
        normalized = paper_trader.normalize_paper_state(paper, live_price)
        acct = paper_trader.accounting_snapshot(normalized, live_price)
        unrealized_pct = acct["unrealized_pnl_pct"]
        btc = acct["btc_amount"]
        signal.update({
            "btc_amount": btc,
            "cash_usdt": acct["cash_usdt"],
            "cost_basis_usdt": acct["cost_basis_usdt"],
            "avg_entry_price_usd": acct["avg_entry_price_usd"],
            "unrealized_pnl_usdt": acct["unrealized_pnl_usdt"],
            "unrealized_pnl_pct": unrealized_pct,
            "portfolio_value_usdt": acct["portfolio_value_usdt"],
            "btc_exposure_pct": acct["btc_exposure_pct"],
        })

        if btc <= 0:
            signal["reason"] = "No BTC position; no profit watcher trigger."
        elif sdate != today:
            signal["reason"] = "State is stale; main bot would refuse execution, so watcher will not trigger."
        elif minutes_ago is not None and minutes_ago < COOLDOWN_MINUTES:
            signal["reason"] = f"Cooldown active: last paper trader run was {minutes_ago:.1f} minutes ago."
        elif unrealized_pct >= float(rules["take_profit_medium_pnl"]):
            signal["trigger_paper_trader"] = True
            signal["trigger_level"] = "medium"
            signal["reason"] = f"Medium adaptive take-profit threshold reached: {unrealized_pct * 100:.2f}% >= {float(rules['take_profit_medium_pnl']) * 100:.2f}%."
        elif unrealized_pct >= float(rules["take_profit_small_pnl"]):
            signal["trigger_paper_trader"] = True
            signal["trigger_level"] = "small"
            signal["reason"] = f"Small adaptive take-profit threshold reached: {unrealized_pct * 100:.2f}% >= {float(rules['take_profit_small_pnl']) * 100:.2f}%."

    SIGNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIGNAL_PATH.write_text(json.dumps(signal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_github_output(signal)

    print("Profit watcher trigger:", signal["trigger_paper_trader"])
    print("Reason:", signal["reason"])
    print("Regime:", signal.get("regime"), "profile:", signal.get("profit_profile"))


if __name__ == "__main__":
    main()
