#!/usr/bin/env python3
# CohesivX Guarded Paper Exit v0.3.25
# PAPER-ONLY idempotent consumer for CohesivX Live Cohesion Guard report.
#
# Fix vs v0.3.23:
#   1) Uses ONLY btc-swing-strategy/cohesivx_scalp_state.json open_position
#      for closing. It no longer falls back to stale scalp report positions.
#   2) Refuses to close if the same entry_timestamp_utc / entry_price already
#      has a PAPER_SCALP_EXIT in cohesivx_scalp_trades.jsonl.
#   3) Refuses stale guard reports if guard position does not match current
#      state.open_position.
#
# Safety:
#   - No real orders.
#   - No API keys.
#   - No Micro-Live interaction.
#   - Mutates only paper scalper local files when one valid open state position exists.

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "btc-swing-strategy"

STATE_PATH = OUT_DIR / "cohesivx_scalp_state.json"
TRADES_PATH = OUT_DIR / "cohesivx_scalp_trades.jsonl"

GUARD_REPORT_PATH = OUT_DIR / "cohesivx_live_cohesion_guard_report.json"
EXIT_REPORT_PATH = OUT_DIR / "cohesivx_guarded_paper_exit_report.json"

BOT_NAME = "CohesivX Guarded Paper Exit v0.3.25-idempotent"
MODE = "PAPER_ONLY_GUARDED_EXIT_IDEMPOTENT"

FEE_RATE_DEFAULT = 0.001

EXIT_ACTIONS = {
    "GUARD_EXIT_HARDBREAK_SUB_PI",
    "GUARD_EXIT_DECAYBREAK_SUB_PI",
    "GUARD_EXIT_POST_ENTRY_BREAK_SUB_PI",
    "GUARD_EXIT_MATURE_2PI_PROTECT",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default if default is not None else {}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def same_entry(a, b):
    if not a or not b:
        return False

    a_ts = a.get("entry_timestamp_utc")
    b_ts = b.get("entry_timestamp_utc")
    if a_ts and b_ts and a_ts != b_ts:
        return False

    try:
        a_entry = float(a.get("entry_price"))
        b_entry = float(b.get("entry_price"))
        if abs(a_entry - b_entry) > 0.01:
            return False
    except Exception:
        # If price is unavailable, timestamp match is enough.
        if not (a_ts and b_ts and a_ts == b_ts):
            return False

    return True


def already_closed(entry_position):
    for r in read_jsonl(TRADES_PATH):
        if r.get("type") != "PAPER_SCALP_EXIT":
            continue
        if same_entry(entry_position, r):
            return True, r
    return False, None


def compute_pnl(open_position, live_price, guard_pnl=None):
    if guard_pnl and all(k in guard_pnl for k in ["current_value_usdt", "exit_fee_usdt", "net_pnl_usdt", "net_pnl_pct"]):
        return {
            "current_value_usdt": float(guard_pnl.get("current_value_usdt")),
            "exit_fee_usdt": float(guard_pnl.get("exit_fee_usdt")),
            "gross_pnl_usdt": float(guard_pnl.get("gross_pnl_usdt", 0.0)),
            "net_pnl_usdt": float(guard_pnl.get("net_pnl_usdt")),
            "net_pnl_pct": float(guard_pnl.get("net_pnl_pct")),
        }

    entry_usdt = float(open_position.get("entry_usdt") or 5.0)
    entry_fee = float(open_position.get("entry_fee_usdt") or (entry_usdt * FEE_RATE_DEFAULT))
    btc = float(open_position.get("btc") or 0.0)
    current_value = btc * float(live_price)
    exit_fee = current_value * FEE_RATE_DEFAULT
    gross = current_value - entry_usdt
    net = gross - entry_fee - exit_fee
    net_pct = net / entry_usdt if entry_usdt else 0.0

    return {
        "current_value_usdt": current_value,
        "exit_fee_usdt": exit_fee,
        "gross_pnl_usdt": gross,
        "net_pnl_usdt": net,
        "net_pnl_pct": net_pct,
    }


def ensure_today(state):
    state.setdefault("today", {})
    today = state["today"]
    today.setdefault("trades", 0)
    today.setdefault("wins", 0)
    today.setdefault("losses", 0)
    today.setdefault("realized_pnl_usdt", 0.0)
    today.setdefault("locked", False)
    today.setdefault("lock_reason", None)
    return today


def update_state_after_exit(state, open_position, pnl, live_price, guard_action, ts):
    current_value_after_exit_fee = pnl["current_value_usdt"] - pnl["exit_fee_usdt"]

    # Keep existing wallet schema if present.
    for cash_key in ["paper_usdt", "usdt", "cash_usdt", "balance_usdt"]:
        if cash_key in state:
            try:
                state[cash_key] = float(state.get(cash_key) or 0.0) + current_value_after_exit_fee
                break
            except Exception:
                pass

    for btc_key in ["paper_btc", "btc"]:
        if btc_key in state:
            try:
                state[btc_key] = 0.0
            except Exception:
                pass

    state["open_position"] = None

    today = ensure_today(state)
    today["trades"] = int(today.get("trades") or 0) + 1
    if pnl["net_pnl_usdt"] >= 0:
        today["wins"] = int(today.get("wins") or 0) + 1
    else:
        today["losses"] = int(today.get("losses") or 0) + 1
    today["realized_pnl_usdt"] = float(today.get("realized_pnl_usdt") or 0.0) + pnl["net_pnl_usdt"]

    state["last_guarded_exit"] = {
        "timestamp_utc": ts,
        "guard_action": guard_action,
        "exit_price": live_price,
        "entry_timestamp_utc": open_position.get("entry_timestamp_utc"),
        "entry_price": open_position.get("entry_price"),
        "net_pnl_usdt": pnl["net_pnl_usdt"],
        "net_pnl_pct": pnl["net_pnl_pct"],
    }

    return state


def main():
    ts = now_iso()
    guard = read_json(GUARD_REPORT_PATH, {})
    state = read_json(STATE_PATH, {})

    base = {
        "bot": BOT_NAME,
        "mode": MODE,
        "timestamp_utc": ts,
        "not_trading_advice": True,
        "real_orders": False,
        "paper_only": True,
        "idempotent": True,
    }

    guard_action = guard.get("guard_action")

    if guard_action not in EXIT_ACTIONS:
        out = {
            **base,
            "status": "PASS",
            "result": "NO_EXIT",
            "guard_action": guard_action,
            "reason": "Guard action is not an exit action.",
        }
        write_json(EXIT_REPORT_PATH, out)
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    # IMPORTANT: v0.3.25 uses state.open_position only.
    open_position = state.get("open_position")
    if not open_position:
        out = {
            **base,
            "status": "PASS",
            "result": "NO_EXIT_NO_OPEN_POSITION",
            "guard_action": guard_action,
            "reason": "Guard requested exit, but state.open_position is empty. Stale reports are not consumed.",
        }
        write_json(EXIT_REPORT_PATH, out)
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    guard_position = guard.get("open_position") or {}
    if not same_entry(open_position, guard_position):
        out = {
            **base,
            "status": "BLOCKED",
            "result": "NO_EXIT_STALE_OR_MISMATCHED_GUARD",
            "guard_action": guard_action,
            "reason": "Guard position does not match current state.open_position.",
            "state_position": open_position,
            "guard_position": guard_position,
        }
        write_json(EXIT_REPORT_PATH, out)
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    closed, existing = already_closed(open_position)
    if closed:
        out = {
            **base,
            "status": "BLOCKED",
            "result": "NO_EXIT_ALREADY_CLOSED",
            "guard_action": guard_action,
            "reason": "This entry already has a PAPER_SCALP_EXIT in trade log.",
            "state_position": open_position,
            "existing_exit": existing,
        }
        write_json(EXIT_REPORT_PATH, out)
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    live_price = float(guard.get("live_price") or 0.0)
    if live_price <= 0:
        out = {
            **base,
            "status": "ERROR",
            "result": "NO_EXIT_INVALID_LIVE_PRICE",
            "guard_action": guard_action,
            "reason": "Guard report has invalid live_price.",
        }
        write_json(EXIT_REPORT_PATH, out)
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    pnl = compute_pnl(open_position, live_price, guard.get("pnl") or {})
    outcome = "WIN" if pnl["net_pnl_usdt"] >= 0 else "LOSS"

    trade = {
        "timestamp_utc": ts,
        "bot": BOT_NAME,
        "type": "PAPER_SCALP_EXIT",
        "reason": f"GUARDED_{guard_action}",
        "outcome": outcome,
        "entry_timestamp_utc": open_position.get("entry_timestamp_utc"),
        "entry_price": open_position.get("entry_price"),
        "exit_price": live_price,
        "entry_usdt": open_position.get("entry_usdt"),
        "entry_fee_usdt": open_position.get("entry_fee_usdt"),
        "btc": open_position.get("btc"),
        "exit_fee_usdt": pnl["exit_fee_usdt"],
        "current_value_usdt": pnl["current_value_usdt"],
        "gross_pnl_usdt": pnl["gross_pnl_usdt"],
        "net_pnl_usdt": pnl["net_pnl_usdt"],
        "net_pnl_pct": pnl["net_pnl_pct"],
        "guard_action": guard_action,
        "guard_Fc_market": (guard.get("Fc_market") or {}).get("Fc"),
        "guard_phase_market": (guard.get("Fc_market") or {}).get("phase"),
        "guard_Fc_position": (guard.get("Fc_position") or {}).get("Fc"),
        "guard_phase_position": (guard.get("Fc_position") or {}).get("phase"),
        "guard_hard_break": (guard.get("Fc_position") or {}).get("hard_break"),
        "guard_decay_break": (guard.get("Fc_position") or {}).get("decay_break"),
        "guard_post_entry_break": (guard.get("Fc_position") or {}).get("post_entry_break"),
    }

    append_jsonl(TRADES_PATH, trade)
    updated_state = update_state_after_exit(state, open_position, pnl, live_price, guard_action, ts)
    write_json(STATE_PATH, updated_state)

    out = {
        **base,
        "status": "PASS",
        "result": "PAPER_POSITION_CLOSED_BY_GUARD",
        "guard_action": guard_action,
        "outcome": outcome,
        "entry_price": open_position.get("entry_price"),
        "exit_price": live_price,
        "net_pnl_usdt": pnl["net_pnl_usdt"],
        "net_pnl_pct": pnl["net_pnl_pct"],
        "trade_appended": str(TRADES_PATH),
        "state_updated": str(STATE_PATH),
        "trade": trade,
        "safety_note": "Paper only. No real orders were sent.",
    }

    write_json(EXIT_REPORT_PATH, out)
    print(json.dumps({
        "status": out["status"],
        "result": out["result"],
        "guard_action": out["guard_action"],
        "outcome": out["outcome"],
        "entry_price": out["entry_price"],
        "exit_price": out["exit_price"],
        "net_pnl_usdt": out["net_pnl_usdt"],
        "net_pnl_pct": out["net_pnl_pct"],
        "state_updated": out["state_updated"],
        "trade_appended": out["trade_appended"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
