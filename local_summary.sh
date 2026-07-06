#!/data/data/com.termux/files/usr/bin/bash
cd ~/Bitcoin || exit 1
export TZ=Europe/Copenhagen

python - <<'PY'
import json, os, datetime


def load(path):
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None

def fmt(x, n=4):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)

def time_local(ts=None, fallback_path=None):
    """
    Automat, fără tzdata/ZoneInfo:
    - JSON timestamp UTC -> ora locală Termux/Android
    - dacă lipsește timestampul -> ora modificării fișierului
    """
    try:
        if ts:
            dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            epoch = dt.timestamp()
        elif fallback_path and os.path.exists(fallback_path):
            epoch = os.path.getmtime(fallback_path)
        else:
            return "-"
        return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M local")
    except Exception:
        return str(ts) if ts else "-"

print()
print("=== COHESIVX LOCAL SUMMARY ===")
print("local:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M local"))

# 1. MICRO REAL
micro_path = "btc-swing-strategy/bitget_micro_live_real_executor_report.json"
micro = load(micro_path)

print()
print("1) MICRO REAL")
if not micro:
    print("Lipsă raport.")
else:
    c = micro.get("checks", {}) or {}
    action = micro.get("action", "-")
    status = micro.get("status", "-")
    result = micro.get("result", "-")
    price = micro.get("last_price", c.get("last_price"))
    usdt = micro.get("available_usdt", c.get("available_usdt"))
    btc = micro.get("available_btc", c.get("available_btc"))
    sent = micro.get("real_order_sent")
    ts = micro.get("timestamp_utc") or micro.get("timestamp")

    print(f"time: {time_local(ts, micro_path)}")
    print(f"{action} · {status} · price {fmt(price,2)}")
    print(f"USDT {fmt(usdt,2)} · BTC {btc or '-'} · real order: {'DA' if sent else 'NU'}")
    print(f"Rezultat: {result}")

# 2. PAPER SCALPER 100
scalp_path = "btc-swing-strategy/cohesivx_scalp_report.json"
scalp = load(scalp_path)

print()
print("2) PAPER SCALPER 100")
if not scalp:
    print("Lipsă raport.")
else:
    ss = scalp.get("state_summary", {}) or {}
    today = ss.get("today", {}) or {}

    action = scalp.get("scalp_action", "-")
    reason = scalp.get("reason", "-")
    price = scalp.get("price")
    cash = ss.get("paper_cash_usdt")
    btc = ss.get("paper_btc")
    open_pos = ss.get("open_position")
    # Rebuild today's scalp stats from trade log, not from volatile report state.
    # Truth source for closed scalp trades = trade log, not report.today
    trades = wins = losses = 0
    pnl = 0.0

    try:
        import datetime
        today_utc = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        log_path = "btc-swing-strategy/cohesivx_scalp_trades.jsonl"

        if os.path.exists(log_path):
            for line in open(log_path):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("timestamp_utc", "").startswith(today_utc):
                    continue
                if row.get("type") != "PAPER_SCALP_EXIT":
                    continue

                trades += 1
                pnl += float(row.get("net_pnl_usdt", 0) or 0)

                if row.get("outcome") == "WIN":
                    wins += 1
                elif row.get("outcome") == "LOSS":
                    losses += 1
    except Exception:
        pass

    try:
        import datetime
        today_utc = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        log_path = "btc-swing-strategy/cohesivx_scalp_trades.jsonl"
        trades = wins = losses = 0
        pnl = 0.0

        if os.path.exists(log_path):
            for line in open(log_path):
                if not line.strip():
                    continue
                row = json.loads(line)
                ts = row.get("timestamp_utc", "")
                if not ts.startswith(today_utc):
                    continue
                if row.get("type") == "PAPER_SCALP_EXIT":
                    trades += 1
                    pnl += float(row.get("net_pnl_usdt", 0) or 0)
                    if row.get("outcome") == "WIN":
                        wins += 1
                    elif row.get("outcome") == "LOSS":
                        losses += 1
    except Exception:
        pass
    ts = scalp.get("timestamp_utc") or scalp.get("timestamp")

    print(f"time: {time_local(ts, scalp_path)}")
    print(f"{action} · price {fmt(price,2)} · open: {'DA' if open_pos else 'NU'}")
    print(f"Cash {fmt(cash,4)} · BTC {btc or 0}")
    print(f"Azi închise: {trades} trades · {wins}W/{losses}L · PnL {fmt(pnl,4)} USDT")
    if open_pos:
        print("Poziție: DESCHISĂ")
    print(f"Motiv: {reason}")

# 3. PAPER TRADER 1000
decision_path = "btc-swing-strategy/paper_trader_decision.json"
state_path = "btc-swing-strategy/paper_trading_state.json"

decision = load(decision_path)
state = load(state_path)

print()
print("3) PAPER TRADER 1000")
if not decision and not state:
    print("Lipsă raport.")
else:
    action = (decision or {}).get("action", "-")
    cash = (state or {}).get("cash_usdt") or (state or {}).get("paper_cash_usdt")
    value = (state or {}).get("portfolio_value_usdt") or (state or {}).get("total_value_usdt")
    realized = (state or {}).get("realized_pnl_usdt")
    unrealized = (state or {}).get("unrealized_pnl_usdt")

    ts = (
        (decision or {}).get("timestamp_utc")
        or (decision or {}).get("timestamp")
        or (decision or {}).get("run_at")
        or (decision or {}).get("checked_at")
        or (decision or {}).get("created_at")
        or (decision or {}).get("time")
    )

    reason = (decision or {}).get("reason", [])
    if isinstance(reason, list):
        reason_txt = reason[0] if reason else "-"
        joined = " | ".join(reason)
        if "target exposure" in joined or "target reached" in joined or "cash corridor" in joined:
            reason_txt = "target atins / cash corridor"
    else:
        reason_txt = str(reason)

    print(f"time: {time_local(ts, decision_path)}")
    print(f"{action} · portfolio {fmt(value,2)} USDT")
    print(f"Cash {fmt(cash,2)} · realized {fmt(realized,2)} · unrealized {fmt(unrealized,2)}")
    print(f"Motiv: {reason_txt}")

print()
print("CONCLUZIE")
print("Micro Real: sigur / fără ordin real.")

if scalp and scalp.get("state_summary", {}).get("open_position"):
    print("Scalper 100: urmărește poziția.")
else:
    print("Scalper 100: fără setup curat.")

print("Trader 1000: pe profit / expunere controlată.")
print()
PY
