#!/data/data/com.termux/files/usr/bin/bash
cd ~/Bitcoin || exit 1

echo
echo "=== COHESIVX LOCAL STATUS ==="
date -u
echo

echo "=== PROCESE ACTIVE ==="
pgrep -af "run_cohesivx_15m|run_scalper_5m|paper_trader|paper_trading|paper_watch|profit_watch" || echo "Niciun proces bot găsit"

echo
echo "=== 1. MICRO REAL / BITGET ==="
python - <<'PY'
import json, os
p="btc-swing-strategy/bitget_micro_live_real_executor_report.json"

if not os.path.exists(p):
    print("Nu există raport Micro Real:", p)
else:
    r=json.load(open(p))
    print("time:", r.get("timestamp_utc"))
    print("bot:", r.get("bot"))
    print("live_mode:", r.get("live_mode"))
    print("action:", r.get("action"))
    print("result:", r.get("result"))
    print("status:", r.get("status"))
    print("audit:", r.get("audit"))
    print("price:", r.get("last_price"))
    print("available_usdt:", r.get("available_usdt"))
    print("available_btc:", r.get("available_btc"))
    print("max_usdt:", r.get("max_usdt"))
    print("real_order_sent:", r.get("real_order_sent"))
PY

echo
echo "=== 2. PAPER SCALPER / 100 USDT ==="
python - <<'PY'
import json, os
report="btc-swing-strategy/cohesivx_scalp_report.json"
state="btc-swing-strategy/cohesivx_scalp_state.json"

if not os.path.exists(report):
    print("Nu există raport Paper Scalper:", report)
else:
    r=json.load(open(report))
    ss=r.get("state_summary",{})
    print("time:", r.get("timestamp_utc"))
    print("action:", r.get("scalp_action"))
    print("reason:", r.get("reason"))
    print("price:", r.get("price"))
    print("cash:", ss.get("paper_cash_usdt"))
    print("btc:", ss.get("paper_btc"))
    print("open_position:", ss.get("open_position"))
    print("open_pnl:", ss.get("open_position_pnl"))
    print("today:", ss.get("today"))

if os.path.exists(state):
    s=json.load(open(state))
    print("state_cash:", s.get("paper_cash_usdt"))
    print("state_btc:", s.get("paper_btc"))
PY

echo
echo "=== ULTIMELE TRADE-URI PAPER SCALPER ==="
tail -n 8 btc-swing-strategy/cohesivx_scalp_trades.jsonl 2>/dev/null || echo "Nu există trade log scalper."

echo
echo "=== 3. PAPER TRADER / 1000 USDT ==="
python - <<'PY'
import json, os

decision="btc-swing-strategy/paper_trader_decision.json"
state="btc-swing-strategy/paper_trading_state.json"
log="btc-swing-strategy/paper_trading_log.csv"

if os.path.exists(decision):
    d=json.load(open(decision))
    print("--- decision ---")
    for k in [
        "timestamp_utc",
        "symbol",
        "action",
        "decision",
        "signal",
        "reason",
        "status",
        "price",
        "last_price",
        "cohesion",
        "regime",
        "risk",
    ]:
        if k in d:
            print(f"{k}:", d.get(k))
else:
    print("Nu există decizie Paper Trader:", decision)

if os.path.exists(state):
    s=json.load(open(state))
    print("--- state ---")
    for k in [
        "paper_cash_usdt",
        "cash_usdt",
        "paper_btc",
        "btc",
        "position_open",
        "open_position",
        "entry_price",
        "entry_usdt",
        "portfolio_value_usdt",
        "total_value_usdt",
        "realized_pnl_usdt",
        "unrealized_pnl_usdt",
    ]:
        if k in s:
            print(f"{k}:", s.get(k))
else:
    print("Nu există stare Paper Trader:", state)

if os.path.exists(log):
    print("--- ultimele randuri log CSV ---")
    try:
        lines=open(log).read().splitlines()
        for line in lines[-6:]:
            print(line)
    except Exception as e:
        print("Eroare citire log:", e)
else:
    print("Nu există log Paper Trader:", log)
PY

echo
echo "=== FIȘIERE ACTUALIZATE RECENT ==="
ls -lt btc-swing-strategy | head -18
