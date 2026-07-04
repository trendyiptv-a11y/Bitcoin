#!/usr/bin/env python3
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "btc-swing-strategy"

BITGET_SAFE = OUT_DIR / "bitget_safe_status.json"
REAL_EXECUTOR = OUT_DIR / "bitget_micro_live_real_executor_report.json"

PRICE_SERIES = OUT_DIR / "cohesivx_scalp_price_series.jsonl"
STATE_PATH = OUT_DIR / "cohesivx_scalp_state.json"
TRADES_PATH = OUT_DIR / "cohesivx_scalp_trades.jsonl"
REPORT_PATH = OUT_DIR / "cohesivx_scalp_report.json"

BOT_NAME = "CohesivX Daily Scalp Paper v0.2.3-4x2-report-signals"

PAPER_START_USDT = 100.0
MAX_TRADE_USDT = 5.0
FEE_RATE = 0.001  # 0.1% per side

MIN_SAMPLES = 30
MAX_SAMPLES_KEEP = 500

TAKE_PROFIT_NET_PCT = 0.0025   # +0.25% net după fees, calibrated for faster scalp harvest
STOP_LOSS_NET_PCT = -0.0035    # -0.35% net după fees
MAX_HOLD_CYCLES = 16           # 16 x 15m = 4 ore

MAX_TRADES_PER_DAY = 4
MAX_DAILY_LOSSES = 2

# v0.2 entry calibration: fewer but cleaner reversion entries.
RSI_ENTRY_MIN = 32.0
RSI_ENTRY_MAX = 48.0
MAX_RANGE_PCT_FOR_ENTRY = 0.015       # v0.1 used 0.035; v0.2 blocks noisy days
MIN_MOMENTUM_1_FOR_ENTRY = 0.0001     # short impulse must turn positive
MIN_MOMENTUM_2_FOR_ENTRY = 0.0002     # two-sample confirmation

# v0.2.1 breakout branch: paper-only, for missed momentum moves.
# This is intentionally controlled: it allows a small test entry only when price
# is near a local high, trend stack is positive, RSI is strong but not euphoric,
# range is still contained, and momentum has not rolled over.
BREAKOUT_RSI_MIN = 55.0
BREAKOUT_RSI_MAX = 76.0
BREAKOUT_MAX_RANGE_PCT = 0.018
BREAKOUT_NEAR_HIGH_PCT = 0.0035       # within 0.35% of recent high
BREAKOUT_MIN_DIST_EMA21 = 0.0010      # above EMA21, not just flat
BREAKOUT_MAX_DIST_EMA21 = 0.0090      # avoid buying too stretched
BREAKOUT_MIN_MOMENTUM_1 = 0.0000
BREAKOUT_MIN_MOMENTUM_2 = -0.0010     # allow tiny pause, block clear rollover

DAILY_STOP_USDT = -2.0
DAILY_LOCK_PROFIT_USDT = 2.0

BLOCK_LONG_ACTIONS = {
    "REDUCE_RISK",
    "TAKE_PROFIT_SMALL",
    "TAKE_PROFIT_MEDIUM",
}

SOFT_ALLOW_ACTIONS = {
    "HOLD",
    "OBSERVE",
    "HOLD_CASH_CORRIDOR",
    "ACCUMULATE_SMALL",
    "OBSERVE_ACCUMULATE_SMALL",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def today_key():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_json(path, default=None):
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def write_json(path, data):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def append_jsonl(path, data):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def extract_price(bitget_safe):
    try:
        data = bitget_safe.get("public_ticker", {}).get("raw", {}).get("data")
        if isinstance(data, list) and data:
            return float(data[0].get("lastPr"))
        if isinstance(data, dict):
            return float(data.get("lastPr"))
    except Exception:
        return None
    return None


def fetch_live_bitget_price(timeout=8):
    """Fetch a fresh public BTCUSDT spot price. No API keys; paper-only.
    This prevents the scalper from reusing a stale bitget_safe_status.json snapshot.
    """
    url = "https://api.bitget.com/api/v2/spot/market/tickers?symbol=BTCUSDT"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cohesivx-scalper-paper/0.2.2"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data")
        if isinstance(data, list) and data:
            price = float(data[0].get("lastPr"))
        elif isinstance(data, dict):
            price = float(data.get("lastPr"))
        else:
            return None
        return price if price > 0 else None
    except Exception:
        return None


def trailing_same_price_count(prices):
    if not prices:
        return 0
    last = prices[-1]
    count = 0
    for p in reversed(prices):
        if p == last:
            count += 1
        else:
            break
    return count


def ema(values, period):
    if not values:
        return None
    k = 2 / (period + 1)
    result = values[0]
    for v in values[1:]:
        result = v * k + result * (1 - k)
    return result


def rsi(values, period=14):
    if len(values) < period + 1:
        return None

    gains = []
    losses = []

    recent = values[-(period + 1):]
    for i in range(1, len(recent)):
        diff = recent[i] - recent[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Flat market: no gains and no losses. Neutral RSI is clearer than a fake 0/100.
    if avg_gain == 0 and avg_loss == 0:
        return 50.0

    if avg_loss == 0:
        return 100.0

    if avg_gain == 0:
        return 0.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def load_state():
    state = load_json(STATE_PATH, default={})
    if not state:
        state = {
            "bot": BOT_NAME,
            "paper_cash_usdt": PAPER_START_USDT,
            "paper_btc": 0.0,
            "open_position": None,
            "daily": {},
            "total_realized_pnl_usdt": 0.0,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
        }
    return state


def ensure_daily(state):
    key = today_key()
    if key not in state["daily"]:
        state["daily"][key] = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "realized_pnl_usdt": 0.0,
            "locked": False,
            "lock_reason": None,
        }

    # Backward compatibility with old v0.1 state files.
    state["daily"][key].setdefault("wins", 0)
    state["daily"][key].setdefault("losses", 0)
    state["daily"][key].setdefault("trades", 0)
    state["daily"][key].setdefault("realized_pnl_usdt", 0.0)
    state["daily"][key].setdefault("locked", False)
    state["daily"][key].setdefault("lock_reason", None)

    return state["daily"][key]


def net_position_pnl(position, current_price):
    entry_usdt = position["entry_usdt"]
    btc = position["btc"]
    entry_fee = position["entry_fee_usdt"]

    current_value = btc * current_price
    exit_fee = current_value * FEE_RATE

    gross = current_value - entry_usdt
    net = gross - entry_fee - exit_fee
    net_pct = net / entry_usdt if entry_usdt > 0 else 0

    return {
        "current_value_usdt": current_value,
        "gross_pnl_usdt": gross,
        "entry_fee_usdt": entry_fee,
        "exit_fee_usdt": exit_fee,
        "net_pnl_usdt": net,
        "net_pnl_pct": net_pct,
    }


def close_position(state, daily, price, reason):
    pos = state["open_position"]
    pnl = net_position_pnl(pos, price)

    state["paper_cash_usdt"] += pnl["current_value_usdt"] - pnl["exit_fee_usdt"]
    state["paper_btc"] = 0.0
    state["open_position"] = None

    state["total_realized_pnl_usdt"] += pnl["net_pnl_usdt"]
    daily["realized_pnl_usdt"] += pnl["net_pnl_usdt"]

    state["total_trades"] += 1
    daily["trades"] += 1

    if pnl["net_pnl_usdt"] >= 0:
        state["wins"] += 1
        daily["wins"] = int(daily.get("wins", 0)) + 1
        outcome = "WIN"
    else:
        state["losses"] += 1
        daily["losses"] = int(daily.get("losses", 0)) + 1
        outcome = "LOSS"

    trade = {
        "timestamp_utc": now_iso(),
        "bot": BOT_NAME,
        "type": "PAPER_SCALP_EXIT",
        "reason": reason,
        "outcome": outcome,
        "entry_price": pos["entry_price"],
        "exit_price": price,
        "entry_usdt": pos["entry_usdt"],
        "btc": pos["btc"],
        **pnl,
    }

    append_jsonl(TRADES_PATH, trade)
    return trade


def open_position(state, daily, price, reason):
    trade_usdt = min(MAX_TRADE_USDT, state["paper_cash_usdt"])

    if trade_usdt <= 0:
        return None

    entry_fee = trade_usdt * FEE_RATE
    net_usdt = trade_usdt - entry_fee
    btc = net_usdt / price

    state["paper_cash_usdt"] -= trade_usdt
    state["paper_btc"] = btc
    state["open_position"] = {
        "entry_timestamp_utc": now_iso(),
        "entry_price": price,
        "entry_usdt": trade_usdt,
        "entry_fee_usdt": entry_fee,
        "btc": btc,
        "hold_cycles": 0,
        "reason": reason,
    }

    trade = {
        "timestamp_utc": now_iso(),
        "bot": BOT_NAME,
        "type": "PAPER_SCALP_BUY",
        "reason": reason,
        "entry_price": price,
        "entry_usdt": trade_usdt,
        "entry_fee_usdt": entry_fee,
        "btc": btc,
    }

    append_jsonl(TRADES_PATH, trade)
    return trade


def main():
    bitget_safe = load_json(BITGET_SAFE)
    real_exec = load_json(REAL_EXECUTOR)

    safe_price = extract_price(bitget_safe)
    live_price = fetch_live_bitget_price()
    if live_price is not None:
        price = live_price
        price_source = "bitget_public_live"
    else:
        price = safe_price
        price_source = "bitget_safe_status_fallback"

    action = real_exec.get("action") or "UNKNOWN"
    result = real_exec.get("result") or "UNKNOWN"

    report = {
        "timestamp_utc": now_iso(),
        "bot": BOT_NAME,
        "mode": "PAPER_ONLY",
        "real_order_sent": False,
        "price": price,
        "price_source": price_source,
        "v063_action": action,
        "v063_result": result,
        "status": "UNKNOWN",
        "scalp_action": "HOLD_SCALP",
        "reason": None,
        "indicators": {},
        "state_summary": {},
        "last_trade": None,
    }

    if price is None or price <= 0:
        report["status"] = "BLOCKED"
        report["reason"] = "NO_VALID_PRICE"
        write_json(REPORT_PATH, report)
        print(json.dumps(report, indent=2))
        return

    # salvăm snapshot preț
    append_jsonl(PRICE_SERIES, {
        "timestamp_utc": now_iso(),
        "price": price,
        "price_source": price_source,
        "v063_action": action,
    })

    rows = read_jsonl(PRICE_SERIES)[-MAX_SAMPLES_KEEP:]
    prices = [float(r["price"]) for r in rows if "price" in r]

    if len(rows) > MAX_SAMPLES_KEEP:
        PRICE_SERIES.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")

    state = load_state()
    daily = ensure_daily(state)

    # daily locks
    if daily["realized_pnl_usdt"] <= DAILY_STOP_USDT:
        daily["locked"] = True
        daily["lock_reason"] = "DAILY_STOP"
    if daily["realized_pnl_usdt"] >= DAILY_LOCK_PROFIT_USDT:
        daily["locked"] = True
        daily["lock_reason"] = "DAILY_PROFIT_LOCK"
    if int(daily.get("losses", 0)) >= MAX_DAILY_LOSSES:
        daily["locked"] = True
        daily["lock_reason"] = "DAILY_LOSS_COUNT_LOCK"

    if len(prices) < MIN_SAMPLES:
        report["status"] = "WARMING_UP"
        report["reason"] = f"Need {MIN_SAMPLES} samples, have {len(prices)}."
        report["state_summary"] = summarize_state(state, price)
        write_json(STATE_PATH, state)
        write_json(REPORT_PATH, report)
        print(json.dumps(report, indent=2))
        return

    ema9 = ema(prices[-30:], 9)
    ema21 = ema(prices[-60:], 21)
    ema50 = ema(prices[-120:] if len(prices) >= 120 else prices, 50)
    rsi14 = rsi(prices, 14)

    last = prices[-1]
    prev = prices[-2]
    momentum_1 = (last - prev) / prev if prev else 0
    prev2 = prices[-3] if len(prices) >= 3 else None
    momentum_2 = (last - prev2) / prev2 if prev2 else 0

    recent_low = min(prices[-24:])   # aproximativ ultimele 6 ore dacă rulează la 15m
    recent_high = max(prices[-24:])
    range_pct = (recent_high - recent_low) / last if last else 0

    dist_ema21 = (last - ema21) / ema21 if ema21 else 0

    report["indicators"] = {
        "samples": len(prices),
        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,
        "rsi14": rsi14,
        "momentum_1": momentum_1,
        "momentum_2": momentum_2,
        "recent_low": recent_low,
        "recent_high": recent_high,
        "range_pct": range_pct,
        "dist_ema21": dist_ema21,
        "fee_rate": FEE_RATE,
        "price_source": price_source,
        "data_quality": {
            "unique_prices_last30": len(set(prices[-30:])),
            "trailing_same_price_count": trailing_same_price_count(prices),
            "stale_price_warning": trailing_same_price_count(prices) >= 6,
        },
        "breakout_config": {
            "rsi_min": BREAKOUT_RSI_MIN,
            "rsi_max": BREAKOUT_RSI_MAX,
            "max_range_pct": BREAKOUT_MAX_RANGE_PCT,
            "near_high_pct": BREAKOUT_NEAR_HIGH_PCT,
            "min_dist_ema21": BREAKOUT_MIN_DIST_EMA21,
            "max_dist_ema21": BREAKOUT_MAX_DIST_EMA21,
            "min_momentum_1": BREAKOUT_MIN_MOMENTUM_1,
            "min_momentum_2": BREAKOUT_MIN_MOMENTUM_2,
        },
    }

    # v0.2.3 reporting fix:
    # calculate these every run, even when an open position exits.
    # This makes the dashboard transparent after SL/TP/TIME exits.
    near_recent_high = (recent_high - last) / last if last else 1.0
    trend_stack_positive = (
        ema9 is not None
        and ema21 is not None
        and ema50 is not None
        and ema9 > ema21 > ema50
    )
    reversion_signal = (
        ema21 is not None
        and rsi14 is not None
        and dist_ema21 <= -0.0015
        and RSI_ENTRY_MIN <= rsi14 <= RSI_ENTRY_MAX
        and momentum_1 > MIN_MOMENTUM_1_FOR_ENTRY
        and momentum_2 > MIN_MOMENTUM_2_FOR_ENTRY
        and range_pct < MAX_RANGE_PCT_FOR_ENTRY
        and action in SOFT_ALLOW_ACTIONS
        and int(daily.get("losses", 0)) < MAX_DAILY_LOSSES
        and not daily.get("locked", False)
    )
    breakout_signal = (
        trend_stack_positive
        and rsi14 is not None
        and BREAKOUT_RSI_MIN <= rsi14 <= BREAKOUT_RSI_MAX
        and 0 <= near_recent_high <= BREAKOUT_NEAR_HIGH_PCT
        and BREAKOUT_MIN_DIST_EMA21 <= dist_ema21 <= BREAKOUT_MAX_DIST_EMA21
        and momentum_1 >= BREAKOUT_MIN_MOMENTUM_1
        and momentum_2 >= BREAKOUT_MIN_MOMENTUM_2
        and range_pct < BREAKOUT_MAX_RANGE_PCT
        and action in SOFT_ALLOW_ACTIONS
        and int(daily.get("losses", 0)) < MAX_DAILY_LOSSES
        and not daily.get("locked", False)
    )

    report["indicators"]["near_recent_high_pct"] = near_recent_high
    report["indicators"]["trend_stack_positive"] = trend_stack_positive
    report["indicators"]["reversion_signal"] = reversion_signal
    report["indicators"]["breakout_signal"] = breakout_signal

    # Dacă există poziție, verificăm exit.
    if state["open_position"]:
        pos = state["open_position"]
        pos["hold_cycles"] = int(pos.get("hold_cycles", 0)) + 1

        pnl = net_position_pnl(pos, price)

        if pnl["net_pnl_pct"] >= TAKE_PROFIT_NET_PCT:
            trade = close_position(state, daily, price, "TAKE_PROFIT_NET")
            report["status"] = "PASS"
            report["scalp_action"] = "PAPER_SCALP_EXIT_TP"
            report["reason"] = "Net take profit reached."
            report["last_trade"] = trade

        elif pnl["net_pnl_pct"] <= STOP_LOSS_NET_PCT:
            trade = close_position(state, daily, price, "STOP_LOSS_NET")
            report["status"] = "PASS"
            report["scalp_action"] = "PAPER_SCALP_EXIT_SL"
            report["reason"] = "Net stop loss reached."
            report["last_trade"] = trade

        elif pos["hold_cycles"] >= MAX_HOLD_CYCLES:
            trade = close_position(state, daily, price, "TIME_EXIT")
            report["status"] = "PASS"
            report["scalp_action"] = "PAPER_SCALP_EXIT_TIME"
            report["reason"] = "Max hold time reached."
            report["last_trade"] = trade

        else:
            report["status"] = "PASS"
            report["scalp_action"] = "HOLD_OPEN_SCALP"
            report["reason"] = "Open paper position held."
            report["open_position_pnl"] = pnl

    else:
        # Dacă nu există poziție, verificăm intrare.
        if daily["locked"]:
            report["status"] = "PASS"
            report["scalp_action"] = "SCALP_DAILY_LOCK"
            report["reason"] = daily["lock_reason"]

        elif daily["trades"] >= MAX_TRADES_PER_DAY:
            report["status"] = "PASS"
            report["scalp_action"] = "SCALP_MAX_TRADES_DAY"
            report["reason"] = "Max daily trades reached."

        elif action in BLOCK_LONG_ACTIONS:
            report["status"] = "PASS"
            report["scalp_action"] = "HOLD_SCALP"
            report["reason"] = f"v0.6.3 action blocks long scalp: {action}"

        else:
            # Signals were calculated above for reporting transparency.
            buy_signal = reversion_signal

            if buy_signal:
                trade = open_position(state, daily, price, "COHESIVE_REVERSION_SCALP_V02")
                report["status"] = "PASS"
                report["scalp_action"] = "PAPER_SCALP_BUY"
                report["reason"] = "v0.2 reversion entry: price below EMA21, RSI recovery corridor, two-step positive momentum, low range."
                report["last_trade"] = trade
            elif breakout_signal:
                trade = open_position(state, daily, price, "COHESIVE_BREAKOUT_SCALP_V021")
                report["status"] = "PASS"
                report["scalp_action"] = "PAPER_SCALP_BUY_BREAKOUT"
                report["reason"] = "v0.2.1 breakout entry: positive EMA stack, RSI strength, near local high, controlled range, momentum not rolled over."
                report["last_trade"] = trade
            else:
                report["status"] = "PASS"
                report["scalp_action"] = "HOLD_SCALP"
                report["reason"] = "No valid cohesive scalp setup."

    report["state_summary"] = summarize_state(state, price)

    write_json(STATE_PATH, state)
    write_json(REPORT_PATH, report)
    print(json.dumps(report, indent=2))


def summarize_state(state, price):
    open_pos = state.get("open_position")
    open_pnl = None

    if open_pos:
        open_pnl = net_position_pnl(open_pos, price)

    total = state.get("paper_cash_usdt", 0) + state.get("paper_btc", 0) * price

    wins = state.get("wins", 0)
    losses = state.get("losses", 0)
    trades = state.get("total_trades", 0)
    win_rate = wins / trades if trades > 0 else None

    return {
        "paper_cash_usdt": state.get("paper_cash_usdt", 0),
        "paper_btc": state.get("paper_btc", 0),
        "paper_total_value_usdt": total,
        "open_position": open_pos,
        "open_position_pnl": open_pnl,
        "total_realized_pnl_usdt": state.get("total_realized_pnl_usdt", 0),
        "total_trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "today": state.get("daily", {}).get(today_key(), {}),
    }


if __name__ == "__main__":
    main()
