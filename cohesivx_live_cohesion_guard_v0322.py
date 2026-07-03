#!/usr/bin/env python3
# CohesivX Live Cohesion Guard v0.3.22
# PAPER-ONLY / REPORT-ONLY guard for an already open scalper position.
#
# Purpose:
#   After the scalper has entered, this guard can run more frequently than the
#   dashboard cycle and evaluate live price against the cohesive formula.
#
# Formula:
#   Fc = (N * M * E) / R^2
#
# Reads:
#   btc-swing-strategy/cohesivx_scalp_state.json
#   btc-swing-strategy/cohesivx_scalp_report.json
#   btc-swing-strategy/cohesivx_scalp_price_series.jsonl
#
# Writes:
#   btc-swing-strategy/cohesivx_live_cohesion_guard_report.json
#   btc-swing-strategy/cohesivx_live_cohesion_guard_log.jsonl
#
# Safety:
#   - No real orders.
#   - No paper state mutation.
#   - It only reports: GUARD_HOLD / GUARD_EXIT_*.
#
# Integration later:
#   If validated, a separate patch can let the scalper paper bot consume
#   guard_action and close the paper position. This file does not do that.

import json
import math
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "btc-swing-strategy"

STATE_PATH = OUT_DIR / "cohesivx_scalp_state.json"
REPORT_PATH = OUT_DIR / "cohesivx_scalp_report.json"
PRICE_SERIES = OUT_DIR / "cohesivx_scalp_price_series.jsonl"

GUARD_REPORT = OUT_DIR / "cohesivx_live_cohesion_guard_report.json"
GUARD_LOG = OUT_DIR / "cohesivx_live_cohesion_guard_log.jsonl"

BOT_NAME = "CohesivX Live Cohesion Guard v0.3.22-formula-only"
MODE = "PAPER_ONLY_REPORT_ONLY"

PI = math.pi
TWO_PI = 2 * math.pi
MID_PI_2PI = (PI + TWO_PI) / 2

FEE_RATE = 0.001
MIN_PRICE_POINTS = 60

BITGET_TICKER_URL = "https://api.bitget.com/api/v2/spot/market/tickers?symbol=BTCUSDT"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default if default is not None else {}


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


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def clamp(x, lo=0.0, hi=1.0):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def safe_div(a, b, fallback=0.0):
    try:
        if b == 0:
            return fallback
        return a / b
    except Exception:
        return fallback


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
    recent = values[-(period + 1):]
    gains, losses = [], []
    for i in range(1, len(recent)):
        diff = recent[i] - recent[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def phase(fc):
    if fc < PI:
        return "SUB_PI"
    if fc < MID_PI_2PI:
        return "PI_TO_MID"
    if fc < TWO_PI:
        return "MID_TO_2PI"
    return "OVER_2PI"


def indicators(prices):
    last = prices[-1]
    prev = prices[-2] if len(prices) >= 2 else last
    prev2 = prices[-3] if len(prices) >= 3 else prev

    ema9 = ema(prices[-30:], 9)
    ema21 = ema(prices[-60:], 21)
    ema50 = ema(prices[-120:] if len(prices) >= 120 else prices, 50)
    rsi14 = rsi(prices, 14)

    momentum_1 = safe_div(last - prev, prev)
    momentum_2 = safe_div(last - prev2, prev2)

    recent = prices[-24:]
    recent_low = min(recent)
    recent_high = max(recent)
    range_pct = safe_div(recent_high - recent_low, last)
    dist_ema21 = safe_div(last - ema21, ema21) if ema21 else 0.0
    near_high = safe_div(recent_high - last, last, 1.0)

    trend_stack_ratio = 0.0
    if ema9 and ema21 and ema50:
        trend_stack_ratio = safe_div((ema9 - ema21) + (ema21 - ema50), last)

    return {
        "price": last,
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
        "near_recent_high_pct": near_high,
        "trend_stack_ratio": trend_stack_ratio,
    }


def adaptive_scale(values, floor=1e-9):
    vals = [abs(v) for v in values if v is not None]
    if not vals:
        return floor
    vals.sort()
    mid = vals[len(vals) // 2]
    return max(mid, floor)


def local_scales(prices):
    returns = []
    for i in range(1, len(prices)):
        returns.append(safe_div(prices[i] - prices[i - 1], prices[i - 1]))

    recent = prices[-60:] if len(prices) >= 60 else prices[:]
    recent_returns = returns[-60:] if len(returns) >= 60 else returns[:]

    if len(recent) < 2:
        base = 1e-6
        return {"ret": base, "range": base, "dist": base}

    ret_scale = adaptive_scale(recent_returns, 1e-6)
    range_scale = safe_div(max(recent) - min(recent), recent[-1], 1e-6)
    dist_scale = max(range_scale / 3, ret_scale, 1e-6)
    return {
        "ret": ret_scale,
        "range": max(range_scale, 1e-6),
        "dist": max(dist_scale, 1e-6),
    }


def cohesion_market(ind, samples, scales):
    # Fc_market = (N * M * E) / R^2

    m1 = ind["momentum_1"]
    m2 = ind["momentum_2"]
    dist = ind["dist_ema21"]
    rng = ind["range_pct"]
    near_high = ind["near_recent_high_pct"]
    trend = ind["trend_stack_ratio"]
    rsi14 = ind["rsi14"] if ind["rsi14"] is not None else 50.0

    ret_s = scales["ret"]
    range_s = scales["range"]
    dist_s = scales["dist"]

    N = 0.5 + 0.5 * clamp(samples / 240.0)

    momentum_flow = math.tanh((m1 + m2) / (2 * ret_s))
    trend_flow = math.tanh(trend / dist_s)
    M = 0.5 + 0.5 * clamp(0.68 * momentum_flow + 0.32 * trend_flow, -1, 1)

    impulse_energy = clamp(abs(m1) / (ret_s * 3))
    structure_energy = clamp(abs(dist) / (dist_s * 2))
    breakout_energy = clamp(1.0 - near_high / max(range_s, 1e-9))
    rsi_distance = abs(rsi14 - 50.0) / 50.0
    E = clamp(0.34 * impulse_energy + 0.26 * structure_energy + 0.25 * breakout_energy + 0.15 * rsi_distance)

    fee_pressure = 2 * FEE_RATE
    range_disorder = clamp(rng / max(range_s * 1.65, 1e-9))
    overextension = clamp(abs(dist) / max(range_s * 0.82, 1e-9))
    rollover = clamp(max(0.0, -m1) / max(ret_s * 1.55, 1e-9))
    disagreement = clamp(abs(m1 - m2) / max(ret_s * 4.0, 1e-9))
    late_cycle_pressure = clamp((abs(rsi14 - 50.0) / 50.0) * (abs(dist) / max(range_s, 1e-9)))
    thin_profit_space = clamp((fee_pressure * 5.0) / max(abs(m1) + abs(m2) + ret_s, 1e-9))

    R = (
        0.46
        + 1.25 * fee_pressure
        + 0.36 * range_disorder
        + 0.34 * overextension
        + 0.30 * rollover
        + 0.22 * disagreement
        + 0.18 * late_cycle_pressure
        + 0.16 * thin_profit_space
    )
    R = max(R, 0.25)

    raw = (N * M * E) / (R ** 2)
    Fc = raw * TWO_PI

    return {
        "Fc": Fc,
        "phase": phase(Fc),
        "N": N,
        "M": M,
        "E": E,
        "R": R,
        "raw": raw,
        "R_components": {
            "fee_pressure": fee_pressure,
            "range_disorder": range_disorder,
            "overextension": overextension,
            "rollover": rollover,
            "disagreement": disagreement,
            "late_cycle_pressure": late_cycle_pressure,
            "thin_profit_space": thin_profit_space,
        },
        "scales": scales,
    }


def net_position_pnl(pos, price):
    entry_usdt = float(pos.get("entry_usdt") or 5.0)
    btc = float(pos.get("btc") or 0.0)
    entry_fee = float(pos.get("entry_fee_usdt") or (entry_usdt * FEE_RATE))
    current_value = btc * price
    exit_fee = current_value * FEE_RATE
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


def cohesion_position(fc_market, pos, pnl):
    # Fc_position = (N * M * E) / R^2

    net_pct = pnl["net_pnl_pct"]
    best = max(float(pos.get("best_net_pct", net_pct) or net_pct), net_pct)
    giveback = max(0.0, best - net_pct)

    fee_unit = max(2 * FEE_RATE, 1e-9)
    entry_price = float(pos["entry_price"])
    current_value = pnl["current_value_usdt"]
    current_price = current_value / max(float(pos.get("btc") or 0.0), 1e-12)

    gross_move_pct = safe_div(current_price - entry_price, entry_price)
    break_even_gap = max(0.0, -net_pct)

    N = fc_market["N"]

    net_flow = math.tanh(net_pct / fee_unit)
    gross_flow = math.tanh(gross_move_pct / fee_unit)
    market_flow = math.tanh((fc_market["Fc"] - PI) / PI)
    M = 0.5 + 0.5 * clamp(0.50 * net_flow + 0.30 * gross_flow + 0.20 * market_flow, -1, 1)

    favorable_energy = clamp(max(net_pct, 0.0) / (3.0 * fee_unit))
    best_energy = clamp(max(best, 0.0) / (3.0 * fee_unit))
    gross_recovery_energy = clamp(max(gross_move_pct, 0.0) / (3.0 * fee_unit))
    market_support_energy = clamp(max(fc_market["Fc"] - PI, 0.0) / PI)
    break_even_recovery = clamp(1.0 - break_even_gap / (2.25 * fee_unit))

    E = clamp(
        0.28 * favorable_energy
        + 0.18 * best_energy
        + 0.24 * gross_recovery_energy
        + 0.18 * market_support_energy
        + 0.12 * break_even_recovery
    )

    true_adverse_pressure = clamp(max(-gross_move_pct, 0.0) / (1.50 * fee_unit))
    net_loss_pressure = clamp(max(-net_pct, 0.0) / (2.40 * fee_unit))
    giveback_pressure = clamp(giveback / (2.00 * fee_unit))
    entry_fc = float((pos.get("entry_fc") or {}).get("Fc") or MID_PI_2PI)
    entry_quality_pressure = clamp(max(0.0, MID_PI_2PI - entry_fc) / MID_PI_2PI)
    market_break_pressure = clamp(max(0.0, PI - fc_market["Fc"]) / PI)
    market_position_divergence = clamp(max(0.0, fc_market["Fc"] - entry_fc) / TWO_PI) if gross_move_pct < 0 else 0.0

    R = (
        0.42
        + 1.15 * 2 * FEE_RATE
        + 0.42 * true_adverse_pressure
        + 0.26 * net_loss_pressure
        + 0.36 * giveback_pressure
        + 0.20 * entry_quality_pressure
        + 0.30 * market_break_pressure
        + 0.18 * market_position_divergence
    )
    R = max(R, 0.25)

    raw = (N * M * E) / (R ** 2)
    Fc = raw * TWO_PI

    sub_pi_depth = clamp(max(0.0, PI - Fc) / PI)

    hard_break = clamp(
        0.40 * market_break_pressure
        + 0.35 * true_adverse_pressure
        + 0.25 * net_loss_pressure
    )

    decay_break = clamp(
        0.35 * market_break_pressure
        + 0.30 * net_loss_pressure
        + 0.20 * true_adverse_pressure
        + 0.15 * sub_pi_depth
    )

    post_entry_break = clamp(
        0.30 * market_break_pressure
        + 0.25 * sub_pi_depth
        + 0.20 * net_loss_pressure
        + 0.15 * true_adverse_pressure
        + 0.10 * clamp(max(0.0, -gross_move_pct) / (1.50 * fee_unit))
    )

    return {
        "Fc": Fc,
        "phase": phase(Fc),
        "N": N,
        "M": M,
        "E": E,
        "R": R,
        "raw": raw,
        "net_pnl_pct": net_pct,
        "best_net_pct": best,
        "giveback_pct": giveback,
        "gross_move_pct": gross_move_pct,
        "break_even_gap": break_even_gap,
        "market_Fc": fc_market["Fc"],
        "market_phase": fc_market["phase"],
        "hard_break": hard_break,
        "decay_break": decay_break,
        "post_entry_break": post_entry_break,
        "hard_break_components": {
            "market_break_pressure": market_break_pressure,
            "true_adverse_pressure": true_adverse_pressure,
            "net_loss_pressure": net_loss_pressure,
        },
        "decay_break_components": {
            "market_break_pressure": market_break_pressure,
            "net_loss_pressure": net_loss_pressure,
            "true_adverse_pressure": true_adverse_pressure,
            "sub_pi_depth": sub_pi_depth,
        },
        "post_entry_break_components": {
            "market_break_pressure": market_break_pressure,
            "sub_pi_depth": sub_pi_depth,
            "net_loss_pressure": net_loss_pressure,
            "true_adverse_pressure": true_adverse_pressure,
            "gross_adverse_pressure": clamp(max(0.0, -gross_move_pct) / (1.50 * fee_unit)),
        },
        "E_components": {
            "favorable_energy": favorable_energy,
            "best_energy": best_energy,
            "gross_recovery_energy": gross_recovery_energy,
            "market_support_energy": market_support_energy,
            "break_even_recovery": break_even_recovery,
        },
        "R_components": {
            "true_adverse_pressure": true_adverse_pressure,
            "net_loss_pressure": net_loss_pressure,
            "giveback_pressure": giveback_pressure,
            "entry_quality_pressure": entry_quality_pressure,
            "market_break_pressure": market_break_pressure,
            "market_position_divergence": market_position_divergence,
        },
    }


def get_live_price():
    with urllib.request.urlopen(BITGET_TICKER_URL, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data") or []
    if not data:
        raise RuntimeError("No Bitget ticker data")
    return float(data[0]["lastPr"])


def find_open_position(state, report):
    # Preferred: raw state file.
    for src in (state, report.get("state_summary") or {}, report):
        if isinstance(src, dict):
            pos = src.get("open_position")
            if pos:
                return pos
    return None


def load_prices_with_live(live_price):
    rows = read_jsonl(PRICE_SERIES)
    prices = []
    for r in rows:
        try:
            p = float(r.get("price"))
            if p > 0:
                prices.append(p)
        except Exception:
            pass
    prices.append(float(live_price))
    return prices


def main():
    ts = now_iso()
    state = read_json(STATE_PATH, {})
    report = read_json(REPORT_PATH, {})

    open_position = find_open_position(state, report)

    base = {
        "bot": BOT_NAME,
        "mode": MODE,
        "timestamp_utc": ts,
        "not_trading_advice": True,
        "real_orders": False,
        "paper_state_mutation": False,
    }

    if not open_position:
        out = {
            **base,
            "status": "PASS",
            "guard_action": "GUARD_NO_POSITION",
            "reason": "No open paper position found.",
        }
        write_json(GUARD_REPORT, out)
        append_jsonl(GUARD_LOG, out)
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    try:
        live_price = get_live_price()
        price_source = "bitget_public_live"
    except Exception as e:
        out = {
            **base,
            "status": "ERROR",
            "guard_action": "GUARD_ERROR_NO_LIVE_PRICE",
            "reason": str(e),
        }
        write_json(GUARD_REPORT, out)
        append_jsonl(GUARD_LOG, out)
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    prices = load_prices_with_live(live_price)

    if len(prices) < MIN_PRICE_POINTS:
        out = {
            **base,
            "status": "INSUFFICIENT_DATA",
            "guard_action": "GUARD_HOLD",
            "reason": f"Need at least {MIN_PRICE_POINTS} prices including live, found {len(prices)}.",
            "live_price": live_price,
            "price_source": price_source,
        }
        write_json(GUARD_REPORT, out)
        append_jsonl(GUARD_LOG, out)
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    ind = indicators(prices)
    scales = local_scales(prices)
    fc_market = cohesion_market(ind, len(prices), scales)
    pnl = net_position_pnl(open_position, live_price)
    fc_pos = cohesion_position(fc_market, open_position, pnl)

    # Formula-only guard action.
    if fc_pos["Fc"] < PI and fc_pos.get("hard_break", 0.0) >= 0.70:
        guard_action = "GUARD_EXIT_HARDBREAK_SUB_PI"
    elif fc_pos["Fc"] < PI and fc_pos.get("post_entry_break", 0.0) >= 0.52:
        guard_action = "GUARD_EXIT_POST_ENTRY_BREAK_SUB_PI"
    elif fc_pos["Fc"] < PI and fc_pos.get("decay_break", 0.0) >= 0.48:
        guard_action = "GUARD_EXIT_DECAYBREAK_SUB_PI"
    elif fc_pos["Fc"] >= TWO_PI:
        guard_action = "GUARD_EXIT_MATURE_2PI_PROTECT"
    else:
        guard_action = "GUARD_HOLD"

    out = {
        **base,
        "status": "PASS",
        "guard_action": guard_action,
        "live_price": live_price,
        "price_source": price_source,
        "open_position": open_position,
        "pnl": pnl,
        "indicators": ind,
        "Fc_market": fc_market,
        "Fc_position": fc_pos,
        "thresholds": {
            "pi": PI,
            "two_pi": TWO_PI,
            "hard_break_exit": 0.70,
            "decay_break_exit": 0.48,
            "post_entry_break_exit": 0.52,
        },
        "safety_note": "Report only. This script does not close paper positions and does not place real orders.",
    }

    write_json(GUARD_REPORT, out)
    append_jsonl(GUARD_LOG, out)
    print(json.dumps({
        "status": out["status"],
        "guard_action": out["guard_action"],
        "live_price": out["live_price"],
        "entry_price": open_position.get("entry_price"),
        "net_pnl_usdt": pnl["net_pnl_usdt"],
        "net_pnl_pct": pnl["net_pnl_pct"],
        "Fc_market": fc_market["Fc"],
        "phase_market": fc_market["phase"],
        "Fc_position": fc_pos["Fc"],
        "phase_position": fc_pos["phase"],
        "hard_break": fc_pos["hard_break"],
        "decay_break": fc_pos["decay_break"],
        "post_entry_break": fc_pos["post_entry_break"],
        "report": str(GUARD_REPORT),
        "log": str(GUARD_LOG),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
