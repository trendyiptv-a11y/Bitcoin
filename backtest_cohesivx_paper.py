#!/usr/bin/env python
from __future__ import annotations

import csv, json, math, statistics, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "btc-swing-strategy"
RESULT_JSON = OUT_DIR / "backtest_report.json"
RESULT_CSV = OUT_DIR / "backtest_equity_curve.csv"

STARTING_BALANCE_USDT = 1000.0
FEE_RATE = 0.001
MIN_TRADE_USDT = 5.0
SAMPLES = 250
BLOCKCHAIN_HASHRATE_URL = "https://api.blockchain.info/charts/hash-rate?timespan=all&format=json"
ELECTRICITY_USD_PER_KWH_BASE = 0.05
PRODUCTION_MARKUP = 1.25

MAX_ENTRY_FRACTION = 0.10
MAX_ENTRY_FRACTION_UNDERPRICED = 0.15
MAX_ENTRY_FRACTION_DEEP = 0.20
MAX_ENTRY_FRACTION_GUARD = 0.08
TARGET_UP_STEP = 0.10
TARGET_UP_STEP_DEEP = 0.14
TARGET_DOWN_STEP = 0.015
TARGET_DOWN_STEP_POSITIVE = 0.006
SELL_HYSTERESIS = 0.20
SEVERE_SELL_HYSTERESIS = 0.12
RISK_REDUCTION_COOLDOWN_DAYS = 21
TRIM_COOLDOWN_DAYS = 14

DRAWDOWN_GUARD = -0.60
DRAWDOWN_GUARD_TARGET_CAP = 0.72
DRAWDOWN_CAUTION = -0.54
ADAPTIVE_BUY_LOCK_DD = -0.58
GUARD_RISK_REDUCTION_COOLDOWN_DAYS = 9
NEGATIVE_MARKET_RISK_COOLDOWN_DAYS = 10
PERIODS = [
    ("2011_2014", "2011-01-01", "2014-12-31"),
    ("2015_2018", "2015-01-01", "2018-12-31"),
    ("2019_2022", "2019-01-01", "2022-12-31"),
    ("2023_2026", "2023-01-01", "2026-12-31"),
]


def f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def dt_from_ms(ms: float) -> datetime:
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)


def block_subsidy_btc(date: datetime) -> float:
    d = date.replace(tzinfo=None)
    if d < datetime(2012, 11, 28): return 50.0
    if d < datetime(2016, 7, 9): return 25.0
    if d < datetime(2020, 5, 11): return 12.5
    if d < datetime(2024, 4, 20): return 6.25
    return 3.125


def efficiency_j_per_th(date: datetime) -> float:
    d = date.replace(tzinfo=None)
    if d < datetime(2013, 1, 1): return 5_000_000.0
    if d < datetime(2016, 1, 1): return 600.0
    if d < datetime(2018, 1, 1): return 120.0
    if d < datetime(2020, 1, 1): return 80.0
    if d < datetime(2023, 1, 1): return 45.0
    if d < datetime(2024, 6, 1): return 33.0
    if d < datetime(2025, 1, 1): return 28.2
    return 26.0


def estimate_cost_usd_per_btc_from_difficulty(difficulty: float, when: datetime) -> float:
    if not (math.isfinite(difficulty) and difficulty > 0): return float("nan")
    subsidy = block_subsidy_btc(when)
    if subsidy <= 0: return float("nan")
    hashes_per_block = difficulty * (2.0 ** 32)
    joules_per_hash = efficiency_j_per_th(when) / 1e12
    energy_kwh = hashes_per_block * joules_per_hash / 3_600_000.0
    return float((energy_kwh * ELECTRICITY_USD_PER_KWH_BASE / subsidy) * PRODUCTION_MARKUP)


def fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "CohesivX-Backtest/0.6.3"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_hashrate_history() -> list[dict[str, Any]]:
    raw = fetch_json(BLOCKCHAIN_HASHRATE_URL)
    out: list[dict[str, Any]] = []
    for r in raw.get("values", []):
        date = datetime.fromtimestamp(float(r.get("x")), tz=timezone.utc).date().isoformat()
        hashrate_ths = f(r.get("y"))
        if hashrate_ths <= 0: continue
        difficulty = hashrate_ths * 1e12 * 600.0 / (2.0 ** 32)
        out.append({"date": date, "hashrate_ths": hashrate_ths, "difficulty_implied": difficulty})
    out.sort(key=lambda x: x["date"])
    return out


def find_ic_file() -> Path:
    for p in [ROOT / "ic_btc_series.json", ROOT / "j-btc-coeziv" / "ic_btc_series.json", ROOT / "data" / "ic_btc_series.json", ROOT / "btc-swing-strategy" / "ic_btc_series.json"]:
        if p.exists() and p.is_file(): return p
    raise FileNotFoundError("Backtest requires ic_btc_series.json in root, data, j-btc-coeziv or btc-swing-strategy.")


def load_ic_series() -> list[dict[str, Any]]:
    raw = json.loads(find_ic_file().read_text(encoding="utf-8"))
    rows = raw.get("series", []) if isinstance(raw, dict) else raw
    clean: list[dict[str, Any]] = []
    for r in rows:
        close, t = f(r.get("close")), r.get("t")
        if close <= 0 or t is None: continue
        dt = dt_from_ms(float(t))
        row = dict(r)
        row.update({"date": dt.date().isoformat(), "dt": dt, "month": dt.date().isoformat()[:7], "close": close})
        for c in ["ic_struct", "ic_dir", "ic_flux", "ic_cycle", "vol30_index"]: row[c] = f(row.get(c), 50.0)
        row["regime"] = str(row.get("regime") or "unknown").lower()
        clean.append(row)
    clean.sort(key=lambda x: x["date"])
    return clean


def attach_production_cost(rows: list[dict[str, Any]], hashrate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not hashrate: raise ValueError("Hashrate history is empty; cannot run v0.6.3 backtest.")
    h_idx, out = 0, []
    for row in rows:
        while h_idx + 1 < len(hashrate) and hashrate[h_idx + 1]["date"] <= row["date"]: h_idx += 1
        h = hashrate[h_idx]
        if h["date"] > row["date"]: continue
        cost = estimate_cost_usd_per_btc_from_difficulty(f(h.get("difficulty_implied")), row["dt"])
        if not (math.isfinite(cost) and cost > 0): continue
        x = dict(row)
        x["difficulty_implied"] = f(h.get("difficulty_implied"))
        x["hashrate_ths"] = f(h.get("hashrate_ths"))
        x["prod_cost_usd"] = cost
        x["historical_multiplier"] = x["close"] / cost
        out.append(x)
    return out


def context_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.sqrt(sum(((f(a.get(c)) - f(b.get(c))) / 100.0) ** 2 for c in ["ic_struct", "ic_dir", "ic_flux", "ic_cycle", "vol30_index"]))


def quantile(values: list[float], q: float) -> float:
    xs = sorted([x for x in values if math.isfinite(x)])
    if not xs: return float("nan")
    if len(xs) == 1: return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi: return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def memory_signal(history: list[dict[str, Any]], current: dict[str, Any]) -> dict[str, Any]:
    if len(history) < SAMPLES: return {"available": False, "reason": "not enough history"}
    price, current_cost = f(current.get("close")), f(current.get("prod_cost_usd"))
    if price <= 0 or current_cost <= 0: return {"available": False, "reason": "missing price or production cost"}
    nearest = sorted(history, key=lambda r: context_distance(r, current))[:SAMPLES]
    same_pool = [r for r in history if str(r.get("regime")) == str(current.get("regime"))]
    same = sorted(same_pool, key=lambda r: context_distance(r, current))[:SAMPLES] if same_pool else []
    similar_mult = [f(r.get("historical_multiplier")) for r in nearest if f(r.get("historical_multiplier")) > 0]
    same_mult = [f(r.get("historical_multiplier")) for r in same if f(r.get("historical_multiplier")) > 0]
    if len(similar_mult) < 25: return {"available": False, "reason": "not enough valid multipliers"}
    sim_m10, sim_m50, sim_m90 = quantile(similar_mult, 0.10), quantile(similar_mult, 0.50), quantile(similar_mult, 0.90)
    same_m10 = quantile(same_mult, 0.10) if same_mult else sim_m10
    same_m50 = quantile(same_mult, 0.50) if same_mult else sim_m50
    same_m90 = quantile(same_mult, 0.90) if same_mult else sim_m90
    wm10, wm50, wm90 = 0.60 * sim_m10 + 0.40 * same_m10, 0.60 * sim_m50 + 0.40 * same_m50, 0.60 * sim_m90 + 0.40 * same_m90
    wp10, wp50, wp90 = current_cost * wm10, current_cost * wm50, current_cost * wm90
    expected = (wp50 - price) / price
    downside = abs(min(0.0, (wp10 - price) / price))
    upside = max(0.0, (wp90 - price) / price)
    underpriced = max(0.0, (wp50 - price) / wp50) if wp50 > 0 else 0.0
    over_p50 = max(0.0, (price - wp50) / wp50) if wp50 > 0 else 0.0
    distance_median = statistics.median([context_distance(r, current) for r in nearest])
    sample_conf = clamp(len(nearest) / SAMPLES, 0.0, 1.0)
    same_conf = clamp(len(same) / max(len(nearest), 1), 0.0, 1.0)
    distance_conf = clamp(1.0 - max(distance_median, 0.0) / 0.80, 0.0, 1.0)
    confidence = clamp(0.35 * sample_conf + 0.25 * same_conf + 0.25 * distance_conf + 0.15 * clamp(f(current.get("ic_struct")) / 100.0, 0.0, 1.0), 0.0, 1.0)
    regime = str(current.get("regime") or "")
    adjust = 0.06 if regime == "bear_late" else (-0.04 if regime.startswith("bear") else (0.04 if regime.startswith("bull") else (0.02 if regime in {"range", "neutral"} else 0.0)))
    if f(current.get("ic_struct")) >= 55 and f(current.get("ic_flux")) >= 50: adjust += 0.03
    edge = confidence * expected - (1.0 - confidence) * downside + adjust
    return {"available": True, "weighted_price_p10": wp10, "weighted_price_p50": wp50, "weighted_price_p90": wp90, "weighted_multiplier_p50": wm50, "expected_30d": expected, "downside_risk": downside, "upside_to_p90": upside, "underpriced_pct": underpriced, "over_p50_pct": over_p50, "confidence": confidence, "decision_edge": edge, "distance_median": distance_median, "samples": len(nearest), "same_regime_samples": len(same)}


def raw_target_exposure(regime: str, sig: dict[str, Any]) -> dict[str, Any]:
    r = (regime or "").lower()
    edge, conf, risk = f(sig.get("decision_edge")), f(sig.get("confidence")), f(sig.get("downside_risk"))
    underpriced, expected = f(sig.get("underpriced_pct")), f(sig.get("expected_30d"))
    if r == "bear_late": low, high, core, profile = 0.32, 0.62, 0.22, "bear_late_allocator_v063"
    elif r.startswith("bear"): low, high, core, profile = 0.18, 0.42, 0.10, "bear_allocator_defensive_v063"
    elif r.startswith("bull"): low, high, core, profile = 0.65, 0.94, 0.55, "bull_allocator_patient_v063"
    else: low, high, core, profile = 0.38, 0.68, 0.25, "range_allocator_v063"
    strength = 0.0
    if edge > 0: strength += clamp(edge / 0.25, 0.0, 1.0) * 0.32
    strength += clamp(conf, 0.0, 1.0) * 0.18 + clamp(underpriced / 0.35, 0.0, 1.0) * 0.38
    if expected > 0: strength += clamp(expected / 0.50, 0.0, 1.0) * 0.12
    strength *= clamp(1.0 - max(0.0, risk - 0.27), 0.35, 1.0)
    target = low + (high - low) * clamp(strength, 0.0, 1.0)
    underpriced_mode = underpriced >= 0.08 and edge > 0 and conf >= 0.42
    deep_underpriced_mode = underpriced >= 0.18 and edge > 0.05 and conf >= 0.45
    extreme_underpriced_mode = underpriced >= 0.28 and edge > 0.07 and conf >= 0.48 and risk < 0.42
    if underpriced_mode:
        target = max(target, min(high, low + (high - low) * 0.68)); core = max(core, min(target, low + (high - low) * 0.45)); profile += "_underpriced"
    if deep_underpriced_mode:
        target = max(target, high); core = max(core, min(target, low + (high - low) * 0.70)); profile += "_deep"
    if extreme_underpriced_mode:
        target = max(target, min(0.96, high + 0.03)); core = max(core, min(target, 0.72)); profile += "_extreme"
    no_sell_zone = underpriced_mode or (edge > 0.08 and conf >= 0.45 and f(sig.get("weighted_price_p50")) > 0)
    return {"profile": profile, "raw_target_exposure": clamp(target, 0.0, 0.96), "raw_core_exposure": clamp(core, 0.0, 0.92), "underpriced_mode": underpriced_mode, "deep_underpriced_mode": deep_underpriced_mode, "extreme_underpriced_mode": extreme_underpriced_mode, "no_sell_zone": no_sell_zone}


def smooth_target(previous: float | None, raw: float, positive_market: bool, deep_mode: bool) -> float:
    raw = clamp(raw, 0.0, 0.96)
    if previous is None: return raw
    if raw > previous: return min(raw, previous + (TARGET_UP_STEP_DEEP if deep_mode else TARGET_UP_STEP))
    return max(raw, previous - (TARGET_DOWN_STEP_POSITIVE if positive_market else TARGET_DOWN_STEP))


def account(cash: float, btc: float, cost_basis: float, realized: float, price: float) -> dict[str, float]:
    value = cash + btc * price
    exposure = btc * price / value if value > 0 else 0.0
    unrealized = btc * price - cost_basis if btc > 0 else 0.0
    unrealized_pct = unrealized / cost_basis if cost_basis > 0 else 0.0
    total_pnl = realized + unrealized
    return {"cash": cash, "btc": btc, "cost_basis": cost_basis, "realized": realized, "value": value, "exposure": exposure, "unrealized": unrealized, "unrealized_pct": unrealized_pct, "total_pnl": total_pnl, "total_pnl_pct": total_pnl / STARTING_BALANCE_USDT}


def execute_buy(cash: float, btc: float, cost_basis: float, price: float, gross_usdt: float) -> tuple[float, float, float, float, float]:
    gross_usdt = min(cash, gross_usdt)
    if gross_usdt < MIN_TRADE_USDT: return cash, btc, cost_basis, 0.0, 0.0
    bought_btc = gross_usdt * (1 - FEE_RATE) / price
    return cash - gross_usdt, btc + bought_btc, cost_basis + gross_usdt, gross_usdt, bought_btc


def execute_sell(cash: float, btc: float, cost_basis: float, realized: float, price: float, sell_btc: float) -> tuple[float, float, float, float, float, float]:
    sell_btc = min(btc, max(0.0, sell_btc))
    gross_usdt = sell_btc * price
    if gross_usdt < MIN_TRADE_USDT or sell_btc <= 0: return cash, btc, cost_basis, realized, 0.0, 0.0
    net_usdt = gross_usdt * (1 - FEE_RATE)
    basis_removed = cost_basis * (sell_btc / btc) if btc > 0 and cost_basis > 0 else 0.0
    return cash + net_usdt, btc - sell_btc, max(0.0, cost_basis - basis_removed), realized + net_usdt - basis_removed, gross_usdt, -sell_btc


def adaptive_buy_allowed(current_dd: float, edge: float, expected: float, underpriced: float, conf: float, risk: float, extreme: bool) -> tuple[bool, str, float]:
    if current_dd > ADAPTIVE_BUY_LOCK_DD: return True, "normal", 1.0
    if extreme: return True, "extreme_exception", 0.85
    if current_dd <= DRAWDOWN_GUARD and underpriced >= 0.18 and edge > 0.06 and expected > 0.05 and conf >= 0.48 and risk < 0.44: return True, "guard_confirmed", 0.60
    if current_dd <= ADAPTIVE_BUY_LOCK_DD and underpriced >= 0.12 and edge > 0.04 and expected > 0.02 and conf >= 0.45 and risk < 0.40: return True, "caution_confirmed", 0.48
    return False, "adaptive_lock", 0.0


def run_backtest(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cash = STARTING_BALANCE_USDT; btc = cost_basis = realized = 0.0
    trades = buys = sells = 0
    blocked_sells_by_cooldown = blocked_sells_by_hysteresis = 0
    adaptive_lock_days = adaptive_buy_days = drawdown_guard_days = 0
    curve: list[dict[str, Any]] = []
    peak, max_dd = STARTING_BALANCE_USDT, 0.0
    smoothed_target: float | None = None
    last_sell_idx = -10_000
    bh_btc = STARTING_BALANCE_USDT / f(rows[0].get("close")) if f(rows[0].get("close")) > 0 else 0.0
    for idx, row in enumerate(rows):
        price = f(row.get("close"))
        if price <= 0: continue
        before = account(cash, btc, cost_basis, realized, price)
        current_dd = (before["value"] - peak) / peak if peak > 0 else 0.0
        sig = memory_signal(rows[:idx], row)
        raw_alloc = raw_target_exposure(str(row.get("regime") or ""), sig) if sig.get("available") else {"profile": "observe", "raw_target_exposure": 0.0, "raw_core_exposure": 0.0, "no_sell_zone": False, "underpriced_mode": False, "deep_underpriced_mode": False, "extreme_underpriced_mode": False}
        edge = f(sig.get("decision_edge")) if sig.get("available") else 0.0
        expected = f(sig.get("expected_30d")) if sig.get("available") else 0.0
        risk = f(sig.get("downside_risk")) if sig.get("available") else 0.0
        underpriced = f(sig.get("underpriced_pct")) if sig.get("available") else 0.0
        conf = f(sig.get("confidence")) if sig.get("available") else 0.0
        regime = str(row.get("regime") or "")
        positive_market = regime.startswith("bull") or (edge > 0.06 and expected > 0)
        negative_market = regime.startswith("bear") or (edge < -0.04 and expected < 0) or risk > 0.42
        deep_mode = bool(raw_alloc.get("deep_underpriced_mode", False)) or bool(raw_alloc.get("extreme_underpriced_mode", False))
        extreme_mode = bool(raw_alloc.get("extreme_underpriced_mode", False))
        if sig.get("available"):
            smoothed_target = smooth_target(smoothed_target, float(raw_alloc["raw_target_exposure"]), positive_market, deep_mode)
        target = float(smoothed_target or 0.0)
        drawdown_guard_active = current_dd <= DRAWDOWN_GUARD
        caution_active = current_dd <= DRAWDOWN_CAUTION
        if drawdown_guard_active:
            drawdown_guard_days += 1
            target = min(target, DRAWDOWN_GUARD_TARGET_CAP)
        core = min(float(raw_alloc.get("raw_core_exposure", 0.0)), target)
        action = "OBSERVE"; executed_usdt = executed_btc = 0.0
        buy_allowed, buy_gate, buy_scale = adaptive_buy_allowed(current_dd, edge, expected, underpriced, conf, risk, extreme_mode)
        if buy_allowed and buy_gate != "normal": adaptive_buy_days += 1
        if not buy_allowed: adaptive_lock_days += 1
        if sig.get("available"):
            gap = target - before["exposure"]
            core_btc = before["value"] * core / price if price > 0 else 0.0
            sellable_btc = max(0.0, btc - core_btc)
            days_since_sell = idx - last_sell_idx
            required_reduce_cooldown = GUARD_RISK_REDUCTION_COOLDOWN_DAYS if drawdown_guard_active else (NEGATIVE_MARKET_RISK_COOLDOWN_DAYS if negative_market else RISK_REDUCTION_COOLDOWN_DAYS)
            allow_reduce = days_since_sell >= required_reduce_cooldown
            allow_trim = days_since_sell >= TRIM_COOLDOWN_DAYS
            if gap > 0.025 and edge > 0 and expected > -0.05 and buy_allowed:
                if extreme_mode: action, buy_fraction = "ALLOCATE_EXTREME_UNDERPRICED", min(MAX_ENTRY_FRACTION_DEEP, gap) * buy_scale
                elif bool(raw_alloc.get("deep_underpriced_mode", False)): action, buy_fraction = "ALLOCATE_DEEP_UNDERPRICED", min(MAX_ENTRY_FRACTION_DEEP, gap) * buy_scale
                elif bool(raw_alloc.get("underpriced_mode", False)): action, buy_fraction = "ALLOCATE_UNDERPRICED", min(MAX_ENTRY_FRACTION_UNDERPRICED, gap) * buy_scale
                else: action, buy_fraction = "ALLOCATE_TO_SMOOTHED_TARGET", min(MAX_ENTRY_FRACTION, gap) * buy_scale
                if drawdown_guard_active: buy_fraction = min(buy_fraction, MAX_ENTRY_FRACTION_GUARD)
                cash, btc, cost_basis, executed_usdt, executed_btc = execute_buy(cash, btc, cost_basis, price, before["value"] * buy_fraction)
            elif gap > 0.025 and not buy_allowed:
                action = "HOLD_ADAPTIVE_BUY_LOCK"
            elif btc > 0 and not bool(raw_alloc.get("no_sell_zone", False)) and sellable_btc > 0:
                excess = before["exposure"] - target
                severe = edge < -0.10 or expected < -0.12 or risk > 0.48 or drawdown_guard_active
                if (edge < -0.06 or expected < -0.08 or drawdown_guard_active) and excess > SEVERE_SELL_HYSTERESIS:
                    if allow_reduce or drawdown_guard_active:
                        action = "REDUCE_RISK_V063_GUARD" if drawdown_guard_active else "REDUCE_RISK_HYSTERESIS_V063"
                        fraction = 0.12 if drawdown_guard_active else (0.10 if positive_market else (0.22 if severe else 0.16))
                        cash, btc, cost_basis, realized, executed_usdt, executed_btc = execute_sell(cash, btc, cost_basis, realized, price, min(sellable_btc, btc * fraction))
                        last_sell_idx = idx if executed_btc < 0 else last_sell_idx
                    else:
                        action = "HOLD_SELL_COOLDOWN"; blocked_sells_by_cooldown += 1
                elif excess > SELL_HYSTERESIS and risk > 0.32:
                    if allow_trim:
                        action = "TRIM_ABOVE_SMOOTHED_TARGET_V063"
                        fraction = 0.06 if positive_market else 0.10
                        cash, btc, cost_basis, realized, executed_usdt, executed_btc = execute_sell(cash, btc, cost_basis, realized, price, min(sellable_btc, btc * fraction))
                        last_sell_idx = idx if executed_btc < 0 else last_sell_idx
                    else:
                        action = "HOLD_TRIM_COOLDOWN"; blocked_sells_by_cooldown += 1
                elif before["unrealized_pct"] > 0.55 and f(sig.get("over_p50_pct")) > 0.34 and edge < 0.03 and excess > SELL_HYSTERESIS:
                    if allow_trim:
                        action = "TAKE_PROFIT_OVERVALUED_V063"
                        cash, btc, cost_basis, realized, executed_usdt, executed_btc = execute_sell(cash, btc, cost_basis, realized, price, min(sellable_btc, btc * 0.06))
                        last_sell_idx = idx if executed_btc < 0 else last_sell_idx
                    else:
                        action = "HOLD_PROFIT_COOLDOWN"; blocked_sells_by_cooldown += 1
                elif excess > 0.0:
                    blocked_sells_by_hysteresis += 1
            if executed_usdt >= MIN_TRADE_USDT:
                trades += 1; buys += 1 if executed_btc > 0 else 0; sells += 1 if executed_btc < 0 else 0
        after = account(cash, btc, cost_basis, realized, price)
        peak = max(peak, after["value"])
        dd = (after["value"] - peak) / peak if peak > 0 else 0.0
        max_dd = min(max_dd, dd)
        curve.append({"date": row["date"], "close": round(price, 8), "prod_cost_usd": round(f(row.get("prod_cost_usd")), 8), "historical_multiplier": round(f(row.get("historical_multiplier")), 8), "action": action, "executed_usdt": round(executed_usdt, 8), "executed_btc": round(executed_btc, 12), "cash_usdt": round(cash, 8), "btc_amount": round(btc, 12), "portfolio_value_usdt": round(after["value"], 8), "btc_exposure_pct": round(after["exposure"] * 100, 4), "raw_target_exposure_pct": round(float(raw_alloc.get("raw_target_exposure", 0.0)) * 100, 4), "smoothed_target_exposure_pct": round(target * 100, 4), "core_exposure_pct": round(core * 100, 4), "total_pnl_usdt": round(after["total_pnl"], 8), "total_pnl_pct": round(after["total_pnl_pct"] * 100, 4), "drawdown_pct": round(dd * 100, 4), "drawdown_guard_active": drawdown_guard_active, "caution_active": caution_active, "adaptive_buy_gate": buy_gate, "regime": row.get("regime", ""), "rule_profile": raw_alloc.get("profile", ""), "underpriced_mode": bool(raw_alloc.get("underpriced_mode", False)), "deep_underpriced_mode": bool(raw_alloc.get("deep_underpriced_mode", False)), "extreme_underpriced_mode": extreme_mode, "no_sell_zone": bool(raw_alloc.get("no_sell_zone", False)), "weighted_price_p50": round(f(sig.get("weighted_price_p50")), 8) if sig.get("available") else "", "underpriced_pct": round(underpriced * 100, 4) if sig.get("available") else "", "decision_edge_pct": round(edge * 100, 4) if sig.get("available") else "", "memory_confidence_pct": round(conf * 100, 4) if sig.get("available") else ""})
    last_price = f(rows[-1].get("close"))
    final = account(cash, btc, cost_basis, realized, last_price)
    bh_value = bh_btc * last_price; bh_pnl = bh_value - STARTING_BALANCE_USDT
    profitable_days = sum(1 for x in curve if f(x.get("total_pnl_usdt")) > 0)
    summary = {"final_portfolio_value_usdt": round(final["value"], 8), "total_pnl_usdt": round(final["total_pnl"], 8), "total_return_pct": round(final["total_pnl_pct"] * 100, 4), "max_drawdown_pct": round(max_dd * 100, 4), "trades": trades, "buys": buys, "sells": sells, "blocked_sells_by_cooldown": blocked_sells_by_cooldown, "blocked_sells_by_hysteresis": blocked_sells_by_hysteresis, "drawdown_guard_days": drawdown_guard_days, "adaptive_lock_days": adaptive_lock_days, "adaptive_buy_days": adaptive_buy_days, "final_cash_usdt": round(cash, 8), "final_btc_amount": round(btc, 12), "final_btc_exposure_pct": round(final["exposure"] * 100, 4), "profitable_days_ratio_pct": round(profitable_days / max(len(curve), 1) * 100, 4)}
    buy_hold = {"final_value_usdt": round(bh_value, 8), "total_pnl_usdt": round(bh_pnl, 8), "total_return_pct": round(bh_pnl / STARTING_BALANCE_USDT * 100, 4), "btc_amount": round(bh_btc, 12)}
    return {"strategy": summary, "buy_and_hold": buy_hold}, curve


def max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 0.0; worst = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0: worst = min(worst, (v - peak) / peak)
    return worst


def benchmark_buy_hold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = f(rows[0]["close"]); btc = STARTING_BALANCE_USDT / first if first > 0 else 0.0
    values = [btc * f(r["close"]) for r in rows]; final = values[-1]
    return {"final_value_usdt": round(final, 8), "total_return_pct": round((final / STARTING_BALANCE_USDT - 1) * 100, 4), "max_drawdown_pct": round(max_drawdown(values) * 100, 4), "btc_amount": round(btc, 12)}


def benchmark_dca_monthly(rows: list[dict[str, Any]]) -> dict[str, Any]:
    months = sorted({r["month"] for r in rows}); contribution = STARTING_BALANCE_USDT / max(len(months), 1)
    cash, btc, seen, values, buys = STARTING_BALANCE_USDT, 0.0, set(), [], 0
    for r in rows:
        price = f(r["close"])
        if r["month"] not in seen and cash >= contribution and price > 0:
            seen.add(r["month"]); gross = min(cash, contribution); btc += gross * (1 - FEE_RATE) / price; cash -= gross; buys += 1
        values.append(cash + btc * price)
    final = values[-1]
    return {"final_value_usdt": round(final, 8), "total_return_pct": round((final / STARTING_BALANCE_USDT - 1) * 100, 4), "max_drawdown_pct": round(max_drawdown(values) * 100, 4), "buys": buys, "btc_amount": round(btc, 12)}


def benchmark_rebalanced(rows: list[dict[str, Any]], target_btc_exposure: float) -> dict[str, Any]:
    cash = STARTING_BALANCE_USDT * (1 - target_btc_exposure); btc = STARTING_BALANCE_USDT * target_btc_exposure * (1 - FEE_RATE) / f(rows[0]["close"])
    values, last_month, trades = [], rows[0]["month"], 1
    for r in rows:
        price = f(r["close"]); value = cash + btc * price
        if r["month"] != last_month and value > 0 and price > 0:
            last_month = r["month"]; delta = value * target_btc_exposure - btc * price
            if abs(delta) >= MIN_TRADE_USDT:
                if delta > 0:
                    gross = min(cash, delta); btc += gross * (1 - FEE_RATE) / price; cash -= gross
                else:
                    sell_btc = min(btc, abs(delta) / price); cash += sell_btc * price * (1 - FEE_RATE); btc -= sell_btc
                trades += 1
            value = cash + btc * price
        values.append(value)
    final = values[-1]
    return {"final_value_usdt": round(final, 8), "total_return_pct": round((final / STARTING_BALANCE_USDT - 1) * 100, 4), "max_drawdown_pct": round(max_drawdown(values) * 100, 4), "trades": trades, "target_btc_exposure_pct": round(target_btc_exposure * 100, 2)}


def summarize_curve_period(curve: list[dict[str, Any]], label: str, start: str, end: str) -> dict[str, Any]:
    part = [r for r in curve if start <= str(r.get("date")) <= end]
    if len(part) < 2: return {"label": label, "start": start, "end": end, "available": False}
    start_v, end_v = f(part[0]["portfolio_value_usdt"]), f(part[-1]["portfolio_value_usdt"])
    values = [f(r["portfolio_value_usdt"]) for r in part]; actions = [str(r.get("action")) for r in part]
    return {"label": label, "start": part[0]["date"], "end": part[-1]["date"], "available": True, "strategy_start_value_usdt": round(start_v, 8), "strategy_end_value_usdt": round(end_v, 8), "strategy_return_pct": round((end_v / start_v - 1) * 100, 4) if start_v > 0 else None, "strategy_max_drawdown_pct": round(max_drawdown(values) * 100, 4), "accumulate_days": sum(1 for a in actions if a.startswith("ALLOCATE")), "sell_days": sum(1 for a in actions if a in {"REDUCE_RISK_V063_GUARD", "REDUCE_RISK_HYSTERESIS_V063", "TRIM_ABOVE_SMOOTHED_TARGET_V063", "TAKE_PROFIT_OVERVALUED_V063"}), "cooldown_hold_days": sum(1 for a in actions if "COOLDOWN" in a), "drawdown_guard_days": sum(1 for r in part if bool(r.get("drawdown_guard_active"))), "adaptive_lock_days": sum(1 for r in part if str(r.get("adaptive_buy_gate")) == "adaptive_lock"), "adaptive_buy_days": sum(1 for r in part if str(r.get("adaptive_buy_gate")) in {"guard_confirmed", "caution_confirmed", "extreme_exception"})}


def benchmark_period(rows: list[dict[str, Any]], label: str, start: str, end: str) -> dict[str, Any]:
    part = [r for r in rows if start <= r["date"] <= end]
    if len(part) < 2: return {"label": label, "available": False}
    return {"label": label, "available": True, "buy_and_hold": benchmark_buy_hold(part), "dca_monthly": benchmark_dca_monthly(part), "rebalance_40_60": benchmark_rebalanced(part, 0.40), "rebalance_60_40": benchmark_rebalanced(part, 0.60)}


def write_outputs(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = list(rows[0].keys()) if rows else []
    with RESULT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    ic_rows = load_ic_series(); hashrate = fetch_hashrate_history(); rows = attach_production_cost(ic_rows, hashrate)
    if len(rows) < 400: raise ValueError(f"Not enough aligned historical rows for v0.6.3 backtest: {len(rows)}. Need at least 400.")
    strategy_result, curve = run_backtest(rows)
    benchmarks = {"buy_and_hold": benchmark_buy_hold(rows), "dca_monthly": benchmark_dca_monthly(rows), "rebalance_40_60_monthly": benchmark_rebalanced(rows, 0.40), "rebalance_60_40_monthly": benchmark_rebalanced(rows, 0.60)}
    period_breakdown = [{"strategy": summarize_curve_period(curve, label, start, end), "benchmarks": benchmark_period(rows, label, start, end)} for label, start, end in PERIODS]
    strategy, bh = strategy_result["strategy"], strategy_result["buy_and_hold"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "cohesivx_backtest_v0.6.3_guarded_accumulation",
        "disclaimer": "Educational paper backtest only. Not financial advice. Uses historical daily close and simplified execution assumptions.",
        "assumptions": {"starting_balance_usdt": STARTING_BALANCE_USDT, "fee_rate": FEE_RATE, "min_trade_usdt": MIN_TRADE_USDT, "samples": SAMPLES, "uses_daily_close": True, "uses_slippage": False, "uses_full_fair_price_v2_production_cost": True, "uses_core_holding": True, "uses_structural_allocator": True, "uses_underpriced_mode": True, "uses_no_sell_zone_below_p50": True, "uses_target_smoothing": True, "uses_sell_hysteresis": True, "uses_risk_reduction_cooldown": True, "uses_selective_deep_underpriced_boost": True, "uses_positive_market_slow_target_down": True, "uses_adaptive_drawdown_guard": True, "uses_conditional_buy_gate": True, "uses_guarded_accumulation_tuning": True, "drawdown_guard": DRAWDOWN_GUARD, "drawdown_guard_target_cap": DRAWDOWN_GUARD_TARGET_CAP, "drawdown_caution": DRAWDOWN_CAUTION, "adaptive_buy_lock_dd": ADAPTIVE_BUY_LOCK_DD, "max_entry_fraction_guard": MAX_ENTRY_FRACTION_GUARD, "guard_confirmed_buy_scale": 0.60, "caution_confirmed_buy_scale": 0.48, "guard_risk_reduction_cooldown_days": GUARD_RISK_REDUCTION_COOLDOWN_DAYS, "negative_market_risk_cooldown_days": NEGATIVE_MARKET_RISK_COOLDOWN_DAYS, "benchmarks": ["buy_and_hold", "dca_monthly", "rebalance_40_60_monthly", "rebalance_60_40_monthly"], "hashrate_source": BLOCKCHAIN_HASHRATE_URL, "electricity_usd_per_kwh": ELECTRICITY_USD_PER_KWH_BASE, "production_markup": PRODUCTION_MARKUP, "note": "v0.6.3 stable candidate: adaptive guard with guarded accumulation tuning."},
        "period": {"start": rows[0]["date"], "end": rows[-1]["date"], "days": len(rows)}, "strategy": strategy, "buy_and_hold": bh, "benchmarks": benchmarks,
        "comparison": {"strategy_minus_buy_hold_usdt": round(strategy["final_portfolio_value_usdt"] - bh["final_value_usdt"], 8), "strategy_minus_buy_hold_pct_points": round(strategy["total_return_pct"] - bh["total_return_pct"], 4), "strategy_minus_dca_usdt": round(strategy["final_portfolio_value_usdt"] - benchmarks["dca_monthly"]["final_value_usdt"], 8), "strategy_minus_rebalance_40_60_usdt": round(strategy["final_portfolio_value_usdt"] - benchmarks["rebalance_40_60_monthly"]["final_value_usdt"], 8), "strategy_minus_rebalance_60_40_usdt": round(strategy["final_portfolio_value_usdt"] - benchmarks["rebalance_60_40_monthly"]["final_value_usdt"], 8)}, "period_breakdown": period_breakdown}
    write_outputs(summary, curve)
    print("Backtest completed"); print("Version:", summary["version"]); print("Period:", summary["period"]); print("Strategy:", summary["strategy"]); print("Benchmarks:", summary["benchmarks"]); print("Comparison:", summary["comparison"])


if __name__ == "__main__":
    main()
