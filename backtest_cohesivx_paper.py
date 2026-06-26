#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
import statistics
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "btc-swing-strategy"
RESULT_JSON = OUT_DIR / "backtest_report.json"
RESULT_CSV = OUT_DIR / "backtest_equity_curve.csv"

STARTING_BALANCE_USDT = 1000.0
MAX_ENTRY_FRACTION = 0.10
MIN_TRADE_USDT = 5.0
FEE_RATE = 0.001
SAMPLES = 250
BLOCKCHAIN_HASHRATE_URL = "https://api.blockchain.info/charts/hash-rate?timespan=all&format=json"
ELECTRICITY_USD_PER_KWH_BASE = 0.05
PRODUCTION_MARKUP = 1.25
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
    if d < datetime(2012, 11, 28):
        return 50.0
    if d < datetime(2016, 7, 9):
        return 25.0
    if d < datetime(2020, 5, 11):
        return 12.5
    if d < datetime(2024, 4, 20):
        return 6.25
    return 3.125


def efficiency_j_per_th(date: datetime) -> float:
    d = date.replace(tzinfo=None)
    if d < datetime(2013, 1, 1):
        return 5_000_000.0
    if d < datetime(2016, 1, 1):
        return 600.0
    if d < datetime(2018, 1, 1):
        return 120.0
    if d < datetime(2020, 1, 1):
        return 80.0
    if d < datetime(2023, 1, 1):
        return 45.0
    if d < datetime(2024, 6, 1):
        return 33.0
    if d < datetime(2025, 1, 1):
        return 28.2
    return 26.0


def estimate_cost_usd_per_btc_from_difficulty(difficulty: float, when: datetime) -> float:
    if not (math.isfinite(difficulty) and difficulty > 0.0):
        return float("nan")
    btc_per_block = block_subsidy_btc(when)
    if btc_per_block <= 0:
        return float("nan")
    hashes_per_block = difficulty * (2.0 ** 32)
    joules_per_hash = efficiency_j_per_th(when) / 1e12
    energy_kwh = (hashes_per_block * joules_per_hash) / 3_600_000.0
    cost_block_electric = energy_kwh * ELECTRICITY_USD_PER_KWH_BASE
    return float((cost_block_electric / btc_per_block) * PRODUCTION_MARKUP)


def fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "CohesivX-Backtest/0.4"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_hashrate_history() -> list[dict[str, Any]]:
    raw = fetch_json(BLOCKCHAIN_HASHRATE_URL)
    out: list[dict[str, Any]] = []
    for r in raw.get("values", []):
        date = datetime.fromtimestamp(float(r.get("x")), tz=timezone.utc).date().isoformat()
        hashrate_ths = f(r.get("y"))
        if hashrate_ths <= 0:
            continue
        difficulty = hashrate_ths * 1e12 * 600.0 / (2.0 ** 32)
        out.append({"date": date, "hashrate_ths": hashrate_ths, "difficulty_implied": difficulty})
    out.sort(key=lambda x: x["date"])
    return out


def find_ic_file() -> Path:
    candidates = [
        ROOT / "ic_btc_series.json",
        ROOT / "j-btc-coeziv" / "ic_btc_series.json",
        ROOT / "data" / "ic_btc_series.json",
        ROOT / "btc-swing-strategy" / "ic_btc_series.json",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError("Backtest requires ic_btc_series.json in root, data, j-btc-coeziv or btc-swing-strategy.")


def load_ic_series() -> list[dict[str, Any]]:
    path = find_ic_file()
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("series", []) if isinstance(raw, dict) else raw
    clean: list[dict[str, Any]] = []
    for r in rows:
        close = f(r.get("close"))
        t = r.get("t")
        if close <= 0 or t is None:
            continue
        dt = dt_from_ms(float(t))
        row = dict(r)
        row["date"] = dt.date().isoformat()
        row["dt"] = dt
        row["month"] = row["date"][:7]
        row["close"] = close
        row["ic_struct"] = f(row.get("ic_struct"), 50.0)
        row["ic_dir"] = f(row.get("ic_dir"), 50.0)
        row["ic_flux"] = f(row.get("ic_flux"), 50.0)
        row["ic_cycle"] = f(row.get("ic_cycle"), 50.0)
        row["vol30_index"] = f(row.get("vol30_index"), 50.0)
        row["regime"] = str(row.get("regime") or "unknown").lower()
        clean.append(row)
    clean.sort(key=lambda x: x["date"])
    return clean


def attach_production_cost(rows: list[dict[str, Any]], hashrate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not hashrate:
        raise ValueError("Hashrate history is empty; cannot run V4 backtest.")
    h_idx = 0
    out: list[dict[str, Any]] = []
    for row in rows:
        while h_idx + 1 < len(hashrate) and hashrate[h_idx + 1]["date"] <= row["date"]:
            h_idx += 1
        h = hashrate[h_idx]
        if h["date"] > row["date"]:
            continue
        cost = estimate_cost_usd_per_btc_from_difficulty(f(h.get("difficulty_implied")), row["dt"])
        if not (math.isfinite(cost) and cost > 0):
            continue
        x = dict(row)
        x["difficulty_implied"] = f(h.get("difficulty_implied"))
        x["hashrate_ths"] = f(h.get("hashrate_ths"))
        x["prod_cost_usd"] = cost
        x["historical_multiplier"] = x["close"] / cost
        out.append(x)
    return out


def context_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    cols = ["ic_struct", "ic_dir", "ic_flux", "ic_cycle", "vol30_index"]
    return math.sqrt(sum(((f(a.get(c)) - f(b.get(c))) / 100.0) ** 2 for c in cols))


def quantile(values: list[float], q: float) -> float:
    xs = sorted([x for x in values if math.isfinite(x)])
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def memory_signal(history: list[dict[str, Any]], current: dict[str, Any]) -> dict[str, Any]:
    if len(history) < SAMPLES:
        return {"available": False, "reason": "not enough history"}
    price = f(current.get("close"))
    current_cost = f(current.get("prod_cost_usd"))
    if price <= 0 or current_cost <= 0:
        return {"available": False, "reason": "missing price or production cost"}

    nearest = sorted(history, key=lambda r: context_distance(r, current))[:SAMPLES]
    same_regime_pool = [r for r in history if str(r.get("regime")) == str(current.get("regime"))]
    same = sorted(same_regime_pool, key=lambda r: context_distance(r, current))[:SAMPLES] if same_regime_pool else []
    similar_mult = [f(r.get("historical_multiplier")) for r in nearest if f(r.get("historical_multiplier")) > 0]
    same_mult = [f(r.get("historical_multiplier")) for r in same if f(r.get("historical_multiplier")) > 0]
    if len(similar_mult) < 25:
        return {"available": False, "reason": "not enough valid multipliers"}

    sim_m10, sim_m50, sim_m90 = quantile(similar_mult, 0.10), quantile(similar_mult, 0.50), quantile(similar_mult, 0.90)
    same_m10 = quantile(same_mult, 0.10) if same_mult else sim_m10
    same_m50 = quantile(same_mult, 0.50) if same_mult else sim_m50
    same_m90 = quantile(same_mult, 0.90) if same_mult else sim_m90
    weighted_m10 = 0.60 * sim_m10 + 0.40 * same_m10
    weighted_m50 = 0.60 * sim_m50 + 0.40 * same_m50
    weighted_m90 = 0.60 * sim_m90 + 0.40 * same_m90
    weighted_p10 = current_cost * weighted_m10
    weighted_p50 = current_cost * weighted_m50
    weighted_p90 = current_cost * weighted_m90

    expected_30d = (weighted_p50 - price) / price
    downside = abs(min(0.0, (weighted_p10 - price) / price))
    upside = max(0.0, (weighted_p90 - price) / price)
    underpriced_pct = max(0.0, (weighted_p50 - price) / weighted_p50) if weighted_p50 > 0 else 0.0
    over_p50_pct = max(0.0, (price - weighted_p50) / weighted_p50) if weighted_p50 > 0 else 0.0
    distance_median = statistics.median([context_distance(r, current) for r in nearest])
    sample_conf = clamp(len(nearest) / SAMPLES, 0.0, 1.0)
    same_conf = clamp(len(same) / max(len(nearest), 1), 0.0, 1.0)
    distance_conf = clamp(1.0 - max(distance_median, 0.0) / 0.80, 0.0, 1.0)
    confidence = clamp(0.35 * sample_conf + 0.25 * same_conf + 0.25 * distance_conf + 0.15 * clamp(f(current.get("ic_struct")) / 100.0, 0.0, 1.0), 0.0, 1.0)

    regime = str(current.get("regime") or "")
    adjust = 0.0
    if regime == "bear_late":
        adjust += 0.06
    elif regime.startswith("bear"):
        adjust -= 0.04
    elif regime.startswith("bull"):
        adjust += 0.04
    elif regime in {"range", "neutral"}:
        adjust += 0.02
    if f(current.get("ic_struct")) >= 55 and f(current.get("ic_flux")) >= 50:
        adjust += 0.03
    edge = confidence * expected_30d - (1.0 - confidence) * downside + adjust
    return {
        "available": True,
        "weighted_price_p10": weighted_p10,
        "weighted_price_p50": weighted_p50,
        "weighted_price_p90": weighted_p90,
        "weighted_multiplier_p50": weighted_m50,
        "expected_30d": expected_30d,
        "downside_risk": downside,
        "upside_to_p90": upside,
        "underpriced_pct": underpriced_pct,
        "over_p50_pct": over_p50_pct,
        "confidence": confidence,
        "decision_edge": edge,
        "distance_median": distance_median,
        "samples": len(nearest),
        "same_regime_samples": len(same),
    }


def target_exposure(regime: str, sig: dict[str, Any]) -> dict[str, Any]:
    r = (regime or "").lower()
    edge = f(sig.get("decision_edge"))
    confidence = f(sig.get("confidence"))
    risk = f(sig.get("downside_risk"))
    underpriced = f(sig.get("underpriced_pct"))
    expected = f(sig.get("expected_30d"))

    if r == "bear_late":
        low, high, core = 0.30, 0.55, 0.20
        profile = "bear_late_allocator"
    elif r.startswith("bear"):
        low, high, core = 0.15, 0.35, 0.08
        profile = "bear_allocator_defensive"
    elif r.startswith("bull"):
        low, high, core = 0.60, 0.90, 0.50
        profile = "bull_allocator_patient"
    else:
        low, high, core = 0.35, 0.60, 0.20
        profile = "range_allocator"

    strength = 0.0
    if edge > 0:
        strength += clamp(edge / 0.25, 0.0, 1.0) * 0.35
    strength += clamp(confidence, 0.0, 1.0) * 0.20
    strength += clamp(underpriced / 0.35, 0.0, 1.0) * 0.35
    if expected > 0:
        strength += clamp(expected / 0.50, 0.0, 1.0) * 0.10
    strength *= clamp(1.0 - max(0.0, risk - 0.25), 0.35, 1.0)

    target = low + (high - low) * clamp(strength, 0.0, 1.0)
    underpriced_mode = underpriced >= 0.08 and edge > 0 and confidence >= 0.42
    deep_underpriced_mode = underpriced >= 0.18 and edge > 0.05 and confidence >= 0.45
    if underpriced_mode:
        target = max(target, min(high, low + (high - low) * 0.65))
        core = max(core, min(target, low + (high - low) * 0.40))
        profile += "_underpriced"
    if deep_underpriced_mode:
        target = max(target, high)
        core = max(core, min(target, low + (high - low) * 0.60))
        profile += "_deep"

    no_sell_zone = underpriced_mode or (edge > 0.08 and confidence >= 0.45 and f(sig.get("weighted_price_p50")) > 0)
    return {
        "profile": profile,
        "target_exposure": clamp(target, 0.0, 0.95),
        "core_exposure": clamp(core, 0.0, 0.90),
        "underpriced_mode": underpriced_mode,
        "deep_underpriced_mode": deep_underpriced_mode,
        "no_sell_zone": no_sell_zone,
    }


def account(cash: float, btc: float, cost_basis: float, realized: float, price: float) -> dict[str, float]:
    value = cash + btc * price
    exposure = (btc * price / value) if value > 0 else 0.0
    unrealized = btc * price - cost_basis if btc > 0 else 0.0
    unrealized_pct = unrealized / cost_basis if cost_basis > 0 else 0.0
    total_pnl = realized + unrealized
    return {"cash": cash, "btc": btc, "cost_basis": cost_basis, "realized": realized, "value": value, "exposure": exposure, "unrealized": unrealized, "unrealized_pct": unrealized_pct, "total_pnl": total_pnl, "total_pnl_pct": total_pnl / STARTING_BALANCE_USDT}


def execute_buy(cash: float, btc: float, cost_basis: float, price: float, gross_usdt: float) -> tuple[float, float, float, float, float]:
    gross_usdt = min(cash, gross_usdt)
    if gross_usdt < MIN_TRADE_USDT:
        return cash, btc, cost_basis, 0.0, 0.0
    fee = gross_usdt * FEE_RATE
    net_usdt = gross_usdt - fee
    bought_btc = net_usdt / price
    return cash - gross_usdt, btc + bought_btc, cost_basis + gross_usdt, gross_usdt, bought_btc


def execute_sell(cash: float, btc: float, cost_basis: float, realized: float, price: float, sell_btc: float) -> tuple[float, float, float, float, float, float]:
    sell_btc = min(btc, max(0.0, sell_btc))
    gross_usdt = sell_btc * price
    if gross_usdt < MIN_TRADE_USDT or sell_btc <= 0:
        return cash, btc, cost_basis, realized, 0.0, 0.0
    fee = gross_usdt * FEE_RATE
    net_usdt = gross_usdt - fee
    basis_removed = cost_basis * (sell_btc / btc) if btc > 0 and cost_basis > 0 else 0.0
    realized += net_usdt - basis_removed
    cost_basis = max(0.0, cost_basis - basis_removed)
    return cash + net_usdt, btc - sell_btc, cost_basis, realized, gross_usdt, -sell_btc


def run_backtest(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cash = STARTING_BALANCE_USDT
    btc = cost_basis = realized = 0.0
    trades = buys = sells = 0
    equity_curve: list[dict[str, Any]] = []
    peak = STARTING_BALANCE_USDT
    max_dd = 0.0
    first_price = f(rows[0].get("close"))
    bh_btc = STARTING_BALANCE_USDT / first_price if first_price > 0 else 0.0

    for idx, row in enumerate(rows):
        price = f(row.get("close"))
        if price <= 0:
            continue
        hist = rows[:idx]
        sig = memory_signal(hist, row)
        alloc = target_exposure(str(row.get("regime") or ""), sig) if sig.get("available") else {"profile": "observe", "target_exposure": 0.0, "core_exposure": 0.0, "no_sell_zone": False, "underpriced_mode": False, "deep_underpriced_mode": False}
        action = "OBSERVE"
        executed_usdt = executed_btc = 0.0
        before = account(cash, btc, cost_basis, realized, price)

        if sig.get("available"):
            edge = f(sig.get("decision_edge"))
            expected = f(sig.get("expected_30d"))
            risk = f(sig.get("downside_risk"))
            target = float(alloc["target_exposure"])
            core = float(alloc["core_exposure"])
            no_sell_zone = bool(alloc["no_sell_zone"])
            target_gap = target - before["exposure"]
            core_btc = before["value"] * core / price if price > 0 else 0.0
            sellable_btc = max(0.0, btc - core_btc)

            if target_gap > 0.025 and edge > 0 and expected > -0.05:
                if bool(alloc["deep_underpriced_mode"]):
                    action = "ALLOCATE_DEEP_UNDERPRICED"
                    buy_fraction = min(0.18, target_gap)
                elif bool(alloc["underpriced_mode"]):
                    action = "ALLOCATE_UNDERPRICED"
                    buy_fraction = min(0.14, target_gap)
                else:
                    action = "ALLOCATE_TO_TARGET"
                    buy_fraction = min(MAX_ENTRY_FRACTION, target_gap)
                cash, btc, cost_basis, executed_usdt, executed_btc = execute_buy(cash, btc, cost_basis, price, before["value"] * buy_fraction)
            elif btc > 0 and not no_sell_zone and sellable_btc > 0:
                if edge < -0.06 or expected < -0.08:
                    action = "REDUCE_RISK_ALLOCATOR"
                    cash, btc, cost_basis, realized, executed_usdt, executed_btc = execute_sell(cash, btc, cost_basis, realized, price, min(sellable_btc, btc * 0.25))
                elif before["exposure"] > target + 0.10 and risk > 0.30:
                    action = "TRIM_ABOVE_TARGET_RISK"
                    cash, btc, cost_basis, realized, executed_usdt, executed_btc = execute_sell(cash, btc, cost_basis, realized, price, min(sellable_btc, btc * 0.18))
                elif before["unrealized_pct"] > 0.30 and f(sig.get("over_p50_pct")) > 0.20 and edge < 0.08:
                    action = "TAKE_PROFIT_OVERVALUED_ONLY"
                    cash, btc, cost_basis, realized, executed_usdt, executed_btc = execute_sell(cash, btc, cost_basis, realized, price, min(sellable_btc, btc * 0.12))

            if executed_usdt >= MIN_TRADE_USDT:
                trades += 1
                buys += 1 if executed_btc > 0 else 0
                sells += 1 if executed_btc < 0 else 0

        after = account(cash, btc, cost_basis, realized, price)
        peak = max(peak, after["value"])
        dd = (after["value"] - peak) / peak if peak > 0 else 0.0
        max_dd = min(max_dd, dd)
        equity_curve.append({
            "date": row["date"],
            "close": round(price, 8),
            "prod_cost_usd": round(f(row.get("prod_cost_usd")), 8),
            "historical_multiplier": round(f(row.get("historical_multiplier")), 8),
            "action": action,
            "executed_usdt": round(executed_usdt, 8),
            "executed_btc": round(executed_btc, 12),
            "cash_usdt": round(cash, 8),
            "btc_amount": round(btc, 12),
            "portfolio_value_usdt": round(after["value"], 8),
            "btc_exposure_pct": round(after["exposure"] * 100, 4),
            "target_exposure_pct": round(float(alloc.get("target_exposure", 0.0)) * 100, 4),
            "core_exposure_pct": round(float(alloc.get("core_exposure", 0.0)) * 100, 4),
            "total_pnl_usdt": round(after["total_pnl"], 8),
            "total_pnl_pct": round(after["total_pnl_pct"] * 100, 4),
            "drawdown_pct": round(dd * 100, 4),
            "regime": row.get("regime", ""),
            "rule_profile": alloc.get("profile", ""),
            "underpriced_mode": bool(alloc.get("underpriced_mode", False)),
            "deep_underpriced_mode": bool(alloc.get("deep_underpriced_mode", False)),
            "no_sell_zone": bool(alloc.get("no_sell_zone", False)),
            "weighted_price_p50": round(f(sig.get("weighted_price_p50")), 8) if sig.get("available") else "",
            "underpriced_pct": round(f(sig.get("underpriced_pct")) * 100, 4) if sig.get("available") else "",
            "decision_edge_pct": round(f(sig.get("decision_edge")) * 100, 4) if sig.get("available") else "",
            "memory_confidence_pct": round(f(sig.get("confidence")) * 100, 4) if sig.get("available") else "",
        })

    last_price = f(rows[-1].get("close"))
    final = account(cash, btc, cost_basis, realized, last_price)
    bh_value = bh_btc * last_price
    bh_pnl = bh_value - STARTING_BALANCE_USDT
    profitable_days = sum(1 for x in equity_curve if f(x.get("total_pnl_usdt")) > 0)
    summary = {
        "final_portfolio_value_usdt": round(final["value"], 8),
        "total_pnl_usdt": round(final["total_pnl"], 8),
        "total_return_pct": round(final["total_pnl_pct"] * 100, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "trades": trades,
        "buys": buys,
        "sells": sells,
        "final_cash_usdt": round(cash, 8),
        "final_btc_amount": round(btc, 12),
        "final_btc_exposure_pct": round(final["exposure"] * 100, 4),
        "profitable_days_ratio_pct": round((profitable_days / max(len(equity_curve), 1)) * 100, 4),
    }
    buy_hold = {"final_value_usdt": round(bh_value, 8), "total_pnl_usdt": round(bh_pnl, 8), "total_return_pct": round((bh_pnl / STARTING_BALANCE_USDT) * 100, 4), "btc_amount": round(bh_btc, 12)}
    return {"strategy": summary, "buy_and_hold": buy_hold}, equity_curve


def max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 0.0
    worst = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = min(worst, (v - peak) / peak)
    return worst


def benchmark_buy_hold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = f(rows[0]["close"])
    btc = STARTING_BALANCE_USDT / first if first > 0 else 0.0
    values = [btc * f(r["close"]) for r in rows]
    final = values[-1]
    return {"final_value_usdt": round(final, 8), "total_return_pct": round((final / STARTING_BALANCE_USDT - 1) * 100, 4), "max_drawdown_pct": round(max_drawdown(values) * 100, 4), "btc_amount": round(btc, 12)}


def benchmark_dca_monthly(rows: list[dict[str, Any]]) -> dict[str, Any]:
    months = sorted({r["month"] for r in rows})
    contribution = STARTING_BALANCE_USDT / max(len(months), 1)
    cash = STARTING_BALANCE_USDT
    btc = 0.0
    seen: set[str] = set()
    values: list[float] = []
    buys = 0
    for r in rows:
        price = f(r["close"])
        if r["month"] not in seen and cash >= contribution and price > 0:
            seen.add(r["month"])
            gross = min(cash, contribution)
            btc += (gross * (1 - FEE_RATE)) / price
            cash -= gross
            buys += 1
        values.append(cash + btc * price)
    final = values[-1]
    return {"final_value_usdt": round(final, 8), "total_return_pct": round((final / STARTING_BALANCE_USDT - 1) * 100, 4), "max_drawdown_pct": round(max_drawdown(values) * 100, 4), "buys": buys, "btc_amount": round(btc, 12)}


def benchmark_rebalanced(rows: list[dict[str, Any]], target_btc_exposure: float) -> dict[str, Any]:
    cash = STARTING_BALANCE_USDT * (1 - target_btc_exposure)
    btc = (STARTING_BALANCE_USDT * target_btc_exposure * (1 - FEE_RATE)) / f(rows[0]["close"])
    values: list[float] = []
    last_month = rows[0]["month"]
    trades = 1
    for r in rows:
        price = f(r["close"])
        value = cash + btc * price
        if r["month"] != last_month and value > 0 and price > 0:
            last_month = r["month"]
            target_btc_value = value * target_btc_exposure
            current_btc_value = btc * price
            delta = target_btc_value - current_btc_value
            if abs(delta) >= MIN_TRADE_USDT:
                if delta > 0:
                    gross = min(cash, delta)
                    btc += (gross * (1 - FEE_RATE)) / price
                    cash -= gross
                else:
                    sell_btc = min(btc, abs(delta) / price)
                    cash += sell_btc * price * (1 - FEE_RATE)
                    btc -= sell_btc
                trades += 1
            value = cash + btc * price
        values.append(value)
    final = values[-1]
    return {"final_value_usdt": round(final, 8), "total_return_pct": round((final / STARTING_BALANCE_USDT - 1) * 100, 4), "max_drawdown_pct": round(max_drawdown(values) * 100, 4), "trades": trades, "target_btc_exposure_pct": round(target_btc_exposure * 100, 2)}


def summarize_curve_period(curve: list[dict[str, Any]], label: str, start: str, end: str) -> dict[str, Any]:
    part = [r for r in curve if start <= str(r.get("date")) <= end]
    if len(part) < 2:
        return {"label": label, "start": start, "end": end, "available": False}
    start_v = f(part[0]["portfolio_value_usdt"])
    end_v = f(part[-1]["portfolio_value_usdt"])
    values = [f(r["portfolio_value_usdt"]) for r in part]
    actions = [str(r.get("action")) for r in part]
    return {
        "label": label,
        "start": part[0]["date"],
        "end": part[-1]["date"],
        "available": True,
        "strategy_start_value_usdt": round(start_v, 8),
        "strategy_end_value_usdt": round(end_v, 8),
        "strategy_return_pct": round((end_v / start_v - 1) * 100, 4) if start_v > 0 else None,
        "strategy_max_drawdown_pct": round(max_drawdown(values) * 100, 4),
        "accumulate_days": sum(1 for a in actions if a.startswith("ALLOCATE")),
        "sell_days": sum(1 for a in actions if a in {"REDUCE_RISK_ALLOCATOR", "TRIM_ABOVE_TARGET_RISK", "TAKE_PROFIT_OVERVALUED_ONLY"}),
    }


def benchmark_period(rows: list[dict[str, Any]], label: str, start: str, end: str) -> dict[str, Any]:
    part = [r for r in rows if start <= r["date"] <= end]
    if len(part) < 2:
        return {"label": label, "available": False}
    return {
        "label": label,
        "available": True,
        "buy_and_hold": benchmark_buy_hold(part),
        "dca_monthly": benchmark_dca_monthly(part),
        "rebalance_40_60": benchmark_rebalanced(part, 0.40),
        "rebalance_60_40": benchmark_rebalanced(part, 0.60),
    }


def write_outputs(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = list(rows[0].keys()) if rows else []
    with RESULT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ic_rows = load_ic_series()
    hashrate = fetch_hashrate_history()
    rows = attach_production_cost(ic_rows, hashrate)
    if len(rows) < 400:
        raise ValueError(f"Not enough aligned historical rows for v0.4 backtest: {len(rows)}. Need at least 400.")

    strategy_result, curve = run_backtest(rows)
    benchmarks = {
        "buy_and_hold": benchmark_buy_hold(rows),
        "dca_monthly": benchmark_dca_monthly(rows),
        "rebalance_40_60_monthly": benchmark_rebalanced(rows, 0.40),
        "rebalance_60_40_monthly": benchmark_rebalanced(rows, 0.60),
    }
    period_breakdown = []
    for label, start, end in PERIODS:
        period_breakdown.append({
            "strategy": summarize_curve_period(curve, label, start, end),
            "benchmarks": benchmark_period(rows, label, start, end),
        })

    strategy = strategy_result["strategy"]
    bh = strategy_result["buy_and_hold"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "cohesivx_backtest_v0.4_structural_allocator_underpriced_no_sell_zone",
        "disclaimer": "Educational paper backtest only. Not financial advice. Uses historical daily close and simplified execution assumptions.",
        "assumptions": {
            "starting_balance_usdt": STARTING_BALANCE_USDT,
            "fee_rate": FEE_RATE,
            "min_trade_usdt": MIN_TRADE_USDT,
            "samples": SAMPLES,
            "uses_daily_close": True,
            "uses_slippage": False,
            "uses_full_fair_price_v2_production_cost": True,
            "uses_core_holding": True,
            "uses_structural_allocator": True,
            "uses_underpriced_mode": True,
            "uses_no_sell_zone_below_p50": True,
            "benchmarks": ["buy_and_hold", "dca_monthly", "rebalance_40_60_monthly", "rebalance_60_40_monthly"],
            "hashrate_source": BLOCKCHAIN_HASHRATE_URL,
            "electricity_usd_per_kwh": ELECTRICITY_USD_PER_KWH_BASE,
            "production_markup": PRODUCTION_MARKUP,
            "note": "v0.4 changes the strategy from a trader into a structural allocator: it targets BTC exposure by regime/undervaluation and blocks profit selling below contextual p50.",
        },
        "period": {"start": rows[0]["date"], "end": rows[-1]["date"], "days": len(rows)},
        "strategy": strategy,
        "buy_and_hold": bh,
        "benchmarks": benchmarks,
        "comparison": {
            "strategy_minus_buy_hold_usdt": round(strategy["final_portfolio_value_usdt"] - bh["final_value_usdt"], 8),
            "strategy_minus_buy_hold_pct_points": round(strategy["total_return_pct"] - bh["total_return_pct"], 4),
            "strategy_minus_dca_usdt": round(strategy["final_portfolio_value_usdt"] - benchmarks["dca_monthly"]["final_value_usdt"], 8),
            "strategy_minus_rebalance_40_60_usdt": round(strategy["final_portfolio_value_usdt"] - benchmarks["rebalance_40_60_monthly"]["final_value_usdt"], 8),
            "strategy_minus_rebalance_60_40_usdt": round(strategy["final_portfolio_value_usdt"] - benchmarks["rebalance_60_40_monthly"]["final_value_usdt"], 8),
        },
        "period_breakdown": period_breakdown,
    }
    write_outputs(summary, curve)
    print("Backtest completed")
    print("Version:", summary["version"])
    print("Period:", summary["period"])
    print("Strategy:", summary["strategy"])
    print("Benchmarks:", summary["benchmarks"])
    print("Comparison:", summary["comparison"])


if __name__ == "__main__":
    main()
