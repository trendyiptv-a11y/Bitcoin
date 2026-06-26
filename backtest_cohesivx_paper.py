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
    eff_j_per_th = efficiency_j_per_th(when)
    hashes_per_block = difficulty * (2.0 ** 32)
    joules_per_hash = eff_j_per_th / 1e12
    energy_joules = hashes_per_block * joules_per_hash
    energy_kwh = energy_joules / 3_600_000.0
    cost_block_electric = energy_kwh * ELECTRICITY_USD_PER_KWH_BASE
    cost_electric_per_btc = cost_block_electric / btc_per_block
    return float(cost_electric_per_btc * PRODUCTION_MARKUP)


def fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "CohesivX-Backtest/0.2"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_hashrate_history() -> list[dict[str, Any]]:
    raw = fetch_json(BLOCKCHAIN_HASHRATE_URL)
    rows = raw.get("values", [])
    out: list[dict[str, Any]] = []
    for r in rows:
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
        raise ValueError("Hashrate history is empty; cannot run V2 backtest.")
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


def active_rules(regime: str, edge: float, confidence: float, risk: float) -> dict[str, float | str]:
    r = (regime or "").lower()
    if r == "bear_late":
        profile, max_exp, core = "bear_late_core", 0.38, 0.10
        tp_small, tp_medium = 0.045, 0.09
    elif r.startswith("bear"):
        profile, max_exp, core = "bear_core_defensive", 0.30, 0.06
        tp_small, tp_medium = 0.035, 0.075
    elif r.startswith("bull"):
        profile, max_exp, core = "bull_core_participation", 0.65, 0.25
        tp_small, tp_medium = 0.08, 0.16
    else:
        profile, max_exp, core = "range_core_neutral", 0.45, 0.12
        tp_small, tp_medium = 0.06, 0.12

    if edge >= 0.18 and confidence >= 0.58 and risk < 0.28:
        max_exp = max(max_exp, 0.50 if not r.startswith("bull") else 0.75)
        core = max(core, 0.20 if not r.startswith("bear") else 0.12)
        profile += "_strong"
    elif edge >= 0.12 and confidence >= 0.52 and risk < 0.32:
        max_exp = min(max(max_exp, max_exp + 0.08), 0.65)
        core = max(core, 0.15)
        profile += "_moderate"

    return {"profile": profile, "tp_small": tp_small, "tp_medium": tp_medium, "max_exposure": max_exp, "core_exposure": core}


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

    sim_m10 = quantile(similar_mult, 0.10)
    sim_m50 = quantile(similar_mult, 0.50)
    sim_m90 = quantile(similar_mult, 0.90)
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
        adjust += 0.03
    elif regime in {"range", "neutral"}:
        adjust += 0.02
    if f(current.get("ic_struct")) >= 55 and f(current.get("ic_flux")) >= 50:
        adjust += 0.03

    raw_edge = confidence * expected_30d - (1.0 - confidence) * downside
    edge = raw_edge + adjust
    return {
        "available": True,
        "weighted_price_p10": weighted_p10,
        "weighted_price_p50": weighted_p50,
        "weighted_price_p90": weighted_p90,
        "weighted_multiplier_p50": weighted_m50,
        "expected_30d": expected_30d,
        "downside_risk": downside,
        "upside_to_p90": upside,
        "confidence": confidence,
        "decision_edge": edge,
        "distance_median": distance_median,
        "samples": len(nearest),
        "same_regime_samples": len(same),
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
    btc = 0.0
    cost_basis = 0.0
    realized = 0.0
    trades = buys = sells = 0
    equity_curve: list[dict[str, Any]] = []
    peak = STARTING_BALANCE_USDT
    max_drawdown = 0.0
    first_price = f(rows[0].get("close"))
    bh_btc = STARTING_BALANCE_USDT / first_price if first_price > 0 else 0.0

    for idx, row in enumerate(rows):
        price = f(row.get("close"))
        if price <= 0:
            continue
        hist = rows[:idx]
        sig = memory_signal(hist, row)
        action = "OBSERVE"
        executed_usdt = 0.0
        executed_btc = 0.0
        before = account(cash, btc, cost_basis, realized, price)
        rules = active_rules(str(row.get("regime") or ""), f(sig.get("decision_edge")), f(sig.get("confidence")), f(sig.get("downside_risk")))

        if sig.get("available"):
            edge = f(sig.get("decision_edge"))
            conf = f(sig.get("confidence"))
            risk = f(sig.get("downside_risk"))
            expected = f(sig.get("expected_30d"))
            max_exp = float(rules["max_exposure"])
            core_exp = float(rules["core_exposure"])
            core_btc = (before["value"] * core_exp / price) if price > 0 else 0.0
            sellable_btc = max(0.0, btc - core_btc)

            if edge > 0.18 and conf >= 0.58 and risk < 0.28 and before["exposure"] < max_exp:
                action = "ACCUMULATE_STRONG"
                target_exp = max_exp
                buy_fraction = min(MAX_ENTRY_FRACTION, max(0.0, target_exp - before["exposure"]))
                cash, btc, cost_basis, executed_usdt, executed_btc = execute_buy(cash, btc, cost_basis, price, before["value"] * buy_fraction)
            elif edge > 0.07 and conf >= 0.45 and risk < 0.38 and before["exposure"] < max_exp:
                action = "ACCUMULATE_SMALL"
                target_exp = min(max_exp, max(core_exp, before["exposure"] + 0.05))
                buy_fraction = min(0.05, max(0.0, target_exp - before["exposure"]))
                cash, btc, cost_basis, executed_usdt, executed_btc = execute_buy(cash, btc, cost_basis, price, before["value"] * buy_fraction)
            elif btc > 0 and before["unrealized_pct"] >= float(rules["tp_medium"]) and sellable_btc > 0:
                action = "TAKE_PROFIT_MEDIUM_CORE_PROTECTED"
                cash, btc, cost_basis, realized, executed_usdt, executed_btc = execute_sell(cash, btc, cost_basis, realized, price, min(sellable_btc, btc * 0.35))
            elif btc > 0 and before["unrealized_pct"] >= float(rules["tp_small"]) and (edge < 0.14 or risk > 0.32) and sellable_btc > 0:
                action = "TAKE_PROFIT_SMALL_CORE_PROTECTED"
                cash, btc, cost_basis, realized, executed_usdt, executed_btc = execute_sell(cash, btc, cost_basis, realized, price, min(sellable_btc, btc * 0.20))
            elif btc > 0 and (edge < -0.06 or expected < -0.08) and sellable_btc > 0:
                action = "REDUCE_RISK_CORE_PROTECTED"
                cash, btc, cost_basis, realized, executed_usdt, executed_btc = execute_sell(cash, btc, cost_basis, realized, price, min(sellable_btc, btc * 0.25))

            if executed_usdt >= MIN_TRADE_USDT:
                trades += 1
                if executed_btc > 0:
                    buys += 1
                elif executed_btc < 0:
                    sells += 1

        after = account(cash, btc, cost_basis, realized, price)
        peak = max(peak, after["value"])
        dd = (after["value"] - peak) / peak if peak > 0 else 0.0
        max_drawdown = min(max_drawdown, dd)
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
            "total_pnl_usdt": round(after["total_pnl"], 8),
            "total_pnl_pct": round(after["total_pnl_pct"] * 100, 4),
            "drawdown_pct": round(dd * 100, 4),
            "regime": row.get("regime", ""),
            "rule_profile": rules.get("profile", ""),
            "weighted_price_p50": round(f(sig.get("weighted_price_p50")), 8) if sig.get("available") else "",
            "decision_edge_pct": round(f(sig.get("decision_edge")) * 100, 4) if sig.get("available") else "",
            "memory_confidence_pct": round(f(sig.get("confidence")) * 100, 4) if sig.get("available") else "",
        })

    last_price = f(rows[-1].get("close"))
    final = account(cash, btc, cost_basis, realized, last_price)
    bh_value = bh_btc * last_price
    bh_pnl = bh_value - STARTING_BALANCE_USDT
    profitable_days = sum(1 for x in equity_curve if f(x.get("total_pnl_usdt")) > 0)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "cohesivx_backtest_v0.2_production_cost_multiplier_core_holding",
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
            "hashrate_source": BLOCKCHAIN_HASHRATE_URL,
            "electricity_usd_per_kwh": ELECTRICITY_USD_PER_KWH_BASE,
            "production_markup": PRODUCTION_MARKUP,
            "note": "v0.2 uses production-cost historical multiplier similarity and protects a core BTC position by regime/signal.",
        },
        "period": {"start": rows[0]["date"], "end": rows[-1]["date"], "days": len(rows)},
        "strategy": {
            "final_portfolio_value_usdt": round(final["value"], 8),
            "total_pnl_usdt": round(final["total_pnl"], 8),
            "total_return_pct": round(final["total_pnl_pct"] * 100, 4),
            "max_drawdown_pct": round(max_drawdown * 100, 4),
            "trades": trades,
            "buys": buys,
            "sells": sells,
            "final_cash_usdt": round(cash, 8),
            "final_btc_amount": round(btc, 12),
            "final_btc_exposure_pct": round(final["exposure"] * 100, 4),
            "profitable_days_ratio_pct": round((profitable_days / max(len(equity_curve), 1)) * 100, 4),
        },
        "buy_and_hold": {
            "final_value_usdt": round(bh_value, 8),
            "total_pnl_usdt": round(bh_pnl, 8),
            "total_return_pct": round((bh_pnl / STARTING_BALANCE_USDT) * 100, 4),
            "btc_amount": round(bh_btc, 12),
        },
        "comparison": {
            "strategy_minus_buy_hold_usdt": round(final["value"] - bh_value, 8),
            "strategy_minus_buy_hold_pct_points": round(final["total_pnl_pct"] * 100 - (bh_pnl / STARTING_BALANCE_USDT) * 100, 4),
        },
    }
    return summary, equity_curve


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
        raise ValueError(f"Not enough aligned historical rows for v0.2 backtest: {len(rows)}. Need at least 400.")
    summary, curve = run_backtest(rows)
    write_outputs(summary, curve)
    print("Backtest completed")
    print("Version:", summary["version"])
    print("Period:", summary["period"])
    print("Strategy:", summary["strategy"])
    print("Buy & hold:", summary["buy_and_hold"])
    print("Comparison:", summary["comparison"])


if __name__ == "__main__":
    main()
