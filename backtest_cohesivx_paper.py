#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
import statistics
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


def load_series() -> list[dict[str, Any]]:
    path = find_ic_file()
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("series", []) if isinstance(raw, dict) else raw
    clean: list[dict[str, Any]] = []
    for r in rows:
        close = f(r.get("close"))
        t = r.get("t")
        if close <= 0 or t is None:
            continue
        dt = datetime.fromtimestamp(float(t) / 1000.0, tz=timezone.utc)
        row = dict(r)
        row["date"] = dt.date().isoformat()
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


def context_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    cols = ["ic_struct", "ic_dir", "ic_flux", "ic_cycle", "vol30_index"]
    return math.sqrt(sum(((f(a.get(c)) - f(b.get(c))) / 100.0) ** 2 for c in cols))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def active_rules(regime: str) -> dict[str, float | str]:
    r = (regime or "").lower()
    if r == "bear_late":
        return {"profile": "bear_late_defensive", "tp_small": 0.045, "tp_medium": 0.09, "max_exposure": 0.34}
    if r.startswith("bear"):
        return {"profile": "bear_defensive", "tp_small": 0.035, "tp_medium": 0.075, "max_exposure": 0.30}
    if r.startswith("bull"):
        return {"profile": "bull_patient", "tp_small": 0.08, "tp_medium": 0.16, "max_exposure": 0.40}
    return {"profile": "range_neutral_default", "tp_small": 0.06, "tp_medium": 0.12, "max_exposure": 0.35}


def memory_signal(history: list[dict[str, Any]], current: dict[str, Any]) -> dict[str, Any]:
    if len(history) < 250:
        return {"available": False, "reason": "not enough history"}
    price = f(current.get("close"))
    nearest = sorted(history, key=lambda r: context_distance(r, current))[:250]
    same_regime_pool = [r for r in history if str(r.get("regime")) == str(current.get("regime"))]
    same = sorted(same_regime_pool, key=lambda r: context_distance(r, current))[:250] if same_regime_pool else []

    similar_prices = [f(r.get("close")) for r in nearest if f(r.get("close")) > 0]
    same_prices = [f(r.get("close")) for r in same if f(r.get("close")) > 0]
    if len(similar_prices) < 25:
        return {"available": False, "reason": "not enough similar prices"}

    sim_p10 = quantile(similar_prices, 0.10)
    sim_p50 = quantile(similar_prices, 0.50)
    sim_p90 = quantile(similar_prices, 0.90)
    same_p10 = quantile(same_prices, 0.10) if same_prices else sim_p10
    same_p50 = quantile(same_prices, 0.50) if same_prices else sim_p50
    same_p90 = quantile(same_prices, 0.90) if same_prices else sim_p90

    weighted_p10 = 0.60 * sim_p10 + 0.40 * same_p10
    weighted_p50 = 0.60 * sim_p50 + 0.40 * same_p50
    weighted_p90 = 0.60 * sim_p90 + 0.40 * same_p90
    expected_30d = (weighted_p50 - price) / price if price > 0 else 0.0
    downside = abs(min(0.0, (weighted_p10 - price) / price)) if price > 0 else 0.0
    upside = max(0.0, (weighted_p90 - price) / price) if price > 0 else 0.0

    distance_median = statistics.median([context_distance(r, current) for r in nearest])
    sample_conf = clamp(len(nearest) / 250.0, 0.0, 1.0)
    same_conf = clamp(len(same) / max(len(nearest), 1), 0.0, 1.0)
    distance_conf = clamp(1.0 - max(distance_median, 0.0) / 0.80, 0.0, 1.0)
    context_conf = clamp(0.40 * sample_conf + 0.25 * same_conf + 0.35 * distance_conf, 0.0, 1.0)

    regime = str(current.get("regime") or "")
    regime_adjust = 0.0
    if regime == "bear_late":
        regime_adjust += 0.06
    elif regime.startswith("bear"):
        regime_adjust -= 0.04
    elif regime.startswith("bull"):
        regime_adjust += 0.03
    elif regime in {"range", "neutral"}:
        regime_adjust += 0.02
    if f(current.get("ic_struct")) >= 55 and f(current.get("ic_flux")) >= 50:
        regime_adjust += 0.03

    raw_edge = context_conf * expected_30d - (1.0 - context_conf) * downside
    edge = raw_edge + regime_adjust
    return {
        "available": True,
        "weighted_p10": weighted_p10,
        "weighted_p50": weighted_p50,
        "weighted_p90": weighted_p90,
        "expected_30d": expected_30d,
        "downside_risk": downside,
        "upside_to_p90": upside,
        "confidence": context_conf,
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
    return {
        "cash": cash,
        "btc": btc,
        "cost_basis": cost_basis,
        "realized": realized,
        "value": value,
        "exposure": exposure,
        "unrealized": unrealized,
        "unrealized_pct": unrealized_pct,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl / STARTING_BALANCE_USDT,
    }


def run_backtest(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cash = STARTING_BALANCE_USDT
    btc = 0.0
    cost_basis = 0.0
    realized = 0.0
    trades = 0
    buys = 0
    sells = 0
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
        rules = active_rules(str(row.get("regime") or ""))
        action = "OBSERVE"
        executed_usdt = 0.0
        executed_btc = 0.0
        before = account(cash, btc, cost_basis, realized, price)

        if sig.get("available"):
            edge = f(sig.get("decision_edge"))
            conf = f(sig.get("confidence"))
            risk = f(sig.get("downside_risk"))
            expected = f(sig.get("expected_30d"))
            max_exp = float(rules["max_exposure"])

            if btc > 0 and before["unrealized_pct"] >= float(rules["tp_medium"]):
                action = "TAKE_PROFIT_MEDIUM"
                sell_fraction = 0.35
            elif btc > 0 and before["unrealized_pct"] >= float(rules["tp_small"]) and (edge < 0.14 or risk > 0.32):
                action = "TAKE_PROFIT_SMALL"
                sell_fraction = 0.20
            elif btc > 0 and (edge < -0.06 or expected < -0.08):
                action = "REDUCE_RISK"
                sell_fraction = 0.25
            elif edge > 0.18 and conf >= 0.58 and risk < 0.28 and before["exposure"] < max_exp:
                action = "ACCUMULATE_SMALL"
                buy_fraction = min(MAX_ENTRY_FRACTION, max_exp - before["exposure"])
            elif edge > 0.07 and conf >= 0.45 and risk < 0.38 and before["exposure"] < max_exp:
                action = "OBSERVE_ACCUMULATE_SMALL"
                buy_fraction = min(0.05, max_exp - before["exposure"])

            if action in {"ACCUMULATE_SMALL", "OBSERVE_ACCUMULATE_SMALL"}:
                gross_usdt = min(cash, before["value"] * buy_fraction)
                if gross_usdt >= MIN_TRADE_USDT:
                    fee = gross_usdt * FEE_RATE
                    net_usdt = gross_usdt - fee
                    executed_btc = net_usdt / price
                    executed_usdt = gross_usdt
                    cash -= gross_usdt
                    btc += executed_btc
                    cost_basis += gross_usdt
                    trades += 1
                    buys += 1
            elif action in {"TAKE_PROFIT_SMALL", "TAKE_PROFIT_MEDIUM", "REDUCE_RISK"}:
                sell_btc = min(btc, btc * sell_fraction)
                gross_usdt = sell_btc * price
                if gross_usdt >= MIN_TRADE_USDT and btc > 0:
                    fee = gross_usdt * FEE_RATE
                    net_usdt = gross_usdt - fee
                    basis_removed = cost_basis * (sell_btc / btc) if cost_basis > 0 else 0.0
                    realized += net_usdt - basis_removed
                    cost_basis = max(0.0, cost_basis - basis_removed)
                    cash += net_usdt
                    btc -= sell_btc
                    executed_btc = -sell_btc
                    executed_usdt = gross_usdt
                    trades += 1
                    sells += 1

        after = account(cash, btc, cost_basis, realized, price)
        peak = max(peak, after["value"])
        dd = (after["value"] - peak) / peak if peak > 0 else 0.0
        max_drawdown = min(max_drawdown, dd)
        equity_curve.append({
            "date": row["date"],
            "close": round(price, 8),
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
        "version": "cohesivx_backtest_v0.1_simple_memory_price_context",
        "disclaimer": "Educational paper backtest only. Not financial advice. Uses historical close data and simplified execution assumptions.",
        "assumptions": {
            "starting_balance_usdt": STARTING_BALANCE_USDT,
            "fee_rate": FEE_RATE,
            "min_trade_usdt": MIN_TRADE_USDT,
            "uses_daily_close": True,
            "uses_slippage": False,
            "uses_full_fair_price_v2_production_cost": False,
            "note": "v0.1 uses IC context similarity over historical close prices, not the full production-cost V2 model. It is meant as a first sanity check.",
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
    rows = load_series()
    if len(rows) < 300:
        raise ValueError(f"Not enough historical rows for backtest: {len(rows)}. Need at least 300.")
    summary, curve = run_backtest(rows)
    write_outputs(summary, curve)
    print("Backtest completed")
    print("Period:", summary["period"])
    print("Strategy:", summary["strategy"])
    print("Buy & hold:", summary["buy_and_hold"])
    print("Comparison:", summary["comparison"])


if __name__ == "__main__":
    main()
