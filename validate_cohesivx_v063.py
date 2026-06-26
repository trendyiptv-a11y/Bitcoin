#!/usr/bin/env python
from __future__ import annotations

import csv
import importlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import backtest_cohesivx_paper as engine

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "btc-swing-strategy"
VALIDATION_JSON = OUT_DIR / "validation_report_v063.json"
VALIDATION_CSV = OUT_DIR / "validation_summary_v063.csv"

BASE_PARAMS = {
    "FEE_RATE": engine.FEE_RATE,
    "DRAWDOWN_GUARD": engine.DRAWDOWN_GUARD,
    "DRAWDOWN_GUARD_TARGET_CAP": engine.DRAWDOWN_GUARD_TARGET_CAP,
    "MAX_ENTRY_FRACTION_GUARD": engine.MAX_ENTRY_FRACTION_GUARD,
    "TARGET_UP_STEP_DEEP": engine.TARGET_UP_STEP_DEEP,
}

FEE_TESTS = [0.001, 0.002, 0.003, 0.005, 0.01]
SLIPPAGE_TESTS = [0.0, 0.001, 0.0025, 0.005, 0.01]
START_DATES = [
    "2011-02-01", "2012-01-01", "2013-01-01", "2014-01-01", "2015-01-01",
    "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01",
    "2021-01-01", "2022-01-01", "2023-01-01",
]
PERIOD_WINDOWS = [
    ("period_2011_2014", "2011-02-01", "2014-12-31"),
    ("period_2015_2018", "2015-01-01", "2018-12-31"),
    ("period_2019_2022", "2019-01-01", "2022-12-31"),
    ("period_2023_2026", "2023-01-01", "2026-12-31"),
]
PARAMETER_TESTS = [
    {"DRAWDOWN_GUARD": -0.58},
    {"DRAWDOWN_GUARD": -0.60},
    {"DRAWDOWN_GUARD": -0.62},
    {"DRAWDOWN_GUARD_TARGET_CAP": 0.70},
    {"DRAWDOWN_GUARD_TARGET_CAP": 0.72},
    {"DRAWDOWN_GUARD_TARGET_CAP": 0.74},
    {"MAX_ENTRY_FRACTION_GUARD": 0.06},
    {"MAX_ENTRY_FRACTION_GUARD": 0.08},
    {"MAX_ENTRY_FRACTION_GUARD": 0.10},
    {"TARGET_UP_STEP_DEEP": 0.12},
    {"TARGET_UP_STEP_DEEP": 0.14},
    {"TARGET_UP_STEP_DEEP": 0.16},
]


def reset_engine() -> None:
    for key, value in BASE_PARAMS.items():
        setattr(engine, key, value)


def set_params(params: dict[str, Any]) -> None:
    reset_engine()
    for key, value in params.items():
        setattr(engine, key, value)


def load_rows() -> list[dict[str, Any]]:
    ic_rows = engine.load_ic_series()
    hashrate = engine.fetch_hashrate_history()
    rows = engine.attach_production_cost(ic_rows, hashrate)
    if len(rows) < 400:
        raise ValueError(f"Not enough rows for validation: {len(rows)}")
    return rows


def filter_rows(rows: list[dict[str, Any]], start: str | None = None, end: str | None = None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        date = str(row.get("date"))
        if start and date < start:
            continue
        if end and date > end:
            continue
        out.append(row)
    return out


def install_slippage(slippage: float):
    original_buy = engine.execute_buy
    original_sell = engine.execute_sell

    def execute_buy_with_slippage(cash: float, btc: float, cost_basis: float, price: float, gross_usdt: float):
        return original_buy(cash, btc, cost_basis, price * (1.0 + slippage), gross_usdt)

    def execute_sell_with_slippage(cash: float, btc: float, cost_basis: float, realized: float, price: float, sell_btc: float):
        return original_sell(cash, btc, cost_basis, realized, price * (1.0 - slippage), sell_btc)

    engine.execute_buy = execute_buy_with_slippage
    engine.execute_sell = execute_sell_with_slippage
    return original_buy, original_sell


def restore_slippage(original_buy, original_sell) -> None:
    engine.execute_buy = original_buy
    engine.execute_sell = original_sell


def run_case(name: str, rows: list[dict[str, Any]], category: str, params: dict[str, Any] | None = None, slippage: float = 0.0) -> dict[str, Any]:
    if len(rows) < 400:
        return {"name": name, "category": category, "available": False, "reason": f"too few rows: {len(rows)}"}
    set_params(params or {})
    original_buy, original_sell = install_slippage(slippage)
    try:
        result, _curve = engine.run_backtest(rows)
        benchmarks = {
            "buy_and_hold": engine.benchmark_buy_hold(rows),
            "dca_monthly": engine.benchmark_dca_monthly(rows),
            "rebalance_40_60_monthly": engine.benchmark_rebalanced(rows, 0.40),
            "rebalance_60_40_monthly": engine.benchmark_rebalanced(rows, 0.60),
        }
    finally:
        restore_slippage(original_buy, original_sell)
        reset_engine()

    strategy = result["strategy"]
    final_value = float(strategy["final_portfolio_value_usdt"])
    dca = float(benchmarks["dca_monthly"]["final_value_usdt"])
    rb40 = float(benchmarks["rebalance_40_60_monthly"]["final_value_usdt"])
    rb60 = float(benchmarks["rebalance_60_40_monthly"]["final_value_usdt"])
    bh = float(benchmarks["buy_and_hold"]["final_value_usdt"])
    dd = float(strategy["max_drawdown_pct"])
    rb60_dd = float(benchmarks["rebalance_60_40_monthly"]["max_drawdown_pct"])

    return {
        "name": name,
        "category": category,
        "available": True,
        "start": rows[0]["date"],
        "end": rows[-1]["date"],
        "days": len(rows),
        "params": params or {},
        "slippage": slippage,
        "final_value_usdt": final_value,
        "total_return_pct": float(strategy["total_return_pct"]),
        "max_drawdown_pct": dd,
        "trades": int(strategy["trades"]),
        "buys": int(strategy["buys"]),
        "sells": int(strategy["sells"]),
        "final_btc_amount": float(strategy["final_btc_amount"]),
        "beats_dca": final_value > dca,
        "beats_40_60": final_value > rb40,
        "beats_60_40": final_value > rb60,
        "beats_buy_hold": final_value > bh,
        "drawdown_better_than_60_40": dd > rb60_dd,
        "strategy_minus_dca_usdt": round(final_value - dca, 8),
        "strategy_minus_40_60_usdt": round(final_value - rb40, 8),
        "strategy_minus_60_40_usdt": round(final_value - rb60, 8),
    }


def score_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [c for c in cases if c.get("available")]
    if not valid:
        return {"available": False}

    def ratio(key: str) -> float:
        return round(sum(1 for c in valid if c.get(key)) / len(valid), 4)

    finals = [float(c["final_value_usdt"]) for c in valid]
    drawdowns = [float(c["max_drawdown_pct"]) for c in valid]
    positive = [c for c in valid if float(c["final_value_usdt"]) > engine.STARTING_BALANCE_USDT]
    robustness_score = round(
        100.0 * (
            0.30 * ratio("beats_dca")
            + 0.25 * ratio("beats_40_60")
            + 0.20 * ratio("drawdown_better_than_60_40")
            + 0.15 * ratio("beats_60_40")
            + 0.10 * (len(positive) / len(valid))
        ),
        2,
    )
    return {
        "available": True,
        "cases": len(valid),
        "robustness_score": robustness_score,
        "beats_dca_ratio": ratio("beats_dca"),
        "beats_40_60_ratio": ratio("beats_40_60"),
        "beats_60_40_ratio": ratio("beats_60_40"),
        "drawdown_better_than_60_40_ratio": ratio("drawdown_better_than_60_40"),
        "positive_final_ratio": round(len(positive) / len(valid), 4),
        "median_final_value_usdt": round(sorted(finals)[len(finals) // 2], 8),
        "worst_final_value_usdt": round(min(finals), 8),
        "best_final_value_usdt": round(max(finals), 8),
        "worst_drawdown_pct": round(min(drawdowns), 4),
        "best_drawdown_pct": round(max(drawdowns), 4),
    }


def write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = report["cases"]
    fields = [
        "category", "name", "available", "start", "end", "days", "slippage",
        "final_value_usdt", "total_return_pct", "max_drawdown_pct", "trades", "buys", "sells",
        "final_btc_amount", "beats_dca", "beats_40_60", "beats_60_40",
        "drawdown_better_than_60_40", "strategy_minus_dca_usdt", "strategy_minus_40_60_usdt",
        "strategy_minus_60_40_usdt", "params", "reason",
    ]
    with VALIDATION_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = load_rows()
    cases: list[dict[str, Any]] = []

    cases.append(run_case("baseline_v063", rows, "baseline"))

    for fee in FEE_TESTS:
        cases.append(run_case(f"fee_{fee:.4f}", rows, "fee_stress", params={"FEE_RATE": fee}))

    for slippage in SLIPPAGE_TESTS:
        cases.append(run_case(f"slippage_{slippage:.4f}", rows, "slippage_stress", slippage=slippage))

    for start in START_DATES:
        cases.append(run_case(f"start_{start}", filter_rows(rows, start=start), "start_date_sensitivity"))

    for label, start, end in PERIOD_WINDOWS:
        cases.append(run_case(label, filter_rows(rows, start=start, end=end), "period_window"))

    for params in PARAMETER_TESTS:
        key = "_".join(f"{k}_{v}" for k, v in params.items())
        cases.append(run_case(key, rows, "parameter_sensitivity", params=params))

    grouped = {}
    for category in sorted({c["category"] for c in cases}):
        grouped[category] = score_cases([c for c in cases if c["category"] == category])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "cohesivx_validation_v063_robustness_harness",
        "engine_version": "cohesivx_backtest_v0.6.3_guarded_accumulation",
        "method": "Uses the existing CohesivX v0.6.3 engine and stress-tests it without changing the core decision logic.",
        "period": {"start": rows[0]["date"], "end": rows[-1]["date"], "days": len(rows)},
        "summary": score_cases(cases),
        "grouped_summary": grouped,
        "cases": cases,
    }
    write_report(report)
    print("Validation completed")
    print("Summary:", report["summary"])
    print("Grouped:", report["grouped_summary"])
    print("JSON:", VALIDATION_JSON)
    print("CSV:", VALIDATION_CSV)


if __name__ == "__main__":
    main()
