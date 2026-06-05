#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_coeziv.py

Backtest independent pentru mecanismul coeziv BTC.

Scop:
- citește data/ic_btc_series.json;
- aplică generate_signals() din btc-swing-strategy/btc_swing_strategy.py;
- verifică ce s-a întâmplat după fiecare semnal pe mai multe orizonturi;
- salvează rezultate fără să modifice mecanismul live.

Output:
- data/coeziv_backtest_results.csv
- data/coeziv_backtest_summary.json

Rulare:
    python backtest_coeziv.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STRATEGY_DIR = ROOT / "btc-swing-strategy"

IC_SERIES_PATH = DATA_DIR / "ic_btc_series.json"
OUT_RESULTS = DATA_DIR / "coeziv_backtest_results.csv"
OUT_SUMMARY = DATA_DIR / "coeziv_backtest_summary.json"

HORIZONS_DAYS = [1, 7, 30, 90]
MOVE_THRESHOLD = 0.005  # 0.5% prag pentru direcție relevantă
FEE = 0.001             # cost ipotetic per schimbare poziție, 0.1%


if str(STRATEGY_DIR) not in sys.path:
    sys.path.append(str(STRATEGY_DIR))

try:
    from btc_swing_strategy import generate_signals
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Nu pot importa generate_signals din btc-swing-strategy/btc_swing_strategy.py"
    ) from exc


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        x = float(value)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def load_ic_series(path: Path = IC_SERIES_PATH) -> pd.DataFrame:
    """Încarcă seria IC oficială într-un DataFrame sortat temporal."""
    if not path.exists():
        raise FileNotFoundError(f"Nu am găsit seria IC: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    series = raw.get("series")
    if not isinstance(series, list) or not series:
        raise RuntimeError("ic_btc_series.json nu conține o listă validă în cheia 'series'.")

    df = pd.DataFrame(series)
    if "t" not in df.columns:
        raise RuntimeError("Seria IC nu conține coloana temporală 't'.")

    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("date").sort_index()

    required = {"close", "ic_struct", "ic_dir", "ic_flux", "ic_cycle", "regime"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Seria IC nu conține coloanele necesare: {missing}")

    for col in ["close", "ic_struct", "ic_dir", "ic_flux", "ic_cycle"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close", "ic_struct", "ic_dir", "ic_flux"])
    if df.empty:
        raise RuntimeError("Seria IC este goală după curățarea valorilor numerice.")

    return df


def classify_outcome(signal: str, ret: float | None, threshold: float = MOVE_THRESHOLD) -> str:
    """Clasifică rezultatul unui semnal pentru un randament viitor."""
    sig = (signal or "").lower()
    if ret is None or not math.isfinite(ret):
        return "unknown"

    if sig == "long":
        if ret >= threshold:
            return "correct"
        if ret <= -threshold:
            return "wrong"
        return "flat"

    if sig == "short":
        if ret <= -threshold:
            return "correct"
        if ret >= threshold:
            return "wrong"
        return "flat"

    # Pentru flat/neutral, corect = piață rămasă în interval mic.
    if abs(ret) < threshold:
        return "correct"
    return "wrong"


def build_backtest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Adaugă randamente viitoare și clasificări pentru fiecare orizont."""
    out = df.copy()

    for horizon in HORIZONS_DAYS:
        ret_col = f"ret_{horizon}d"
        outcome_col = f"outcome_{horizon}d"
        out[ret_col] = out["close"].shift(-horizon) / out["close"] - 1.0
        out[outcome_col] = [
            classify_outcome(sig, _safe_float(ret))
            for sig, ret in zip(out["signal"], out[ret_col])
        ]

    return out


def summarize_by_signal(df: pd.DataFrame) -> Dict[str, Any]:
    """Construiește sumar pe semnal și orizont."""
    summary: Dict[str, Any] = {
        "meta": {
            "rows": int(len(df)),
            "from": df.index.min().isoformat() if len(df) else None,
            "to": df.index.max().isoformat() if len(df) else None,
            "horizons_days": HORIZONS_DAYS,
            "move_threshold": MOVE_THRESHOLD,
            "fee_assumption": FEE,
        },
        "signals": {},
    }

    for sig in ["long", "short", "flat"]:
        sig_df = df[df["signal"].astype(str).str.lower() == sig]
        sig_block: Dict[str, Any] = {"count": int(len(sig_df)), "horizons": {}}

        for horizon in HORIZONS_DAYS:
            ret_col = f"ret_{horizon}d"
            outcome_col = f"outcome_{horizon}d"
            valid = sig_df[sig_df[outcome_col] != "unknown"].copy()

            total = int(len(valid))
            correct = int((valid[outcome_col] == "correct").sum()) if total else 0
            wrong = int((valid[outcome_col] == "wrong").sum()) if total else 0
            flat = int((valid[outcome_col] == "flat").sum()) if total else 0

            mean_ret = float(valid[ret_col].mean()) if total else None
            median_ret = float(valid[ret_col].median()) if total else None
            p10 = float(valid[ret_col].quantile(0.10)) if total else None
            p90 = float(valid[ret_col].quantile(0.90)) if total else None

            sig_block["horizons"][f"{horizon}d"] = {
                "samples": total,
                "correct": correct,
                "wrong": wrong,
                "flat": flat,
                "accuracy": (correct / total) if total else None,
                "wrong_rate": (wrong / total) if total else None,
                "flat_rate": (flat / total) if total else None,
                "mean_return": mean_ret,
                "median_return": median_ret,
                "p10_return": p10,
                "p90_return": p90,
            }

        summary["signals"][sig] = sig_block

    return summary


def simulate_position_equity(df: pd.DataFrame, fee: float = FEE) -> pd.DataFrame:
    """
    Simulare simplă de equity:
    - long = +1
    - short = -1
    - flat/neutral = 0

    Notă: acesta nu este sistem de execuție real; este doar verificare structurală.
    """
    out = df.copy()
    signal_to_pos = {"long": 1, "short": -1, "flat": 0, "neutral": 0}
    out["position"] = out["signal"].astype(str).str.lower().map(signal_to_pos).fillna(0).astype(int)

    close = out["close"].astype(float)
    daily_ret = close.pct_change().fillna(0.0)

    # poziția de ieri câștigă randamentul de azi, evitând look-ahead direct.
    pos_prev = out["position"].shift(1).fillna(0)
    strat_ret = pos_prev * daily_ret

    # cost la schimbarea poziției
    position_change = out["position"].diff().abs().fillna(0)
    strat_ret = strat_ret - position_change * fee

    out["strategy_return"] = strat_ret
    out["equity_coeziv"] = (1.0 + strat_ret).cumprod()
    out["buy_hold_equity"] = close / close.iloc[0]

    running_max = out["equity_coeziv"].cummax()
    out["drawdown"] = out["equity_coeziv"] / running_max - 1.0

    return out


def add_equity_summary(summary: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "equity_coeziv" not in df.columns:
        return summary

    final_equity = float(df["equity_coeziv"].iloc[-1])
    buy_hold = float(df["buy_hold_equity"].iloc[-1])
    max_drawdown = float(df["drawdown"].min())
    trades = int((df["position"].diff().abs().fillna(0) > 0).sum())

    summary["equity_simulation"] = {
        "final_equity_coeziv": final_equity,
        "final_equity_buy_hold": buy_hold,
        "max_drawdown": max_drawdown,
        "position_changes": trades,
        "note": "Simulare orientativă: poziția din ziua precedentă aplicată pe randamentul zilei curente; cost la schimbare poziție.",
    }
    return summary


def main() -> None:
    df = load_ic_series()
    df = generate_signals(df)

    if "signal" not in df.columns:
        raise RuntimeError("generate_signals nu a produs coloana 'signal'.")

    df_bt = build_backtest_rows(df)
    df_bt = simulate_position_equity(df_bt)
    summary = summarize_by_signal(df_bt)
    summary = add_equity_summary(summary, df_bt)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    export_cols = [
        "close", "ic_struct", "ic_dir", "ic_flux", "ic_cycle", "regime", "signal",
        "position", "strategy_return", "equity_coeziv", "buy_hold_equity", "drawdown",
    ]
    for horizon in HORIZONS_DAYS:
        export_cols.extend([f"ret_{horizon}d", f"outcome_{horizon}d"])

    existing_cols = [c for c in export_cols if c in df_bt.columns]
    df_bt[existing_cols].to_csv(OUT_RESULTS, index_label="date")

    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Backtest coeziv salvat în: {OUT_RESULTS}")
    print(f"[OK] Sumar backtest salvat în: {OUT_SUMMARY}")
    print(json.dumps(summary.get("equity_simulation", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
