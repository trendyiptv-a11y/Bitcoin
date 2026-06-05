#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_coeziv.py

Backtest independent pentru mecanismul coeziv BTC.

Scop:
- citește data/ic_btc_series.json;
- aplică generate_signals() din btc-swing-strategy/btc_swing_strategy.py;
- verifică ce s-a întâmplat după fiecare semnal pe mai multe orizonturi;
- testează evenimentele de degradare structurală (bear_struct) versus drawdown-uri ulterioare;
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

# Test special pentru ipoteza: degradarea structurală apare înaintea scăderii majore.
DEGRADATION_REGIMES = {"bear_struct", "bear_late", "accum_bear"}
MAJOR_DRAWDOWN_THRESHOLD = -0.20
DEGRADATION_WINDOWS_DAYS = [30, 60, 90, 180]
COOLDOWN_DAYS = 30  # evită numărarea aceleiași faze bear ca evenimente zilnice multiple


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


def _is_degradation_regime(regime: Any) -> bool:
    return str(regime or "").lower() in DEGRADATION_REGIMES


def find_structural_degradation_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifică primele intrări în regim de degradare structurală.

    Un eveniment este păstrat doar dacă:
    - ziua curentă este în DEGRADATION_REGIMES;
    - ziua precedentă nu era în DEGRADATION_REGIMES;
    - au trecut cel puțin COOLDOWN_DAYS de la ultimul eveniment păstrat.

    Astfel evităm să numărăm aceeași fază bear de zeci/sute de ori.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out["is_degradation_regime"] = out["regime"].apply(_is_degradation_regime)
    out["was_degradation_regime"] = out["is_degradation_regime"].shift(1).fillna(False)
    candidates = out[out["is_degradation_regime"] & (~out["was_degradation_regime"])].copy()

    kept_rows = []
    last_event_date = None
    for ts, row in candidates.iterrows():
        if last_event_date is not None:
            delta_days = (ts - last_event_date).days
            if delta_days < COOLDOWN_DAYS:
                continue
        kept_rows.append((ts, row))
        last_event_date = ts

    if not kept_rows:
        return out.iloc[0:0].copy()

    events = pd.DataFrame([row for _, row in kept_rows], index=[ts for ts, _ in kept_rows])
    return events


def analyze_structural_degradation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Testează ipoteza:
    Prima apariție a degradării structurale este urmată de drawdown major?

    Pentru fiecare eveniment, calculează drawdown-ul minim față de prețul de intrare
    în ferestrele 30/60/90/180 zile și timpul până la atingerea pragului -20%.
    """
    events = find_structural_degradation_events(df)

    result: Dict[str, Any] = {
        "definition": {
            "degradation_regimes": sorted(DEGRADATION_REGIMES),
            "major_drawdown_threshold": MAJOR_DRAWDOWN_THRESHOLD,
            "windows_days": DEGRADATION_WINDOWS_DAYS,
            "cooldown_days": COOLDOWN_DAYS,
            "interpretation": "Confirmat dacă minimul din fereastră scade cu cel puțin 20% față de prețul din ziua evenimentului.",
        },
        "event_count": int(len(events)),
        "windows": {},
        "events": [],
    }

    if events.empty:
        return result

    close = df["close"].astype(float)

    for ts, event in events.iterrows():
        entry_price = float(event["close"])
        event_record: Dict[str, Any] = {
            "date": ts.isoformat(),
            "regime": str(event.get("regime")),
            "signal": str(event.get("signal")),
            "entry_price": entry_price,
            "ic_struct": _safe_float(event.get("ic_struct")),
            "ic_dir": _safe_float(event.get("ic_dir")),
            "ic_flux": _safe_float(event.get("ic_flux")),
            "windows": {},
        }

        for window in DEGRADATION_WINDOWS_DAYS:
            end_ts = ts + pd.Timedelta(days=window)
            future = close[(close.index > ts) & (close.index <= end_ts)]

            if future.empty or entry_price <= 0:
                event_record["windows"][f"{window}d"] = {
                    "available": False,
                    "confirmed": None,
                    "min_drawdown": None,
                    "days_to_threshold": None,
                    "min_price": None,
                }
                continue

            drawdowns = future / entry_price - 1.0
            min_drawdown = float(drawdowns.min())
            min_ts = drawdowns.idxmin()
            min_price = float(future.loc[min_ts])

            threshold_hits = drawdowns[drawdowns <= MAJOR_DRAWDOWN_THRESHOLD]
            if threshold_hits.empty:
                confirmed = False
                days_to_threshold = None
            else:
                confirmed = True
                first_hit_ts = threshold_hits.index[0]
                days_to_threshold = int((first_hit_ts - ts).days)

            event_record["windows"][f"{window}d"] = {
                "available": True,
                "confirmed": confirmed,
                "min_drawdown": min_drawdown,
                "days_to_threshold": days_to_threshold,
                "min_price": min_price,
            }

        result["events"].append(event_record)

    for window in DEGRADATION_WINDOWS_DAYS:
        key = f"{window}d"
        available = [e for e in result["events"] if e["windows"][key]["available"]]
        confirmed = [e for e in available if e["windows"][key]["confirmed"] is True]
        min_drawdowns = [e["windows"][key]["min_drawdown"] for e in available]
        days_to_threshold = [
            e["windows"][key]["days_to_threshold"]
            for e in confirmed
            if e["windows"][key]["days_to_threshold"] is not None
        ]

        result["windows"][key] = {
            "available_events": len(available),
            "confirmed_events": len(confirmed),
            "confirmation_rate": (len(confirmed) / len(available)) if available else None,
            "avg_min_drawdown": float(pd.Series(min_drawdowns).mean()) if min_drawdowns else None,
            "median_min_drawdown": float(pd.Series(min_drawdowns).median()) if min_drawdowns else None,
            "avg_days_to_threshold": float(pd.Series(days_to_threshold).mean()) if days_to_threshold else None,
            "median_days_to_threshold": float(pd.Series(days_to_threshold).median()) if days_to_threshold else None,
        }

    return result


def main() -> None:
    df = load_ic_series()
    df = generate_signals(df)

    if "signal" not in df.columns:
        raise RuntimeError("generate_signals nu a produs coloana 'signal'.")

    df_bt = build_backtest_rows(df)
    df_bt = simulate_position_equity(df_bt)
    summary = summarize_by_signal(df_bt)
    summary = add_equity_summary(summary, df_bt)
    summary["structural_degradation_events"] = analyze_structural_degradation(df_bt)

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
    print(json.dumps(summary.get("structural_degradation_events", {}).get("windows", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
