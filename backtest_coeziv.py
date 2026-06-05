#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_coeziv.py

Backtest independent pentru mecanismul coeziv BTC.

Scop:
- citește data/ic_btc_series.json;
- aplică generate_signals() din btc-swing-strategy/btc_swing_strategy.py;
- verifică ce s-a întâmplat după fiecare semnal pe mai multe orizonturi;
- testează evenimentele de degradare structurală versus drawdown-uri ulterioare;
- exportă și cardul automat btc-swing-strategy/risk_window.json;
- salvează rezultate fără să modifice mecanismul live.
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
OUT_RISK_WINDOW = STRATEGY_DIR / "risk_window.json"

HORIZONS_DAYS = [1, 7, 30, 90]
MOVE_THRESHOLD = 0.005
FEE = 0.001

DEGRADATION_REGIMES = {"bear_struct", "bear_late", "accum_bear"}
MAJOR_DRAWDOWN_THRESHOLD = -0.20
DEGRADATION_WINDOWS_DAYS = [30, 60, 90, 180]
RISK_WINDOW_DAYS = 180
COOLDOWN_DAYS = 30

if str(STRATEGY_DIR) not in sys.path:
    sys.path.append(str(STRATEGY_DIR))

try:
    from btc_swing_strategy import generate_signals
except Exception as exc:
    raise RuntimeError("Nu pot importa generate_signals din btc-swing-strategy/btc_swing_strategy.py") from exc


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def load_ic_series(path: Path = IC_SERIES_PATH) -> pd.DataFrame:
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

    if abs(ret) < threshold:
        return "correct"
    return "wrong"


def build_backtest_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for horizon in HORIZONS_DAYS:
        ret_col = f"ret_{horizon}d"
        outcome_col = f"outcome_{horizon}d"
        out[ret_col] = out["close"].shift(-horizon) / out["close"] - 1.0
        out[outcome_col] = [classify_outcome(sig, _safe_float(ret)) for sig, ret in zip(out["signal"], out[ret_col])]
    return out


def summarize_by_signal(df: pd.DataFrame) -> Dict[str, Any]:
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

            sig_block["horizons"][f"{horizon}d"] = {
                "samples": total,
                "correct": correct,
                "wrong": wrong,
                "flat": flat,
                "accuracy": (correct / total) if total else None,
                "wrong_rate": (wrong / total) if total else None,
                "flat_rate": (flat / total) if total else None,
                "mean_return": float(valid[ret_col].mean()) if total else None,
                "median_return": float(valid[ret_col].median()) if total else None,
                "p10_return": float(valid[ret_col].quantile(0.10)) if total else None,
                "p90_return": float(valid[ret_col].quantile(0.90)) if total else None,
            }
        summary["signals"][sig] = sig_block
    return summary


def simulate_position_equity(df: pd.DataFrame, fee: float = FEE) -> pd.DataFrame:
    out = df.copy()
    signal_to_pos = {"long": 1, "short": -1, "flat": 0, "neutral": 0}
    out["position"] = out["signal"].astype(str).str.lower().map(signal_to_pos).fillna(0).astype(int)
    close = out["close"].astype(float)
    daily_ret = close.pct_change().fillna(0.0)
    pos_prev = out["position"].shift(1).fillna(0)
    strat_ret = pos_prev * daily_ret
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
    summary["equity_simulation"] = {
        "final_equity_coeziv": float(df["equity_coeziv"].iloc[-1]),
        "final_equity_buy_hold": float(df["buy_hold_equity"].iloc[-1]),
        "max_drawdown": float(df["drawdown"].min()),
        "position_changes": int((df["position"].diff().abs().fillna(0) > 0).sum()),
        "note": "Simulare orientativă: poziția din ziua precedentă aplicată pe randamentul zilei curente; cost la schimbare poziție.",
    }
    return summary


def _is_degradation_regime(regime: Any) -> bool:
    return str(regime or "").lower() in DEGRADATION_REGIMES


def current_degradation_streak(df: pd.DataFrame) -> int:
    count = 0
    for _, row in df.iloc[::-1].iterrows():
        if _is_degradation_regime(row.get("regime")):
            count += 1
        else:
            break
    return count


def find_structural_degradation_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["is_degradation_regime"] = out["regime"].apply(_is_degradation_regime)
    out["was_degradation_regime"] = out["is_degradation_regime"].shift(1).fillna(False)
    candidates = out[out["is_degradation_regime"] & (~out["was_degradation_regime"])].copy()
    kept_rows = []
    last_event_date = None
    for ts, row in candidates.iterrows():
        if last_event_date is not None and (ts - last_event_date).days < COOLDOWN_DAYS:
            continue
        kept_rows.append((ts, row))
        last_event_date = ts
    if not kept_rows:
        return out.iloc[0:0].copy()
    return pd.DataFrame([row for _, row in kept_rows], index=[ts for ts, _ in kept_rows])


def analyze_structural_degradation(df: pd.DataFrame) -> Dict[str, Any]:
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
            key = f"{window}d"
            future = close[(close.index > ts) & (close.index <= ts + pd.Timedelta(days=window))]
            if future.empty or entry_price <= 0:
                event_record["windows"][key] = {"available": False, "confirmed": None, "min_drawdown": None, "days_to_threshold": None, "min_price": None}
                continue
            drawdowns = future / entry_price - 1.0
            min_ts = drawdowns.idxmin()
            threshold_hits = drawdowns[drawdowns <= MAJOR_DRAWDOWN_THRESHOLD]
            event_record["windows"][key] = {
                "available": True,
                "confirmed": not threshold_hits.empty,
                "min_drawdown": float(drawdowns.min()),
                "days_to_threshold": None if threshold_hits.empty else int((threshold_hits.index[0] - ts).days),
                "min_price": float(future.loc[min_ts]),
            }
        result["events"].append(event_record)

    for window in DEGRADATION_WINDOWS_DAYS:
        key = f"{window}d"
        available = [e for e in result["events"] if e["windows"][key]["available"]]
        confirmed = [e for e in available if e["windows"][key]["confirmed"] is True]
        min_drawdowns = [e["windows"][key]["min_drawdown"] for e in available]
        days_to_threshold = [e["windows"][key]["days_to_threshold"] for e in confirmed if e["windows"][key]["days_to_threshold"] is not None]
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


def _risk_level(rate: float | None) -> str:
    if rate is None:
        return "unknown"
    if rate >= 0.50:
        return "high"
    if rate >= 0.33:
        return "moderate"
    return "low"


def build_risk_window_card(summary: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    block = summary.get("structural_degradation_events", {})
    window_key = f"{RISK_WINDOW_DAYS}d"
    window = (block.get("windows") or {}).get(window_key, {})
    rate = window.get("confirmation_rate")
    median_days = window.get("median_days_to_threshold")
    avg_days = window.get("avg_days_to_threshold")
    active = _is_degradation_regime(last.get("regime"))

    if active and rate is not None and median_days is not None:
        main_text = (
            f"Contextul actual aparține unei familii istorice care a generat scăderi importante "
            f"în aproximativ {rate * 100:.0f}% din cazuri. Când acestea s-au confirmat, "
            f"confirmarea a apărut în medie după ~{median_days:.0f} zile."
        )
    elif active:
        main_text = "Contextul actual indică degradare structurală, dar eșantionul istoric este insuficient pentru o estimare robustă."
    else:
        main_text = "Contextul curent nu este într-un regim de degradare structurală. Fereastra de risc major nu este activă în acest snapshot."

    return {
        "title": "FEREASTRA_DE_RISC",
        "active": active,
        "level": _risk_level(rate) if active else "normal",
        "current_regime": str(last.get("regime")),
        "current_signal": str(last.get("signal")),
        "consecutive_degradation_days": current_degradation_streak(df),
        "window_days": RISK_WINDOW_DAYS,
        "major_drawdown_threshold": MAJOR_DRAWDOWN_THRESHOLD,
        "historical_available_events": window.get("available_events"),
        "historical_confirmed_events": window.get("confirmed_events"),
        "historical_confirmation_rate": rate,
        "median_days_to_confirmation": median_days,
        "average_days_to_confirmation": avg_days,
        "median_min_drawdown": window.get("median_min_drawdown"),
        "average_min_drawdown": window.get("avg_min_drawdown"),
        "main_text": main_text,
        "footer": "Interpretare statistică a degradării structurale, nu recomandare de tranzacționare.",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }


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
    risk_window = build_risk_window_card(summary, df_bt)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

    export_cols = ["close", "ic_struct", "ic_dir", "ic_flux", "ic_cycle", "regime", "signal", "position", "strategy_return", "equity_coeziv", "buy_hold_equity", "drawdown"]
    for horizon in HORIZONS_DAYS:
        export_cols.extend([f"ret_{horizon}d", f"outcome_{horizon}d"])
    existing_cols = [c for c in export_cols if c in df_bt.columns]
    df_bt[existing_cols].to_csv(OUT_RESULTS, index_label="date")

    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with OUT_RISK_WINDOW.open("w", encoding="utf-8") as f:
        json.dump(risk_window, f, ensure_ascii=False, indent=2)

    print(f"[OK] Backtest coeziv salvat în: {OUT_RESULTS}")
    print(f"[OK] Sumar backtest salvat în: {OUT_SUMMARY}")
    print(f"[OK] Card Fereastră de risc salvat în: {OUT_RISK_WINDOW}")
    print(json.dumps(risk_window, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
