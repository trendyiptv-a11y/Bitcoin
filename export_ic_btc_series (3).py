#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_ic_btc_series.py

Generează seria istorică IC_BTC / ICD_BTC / ICF_BTC etc. pentru front-end,
folosind **exact aceleași date daily** ca snapshot-ul live:
data/btc_daily.csv

Scop:
- să fie aliniat cu modelul coeziv oficial din
  update_btc_state_latest_from_daily.py
- ultimul punct din serie == valorile din btc_state_latest.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import math
from statistics import stdev

import csv

from update_btc_state_latest_from_daily import (
    classify_regime,
    clamp,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INPUT_DAILY = DATA_DIR / "btc_daily.csv"
OUT_PATH = DATA_DIR / "ic_btc_series.json"


# --------- utilitare simple (copiate compatibil cu scriptul oficial) ---------


def ema(series: List[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period trebuie să fie > 0")
    if len(series) == 0:
        return []

    k = 2 / (period + 1.0)
    out: List[Optional[float]] = [None] * len(series)

    if len(series) < period:
        return out

    sma = sum(series[:period]) / period
    out[period - 1] = sma
    prev = sma

    for i in range(period, len(series)):
        price = series[i]
        prev = price * k + prev * (1 - k)
        out[i] = prev

    return out


def rolling_std(series: List[float], window: int) -> List[Optional[float]]:
    if window <= 1:
        raise ValueError("window trebuie să fie > 1")
    n = len(series)
    out: List[Optional[float]] = [None] * n
    if n < window:
        return out

    for i in range(window - 1, n):
        chunk = series[i - window + 1 : i + 1]
        if len(chunk) < 2:
            out[i] = None
        else:
            out[i] = stdev(chunk)  # type: ignore[arg-type]
    return out


def percentile_rank(history: List[float], value: float) -> float:
    """
    Percentila poziției lui `value` într-un istoric.
    0 = cel mai mic, 100 = cel mai mare.
    (identic ca în update_btc_state_latest_from_daily.py)
    """
    if not history:
        return 50.0
    sorted_vals = sorted(history)
    count = 0
    for v in sorted_vals:
        if v <= value:
            count += 1
        else:
            break
    return 100.0 * count / len(sorted_vals)


# --------- citire date BTC din btc_daily.csv (la fel ca scriptul oficial) ---------


def read_btc_daily(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Nu am găsit fișierul {path}")

    dates: List[datetime] = []
    closes: List[float] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            date_idx = header.index("date")
        except ValueError:
            date_idx = header.index("Date")
        try:
            close_idx = header.index("close")
        except ValueError:
            close_idx = header.index("Close")

        for row in reader:
            if not row or not row[date_idx]:
                continue
            d_str = row[date_idx].split()[0]
            d = datetime.fromisoformat(d_str)
            try:
                c = float(row[close_idx])
            except ValueError:
                continue
            dates.append(d)
            closes.append(c)

    combined = sorted(zip(dates, closes), key=lambda x: x[0])
    dates_sorted, closes_sorted = zip(*combined)
    return list(dates_sorted), list(closes_sorted)


# --------- serie coezivă (0–100) pe toată istoria ---------


def build_ic_series() -> Dict[str, Any]:
    dates, closes = read_btc_daily(INPUT_DAILY)
    n = len(closes)
    if n < 260:
        raise RuntimeError("Prea puține date BTC (ai nevoie de ~260 zile minim).")

    # log-returns
    log_ret: List[float] = [0.0]
    for i in range(1, n):
        if closes[i - 1] <= 0 or closes[i] <= 0:
            log_ret.append(0.0)
        else:
            log_ret.append(math.log(closes[i] / closes[i - 1]))

    # EMA-uri & trend_strength (ca în scriptul oficial)
    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200)

    spread: List[float] = []
    for i in range(n):
        if ema50[i] is None or ema200[i] is None:
            spread.append(0.0)
        else:
            spread.append(abs(ema50[i] - ema200[i]))

    vol200 = rolling_std(closes, 200)

    trend_strength: List[Optional[float]] = [None] * n
    for i in range(n):
        if vol200[i] is None or vol200[i] == 0:
            trend_strength[i] = None
        else:
            trend_strength[i] = spread[i] / vol200[i]

    # istoric pentru percentile (toată istoria, ca în snapshot)
    ts_hist: List[float] = [
        v for v in trend_strength if v is not None and v != 0.0
    ]

    # directionalitate – cumulated return pe 60 zile
    window_dir = 60
    cum_ret: List[Optional[float]] = [None] * n
    for i in range(window_dir, n):
        base = closes[i - window_dir]
        if base <= 0:
            cum_ret[i] = None
        else:
            cum_ret[i] = closes[i] / base - 1.0

    cr_hist: List[float] = [v for v in cum_ret if v is not None]

    # volatilitate 30d anualizată
    window_vol = 30
    vol30: List[Optional[float]] = [None] * n
    for i in range(window_vol, n):
        chunk = log_ret[i - window_vol + 1 : i + 1]
        if len(chunk) < 2:
            vol30[i] = None
        else:
            s = stdev(chunk)  # type: ignore[arg-type]
            vol30[i] = s * math.sqrt(365.0) * 100.0

    vol_hist: List[float] = [v for v in vol30 if v is not None]

    series_records: List[Dict[str, Any]] = []

    for i in range(n):
        ts_val = trend_strength[i]
        cr_val = cum_ret[i]
        vol_val = vol30[i]

        # sărim punctele foarte timpurii fără structură/volatilitate definită
        if ts_val is None or cr_val is None or vol_val is None:
            continue

        ic_struct = percentile_rank(ts_hist, ts_val)
        ic_dir = percentile_rank(cr_hist, cr_val)
        vol_index = percentile_rank(vol_hist, vol_val)
        ic_flux = clamp(100.0 - vol_index, 0.0, 100.0)

        regime = classify_regime(ic_struct, ic_dir)

        rec = {
            "t": int(dates[i].timestamp() * 1000),
            "close": float(closes[i]),
            "ic_struct": float(clamp(ic_struct, 0.0, 100.0)),
            "ic_dir": float(clamp(ic_dir, 0.0, 100.0)),
            "ic_flux": float(ic_flux),
            # pentru moment păstrăm ciclul ca 50 fix – ai deja logica macro-ciclu separat
            "ic_cycle": 50.0,
            "vol30_ann_pct": float(vol_val),
            "vol30_index": float(clamp(vol_index, 0.0, 100.0)),
            "regime": regime.code,
            "regime_label": regime.label,
            "regime_short": regime.short,
            "regime_color": regime.color,
        }
        series_records.append(rec)

    meta = {
        "as_of": dates[-1].strftime("%Y-%m-%d"),
        "points": len(series_records),
        "source": "coeziv-btc-official-daily",
    }

    return {"meta": meta, "series": series_records}


def main() -> None:
    data = build_ic_series()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"[Coeziv] Am generat {len(data['series'])} puncte în {OUT_PATH}")


if __name__ == "__main__":
    main()
