#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_global_coeziv_state.py

Model Coeziv extins pentru piața globală de active.

- Folosește seriile din data_global/*.csv (SPX, VIX, DXY, GOLD, OIL),
  descărcate anterior cu update_global_coeziv_state.py.
- Calculează:
    * IC_GLOBAL  (structură)  – coeziunea internă între pieţe (corelaţii)
    * ICD_GLOBAL (direcţie)   – fluxul de risc global (risk-on vs risk-off)
    * energie de fază coezivă (sinus pe 2π)
    * risk_score global (-1 … +1)
    * macro_signal: "risk-on" / "risk-off" / "echilibrat"
    * regim global: "bull" / "bear" / "neutral"

Rezultatul este serializat în data/global_coeziv_state.json
şi este folosit de:
- ic_global.html
- update_btc_state_latest_from_daily.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---- locaţii fişiere --------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_GLOBAL = ROOT / "data_global"
DATA_OUT = ROOT / "data"
OUTPUT_JSON = DATA_OUT / "global_coeziv_state.json"

# numele fişierelor din data_global/*.csv
SERIES = ["spx", "vix", "dxy", "gold", "oil"]

# ferestre temporale (în zile)
WINDOW_STRUCT = 120  # structură / corelaţii
WINDOW_DIR = 60      # direcţionalitate / fluxuri de risc


# ---- utilitare --------------------------------------------------------------

def log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[GlobalCoeziv] {now} | {msg}", flush=True)


def percentile_from_sorted(sorted_vals: List[float], value: float) -> float:
    """Percentilă dintr-o listă sortată (0–100)."""
    if not sorted_vals:
        return 50.0
    import bisect
    pos = bisect.bisect_right(sorted_vals, value)
    return 100.0 * pos / len(sorted_vals)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class GlobalRegime:
    regime: str
    description: str


# ---- citire serii globale ---------------------------------------------------

def load_single_series(name: str) -> pd.Series:
    """
    Încarcă o serie din data_global/<name>.csv cu coloanele:
       timestamp (ms), close (float)
    și o întoarce ca pd.Series indexată pe dată.
    """
    path = DATA_GLOBAL / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Lipsește fișierul {path}")

    df = pd.read_csv(path)

    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{path} trebuie să aibă coloanele 'timestamp' și 'close'")

    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # conversie robustă la float (dacă apare accidental text, e filtrat ca NaN)
    close = pd.to_numeric(df["close"], errors="coerce")
    s = pd.Series(close.values, index=ts).dropna()

    return s.sort_index()


def load_all_series() -> pd.DataFrame:
    """
    Concatenează seriile într-un singur DataFrame:
       index = dată, coloane = ['spx', 'vix', 'dxy', 'gold', 'oil']
    doar pe intervalul comun tuturor seriilor.
    """
    log("Încarc seriile globale din data_global/ ...")
    data: Dict[str, pd.Series] = {}
    for name in SERIES:
        data[name] = load_single_series(name)

    df = pd.concat(data, axis=1, join="inner").dropna()
    if len(df) < 260:
        raise RuntimeError(
            "Prea puține date comune pentru seriile globale (min ~260 zile)."
        )

    log(f"Interval comun: {df.index[0].date()} – {df.index[-1].date()} ({len(df)} zile)")
    return df


# ---- IC_GLOBAL structură (coeziune între pieţe) ------------------------------

def compute_ic_global_structural(df: pd.DataFrame) -> pd.Series:
    """
    IC_GLOBAL: măsoară coeziunea structurală dintre pieţele de active
    folosind media corelaţiilor absolute dintre randamentele zilnice,
    pe o fereastră rulantă de WINDOW_STRUCT zile, normalizată pe 0–100.
    """
    rets = df.pct_change().dropna()
    assets = list(df.columns)

    if len(rets) < WINDOW_STRUCT:
        raise RuntimeError("Insuficiente date pentru fereastra structurală.")

    log(f"Calculez IC_GLOBAL structural (corelații, {WINDOW_STRUCT} zile)...")

    num_assets = len(assets)
    pair_count = num_assets * (num_assets - 1) // 2

    ic_raw = pd.Series(index=rets.index, dtype=float)
    ic_raw[:] = 0.0

    # sumă de |corr| pe toate perechile
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            a, b = assets[i], assets[j]
            c = rets[a].rolling(WINDOW_STRUCT).corr(rets[b]).abs()
            ic_raw = ic_raw.add(c, fill_value=0.0)

    ic_raw = (ic_raw / float(pair_count)).dropna()

    hist_vals = [float(x) for x in ic_raw.values if np.isfinite(x)]
    sorted_hist = sorted(hist_vals)

    ic_idx_vals: Dict[pd.Timestamp, float] = {}
    for ts, v in ic_raw.items():
        if not np.isfinite(v):
            continue
        p = percentile_from_sorted(sorted_hist, float(v))
        ic_idx_vals[ts] = clamp(p, 0.0, 100.0)

    ic_index = pd.Series(ic_idx_vals).sort_index()
    ic_index.name = "ic_global"
    return ic_index


# ---- ICD_GLOBAL direcțional (flux de risc) -----------------------------------

def compute_icd_global_directional(df: pd.DataFrame) -> pd.Series:
    """
    ICD_GLOBAL: direcţionalitatea globală (bias de risc) definită ca
    randament cumulativ pe 60 de zile al unui coş:
      + SPX, GOLD, OIL  (risk-on / active ciclice)
      - VIX, DXY        (refugii / risk-off)
    Normalizăm apoi în percentilă 0–100.
    """
    if len(df) < WINDOW_DIR + 1:
        raise RuntimeError("Insuficiente date pentru fereastra direcțională.")

    log(f"Calculez ICD_GLOBAL direcțional (randamente {WINDOW_DIR} zile)...")

    cum = {}
    for name in SERIES:
        series = df[name]
        base = series.shift(WINDOW_DIR)
        cum[name] = series / base - 1.0

    cum_df = pd.DataFrame(cum).dropna()

    # ponderi pentru coşul de risc global
    w_spx, w_gold, w_oil, w_vix, w_dxy = 0.4, 0.15, 0.15, 0.15, 0.15

    dir_raw = (
        w_spx * cum_df["spx"]
        + w_gold * cum_df["gold"]
        + w_oil * cum_df["oil"]
        - w_vix * cum_df["vix"]
        - w_dxy * cum_df["dxy"]
    ).dropna()

    hist_vals = [float(x) for x in dir_raw.values if np.isfinite(x)]
    sorted_hist = sorted(hist_vals)

    icd_vals: Dict[pd.Timestamp, float] = {}
    for ts, v in dir_raw.items():
        if not np.isfinite(v):
            continue
        p = percentile_from_sorted(sorted_hist, float(v))
        icd_vals[ts] = clamp(p, 0.0, 100.0)

    icd_index = pd.Series(icd_vals).sort_index()
    icd_index.name = "icd_global"
    return icd_index


# ---- praguri dinamice & fază coezivă ----------------------------------------

def dynamic_thresholds(ic_series: pd.Series, icd_series: pd.Series) -> Dict[str, float]:
    """
    Praguri nu sunt hardcodate, ci extrase din distribuţia istorică.
    Folosim percentile 30 / 70 pentru IC şi ICD.
    """
    ic_vals = np.array(ic_series.values, dtype=float)
    icd_vals = np.array(icd_series.values, dtype=float)

    return {
        "ic_low": float(np.percentile(ic_vals, 30)),
        "ic_high": float(np.percentile(ic_vals, 70)),
        "icd_low": float(np.percentile(icd_vals, 30)),
        "icd_high": float(np.percentile(icd_vals, 70)),
    }


def coeziv_phase(ic: float, icd: float) -> float:
    """
    Fază coezivă globală în [0, 2π], derivată din structură & direcţionalitate.
    Folosim produsul normalizat al indicilor ca „densitate de coeziune”.
    """
    ic_n = clamp(ic / 100.0, 0.0, 1.0)
    icd_n = clamp(icd / 100.0, 0.0, 1.0)
    # fază 0…2π (π şi 2π din modelul coeziv)
    return 2.0 * np.pi * (ic_n * icd_n)


def coeziv_energy(ic: float, icd: float) -> float:
    """
    Energia de fază coezivă – analog „apa din celulă”:
      E = sin(phase)
    E > 0   → expansiune / risk-on
    E < 0   → contracţie / risk-off
    """
    phase = coeziv_phase(ic, icd)
    return float(np.sin(phase))


def classify_global_regime_coeziv(ic_val: float, icd_val: float) -> GlobalRegime:
    """
    Regim global în modelul coeziv:
      - bull   : energie > +0.35
      - bear   : energie < -0.35
      - neutral: altfel
    """
    energy = coeziv_energy(ic_val, icd_val)

    if energy > 0.35:
        return GlobalRegime(
            regime="bull",
            description="Fază globală de expansiune coezivă: structură ridicată + fluxuri pro-risc."
        )
    if energy < -0.35:
        return GlobalRegime(
            regime="bear",
            description="Fază globală de contracție coezivă: tendință defensivă / orientare spre refugii."
        )
    return GlobalRegime(
        regime="neutral",
        description="Zonă de echilibru fazal / tranziție: structură & direcționalitate amestecate."
    )


def compute_risk_score_and_macro(ic_val: float, icd_val: float) -> Tuple[float, str]:
    """
    Risk score global derivat direct din energia de fază coezivă:
        risk_score = sin(phase) ∈ [-1, +1]
    """
    energy = coeziv_energy(ic_val, icd_val)
    risk_score = clamp(energy, -1.0, 1.0)

    if risk_score > 0.2:
        macro_signal = "risk-on"
    elif risk_score < -0.2:
        macro_signal = "risk-off"
    else:
        macro_signal = "echilibrat"

    return risk_score, macro_signal


# ---- orchestrare ------------------------------------------------------------

def main() -> None:
    log("Pornesc build_global_coeziv_state.py (model coeziv extins)")

    df = load_all_series()
    ic_series = compute_ic_global_structural(df)
    icd_series = compute_icd_global_directional(df)

    # aliniază pe acelaşi index
    common_index = ic_series.index.intersection(icd_series.index)
    ic_series = ic_series.loc[common_index]
    icd_series = icd_series.loc[common_index]

    if not len(common_index):
        raise RuntimeError("Nu există intersecție de date IC/ICD.")

    records: List[Dict[str, float]] = []
    for ts in common_index:
        t_ms = int(ts.timestamp() * 1000)
        ic_val = float(ic_series.loc[ts])
        icd_val = float(icd_series.loc[ts])
        records.append(
            {
                "t": t_ms,
                "ic_global": round(ic_val, 2),
                "icd_global": round(icd_val, 2),
            }
        )

    # sortare pentru siguranţă
    records.sort(key=lambda x: x["t"])

    # punctul curent = ultima zi
    current = records[-1]
    ic_val, icd_val = current["ic_global"], current["icd_global"]

    # regim + risk score + macro signal în model coeziv
    regime = classify_global_regime_coeziv(ic_val, icd_val)
    risk_score, macro_signal = compute_risk_score_and_macro(ic_val, icd_val)
    current["regime"] = regime.regime

    # praguri dinamice extrase din istoric
    thresholds = dynamic_thresholds(ic_series, icd_series)

    as_of_date = datetime.fromtimestamp(
        current["t"] / 1000, tz=timezone.utc
    ).strftime("%Y-%m-%d")

    payload = {
        "as_of": as_of_date,
        "risk_score": round(risk_score, 4),
        "macro_signal": macro_signal,
        "series": records,
        "current": current,
        "thresholds": thresholds,
        "meta": {
            "source": "coeziv-global-python",
            "window_struct_days": WINDOW_STRUCT,
            "window_dir_days": WINDOW_DIR,
            "description": (
                "IC_GLOBAL structural din corelații multi-asset; "
                "ICD_GLOBAL direcțional din randamente cumulative risk-on/risk-off; "
                "risk_score derivat din faza coezivă sin(2π·IC·ICD)."
            ),
        },
    }

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log(f"Am scris {len(records)} puncte în {OUTPUT_JSON.relative_to(ROOT)}")
    log(
        f"Stare curentă: IC_GLOBAL={ic_val:.2f}, "
        f"ICD_GLOBAL={icd_val:.2f}, regim={regime.regime}, "
        f"risk_score={risk_score:.3f}, macro_signal={macro_signal}"
    )


if __name__ == "__main__":
    main()
