#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
update_btc_state_latest_from_daily.py

Script Coeziv: calculează starea BTC pe baza datelor daily
și salvează un snapshot "btc_state_latest.json" folosit în front-end.

Pași (backend pur, fără dependențe exotice):
- citește data/btc_daily.csv (minim coloanele: date, close)
- calculează log-return-uri, EMA50 / EMA200, volatilitate pe 30 de zile
- derivă indici IC_BTC structural & ICD_BTC direcțional (0–100)
  folosind normalizare pe istoric (percentile)
- clasifică regimul coeziv (acumulare, bull structural, bear structural etc.)
- sintetizează un context scurt (trend, volatilitate, macro)
- scrie rezultatul în data/btc_state_latest.json

Important: nu modifică alte fișiere și nu descarcă date noi.
Rulezi scriptul după ce ai actualizat deja btc_daily.csv.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
DATA_BTC = ROOT / "data"
DATA_GLOBAL = ROOT / "data_global"
INPUT_DAILY = DATA_BTC / "btc_daily.csv"
OUTPUT_STATE = DATA_BTC / "btc_state_latest.json"
GLOBAL_STATE_FILE = DATA_GLOBAL / "global_coeziv_state.json"


# ---------- utilitare numerice ----------

def ema(series: List[float], period: int) -> List[Optional[float]]:
    """
    EMA clasică. Returnează o listă de aceeași lungime cu valori None la început
    până când se umple fereastra.
    """
    if period <= 0:
        raise ValueError("period trebuie să fie > 0")
    if len(series) == 0:
        return []

    k = 2 / (period + 1.0)
    out: List[Optional[float]] = [None] * len(series)

    # seed = media simplă pe primele `period` valori
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
    """
    Deviație standard rulantă (similar cu pandas.Series.rolling.std).
    """
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
    """
    if not history:
        return 50.0
    sorted_vals = sorted(history)
    # număr de valori <= value
    count = 0
    for v in sorted_vals:
        if v <= value:
            count += 1
        else:
            break
    return 100.0 * count / len(sorted_vals)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------- structuri pentru regim ----------

@dataclass
class Regime:
    code: str
    label: str
    short: str
    color: str  # doar pentru front-end: "green", "red", "orange", etc.


def classify_regime(ic_struct: float, ic_dir: float) -> Regime:
    """
    Clasificare coezivă a regimului BTC pe baza indicilor structurali & direcționali.

    Convenție:
    - IC struct < 20  => structură foarte slabă / bază
    - 20–60          => tranziție / trend în formare
    - > 60           => trend structural puternic / fază superioară
    - ICD < 45       => bias bear
    - 45–55          => mixt / de tranziție
    - > 55           => bias bull
    """
    struct = ic_struct
    dir_ = ic_dir

    if struct < 20 and dir_ < 45:
        return Regime(
            code="accum_bear",
            label="Acumulare bearish / bază descendentă",
            short="Structură foarte slabă, cu flux ușor orientat în jos.",
            color="red",
        )
    if struct < 20 and dir_ > 55:
        return Regime(
            code="accum_bull",
            label="Acumulare bullish / bază ascendentă",
            short="Bază slabă, dar cu bias ușor pozitiv al fluxului.",
            color="green",
        )
    if 20 <= struct < 60 and dir_ > 55:
        return Regime(
            code="bull_struct",
            label="Bull structural",
            short="Trend ascendent în formare / consolidare structurală.",
            color="green",
        )
    if 20 <= struct < 60 and dir_ < 45:
        return Regime(
            code="bear_struct",
            label="Bear structural",
            short="Trend descendent în formare / structură în răcire.",
            color="red",
        )
    if struct >= 60 and dir_ > 55:
        return Regime(
            code="bull_late",
            label="Bull târziu / început de top structural",
            short="Structură puternic ascendentă, dar matură, cu risc de epuizare.",
            color="orange",
        )
    if struct >= 60 and dir_ < 45:
        return Regime(
            code="bear_late",
            label="Bear târziu / capitulare",
            short="Structură descendentă avansată, cu risc de mișcări extreme.",
            color="orange",
        )
    # fallback – când ICD ~ 50 sau struct între zone
    return Regime(
        code="mixed",
        label="Regim mixt / de tranziție",
        short="Configurație neclară: structură și direcționalitate amestecate.",
        color="grey",
    )


# ---------- citire date BTC ----------

def read_btc_daily(path: Path) -> Tuple[List[datetime], List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Nu am găsit fișierul {path}")

    dates: List[datetime] = []
    closes: List[float] = []

    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        # căutăm coloanele de interes
        try:
            date_idx = header.index("date")
        except ValueError:
            date_idx = header.index("Date")  # alternativă
        try:
            close_idx = header.index("close")
        except ValueError:
            close_idx = header.index("Close")

        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            try:
                d = datetime.fromisoformat(parts[date_idx])
            except Exception:
                # suport și pentru format gen 2025-01-31 00:00:00
                d = datetime.strptime(parts[date_idx].split()[0], "%Y-%m-%d")
            try:
                c = float(parts[close_idx])
            except ValueError:
                continue
            dates.append(d)
            closes.append(c)

    # sortăm în caz că nu sunt deja ordonate
    combined = sorted(zip(dates, closes), key=lambda x: x[0])
    dates_sorted, closes_sorted = zip(*combined)
    return list(dates_sorted), list(closes_sorted)


# ---------- calcule pentru IC / volatilitate ----------

def compute_state_from_prices(
    dates: List[datetime], closes: List[float]
) -> Dict[str, float]:
    if len(dates) != len(closes):
        raise ValueError("dates și closes trebuie să aibă aceeași lungime")
    if len(closes) < 260:
        raise RuntimeError("Prea puține date BTC pentru a calcula indicii (~260 zile).")

    # log-return-uri
    log_ret: List[float] = [0.0]
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0 or closes[i] <= 0:
            log_ret.append(0.0)
        else:
            log_ret.append(math.log(closes[i] / closes[i - 1]))

    # EMA-uri pentru structură
    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200)

    # "trend_strength" ca distanță EMA50–EMA200 raportată la volatilitatea pe 200 zile
    spread: List[float] = []
    for i in range(len(closes)):
        if ema50[i] is None or ema200[i] is None:
            spread.append(0.0)
        else:
            spread.append(abs(ema50[i] - ema200[i]))

    vol200 = rolling_std(closes, 200)
    trend_strength_hist: List[float] = []
    for i in range(len(closes)):
        if vol200[i] is None or vol200[i] == 0:
            continue
        trend_strength_hist.append(spread[i] / vol200[i])

    if not trend_strength_hist:
        raise RuntimeError("Nu am putut calcula trend_strength_hist.")

    latest_trend_strength = trend_strength_hist[-1]
    ic_struct = percentile_rank(trend_strength_hist, latest_trend_strength)

    # directionalitate: folosim randamentul cumulat pe 60 de zile, normalizat pe istoric
    window_dir = 60
    cum_ret_hist: List[float] = []
    for i in range(window_dir, len(closes)):
        base = closes[i - window_dir]
        if base <= 0:
            continue
        cum_ret_hist.append(closes[i] / base - 1.0)

    if not cum_ret_hist:
        raise RuntimeError("Nu am putut calcula istoric pentru ICD_BTC.")

    latest_base = closes[-window_dir]
    latest_cum_ret = closes[-1] / latest_base - 1.0
    pct = percentile_rank(cum_ret_hist, latest_cum_ret)
    # transformăm percentila 0–100 într-un index 0–100, dar centrat în jurul lui 50
    ic_dir = pct

    # volatilitate pe 30 de zile (annualizată, %)
    if len(log_ret) < 30:
        raise RuntimeError("Prea puține date pentru volatilitate 30d.")
    vol30_hist: List[float] = []
    window_vol = 30
    for i in range(window_vol, len(log_ret)):
        chunk = log_ret[i - window_vol + 1 : i + 1]
        if len(chunk) < 2:
            continue
        s = stdev(chunk)  # type: ignore[arg-type]
        vol30_hist.append(s * math.sqrt(365.0) * 100.0)

    latest_chunk = log_ret[-window_vol:]
    latest_std = stdev(latest_chunk)  # type: ignore[arg-type]
    latest_vol30 = latest_std * math.sqrt(365.0) * 100.0

    vol30_index = percentile_rank(vol30_hist, latest_vol30)

    return {
        "ic_struct": clamp(ic_struct, 0.0, 100.0),
        "ic_dir": clamp(ic_dir, 0.0, 100.0),
        "vol30_ann_pct": max(latest_vol30, 0.0),
        "vol30_index": clamp(vol30_index, 0.0, 100.0),
    }


# ---------- integrare cu macro global (opțional) ----------

def read_global_macro_state(path: Path) -> Dict[str, Optional[float]]:
    """
    Încearcă să citească scorul de risc global & semnal macro
    din fișierul generat de scriptul global (dacă există).
    Structura exactă poate fi adaptată ușor – aici folosim un format generic:
    {
        "as_of": "...",
        "risk_score": float (-1..1),
        "macro_signal": "echilibrat" / "risk-on" / "risk-off"
    }
    """
    if not path.exists():
        return {"risk_score": None}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"risk_score": None}

    # suportăm fie obiect simplu, fie listă de snapshot-uri
    if isinstance(data, list) and data:
        last = data[-1]
    elif isinstance(data, dict):
        last = data
    else:
        return {"risk_score": None}

    risk = last.get("risk_score")
    macro_signal = last.get("macro_signal") or last.get("macro_label")
    return {
        "risk_score": float(risk) if isinstance(risk, (int, float)) else None,
        "macro_signal": macro_signal,
    }


# ---------- asamblare snapshot & salvare ----------

def build_context_short(
    ic_struct: float,
    ic_dir: float,
    vol30_ann_pct: float,
    vol30_index: float,
    global_macro: Dict[str, Optional[float]],
) -> Dict[str, Dict[str, str]]:
    # Trend
    if ic_struct < 20:
        trend_txt = f"Trend: structură slabă (IC {ic_struct:.0f}), bias {'bear' if ic_dir < 50 else 'bull'} (ICD {ic_dir:.0f})."
        trend_color = "red" if ic_dir < 50 else "orange"
    elif ic_struct < 60:
        trend_txt = f"Trend: zonă de tranziție (IC {ic_struct:.0f}), bias direcțional {('bull' if ic_dir > 50 else 'bear')} (ICD {ic_dir:.0f})."
        trend_color = "orange"
    else:
        trend_txt = f"Trend: structură ridicată (IC {ic_struct:.0f}), bias {'bull' if ic_dir >= 50 else 'bear'} (ICD {ic_dir:.0f})."
        trend_color = "green" if ic_dir >= 50 else "red"

    # Volatilitate
    if vol30_index < 33:
        vol_label = "scăzută"
        vol_color = "green"
    elif vol30_index < 66:
        vol_label = "moderată"
        vol_color = "orange"
    else:
        vol_label = "ridicată"
        vol_color = "red"
    vol_txt = f"Volatilitate relativă: {vol_label} (vol30 ~ {vol30_ann_pct:.1f}%)."

    # Macro
    risk_score = global_macro.get("risk_score")
    macro_signal = global_macro.get("macro_signal") or "–"
    if risk_score is None:
        macro_txt = f"Macro (scurt): {macro_signal}."
        macro_color = "grey"
    else:
        if risk_score > 0.15:
            macro_color = "green"
        elif risk_score < -0.15:
            macro_color = "red"
        else:
            macro_color = "orange"
        macro_txt = f"Macro (scurt): {macro_signal or 'echilibrat'} (scor risc {risk_score:.2f})."

    return {
        "trend": {"text": trend_txt, "color": trend_color},
        "volatility": {"text": vol_txt, "color": vol_color},
        "macro": {"text": macro_txt, "color": macro_color},
    }


def main() -> None:
    dates, closes = read_btc_daily(INPUT_DAILY)
    metrics = compute_state_from_prices(dates, closes)
    ic_struct = metrics["ic_struct"]
    ic_dir = metrics["ic_dir"]
    vol30_ann_pct = metrics["vol30_ann_pct"]
    vol30_index = metrics["vol30_index"]

    regime = classify_regime(ic_struct, ic_dir)
    global_macro = read_global_macro_state(GLOBAL_STATE_FILE)
    context_short = build_context_short(
        ic_struct, ic_dir, vol30_ann_pct, vol30_index, global_macro
    )

    last_date = dates[-1]
    last_close = closes[-1]

    state = {
        "as_of": last_date.strftime("%Y-%m-%d"),
        "close": round(last_close, 2),
        # indici coezivi
        "ic_struct": round(ic_struct, 2),
        "ic_dir": round(ic_dir, 2),
        "vol30_ann_pct": round(vol30_ann_pct, 2),
        "vol30_index": round(vol30_index, 2),
        # regim
        "regime_code": regime.code,
        "regime_label": regime.label,
        "regime_short": regime.short,
        "regime_color": regime.color,
        # context scurt pentru front-end
        "context_short": context_short,
    }

    OUTPUT_STATE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_STATE.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print(f"[Coeziv] Am salvat starea BTC în {OUTPUT_STATE}")


if __name__ == "__main__":
    main()
