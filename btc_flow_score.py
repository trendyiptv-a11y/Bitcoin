import os
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

# Presupunem aceeași structură de directoare ca în coeziv_state.py
BASE_DIR = os.path.dirname(__file__)
STRATEGY_DIR = os.path.join(BASE_DIR, "btc-swing-strategy")


# -----------------------------
#  ÎNCĂRCARE OHLC + VOLUM (INTRADAY)
# -----------------------------

def load_intraday_ohlc(path: Optional[str] = None) -> pd.DataFrame:
    """
    Încărcăm seriile intraday BTC pentru calculul Flow Score.

    Caută fișierul în mai multe locații posibile:
    - path explicit (dacă este dat)
    - ./j_btc_series.csv
    - ./data/j_btc_series.csv
    - ./btc-swing-strategy/j_btc_series.csv

    Așteptări minimum pentru coloane:
    - 'timestamp' sau 'date' (timpul)
    - 'open', 'high', 'low', 'close'
    - 'volume' (sau 'vol')

    Returnează un DataFrame indexat pe datetime, sortat crescător.
    """
    candidates = []

    if path is not None:
        candidates.append(path)
    else:
        candidates.extend([
            os.path.join(BASE_DIR, "j_btc_series.csv"),
            os.path.join(BASE_DIR, "data", "j_btc_series.csv"),
            os.path.join(STRATEGY_DIR, "j_btc_series.csv"),
        ])

    chosen = None
    for c in candidates:
        if os.path.exists(c):
            chosen = c
            break

    if chosen is None:
        raise FileNotFoundError(
            "Nu am găsit j_btc_series.csv în niciuna din locațiile așteptate:\n"
            + "\n".join(candidates)
        )

    df = pd.read_csv(chosen)

    # Detectăm coloana de timp
    time_col = None
    for cand in ["timestamp", "time", "date", "datetime"]:
        if cand in df.columns:
            time_col = cand
            break

    if time_col is None:
        raise ValueError("Nu am găsit coloană de timp (timestamp/date) în j_btc_series.csv")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

    # Normalizăm numele coloanelor la lower
    df.columns = [c.lower() for c in df.columns]

    # Asigurăm coloane critice
    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Nu am găsit coloana obligatorie '{col}' în j_btc_series.csv")

    # volum – încercăm mai multe denumiri
    volume_col = None
    for cand in ["volume", "vol", "quote_volume"]:
        if cand in df.columns:
            volume_col = cand
            break

    if volume_col is None:
        raise ValueError("Nu am găsit o coloană de volum (volume/vol/quote_volume).")

    df = df[["open", "high", "low", "close", volume_col]].rename(columns={volume_col: "volume"})

    return df


# -----------------------------
#  COMPONENTE FLOW
# -----------------------------

def _safe_tanh(x: float, cap: float = 3.0) -> float:
    """
    Aplica tanh pe un x limitat în [-cap, cap], ca să evităm valori extreme.
    Întoarce un scor în [-1, 1].
    """
    x_clipped = max(-cap, min(cap, x))
    return float(np.tanh(x_clipped))


def _momentum_component(df: pd.DataFrame, hours_short: int = 24) -> float:
    """
    Componentă de momentum pe termen scurt:
    - ultimul raport close_now / close_în_urmă cu 'hours_short' ore
    - normalizat de volatilitatea recentă
    """
    if len(df) < hours_short + 10:
        return 0.0

    # Presupunem date orare – dacă nu, este totuși un lookback relativ scurt.
    last_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-hours_short - 1]

    ret = (last_close / prev_close) - 1.0

    # volatilitate recentă (std anualizată aproximativ)
    ret_series = df["close"].pct_change().dropna()
    recent_vol = ret_series.tail(hours_short).std()

    if recent_vol is None or recent_vol == 0 or np.isnan(recent_vol):
        norm_ret = 0.0
    else:
        norm_ret = ret / (recent_vol * np.sqrt(hours_short))

    return _safe_tanh(norm_ret)


def _volume_component(df: pd.DataFrame, hours_short: int = 24, days_long: int = 30) -> float:
    """
    Componentă de volum relativ:
    - volum în ultimele 'hours_short' ore vs. volum mediu per 24h
      în ultimele 'days_long' zile.
    """
    if len(df) < 24 * 5:  # măcar câteva zile
        return 0.0

    # volum ultimele 24h
    vol_short = df["volume"].tail(hours_short).sum()

    # aproximăm „zile” în pași de 24 de bare
    bars_per_day = 24
    needed_bars = days_long * bars_per_day
    if len(df) < needed_bars:
        hist = df["volume"]
    else:
        hist = df["volume"].tail(needed_bars)

    # volum agregat per zi
    # tăiem la multiplu de 24
    usable = hist.tail((len(hist) // bars_per_day) * bars_per_day)
    daily = usable.values.reshape(-1, bars_per_day).sum(axis=1)

    if len(daily) == 0:
        return 0.0

    median_daily = float(np.median(daily))

    if median_daily <= 0:
        return 0.0

    ratio = (vol_short / median_daily) - 1.0  # >0 înseamnă volum peste medie
    # normalizăm prin tanh
    return _safe_tanh(ratio)


def _range_position_component(df: pd.DataFrame, days_long: int = 30) -> float:
    """
    Componentă de poziție în range:
    - folosim ultimul close vs. min/max pe ultimele 'days_long' zile.
    - întoarcem scor în [-1, 1], unde:
        -1 ~ josul range-ului (context de potențială acumulare)
        +1 ~ vârful range-ului (context de distribuție)
    """
    bars_per_day = 24
    needed_bars = days_long * bars_per_day

    if len(df) < needed_bars:
        window = df["close"]
    else:
        window = df["close"].tail(needed_bars)

    c_now = window.iloc[-1]
    c_min = window.min()
    c_max = window.max()

    if c_max == c_min:
        return 0.0

    pos = (c_now - c_min) / (c_max - c_min)  # 0 jos, 1 sus
    # mapăm [0,1] -> [-1,1]
    return float(pos * 2.0 - 1.0)


# -----------------------------
#  FLOW SCORE – AGREGAȚIE
# -----------------------------

def compute_flow_score(df: pd.DataFrame) -> Dict[str, object]:
    """
    Calculează Flow Score-ul pe baza unui DataFrame intraday cu:
    - open, high, low, close, volume

    Returnează un dict cu:
    - 'flow_score' (0-100)
    - 'flow_bias' ('pozitiv' / 'negativ' / 'neutru')
    - 'flow_strength' ('slab' / 'moderat' / 'puternic')
    - 'components' (dict cu valorile brute ale componentelor)
    """
    # componente brute în [-1,1]
    m = _momentum_component(df)
    v = _volume_component(df)
    r = _range_position_component(df)

    # ponderi – poți ajusta ulterior
    w_m = 0.5   # momentum
    w_v = 0.3   # volum
    w_r = 0.2   # poziție în range

    flow_raw = w_m * m + w_v * v + w_r * r  # încă în [-1,1] aproximativ

    # mapare la [0,100]
    flow_score = int(round((flow_raw + 1.0) * 50.0))  # -1 -> 0, 0 -> 50, +1 -> 100
    flow_score = max(0, min(100, flow_score))

    # bias
    if flow_raw > 0.15:
        flow_bias = "pozitiv"
    elif flow_raw < -0.15:
        flow_bias = "negativ"
    else:
        flow_bias = "neutru"

    # forță
    abs_raw = abs(flow_raw)
    if abs_raw < 0.25:
        flow_strength = "slab"
    elif abs_raw < 0.6:
        flow_strength = "moderat"
    else:
        flow_strength = "puternic"

    return {
        "flow_score": flow_score,
        "flow_bias": flow_bias,
        "flow_strength": flow_strength,
        "components": {
            "momentum": m,
            "volume": v,
            "range_position": r,
            "raw": flow_raw,
        },
    }


def compute_flow_from_file(path: Optional[str] = None) -> Dict[str, object]:
    """
    Helper simplu: încarcă intraday OHLC + volum și calculează Flow Score.
    """
    df = load_intraday_ohlc(path=path)
    return compute_flow_score(df)
