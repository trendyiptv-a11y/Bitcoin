import os
from typing import Optional, Dict

import numpy as np
import pandas as pd

# Folosim aceeași structură de directoare ca în coeziv_state.py
BASE_DIR = os.path.dirname(__file__)
STRATEGY_DIR = os.path.join(BASE_DIR, "btc-swing-strategy")

# Refolosim loader-ul de OHLCV din btc_flow_score.py
from btc_flow_score import load_intraday_ohlc


# -----------------------------
#  UTILITARE
# -----------------------------

def _safe_tanh(x: float, cap: float = 3.0) -> float:
    """
    Aplica tanh pe un x limitat în [-cap, cap], pentru a evita valori extreme.
    Întoarce un scor în [-1, 1].
    """
    x_clipped = max(-cap, min(cap, x))
    return float(np.tanh(x_clipped))


# -----------------------------
#  COMPONENTE DE LICHIDITATE
# -----------------------------

def _range_component_liq(df: pd.DataFrame,
                         hours_short: int = 24,
                         days_long: int = 30) -> float:
    """
    Componentă bazată pe range intrabar:

    - calculăm media (high - low) / close pentru ultimele 'hours_short' bare,
      proxy pentru „cât se mișcă prețul pe fiecare bară”.
    - comparăm cu media istorică pe ultimele 'days_long' zile.

    Range mai mic decât normal => lichiditate mai bună (scor pozitiv).
    Range mai mare => lichiditate mai slabă (scor negativ).
    """
    if len(df) < hours_short + 10:
        return 0.0

    bars_per_day = 24
    needed_bars = days_long * bars_per_day

    # short window
    short = df.tail(hours_short)
    short_range_pct = ((short["high"] - short["low"]) / short["close"]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(short_range_pct) == 0:
        return 0.0
    short_mean = float(short_range_pct.mean())

    # long window
    if len(df) < needed_bars:
        long = df
    else:
        long = df.tail(needed_bars)

    long_range_pct = ((long["high"] - long["low"]) / long["close"]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(long_range_pct) == 0:
        return 0.0
    long_mean = float(long_range_pct.mean())
    long_std = float(long_range_pct.std()) if len(long_range_pct) > 5 else 0.0

    if long_std == 0 or np.isnan(long_std):
        # fallback: doar raport simplu
        ratio = (long_mean - short_mean) / long_mean  # range mai mic => pozitiv
        return _safe_tanh(ratio)

    # z-score: cât de „sub normal” e range-ul actual
    z = (long_mean - short_mean) / long_std
    return _safe_tanh(z)


def _volume_density_component_liq(df: pd.DataFrame,
                                  hours_short: int = 24,
                                  days_long: int = 30) -> float:
    """
    Componentă bazată pe densitatea volumului:

    - volum per unitate de range: volume / max(high - low, eps)
    - densitate mare => mult volum pentru un range relativ mic => lichiditate bună.
    """
    if len(df) < hours_short + 10:
        return 0.0

    eps = 1e-8
    bars_per_day = 24
    needed_bars = days_long * bars_per_day

    # short window
    short = df.tail(hours_short).copy()
    short_range = (short["high"] - short["low"]).abs().replace(0, eps)
    short_density = (short["volume"] / short_range).replace([np.inf, -np.inf], np.nan).dropna()
    if len(short_density) == 0:
        return 0.0
    short_med = float(np.median(short_density))

    # long window
    if len(df) < needed_bars:
        long = df
    else:
        long = df.tail(needed_bars)

    long_range = (long["high"] - long["low"]).abs().replace(0, eps)
    long_density = (long["volume"] / long_range).replace([np.inf, -np.inf], np.nan).dropna()
    if len(long_density) == 0:
        return 0.0
    long_med = float(np.median(long_density))
    long_std = float(long_density.std()) if len(long_density) > 5 else 0.0

    if long_std == 0 or np.isnan(long_std):
        ratio = (short_med / long_med) - 1.0  # >0 => densitate peste normal
        return _safe_tanh(ratio)

    z = (short_med - long_med) / long_std
    return _safe_tanh(z)


def _micro_vol_component_liq(df: pd.DataFrame,
                             hours_short: int = 24,
                             days_long: int = 30) -> float:
    """
    Componentă bazată pe micro-volatilitate:

    - folosim pct_change la close pe ultimele 'hours_short' bare
    - comparăm std-ul cu std-ul istoric pe 'days_long' zile.

    Volatilitate sub normal => piață „smooth” => lichiditate bună (pozitiv).
    Volatilitate peste normal => lichiditate mai slabă (negativ).
    """
    if len(df) < hours_short + 10:
        return 0.0

    bars_per_day = 24
    needed_bars = days_long * bars_per_day

    ret = df["close"].pct_change().dropna()
    if len(ret) < hours_short + 10:
        return 0.0

    short_vol = float(ret.tail(hours_short).std())
    if len(ret) < needed_bars:
        long_ret = ret
    else:
        long_ret = ret.tail(needed_bars)

    long_vol = float(long_ret.std())
    if long_vol == 0 or np.isnan(long_vol):
        return 0.0

    # volatilitate sub normal => pozitiv (mai lichid)
    z = (long_vol - short_vol) / long_vol
    return _safe_tanh(z)


# -----------------------------
#  LIQUIDITY SCORE – AGREGAȚIE
# -----------------------------

def compute_liquidity_score(df: pd.DataFrame) -> Dict[str, object]:
    """
    Calculează Liquidity Score pe baza unui DataFrame intraday cu:
    - open, high, low, close, volume

    Returnează:
    - liquidity_score (0-100)
    - liquidity_regime ('scăzută' / 'normală' / 'ridicată')
    - liquidity_strength ('slabă' / 'moderată' / 'puternică')
    - components (dict cu valorile brute ale componentelor)
    """
    r = _range_component_liq(df)
    d = _volume_density_component_liq(df)
    mv = _micro_vol_component_liq(df)

    # Ponderi – le poți ajusta ulterior
    w_r = 0.4   # range relativ
    w_d = 0.4   # densitatea volumului
    w_mv = 0.2  # micro-volatilitate

    liq_raw = w_r * r + w_d * d + w_mv * mv  # aproximativ în [-1, 1]

    # mapăm la [0,100]
    liquidity_score = int(round((liq_raw + 1.0) * 50.0))
    liquidity_score = max(0, min(100, liquidity_score))

    # regim de lichiditate
    if liq_raw > 0.25:
        regime = "ridicată"
    elif liq_raw < -0.25:
        regime = "scăzută"
    else:
        regime = "normală"

    # intensitate (cât de mult diferă față de normal)
    abs_raw = abs(liq_raw)
    if abs_raw < 0.25:
        strength = "slabă"
    elif abs_raw < 0.6:
        strength = "moderată"
    else:
        strength = "puternică"

    return {
        "liquidity_score": liquidity_score,
        "liquidity_regime": regime,
        "liquidity_strength": strength,
        "components": {
            "range_component": r,
            "volume_density_component": d,
            "micro_vol_component": mv,
            "raw": liq_raw,
        },
    }


def compute_liquidity_from_file(path: Optional[str] = None) -> Dict[str, object]:
    """
    Helper simplu: încarcă intraday OHLC + volum și calculează Liquidity Score.
    """
    df = load_intraday_ohlc(path=path)
    return compute_liquidity_score(df)
