import math
from typing import Any, Dict, Optional

import pandas as pd

import coeziv_state


def _finite_float(value: Any, fallback: float) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else fallback
    except Exception:
        return fallback


def classify_market_regime(row: pd.Series, dev_pct: Optional[float] = None) -> Optional[Dict[str, str]]:
    """
    Traducător coeziv pentru cele 8 regimuri standard de piață.

    Principiu:
    - signal = verdictul curent al mecanismului.
    - regime = memorie / structură internă.
    - ic_dir = presiune direcțională internă.
    - ic_flux = tensiune / intensitate, nu direcție.
    - dev_pct = deviație față de model.

    Regula de coerență:
    - signal=flat nu poate afișa trend descendent/ascendent activ (#1, #2, #6, #7).
    - signal=short permite doar familia negativă (#5, #6, #7).
    - signal=long permite doar familia pozitivă (#1, #2, #3).
    """
    if row is None or not isinstance(row, pd.Series):
        return None

    regime_name = str(row.get("regime", "") or "").lower()
    signal = str(row.get("signal", "") or "").lower()
    ic_dir = _finite_float(row.get("ic_dir"), 50.0)
    ic_flux = _finite_float(row.get("ic_flux"), 0.0)

    is_bull = regime_name.startswith("bull")
    is_bear = regime_name in {"bear_struct", "bear_late"}
    is_accum_bear = regime_name == "accum_bear"
    is_mixed = regime_name == "mixed"

    flux_active = ic_flux >= 20.0
    flux_tense = ic_flux >= 40.0

    # 1) Verdictul curent are prioritate: flat înseamnă context de tranziție/range,
    # chiar dacă memoria structurală mai poartă inerție bearish/bullish.
    if signal == "flat":
        if is_mixed:
            if ic_dir > 55:
                trend_label = "3. Range cu bias pozitiv"
                trend_code = "range_bias_up"
            elif ic_dir < 45:
                trend_label = "5. Range cu bias negativ"
                trend_code = "range_bias_down"
            else:
                trend_label = "4. Range neutru"
                trend_code = "range_neutral"
        elif is_accum_bear:
            trend_label = "8. Regim neutru / tranziție"
            trend_code = "transition_accumulation"
        else:
            if 45 <= ic_dir <= 55:
                trend_label = "4. Range neutru"
                trend_code = "range_neutral"
            else:
                trend_label = "8. Regim neutru / tranziție"
                trend_code = "transition_accumulation"

    # 2) Verdict negativ: familia negativă.
    elif signal == "short":
        if is_mixed:
            trend_label = "5. Range cu bias negativ"
            trend_code = "range_bias_down"
        elif is_bear and ic_dir <= 35 and flux_tense:
            trend_label = "7. Trend descendent puternic"
            trend_code = "down_trend_strong"
        else:
            trend_label = "6. Trend descendent moderat"
            trend_code = "down_trend_moderate"

    # 3) Verdict pozitiv: familia pozitivă.
    elif signal == "long":
        if is_mixed:
            trend_label = "3. Range cu bias pozitiv"
            trend_code = "range_bias_up"
        elif is_bull and ic_dir >= 65 and flux_active:
            trend_label = "1. Trend ascendent puternic"
            trend_code = "up_trend_strong"
        else:
            trend_label = "2. Trend ascendent moderat"
            trend_code = "up_trend_moderate"

    # 4) Fallback dacă signal lipsește sau are o valoare necunoscută.
    else:
        if is_bull and ic_dir >= 65 and flux_active:
            trend_label = "1. Trend ascendent puternic"
            trend_code = "up_trend_strong"
        elif is_bull:
            trend_label = "2. Trend ascendent moderat"
            trend_code = "up_trend_moderate"
        elif is_mixed:
            if ic_dir > 55:
                trend_label = "3. Range cu bias pozitiv"
                trend_code = "range_bias_up"
            elif ic_dir < 45:
                trend_label = "5. Range cu bias negativ"
                trend_code = "range_bias_down"
            else:
                trend_label = "4. Range neutru"
                trend_code = "range_neutral"
        elif is_accum_bear:
            trend_label = "8. Regim neutru / tranziție"
            trend_code = "transition_accumulation"
        elif is_bear and ic_dir <= 35 and flux_tense:
            trend_label = "7. Trend descendent puternic"
            trend_code = "down_trend_strong"
        elif is_bear:
            trend_label = "6. Trend descendent moderat"
            trend_code = "down_trend_moderate"
        else:
            trend_label = "4. Range neutru"
            trend_code = "range_neutral"

    dev_label = "cu deviație nedefinită față de model"
    dev_code = "dev_unknown"

    if dev_pct is not None and math.isfinite(dev_pct):
        dev_abs = abs(dev_pct)
        if dev_abs < 0.005:
            dev_label = "cu deviație normală față de model"
            dev_code = "dev_normal"
        elif dev_abs < 0.02:
            dev_label = "cu deviație moderată față de model"
            dev_code = "dev_moderate"
        elif dev_abs < 0.04:
            dev_label = "cu deviație tensionată față de model"
            dev_code = "dev_tension"
        else:
            dev_label = "cu deviație extremă față de model"
            dev_code = "dev_extreme"

    tension_suffix = ""
    tension_code = ""
    if flux_tense:
        tension_suffix = " — flux tensionat"
        tension_code = "_flux_tense"
    elif flux_active:
        tension_suffix = " — flux activ"
        tension_code = "_flux_active"

    return {
        "label": f"{trend_label} {dev_label}{tension_suffix}",
        "code": f"{trend_code}_{dev_code}{tension_code}",
    }


coeziv_state.classify_market_regime = classify_market_regime

if __name__ == "__main__":
    coeziv_state.main()
