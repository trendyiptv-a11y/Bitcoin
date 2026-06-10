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
    - `regime` textual descrie structura mare.
    - `ic_dir` descrie biasul direcțional.
    - `ic_flux` descrie tensiunea / intensitatea, nu direcția.
    - `dev_pct` descrie deviația față de model.
    """
    if row is None or not isinstance(row, pd.Series):
        return None

    regime_name = str(row.get("regime", "") or "").lower()
    ic_dir = _finite_float(row.get("ic_dir"), 50.0)
    ic_flux = _finite_float(row.get("ic_flux"), 0.0)

    is_bull = regime_name.startswith("bull")
    is_bear = regime_name in {"bear_struct", "bear_late"}
    is_accum_bear = regime_name == "accum_bear"
    is_mixed = regime_name == "mixed"

    # Fluxul este pe scară 0–100 în IC series.
    # Îl folosim ca intensitate/tensiune, nu ca direcție.
    flux_active = ic_flux >= 20.0
    flux_tense = ic_flux >= 40.0

    # 8 regimuri standard, derivate din structura mare + direcție + tensiune.
    if is_bull and ic_dir >= 65:
        if flux_active:
            trend_label = "1. Trend ascendent puternic"
            trend_code = "up_trend_strong"
        else:
            trend_label = "2. Trend ascendent moderat"
            trend_code = "up_trend_moderate"
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
        trend_label = "8. Tranziție / acumulare"
        trend_code = "transition_accumulation"
    elif is_bear and ic_dir <= 35:
        if flux_tense:
            trend_label = "7. Trend descendent puternic"
            trend_code = "down_trend_strong"
        else:
            trend_label = "6. Trend descendent moderat"
            trend_code = "down_trend_moderate"
    elif is_bear:
        trend_label = "6. Trend descendent moderat"
        trend_code = "down_trend_moderate"
    else:
        if ic_dir > 55:
            trend_label = "3. Range cu bias pozitiv"
            trend_code = "range_bias_up"
        elif ic_dir < 45:
            trend_label = "5. Range cu bias negativ"
            trend_code = "range_bias_down"
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
