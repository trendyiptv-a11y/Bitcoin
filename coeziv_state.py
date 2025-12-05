import json
import os
import sys
import math
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, List, Any

import pandas as pd
import requests

# ============================
#  SETĂRI DE PATH
# ============================

BASE_DIR = os.path.dirname(__file__)
STRATEGY_DIR = os.path.join(BASE_DIR, "btc-swing-strategy")

if STRATEGY_DIR not in sys.path:
    sys.path.append(STRATEGY_DIR)

from btc_swing_strategy import generate_signals

# Flow & Liquidity – pot fi opționale, tratăm robust

# --- FLOW SCORE ---
try:
    # folosim implementarea din btc_flow_score.py
    from btc_flow_score import compute_flow_from_daily_csv as _compute_flow_from_daily_csv

    def compute_flow_from_file(path: Optional[str] = None) -> Dict[str, Any]:
        """
        Wrapper peste compute_flow_from_daily_csv astfel încât să întoarcă
        structura așteptată de coeziv_state.py:
          {
            "flow_score": float,
            "flow_bias": "pozitiv/negativ/neutru",
            "flow_strength": "slab/moderat/puternic",
            "components": {...}
          }
        """
        try:
            if path is None:
                raw = _compute_flow_from_daily_csv()
            else:
                raw = _compute_flow_from_daily_csv(path)
        except TypeError:
            # dacă semnătura nu acceptă path, cădem pe varianta simplă
            raw = _compute_flow_from_daily_csv()

        bias = raw.get("bias")
        strength = raw.get("strength")

        try:
            flow_value = float(raw.get("flow_value", 0.0) or 0.0)
        except Exception:
            flow_value = 0.0

        components = {
            "bias": bias,
            "strength": strength,
            "flow_value": flow_value,
            "open": raw.get("open"),
            "close": raw.get("close"),
        }

        return {
            "flow_score": flow_value,
            "flow_bias": bias,
            "flow_strength": strength,
            "components": components,
        }

except ImportError:
    # fallback foarte defensiv
    def compute_flow_from_file(path: Optional[str] = None) -> Dict[str, Any]:
        return {
            "flow_score": None,
            "flow_bias": None,
            "flow_strength": None,
            "components": {},
        }


# --- LIQUIDITY SCORE ---
try:
    # folosim implementarea din btc_liquidity_score.py
    from btc_liquidity_score import compute_liquidity_from_daily_csv as _compute_liquidity_from_daily_csv

    def compute_liquidity_from_file(path: Optional[str] = None) -> Dict[str, Any]:
        """
        Wrapper peste compute_liquidity_from_daily_csv astfel încât să întoarcă
        structura așteptată de coeziv_state.py:
          {
            "liquidity_score": float,
            "liquidity_regime": "ridicată/moderată/scăzută",
            "liquidity_strength": "slabă/moderată/puternică",
            "components": {...}
          }
        """
        try:
            if path is None:
                raw = _compute_liquidity_from_daily_csv()
            else:
                raw = _compute_liquidity_from_daily_csv(path)
        except TypeError:
            raw = _compute_liquidity_from_daily_csv()

        regime = raw.get("regime")
        strength = raw.get("strength")

        try:
            liquidity_value = float(raw.get("liquidity_value", 0.0) or 0.0)
        except Exception:
            liquidity_value = 0.0

        components = {
            "regime": regime,
            "strength": strength,
            "liquidity_value": liquidity_value,
            "avg7": raw.get("avg7"),
        }

        return {
            "liquidity_score": liquidity_value,
            "liquidity_regime": regime,
            "liquidity_strength": strength,
            "components": components,
        }

except ImportError:
    def compute_liquidity_from_file(path: Optional[str] = None) -> Dict[str, Any]:
        return {
            "liquidity_score": None,
            "liquidity_regime": None,
            "liquidity_strength": None,
            "components": {},
        }

# cost de producție – versiunea automată; fallback la None dacă nu există modulul
try:
    from btc_production_auto import estimate_production
except ImportError:
    def estimate_production() -> Optional[float]:
        return None

# ============================
#  HELPER PENTRU PATH-URI
# ============================


def find_file(candidates: List[str]) -> Optional[str]:
    """
    Găsește primul fișier existent dintr-o listă de căi relative/absolute.
    Întoarce path-ul complet sau None.
    """
    for rel in candidates:
        if os.path.isabs(rel):
            candidate = rel
        else:
            candidate = os.path.join(BASE_DIR, rel)

        if os.path.exists(candidate):
            return candidate
    return None


# ============================
#  ÎNCĂRCARE IC SERIES
# ============================

def load_ic_series(path: Optional[str] = None) -> pd.DataFrame:
    """
    Încarcă seria IC BTC din ic_btc_series.json.

    Caută fișierul în mai multe locații posibile:
      - ./ic_btc_series.json
      - ./j-btc-coeziv/ic_btc_series.json
      - ./data/ic_btc_series.json
      - ./btc-swing-strategy/ic_btc_series.json
    """
    candidates: List[str] = []

    if path is not None:
        candidates.append(path)

    candidates.extend(
        [
            "ic_btc_series.json",
            os.path.join("j-btc-coeziv", "ic_btc_series.json"),
            os.path.join("data", "ic_btc_series.json"),
            os.path.join("btc-swing-strategy", "ic_btc_series.json"),
        ]
    )

    found = find_file(candidates)
    if found is None:
        raise FileNotFoundError("Nu s-a găsit ic_btc_series.json în locațiile așteptate.")

    with open(found, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    return df


# ============================
#  ÎNCĂRCARE SCORURI TREND/MEAN-REVERT/REGIME
# ============================

def load_regime_state(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Încarcă un fișier JSON cu informații despre regimul pieței (trend / mean reversion / sideways etc.).
    """
    candidates = []

    if path is not None:
        candidates.append(path)

    candidates.extend(
        [
            "btc_regime_state.json",
            os.path.join("data", "btc_regime_state.json"),
            os.path.join("btc-swing-strategy", "btc_regime_state.json"),
        ]
    )

    found = find_file(candidates)
    if found is None:
        return {}

    with open(found, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================
#  RISC MACRO (HEADWIND / TAILWIND)
# ============================

def load_macro_risk_state(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Încarcă un fișier JSON cu informații despre regimul macro (risk-on / risk-off etc.).
    """
    candidates = []

    if path is not None:
        candidates.append(path)

    candidates.extend(
        [
            "macro_risk_state.json",
            os.path.join("data", "macro_risk_state.json"),
            os.path.join("btc-swing-strategy", "macro_risk_state.json"),
        ]
    )

    found = find_file(candidates)
    if found is None:
        return {}

    with open(found, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================
#  FETCH PREȚ BTC (API)
# ============================

def fetch_btc_price_from_api() -> Optional[float]:
    """
    Fetch prețul curent BTCUSD dintr-un API public simplu.
    Dacă nu reușește, întoarce None.
    """
    urls = [
        "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
        "https://api.kraken.com/0/public/Ticker?pair=XXBTZUSD",
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                continue

            data = resp.json()

            if "price" in data:
                return float(data["price"])

            if "result" in data:
                result = data["result"]
                if isinstance(result, dict):
                    first_key = list(result.keys())[0]
                    last = result[first_key]
                    if "c" in last and last["c"]:
                        return float(last["c"][0])

        except Exception:
            continue

    return None


# ============================
#  COST DE PRODUCȚIE BTC
# ============================

def get_production_cost_fallback() -> Optional[float]:
    """
    Fallback foarte simplu dacă nu avem modul de producție:
      - încearcă să citească dintr-un JSON/CSV local
      - altfel întoarce None
    """
    candidates = [
        "btc_production_cost.json",
        os.path.join("data", "btc_production_cost.json"),
        os.path.join("btc-swing-strategy", "btc_production_cost.json"),
    ]

    found = find_file(candidates)
    if not found:
        return None

    try:
        with open(found, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "latest_cost" in data:
            return float(data["latest_cost"])
    except Exception:
        return None

    return None


def get_btc_production_cost() -> Optional[float]:
    """
    Wrapper standardizat:
      - dacă există modulul estimate_production(), îl folosim
      - altfel fallback la citire locală
    """
    try:
        v = estimate_production()
        if v is not None:
            return float(v)
    except Exception:
        pass

    return get_production_cost_fallback()


# ============================
#  UTILITARE SCOR / TEXT
# ============================

def classify_strength(value: float, thresholds: Tuple[float, float]) -> str:
    """
    Împarte un scor numeric în: slab / moderat / puternic
    pe baza a două praguri (low, high).
    """
    low, high = thresholds
    abs_v = abs(value)
    if abs_v < low:
        return "slab"
    elif abs_v < high:
        return "moderat"
    else:
        return "puternic"


def sign_bias(value: float, tol: float = 1e-6) -> str:
    """
    Întoarce 'pozitiv', 'negativ' sau 'neutru' în funcție de semnul lui value.
    """
    if value > tol:
        return "pozitiv"
    elif value < -tol:
        return "negativ"
    return "neutru"


# ============================
#  BUILD STATE REGIM TEHNIC
# ============================

def build_technical_state(regime_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construiește un dicționar cu informații despre regimul tehnic (trend / mean reversion etc.)
    pornind de la btc_regime_state.json sau un alt input similar.
    """
    if not regime_state:
        return {
            "trend_score": None,
            "volatility_regime": None,
            "mean_reversion_score": None,
            "raw": {},
        }

    trend_score = regime_state.get("trend_score")
    vol_regime = regime_state.get("vol_regime")
    mean_rev = regime_state.get("mean_reversion_score")

    return {
        "trend_score": trend_score,
        "volatility_regime": vol_regime,
        "mean_reversion_score": mean_rev,
        "raw": regime_state,
    }


# ============================
#  BUILD MACRO STATE
# ============================

def build_macro_state(macro_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construiește un dicționar cu informații despre regimul macro (risk-on / risk-off / neutru).
    """
    if not macro_state:
        return {
            "macro_bias": None,
            "macro_strength": None,
            "raw": {},
        }

    bias = macro_state.get("bias")
    strength = macro_state.get("strength")

    return {
        "macro_bias": bias,
        "macro_strength": strength,
        "raw": macro_state,
    }


# ============================
#  COEZIV SCORE DIN IC SERIES
# ============================

def compute_ic_snapshot(ic_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Construiește un snapshot simplu din ic_btc_series.json:
      - ultimul IC
      - media pe ultimele N zile
      - bias + strength
    """
    if ic_df.empty:
        return {
            "ic_latest": None,
            "ic_avg_7d": None,
            "ic_bias": None,
            "ic_strength": None,
        }

    ic_df = ic_df.sort_values("date").reset_index(drop=True)

    if "ic" not in ic_df.columns:
        raise ValueError("ic_btc_series.json nu conține coloana 'ic'.")

    latest_ic = float(ic_df["ic"].iloc[-1])

    window = ic_df["ic"].tail(7)
    ic_avg_7d = float(window.mean())

    bias = sign_bias(latest_ic, tol=0.01)
    strength = classify_strength(latest_ic, thresholds=(0.02, 0.05))

    return {
        "ic_latest": latest_ic,
        "ic_avg_7d": ic_avg_7d,
        "ic_bias": bias,
        "ic_strength": strength,
    }


# ============================
#  GENERARE SEMNALE SWING-STRATEGY
# ============================

def generate_strategy_signals() -> Dict[str, Any]:
    """
    Folosește btc_swing_strategy.generate_signals pentru a obține semnalele strategiei.
    """
    try:
        signals = generate_signals()
        if not isinstance(signals, dict):
            return {"raw": signals}
        return signals
    except Exception:
        return {"raw": None}


# ============================
#  AGREGARE COMPLETĂ COEZIV STATE
# ============================

def build_coeziv_state(
    ic_path: Optional[str] = None,
    regime_path: Optional[str] = None,
    macro_path: Optional[str] = None,
    flow_path: Optional[str] = None,
    liquidity_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Agregă toate componentele:
      - IC Series
      - Regim tehnic (trend/mean reversion/vol)
      - Regim macro (risk-on/off)
      - Flow score (din btc_flow_score)
      - Liquidity score (din btc_liquidity_score)
      - Cost de producție BTC
      - Semnale din btc_swing_strategy
    """
    # 1. IC
    try:
        ic_df = load_ic_series(ic_path)
        ic_snapshot = compute_ic_snapshot(ic_df)
    except Exception as e:
        ic_df = pd.DataFrame()
        ic_snapshot = {
            "ic_latest": None,
            "ic_avg_7d": None,
            "ic_bias": None,
            "ic_strength": None,
            "error": str(e),
        }

    # 2. Regim tehnic
    regime_state = load_regime_state(regime_path)
    technical_state = build_technical_state(regime_state)

    # 3. Macro
    macro_state_raw = load_macro_risk_state(macro_path)
    macro_state = build_macro_state(macro_state_raw)

    # 4. Flow
    flow_state = compute_flow_from_file(flow_path)

    # 5. Liquidity
    liquidity_state = compute_liquidity_from_file(liquidity_path)

    # 6. Cost de producție
    production_cost = get_btc_production_cost()

    # 7. Preț BTC
    spot = fetch_btc_price_from_api()

    # 8. Semnale strategie
    strategy_signals = generate_strategy_signals()

    now_utc = datetime.now(timezone.utc)

    return {
        "timestamp": now_utc.isoformat(),
        "spot": spot,
        "production_cost": production_cost,
        "ic": ic_snapshot,
        "technical": technical_state,
        "macro": macro_state,
        "flow": flow_state,
        "liquidity": liquidity_state,
        "strategy": strategy_signals,
    }


# ============================
#  MAIN (DUMP JSON)
# ============================

def main():
    """
    Rulează build_coeziv_state() și printează JSON-ul rezultat.
    """
    state = build_coeziv_state()
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
