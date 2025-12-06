import json
import os
import sys
import math
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any, Literal

import pandas as pd
import requests

# ============================
#  SETĂRI DE PATH
# ============================

BASE_DIR = os.path.dirname(__file__)
STRATEGY_DIR = os.path.join(BASE_DIR, "btc-swing-strategy")
DATA_DIR = os.path.join(STRATEGY_DIR, "data")
IC_SERIES_PATH = os.path.join(DATA_DIR, "ic_series.csv")
BTC_DAILY_PATH = os.path.join(DATA_DIR, "btc_daily.csv")


# ============================
#  UTILITARE
# ============================

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _safe_int(x: Any) -> Optional[int]:
    try:
        v = int(x)
    except Exception:
        return None
    return v


# ============================
#  CLASIFICARE REGIM STRUCTURAL (8 SITUAȚII STANDARD)
# ============================

StructuralPosition = Literal["below", "around", "above"]
StructuralFlowDir = Literal["buy", "sell", "neutral"]
StructuralLiqLevel = Literal["low", "mid", "high"]


@dataclass
class StructuralRegime:
    id: int
    key: str
    label: str
    description: str


def _normalize_structural_flow_bias(flow_bias: Optional[str]) -> Optional[StructuralFlowDir]:
    if not flow_bias:
        return None
    fb = flow_bias.lower()
    if fb.startswith("pozit"):
        return "buy"
    if fb.startswith("negat"):
        return "sell"
    if fb.startswith("neutr"):
        return "neutral"
    return None


def _normalize_structural_liquidity_regime(liq_regime: Optional[str]) -> Optional[StructuralLiqLevel]:
    if not liq_regime:
        return None
    lr = liq_regime.lower()
    if "scaz" in lr or "scăz" in lr:
        return "low"
    if "ridic" in lr:
        return "high"
    if "moder" in lr or "normal" in lr:
        return "mid"
    return None


def _compute_structural_position(model_price: Optional[float], spot_price: Optional[float]) -> Tuple[Optional[StructuralPosition], Optional[float]]:
    if model_price is None or spot_price is None:
        return None, None
    if not (isinstance(model_price, (int, float)) and isinstance(spot_price, (int, float))):
        return None, None
    if model_price <= 0 or not math.isfinite(model_price) or not math.isfinite(spot_price):
        return None, None

    diff = spot_price - model_price
    pct = diff / model_price  # ex: 0.01 = +1%

    # praguri: ±0.5% = "la model"
    if pct < -0.005:
        pos: StructuralPosition = "below"
    elif pct > 0.005:
        pos = "above"
    else:
        pos = "around"

    return pos, pct


def classify_structural_regime(
    model_price: Optional[float],
    spot_price: Optional[float],
    flow_bias: Optional[str],
    flow_strength: Optional[str],
    liquidity_regime: Optional[str],
    liquidity_strength: Optional[str],
) -> Optional[StructuralRegime]:
    """Clasifică contextul curent într-unul dintre cele 8 regimuri standardizate.

    Dacă informațiile sunt insuficiente sau combinația nu este clară,
    întoarce None (context non-canonic).
    """

    pos, pct = _compute_structural_position(model_price, spot_price)
    flow = _normalize_structural_flow_bias(flow_bias)
    liq = _normalize_structural_liquidity_regime(liquidity_regime)

    flow_strength_norm = (flow_strength or "").lower()
    liq_strength_norm = (liquidity_strength or "").lower()

    if not pos or not flow or not liq:
        return None

    # 1️⃣ Fragilitate maximă
    if (
        pos == "below"
        and flow == "sell"
        and liq == "low"
        and flow_strength_norm in ("medie", "puternică")
    ):
        return StructuralRegime(
            id=1,
            key="fragility_max",
            label="Fragilitate maximă",
            description="Sub model, presiune de vânzare și lichiditate scăzută – mișcările sunt amplificate de absența market makerilor.",
        )

    # 2️⃣ Bearish structural
    if (
        pos == "below"
        and flow == "sell"
        and liq in ("mid", "high")
    ):
        return StructuralRegime(
            id=2,
            key="bearish_structural",
            label="Bearish structural",
            description="Sub model, vânzare în condiții de lichiditate cel puțin moderată – piața acceptă niveluri mai joase.",
        )

    # 3️⃣ Piață inertă
    if (
        pos == "below"
        and flow == "neutral"
        and liq in ("low", "mid")
    ):
        return StructuralRegime(
            id=3,
            key="inert_below",
            label="Piață inertă",
            description="Sub model, flux echilibrat și lichiditate redusă – lipsă de apetit clar pentru direcție.",
        )

    # 4️⃣ Echilibru sănătos
    if (
        pos == "around"
        and flow in ("neutral",)
        and liq in ("mid", "high")
    ):
        return StructuralRegime(
            id=4,
            key="equilibrium",
            label="Echilibru sănătos",
            description="BTC este la nivelul modelului, fluxul este relativ echilibrat, iar lichiditatea este confortabilă.",
        )

    # 5️⃣ Accumulare ordonată
    if (
        pos == "around"
        and flow == "buy"
        and liq == "high"
    ):
        return StructuralRegime(
            id=5,
            key="accumulation",
            label="Accumulare ordonată",
            description="La model, cumpărare în regim de lichiditate ridicată – interes real, dar fără grabă.",
        )

    # 6️⃣ Expansiune sănătoasă
    if (
        pos == "above"
        and flow == "buy"
        and liq == "high"
    ):
        return StructuralRegime(
            id=6,
            key="expansion",
            label="Expansiune sănătoasă",
            description="Peste model, cumpărare în condiții de lichiditate ridicată – momentum susținut într-o piață elastică.",
        )

    # 7️⃣ Fragilitate bullish
    if (
        pos == "above"
        and flow == "buy"
        and liq == "low"
    ):
        return StructuralRegime(
            id=7,
            key="fragility_bullish",
            label="Fragilitate bullish",
            description="Peste model, cumpărare într-o piață subțire – raliu rapid, dar instabil.",
        )

    # 8️⃣ Tranziție de regim
    if (
        pos == "above"
        and flow == "sell"
        and liq == "low"
    ):
        return StructuralRegime(
            id=8,
            key="regime_transition",
            label="Tranziție de regim",
            description="Peste model, vânzare în lichiditate scăzută – distribuție sau schimbare de comportament al pieței.",
        )

    # context valid, dar non-canonic
    return None


# ============================
#  ÎNCĂRCARE SERIE IC
# ============================

def load_ic_series() -> pd.DataFrame:
    if not os.path.exists(IC_SERIES_PATH):
        raise FileNotFoundError(f"Nu găsesc fișierul IC series: {IC_SERIES_PATH}")

    df = pd.read_csv(IC_SERIES_PATH, parse_dates=["timestamp"], index_col="timestamp")
    df.sort_index(inplace=True)
    return df


# ============================
#  GENERARE SEMNALE COEZIVE
# ============================

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    # ne asigurăm că avem coloanele necesare
    for col in ("ic_trend", "ic_flux", "ic_dir", "ic_regime"):
        if col not in df.columns:
            raise ValueError(f"Coloana lipsă în IC series: {col}")

    # semnal brut: folosim semnalul de swing deja calculat (dacă există)
    # sau derivăm un semnal simplu din ic_trend / ic_dir
    signal = []

    for _, row in df.iterrows():
        s = str(row.get("signal", "")).lower()
        if s in ("long", "short", "flat"):
            signal.append(s)
            continue

        ic_trend = _safe_float(row.get("ic_trend")) or 0.0
        ic_dir = _safe_float(row.get("ic_dir")) or 0.0

        if ic_trend > 0 and ic_dir > 0:
            signal.append("long")
        elif ic_trend < 0 and ic_dir < 0:
            signal.append("short")
        else:
            signal.append("flat")

    df["signal"] = signal
    return df


# ============================
#  ISTORIC CONTEXTE PENTRU UI
# ============================

def build_signal_history(df_signals: pd.DataFrame, limit: int = 30) -> List[Dict[str, Any]]:
    """
    Construiește un istoric simplificat al contextelor (semnalelor) pentru UI.
    Păstrăm doar ultimele `limit` snapshot-uri (implicit 30).
    """

    history: List[Dict[str, Any]] = []
    if df_signals is None or df_signals.empty:
        return history

    tail = df_signals.tail(limit)

    for ts, row in tail.iterrows():
        sig = str(row.get("signal", "")).lower()
        if sig not in ("long", "short", "flat"):
            sig = "flat"

        price = _safe_float(row.get("price_usd"))
        model_price = _safe_float(row.get("model_price_usd"))

        dev_pct: Optional[float] = None
        if price is not None and model_price is not None and model_price > 0:
            dev_pct = (price - model_price) / model_price

        history.append(
            {
                "timestamp": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
                "signal": sig,
                "price_usd": price,
                "model_price_usd": model_price,
                "deviation_from_model": dev_pct,
            }
        )

    return history


# ============================
#  CLASIFICARE REGIM PIAȚĂ (TREND + MODEL)
# ============================

def classify_market_regime(row: pd.Series, dev_pct: Optional[float] = None) -> Optional[Dict[str, str]]:
    """
    Construiește o etichetă textuală de regim de piață pe baza coloanelor
    din ultimul snapshot (regime, ic_flux, ic_dir) și, opțional, a deviației
    prețului față de model (dev_pct).

    Întoarce un dict de forma:
      {
        "label": "Trend descendent cu deviație moderată față de model",
        "code": "down_trend_moderate_dev"
      }
    sau None, dacă nu avem destule informații.
    """
    if row is None or not isinstance(row, pd.Series):
        return None

    try:
        regime_val = float(row.get("ic_regime"))
    except Exception:
        regime_val = 0.0

    try:
        flux_val = float(row.get("ic_flux"))
    except Exception:
        flux_val = 0.0

    # clasificare de bază a trendului dinamic
    trend_label = "Regim neutru"
    trend_code = "neutral"

    if regime_val >= 0.4:
        if flux_val >= 0.2:
            trend_label = "Trend ascendent susținut"
            trend_code = "up_trend_strong"
        else:
            trend_label = "Trend ascendent moderat"
            trend_code = "up_trend_moderate"
    elif regime_val <= -0.4:
        if flux_val <= -0.2:
            trend_label = "Trend descendent susținut"
            trend_code = "down_trend_strong"
        else:
            trend_label = "Trend descendent moderat"
            trend_code = "down_trend_moderate"
    else:
        # zonă de range / tranziție
        if abs(flux_val) < 0.1:
            trend_label = "Range neutru"
            trend_code = "range_neutral"
        elif flux_val > 0:
            trend_label = "Range cu bias pozitiv"
            trend_code = "range_bias_up"
        else:
            trend_label = "Range cu bias negativ"
            trend_code = "range_bias_down"

    # clasificare deviație față de model
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

    label = f"{trend_label} {dev_label}"
    code = f"{trend_code}_{dev_code}"

    return {"label": label, "code": code}


# ============================
#  PREȚ LIVE BTC (API GRATUIT)
# ============================

def get_live_btc_price() -> float:
    """
    Ia prețul BTC/USD din CoinGecko (API gratuit, fără cheie).
    Dacă apare o eroare, excepția va fi prinsă în main() și
    vom cădea pe prețul din model.
    """
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin", "vs_currencies": "usd"}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return float(data["bitcoin"]["usd"])


# ============================
#  STATISTICĂ ISTORICĂ PENTRU SEMNAL
# ============================

def compute_signal_stats(
    df: pd.DataFrame,
    horizon_hours: int = 24,
    move_threshold: float = 0.005,
    min_samples: int = 100,
) -> Dict[str, Any]:
    """
    Calculează statistică istorică pentru semnalul curent (long/short/flat),
    pe baza seriei de semnale și a randamentelor viitoare.

    Întoarce un dict cu:
      {
        "probability": 0.37,        # probabilitatea ca viitorul să fie în direcția semnalului
        "samples": 1462,            # câte cazuri similare în istoric
        "horizon_hours": 24,
        "source": "historical_ic",
        "breakdown": {
            "in_direction": p_in_dir,
            "opposite": p_opp,
            "flat": p_flat,
        },
        "expected_drift": {
            "horizon_hours": horizon_hours,
            "p10": ...,
            "p50": ...,
            "p90": ...,
        }
      }
    """

    out: Dict[str, Any] = {
        "probability": None,
        "samples": 0,
        "horizon_hours": horizon_hours,
        "source": "historical_ic",
        "breakdown": None,
        "expected_drift": None,
    }

    if df is None or df.empty:
        return out

    # avem nevoie de coloane: signal, price_usd
    if "signal" not in df.columns or "price_usd" not in df.columns:
        return out

    # calculăm randamente viitoare pe orizontul dat
    df = df.copy()
    df["price_future"] = df["price_usd"].shift(-horizon_hours)
    df["return_future"] = (df["price_future"] - df["price_usd"]) / df["price_usd"]

    last = df.iloc[-1]
    last_signal = str(last.get("signal", "neutral")).lower()
    if last_signal not in ("long", "short", "flat"):
        return out

    # filtrăm cazurile similare (același semnal)
    same = df[df["signal"].str.lower() == last_signal].dropna(subset=["return_future"])
    if same.empty:
        return out

    n = len(same)
    out["samples"] = int(n)

    # expected drift (p10 / p50 / p90)
    rets = same["return_future"].astype(float).dropna()
    if len(rets) >= max(30, min_samples // 2):
        p10 = float(rets.quantile(0.1))
        p50 = float(rets.quantile(0.5))
        p90 = float(rets.quantile(0.9))
        out["expected_drift"] = {
            "horizon_hours": horizon_hours,
            "p10": p10,
            "p50": p50,
            "p90": p90,
        }

    # dacă nu avem suficiente cazuri, păstrăm doar expected_drift (dacă s-a putut calcula)
    if n < max(1, min_samples):
        return out

    in_dir = 0
    opp = 0
    flat = 0

    for _, ret in same["return_future"].items():
        if not (isinstance(ret, (int, float)) and math.isfinite(ret)):
            continue

        if last_signal == "long":
            if ret > move_threshold:
                in_dir += 1
            elif ret < -move_threshold:
                opp += 1
            else:
                flat += 1
        elif last_signal == "short":
            if ret < -move_threshold:
                in_dir += 1
            elif ret > move_threshold:
                opp += 1
            else:
                flat += 1
        else:
            # flat: nu avem direcție explicită, tratăm simetric
            if abs(ret) <= move_threshold:
                in_dir += 1
            elif ret > move_threshold:
                opp += 1
            else:
                flat += 1

    total = in_dir + opp + flat
    if total == 0:
        return out

    p_in_dir = in_dir / total
    p_opp = opp / total
    p_flat = flat / total

    out["probability"] = p_in_dir
    out["breakdown"] = {
        "in_direction": p_in_dir,
        "opposite": p_opp,
        "flat": p_flat,
    }

    return out


# ============================
#  COST DE PRODUCȚIE (ESTIMARE)
# ============================

def estimate_production_cost() -> Tuple[Optional[float], Optional[str]]:
    """
    Estimează costul de producție al BTC pe baza fișierului btc_daily.csv.
    Întoarce (cost_usd, as_of_date_str)
    """
    path = BTC_DAILY_PATH
    if not os.path.exists(path):
        return None, None

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)

    if "production_cost_usd" not in df.columns:
        return None, None

    last = df.iloc[-1]
    cost = _safe_float(last.get("production_cost_usd"))
    if cost is None:
        return None, None

    ts = last.get("timestamp")
    if isinstance(ts, pd.Timestamp):
        as_of = ts.date().isoformat()
    else:
        as_of = str(ts)

    return cost, as_of


# ============================
#  FLOW SCORE & LIQUIDITY SCORE
# ============================

from btc_flow_score import compute_flow_from_file
from btc_liquidity_score import compute_liquidity_from_file


# ============================
#  MESAJ COEZIV PENTRU UI
# ============================

def build_message(
    signal: str,
    price: Optional[float],
    flow_bias: Optional[str],
    flow_strength: Optional[str],
    liquidity_regime: Optional[str],
    stats: Dict[str, Any],
    deviation_from_production: Optional[float],
) -> str:
    """
    Construiește mesajul narativ pentru utilizator pe baza:
      - semnalului (long/short/flat)
      - poziționării față de costul de producție (dacă există)
      - Flow Regime
      - Liquidity Regime
      - statisticii istorice (probabilitate + expected drift)
    """
    parts: List[str] = []

    # 1. descriere de bază a semnalului
    if signal == "long":
        parts.append(
            "Context structural ușor pozitiv, dar cu probabilitate istorică slabă. "
            "Pentru utilizatori obișnuiți: atenție la poziții foarte mari pe upside; "
            "nu este un moment de euforie, ci mai degrabă de reacție controlată."
        )
    elif signal == "short":
        parts.append(
            "Context structural ușor negativ, dar cu probabilitate istorică slabă. "
            "Pentru utilizatori obișnuiți: prudență cu pozițiile mari, în special cele long, "
            "dar nu este un moment de acțiune agresivă."
        )
    else:
        parts.append(
            "Context structural mai degrabă neutru; mecanismul nu vede un avantaj clar "
            "nici pentru upside, nici pentru downside în acest snapshot."
        )

    # 2. Flow Regime
    if flow_bias in ("pozitiv", "negativ", "neutru"):
        if flow_bias == "pozitiv":
            txt = "Fluxul actual de piață este orientat spre cumpărare"
        elif flow_bias == "negativ":
            txt = "Fluxul actual de piață este orientat spre vânzare"
        else:
            txt = "Fluxul actual de piață este relativ echilibrat"

        if flow_strength in ("slab", "moderat", "puternic"):
            txt += f" ({flow_strength})."

        parts.append(txt)

    # 3. lichiditate
    if liquidity_regime in ("scăzută", "normală", "ridicată"):
        if liquidity_regime == "ridicată":
            txt = "Lichiditatea este ridicată, iar mișcările de preț tind să fie mai stabile."
        elif liquidity_regime == "scăzută":
            txt = "Lichiditatea este scăzută, iar mișcările de preț pot fi mai bruște decât de obicei."
        else:
            txt = "Lichiditatea este la un nivel moderat, specific contextelor obișnuite de piață."

        parts.append(txt)

    # 4. deviația față de costul de producție, dacă există
    if deviation_from_production is not None and math.isfinite(deviation_from_production):
        pct = deviation_from_production * 100
        if abs(pct) < 10:
            parts.append(
                "Prețul este într-o zonă de echilibru față de costul estimat de producție al rețelei."
            )
        elif pct <= 50:
            parts.append(
                "Prețul este semnificativ peste costul estimat de producție, "
                "zonă specifică fazelor speculative moderate."
            )
        else:
            parts.append(
                "Prețul este mult peste costul estimat de producție, "
                "ceea ce sugerează o fază speculativă ridicată."
            )

    # 5. așteptări bazate pe istoric (expected drift)
    if stats:
        drift = stats.get("expected_drift") or {}
        p10 = drift.get("p10")
        p50 = drift.get("p50")
        p90 = drift.get("p90")
        horizon = drift.get("horizon_hours", 24)

        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (p10, p50, p90)):
            parts.append(
                f"Istoric, în contexte similare, mișcarea pe următoarele ~{horizon} ore "
                f"a fost de aproximativ {p10 * 100:.1f}% în scenariile mai pesimiste, "
                f"{p50 * 100:.1f}% în scenariile mediene și până la {p90 * 100:.1f}% "
                f"în scenariile mai favorabile."
            )

    parts.append("Nu este o recomandare de tranzacționare, ci o interpretare structurată a riscului.")

    return " ".join(parts)


# ============================
#  MAIN – GENERAREA STĂRII COEZIVE
# ============================

def main() -> None:
    # 1. încărcăm datele IC (structură, direcție, flux, regim)
    df = load_ic_series()

    # 2. generăm semnalele coezive
    df = generate_signals(df)
    if df is None or df.empty:
        raise RuntimeError("DataFrame-ul cu semnale este gol după generate_signals().")

    # 3. extragem ultimul punct din serie
    last = df.iloc[-1]
    ts = last.name  # index datetime

    # semnal brut
    signal = str(last.get("signal", "neutral")).lower()
    if signal not in ("long", "short", "flat", "neutral"):
        signal = "neutral"

    # prețuri din IC / model
    price_from_series = _safe_float(last.get("price_usd"))
    model_price = _safe_float(last.get("model_price_usd"))

    # 4. preț "live" pentru UI
    price_source = "spot"
    price_for_text: Optional[float] = None

    try:
        price_for_text = get_live_btc_price()
        if not (isinstance(price_for_text, (int, float)) and math.isfinite(price_for_text)):
            price_for_text = None
    except Exception as e:
        print("Nu am putut lua prețul BTC live, folosesc modelul / seria locală.", e)
        price_for_text = None

    if price_for_text is None:
        # fallback: folosim prețul din serie sau, dacă lipsește, prețul model
        if price_from_series is not None:
            price_for_text = price_from_series
            price_source = "spot_series"
        elif model_price is not None:
            price_for_text = model_price
            price_source = "model"
        else:
            price_for_text = None
            price_source = "unknown"

    # 5. cost de producție
    production_cost: Optional[float] = None
    production_as_of: Optional[str] = None
    try:
        production_cost, production_as_of = estimate_production_cost()
    except Exception as e:
        print("Nu am putut estima costul de producție BTC.", e)
        production_cost, production_as_of = None, None

    deviation_from_production: Optional[float] = None
    try:
        if production_cost is not None and math.isfinite(production_cost) and production_cost > 0:
            ref_price = (
                price_for_text
                if (math.isfinite(price_for_text) and price_for_text > 0)
                else model_price
            )
            if ref_price is not None and math.isfinite(ref_price) and ref_price > 0:
                deviation_from_production = (ref_price - production_cost) / production_cost
    except Exception:
        deviation_from_production = None

    # 6. deviația curentă față de prețul modelului (pentru regim de piață)
    dev_pct_model: Optional[float] = None
    try:
        if math.isfinite(model_price) and model_price > 0 and math.isfinite(price_for_text):
            dev_pct_model = (price_for_text - model_price) / model_price
    except Exception:
        dev_pct_model = None

    # 7. clasificarea regimului de piață (trend + model)
    market_regime = classify_market_regime(last, dev_pct_model)

    # 8. Flow Score și Liquidity Score
    try:
        flow = compute_flow_from_file()
    except Exception as e:
        print("Nu am putut calcula Flow Score:", e)
        flow = {
            "flow_score": None,
            "flow_bias": None,
            "flow_strength": None,
            "components": {},
        }

    try:
        liq = compute_liquidity_from_file()
    except Exception as e:
        print("Nu am putut calcula Liquidity Score:", e)
        liq = {
            "liquidity_score": None,
            "liquidity_regime": None,
            "liquidity_strength": None,
            "components": {},
        }

    # 8b. clasificare regim structural standardizat (8 situații)
    structural_regime = classify_structural_regime(
        model_price=model_price,
        spot_price=price_for_text,
        flow_bias=flow.get("flow_bias"),
        flow_strength=flow.get("flow_strength"),
        liquidity_regime=liq.get("liquidity_regime"),
        liquidity_strength=liq.get("liquidity_strength"),
    )

    # 9. istoric de contexte (ultimele N puncte)
    signal_history = build_signal_history(df, limit=30)

    # 10. statistică istorică pentru semnalul curent
    stats = compute_signal_stats(
        df,
        horizon_hours=24,
        move_threshold=0.005,
        min_samples=100,
    )

    # 11. generăm mesajul coeziv pentru utilizator (ce se întâmplă + ce se poate întâmpla)
    message = build_message(
        signal=signal,
        price=price_for_text,
        flow_bias=flow.get("flow_bias"),
        flow_strength=flow.get("flow_strength"),
        liquidity_regime=liq.get("liquidity_regime"),
        stats=stats,
        deviation_from_production=deviation_from_production,
    )

    # 12. construim starea finală
    state: Dict[str, Any] = {
        "timestamp": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),

        # prețuri
        "price_usd": price_for_text,      # ce vede utilizatorul ca preț "acum"
        "model_price_usd": model_price,   # prețul intern al mecanismului
        "price_source": price_source,     # "spot" sau "model"

        # context principal
        "signal": signal,
        "message": message,
        "generated_at": datetime.now(timezone.utc).isoformat(),

        # istoric contexte
        "signal_history": signal_history,

        # statistică istorică
        "signal_probability": stats.get("probability"),
        "signal_prob_samples": stats.get("samples"),
        "signal_prob_horizon_hours": stats.get("horizon_hours"),
        "signal_prob_source": stats.get("source"),
        "signal_prob_breakdown": stats.get("breakdown"),
        "signal_expected_drift": stats.get("expected_drift"),

        # regim de piață (dinamic, trend + deviație față de model)
        "market_regime": market_regime,

        # regim structural standardizat (8 situații)
        "market_regime_id": getattr(structural_regime, "id", None),
        "market_regime_key": getattr(structural_regime, "key", None),
        "market_regime_label": getattr(structural_regime, "label", None),
        "market_regime_description": getattr(structural_regime, "description", None),

        # cost de producție
        "production_cost_usd": production_cost,
        "production_cost_as_of": production_as_of,
        "deviation_from_production": deviation_from_production,

        # Flow Score
        "flow_score": flow.get("flow_score"),
        "flow_bias": flow.get("flow_bias"),
        "flow_strength": flow.get("flow_strength"),
        "flow_components": flow.get("components", {}),

        # Liquidity Score
        "liquidity_score": liq.get("liquidity_score"),
        "liquidity_regime": liq.get("liquidity_regime"),
        "liquidity_strength": liq.get("liquidity_strength"),
        "liquidity_components": liq.get("components", {}),
    }

    # 13. scriem JSON în folderul frontend-ului
    os.makedirs(STRATEGY_DIR, exist_ok=True)
    output_path = os.path.join(STRATEGY_DIR, "coeziv_state.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print(f"Am scris starea coezivă în: {output_path}")

    # log sumar în consolă (opțional)
    prob = state.get("signal_probability")
    samples = state.get("signal_prob_samples")
    drift = state.get("signal_expected_drift")

    print("Semnal brut:", signal)
    print("Preț UI:", state.get("price_usd"), "| Model:", state.get("model_price_usd"))
    print("Regim piață:", market_regime)
    print("Flow bias:", flow.get("flow_bias"), "| Flow strength:", flow.get("flow_strength"))
    print("Liquidity regime:", liq.get("liquidity_regime"), "| Liquidity strength:", liq.get("liquidity_strength"))

    if prob is not None and samples:
        try:
            print(
                "Probabilitate istorică ca scenariul să evolueze în direcția contextului "
                f"(~{state['signal_prob_horizon_hours']}h): "
                f"~{prob * 100:.1f}% (bazat pe {samples} situații similare)."
            )
        except Exception:
            pass

    if drift:
        try:
            p10 = drift.get("p10")
            p50 = drift.get("p50")
            p90 = drift.get("p90")
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (p10, p50, p90)):
                print(
                    "Expected drift (24h) aproximativ: "
                    f"{p10 * 100:.1f}% ... {p50 * 100:.1f}% ... {p90 * 100:.1f}%."
                )
        except Exception:
            pass


# ============================
#  ENTRY POINT
# ============================

if __name__ == "__main__":
    main()
