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

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if STRATEGY_DIR not in sys.path:
    sys.path.append(STRATEGY_DIR)

from btc_swing_strategy import generate_signals

from btc_swing_strategy import generate_signals

# Flow & Liquidity – pot fi opționale, tratăm robust și normalizăm structura
try:
    from btc_flow_score import compute_flow_from_daily_csv
except ImportError:
    compute_flow_from_daily_csv = None  # type: ignore[assignment]

try:
    from btc_liquidity_score import compute_liquidity_from_daily_csv
except ImportError:
    compute_liquidity_from_daily_csv = None  # type: ignore[assignment]


def _empty_flow() -> Dict[str, Any]:
    return {
        "flow_score": None,
        "flow_bias": None,
        "flow_strength": None,
        "components": {},
    }


def _empty_liquidity() -> Dict[str, Any]:
    return {
        "liquidity_score": None,
        "liquidity_regime": None,
        "liquidity_strength": None,
        "components": {},
    }


def compute_flow_from_file(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Wrapper peste btc_flow_score.compute_flow_from_daily_csv.
    Convertește structura (bias/strength/flow_value) în (flow_score/flow_bias/flow_strength)
    pe care o așteaptă restul mecanismului.
    """
    if compute_flow_from_daily_csv is None:
        return _empty_flow()

    default_path = os.path.join(BASE_DIR, "data", "btc_daily.csv")
    use_path = path or default_path

    try:
        raw = compute_flow_from_daily_csv(use_path)
    except Exception as e:
        print("Nu am putut calcula Flow Score din btc_flow_score:", e)
        return _empty_flow()

    if not isinstance(raw, dict):
        return _empty_flow()

    return {
        "flow_score": raw.get("flow_value"),
        "flow_bias": raw.get("bias"),
        "flow_strength": raw.get("strength"),
        "components": {},
    }


def compute_liquidity_from_file(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Wrapper peste btc_liquidity_score.compute_liquidity_from_daily_csv.
    Convertește (regime/strength/liquidity_value) în câmpurile așteptate de coeziv_state.
    """
    if compute_liquidity_from_daily_csv is None:
        return _empty_liquidity()

    default_path = os.path.join(BASE_DIR, "data", "btc_daily.csv")
    use_path = path or default_path

    try:
        raw = compute_liquidity_from_daily_csv(use_path)
    except Exception as e:
        print("Nu am putut calcula Liquidity Score din btc_liquidity_score:", e)
        return _empty_liquidity()

    if not isinstance(raw, dict):
        return _empty_liquidity()

    return {
        "liquidity_score": raw.get("liquidity_value"),
        "liquidity_regime": raw.get("regime"),
        "liquidity_strength": raw.get("strength"),
        "components": {},
    }

# cost de producție – versiunea automată; fallback la None dacă nu există modulul
try:
    from btc_production_auto import estimate_production_cost
except ImportError:
    def estimate_production_cost(
        electricity_price_usd_per_kwh: float = 0.07,
        hw_efficiency_j_per_th: float = 25.0,
        profile: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[str]]:
        return None, None


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
    else:
        candidates.extend(
            [
                os.path.join(BASE_DIR, "ic_btc_series.json"),
                os.path.join(BASE_DIR, "j-btc-coeziv", "ic_btc_series.json"),
                os.path.join(BASE_DIR, "data", "ic_btc_series.json"),
                os.path.join(STRATEGY_DIR, "ic_btc_series.json"),
            ]
        )

    chosen: Optional[str] = None
    for c in candidates:
        if os.path.exists(c):
            chosen = c
            break

    if chosen is None:
        raise FileNotFoundError(
            "Nu am găsit ic_btc_series.json în niciuna din locațiile așteptate:\n"
            + "\n".join(candidates)
        )

    with open(chosen, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw["series"])
    # presupunem timestamp în ms în coloana "t"
    if "t" in df.columns:
        df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.set_index("date").sort_index()
    else:
        # fallback: căutăm coloană de timp generică
        time_col = None
        for cand in ["timestamp", "time", "date", "datetime"]:
            if cand in df.columns:
                time_col = cand
                break
        if time_col is None:
            raise ValueError("Nu am găsit coloană de timp în ic_btc_series.json.")
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.set_index(time_col).sort_index()

    return df[
        [
            "close",
            "ic_struct",
            "ic_dir",
            "ic_flux",
            "ic_cycle",
            "regime",
        ]
    ]


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
    Calculează statistică istorică pentru semnalul curent.

    - horizon_hours: orizontul de timp pe care măsurăm randamentul (ex: 24h)
    - move_threshold: prag relativ (ex: 0.005 = 0.5%) pentru a considera mișcarea "relevantă"
    - min_samples: număr minim de cazuri similare pentru o probabilitate robustă

    Întoarce un dict cu:
      {
        "probability": p_in_direction (sau None),
        "samples": n,
        "horizon_hours": horizon_hours,
        "source": "ohlc",
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
        "source": "ohlc",
        "breakdown": None,
        "expected_drift": None,
    }

    if df is None or df.empty:
        return out

    if "close" not in df.columns or "signal" not in df.columns:
        return out

    df = df.sort_index()
    if df.index.tz is None:
        df = df.tz_localize("UTC")

    closes = df["close"].astype(float)
    signals = df["signal"].astype(str).str.lower().replace({"nan": ""})

    idx = closes.index
    if len(idx) < 3:
        return out

    step_seconds = (idx[1] - idx[0]).total_seconds()
    if step_seconds <= 0:
        return out

    horizon_steps = max(1, round(horizon_hours * 3600 / step_seconds))

    triplets: List[Tuple[pd.Timestamp, str, float]] = []
    for i in range(0, len(closes) - horizon_steps):
        sig = signals.iloc[i]
        if sig not in ("long", "short", "flat", "neutral"):
            continue

        p0 = closes.iloc[i]
        pH = closes.iloc[i + horizon_steps]
        if not (math.isfinite(p0) and math.isfinite(pH) and p0 > 0):
            continue

        ret = (pH - p0) / p0
        ts0 = closes.index[i]
        triplets.append((ts0, sig, ret))

    if not triplets:
        return out

    last_sig = signals.iloc[-1]
    if last_sig not in ("long", "short", "flat", "neutral"):
        return out

    # normalizăm "neutral" -> "flat"
    if last_sig == "neutral":
        last_sig = "flat"

    # filtrăm doar cazurile cu semnal identic cu cel actual
    same: List[Tuple[pd.Timestamp, float]] = [
        (ts, ret) for (ts, sig, ret) in triplets if sig == last_sig
    ]
    n = len(same)
    out["samples"] = n

    if n < 1:
        return out

    # distribuția randamentelor în contexte similare (pentru expected_drift)
    rets = [ret for _, ret in same if isinstance(ret, (int, float)) and math.isfinite(ret)]
    if rets:
        s = pd.Series(rets)
        try:
            p10 = float(s.quantile(0.10))
            p50 = float(s.quantile(0.50))
            p90 = float(s.quantile(0.90))
            out["expected_drift"] = {
                "horizon_hours": horizon_hours,
                "p10": p10,
                "p50": p50,
                "p90": p90,
            }
        except Exception:
            pass

    # dacă nu avem suficiente cazuri, păstrăm doar expected_drift (dacă s-a putut calcula)
    if n < max(1, min_samples):
        return out

    in_dir = 0
    opp = 0
    flat = 0

    for _, ret in same:
        if not (isinstance(ret, (int, float)) and math.isfinite(ret)):
            continue

        if last_sig == "long":
            if ret >= move_threshold:
                in_dir += 1
            elif ret <= -move_threshold:
                opp += 1
            else:
                flat += 1
        elif last_sig == "short":
            if ret <= -move_threshold:
                in_dir += 1
            elif ret >= move_threshold:
                opp += 1
            else:
                flat += 1
        else:
            # semnal flat/neutral: "în direcție" = rămâne într-un range mic
            if abs(ret) < move_threshold:
                in_dir += 1
            elif ret > 0:
                opp += 1
            else:
                flat += 1

    total = in_dir + opp + flat
    if total <= 0:
        return out

    prob_dir = in_dir / total

    out["probability"] = prob_dir
    out["breakdown"] = {
        "in_direction": prob_dir,
        "opposite": opp / total,
        "flat": flat / total,
    }

    return out


# ============================
#  ISTORIC SEMNALE / CONTEXTE
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
        if sig not in ("long", "short", "flat", "neutral"):
            sig = "neutral"

        close_val = row.get("close", None)
        try:
            close_val = float(close_val) if close_val is not None else None
        except Exception:
            close_val = None

        history.append(
            {
                "timestamp": ts.isoformat(),
                "signal": sig,
                "model_price_usd": close_val,
            }
        )

    return history


# ============================
#  CLASIFICATOR DE REGIM DE PIAȚĂ
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
        regime_val = float(row.get("regime", 0.0))
    except Exception:
        regime_val = 0.0

    try:
        flux_val = float(row.get("ic_flux", 0.0))
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

    return {
        "label": label,
        "code": code,
    }

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


def _fg_zone(score: float) -> str:
    score = _clamp(score, 0.0, 100.0)
    if score < 20.0:
        return "extreme_fear"
    if score < 40.0:
        return "fear"
    if score < 60.0:
        return "neutral"
    if score < 80.0:
        return "greed"
    return "extreme_greed"


def _compute_fg_structural(state: Dict[str, Any]) -> float:
    """
    FG structural (0–100) derivat din:
      - market_regime.code
      - deviation_from_production
    """
    regime = state.get("market_regime") or {}
    code = str(regime.get("code", ""))

    # bază după trend (doar prefixul contează)
    if code.startswith("up_trend_strong"):
        regime_base = 80.0
    elif code.startswith("up_trend_moderate"):
        regime_base = 70.0
    elif code.startswith("down_trend_strong"):
        regime_base = 20.0
    elif code.startswith("down_trend_moderate"):
        regime_base = 30.0
    elif code.startswith("range_bias_up"):
        regime_base = 60.0
    elif code.startswith("range_bias_down"):
        regime_base = 40.0
    elif code.startswith("range_neutral"):
        regime_base = 50.0
    else:
        regime_base = 50.0  # fallback neutru

    dev = state.get("deviation_from_production")
    try:
        dev = float(dev) if dev is not None else 0.0
    except Exception:
        dev = 0.0

    dev_clamped = _clamp(dev, -0.5, 1.0)       # [-50%, +100%]
    dev_score = 50.0 + dev_clamped * 50.0      # -0.5 -> 25, 0 -> 50, +1 -> 100

    fg_struct = 0.5 * regime_base + 0.5 * dev_score
    return _clamp(fg_struct, 0.0, 100.0)


def _compute_fg_tactical(state: Dict[str, Any]) -> float:
    """
    FG tactic (24h) derivat din:
      - signal
      - signal_prob_breakdown.{in_direction, opposite}
      - signal_expected_drift.{p10, p90}
    """
    signal = str(state.get("signal", "neutral")).lower()

    breakdown = state.get("signal_prob_breakdown") or {}
    p_in = float(breakdown.get("in_direction") or 0.0)
    p_opp = float(breakdown.get("opposite") or 0.0)

    drift = state.get("signal_expected_drift") or {}
    p10 = float(drift.get("p10") or 0.0)
    p90 = float(drift.get("p90") or 0.0)

    # intensitatea semnalului (edge)
    edge_mag = abs(p_in - p_opp)
    intensity = _clamp(edge_mag / 0.25, 0.0, 1.0)  # 0.25 edge ~ full strength

    if signal == "short":
        fear_side = 1.0
    elif signal == "long":
        fear_side = -1.0
    else:
        fear_side = 0.0

    fg_dir = 50.0 - fear_side * intensity * 40.0  # short puternic -> ~10, long puternic -> ~90

    # volatilitatea intervalului așteptat
    vol_range = p90 - p10            # ex. 0.06 ~ 6%
    vol_mag = _clamp(vol_range / 0.08, 0.0, 1.0)  # 8% range ~ 1.0

    stress_adjust = (vol_mag - 0.5) * 20.0        # -10 .. +10

    fg_tactical = fg_dir - fear_side * stress_adjust
    return _clamp(fg_tactical, 0.0, 100.0)


def _combine_fg(fg_struct: float, fg_tactical: float) -> float:
    """
    Structura ancorează, tacticul deviază în interiorul spațiului rămas până la extreme.
    """
    fg_struct = _clamp(fg_struct, 0.0, 100.0)
    fg_tactical = _clamp(fg_tactical, 0.0, 100.0)

    max_dev = min(fg_struct, 100.0 - fg_struct)
    emotion_delta = (fg_tactical - 50.0) / 50.0   # -1..+1

    fg_combined = fg_struct + emotion_delta * max_dev
    return _clamp(fg_combined, 0.0, 100.0)


def _enrich_state_with_fg(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adaugă blocul `fg` în state:
      "fg": {
        "structural": ...,
        "tactical_24h": ...,
        "combined": ...,
        "tension": ...,
        "structural_zone": "...",
        "tactical_zone": "...",
        "combined_zone": "..."
      }
    """
    fg_struct = _compute_fg_structural(state)
    fg_tact = _compute_fg_tactical(state)
    fg_comb = _combine_fg(fg_struct, fg_tact)
    tension = abs(fg_struct - fg_tact)

    fg_block = {
        "structural": round(fg_struct, 2),
        "tactical_24h": round(fg_tact, 2),
        "combined": round(fg_comb, 2),
        "tension": round(tension, 2),
        "structural_zone": _fg_zone(fg_struct),
        "tactical_zone": _fg_zone(fg_tact),
        "combined_zone": _fg_zone(fg_comb),
    }

    new_state = dict(state)
    new_state["fg"] = fg_block
    return new_state

# ============================
#  MESAJUL COEZIV PENTRU UTILIZATOR
# ============================

def build_message(
    signal: str,
    price: float,
    flow_bias: Optional[str] = None,
    flow_strength: Optional[str] = None,
    liquidity_regime: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    deviation_from_production: Optional[float] = None,
) -> str:
    """
    Construiește un mesaj text simplu, dar profesionist, care explică:
      - ce vede mecanismul acum (context)
      - ce spun fluxul și lichiditatea
      - ce s-a întâmplat în contexte similare (expected drift)
    """
    signal = (signal or "").lower()
    ctx = "neutru"

    if signal == "long":
        ctx = "presiune de creștere"
    elif signal == "short":
        ctx = "risc de scădere"
    else:
        ctx = "neutru"

    parts: List[str] = []

    # 1. context principal
    parts.append(
        f"Prețul de tranzacționare la actualizarea sistemului: în jur de ~{price:,.0f} USD. "
        f"Mecanismul vede un {ctx}."
    )

    # 2. flow (flux de piață)
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
            txt = "Lichiditatea este într-o zonă normală pentru acest regim de piață."
        parts.append(txt)

    # 4. cost de producție (dacă avem deviație)
    if deviation_from_production is not None and math.isfinite(deviation_from_production):
        pct = deviation_from_production * 100.0
        if pct < -20:
            parts.append(
                "Prețul este semnificativ sub costul estimat de producție al rețelei, "
                "zonă asociată istoric cu stres ridicat asupra minerilor."
            )
        elif pct < -5:
            parts.append(
                "Prețul este ușor sub costul estimat de producție, ceea ce indică "
                "presiune ridicată pe minerii mai puțin eficienți."
            )
        elif pct <= 20:
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

    # prețul din snapshot-ul modelului
    try:
        model_price = float(last["close"])
    except Exception:
        raise RuntimeError("Coloana 'close' lipsește sau nu poate fi convertită la float.")

    # 4. preț spot live (cu fallback la model_price)
    price_source = "model"
    price_for_text = model_price

    try:
        live_price = get_live_btc_price()
        if math.isfinite(live_price) and live_price > 0:
            price_for_text = float(live_price)
            price_source = "spot"
    except Exception as e:  # log, dar nu cădem
        print("Nu am putut obține prețul live BTC. Folosesc prețul din model.", e)

    # 5. cost de producție BTC – multiple profile (cheap / average / expensive)
    production_costs: Dict[str, Optional[float]] = {}
    production_as_of: Optional[str] = None

    try:
        cheap_cost, as_of = estimate_production_cost(profile="cheap")
        avg_cost, as_of = estimate_production_cost(profile="average")
        exp_cost, as_of = estimate_production_cost(profile="expensive")

        production_costs = {
            "cheap": cheap_cost,
            "average": avg_cost,
            "expensive": exp_cost,
        }
        production_as_of = as_of
    except Exception as e:
        print("Nu am putut estima costurile de producție BTC.", e)
        production_costs = {}
        production_as_of = None

    # deviația pentru mesajul principal o calculăm față de profilul 'average'
    deviation_from_production: Optional[float] = None
    try:
        ref_cost = production_costs.get("average")
        if ref_cost is not None and math.isfinite(ref_cost) and ref_cost > 0:
            ref_price = (
                price_for_text
                if (math.isfinite(price_for_text) and price_for_text > 0)
                else model_price
            )
            if math.isfinite(ref_price) and ref_price > 0:
                deviation_from_production = (ref_price - ref_cost) / ref_cost
    except Exception:
        deviation_from_production = None

# deviația față de prețul modelului (pentru market regime)
    dev_pct_model: Optional[float] = None
    try:
        if (
            model_price is not None and math.isfinite(model_price) and model_price > 0
            and price_for_text is not None and math.isfinite(price_for_text)
        ):
            dev_pct_model = (price_for_text - model_price) / model_price
    except Exception:
        dev_pct_model = None
    
    # 7. clasificarea regimului de piață
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

        # regim de piață
        "market_regime": market_regime,

        # cost de producție (multiple profile)
        "production_costs_usd": production_costs,
        "production_cost_reference": "average",  # profilul folosit pentru deviație și mesaje
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
    # 12b. === Fear & Greed Coeziv ===
    state = _enrich_state_with_fg(state)
    # 13. scriem JSON în folderul frontend-ului
    os.makedirs(STRATEGY_DIR, exist_ok=True)
    output_path = os.path.join(STRATEGY_DIR, "coeziv_state.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    # 14. log simplu în consolă
    print("Stare coezivă generată:", output_path)
    print(
        "Semnal:", signal,
        "| Sursă preț:", price_source,
        "| Preț mesaj:", f"{price_for_text:,.2f} USD",
        "| Preț model:", f"{model_price:,.2f} USD",
    )

    prob = state.get("signal_probability")
    samples = state.get("signal_prob_samples")
    drift = state.get("signal_expected_drift") or {}
    regime_label = (market_regime or {}).get("label", "n/a")

    if prob is not None and samples:
        try:
            print(
                f"Probabilitate istorică (în contextul actual, "
                f"{state['signal_prob_horizon_hours']}h): "
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
                    f"{p10 * 100:.1f}% ... {p50 * 100:.1f}% ... {p90 * 100:.1f}%"
                )
        except Exception:
            pass

    # log cost de producție – folosim profilul 'average' ca referință
    ref_cost = production_costs.get("average")
    if ref_cost is not None and math.isfinite(ref_cost):
        print(
            "Cost de producție BTC (fundamental, profil average):",
            f"{ref_cost:,.2f} USD",
            "| Deviație față de cost:",
            "n/a"
            if deviation_from_production is None
            else f"{deviation_from_production * 100:.1f}%",
        )

        # dacă vrei, poți loga și celelalte profile
        cheap_cost = production_costs.get("cheap")
        exp_cost = production_costs.get("expensive")
        print(
            "Costuri producție (cheap / average / expensive):",
            "cheap=" + (f"{cheap_cost:,.2f} USD" if cheap_cost is not None else "n/a"),
            "| average=" + f"{ref_cost:,.2f} USD",
            "| expensive=" + (f"{exp_cost:,.2f} USD" if exp_cost is not None else "n/a"),
        )

    print("Regim de piață:", regime_label)
    print(
        "Flow Score:",
        flow.get("flow_score"),
        "| Flow bias:",
        flow.get("flow_bias"),
        "| Flow strength:",
        flow.get("flow_strength"),
    )
    print(
        "Liquidity Score:",
        liq.get("liquidity_score"),
        "| Regim lichiditate:",
        liq.get("liquidity_regime"),
        "| Forță lichiditate:",
        liq.get("liquidity_strength"),
    )


# ============================
#  ENTRY POINT
# ============================

if __name__ == "__main__":
    main()
