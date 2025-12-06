import json
import os
import sys
import math
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, List, Any

import pandas as pd
import requests

from btc_flow_score import compute_flow_from_daily_csv
from btc_liquidity_score import compute_liquidity_from_daily_csv  # 

# ============================
#  SETĂRI DE PATH
# ============================

BASE_DIR = os.path.dirname(__file__)
STRATEGY_DIR = os.path.join(BASE_DIR, "btc-swing-strategy")

if STRATEGY_DIR not in sys.path:
    sys.path.append(STRATEGY_DIR)

from btc_swing_strategy import generate_signals  # din btc-swing-strategy 2

# cost de producție – versiunea automată; fallback la None dacă nu există modulul
try:
    from btc_production_auto import estimate_production_cost
except ImportError:
    def estimate_production_cost(
        electricity_price_usd_per_kwh: float = 0.07,
        hw_efficiency_j_per_th: float = 25.0,
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

    # Majoritatea feedurilor au structura {"meta": ..., "series": [...]}
    if isinstance(raw, dict) and "series" in raw:
        series = raw["series"]
    else:
        series = raw

    df = pd.DataFrame(series)

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
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="raise")
        df = df.set_index(time_col).sort_index()

    # păstrăm doar coloanele necesare pentru generate_signals
    cols = [
        "close",
        "ic_struct",
        "ic_dir",
        "ic_flux",
        "ic_cycle",
        "regime",
    ]
    existing = [c for c in cols if c in df.columns]
    if "close" not in existing:
        raise ValueError("Seria IC nu conține coloana 'close' necesară modelului.")
    return df[existing]


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

    if last_sig == "neutral":
        last_sig = "flat"

    same: List[Tuple[pd.Timestamp, float]] = [
        (ts, ret) for (ts, sig, ret) in triplets if sig == last_sig
    ]
    n = len(same)
    out["samples"] = n

    if n < 1:
        return out

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
    din ultimul snapshot (regime, ic_flux) și, opțional, a deviației
    prețului față de model (dev_pct).
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
        if abs(flux_val) < 0.1:
            trend_label = "Range neutru"
            trend_code = "range_neutral"
        elif flux_val > 0:
            trend_label = "Range cu bias pozitiv"
            trend_code = "range_bias_up"
        else:
            trend_label = "Range cu bias negativ"
            trend_code = "range_bias_down"

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
    deviation_from_production: Optional[float] = None,  # păstrat în semnătură, nu îl mai folosim în text
) -> str:
    """
    Mesaj instituțional care explică:
      - contextul de piață identificat de mecanism (Market Context)
      - Flow Regime și Liquidity Regime
      - câteva repere istorice (frecvență, drift)
    """

    signal_norm = (signal or "").lower()
    if signal_norm == "long":
        ctx_label = "bias pozitiv al contextului de piață"
    elif signal_norm == "short":
        ctx_label = "bias negativ al contextului de piață"
    else:
        ctx_label = "context de piață neutru"

    parts: List[str] = []

    # 1. Context principal + preț de referință
    if isinstance(price, (int, float)) and math.isfinite(price) and price > 0:
        parts.append(
            f"În acest snapshot, Bitcoin este în jur de ~{price:,.0f} USD, "
            f"iar mecanismul identifică un {ctx_label}."
        )
    else:
        parts.append(
            f"Mecanismul a identificat un {ctx_label} pentru contextul curent al pieței."
        )

    # 2. Flow Regime
    if flow_bias in ("pozitiv", "negativ", "neutru"):
        if flow_bias == "pozitiv":
            txt = "Flow Regime indică o presiune de cumpărare"
        elif flow_bias == "negativ":
            txt = "Flow Regime indică o presiune de vânzare"
        else:
            txt = "Flow Regime indică un flux relativ echilibrat între cumpărători și vânzători"

        if flow_strength in ("slab", "moderat", "puternic"):
            txt += f" ({flow_strength})."
        else:
            txt += "."
        parts.append(txt)

    # 3. Liquidity Regime
    if liquidity_regime in ("scăzută", "moderată", "ridicată"):
        if liquidity_regime == "ridicată":
            txt = (
                "Liquidity Regime este unul de lichiditate ridicată, "
                "unde mișcările de preț tind să fie mai bine ancorate în fluxul agregat."
            )
        elif liquidity_regime == "scăzută":
            txt = (
                "Liquidity Regime este unul de lichiditate scăzută, "
                "iar mișcările de preț pot fi mai bruște și mai sensibile la ordine punctuale."
            )
        else:
            txt = (
                "Liquidity Regime este unul de lichiditate moderată, "
                "specific condițiilor obișnuite de tranzacționare."
            )
        parts.append(txt)

    # 4. Statistică istorică – frecvență + drift
    if stats:
        try:
            prob = stats.get("probability")
            samples = stats.get("samples") or 0
            horizon = stats.get("horizon_hours", 24)

            if isinstance(prob, (int, float)) and math.isfinite(prob) and samples > 0:
                pct = prob * 100.0
                parts.append(
                    f"Istoric, pentru contexte similare, aproximativ {pct:.1f}% dintre episoade "
                    f"au evoluat în direcția acestui context pe un orizont de circa {horizon} ore "
                    f"(pe baza a {samples} observații)."
                )
        except Exception:
            pass

        try:
            drift = stats.get("expected_drift") or {}
            p10 = drift.get("p10")
            p50 = drift.get("p50")
            p90 = drift.get("p90")
            horizon = drift.get("horizon_hours", 24)

            if all(
                isinstance(v, (int, float)) and math.isfinite(v)
                for v in (p10, p50, p90)
            ):
                parts.append(
                    f"În aceleași tipuri de contexte, pe un orizont de ~{horizon} ore, "
                    f"randamentele relative au avut de regulă un interval aproximativ "
                    f"între {p10 * 100:.1f}% și {p90 * 100:.1f}%, "
                    f"cu o valoare mediană în jur de {p50 * 100:.1f}%."
                )
        except Exception:
            pass

    parts.append(
        "Acest mesaj descrie contextul structural al pieței pe baza datelor istorice și a modelului intern "
        "și nu reprezintă o recomandare de tranzacționare."
    )

    return " ".join(parts)


# ============================
#  MAIN – GENERAREA STĂRII COEZIVE
# ============================

def main() -> None:
    # 1. încărcăm datele IC (structură, direcție, flux, regim)
    df = load_ic_series()
    df = generate_signals(df)
    if df is None or df.empty:
        raise RuntimeError("DataFrame-ul cu semnale este gol după generate_signals().")

    # 2. ultimul snapshot din serie
    last = df.iloc[-1]
    ts = last.name  # index datetime

    signal = str(last.get("signal", "neutral")).lower()
    if signal not in ("long", "short", "flat", "neutral"):
        signal = "neutral"

    try:
        model_price = float(last["close"])
    except Exception:
        raise RuntimeError("Coloana 'close' lipsește sau nu poate fi convertită la float.")

    # 3. preț spot live (fallback la model_price)
    price_source = "model"
    price_for_text = model_price
    try:
        live_price = get_live_btc_price()
        if math.isfinite(live_price) and live_price > 0:
            price_for_text = float(live_price)
            price_source = "spot"
    except Exception as e:
        print("Nu am putut obține prețul live BTC. Folosesc prețul din model.", e)

    # 4. cost de producție (opțional)
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
            if math.isfinite(ref_price) and ref_price > 0:
                deviation_from_production = (ref_price - production_cost) / production_cost
    except Exception:
        deviation_from_production = None

    # 5. deviația față de model (pentru regim de piață)
    dev_pct_model: Optional[float] = None
    try:
        if math.isfinite(model_price) and model_price > 0 and math.isfinite(price_for_text):
            dev_pct_model = (price_for_text - model_price) / model_price
    except Exception:
        dev_pct_model = None

    # 6. clasificare regim de piață
    market_regime = classify_market_regime(last, dev_pct_model)

    # 7. Flow Score & Liquidity Score din btc_daily.csv
    try:
        flow_raw = compute_flow_from_daily_csv()  # 3
        flow_score = float(flow_raw.get("flow_value", 0.0) or 0.0)
        flow_state = {
            "flow_score": flow_score,
            "flow_bias": flow_raw.get("bias"),
            "flow_strength": flow_raw.get("strength"),
            "flow_components": {
                "open": flow_raw.get("open"),
                "close": flow_raw.get("close"),
                "flow_value": flow_score,
                "bias": flow_raw.get("bias"),
                "strength": flow_raw.get("strength"),
            },
        }
    except Exception as e:
        print("Nu am putut calcula Flow Score din btc_daily.csv:", e)
        flow_state = {
            "flow_score": None,
            "flow_bias": None,
            "flow_strength": None,
            "flow_components": {"error": str(e)},
        }

    try:
        liq_raw = compute_liquidity_from_daily_csv()  # 4
        liq_score = float(liq_raw.get("liquidity_value", 0.0) or 0.0)
        liq_state = {
            "liquidity_score": liq_score,
            "liquidity_regime": liq_raw.get("regime"),
            "liquidity_strength": liq_raw.get("strength"),
            "liquidity_components": {
                "liquidity_value": liq_score,
                "regime": liq_raw.get("regime"),
                "strength": liq_raw.get("strength"),
                "avg7": liq_raw.get("avg7"),
            },
        }
    except Exception as e:
        print("Nu am putut calcula Liquidity Score din btc_daily.csv:", e)
        liq_state = {
            "liquidity_score": None,
            "liquidity_regime": None,
            "liquidity_strength": None,
            "liquidity_components": {"error": str(e)},
        }

    # 8. istoric + statistică istorică
    signal_history = build_signal_history(df, limit=30)
    stats = compute_signal_stats(
        df,
        horizon_hours=24,
        move_threshold=0.005,
        min_samples=100,
    )

    # 9. mesaj coeziv – folosim Flow & Liquidity din btc_daily
    message = build_message(
        signal=signal,
        price=price_for_text,
        flow_bias=flow_state["flow_bias"],
        flow_strength=flow_state["flow_strength"],
        liquidity_regime=liq_state["liquidity_regime"],
        stats=stats,
        deviation_from_production=deviation_from_production,
    )

    # 10. construim starea finală pentru frontend
    state: Dict[str, Any] = {
        "timestamp": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),

        # prețuri
        "price_usd": price_for_text,      # preț folosit în mesaj
        "model_price_usd": model_price,   # preț mecanism
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

        # cost de producție
        "production_cost_usd": production_cost,
        "production_cost_as_of": production_as_of,
        "deviation_from_production": deviation_from_production,

        # Flow Score (din btc_daily.csv)
        "flow_score": flow_state["flow_score"],
        "flow_bias": flow_state["flow_bias"],
        "flow_strength": flow_state["flow_strength"],
        "flow_components": flow_state["flow_components"],

        # Liquidity Score (din btc_daily.csv)
        "liquidity_score": liq_state["liquidity_score"],
        "liquidity_regime": liq_state["liquidity_regime"],
        "liquidity_strength": liq_state["liquidity_strength"],
        "liquidity_components": liq_state["liquidity_components"],
    }

    # 11. scriem JSON în folderul frontend-ului
    os.makedirs(STRATEGY_DIR, exist_ok=True)
    output_path = os.path.join(STRATEGY_DIR, "coeziv_state.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    # 12. log simplu în consolă
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

    if production_cost is not None and math.isfinite(production_cost):
        print(
            "Cost de producție BTC (fundamental):",
            f"{production_cost:,.2f} USD",
            "| Deviație față de cost:",
            "n/a"
            if deviation_from_production is None
            else f"{deviation_from_production * 100:.1f}%",
        )

    print("Regim de piață:", regime_label)
    print(
        "Flow Score:",
        flow_state["flow_score"],
        "| Flow bias:",
        flow_state["flow_bias"],
        "| Flow strength:",
        flow_state["flow_strength"],
    )
    print(
        "Liquidity Score:",
        liq_state["liquidity_score"],
        "| Regim lichiditate:",
        liq_state["liquidity_regime"],
        "| Forță lichiditate:",
        liq_state["liquidity_strength"],
    )


# ============================
#  ENTRY POINT
# ============================

if __name__ == "__main__":
    main()
