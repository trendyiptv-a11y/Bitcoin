import json
import os
import sys
import math
from datetime import datetime, timezone

import pandas as pd
import requests


# ============================
#  SETĂRI DE PATH
# ============================

BASE_DIR = os.path.dirname(__file__)
STRATEGY_DIR = os.path.join(BASE_DIR, "btc-swing-strategy")

# adăugăm folderul în sys.path
if STRATEGY_DIR not in sys.path:
    sys.path.append(STRATEGY_DIR)

from btc_swing_strategy import generate_signals


# ============================
#  ÎNCĂRCARE IC SERIES
# ============================

def load_ic_series(path=None):
    candidates = []

    if path is not None:
        candidates.append(path)
    else:
        candidates.extend([
            os.path.join(BASE_DIR, "ic_btc_series.json"),
            os.path.join(BASE_DIR, "j-btc-coeziv", "ic_btc_series.json"),
            os.path.join(BASE_DIR, "data", "ic_btc_series.json"),
        ])

    chosen = None
    for c in candidates:
        if os.path.exists(c):
            chosen = c
            break

    if chosen is None:
        raise FileNotFoundError("Nu am găsit ic_btc_series.json în locațiile:\n" + "\n".join(candidates))

    with open(chosen, "r") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw["series"])
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.set_index("date").sort_index()

    return df[["close", "ic_struct", "ic_dir", "ic_flux", "ic_cycle", "regime"]]


# ============================
#  PREȚ LIVE BTC
# ============================

def get_live_btc_price():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin", "vs_currencies": "usd"}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return float(data["bitcoin"]["usd"])


# ============================
#  MESAJ COEZIV
# ============================

def build_message(signal: str, price: float) -> str:
    if signal == "long":
        return (
            f"La prețul actual de ~{price:,.0f} USD, mecanismul vede context "
            f"favorabil acumulării. Decizia finală îți aparține."
        )
    elif signal == "short":
        return (
            f"La un preț de ~{price:,.0f} USD, mecanismul observă risc crescut "
            f"de scădere. Redu expunerea dacă este necesar."
        )
    else:
        return (
            f"Bitcoin se tranzacționează în jurul valorii de ~{price:,.0f} USD. "
            f"Mecanismul este neutru: context echilibrat."
        )
        # ============================
#  STATISTICĂ ISTORICĂ PENTRU SEMNAL
# ============================

def compute_signal_stats(
    df: pd.DataFrame,
    horizon_hours: int = 24,
    move_threshold: float = 0.005,
    min_samples: int = 100,
) -> dict:
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

    out = {
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

    # estimăm pasul de timp tipic
    idx = closes.index
    if len(idx) < 3:
        return out

    step_seconds = (idx[1] - idx[0]).total_seconds()
    if step_seconds <= 0:
        return out

    horizon_steps = max(1, round(horizon_hours * 3600 / step_seconds))

    # construim lista de (semnal, ret_24h)
    triplets = []
    for i in range(0, len(closes) - horizon_steps):
        sig = signals.iloc[i]
        if sig not in ("long", "short", "flat"):
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
    if last_sig not in ("long", "short", "flat"):
        return out

    # filtrăm doar cazurile cu semnal identic cu cel actual
    same = [(ts, ret) for (ts, sig, ret) in triplets if sig == last_sig]
    n = len(same)
    out["samples"] = n

    if n < max(1, min_samples):
        # avem ceva istoric, dar nu suficient de bogat pentru o probabilitate robustă
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
            # dacă din orice motiv percentilele nu pot fi calculate, ignorăm expected_drift
            pass

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
            # semnal flat: considerăm "în direcție" dacă rămâne într-un range mic
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
#  ISTORIC SEMNALE – SNAPSHOT-URI
# ============================

def build_signal_history(df_signals: pd.DataFrame, limit: int = 30) -> list[dict]:
    """
    Construiește un istoric simplificat al semnalelor pentru UI.
    Păstrăm doar ultimele `limit` snapshot-uri (implicit 30).
    """

    history: list[dict] = []
    if df_signals is None or df_signals.empty:
        return history

    tail = df_signals.tail(limit)

    for ts, row in tail.iterrows():
        sig = str(row.get("signal", "")).lower()
        if sig not in ("long", "short", "flat"):
            sig = "flat"

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

def classify_market_regime(row: pd.Series, dev_pct=None) -> dict | None:
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

    # extragem valorile brute, cu fallback la 0.0 dacă lipsesc
    try:
        regime_val = float(row.get("regime", 0.0))
    except Exception:
        regime_val = 0.0

    try:
        flux_val = float(row.get("ic_flux", 0.0))
    except Exception:
        flux_val = 0.0

    try:
        dir_val = float(row.get("ic_dir", 0.0))
    except Exception:
        dir_val = 0.0

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

    # semnal
    signal = str(last.get("signal", "flat")).lower()
    if signal not in ("long", "short", "flat"):
        signal = "flat"

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
    except Exception as e:
        # dacă API-ul nu merge, folosim prețul din model și lăsăm price_source="model"
        print("Nu am putut obține prețul live BTC. Folosesc prețul din model.", e)

    # 5. generăm mesajul pe baza semnalului + prețul folosit în text
    message = build_message(signal, price_for_text)

    # 6. istoric de semnale (ultimele N puncte)
    signal_history = build_signal_history(df, limit=30)

    # 7. statistică istorică pentru semnalul curent (inclusiv expected drift)
    stats = compute_signal_stats(
        df,
        horizon_hours=24,
        move_threshold=0.005,
        min_samples=100,
    )

    # 8. deviația curentă față de prețul modelului (pentru regim de piață)
    dev_pct = None
    try:
        if math.isfinite(model_price) and model_price > 0 and math.isfinite(price_for_text):
            dev_pct = (price_for_text - model_price) / model_price
    except Exception:
        dev_pct = None

    market_regime = classify_market_regime(last, dev_pct)

    # 9. construim starea finală
    state = {
        "timestamp": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
        "price_usd": price_for_text,          # ce vezi mare în UI
        "model_price_usd": model_price,       # prețul folosit în snapshot-ul IC
        "price_source": price_source,         # "spot" sau "model"
        "signal": signal,
        "message": message,

        "signal_history": signal_history,     # folosit de cardul de istoric

        "signal_probability": stats.get("probability"),
        "signal_prob_samples": stats.get("samples"),
        "signal_prob_horizon_hours": stats.get("horizon_hours"),
        "signal_prob_source": stats.get("source"),
        "signal_prob_breakdown": stats.get("breakdown"),
        "signal_expected_drift": stats.get("expected_drift"),

        "market_regime": market_regime,

        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # 10. scriem JSON în folderul frontend-ului
    output_path = os.path.join(STRATEGY_DIR, "coeziv_state.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    # 11. log simplu în consolă
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
        print(
            f"Probabilitate istorică (în direcția semnalului, {state['signal_prob_horizon_hours']}h): "
            f"~{prob * 100:.1f}% (bazat pe {samples} situații similare)."
        )

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

    print("Regim de piață:", regime_label)


# ============================
#  ENTRY POINT
# ============================

if __name__ == "__main__":
    main()
