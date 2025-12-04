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

# adăugăm folderul btc-swing-strategy în sys.path
if STRATEGY_DIR not in sys.path:
    sys.path.append(STRATEGY_DIR)

# importă generate_signals din fișierul tău principal de strategie
from btc_swing_strategy import generate_signals


# ============================
#  ÎNCĂRCARE IC SERIES
# ============================

def load_ic_series(path: str | None = None) -> pd.DataFrame:
    """
    Încarcă seria IC BTC din ic_btc_series.json.

    Caută fișierul în mai multe locații posibile:
      - ./ic_btc_series.json
      - ./j-btc-coeziv/ic_btc_series.json
      - ./data/ic_btc_series.json

    Returnează un DataFrame indexat pe dată, cu cel puțin coloanele:
      - close
      - ic_struct
      - ic_dir
      - ic_flux
      - ic_cycle
      - regime
    """
    candidates: list[str] = []

    if path is not None:
        candidates.append(path)
    else:
        candidates.extend(
            [
                os.path.join(BASE_DIR, "ic_btc_series.json"),
                os.path.join(BASE_DIR, "j-btc-coeziv", "ic_btc_series.json"),
                os.path.join(BASE_DIR, "data", "ic_btc_series.json"),
            ]
        )

    chosen: str | None = None
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
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("date").sort_index()

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
#  MESAJUL COEZIV
# ============================

def build_message(signal: str, price: float) -> str:
    """Textul coeziv pentru dashboard, bazat pe semnal + prețul folosit în mesaj."""

    signal = (signal or "").lower()

    if signal == "long":
        return (
            f"La prețul actual de ~{price:,.0f} USD, mecanismul coeziv vede "
            f"context favorabil pentru acumulare. Poți cumpăra, dar decizia finală îți aparține."
        )

    if signal == "short":
        return (
            f"În jurul valorii de ~{price:,.0f} USD, mecanismul coeziv detectează "
            f"riscuri crescute de scădere. Poți vinde sau reduce expunerea."
        )

    # neutru / orice altceva
    return (
        f"Bitcoin se tranzacționează în jur de ~{price:,.0f} USD. "
        f"Mecanismul coeziv este neutru: poți cumpăra și poți vinde, "
        f"dar poate fi util să aștepți claritate suplimentară."
    )


# ============================
#  ISTORIC DE SEMNALE
# ============================

def build_signal_history(df: pd.DataFrame, limit: int = 30) -> list[dict]:
    """Construiește un istoric scurt de semnale pentru UI (ultimele `limit` snapshot-uri)."""
    history: list[dict] = []

    tail = df.tail(limit)
    for ts, row in tail.iterrows():
        sig = str(row.get("signal", "")).lower()
        if sig not in ("long", "short", "flat"):
            # normalizăm orice altceva la 'flat' pentru UI
            sig = "flat"

        close_val = row.get("close")
        try:
            close_val = float(close_val) if close_val is not None else None
        except (TypeError, ValueError):
            close_val = None

        history.append(
            {
                "timestamp": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
                "signal": sig,
                "model_price_usd": close_val,
            }
        )

    return history


# ============================
#  STATISTICĂ ISTORICĂ A SEMNALULUI
# ============================

def compute_signal_stats(
    df: pd.DataFrame,
    horizon_hours: int = 24,
    move_threshold: float = 0.005,
    min_samples: int = 100,
) -> dict:
    """Calculează probabilitatea istorică a semnalului curent + breakdown.

    Pentru fiecare snapshot istoric cu un semnal valid (long / short / flat),
    măsurăm randamentul pe o fereastră de `horizon_hours` în viitor și
    clasificăm:
      - "în direcție"   (mișcare relevantă în direcția semnalului)
      - "contra"        (mișcare relevantă în sens opus)
      - "flat/zgomot"   (mișcare mică, sub pragul move_threshold)

    Întoarce un dict cu cheile:
      - probability         (float sau None)
      - samples             (număr de cazuri comparabile)
      - horizon_hours       (int)
      - source              (string, de ex. "ohlc")
      - breakdown           (dict sau None) cu cheile in_direction / opposite / flat
    """

    out = {
        "probability": None,
        "samples": 0,
        "horizon_hours": horizon_hours,
        "source": "ohlc",
        "breakdown": None,
    }

    if df is None or df.empty:
        return out

    if "close" not in df.columns or "signal" not in df.columns:
        return out

    df = df.copy()
    df = df.dropna(subset=["close"])
    if df.empty:
        return out

    # normalizăm semnalele
    df["signal_clean"] = df["signal"].astype(str).str.lower()

    # deducem pasul de timp tipic (presupunem serie regulată)
    idx = df.index.to_series().sort_values()
    if len(idx) >= 2:
        step_sec = (idx.iloc[1] - idx.iloc[0]).total_seconds()
        if not step_sec or not math.isfinite(step_sec):
            step_sec = 3600.0
    else:
        step_sec = 3600.0

    horizon_steps = max(1, int(round(horizon_hours * 3600.0 / step_sec)))

    closes = df["close"].astype(float).values
    sig_series = df["signal_clean"]

    records: list[tuple[str, float]] = []
    limit = len(df) - horizon_steps
    for i in range(max(0, limit)):
        sig = sig_series.iloc[i]
        if sig not in ("long", "short", "flat"):
            continue

        p0 = closes[i]
        pH = closes[i + horizon_steps]
        if not (math.isfinite(p0) and math.isfinite(pH) and p0 > 0):
            continue

        ret = (pH - p0) / p0
        records.append((sig, ret))

    if not records:
        return out

    # filtrăm doar cazurile cu același semnal ca ultimul snapshot
    last_sig = sig_series.iloc[-1]
    same = [r for r in records if r[0] == last_sig]
    n = len(same)
    out["samples"] = n

    if n < max(1, min_samples):
        # avem ceva istoric, dar nu suficient de bogat pentru o probabilitate robustă
        return out

    in_dir = 0
    opp = 0
    flat = 0

    for _, ret in same:
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
        else:  # last_sig == "flat" sau altceva
            if abs(ret) < move_threshold:
                in_dir += 1  # tratăm "flat" drept succes când piața rămâne în range
            elif ret > 0:
                opp += 1
            else:
                flat += 1

    total = in_dir + opp + flat
    if total == 0:
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
#  MAIN – GENERAREA STĂRII COEZIVE
# ============================

def main() -> None:
    # 1. încărcăm datele IC (structură, direcție, flux, regim)
    df = load_ic_series()

    # 2. generăm semnalele coezive (completează coloana "signal")
    df = generate_signals(df)

    # 3. extragem ultimul punct din serie
    last = df.iloc[-1]
    model_price = float(last["close"])  # prețul din snapshot-ul modelului
    signal = str(last.get("signal", "flat"))
    ts = last.name  # index datetime

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

    # 5. generăm mesajul pe baza semnalului + prețul folosit în text
    message = build_message(signal, price_for_text)

    # 6. istoric de semnale (ultimele N puncte)
    signal_history = build_signal_history(df, limit=30)

    # 7. statistică istorică pentru semnalul curent
    stats = compute_signal_stats(df, horizon_hours=24, move_threshold=0.005, min_samples=100)

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
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # 8. scriem JSON în folderul frontend-ului
    output_path = os.path.join(STRATEGY_DIR, "coeziv_state.json")
    os.makedirs(STRATEGY_DIR, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print("Stare coezivă generată:", output_path)
    print(
        "Semnal:",
        signal,
        "| Sursă preț:",
        price_source,
        "| Preț mesaj:",
        price_for_text,
    )
    print("Istoric semnale livrat:", len(signal_history), "puncte")
    print(
        "Prob semnal:",
        stats.get("probability"),
        "| samples:",
        stats.get("samples"),
        "| breakdown:",
        stats.get("breakdown"),
    )


# ============================
#  ENTRY POINT
# ============================

if __name__ == "__main__":
    main()
