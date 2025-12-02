import json
import os
import sys
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

# importă generate_signals din fișierul tău
from btc_swing_strategy import generate_signals


# ============================
#  ÎNCĂRCARE IC SERIES
# ============================

def load_ic_series(path=None):
    """
    Încarcă seria IC BTC din ic_btc_series.json.

    Caută fișierul în mai multe locații posibile:
    - ./ic_btc_series.json
    - ./j-btc-coeziv/ic_btc_series.json
    - ./data/ic_btc_series.json
    """

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
        raise FileNotFoundError(
            "Nu am găsit ic_btc_series.json în niciuna din locațiile așteptate:\n"
            + "\n".join(candidates)
        )

    with open(chosen, "r") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw["series"])
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.set_index("date").sort_index()

    # returnăm strict coloanele necesare pentru generate_signals
    return df[[
        "close",
        "ic_struct",
        "ic_dir",
        "ic_flux",
        "ic_cycle",
        "regime",
    ]]


# ============================
#  PREȚ LIVE BTC (API GRATUIT)
# ============================

def get_live_btc_price():
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

    if signal == "long":
        return (
            f"La prețul actual de ~{price:,.0f} USD, mecanismul coeziv vede "
            f"context favorabil pentru acumulare. Poți cumpăra, dar decizia finală îți aparține."
        )

    elif signal == "short":
        return (
            f"În jurul valorii de ~{price:,.0f} USD, mecanismul coeziv detectează "
            f"riscuri crescute de scădere. Poți vinde sau reduce expunerea."
        )

    else:
        return (
            f"Bitcoin se tranzacționează în jur de ~{price:,.0f} USD. "
            f"Mecanismul coeziv este neutru: poți cumpăra și poți vinde, "
            f"dar poate fi util să aștepți claritate suplimentară."
        )


# ============================
#  MAIN – GENERAREA STĂRII COEZIVE
# ============================

def main():
    # 1. încărcăm datele IC (structură, direcție, flux, regim)
    df = load_ic_series()

    # 2. generăm semnalele coezive
    df = generate_signals(df)

    # 3. extragem ultimul punct din serie
    last = df.iloc[-1]
    model_price = float(last["close"])   # prețul din snapshot-ul modelului
    signal = str(last["signal"])
    ts = last.name  # index datetime

    # 4. preț spot live (cu fallback la model_price)
    price_source = "model"
    price_for_text = model_price

    try:
        live_price = get_live_btc_price()
        price_for_text = live_price
        price_source = "spot"
    except Exception as e:
        # dacă API-ul nu merge, folosim prețul din model și lăsăm price_source="model"
        print("Nu am putut obține prețul live BTC. Folosesc prețul din model.", e)

    # 5. generăm mesajul pe baza semnalului + prețul folosit în text
    message = build_message(signal, price_for_text)

    state = {
        "timestamp": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
        "price_usd": price_for_text,        # ce vezi mare în UI
        "model_price_usd": model_price,     # prețul folosit în snapshot-ul IC
        "price_source": price_source,       # "spot" sau "model"
        "signal": signal,
        "message": message,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # 6. scriem JSON în folderul frontend-ului
    output_path = os.path.join(STRATEGY_DIR, "coeziv_state.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print("Stare coezivă generată:", output_path)
    print("Semnal:", signal, "| Sursă preț:", price_source, "| Preț mesaj:", price_for_text)


# ============================
#  ENTRY POINT
# ============================

if __name__ == "__main__":
    main()
