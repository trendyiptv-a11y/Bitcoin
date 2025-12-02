import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

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
#  MESAJUL COEZIV
# ============================

def build_message(signal: str, price: float) -> str:
    """Textul coeziv pentru dashboard."""

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
    # 1. încărcăm datele IC
    df = load_ic_series()

    # 2. generăm semnalele coezive
    df = generate_signals(df)

    # 3. extragem ultimul punct
    last = df.iloc[-1]
    price = float(last["close"])
    signal = str(last["signal"])
    ts = last.name  # index datetime

    # 4. generăm mesajul
    message = build_message(signal, price)

    state = {
        "timestamp": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
        "price_usd": price,
        "signal": signal,
        "message": message,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # 5. scriem JSON în folderul frontend-ului
    output_path = os.path.join(STRATEGY_DIR, "coeziv_state.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print("Stare coezivă generată:", output_path)


# ============================
#  ENTRY POINT
# ============================

if __name__ == "__main__":
    main()
