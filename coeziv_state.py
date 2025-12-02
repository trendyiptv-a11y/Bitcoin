import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

from data_loader import load_ic_series  # e în root

# 🔧 adăugăm folderul btc-swing-strategy în sys.path
BASE_DIR = os.path.dirname(__file__)
STRATEGY_DIR = os.path.join(BASE_DIR, "btc-swing-strategy")
if STRATEGY_DIR not in sys.path:
    sys.path.append(STRATEGY_DIR)

from btc_swing_strategy import generate_signals  # fișierul tău din screenshot


def build_message(signal: str, price: float) -> str:
    """Textul coeziv, decis în backend."""
    if signal == "long":
        return (
            f"La prețul actual de ~{price:,.0f} USD, mecanismul coeziv "
            f"vede context favorabil de acumulare. Poți cumpăra, dar decizia finală e a ta."
        )
    elif signal == "short":
        return (
            f"În jurul valorii de ~{price:,.0f} USD, mecanismul coeziv vede "
            f"risc crescut de scădere. Poți vinde sau poți reduce expunerea."
        )
    else:
        return (
            f"Bitcoin se tranzacționează în jur de ~{price:,.0f} USD. "
            f"Mecanismul coeziv este neutru: poți cumpăra și poți vinde, "
            f"dar cel mai valoros poate fi să mai aștepți claritate."
        )


def main():
    # 1. încărcăm seria cu IC-uri (close + ic_struct + ic_dir + ic_flux + regime)
    df = load_ic_series("ic_btc_series.json")

    # 2. generăm semnalele coezive (din btc_swing_strategy.py)
    df = generate_signals(df)

    # 3. ultimul punct din serie
    last = df.iloc[-1]
    price = float(last["close"])
    signal = str(last["signal"])
    ts = last.name  # index datetime, cel mai probabil

    # 4. mesaj final
    message = build_message(signal, price)

    state = {
        "timestamp": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
        "price_usd": price,
        "signal": signal,
        "message": message,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # 5. scriem JSON lângă index.html
    out_path = os.path.join(BASE_DIR, "btc-swing-strategy", "coeziv_state.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
