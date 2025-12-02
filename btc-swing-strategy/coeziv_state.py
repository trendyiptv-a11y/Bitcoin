import json
from datetime import datetime, timezone

import pandas as pd

from btc_swing_strategy import generate_signals   # funcția ta coezivă
from data_loader import load_ic_series            # dacă numele e altul, adaptează


def build_message(signal: str, price: float) -> str:
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
    # 1. încărcăm seria ta cu IC-uri
    df = load_ic_series("ic_btc_series.json")  # adaptează path-ul dacă e altul

    # 2. generăm semnalele coezive (intern poate folosi și .pt)
    df = generate_signals(df)

    # 3. luăm ultimul punct
    last = df.iloc[-1]
    price = float(last["close"])
    signal = str(last["signal"])
    ts = last.name  # index datetime, cel mai probabil

    message = build_message(signal, price)

    state = {
        "timestamp": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
        "price_usd": price,
        "signal": signal,
        "message": message,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = "btc-swing-strategy/coeziv_state.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
