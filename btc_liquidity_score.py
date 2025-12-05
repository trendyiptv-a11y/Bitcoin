import os
import pandas as pd


DATA_PATH = os.path.join("data", "btc_daily.csv")


def compute_liquidity_from_daily_csv(path: str = DATA_PATH):
    """
    Calculează lichiditatea pieței pe baza fișierului btc_daily.csv.
    Returnează:
      - regime: ridicată / moderată / scăzută
      - strength: slabă / moderată / puternică
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Nu am găsit fișierul de date: {path}")

    df = pd.read_csv(path)

    # Asigură coloanele necesare
    expected = {"timestamp", "open", "high", "low", "close", "volume"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"Fișierul {path} nu conține coloanele obligatorii: {expected}"
        )

    df = df.sort_values("timestamp")

    # Avem nevoie de minim 7 zile pentru intensitate
    if len(df) < 7:
        raise ValueError("Fișierul trebuie să conțină cel puțin 7 zile de date.")

    last = df.iloc[-1]
    high = float(last["high"])
    low = float(last["low"])
    close = float(last["close"])

    # Range-ul ultimei zile
    if close == 0:
        liquidity_range = 0.0
    else:
        liquidity_range = (high - low) / close

    # Regim de lichiditate
    if liquidity_range < 0.02:             # <2%
        regime = "ridicată"
    elif liquidity_range < 0.04:           # 2% – 4%
        regime = "moderată"
    else:                                  # >4%
        regime = "scăzută"

    # Intensitate pe baza mediei ultimelor 7 zile
    df["range"] = (df["high"] - df["low"]) / df["close"]
    last7 = df["range"].tail(7)
    avg7 = last7.mean()

    if avg7 == 0:
        strength = "slabă"
    else:
        deviation = abs(liquidity_range - avg7) / avg7

        if deviation < 0.15:           # <15% deviere
            strength = "slabă"
        elif deviation < 0.35:         # 15–35%
            strength = "moderată"
        else:                          # >35%
            strength = "puternică"

    return {
        "regime": regime,
        "strength": strength,
        "liquidity_value": float(liquidity_range),
        "avg7": float(avg7),
    }
