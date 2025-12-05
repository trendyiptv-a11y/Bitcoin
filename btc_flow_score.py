import os
import pandas as pd


DATA_PATH = os.path.join("data", "btc_daily.csv")


def compute_flow_from_daily_csv(path: str = DATA_PATH):
    """
    Calculează fluxul pieței pe ultimele 24h pe baza fișierului btc_daily.csv.
    Returnează:
      - bias: pozitiv / negativ / neutru
      - strength: slab / moderat / puternic
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Nu am găsit fișierul de date: {path}")

    df = pd.read_csv(path)

    # Asigurare coloane minime
    required_cols = {"open", "high", "low", "close", "volume"}

    # Acceptăm fie 'timestamp', fie 'date' ca și coloană de timp
    if "timestamp" in df.columns:
        timestamp_col = "timestamp"
    elif "date" in df.columns:
        timestamp_col = "date"
    else:
        raise ValueError(
            f"Fișierul {path} nu conține coloana obligatorie 'timestamp' sau 'date'. "
            f"Coloane găsite: {set(df.columns)}"
        )

    # Verificăm restul coloanelor
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Fișierul {path} nu conține coloanele obligatorii: {required_cols}. "
            f"Coloane găsite: {set(df.columns)}"
        )

    # Normalizăm numele coloanei de timp și sortăm
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df.sort_values("timestamp")

    # Alegem ultima zi
    last = df.iloc[-1]

    # Calculează fluxul zilnic
    open_p = float(last["open"])
    close_p = float(last["close"])
    if open_p == 0:
        flow = 0.0
    else:
        flow = (close_p - open_p) / open_p

    # Determinare bias
    if flow > 0.01:            # +1%
        bias = "pozitiv"
    elif flow < -0.01:         # -1%
        bias = "negativ"
    else:
        bias = "neutru"

    # Determinare strength
    abs_flow = abs(flow)

    if abs_flow < 0.01:        # <1%
        strength = "slab"
    elif abs_flow < 0.025:     # 1% – 2.5%
        strength = "moderat"
    else:                      # >2.5%
        strength = "puternic"

    return {
        "bias": bias,
        "strength": strength,
        "flow_value": float(flow),
        "open": open_p,
        "close": close_p,
    }
