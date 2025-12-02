"""
data_loader.py

Script simplu pentru descărcarea datelor zilnice BTC-USD
de la Yahoo Finance (prin yfinance) și salvarea lor în
data/btc_1d.csv într-un format compatibil cu backtest.py.

Rulare:

    python data_loader.py

"""

import os
from datetime import datetime

import pandas as pd
import yfinance as yf


DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "btc_1d.csv")


def download_btc_daily(start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Descarcă date zilnice BTC-USD de la Yahoo Finance.

    :param start: data de început (YYYY-MM-DD)
    :param end:   data de sfârșit (YYYY-MM-DD) sau None pentru până azi
    :return:      DataFrame cu coloanele: timestamp, open, high, low, close, volume
    """
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")

    print(f"[INFO] Descarc BTC-USD de la {start} la {end}...")

    df = yf.download("BTC-USD", start=start, end=end, interval="1d")

    if df.empty:
        raise RuntimeError("Nu am primit date de la Yahoo Finance. Verifică conexiunea la internet sau simbolul.")

    # Curățăm și redenumim coloanele ca să fie compatibile cu backtest.py
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.reset_index(inplace=True)
    df.rename(
        columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )

    # Asigurăm tipurile de date
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    print(f"[INFO] Am descărcat {len(df)} bare zilnice.")
    return df


def save_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Salvează DataFrame-ul în CSV la path-ul dat.
    Creează folderul dacă nu există.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"[INFO] Am creat directorul: {directory}")

    df.to_csv(path, index=False)
    print(f"[INFO] Am salvat fișierul CSV: {path}")


def main():
    # Poți ajusta perioada dacă vrei mai puține date
    start_date = "2010-01-01"
    df = download_btc_daily(start=start_date)
    save_to_csv(df, DATA_PATH)
    print("[DONE] Datele BTC zilnice sunt pregătite pentru backtest.py.")


if __name__ == "__main__":
    main()
