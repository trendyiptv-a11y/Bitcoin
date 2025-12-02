"""
plot_backtest.py

Script pentru vizualizarea:
- prețului BTC (daily)
- equity curve a strategiei de swing

Rulare:

    python plot_backtest.py
"""

import pandas as pd
import matplotlib.pyplot as plt

PRICE_PATH = "data/btc_daily.csv"
EQUITY_PATH = "equity_curve.csv"


def load_price(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise KeyError("btc_daily.csv trebuie să aibă coloana 'timestamp'.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    return df["close"]


def load_equity(path: str) -> pd.Series:
    df = pd.read_csv(path)
    # equity_curve.csv are două coloane: index (timestamp) și equity
    # dacă indexul e salvat ca 'Unnamed: 0', îl tratăm ca dată
    if "Unnamed 0" in df.columns:
        df.rename(columns={"Unnamed 0": "timestamp"}, inplace=True)
    elif "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp")

    if "equity" not in df.columns:
        # equity e probabil singura coloană
        equity_col = [c for c in df.columns if c != "timestamp"][0]
        df.rename(columns={equity_col: "equity"}, inplace=True)

    return df["equity"]


def main():
    price = load_price(PRICE_PATH)
    equity = load_equity(EQUITY_PATH)

    # aliniem seriile pe perioada comună
    combined = pd.concat([price, equity], axis=1, join="inner")
    combined.columns = ["close", "equity"]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Timp")
    ax1.set_ylabel("Preț BTC (close)")
    ax1.plot(combined.index, combined["close"], label="BTC Close", alpha=0.7)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Equity")
    ax2.plot(combined.index, combined["equity"], label="Equity", alpha=0.7)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("BTC Swing Strategy – Preț vs Equity")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
