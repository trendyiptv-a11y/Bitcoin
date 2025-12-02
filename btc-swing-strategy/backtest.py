import pandas as pd


def run_backtest(df, fee=0.001):
    """
    Rulează backtest-ul pe un DataFrame care conține cel puțin coloanele:
    - 'close'  : prețul de închidere
    - 'signal' : 'flat' / 'long' / 'short'
    """
    df = df.copy()

    position = 0
    equity = 1.0
    last_price = df["close"].iloc[0]

    equity_curve = []

    for i in range(len(df)):
        price = df["close"].iloc[i]
        signal = df["signal"].iloc[i]

        # Închide poziția dacă semnalul se schimbă
        if signal == "flat":
            position = 0

        elif signal == "long":
            if position != 1:
                equity *= (1 - fee)
                position = 1

        elif signal == "short":
            if position != -1:
                equity *= (1 - fee)
                position = -1

        # Profit / pierdere zilnic
        if position == 1:
            equity *= price / last_price
        elif position == -1:
            equity *= last_price / price

        last_price = price
        equity_curve.append(equity)

    df["equity"] = equity_curve
    return df


def generate_signals(df, short_window=20, long_window=50, threshold=0.0):
    """
    Generează semnale de tip swing folosind un crossover de medii mobile:

    - 'long'  când SMA scurtă > SMA lungă
    - 'short' când SMA scurtă < SMA lungă
    - 'flat'  când nu avem date suficiente sau diferența e mică
    """
    df = df.copy()

    df["sma_short"] = df["close"].rolling(short_window).mean()
    df["sma_long"] = df["close"].rolling(long_window).mean()

    df["signal"] = "flat"  # implicit

    cond_long = df["sma_short"] > df["sma_long"] * (1 + threshold)
    cond_short = df["sma_short"] < df["sma_long"] * (1 - threshold)

    df.loc[cond_long, "signal"] = "long"
    df.loc[cond_short, "signal"] = "short"

    return df


def main():
    # Rulat din rădăcina repo-ului:
    #   python btc-swing-strategy/backtest.py
    # → cwd = root, deci citim direct din data/btc_daily.csv
    df = pd.read_csv("data/btc_daily.csv")

    # 1. generează semnale
    df_with_signals = generate_signals(df, short_window=20, long_window=50)

    # 2. rulează backtest-ul
    df_bt = run_backtest(df_with_signals, fee=0.001)

    # 3. salvează equity curve-ul în fișierul așteptat de workflow
    df_bt.to_csv("equity_curve.csv", index=False)


if __name__ == "__main__":
    main()
