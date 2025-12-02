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


def main():
    # 1. încarcă datele exact din locul în care sunt în repo
    df = pd.read_csv("data/btc_daily.csv")

    # 2. rulează backtest-ul
    df_bt = run_backtest(df, fee=0.001)

    # 3. salvează equity curve-ul în fișierul așteptat de workflow
    df_bt.to_csv("equity_curve.csv", index=False)


if __name__ == "__main__":
    main()
