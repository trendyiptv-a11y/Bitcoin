import pandas as pd

def run_backtest(df, fee=0.001):

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
