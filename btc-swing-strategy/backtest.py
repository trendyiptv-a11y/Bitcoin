"""
backtest.py

Backtest simplu pentru BTCSwingStrategy folosind date zilnice BTC dintr-un CSV.

Presupuneri:
- folosim doar timeframe 1D (ca "bară" de execuție) pentru versiunea 1.
- indicatorii coezivi CT, CC, CS, CN sunt aproximați foarte simplu (TODO: de înlocuit cu modelul tău).
- trailing HL/LH sunt aproximați cu min/max local pe câteva zile.

Pentru rulare:

    python backtest.py

Asigură-te că ai fișierul data/btc_1d.csv cu coloanele:
timestamp,open,high,low,close,volume
"""

import os
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd
import numpy as np

from btc_swing_strategy import (
    BTCSwingStrategy,
    Regime,
    Bias,
    TradeSide,
    Trade,
)


DATA_PATH = "data/btc_1d.csv"


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Trade]
    starting_equity: float
    ending_equity: float
    pnl_percent: float
    num_trades: int


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nu găsesc fișierul de date: {path}")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # calculează MA50 și MA200
    df["ma50"] = df["close"].rolling(window=50).mean()
    df["ma200"] = df["close"].rolling(window=200).mean()

    return df.dropna()  # pentru a elimina primele rânduri fără MA200


def detect_trend_flags(df: pd.DataFrame, idx: pd.Timestamp) -> dict:
    """
    Detectează uptrend/downtrend simplificat:
    - uptrend: close > ma50 și ma50 > ma200
    - downtrend: close < ma50 și ma50 < ma200
    """
    row = df.loc[idx]
    uptrend_1d = row["close"] > row["ma50"] > row["ma200"]
    downtrend_1d = row["close"] < row["ma50"] < row["ma200"]
    golden_cross = row["ma50"] > row["ma200"]
    death_cross = row["ma50"] < row["ma200"]
    return dict(
        uptrend_1d=uptrend_1d,
        downtrend_1d=downtrend_1d,
        golden_cross=golden_cross,
        death_cross=death_cross,
    )


def approx_cohesive_indices(df: pd.DataFrame, idx: pd.Timestamp) -> tuple:
    """
    Aproximare foarte simplă pentru CT, CC, CS, CN, doar ca exemplu.

    TODO:
    - înlocuiește cu modelul tău real de indicatori coezivi.
    """

    # CT: coeziune trend ~ cât de "smooth" e trendul pe ultimele 180 zile
    window = 180
    hist = df["close"].iloc[max(0, df.index.get_loc(idx) - window + 1) : df.index.get_loc(idx) + 1]
    if len(hist) < 30:
        return 0.5, 0.5, 0.5, 0.5

    log_price = np.log(hist.values)
    x = np.arange(len(log_price))
    slope, intercept = np.polyfit(x, log_price, 1)
    fitted = intercept + slope * x
    ss_res = np.sum((log_price - fitted) ** 2)
    ss_tot = np.sum((log_price - np.mean(log_price)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    CT = float(np.clip(r2, 0, 1))

    # CC: poziție aproximativă în ciclu (folosim % față de minim și maxim din ultimii 4 ani)
    years4 = 365 * 4
    hist4 = df["close"].iloc[max(0, df.index.get_loc(idx) - years4 + 1) : df.index.get_loc(idx) + 1]
    if len(hist4) < 30:
        CC = 0.5
    else:
        low4, high4 = hist4.min(), hist4.max()
        if high4 == low4:
            CC = 0.5
        else:
            CC = float((hist4.iloc[-1] - low4) / (high4 - low4))  # între 0 și 1

    # CS: structural cohesion ~ cât de aproape e prețul de MA200 (mai aproape = mai stabil)
    row = df.loc[idx]
    cs_raw = 1 - abs(row["close"] - row["ma200"]) / row["ma200"]
    CS = float(np.clip(cs_raw, 0, 1))

    # CN: narativ cohesion ~ aici îl aproximăm ca funcție de drawdown (bull = narativ puternic)
    max_close = hist4.max() if len(hist4) > 0 else row["close"]
    drawdown = (row["close"] - max_close) / max_close
    CN = float(np.clip(1 + drawdown, 0, 1))  # dacă suntem aproape de top, CN ~ 1

    return CT, CC, CS, CN


def detect_structural_levels(df: pd.DataFrame, idx: pd.Timestamp, lookback: int = 10) -> tuple:
    """
    Aproximare pentru:
    - last_higher_low (HL)
    - last_lower_high (LH)
    folosind min/max local pe ultimele N zile.
    """

    loc = df.index.get_loc(idx)
    start = max(0, loc - lookback)
    window = df.iloc[start : loc + 1]

    # HL ~ minimului local al ultimelor câteva zile
    last_hl = window["low"].min()
    # LH ~ maximul local al ultimelor câteva zile
    last_lh = window["high"].max()

    return last_hl, last_lh


def run_backtest(
    starting_equity: float = 10_000.0,
    buffer: float = 0.01,
) -> BacktestResult:
    df = load_data(DATA_PATH)

    strategy = BTCSwingStrategy(account_equity=starting_equity)
    equity_curve = []

    # vom presupune că datele sunt zilnice; folosim close drept "current_price"
    for idx, row in df.iterrows():
        current_price = row["close"]

        # 1. Calculează trend flags
        trend_flags = detect_trend_flags(df, idx)

        # 2. Calculează indicatorii coezivi (foarte simplificați)
        CT, CC, CS, CN = approx_cohesive_indices(df, idx)

        # 3. Update macro/bias zilnic
        strategy.daily_update_macro(
            CT=CT,
            CC=CC,
            CS=CS,
            CN=CN,
            uptrend_1d=trend_flags["uptrend_1d"],
            downtrend_1d=trend_flags["downtrend_1d"],
            golden_cross=trend_flags["golden_cross"],
            death_cross=trend_flags["death_cross"],
        )

        # 4. Detectează niveluri structurale HL/LH aproximativ
        last_hl, last_lh = detect_structural_levels(df, idx, lookback=5)

        # 5. Construiește semnale de intrare simplificate (doar exemplu)
        # Impuls: zi cu range relativ mare > medie
        range_today = row["high"] - row["low"]
        avg_range_20 = (df["high"] - df["low"]).rolling(20).mean().loc[idx]
        impuls_up = row["close"] > row["open"] and range_today > avg_range_20
        impuls_down = row["close"] < row["open"] and range_today > avg_range_20

        # Retracements simplificate: folosim variația față de ma20
        ma20 = df["close"].rolling(20).mean().loc[idx]
        if not np.isnan(ma20) and ma20 != 0:
            retracement_percent = max(0.0, (ma20 - current_price) / ma20 * 100)  # pentru long
            retracement_percent_up = max(0.0, (current_price - ma20) / ma20 * 100)  # pentru short
        else:
            retracement_percent = 0.0
            retracement_percent_up = 0.0

        orderly_pullback = True  # placeholder
        breakout = impuls_up     # placeholder: considerăm impulsul ca breakout
        breakdown = impuls_down  # idem

        long_signal_data = dict(
            impuls_up=impuls_up,
            retracement_percent=retracement_percent,
            orderly_pullback=orderly_pullback,
            breakout=breakout,
            entry_price=current_price,
            pullback_low=row["low"],
        )

        short_signal_data = dict(
            impuls_down=impuls_down,
            retracement_percent_up=retracement_percent_up,
            has_lower_high=True,  # placeholder
            breakdown=breakdown,
            entry_price=current_price,
            recent_swing_high=row["high"],
        )

        # 6. Rulează logica strategiei pentru bara curentă
        strategy.on_new_bar(
            current_price=current_price,
            long_signal_data=long_signal_data,
            short_signal_data=short_signal_data,
            last_higher_low=last_hl,
            last_lower_high=last_lh,
            buffer=current_price * buffer,  # buffer ca % din preț
        )

        # 7. Actualizează equity (foarte simplificat: le considerăm mark-to-market liniar)
        #    Aici, pentru versiunea 1, doar o aproximăm ca fiind constantă + profit/pierderi realizate
        #    (nu calculăm PnL flotant sofisticat).
        equity_curve.append(strategy.account_equity)

    equity_series = pd.Series(equity_curve, index=df.index)

    result = BacktestResult(
        equity_curve=equity_series,
        trades=strategy.open_trades,
        starting_equity=starting_equity,
        ending_equity=strategy.account_equity,
        pnl_percent=(strategy.account_equity / starting_equity - 1) * 100,
        num_trades=len([t for t in strategy.open_trades if t.is_closed]),
    )

    return result


def main():
    result = run_backtest()

    print("===== Backtest BTC Swing Strategy =====")
    print(f"Capital inițial : {result.starting_equity:,.2f}")
    print(f"Capital final   : {result.ending_equity:,.2f}")
    print(f"PNL %           : {result.pnl_percent:.2f}%")
    print(f"Număr tranzacții: {result.num_trades}")

    # Poți salva și equity curve în CSV
    out_path = "equity_curve.csv"
    result.equity_curve.to_csv(out_path, header=["equity"])
    print(f"Equity curve salvată în {out_path}")


if __name__ == "__main__":
    main()
