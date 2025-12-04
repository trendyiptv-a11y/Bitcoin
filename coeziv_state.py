import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# ============================
#  SETĂRI DE PATH
# ============================

BASE_DIR = os.path.dirname(__file__)
STRATEGY_DIR = os.path.join(BASE_DIR, "btc-swing-strategy")

# adăugăm folderul btc-swing-strategy în sys.path
if STRATEGY_DIR not in sys.path:
    sys.path.append(STRATEGY_DIR)

# importă generate_signals din fișierul tău de strategie
from btc_swing_strategy import generate_signals


# ============================
#  ÎNCĂRCARE IC SERIES
# ============================

def load_ic_series(path: str | None = None) -> pd.DataFrame:
    """
    Încarcă seria IC BTC din ic_btc_series.json.

    Caută fișierul în mai multe locații posibile:
    - ./ic_btc_series.json
    - ./j-btc-coeziv/ic_btc_series.json
    - ./data/ic_btc_series.json
    """

    candidates: list[str] = []

    if path is not None:
        candidates.append(path)
    else:
        candidates.extend(
            [
                os.path.join(BASE_DIR, "ic_btc_series.json"),
                os.path.join(BASE_DIR, "j-btc-coeziv", "ic_btc_series.json"),
                os.path.join(BASE_DIR, "data", "ic_btc_series.json"),
            ]
        )

    chosen: str | None = None
    for c in candidates:
        if os.path.exists(c):
            chosen = c
            break

    if chosen is None:
        raise FileNotFoundError(
            "Nu am găsit ic_btc_series.json în niciuna din locațiile așteptate:\n"
            + "\n".join(candidates)
        )

    with open(chosen, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw["series"])
    # presupunem că 't' este în milisecunde epoch
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("date").sort_index()

    # returnăm strict coloanele necesare pentru generate_signals
    return df[
        [
            "close",
            "ic_struct",
            "ic_dir",
            "ic_flux",
            "ic_cycle",
            "regime",
        ]
    ]


# ============================
#  ÎNCĂRCARE OHLC REAL (PIAȚĂ)
# ============================

def load_ohlc(path: str | None = None) -> pd.DataFrame | None:
    """
    Încarcă seriile OHLC reale pentru BTC dintr-un fișier local (CSV).

    Caută fișierul în mai multe locații posibile:
    - ./data/btc_ohlc.csv
    - ./data/btc_daily.csv
    - ./btc_ohlc.csv
    - ./btc_daily.csv

    Presupunem că fișierul are cel puțin:
    - o coloană de timp: una dintre ['timestamp','time','date','Date','open_time','t']
    - o coloană de preț de închidere: una dintre ['close','Close','c','C']
    """

    candidates: list[str] = []

    if path is not None:
        candidates.append(path)
    else:
        candidates.extend(
            [
                os.path.join(BASE_DIR, "data", "btc_ohlc.csv"),
                os.path.join(BASE_DIR, "data", "btc_daily.csv"),
                os.path.join(BASE_DIR, "btc_ohlc.csv"),
                os.path.join(BASE_DIR, "btc_daily.csv"),
            ]
        )

    chosen: str | None = None
    for c in candidates:
        if os.path.exists(c):
            chosen = c
            break

    if chosen is None:
        # nu forțăm eroare: întoarcem None => vom cădea pe varianta doar cu model
        return None

    df = pd.read_csv(chosen)

    # detectăm coloana de timp
    time_col = None
    for cand in ["timestamp", "time", "date", "Date", "open_time", "t"]:
        if cand in df.columns:
            time_col = cand
            break

    if time_col is None:
        raise ValueError(
            f"Nu am găsit nicio coloană de timp în fișierul OHLC {chosen}. "
            "Aștept una dintre: timestamp, time, date, Date, open_time, t."
        )

    # detectăm coloana de close
    close_col = None
    for cand in ["close", "Close", "c", "C"]:
        if cand in df.columns:
            close_col = cand
            break

    if close_col is None:
        raise ValueError(
            f"Nu am găsit nicio coloană de close în fișierul OHLC {chosen}. "
            "Aștept una dintre: close, Close, c, C."
        )

    # parsăm timpul
    if time_col == "t":
        # presupunem milisecunde epoch
        df["dt"] = pd.to_datetime(df[time_col], unit="ms", utc=True)
    else:
        df["dt"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    df = df.dropna(subset=["dt"])
    df = df.set_index("dt").sort_index()

    out = pd.DataFrame(index=df.index.copy())
    out["close"] = pd.to_numeric(df[close_col], errors="coerce")
    out = out.dropna(subset=["close"])

    return out


# ============================
#  PREȚ LIVE BTC (API GRATUIT)
# ============================

def get_live_btc_price() -> float:
    """
    Ia prețul BTC/USD din CoinGecko (API gratuit, fără cheie).
    Dacă apare o eroare, excepția va fi prinsă în main() și
    vom cădea pe prețul din model.
    """
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin", "vs_currencies": "usd"}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return float(data["bitcoin"]["usd"])


# ============================
#  MESAJUL COEZIV
# ============================

def build_message(signal: str, price: float) -> str:
    """
    Textul coeziv pentru dashboard, bazat pe semnal + prețul folosit în mesaj.
    """

    s = (signal or "").lower()

    if s == "long":
        return (
            f"La prețul actual de ~{price:,.0f} USD, mecanismul coeziv vede "
            f"context favorabil pentru acumulare. Poți cumpăra, dar decizia finală îți aparține."
        )

    if s == "short":
        return (
            f"În jurul valorii de ~{price:,.0f} USD, mecanismul coeziv detectează "
            f"riscuri crescute de scădere. Poți vinde sau reduce expunerea."
        )

    return (
        f"Bitcoin se tranzacționează în jur de ~{price:,.0f} USD. "
        f"Mecanismul coeziv este neutru: poți cumpăra și poți vinde, "
        f"dar poate fi util să aștepți claritate suplimentară."
    )


# ============================
#  ISTORIC DE SEMNALE PENTRU UI
# ============================

def build_signal_history(df: pd.DataFrame, limit: int = 30) -> list[dict]:
    """
    Construiește un mic istoric de semnale pentru UI:
    ultimele `limit` intrări din serie, fiecare cu:
      - timestamp
      - signal
      - model_price_usd (close)
    """
    history: list[dict] = []
    tail = df.tail(limit)

    for ts, row in tail.iterrows():
        sig = str(row.get("signal", "")).lower()
        if sig not in ("long", "short", "flat"):
            # dacă generate_signals nu a populat încă semnalul pentru rândul ăsta,
            # îl ignorăm în istoric
            continue

        close_val = row.get("close", None)
        try:
            close_val = float(close_val) if close_val is not None else None
        except (TypeError, ValueError):
            close_val = None

        history.append(
            {
                "timestamp": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
                "signal": sig,
                "model_price_usd": close_val,
            }
        )

    return history


# ============================
#  OUTCOME & PROBABILITATE
# ============================

def compute_signal_outcomes(
    df_signals: pd.DataFrame,
    df_ohlc: pd.DataFrame | None = None,
    horizon_hours: int = 24,
    threshold_pct: float = 0.005,
) -> pd.DataFrame:
    """
    Pentru fiecare snapshot cu semnal Long/Short, calculăm dacă semnalul
    a fost 'reușit' în următoarele horizon_hours.

    Outcome:
      1 = prețul a mers în direcția semnalului cu cel puțin threshold_pct
      0 = altfel

    Dacă df_ohlc este furnizat, folosim datele reale OHLC (piață).
    Dacă nu, folosim doar seria IC (close) ca proxy.
    """

    df = df_signals.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame-ul de semnale trebuie să aibă index datetime.")

    df = df.sort_index()

    # Alegem sursa de preț pentru viitor
    if df_ohlc is not None and not df_ohlc.empty:
        price_index = df_ohlc.index.sort_values()
        price_series = df_ohlc["close"].sort_index()
        use_ohlc = True
    else:
        price_index = df.index.sort_values()
        price_series = df["close"].sort_index()
        use_ohlc = False

    horizon = timedelta(hours=horizon_hours)
    future_prices: list[float | None] = []

    for ts in df.index:
        target = ts + horizon
        # găsim primul punct de preț cu timestamp >= target
        pos = price_index.searchsorted(target, side="left")
        if pos >= len(price_index):
            # nu avem date suficient de în viitor; marcăm None
            future_prices.append(None)
        else:
            future_ts = price_index[pos]
            future_prices.append(float(price_series.loc[future_ts]))

    df["future_close"] = future_prices
    df["outcome"] = None

    for idx, row in df.iterrows():
        sig = str(row.get("signal", "")).lower()
        c0 = row.get("close", None)
        cf = row.get("future_close", None)

        if sig not in ("long", "short") or c0 is None or cf is None:
            df.at[idx, "outcome"] = None
            continue

        try:
            c0 = float(c0)
            cf = float(cf)
        except (TypeError, ValueError):
            df.at[idx, "outcome"] = None
            continue

        if c0 <= 0:
            df.at[idx, "outcome"] = None
            continue

        ret = (cf - c0) / c0  # randament relativ

        if sig == "long":
            df.at[idx, "outcome"] = 1 if ret >= threshold_pct else 0
        else:  # short
            df.at[idx, "outcome"] = 1 if ret <= -threshold_pct else 0

    df.attrs["outcome_source"] = "ohlc" if use_ohlc else "model"

    return df


def estimate_signal_probability(
    df_signals: pd.DataFrame,
    df_ohlc: pd.DataFrame | None = None,
    horizon_hours: int = 24,
    threshold_pct: float = 0.005,
) -> dict:
    """
    Estimăm probabilitatea ca semnalul curent să fie 'reușit'
    pe baza istoricului de semnale similare.

    Returnăm:
      {
        "probability": float în [0,1] sau None,
        "samples": int,
        "horizon_hours": int,
        "source": "ohlc" | "model"
      }
    """

    df = compute_signal_outcomes(
        df_signals, df_ohlc=df_ohlc, horizon_hours=horizon_hours, threshold_pct=threshold_pct
    )

    # rândul curent (ultimul)
    last = df.iloc[-1]
    sig = str(last.get("signal", "")).lower()
    if sig not in ("long", "short"):
        return {
            "probability": None,
            "samples": 0,
            "horizon_hours": horizon_hours,
            "source": df.attrs.get("outcome_source", "unknown"),
        }

    last_regime = last.get("regime", None)
    last_dir = last.get("ic_dir", None)

    # filtrăm istoricul (excludem ultimul rând)
    hist = df.iloc[:-1].copy()
    hist = hist[hist["signal"].astype(str).str.lower().isin(["long", "short"])]
    hist = hist[hist["outcome"].isin([0, 1])]

    # condiții de similaritate
    mask = hist["signal"].astype(str).str.lower() == sig

    # regim similar (același semn, dacă este numeric)
    if pd.notna(last_regime):
        try:
            reg_sign = 1 if float(last_regime) > 0 else (-1 if float(last_regime) < 0 else 0)
            hist_reg_sign = hist["regime"].astype(float).apply(
                lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
            )
            mask &= hist_reg_sign.eq(reg_sign)
        except Exception:
            # dacă nu putem interpreta regimul numeric, ignorăm criteriul
            pass

    # direcție IC similară (semn)
    if pd.notna(last_dir):
        try:
            dir_sign = 1 if float(last_dir) > 0 else (-1 if float(last_dir) < 0 else 0)
            hist_dir_sign = hist["ic_dir"].astype(float).apply(
                lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
            )
            mask &= hist_dir_sign.eq(dir_sign)
        except Exception:
            # ignorăm criteriul dacă nu e numeric
            pass

    subset = hist[mask]

    n_total = int(subset.shape[0])
    if n_total == 0:
        return {
            "probability": None,
            "samples": 0,
            "horizon_hours": horizon_hours,
            "source": df.attrs.get("outcome_source", "unknown"),
        }

    n_success = int(subset["outcome"].sum())

    # Laplace smoothing (evităm 0% / 100% pe eșantioane mici)
    prob = (n_success + 1.0) / (n_total + 2.0)

    return {
        "probability": float(prob),
        "samples": n_total,
        "horizon_hours": horizon_hours,
        "source": df.attrs.get("outcome_source", "unknown"),
    }


# ============================
#  MAIN – GENERAREA STĂRII COEZIVE
# ============================

def main() -> None:
    # 1. încărcăm datele IC (structură, direcție, flux, regim)
    df_ic = load_ic_series()

    # 2. generăm semnalele coezive
    df_ic = generate_signals(df_ic)

    # 3. încărcăm (opțional) seriile OHLC reale pentru probabilitate
    try:
        df_ohlc = load_ohlc()
    except Exception as e:
        print("Nu am putut încărca fișierul OHLC. Vom folosi doar seria IC pentru probabilitate.", e)
        df_ohlc = None

    # 4. extragem ultimul punct din serie
    last = df_ic.iloc[-1]
    model_price = float(last["close"])   # prețul din snapshot-ul modelului
    signal = str(last["signal"])
    ts = last.name  # index datetime

    # 5. preț spot live (cu fallback la model_price)
    price_source = "model"
    price_for_text = model_price

    try:
        live_price = get_live_btc_price()
        price_for_text = live_price
        price_source = "spot"
    except Exception as e:
        # dacă API-ul nu merge, folosim prețul din model și lăsăm price_source="model"
        print("Nu am putut obține prețul live BTC. Folosesc prețul din model.", e)

    # 6. generăm mesajul pe baza semnalului + prețul folosit în text
    message = build_message(signal, price_for_text)

    # 7. istoric de semnale (ultimele N puncte)
    signal_history = build_signal_history(df_ic, limit=30)

    # 8. probabilitate istorică pentru semnalul curent
    prob_info = estimate_signal_probability(
        df_ic,
        df_ohlc=df_ohlc,
        horizon_hours=24,
        threshold_pct=0.005,
    )
    signal_probability = prob_info.get("probability")
    signal_prob_samples = prob_info.get("samples", 0)
    signal_prob_horizon = prob_info.get("horizon_hours", 24)
    signal_prob_source = prob_info.get("source", "unknown")

    state = {
        "timestamp": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
        "price_usd": price_for_text,        # ce vezi mare în UI
        "model_price_usd": model_price,     # prețul folosit în snapshot-ul IC
        "price_source": price_source,       # "spot" sau "model"
        "signal": signal,
        "message": message,
        "signal_history": signal_history,   # folosit de cardul de istoric
        "signal_probability": signal_probability,                # 0–1 sau null
        "signal_prob_samples": signal_prob_samples,              # câte cazuri similare
        "signal_prob_horizon_hours": signal_prob_horizon,        # ex. 24
        "signal_prob_source": signal_prob_source,                # "ohlc" sau "model"
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # 9. scriem JSON în folderul frontend-ului
    output_path = os.path.join(STRATEGY_DIR, "coeziv_state.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print("Stare coezivă generată:", output_path)
    print("Semnal:", signal, "| Sursă preț:", price_source, "| Preț mesaj:", price_for_text)
    print("Istoric semnale livrat:", len(signal_history), "puncte")
    print(
        "Probabilitate semnal:",
        signal_probability,
        "| Eșantioane similare:",
        signal_prob_samples,
        "| Orizont ore:",
        signal_prob_horizon,
        "| Sursă outcome:",
        signal_prob_source,
    )


# ============================
#  ENTRY POINT
# ============================

if __name__ == "__main__":
    main()
