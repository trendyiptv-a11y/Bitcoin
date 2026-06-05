import json
import math
import sys
from pathlib import Path
from typing import Any, List

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FRONT = ROOT / "btc-swing-strategy"
SRC = DATA / "ic_btc_series.json"
OUT = FRONT / "risk_window.json"

REGIMES = {"bear_struct", "bear_late", "accum_bear"}
THRESHOLD = -0.20
WINDOW = 180
COOLDOWN = 30

sys.path.append(str(FRONT))
from btc_swing_strategy import generate_signals


def sf(x: Any):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def load_df():
    raw = json.loads(SRC.read_text(encoding="utf-8"))
    df = pd.DataFrame(raw["series"])
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("date").sort_index()
    for c in ["close", "ic_struct", "ic_dir", "ic_flux", "ic_cycle"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close", "ic_struct", "ic_dir", "ic_flux"])


def is_deg(r):
    return str(r or "").lower() in REGIMES


def streak(df):
    n = 0
    for _, row in df.iloc[::-1].iterrows():
        if is_deg(row.get("regime")):
            n += 1
        else:
            break
    return n


def events(df):
    x = df.copy()
    x["deg"] = x["regime"].apply(is_deg)
    x["prev"] = x["deg"].shift(1).fillna(False)
    cand = x[x["deg"] & (~x["prev"])]
    keep = []
    last = None
    for ts, row in cand.iterrows():
        if last is not None and (ts - last).days < COOLDOWN:
            continue
        keep.append((ts, row))
        last = ts
    if not keep:
        return x.iloc[0:0]
    return pd.DataFrame([r for _, r in keep], index=[ts for ts, _ in keep])


def stats(df):
    ev = events(df)
    close = df["close"].astype(float)
    confirmed = 0
    available = 0
    mins: List[float] = []
    days: List[int] = []
    for ts, row in ev.iterrows():
        p = sf(row.get("close"))
        if not p or p <= 0:
            continue
        fut = close[(close.index > ts) & (close.index <= ts + pd.Timedelta(days=WINDOW))]
        if fut.empty:
            continue
        available += 1
        rel = fut / p - 1.0
        mins.append(float(rel.min()))
        hit = rel[rel <= THRESHOLD]
        if not hit.empty:
            confirmed += 1
            days.append(int((hit.index[0] - ts).days))
    return {
        "available_events": available,
        "confirmed_events": confirmed,
        "confirmation_rate": confirmed / available if available else None,
        "median_days": float(pd.Series(days).median()) if days else None,
        "average_days": float(pd.Series(days).mean()) if days else None,
        "median_min_move": float(pd.Series(mins).median()) if mins else None,
        "average_min_move": float(pd.Series(mins).mean()) if mins else None,
    }


def main():
    df = generate_signals(load_df())
    last = df.iloc[-1]
    s = stats(df)
    active = is_deg(last.get("regime"))
    card = {
        "title": "FEREASTRA_DE_RISC",
        "active": active,
        "current_regime": str(last.get("regime")),
        "current_signal": str(last.get("signal")),
        "streak_days": streak(df),
        "window_days": WINDOW,
        "threshold": THRESHOLD,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        **s,
    }
    OUT.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(card, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
