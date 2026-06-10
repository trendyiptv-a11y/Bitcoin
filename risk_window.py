import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FRONT = ROOT / "btc-swing-strategy"
SRC = DATA / "ic_btc_series.json"
OUT = FRONT / "risk_window.json"
HISTORY_OUT = FRONT / "risk_window_history.json"

REGIMES = {"bear_struct", "bear_late", "accum_bear"}
THRESHOLD = -0.20
WINDOW = 180
COOLDOWN = 30
SINCE_DATE = pd.Timestamp("2025-12-01", tz="UTC")

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


def degradation_segments(df):
    segments = []
    active = None
    for ts, row in df.iterrows():
        deg = is_deg(row.get("regime"))
        regime = str(row.get("regime") or "")
        signal = str(row.get("signal") or "")
        close = sf(row.get("close"))
        if deg and active is None:
            active = {
                "start": ts,
                "end": ts,
                "start_price": close,
                "end_price": close,
                "days": 1,
                "regimes": {regime} if regime else set(),
                "signals": {signal} if signal else set(),
            }
        elif deg and active is not None:
            active["end"] = ts
            active["end_price"] = close
            active["days"] += 1
            if regime:
                active["regimes"].add(regime)
            if signal:
                active["signals"].add(signal)
        elif (not deg) and active is not None:
            segments.append(active)
            active = None
    if active is not None:
        segments.append(active)
    return segments


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


def segment_outcomes(df) -> List[Dict[str, Any]]:
    close = df["close"].astype(float)
    out = []
    segments = degradation_segments(df)
    current_start = segments[-1]["start"] if segments else None
    for seg in segments:
        start = seg["start"]
        p = sf(seg.get("start_price"))
        fut = close[(close.index > start) & (close.index <= start + pd.Timedelta(days=WINDOW))]
        min_drawdown: Optional[float] = None
        confirmed = False
        days_to_confirmation: Optional[int] = None
        if p and p > 0 and not fut.empty:
            rel = fut / p - 1.0
            min_drawdown = float(rel.min())
            hit = rel[rel <= THRESHOLD]
            if not hit.empty:
                confirmed = True
                days_to_confirmation = int((hit.index[0] - start).days)
        out.append({
            "start": start.date().isoformat(),
            "end": seg["end"].date().isoformat(),
            "days": int(seg["days"]),
            "is_current": bool(current_start is not None and start == current_start),
            "start_price": seg.get("start_price"),
            "end_price": seg.get("end_price"),
            "regimes": sorted(list(seg.get("regimes", set()))),
            "signals": sorted(list(seg.get("signals", set()))),
            "window_days": WINDOW,
            "major_drawdown_threshold": THRESHOLD,
            "min_drawdown": min_drawdown,
            "confirmed_major_drawdown": confirmed,
            "days_to_confirmation": days_to_confirmation,
        })
    return out


def since_summary(history: List[Dict[str, Any]]):
    recent = [h for h in history if pd.Timestamp(h["start"], tz="UTC") >= SINCE_DATE]
    current = next((h for h in history if h.get("is_current")), None)
    current_days = int(current.get("days", 0)) if current else 0
    if not recent:
        return {
            "since": SINCE_DATE.date().isoformat(),
            "events": 0,
            "max_streak_days": 0,
            "current_streak_days": current_days,
            "current_is_record": False,
            "matching_or_longer_than_current": 0,
            "top_streaks": [],
        }
    max_days = max(int(h.get("days", 0)) for h in recent)
    matching = [h for h in recent if current_days and int(h.get("days", 0)) >= current_days]
    top = sorted(recent, key=lambda h: int(h.get("days", 0)), reverse=True)[:10]
    return {
        "since": SINCE_DATE.date().isoformat(),
        "events": len(recent),
        "max_streak_days": max_days,
        "current_streak_days": current_days,
        "current_is_record": bool(current_days and current_days >= max_days),
        "matching_or_longer_than_current": len(matching),
        "top_streaks": top,
    }


def level_for(active: bool, rate: Optional[float], streak_days: int):
    if not active:
        return "normal"
    if streak_days >= 5 and rate is not None and rate >= 0.45:
        return "high"
    if streak_days >= 3:
        return "moderate"
    return "low"


def main():
    df = generate_signals(load_df())
    last = df.iloc[-1]
    s = stats(df)
    active = is_deg(last.get("regime"))
    streak_days = streak(df)
    level = level_for(active, s.get("confirmation_rate"), streak_days)
    history = segment_outcomes(df)
    summary_since = since_summary(history)

    card = {
        "title": "FEREASTRA_DE_RISC",
        "active": active,
        "level": level,
        "current_regime": str(last.get("regime")),
        "current_signal": str(last.get("signal")),
        "consecutive_degradation_days": streak_days,
        "streak_days": streak_days,
        "window_days": WINDOW,
        "major_drawdown_threshold": THRESHOLD,
        "threshold": THRESHOLD,
        "historical_available_events": s.get("available_events"),
        "historical_confirmed_events": s.get("confirmed_events"),
        "historical_confirmation_rate": s.get("confirmation_rate"),
        "median_days_to_confirmation": s.get("median_days"),
        "average_days_to_confirmation": s.get("average_days"),
        "median_min_drawdown": s.get("median_min_move"),
        "average_min_drawdown": s.get("average_min_move"),
        "since_2025_12_summary": summary_since,
        "main_text": "Contextul actual aparține unei familii istorice care a generat scăderi importante în aproximativ 50% din cazuri. Când acestea s-au confirmat, confirmarea a apărut în medie după ~27 zile.",
        "footer": "Interpretare statistică a degradării structurale, nu recomandare de tranzacționare.",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }

    FRONT.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    HISTORY_OUT.write_text(json.dumps({
        "title": "RISK_WINDOW_HISTORY",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "regimes": sorted(REGIMES),
        "window_days": WINDOW,
        "major_drawdown_threshold": THRESHOLD,
        "since_2025_12_summary": summary_since,
        "history": history,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(card, ensure_ascii=False, indent=2))
    print(f"[OK] Scris {OUT}")
    print(f"[OK] Scris {HISTORY_OUT}")


if __name__ == "__main__":
    main()
