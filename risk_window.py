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
BOTTOM_SUMMARY = FRONT / "adaptive_bottom_zone_summary.json"

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


def fmt_pct(rate: Optional[float]) -> str:
    if rate is None or not math.isfinite(float(rate)):
        return "n/a"
    return f"{float(rate) * 100:.0f}%"


def fmt_days(days: Optional[float]) -> str:
    if days is None or not math.isfinite(float(days)):
        return "n/a"
    return f"~{float(days):.0f} zile"


def fmt_k(value: Any) -> str:
    n = sf(value)
    if n is None:
        return "n/a"
    txt = f"{n / 1000:.1f}K"
    return txt.replace(".0K", "K")


def iso_day(value: Any) -> Optional[str]:
    if value is None:
        return None
    m = str(value)[:10]
    if len(m) == 10 and m[4] == "-" and m[7] == "-":
        return m
    return None


def days_between(start: Any, end: Any) -> Optional[int]:
    s = iso_day(start)
    e = iso_day(end)
    if not s or not e:
        return None
    try:
        a = pd.Timestamp(s, tz="UTC")
        b = pd.Timestamp(e, tz="UTC")
    except Exception:
        return None
    if b < a:
        return None
    return int((b - a).days)


def load_bottom_summary() -> Optional[Dict[str, Any]]:
    if not BOTTOM_SUMMARY.exists():
        return None
    try:
        return json.loads(BOTTOM_SUMMARY.read_text(encoding="utf-8"))
    except Exception:
        return None


def bottom_bounds(summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    c = (summary or {}).get("radar_candidate") or {}
    bounds = {
        "ath": sf(c.get("ath_price")),
        "bear": sf(c.get("bear_warning_threshold")),
        "bottom_low": sf(c.get("bottom_risk_zone_low")),
        "bottom_mid": sf(c.get("bottom_risk_zone_mid")),
        "bottom_high": sf(c.get("bottom_risk_zone_high")),
        "hard": sf(c.get("hard_capitulation_below")),
    }
    required = ["bear", "bottom_low", "bottom_mid", "bottom_high"]
    if any(bounds[k] is None for k in required):
        return None
    return bounds  # type: ignore[return-value]


def direction_from_signal(signal: Any) -> str:
    s = str(signal or "").lower()
    if s in {"long", "buy", "bull", "up"}:
        return "up"
    if s in {"short", "sell", "bear", "down"}:
        return "down"
    return "flat"


def structural_timing(summary: Dict[str, Any], as_of: str) -> Dict[str, Any]:
    t = summary.get("structural_timing") or {}
    touch = iso_day(t.get("fragility_touch_date"))
    adaptive_touch = iso_day(t.get("adaptive_bear_warning_touch_date"))
    confirm = iso_day(t.get("close_confirmation_date"))
    touch_days = days_between(touch, as_of)
    confirm_days = days_between(confirm, as_of)
    lag = sf(t.get("days_from_fragility_touch_to_confirmation"))
    if lag is None:
        lag_days = days_between(touch, confirm)
    else:
        lag_days = int(lag)
    confirmed = bool(confirm and confirm_days is not None and confirm_days >= 0)
    return {
        "as_of": as_of,
        "fragility_touch_date": touch,
        "adaptive_bear_warning_touch_date": adaptive_touch,
        "close_confirmation_date": confirm,
        "days_from_fragility_touch": touch_days,
        "days_from_close_confirmation": confirm_days,
        "days_to_close_confirmation": lag_days,
        "confirmed": confirmed,
        "fragility_reference_price": sf(t.get("fragility_reference_price")),
        "adaptive_bear_warning_threshold": sf(t.get("adaptive_bear_warning_threshold")),
    }


def bottom_observation(df, bounds: Dict[str, float], timing: Dict[str, Any], last_price: float) -> Dict[str, Any]:
    touch = timing.get("fragility_touch_date")
    if touch:
        x = df[df.index >= pd.Timestamp(touch, tz="UTC")]
    else:
        x = df
    close = x["close"].astype(float) if not x.empty else df["close"].astype(float)
    min_close = float(close.min()) if not close.empty else None
    touched_zone = bool(min_close is not None and min_close <= bounds["bottom_high"])
    touched_hard = bool(min_close is not None and min_close < bounds["bottom_low"])
    confirmed_exit = bool(touched_zone and last_price > bounds["bottom_high"])

    if last_price < bounds["bottom_low"]:
        status = "capitulare sub zonă"
    elif last_price <= bounds["bottom_high"]:
        status = "în test"
    elif confirmed_exit:
        status = "testată anterior; preț deasupra zonei"
    else:
        status = "neatinsă"

    return {
        "status": status,
        "touched_model_zone": touched_zone,
        "touched_hard_capitulation": touched_hard,
        "confirmed_exit_above_zone": confirmed_exit,
        "min_close_after_fragility_touch": min_close,
    }


def structural_zone(last_price: float, signal: Any, bounds: Dict[str, float], observation: Dict[str, Any]) -> Dict[str, Any]:
    direction = direction_from_signal(signal)
    touched = bool(observation.get("touched_model_zone"))
    ath = bounds.get("ath")

    def pack(key: str, label: str, active: bool, level: str):
        return {"key": key, "label": label, "active": active, "level": level, "direction": direction}

    if ath is not None and last_price >= ath:
        if direction == "down":
            return pack("expansion", "Expansiune sub presiune", False, "low")
        if direction == "up":
            return pack("expansion", "Expansiune activă", False, "normal")
        return pack("expansion", "Expansiune structurală", False, "normal")

    if last_price >= bounds["bear"]:
        if direction == "down":
            return pack("repaired", "Retest de ruptură", True, "low")
        if direction == "up":
            return pack("repaired", "Creștere structurală", False, "normal")
        return pack("repaired", "Structură refăcută", False, "normal")

    if last_price > bounds["bottom_high"]:
        suffix = "zona istoric modelată neatinsă" if not touched else "ieșire peste zona modelată"
        if direction == "down":
            return pack("deep", f"Risc de adâncire · {suffix}", True, "high")
        if direction == "up":
            return pack("deep", f"Revenire parțială · {suffix}", True, "moderate")
        return pack("deep", f"Degradare profundă · {suffix}", True, "high")

    if last_price >= bounds["bottom_low"]:
        if direction == "down":
            return pack("bottom", "Risc în zona istoric modelată", True, "high")
        if direction == "up":
            return pack("bottom", "Ieșire din zona istoric modelată", True, "moderate")
        return pack("bottom", "Zona istoric modelată în test", True, "high")

    if direction == "up":
        return pack("capitulation", "Revenire din capitulare", True, "high")
    return pack("capitulation", "Capitulare sub zona modelată", True, "high")


def build_main_text(active: bool, s: Dict[str, Any], streak_days: int) -> str:
    rate = s.get("confirmation_rate")
    available = s.get("available_events") or 0
    confirmed = s.get("confirmed_events") or 0
    median_days = s.get("median_days")
    rate_txt = fmt_pct(rate)
    days_txt = fmt_days(median_days)

    if not active:
        return "Fereastra de risc structural nu este activă în snapshotul curent. Contextul actual nu se află în familia de degradare urmărită de acest card."

    if available <= 0:
        return f"Mecanismul indică {streak_days} zile consecutive de degradare structurală, dar nu există încă suficiente evenimente istorice comparabile pentru o rată robustă de confirmare."

    return (
        f"Contextul actual aparține unei familii istorice care a generat scăderi importante în {rate_txt} din cazuri "
        f"({confirmed} confirmări din {available} episoade comparabile). "
        f"Când acestea s-au confirmat, confirmarea a apărut în medie după {days_txt}."
    )


def build_structural_main_text(zone: Dict[str, Any], timing: Dict[str, Any], bounds: Dict[str, float], observation: Dict[str, Any]) -> str:
    touch = timing.get("fragility_touch_date") or "n/a"
    confirm = timing.get("close_confirmation_date") or "n/a"
    touch_days = timing.get("days_from_fragility_touch")
    lag = timing.get("days_to_close_confirmation")
    ref = fmt_k(timing.get("fragility_reference_price") or bounds.get("bear"))
    low = fmt_k(bounds.get("bottom_low"))
    high = fmt_k(bounds.get("bottom_high"))
    label = str(zone.get("label") or "Context structural")
    bottom_status = observation.get("status") or "n/a"

    if timing.get("confirmed"):
        first = f"Ruptura structurală a început la ~{ref} pe {touch} și s-a confirmat pe închidere în {lag} zile, pe {confirm}."
    else:
        first = f"Prețul a intrat în test structural la ~{ref} pe {touch}; confirmarea pe închidere nu este încă validată."

    days_txt = f"~{touch_days} zile" if touch_days is not None else "n/a"
    return (
        f"{first} Structura fragilă este urmărită de {days_txt}. "
        f"Regimul curent este: {label}. "
        f"Zona istoric modelată de mecanism, {low}–{high}, este {bottom_status}."
    )


def build_legacy_card(last, active: bool, level: str, streak_days: int, s: Dict[str, Any], summary_since: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": "FEREASTRA_DE_RISC",
        "active": bool(active),
        "level": level,
        "current_regime": str(last.get("regime")),
        "current_signal": str(last.get("signal")),
        "consecutive_degradation_days": int(streak_days),
        "streak_days": int(streak_days),
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
        "main_text": build_main_text(active, s, streak_days),
        "footer": "Interpretare statistică a degradării structurale, nu recomandare de tranzacționare.",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }


def build_structural_card(df, last, legacy: Dict[str, Any], summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    bounds = bottom_bounds(summary)
    last_price = sf(last.get("close"))
    if not bounds or last_price is None:
        return None

    as_of = df.index[-1].date().isoformat()
    timing = structural_timing(summary, as_of)
    observation = bottom_observation(df, bounds, timing, last_price)
    zone = structural_zone(last_price, last.get("signal"), bounds, observation)

    confirm_rate = 1.0 if timing.get("confirmed") else 0.0
    confirm_days = timing.get("days_to_close_confirmation")
    structural_days = timing.get("days_from_fragility_touch")

    card = dict(legacy)
    card.update({
        "title": "FEREASTRA_STRUCTURALA_BTC",
        "active": bool(zone.get("active")),
        "level": str(zone.get("level")),
        "current_regime": str(zone.get("label")),
        "current_signal": str(last.get("signal")),
        "consecutive_degradation_days": int(structural_days) if structural_days is not None else legacy.get("consecutive_degradation_days"),
        "streak_days": int(structural_days) if structural_days is not None else legacy.get("streak_days"),
        "historical_confirmation_rate": confirm_rate,
        "median_days_to_confirmation": float(confirm_days) if confirm_days is not None else None,
        "average_days_to_confirmation": float(confirm_days) if confirm_days is not None else None,
        "structural_source": str(BOTTOM_SUMMARY.relative_to(ROOT)),
        "structural_as_of": as_of,
        "structural_zone": zone,
        "structural_timing": timing,
        "bottom_observation": observation,
        "fragility_reference_price": timing.get("fragility_reference_price"),
        "adaptive_bear_warning_threshold": timing.get("adaptive_bear_warning_threshold") or bounds.get("bear"),
        "fragility_touch_date": timing.get("fragility_touch_date"),
        "close_confirmation_date": timing.get("close_confirmation_date"),
        "days_from_fragility_touch": structural_days,
        "days_to_close_confirmation": confirm_days,
        "bottom_risk_zone_low": bounds.get("bottom_low"),
        "bottom_risk_zone_mid": bounds.get("bottom_mid"),
        "bottom_risk_zone_high": bounds.get("bottom_high"),
        "bottom_status": observation.get("status"),
        "legacy_risk_window": {
            "active": legacy.get("active"),
            "level": legacy.get("level"),
            "current_regime": legacy.get("current_regime"),
            "current_signal": legacy.get("current_signal"),
            "consecutive_degradation_days": legacy.get("consecutive_degradation_days"),
            "historical_available_events": legacy.get("historical_available_events"),
            "historical_confirmed_events": legacy.get("historical_confirmed_events"),
            "historical_confirmation_rate": legacy.get("historical_confirmation_rate"),
            "median_days_to_confirmation": legacy.get("median_days_to_confirmation"),
            "average_days_to_confirmation": legacy.get("average_days_to_confirmation"),
            "median_min_drawdown": legacy.get("median_min_drawdown"),
            "average_min_drawdown": legacy.get("average_min_drawdown"),
        },
        "main_text": build_structural_main_text(zone, timing, bounds, observation),
        "footer": "Interpretare structurală din harta istoric modelată de mecanism; nu modifică modelul de bază, tradingul și nu reprezintă recomandare de tranzacționare.",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    })
    return card


def main():
    df = generate_signals(load_df())
    last = df.iloc[-1]
    s = stats(df)
    active = is_deg(last.get("regime"))
    streak_days = streak(df)
    level = level_for(active, s.get("confirmation_rate"), streak_days)
    history = segment_outcomes(df)
    summary_since = since_summary(history)

    legacy_card = build_legacy_card(last, active, level, streak_days, s, summary_since)
    bottom_summary = load_bottom_summary()
    card = build_structural_card(df, last, legacy_card, bottom_summary) if bottom_summary else None
    if card is None:
        card = legacy_card

    FRONT.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    HISTORY_OUT.write_text(json.dumps({
        "title": "RISK_WINDOW_HISTORY",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "regimes": sorted(REGIMES),
        "window_days": WINDOW,
        "major_drawdown_threshold": THRESHOLD,
        "since_2025_12_summary": summary_since,
        "structural_source": str(BOTTOM_SUMMARY.relative_to(ROOT)) if bottom_summary else None,
        "history": history,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(card, ensure_ascii=False, indent=2))
    print(f"[OK] Scris {OUT}")
    print(f"[OK] Scris {HISTORY_OUT}")


if __name__ == "__main__":
    main()
