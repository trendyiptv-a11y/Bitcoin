#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
update_btc_72h_statistics.py

Adaugă statistica istorică pe 3 zile / 72h pentru mecanismul BTC.

Principiu:
- folosește seria istorică IC BTC generată din data/btc_daily.csv;
- caută contexte istorice similare cu contextul curent;
- măsoară evoluția close[T] -> close[T+3 zile];
- scrie distribuția obiectivă în JSON-urile folosite de front-end.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SERIES_FILE = DATA_DIR / "ic_btc_series.json"
DATA_STATE_FILE = DATA_DIR / "btc_state_latest.json"
WEB_STATE_FILE = ROOT / "btc-swing-strategy" / "coeziv_state.json"

HORIZON_DAYS = 3
HORIZON_HOURS = 72
FLAT_THRESHOLD = 0.0075  # +/-0.75% pe 3 zile = zgomot/neutru
MAX_SIMILAR_SAMPLES = 750
MIN_SIMILAR_SAMPLES = 120


def safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        x = float(v)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def context_direction(ic_dir: float) -> str:
    if ic_dir >= 55:
        return "up"
    if ic_dir <= 45:
        return "down"
    return "neutral"


def classify_return(ret: float, direction: str) -> str:
    if abs(ret) <= FLAT_THRESHOLD:
        return "flat"
    if direction == "up":
        return "in_direction" if ret > 0 else "opposite"
    if direction == "down":
        return "in_direction" if ret < 0 else "opposite"
    return "in_direction" if ret > 0 else "opposite"


def distance_to_latest(rec: Dict[str, Any], latest: Dict[str, Any]) -> float:
    ic_dir = safe_float(rec.get("ic_dir"))
    ic_struct = safe_float(rec.get("ic_struct"))
    vol = safe_float(rec.get("vol30_index"))
    latest_dir = safe_float(latest.get("ic_dir"))
    latest_struct = safe_float(latest.get("ic_struct"))
    latest_vol = safe_float(latest.get("vol30_index"))
    if ic_dir is None or ic_struct is None or latest_dir is None or latest_struct is None:
        return float("inf")
    d = abs(ic_dir - latest_dir) * 1.0 + abs(ic_struct - latest_struct) * 0.6
    if vol is not None and latest_vol is not None:
        d += abs(vol - latest_vol) * 0.4
    return d


def compute_distribution(series: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(series) <= HORIZON_DAYS + 50:
        raise RuntimeError("Seria IC BTC este prea scurtă pentru statistica 72h.")

    latest = series[-1]
    latest_ic_dir = safe_float(latest.get("ic_dir"))
    if latest_ic_dir is None:
        raise RuntimeError("Ultimul punct din serie nu are ic_dir valid.")

    direction = context_direction(latest_ic_dir)
    candidates: List[Dict[str, Any]] = []

    for i in range(0, len(series) - HORIZON_DAYS):
        rec = series[i]
        fut = series[i + HORIZON_DAYS]
        close0 = safe_float(rec.get("close"))
        close1 = safe_float(fut.get("close"))
        if close0 is None or close1 is None or close0 <= 0:
            continue
        dist = distance_to_latest(rec, latest)
        if not math.isfinite(dist):
            continue
        ret = close1 / close0 - 1.0
        candidates.append({"distance": dist, "return": ret, "t": rec.get("t"), "future_t": fut.get("t")})

    if not candidates:
        raise RuntimeError("Nu există candidați istorici pentru statistica 72h.")

    candidates.sort(key=lambda x: x["distance"])
    sample_size = min(MAX_SIMILAR_SAMPLES, len(candidates))
    if sample_size < MIN_SIMILAR_SAMPLES:
        sample_size = len(candidates)
    sample = candidates[:sample_size]
    returns = [float(x["return"]) for x in sample]

    counts = {"in_direction": 0, "opposite": 0, "flat": 0}
    for ret in returns:
        counts[classify_return(ret, direction)] += 1

    n = len(returns)
    breakdown = {k: (counts[k] / n if n else 0.0) for k in counts}

    return {
        # Câmpul pe care UI-ul îl afișează la „Probabilitate istorică”.
        # Trebuie sincronizat cu distribuția 72h, altfel rămâne vechiul procent 24h.
        "signal_probability": breakdown["in_direction"],
        "signal_prob_horizon_hours": HORIZON_HOURS,
        "signal_prob_samples": n,
        "signal_prob_source": "historical_72h_ic_similarity",
        "signal_prob_breakdown": breakdown,
        "signal_expected_drift": {
            "horizon_hours": HORIZON_HOURS,
            "p10": quantile(returns, 0.10),
            "p50": quantile(returns, 0.50),
            "p90": quantile(returns, 0.90),
            "mean": mean(returns) if returns else None,
        },
        "signal_context_direction": direction,
        "signal_stat_window": {
            "horizon_days": HORIZON_DAYS,
            "horizon_hours": HORIZON_HOURS,
            "flat_threshold": FLAT_THRESHOLD,
            "sample_method": "nearest historical IC context",
            "max_similar_samples": MAX_SIMILAR_SAMPLES,
            "samples_used": n,
        },
    }


def patch_legacy_message(data: Dict[str, Any]) -> None:
    msg = data.get("message")
    if not isinstance(msg, str):
        return
    msg = msg.replace("următoarele ~24 ore", "următoarele ~3 zile")
    msg = msg.replace("urmatoarele ~24 ore", "următoarele ~3 zile")
    msg = msg.replace("următoarele 24 de ore", "următoarele 3 zile")
    msg = msg.replace("urmatoarele 24 de ore", "următoarele 3 zile")
    data["message"] = msg


def apply_to_state_file(path: Path, stats: Dict[str, Any]) -> None:
    data = load_json(path)
    if not data:
        print(f"[72h] Sar peste {path}: fișier lipsă sau gol.")
        return
    data.update(stats)
    patch_legacy_message(data)
    save_json(path, data)
    print(f"[72h] Actualizat {path}")


def main() -> None:
    if not SERIES_FILE.exists():
        raise RuntimeError(f"Lipsă {SERIES_FILE}. Rulează întâi export_ic_btc_series.py")
    with SERIES_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    series = payload.get("series", []) if isinstance(payload, dict) else []
    if not isinstance(series, list) or not series:
        raise RuntimeError("ic_btc_series.json nu conține o serie validă.")

    stats = compute_distribution(series)
    apply_to_state_file(DATA_STATE_FILE, stats)
    apply_to_state_file(WEB_STATE_FILE, stats)

    b = stats["signal_prob_breakdown"]
    d = stats["signal_expected_drift"]
    print(
        "[72h] Distribuție: "
        f"in_direction={b['in_direction']:.2%}, opposite={b['opposite']:.2%}, flat={b['flat']:.2%}; "
        f"p10={d['p10']:.2%}, p50={d['p50']:.2%}, p90={d['p90']:.2%}"
    )


if __name__ == "__main__":
    main()
