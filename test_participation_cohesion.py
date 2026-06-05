#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "btc-swing-strategy" / "coeziv_state.json"
OUT_PATH = ROOT / "data" / "participation_cohesion_test.json"


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def num(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def norm_abs_small(x: float, scale: float) -> float:
    # 100 = abatere mică, 0 = abatere mare
    return clamp(100.0 * (1.0 - min(abs(x) / scale, 1.0)))


def load_state() -> Dict[str, Any]:
    with STATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def score_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    flow = num(state.get("flow_score"))
    liquidity = num(state.get("liquidity_score"))
    deviation = num(state.get("deviation_from_production"))
    prob = num(state.get("signal_probability"), 0.5)
    signal = str(state.get("signal", "flat")).lower()
    history = state.get("signal_history") or []

    recent = history[-14:] if isinstance(history, list) else []
    short_days = sum(1 for r in recent if str(r.get("signal", "")).lower() == "short")
    long_days = sum(1 for r in recent if str(r.get("signal", "")).lower() == "long")
    flat_days = sum(1 for r in recent if str(r.get("signal", "")).lower() == "flat")
    total_days = max(len(recent), 1)

    # Interpretare experimentală:
    # lichiditate bună = participare prezentă
    # flux negativ persistent = interes orientat spre ieșire/speculație defensivă
    # deviație mare față de cost = tensiune economică
    # semnal short persistent = coeziune participativă mai slabă
    liquidity_component = clamp(50.0 + liquidity * 900.0)
    flow_component = clamp(50.0 + flow * 1200.0)
    production_component = norm_abs_small(deviation, 0.35)
    persistence_component = clamp(100.0 - (short_days / total_days) * 70.0 + (long_days / total_days) * 20.0)
    probability_component = clamp(100.0 - abs(prob - 0.5) * 120.0)

    score = (
        0.30 * liquidity_component +
        0.25 * flow_component +
        0.20 * production_component +
        0.15 * persistence_component +
        0.10 * probability_component
    )
    score = round(clamp(score), 2)

    if score >= 70:
        label = "participare coezivă"
    elif score >= 50:
        label = "participare tensionată"
    elif score >= 30:
        label = "participare fragilă"
    else:
        label = "participare degradată"

    return {
        "score": score,
        "label": label,
        "signal": signal,
        "components": {
            "liquidity_component": round(liquidity_component, 2),
            "flow_component": round(flow_component, 2),
            "production_component": round(production_component, 2),
            "persistence_component": round(persistence_component, 2),
            "probability_component": round(probability_component, 2),
        },
        "inputs": {
            "flow_score": flow,
            "liquidity_score": liquidity,
            "deviation_from_production": deviation,
            "signal_probability": prob,
            "recent_days": total_days,
            "recent_short_days": short_days,
            "recent_long_days": long_days,
            "recent_flat_days": flat_days,
        },
        "interpretation": "Test experimental: estimează dacă interesul participanților pare coeziv, tensionat, fragil sau degradat folosind doar datele deja existente în coeziv_state.json.",
    }


def main() -> None:
    state = load_state()
    result = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source_timestamp": state.get("timestamp"),
        "source_generated_at": state.get("generated_at"),
        "participation_cohesion_test": score_from_state(state),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
