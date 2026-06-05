#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "btc-swing-strategy" / "coeziv_state.json"
IC_SERIES_PATH = ROOT / "data" / "ic_btc_series.json"
OUT_PATH = ROOT / "data" / "participation_cohesion_test.json"
OUT_SERIES = ROOT / "data" / "participation_cohesion_series.csv"
OUT_SUMMARY = ROOT / "data" / "participation_cohesion_history_summary.json"
OUT_CARD = ROOT / "btc-swing-strategy" / "participation_cohesion.json"


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def num(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def norm_abs_small(x: float, scale: float) -> float:
    return clamp(100.0 * (1.0 - min(abs(x) / scale, 1.0)))


def label_from_score(score: float) -> str:
    if score >= 70:
        return "participare coezivă"
    if score >= 50:
        return "participare tensionată"
    if score >= 30:
        return "participare fragilă"
    return "participare degradată"


def level_from_score(score: float) -> str:
    if score >= 70:
        return "cohesive"
    if score >= 50:
        return "tense"
    if score >= 30:
        return "fragile"
    return "degraded"


def load_state() -> Dict[str, Any]:
    with STATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ic_series() -> pd.DataFrame:
    with IC_SERIES_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw.get("series", []))
    if df.empty:
        raise RuntimeError("ic_btc_series.json nu conține date")
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("date").sort_index()
    for col in ["close", "ic_struct", "ic_dir", "ic_flux", "ic_cycle"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def score_core(flow: float, liquidity: float, deviation: float, prob: float, short_ratio: float, long_ratio: float) -> Dict[str, Any]:
    liquidity_component = clamp(50.0 + liquidity * 900.0)
    flow_component = clamp(50.0 + flow * 1200.0)
    production_component = norm_abs_small(deviation, 0.35)
    persistence_component = clamp(100.0 - short_ratio * 70.0 + long_ratio * 20.0)
    probability_component = clamp(100.0 - abs(prob - 0.5) * 120.0)

    score = (
        0.30 * liquidity_component +
        0.25 * flow_component +
        0.20 * production_component +
        0.15 * persistence_component +
        0.10 * probability_component
    )
    score = round(clamp(score), 2)

    return {
        "score": score,
        "label": label_from_score(score),
        "level": level_from_score(score),
        "components": {
            "liquidity_component": round(liquidity_component, 2),
            "flow_component": round(flow_component, 2),
            "production_component": round(production_component, 2),
            "persistence_component": round(persistence_component, 2),
            "probability_component": round(probability_component, 2),
        },
    }


def score_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    flow = num(state.get("flow_score"))
    liquidity = num(state.get("liquidity_score"))
    deviation = num(state.get("deviation_from_production"))
    prob = num(state.get("signal_probability"), 0.5)
    signal = str(state.get("signal", "flat")).lower()
    history = state.get("signal_history") or []

    recent = history[-14:] if isinstance(history, list) else []
    total_days = max(len(recent), 1)
    short_days = sum(1 for r in recent if str(r.get("signal", "")).lower() == "short")
    long_days = sum(1 for r in recent if str(r.get("signal", "")).lower() == "long")
    flat_days = sum(1 for r in recent if str(r.get("signal", "")).lower() == "flat")

    out = score_core(flow, liquidity, deviation, prob, short_days / total_days, long_days / total_days)
    out.update({
        "signal": signal,
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
        "interpretation": "Test experimental: estimează coeziunea participativă folosind datele deja existente în coeziv_state.json.",
    })
    return out


def infer_signal(row: pd.Series) -> str:
    regime = str(row.get("regime", "")).lower()
    ic_dir = num(row.get("ic_dir"))
    ic_flux = num(row.get("ic_flux"))
    if "bear" in regime or (ic_dir < -0.04 and ic_flux < 0):
        return "short"
    if "bull" in regime or (ic_dir > 0.04 and ic_flux > 0):
        return "long"
    return "flat"


def build_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signal_proxy"] = out.apply(infer_signal, axis=1)
    out["flow_proxy"] = out.get("ic_flux", 0).fillna(0).clip(-0.08, 0.08)
    out["liquidity_proxy"] = (out.get("ic_struct", 0).fillna(0).abs() * 0.04 + out.get("ic_flux", 0).fillna(0).abs() * 0.25).clip(0, 0.08)

    if "ic_cycle" in out.columns:
        cycle = out["ic_cycle"].replace(0, pd.NA)
        out["deviation_proxy"] = ((out["close"] / cycle) - 1.0).replace([math.inf, -math.inf], pd.NA).fillna(0).clip(-0.8, 0.8)
    else:
        ma = out["close"].rolling(200, min_periods=30).mean()
        out["deviation_proxy"] = ((out["close"] / ma) - 1.0).replace([math.inf, -math.inf], pd.NA).fillna(0).clip(-0.8, 0.8)

    out["prob_proxy"] = 0.5
    out.loc[out["signal_proxy"] == "short", "prob_proxy"] = 0.42
    out.loc[out["signal_proxy"] == "long", "prob_proxy"] = 0.58

    short_roll = (out["signal_proxy"] == "short").rolling(14, min_periods=1).mean()
    long_roll = (out["signal_proxy"] == "long").rolling(14, min_periods=1).mean()

    records: List[Dict[str, Any]] = []
    for ts, row in out.iterrows():
        sc = score_core(
            flow=num(row.get("flow_proxy")),
            liquidity=num(row.get("liquidity_proxy")),
            deviation=num(row.get("deviation_proxy")),
            prob=num(row.get("prob_proxy"), 0.5),
            short_ratio=num(short_roll.loc[ts]),
            long_ratio=num(long_roll.loc[ts]),
        )
        comps = sc["components"]
        records.append({
            "date": ts.isoformat(),
            "close": num(row.get("close")),
            "regime": str(row.get("regime", "")),
            "signal_proxy": row.get("signal_proxy"),
            "participation_score": sc["score"],
            "participation_label": sc["label"],
            "participation_level": sc["level"],
            "liquidity_component": comps["liquidity_component"],
            "flow_component": comps["flow_component"],
            "production_component": comps["production_component"],
            "persistence_component": comps["persistence_component"],
            "probability_component": comps["probability_component"],
            "flow_proxy": num(row.get("flow_proxy")),
            "liquidity_proxy": num(row.get("liquidity_proxy")),
            "deviation_proxy": num(row.get("deviation_proxy")),
        })
    return pd.DataFrame(records)


def summarize_history(hist: pd.DataFrame) -> Dict[str, Any]:
    labels = hist["participation_label"].value_counts().to_dict()
    worst = hist.nsmallest(10, "participation_score")[["date", "close", "regime", "signal_proxy", "participation_score", "participation_label"]].to_dict("records")
    best = hist.nlargest(10, "participation_score")[["date", "close", "regime", "signal_proxy", "participation_score", "participation_label"]].to_dict("records")
    return {
        "rows": int(len(hist)),
        "from": hist["date"].iloc[0] if len(hist) else None,
        "to": hist["date"].iloc[-1] if len(hist) else None,
        "mean_score": float(hist["participation_score"].mean()) if len(hist) else None,
        "median_score": float(hist["participation_score"].median()) if len(hist) else None,
        "min_score": float(hist["participation_score"].min()) if len(hist) else None,
        "max_score": float(hist["participation_score"].max()) if len(hist) else None,
        "label_counts": labels,
        "worst_10": worst,
        "best_10": best,
        "note": "Serie istorică experimentală. Folosește proxy-uri din ic_btc_series.json, nu încă date on-chain directe despre utilizarea peer-to-peer.",
    }


def build_card(snapshot: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    test = snapshot["participation_cohesion_test"]
    score = num(test.get("score"))
    label = str(test.get("label", "participare necunoscută"))
    level = str(test.get("level", level_from_score(score)))
    comps = test.get("components") or {}
    inputs = test.get("inputs") or {}

    if score >= 70:
        main_text = "Participanții par activi și relativ coezivi. Interesul pentru ecosistem persistă, iar comportamentul observabil nu indică abandon."
    elif score >= 50:
        main_text = "Participanții rămân activi, dar comportamentul este tensionat. Interesul persistă, însă fluxul dominant poate fi defensiv sau orientat spre ieșire."
    elif score >= 30:
        main_text = "Participarea pare fragilă. Activitatea există, dar coeziunea comportamentală este slăbită."
    else:
        main_text = "Participarea pare degradată. Modelul observă semne de retragere puternică a interesului participanților."

    return {
        "title": "COEZIUNE_PARTICIPATIVA",
        "score": score,
        "label": label,
        "level": level,
        "signal": test.get("signal"),
        "main_text": main_text,
        "components": comps,
        "inputs": inputs,
        "history": {
            "rows": summary.get("rows"),
            "from": summary.get("from"),
            "to": summary.get("to"),
            "mean_score": summary.get("mean_score"),
            "median_score": summary.get("median_score"),
            "min_score": summary.get("min_score"),
            "max_score": summary.get("max_score"),
            "label_counts": summary.get("label_counts"),
        },
        "footer": "Indicator experimental derivat din flux, lichiditate, persistență și tensiune față de model. Nu măsoară încă direct utilizarea peer-to-peer on-chain.",
        "generated_at": snapshot.get("generated_at"),
        "source_timestamp": snapshot.get("source_timestamp"),
    }


def main() -> None:
    state = load_state()
    snapshot_result = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source_timestamp": state.get("timestamp"),
        "source_generated_at": state.get("generated_at"),
        "participation_cohesion_test": score_from_state(state),
    }

    df = load_ic_series()
    hist = build_history(df)
    hist_summary = summarize_history(hist)
    card = build_card(snapshot_result, hist_summary)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_CARD.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(snapshot_result, ensure_ascii=False, indent=2), encoding="utf-8")
    hist.to_csv(OUT_SERIES, index=False)
    OUT_SUMMARY.write_text(json.dumps(hist_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_CARD.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "snapshot": snapshot_result,
        "history_summary": hist_summary,
        "card": card,
        "outputs": {
            "snapshot_json": str(OUT_PATH),
            "history_csv": str(OUT_SERIES),
            "history_summary_json": str(OUT_SUMMARY),
            "card_json": str(OUT_CARD),
        }
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
