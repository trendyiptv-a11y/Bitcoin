#!/usr/bin/env python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "btc-swing-strategy" / "coeziv_state.json"
SUMMARY_CANDIDATES = [
    ROOT / "btc-swing-strategy" / "comparative_backtest_summary.json",
    ROOT / "comparative_backtest_summary.json",
]

TACTICAL_KEYS = [
    "signal_probability",
    "signal_prob_samples",
    "signal_prob_horizon_hours",
    "signal_prob_source",
    "signal_prob_breakdown",
    "signal_expected_drift",
]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON invalid în {path}")
    return data


def _find_summary_path() -> Optional[Path]:
    for p in SUMMARY_CANDIDATES:
        if p.exists() and p.is_file():
            return p
    return None


def _get_threshold_block(summary: Dict[str, Any], horizon_days: int, threshold_name: str = "threshold_10pct") -> Optional[Dict[str, Any]]:
    try:
        block = summary["models"]["cohesive_v2"][f"horizon_{horizon_days}d"][threshold_name]
        rate = float(block["directional_hit_rate"])
        events = int(block["events"])
        below = int(block.get("below_model_events", 0))
        above = int(block.get("above_model_events", 0))
        return {
            "directional_hit_rate": rate,
            "events": events,
            "below_model_events": below,
            "above_model_events": above,
        }
    except Exception:
        return None


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _recent_signal_count(state: Dict[str, Any], signal: str) -> int:
    wanted = _norm(signal)
    history = state.get("signal_history") or []
    if not isinstance(history, list):
        return 0
    return sum(1 for row in history if isinstance(row, dict) and _norm(row.get("signal")) == wanted)


def _build_cohesive_fg_view(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Traducere publică a Fear & Greed prin contextul întreg al mecanismului.

    Păstrăm scorul brut și zona brută, dar adăugăm un strat semantic coeziv:
    - raw_zone / combined_zone = scor numeric brut
    - cohesive_* = mesaj final pentru UI
    """
    fg = state.get("fg") or {}
    raw_zone = _norm(fg.get("combined_zone") or "neutral")
    score = _safe_float(fg.get("combined"), 50.0) or 50.0

    signal = _norm(state.get("signal"))
    flow_bias = _norm(state.get("flow_bias"))
    flow_strength = _norm(state.get("flow_strength"))
    liquidity_regime = _norm(state.get("liquidity_regime"))

    model_deviation = _safe_float(state.get("model_price_deviation"), 0.0) or 0.0
    production_deviation = _safe_float(state.get("deviation_from_production"), 0.0) or 0.0

    structural = state.get("structural_confirmation") or {}
    structural_regime = _norm(structural.get("regime") or (state.get("model_price_context") or {}).get("regime"))
    structural_deviation = _safe_float(structural.get("current_deviation"), model_deviation) or model_deviation

    recent_growth_contexts = _recent_signal_count(state, "long")
    recent_downside_contexts = _recent_signal_count(state, "short")

    structural_bearish = structural_regime in {"bear_late", "bear_struct", "accum_bear"}
    deeply_below_internal_value = structural_deviation <= -0.20 or model_deviation <= -0.20
    weak_neutral_flow = flow_bias == "neutru" and flow_strength == "slab"
    no_growth_confirmation = recent_growth_contexts == 0 and signal in {"flat", "neutral", ""}
    liquidity_support = liquidity_regime in {"ridicată", "ridicata", "high"}

    structural_tension = structural_bearish or deeply_below_internal_value

    if raw_zone == "greed" and score < 65.0 and (structural_tension or (weak_neutral_flow and no_growth_confirmation)):
        return {
            "raw_zone": raw_zone,
            "cohesive_zone": "optimism_tensionat",
            "cohesive_label": "Optimism tensionat",
            "cohesive_description": (
                "Apetit de risc ușor, dar fără confirmare structurală. Piața nu arată panică, "
                "însă fluxul este slab, contextele de creștere lipsesc, iar structura rămâne nereparată."
            ),
            "cohesive_label_en": "Tense optimism",
            "cohesive_description_en": (
                "Mild risk appetite, but without structural confirmation. The market does not show panic, "
                "yet flow is weak, growth contexts are absent, and the structure remains unrepaired."
            ),
        }

    if raw_zone == "greed" and (structural_tension or weak_neutral_flow or no_growth_confirmation):
        return {
            "raw_zone": raw_zone,
            "cohesive_zone": "greed_fragil",
            "cohesive_label": "Greed fragil",
            "cohesive_description": (
                "Există apetit de risc, dar confirmarea este incompletă. Greed-ul devine sănătos doar "
                "dacă este susținut de flux pozitiv persistent, participare coezivă și reparare structurală."
            ),
            "cohesive_label_en": "Fragile greed",
            "cohesive_description_en": (
                "There is risk appetite, but confirmation is incomplete. Greed becomes healthy only if it is "
                "supported by persistent positive flow, cohesive participation and structural repair."
            ),
        }

    if raw_zone == "greed":
        return {
            "raw_zone": raw_zone,
            "cohesive_zone": "greed",
            "cohesive_label": "Greed",
            "cohesive_description": (
                "Apetit de risc prezent. Contextul este sănătos doar cât timp fluxul, participarea, "
                "lichiditatea și structura confirmă împreună."
            ),
            "cohesive_label_en": "Greed",
            "cohesive_description_en": (
                "Risk appetite is present. The context is healthy only while flow, participation, liquidity "
                "and structure confirm together."
            ),
        }

    if raw_zone == "extreme_greed":
        return {
            "raw_zone": raw_zone,
            "cohesive_zone": "extreme_greed",
            "cohesive_label": "Extreme Greed",
            "cohesive_description": "Euforie sau supra-extindere. Context sensibil la răcire, mai ales dacă fluxul sau participarea slăbesc.",
            "cohesive_label_en": "Extreme Greed",
            "cohesive_description_en": "Euphoria or over-extension. Sensitive to cooling, especially if flow or participation weakens.",
        }

    if raw_zone == "fear":
        return {
            "raw_zone": raw_zone,
            "cohesive_zone": "fear",
            "cohesive_label": "Fear",
            "cohesive_description": "Aversiune la risc și poziționare defensivă. Prioritatea este controlul expunerii.",
            "cohesive_label_en": "Fear",
            "cohesive_description_en": "Risk aversion and defensive positioning. Exposure control is the priority.",
        }

    if raw_zone == "extreme_fear":
        return {
            "raw_zone": raw_zone,
            "cohesive_zone": "extreme_fear",
            "cohesive_label": "Extreme Fear",
            "cohesive_description": "Panică sau capitulare structurală. Context fragil, cu risc de mișcări rapide și dezordonate.",
            "cohesive_label_en": "Extreme Fear",
            "cohesive_description_en": "Panic or structural capitulation. Fragile context, with risk of fast disorderly moves.",
        }

    label = "Neutru"
    description = "Echilibru relativ între risc și oportunitate; mecanismul nu vede un avantaj emoțional decisiv."
    if raw_zone == "neutral" and liquidity_support and production_deviation > 0.0 and structural_tension:
        label = "Neutru tensionat"
        description = "Nu există panică, dar contextul structural rămâne tensionat. Confirmarea direcției trebuie așteptată."

    return {
        "raw_zone": raw_zone,
        "cohesive_zone": "neutral" if raw_zone == "neutral" else raw_zone,
        "cohesive_label": label,
        "cohesive_description": description,
        "cohesive_label_en": "Tense neutral" if label == "Neutru tensionat" else "Neutral",
        "cohesive_description_en": (
            "There is no panic, but the structural context remains tense. Direction confirmation should be awaited."
            if label == "Neutru tensionat"
            else "Relative balance between risk and opportunity; the mechanism sees no decisive emotional edge."
        ),
    }


def _enrich_fg_with_cohesive_view(state: Dict[str, Any]) -> None:
    fg = state.get("fg")
    if not isinstance(fg, dict):
        return
    fg.update(_build_cohesive_fg_view(state))


def build_structural_confirmation(summary: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    h7 = _get_threshold_block(summary, 7)
    h30 = _get_threshold_block(summary, 30)
    if not h7 or not h30:
        raise RuntimeError("Nu pot extrage confirmarea structurală 7d/30d din comparative_backtest_summary.json")

    components = state.get("model_price_components") or {}
    context = state.get("model_price_context") or {}

    return {
        "source": "comparative_backtest_summary.json",
        "model": "cohesive_mechanism",
        "threshold": 0.10,
        "threshold_label": "deviații ample față de reperul coeziv",
        "horizon_7d": h7,
        "horizon_30d": h30,
        "similar_context_samples": components.get("similar_context_samples"),
        "regime": context.get("regime") or (state.get("market_regime") or {}).get("code"),
        "current_deviation": state.get("model_price_deviation"),
        "label": "Semnal structural, nu intraday",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"Nu există {STATE_PATH}")

    summary_path = _find_summary_path()
    if not summary_path:
        raise FileNotFoundError("Nu am găsit comparative_backtest_summary.json")

    state = _load_json(STATE_PATH)
    summary = _load_json(summary_path)

    state["structural_confirmation"] = build_structural_confirmation(summary, state)
    _enrich_fg_with_cohesive_view(state)

    # Scoatem vechea statistică tactică 24h/72h din state-ul principal, ca UI-ul să nu o mai trateze ca mecanism.
    for key in TACTICAL_KEYS:
        state.pop(key, None)

    # Dacă mesajul conține fraza veche cu 24h/72h, păstrăm mesajul general și explicația prețului coeziv.
    # Nu hardcodăm valori; doar eliminăm propozițiile vechi tactice, când apar.
    msg = str(state.get("message") or "")
    for marker in ["Istoric, în contexte similare, mișcarea pe următoarele", "Nu este o recomandare de tranzacționare"]:
        idx = msg.find(marker)
        if idx >= 0:
            if marker.startswith("Istoric"):
                tail_idx = msg.find("Nu este o recomandare de tranzacționare", idx)
                if tail_idx >= 0:
                    msg = msg[:idx].rstrip() + " " + msg[tail_idx:].lstrip()
            break
    state["message"] = msg.strip()

    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Confirmare structurală injectată din {summary_path} în {STATE_PATH}")


if __name__ == "__main__":
    main()
