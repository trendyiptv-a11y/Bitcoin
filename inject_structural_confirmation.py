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
