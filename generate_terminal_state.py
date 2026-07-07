from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from engine import build_reasoning, build_terminal_scores

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "btc-swing-strategy" / "coeziv_state.json"
TERMINAL_PATH = ROOT / "btc-swing-strategy" / "cohesivx_terminal_state.json"


def _safe_get(data: Dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_terminal_state(state: Dict[str, Any]) -> Dict[str, Any]:
    scores = build_terminal_scores(state)
    reasoning = build_reasoning(state, scores)

    return {
        "schema": "cohesivx_terminal_state_v0.1",
        "version": "0.1-genesis",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "state_file": "btc-swing-strategy/coeziv_state.json",
            "timestamp": state.get("timestamp"),
            "generated_at": state.get("generated_at"),
            "price_source": state.get("price_source"),
            "model_price_source": state.get("model_price_source"),
            "model_price_method": state.get("model_price_method"),
        },
        "summary": {
            "market_state": scores.get("market_state"),
            "headline": reasoning.get("headline"),
            "text": reasoning.get("summary"),
            "signal": state.get("signal"),
            "market_regime": state.get("market_regime"),
        },
        "scores": scores,
        "market": {
            "monitor_spot_usd": state.get("price_usd"),
            "ic_close_usd": state.get("ic_close_usd"),
            "cohesive_central_usd": state.get("model_price_usd"),
            "model_deviation": state.get("model_price_deviation"),
            "model_deviation_pct": scores.get("model_deviation_pct"),
            "model_price_explanation": state.get("model_price_explanation"),
        },
        "structure": {
            "bands": state.get("model_price_bands"),
            "components": state.get("model_price_components"),
            "context": state.get("model_price_context"),
            "diagnostics": state.get("model_price_diagnostics"),
        },
        "miners": {
            "production_costs_usd": state.get("production_costs_usd"),
            "production_cost_reference": state.get("production_cost_reference"),
            "production_cost_as_of": state.get("production_cost_as_of"),
            "deviation_from_production": state.get("deviation_from_production"),
            "production_ratio": scores.get("production_ratio"),
            "miner_health": scores.get("miner_health"),
        },
        "sources": {
            "source_agreement": scores.get("source_agreement"),
            "source_agreement_note": scores.get("source_agreement_note"),
            "tradingview_anchors_available": bool(_safe_get(state, ["tradingview_anchors", "yearly"], {})),
            "active_year_override": _safe_get(state, ["tradingview_anchors", "active_year_override"]),
        },
        "flow_liquidity": {
            "flow_score": state.get("flow_score"),
            "flow_bias": state.get("flow_bias"),
            "flow_strength": state.get("flow_strength"),
            "liquidity_score": state.get("liquidity_score"),
            "liquidity_regime": state.get("liquidity_regime"),
            "liquidity_strength": state.get("liquidity_strength"),
        },
        "reasoning": reasoning.get("reasoning", []),
        "raw_reasoning": reasoning,
        "warnings": [
            "This terminal state is a structural diagnostic, not financial advice.",
            "TradingView chart close may differ from the official monitor spot when the selected symbol is an index such as BLX.",
        ],
    }


def main() -> None:
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"Lipsește {STATE_PATH}")

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    terminal_state = build_terminal_state(state)

    TERMINAL_PATH.write_text(
        json.dumps(terminal_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("CohesivX terminal state generat:", TERMINAL_PATH)
    print("Market state:", terminal_state["summary"].get("market_state"))
    print("Structural score:", terminal_state["scores"].get("structural_score"))
    print("Fragility:", terminal_state["scores"].get("fragility_index"))
    print("Confidence:", terminal_state["scores"].get("confidence_score"))


if __name__ == "__main__":
    main()
