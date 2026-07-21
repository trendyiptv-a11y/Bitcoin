from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

import coeziv_state as base
from cohesive_fair_price import compute_cohesive_fair_price_v2
from trader_signal_card import build_trader_signal_card

BASE_DIR = base.BASE_DIR
STRATEGY_DIR = base.STRATEGY_DIR


def _safe_get_costs() -> tuple[Dict[str, Optional[float]], Optional[str]]:
    try:
        cheap_cost, as_of = base.estimate_production_cost(profile="cheap")
        avg_cost, as_of = base.estimate_production_cost(profile="average")
        exp_cost, as_of = base.estimate_production_cost(profile="expensive")
        return {"cheap": cheap_cost, "average": avg_cost, "expensive": exp_cost}, as_of
    except Exception as exc:
        print("Nu am putut estima costurile de producție BTC.", exc)
        return {}, None


def _safe_flow() -> Dict[str, Any]:
    try:
        return base.compute_flow_from_file()
    except Exception as exc:
        print("Nu am putut calcula Flow Score:", exc)
        return {"flow_score": None, "flow_bias": None, "flow_strength": None, "components": {}}


def _safe_liquidity() -> Dict[str, Any]:
    try:
        return base.compute_liquidity_from_file()
    except Exception as exc:
        print("Nu am putut calcula Liquidity Score:", exc)
        return {"liquidity_score": None, "liquidity_regime": None, "liquidity_strength": None, "components": {}}


def _fmt_usd(value: Any) -> str:
    try:
        v = float(value)
        if math.isfinite(v):
            return f"~{v:,.0f} USD"
    except Exception:
        pass
    return "n/a"


def _fmt_pct(value: Any) -> str:
    try:
        v = float(value)
        if math.isfinite(v):
            sign = "+" if v > 0 else "−" if v < 0 else ""
            return f"{sign}{abs(v) * 100:.1f}%"
    except Exception:
        pass
    return "n/a"


def _build_model_price_explanation(fair_price: Optional[Dict[str, Any]], model_price: float, price_for_text: float, dev_pct_model: Optional[float]) -> Optional[str]:
    if not fair_price or not (math.isfinite(model_price) and model_price > 0):
        return None
    comps = fair_price.get("components") or {}
    bands = fair_price.get("bands") or {}
    samples = comps.get("similar_context_samples")
    multiplier = comps.get("historical_multiplier_p50")
    cost_anchor = comps.get("production_cost_anchor_usd")
    p10 = bands.get("p10")
    p90 = bands.get("p90")
    relation = "sub" if dev_pct_model is not None and dev_pct_model < 0 else "peste"
    first = f"Prețul coeziv central este {_fmt_usd(model_price)}, calculat din costul de producție actual"
    if cost_anchor is not None:
        first += f" ({_fmt_usd(cost_anchor)})"
    pieces = [first]
    if samples is not None and multiplier is not None:
        try:
            pieces.append(f"și din {int(samples)} contexte istorice similare, unde multiplicatorul median preț/cost este {float(multiplier):.3f}×.")
        except Exception:
            pieces.append("și din contexte istorice similare ale mecanismului.")
    elif samples is not None:
        try:
            pieces.append(f"și din {int(samples)} contexte istorice similare.")
        except Exception:
            pieces.append("și din contexte istorice similare ale mecanismului.")
    else:
        pieces.append("și din contexte istorice similare ale mecanismului.")
    if dev_pct_model is not None and math.isfinite(dev_pct_model):
        pieces.append(f"Prețul live este {_fmt_pct(dev_pct_model)} {relation} acest reper coeziv.")
    if p10 is not None and p90 is not None:
        pieces.append(f"Banda statistică a modelului este {_fmt_usd(p10)} – {_fmt_usd(p90)}; banda superioară nu este target, ci limită statistică a multiplicatorului istoric.")
    return " ".join(pieces)


def main() -> None:
    df = base.generate_signals(base.load_ic_series())
    if df is None or df.empty:
        raise RuntimeError("DataFrame-ul cu semnale este gol după generate_signals().")
    last = df.iloc[-1]
    ts = last.name
    signal = str(last.get("signal", "neutral")).lower()
    if signal not in ("long", "short", "flat", "neutral"):
        signal = "neutral"
    try:
        ic_close_price = float(last["close"])
    except Exception:
        raise RuntimeError("Coloana 'close' lipsește sau nu poate fi convertită la float.")

    price_source = "model"
    price_for_text = ic_close_price
    try:
        live_price = base.get_live_btc_price()
        if math.isfinite(live_price) and live_price > 0:
            price_for_text = float(live_price)
            price_source = "spot"
    except Exception as exc:
        print("Nu am putut obține prețul live BTC. Folosesc prețul din IC.", exc)

    production_costs, production_as_of = _safe_get_costs()
    deviation_from_production = None
    try:
        ref_cost = production_costs.get("average")
        if ref_cost is not None and math.isfinite(ref_cost) and ref_cost > 0:
            ref_price = price_for_text if math.isfinite(price_for_text) and price_for_text > 0 else ic_close_price
            deviation_from_production = (ref_price - ref_cost) / ref_cost
    except Exception:
        deviation_from_production = None

    model_price = ic_close_price
    model_price_method = "ic_close_fallback"
    model_price_source = "ic_close_fallback"
    fair_price = None
    try:
        avg_cost = production_costs.get("average")
        if avg_cost is not None and math.isfinite(avg_cost) and avg_cost > 0:
            fair_price = compute_cohesive_fair_price_v2(BASE_DIR, current_production_cost=float(avg_cost), spot_price=float(price_for_text), samples=250)
            candidate = fair_price.get("model_price_usd")
            if candidate is not None and math.isfinite(float(candidate)) and float(candidate) > 0:
                model_price = float(candidate)
                model_price_method = str(fair_price.get("method") or "production_cost_context_similarity_v2")
                model_price_source = "cohesive_fair_price_v2"
    except Exception as exc:
        print("Nu am putut calcula prețul coeziv V2. Folosesc fallback IC close.", exc)
        fair_price = None

    dev_pct_model = None
    try:
        if math.isfinite(model_price) and model_price > 0 and math.isfinite(price_for_text):
            dev_pct_model = (price_for_text - model_price) / model_price
    except Exception:
        dev_pct_model = None

    market_regime = base.classify_market_regime(last, dev_pct_model)
    flow = _safe_flow()
    liq = _safe_liquidity()
    signal_history = base.build_signal_history(df, limit=30)
    stats = base.compute_signal_stats(df, horizon_hours=72, move_threshold=0.0075, min_samples=100)
    message = base.build_message(signal=signal, price=price_for_text, flow_bias=flow.get("flow_bias"), flow_strength=flow.get("flow_strength"), liquidity_regime=liq.get("liquidity_regime"), stats=stats, deviation_from_production=deviation_from_production)
    model_price_explanation = _build_model_price_explanation(fair_price, model_price, price_for_text, dev_pct_model)
    if model_price_explanation:
        message = f"{message} {model_price_explanation}"

    trader_signal_card = build_trader_signal_card(signal=signal, flow=flow, liq=liq, stats=stats, market_regime=market_regime, dev_pct_model=dev_pct_model, deviation_from_production=deviation_from_production)

    state: Dict[str, Any] = {
        "timestamp": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
        "price_usd": price_for_text,
        "model_price_usd": model_price,
        "ic_close_usd": ic_close_price,
        "price_source": price_source,
        "model_price_source": model_price_source,
        "model_price_method": model_price_method,
        "model_price_deviation": dev_pct_model,
        "model_price_explanation": model_price_explanation,
        "signal": signal,
        "trader_signal_card": trader_signal_card,
        "message": message,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signal_history": signal_history,
        "signal_probability": stats.get("probability"),
        "signal_prob_samples": stats.get("samples"),
        "signal_prob_horizon_hours": stats.get("horizon_hours"),
        "signal_prob_source": stats.get("source"),
        "signal_prob_breakdown": stats.get("breakdown"),
        "signal_expected_drift": stats.get("expected_drift"),
        "market_regime": market_regime,
        "production_costs_usd": production_costs,
        "production_cost_reference": "average",
        "production_cost_as_of": production_as_of,
        "deviation_from_production": deviation_from_production,
        "flow_score": flow.get("flow_score"),
        "flow_bias": flow.get("flow_bias"),
        "flow_strength": flow.get("flow_strength"),
        "flow_components": flow.get("components", {}),
        "liquidity_score": liq.get("liquidity_score"),
        "liquidity_regime": liq.get("liquidity_regime"),
        "liquidity_strength": liq.get("liquidity_strength"),
        "liquidity_components": liq.get("components", {}),
    }
    if fair_price:
        state["model_price_bands"] = fair_price.get("bands")
        state["model_price_components"] = fair_price.get("components")
        state["model_price_context"] = fair_price.get("current")
        state["model_price_diagnostics"] = {"aligned_historical_points": fair_price.get("aligned_historical_points"), "v2a_nearest_contexts": fair_price.get("v2a_nearest_contexts"), "v2b_same_regime": fair_price.get("v2b_same_regime"), "source": fair_price.get("source")}
    state = base._enrich_state_with_fg(state)
    os.makedirs(STRATEGY_DIR, exist_ok=True)
    output_path = os.path.join(STRATEGY_DIR, "coeziv_state.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print("Stare coezivă V2 generată:", output_path)
    print("Semnal:", signal, "| Sursă preț:", price_source)
    print("Preț spot/mesaj:", f"{price_for_text:,.2f} USD")
    print("IC close:", f"{ic_close_price:,.2f} USD")
    print("Preț coeziv model:", f"{model_price:,.2f} USD", "| metodă:", model_price_method)
    if dev_pct_model is not None:
        print("Deviație spot față de preț coeziv:", f"{dev_pct_model * 100:.2f}%")
    if model_price_explanation:
        print("Explicație preț coeziv:", model_price_explanation)


if __name__ == "__main__":
    main()
