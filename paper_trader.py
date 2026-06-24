#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "btc-swing-strategy" / "coeziv_state.json"
PAPER_STATE_PATH = ROOT / "btc-swing-strategy" / "paper_trading_state.json"
PAPER_LOG_PATH = ROOT / "btc-swing-strategy" / "paper_trading_log.csv"

STARTING_BALANCE_USDT = 1000.0
MAX_ENTRY_FRACTION = 0.10
MAX_BTC_EXPOSURE_FRACTION = 0.40
MIN_STRUCTURAL_CONFIRMATION = 0.55
MAX_CONTEXT_DISTANCE_FOR_ACCUMULATION = 0.60
STALE_DATA_ACTION = "OBSERVE_STALE_DATA"
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def state_date(state: dict[str, Any]) -> str:
    context = state.get("model_price_context") or {}
    date_value = context.get("date") or str(state.get("timestamp") or "")[:10]
    return str(date_value or "")[:10]


def fetch_live_btc_price() -> tuple[float | None, str]:
    try:
        req = urllib.request.Request(BINANCE_TICKER_URL, headers={"User-Agent": "CohesivX-Paper-Trader/1.0"})
        with urllib.request.urlopen(req, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
        price = as_float(payload.get("price"))
        if price > 0:
            return price, "binance_live"
    except Exception as exc:
        return None, f"binance_unavailable:{exc.__class__.__name__}"
    return None, "binance_unavailable:invalid_price"


def initial_paper_state() -> dict[str, Any]:
    return {
        "mode": "paper",
        "starting_balance_usdt": STARTING_BALANCE_USDT,
        "cash_usdt": STARTING_BALANCE_USDT,
        "btc_amount": 0.0,
        "portfolio_value_usdt": STARTING_BALANCE_USDT,
        "last_action": "INIT",
        "last_confidence": "none",
        "last_reason": ["Paper trader initialized. No real funds are used."],
        "last_run_at": None,
        "not_trading_advice": True,
    }


def get_structural_confirmation(state: dict[str, Any]) -> float:
    structural = state.get("structural_confirmation") or {}
    h7 = structural.get("horizon_7d") or {}
    h30 = structural.get("horizon_30d") or {}
    rates = [as_float(h7.get("directional_hit_rate")), as_float(h30.get("directional_hit_rate"))]
    rates = [r for r in rates if r > 0]
    return sum(rates) / len(rates) if rates else 0.0


def hydrate_cohesivx_state(state: dict[str, Any], paper: dict[str, Any], live_price: float | None, live_source: str) -> dict[str, Any]:
    context = state.get("model_price_context") or {}
    market_regime = state.get("market_regime") or {}
    production_costs = state.get("production_costs_usd") or {}
    model_bands = state.get("model_price_bands") or {}
    components = state.get("model_price_components") or {}
    diagnostics = state.get("model_price_diagnostics") or {}
    nearest = diagnostics.get("v2a_nearest_contexts") or {}
    same_regime = diagnostics.get("v2b_same_regime") or {}
    structural = state.get("structural_confirmation") or {}

    snapshot_price = as_float(state.get("price_usd"))
    execution_price = live_price if live_price and live_price > 0 else snapshot_price
    model_price = as_float(state.get("model_price_usd"))
    live_deviation = ((execution_price - model_price) / model_price) if model_price > 0 and execution_price > 0 else as_float(state.get("model_price_deviation"))

    cash = as_float(paper.get("cash_usdt"), STARTING_BALANCE_USDT)
    btc = as_float(paper.get("btc_amount"))
    portfolio_value = cash + btc * execution_price if execution_price > 0 else cash
    btc_exposure = (btc * execution_price / portfolio_value) if portfolio_value > 0 and execution_price > 0 else 0.0

    return {
        "state_date": state_date(state),
        "run_date_utc": utc_today(),
        "is_fresh_for_today": state_date(state) == utc_today(),
        "price": {
            "execution_price_usd": execution_price,
            "execution_price_source": live_source if live_price else "coeziv_state_snapshot_fallback",
            "snapshot_price_usd": snapshot_price,
            "live_price_usd": live_price,
            "ic_close_usd": as_float(state.get("ic_close_usd")),
            "cohesive_fair_price_usd": model_price,
            "cohesive_deviation": live_deviation,
            "cohesive_deviation_pct": live_deviation * 100,
            "snapshot_model_deviation": as_float(state.get("model_price_deviation")),
            "model_source": state.get("model_price_source"),
            "model_method": state.get("model_price_method"),
        },
        "production": {
            "reference": state.get("production_cost_reference"),
            "cheap_usd": as_float(production_costs.get("cheap")),
            "average_usd": as_float(production_costs.get("average")),
            "expensive_usd": as_float(production_costs.get("expensive")),
            "deviation_from_production": ((execution_price - as_float(production_costs.get("average"))) / as_float(production_costs.get("average"))) if as_float(production_costs.get("average")) > 0 and execution_price > 0 else as_float(state.get("deviation_from_production")),
            "snapshot_deviation_from_production": as_float(state.get("deviation_from_production")),
            "as_of": state.get("production_cost_as_of"),
        },
        "bands": {"p10": as_float(model_bands.get("p10")), "p50": as_float(model_bands.get("p50")), "p90": as_float(model_bands.get("p90"))},
        "flow": {"score": as_float(state.get("flow_score")), "bias": str(state.get("flow_bias") or "").lower(), "strength": str(state.get("flow_strength") or "").lower(), "components": state.get("flow_components") or {}},
        "liquidity": {"score": as_float(state.get("liquidity_score")), "regime": str(state.get("liquidity_regime") or "").lower(), "strength": str(state.get("liquidity_strength") or "").lower(), "components": state.get("liquidity_components") or {}},
        "regime": {"market_label": market_regime.get("label"), "market_code": str(market_regime.get("code") or "").lower(), "structural_code": str(context.get("regime") or "").lower(), "signal": str(state.get("signal") or "flat").lower()},
        "ic_vector": {"ic_struct": as_float(context.get("ic_struct")), "ic_dir": as_float(context.get("ic_dir")), "ic_flux": as_float(context.get("ic_flux")), "ic_cycle": as_float(context.get("ic_cycle")), "vol30_index": as_float(context.get("vol30_index"))},
        "historical_memory": {
            "aligned_points": as_int(diagnostics.get("aligned_historical_points")),
            "similar_context_samples": as_int(components.get("similar_context_samples") or nearest.get("samples")),
            "similar_context_distance_median": as_float(components.get("similar_context_distance_median") or nearest.get("distance_median")),
            "similar_multiplier_p10": as_float(nearest.get("multiplier_p10") or components.get("historical_multiplier_p10")),
            "similar_multiplier_p50": as_float(nearest.get("multiplier_p50") or components.get("historical_multiplier_p50")),
            "similar_multiplier_p90": as_float(nearest.get("multiplier_p90") or components.get("historical_multiplier_p90")),
            "same_regime_samples": as_int(components.get("same_regime_samples") or same_regime.get("samples")),
            "same_regime_price_p50": as_float(components.get("same_regime_price_p50") or same_regime.get("price_p50")),
            "same_regime_multiplier_p50": as_float(components.get("same_regime_multiplier_p50") or same_regime.get("multiplier_p50")),
            "same_regime_spot_deviation": as_float(same_regime.get("spot_deviation_from_p50")),
        },
        "structural_confirmation": {"available": bool(structural), "combined_rate": get_structural_confirmation(state), "raw": structural},
        "paper_portfolio": {"cash_usdt": cash, "btc_amount": btc, "portfolio_value_usdt": portfolio_value, "btc_exposure": btc_exposure, "btc_exposure_pct": btc_exposure * 100},
    }


def score_cohesivx_opportunity(snapshot: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    deviation = snapshot["price"]["cohesive_deviation"]
    production_dev = snapshot["production"]["deviation_from_production"]
    structural_regime = snapshot["regime"]["structural_code"]
    market_code = snapshot["regime"]["market_code"]
    flow_bias = snapshot["flow"]["bias"]
    flow_strength = snapshot["flow"]["strength"]
    liquidity_regime = snapshot["liquidity"]["regime"]
    liquidity_strength = snapshot["liquidity"]["strength"]
    confirmation = snapshot["structural_confirmation"]["combined_rate"]
    similar_samples = snapshot["historical_memory"]["similar_context_samples"]
    same_regime_samples = snapshot["historical_memory"]["same_regime_samples"]
    context_distance = snapshot["historical_memory"]["similar_context_distance_median"]
    ic_struct = snapshot["ic_vector"]["ic_struct"]
    ic_flux = snapshot["ic_vector"]["ic_flux"]
    btc_exposure = snapshot["paper_portfolio"]["btc_exposure"]

    if deviation <= -0.05:
        score += 2
        reasons.append(f"Live/execution price is {deviation * 100:.1f}% below the cohesive fair price.")
    if deviation <= -0.15:
        score += 1
        reasons.append("Cohesive discount is deep, but still structural, not automatic buy.")
    if production_dev >= 0.05:
        score += 1
        reasons.append(f"Execution price is {production_dev * 100:.1f}% above average production cost.")
    elif production_dev < -0.05:
        score -= 2
        reasons.append("Execution price is below average production cost; capital protection dominates.")
    if structural_regime == "bear_late":
        score += 2
        reasons.append("Structural regime is bear_late: pressure is mature rather than early panic.")
    elif structural_regime in {"range", "neutral"}:
        score += 1
        reasons.append(f"Structural regime is {structural_regime}: acceptable for observation.")
    elif structural_regime:
        score -= 1
        reasons.append(f"Structural regime is {structural_regime}: not ideal for accumulation.")
    if "dev_extreme" in market_code:
        score -= 1
        reasons.append("Market regime reports extreme deviation; sizing remains conservative.")
    if "ridic" in liquidity_regime or "putern" in liquidity_strength or "strong" in liquidity_strength:
        score += 1
        reasons.append("Liquidity is high/strong, reducing paper-execution fragility.")
    if flow_bias in {"neutru", "neutral"} and flow_strength in {"slab", "weak"}:
        score += 1
        reasons.append("Flow is neutral/weak, so no aggressive directional chase.")
    elif "neg" in flow_bias or "bear" in flow_bias:
        score -= 1
        reasons.append("Flow is negative; confidence is reduced.")
    if confirmation >= MIN_STRUCTURAL_CONFIRMATION:
        score += 1
        reasons.append(f"Structural confirmation is {confirmation * 100:.1f}%, above threshold.")
    elif snapshot["structural_confirmation"]["available"]:
        score -= 1
        reasons.append("Structural confirmation exists but is below threshold.")
    else:
        reasons.append("Structural confirmation is not available; no confirmation bonus is applied.")
    if similar_samples >= 200:
        score += 1
        reasons.append(f"Historical memory is well populated: {similar_samples} similar contexts.")
    if same_regime_samples >= 50:
        score += 1
        reasons.append(f"Same-regime memory has {same_regime_samples} samples.")
    if context_distance and context_distance > MAX_CONTEXT_DISTANCE_FOR_ACCUMULATION:
        score -= 1
        reasons.append(f"Median context distance is {context_distance:.3f}, above preferred threshold.")
    if ic_struct >= 55 and ic_flux >= 50:
        score += 1
        reasons.append(f"IC vector is structurally alive: ic_struct={ic_struct:.1f}, ic_flux={ic_flux:.1f}.")
    if btc_exposure >= MAX_BTC_EXPOSURE_FRACTION:
        score -= 4
        reasons.append("Paper BTC exposure is already at or above max cap.")
    return score, reasons


def decide(state: dict[str, Any], paper: dict[str, Any], live_price: float | None, live_source: str) -> tuple[str, str, list[str], float, dict[str, Any], int]:
    snapshot = hydrate_cohesivx_state(state, paper, live_price, live_source)
    price = snapshot["price"]["execution_price_usd"]
    model_price = snapshot["price"]["cohesive_fair_price_usd"]
    btc = snapshot["paper_portfolio"]["btc_amount"]
    cash = snapshot["paper_portfolio"]["cash_usdt"]
    btc_exposure = snapshot["paper_portfolio"]["btc_exposure"]

    if not snapshot["is_fresh_for_today"]:
        return STALE_DATA_ACTION, "none", [f"State date {snapshot['state_date']} is not UTC today {snapshot['run_date_utc']}.", "Paper trader refuses to act on stale CohesivX state."], 0.0, snapshot, -999
    if price <= 0 or model_price <= 0:
        return "OBSERVE", "low", ["Missing valid execution price or cohesive model price."], 0.0, snapshot, -999

    score, reasons = score_cohesivx_opportunity(snapshot)
    if score >= 7 and cash > 10 and btc_exposure < MAX_BTC_EXPOSURE_FRACTION:
        remaining = max(0.0, MAX_BTC_EXPOSURE_FRACTION - btc_exposure)
        fraction = min(MAX_ENTRY_FRACTION, remaining)
        reasons.append(f"CohesivX score {score}: small paper accumulation allowed at {fraction * 100:.1f}% portfolio allocation.")
        return "ACCUMULATE_SMALL", "moderate", reasons, fraction, snapshot, score
    if score >= 4 and cash > 10 and btc_exposure < MAX_BTC_EXPOSURE_FRACTION:
        remaining = max(0.0, MAX_BTC_EXPOSURE_FRACTION - btc_exposure)
        fraction = min(MAX_ENTRY_FRACTION / 2, remaining)
        reasons.append(f"CohesivX score {score}: half-size paper accumulation allowed.")
        return "OBSERVE_ACCUMULATE_SMALL", "moderate_low", reasons, fraction, snapshot, score
    if btc > 0 and snapshot["price"]["cohesive_deviation"] > 0.10:
        reasons.append("Execution price is more than 10% above cohesive fair price; paper exposure is reduced.")
        return "REDUCE_RISK", "moderate", reasons, min(MAX_ENTRY_FRACTION, btc_exposure), snapshot, score
    if btc > 0:
        reasons.append(f"CohesivX score {score}: existing paper BTC is held.")
        return "HOLD", "low", reasons, 0.0, snapshot, score
    reasons.append(f"CohesivX score {score}: no paper entry; conditions are incomplete.")
    return "OBSERVE", "low", reasons, 0.0, snapshot, score


def apply_paper_action(paper: dict[str, Any], execution_price: float, action: str, fraction: float) -> dict[str, Any]:
    cash = as_float(paper.get("cash_usdt"), STARTING_BALANCE_USDT)
    btc = as_float(paper.get("btc_amount"))
    portfolio_before = cash + btc * execution_price if execution_price > 0 else cash
    executed_usdt = 0.0
    executed_btc = 0.0
    if execution_price > 0 and action in {"ACCUMULATE_SMALL", "OBSERVE_ACCUMULATE_SMALL"} and fraction > 0:
        target = min(cash, portfolio_before * fraction)
        if target >= 5:
            executed_usdt = target
            executed_btc = target / execution_price
            cash -= target
            btc += executed_btc
    elif execution_price > 0 and action == "REDUCE_RISK" and fraction > 0 and btc > 0:
        target = min(btc * execution_price, portfolio_before * fraction)
        if target >= 5:
            executed_usdt = target
            executed_btc = -(target / execution_price)
            cash += target
            btc += executed_btc
    portfolio_after = cash + btc * execution_price if execution_price > 0 else cash
    return {"cash_usdt": cash, "btc_amount": btc, "executed_usdt": executed_usdt, "executed_btc": executed_btc, "portfolio_value_before": portfolio_before, "portfolio_value_after": portfolio_after}


def append_log(row: dict[str, Any]) -> None:
    PAPER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_at", "state_timestamp", "state_date", "execution_price_source", "execution_price_usd", "snapshot_price_usd", "model_price_usd", "deviation_pct", "production_deviation_pct", "structural_regime", "market_regime_code", "signal", "flow_bias", "flow_strength", "liquidity_regime", "liquidity_strength", "similar_context_samples", "same_regime_samples", "structural_confirmation_pct", "cohesivx_score", "action", "confidence", "executed_usdt", "executed_btc", "cash_usdt", "btc_amount", "btc_exposure_pct", "portfolio_value_usdt", "reason"]
    exists = PAPER_LOG_PATH.exists()
    with PAPER_LOG_PATH.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"Missing {STATE_PATH}")
    state = load_json(STATE_PATH, {})
    paper = load_json(PAPER_STATE_PATH, initial_paper_state()) or initial_paper_state()
    live_price, live_source = fetch_live_btc_price()
    action, confidence, reasons, fraction, snapshot, score = decide(state, paper, live_price, live_source)
    execution_price = snapshot["price"]["execution_price_usd"]
    execution = apply_paper_action(paper, execution_price, action, fraction)
    run_at = now_iso()
    portfolio_after = execution["portfolio_value_after"]
    btc_exposure_after = (execution["btc_amount"] * execution_price / portfolio_after) if execution_price > 0 and portfolio_after > 0 else 0.0

    updated = {**paper, "mode": "paper", "cash_usdt": round(execution["cash_usdt"], 8), "btc_amount": round(execution["btc_amount"], 12), "portfolio_value_usdt": round(portfolio_after, 8), "btc_exposure_pct": round(btc_exposure_after * 100, 4), "last_action": action, "last_confidence": confidence, "last_cohesivx_score": score, "last_reason": reasons, "last_run_at": run_at, "last_execution_price_usd": execution_price, "last_execution_price_source": snapshot["price"]["execution_price_source"], "last_snapshot_price_usd": snapshot["price"]["snapshot_price_usd"], "last_model_price_usd": snapshot["price"]["cohesive_fair_price_usd"], "last_deviation_pct": round(snapshot["price"]["cohesive_deviation_pct"], 4), "decision_snapshot": snapshot, "not_trading_advice": True, "rules": {"starting_balance_usdt": STARTING_BALANCE_USDT, "max_entry_fraction": MAX_ENTRY_FRACTION, "max_btc_exposure_fraction": MAX_BTC_EXPOSURE_FRACTION, "min_structural_confirmation": MIN_STRUCTURAL_CONFIRMATION, "max_context_distance_for_accumulation": MAX_CONTEXT_DISTANCE_FOR_ACCUMULATION, "logic": "CohesivX structural paper trading using live execution price plus cohesive fair price, production cost, regime, flow, liquidity, IC vector, historical memory and structural confirmation. No RSI/MACD/scalping logic."}}
    save_json(PAPER_STATE_PATH, updated)

    row = {"run_at": run_at, "state_timestamp": state.get("timestamp"), "state_date": snapshot["state_date"], "execution_price_source": snapshot["price"]["execution_price_source"], "execution_price_usd": execution_price, "snapshot_price_usd": snapshot["price"]["snapshot_price_usd"], "model_price_usd": snapshot["price"]["cohesive_fair_price_usd"], "deviation_pct": round(snapshot["price"]["cohesive_deviation_pct"], 4), "production_deviation_pct": round(snapshot["production"]["deviation_from_production"] * 100, 4), "structural_regime": snapshot["regime"]["structural_code"], "market_regime_code": snapshot["regime"]["market_code"], "signal": snapshot["regime"]["signal"], "flow_bias": snapshot["flow"]["bias"], "flow_strength": snapshot["flow"]["strength"], "liquidity_regime": snapshot["liquidity"]["regime"], "liquidity_strength": snapshot["liquidity"]["strength"], "similar_context_samples": snapshot["historical_memory"]["similar_context_samples"], "same_regime_samples": snapshot["historical_memory"]["same_regime_samples"], "structural_confirmation_pct": round(snapshot["structural_confirmation"]["combined_rate"] * 100, 4), "cohesivx_score": score, "action": action, "confidence": confidence, "executed_usdt": round(execution["executed_usdt"], 8), "executed_btc": round(execution["executed_btc"], 12), "cash_usdt": round(updated["cash_usdt"], 8), "btc_amount": round(updated["btc_amount"], 12), "btc_exposure_pct": round(updated["btc_exposure_pct"], 4), "portfolio_value_usdt": round(updated["portfolio_value_usdt"], 8), "reason": " | ".join(reasons)}
    append_log(row)
    print(f"Paper trader action: {action}")
    print(f"Confidence: {confidence}")
    print(f"CohesivX score: {score}")
    print(f"Execution price: {execution_price} ({snapshot['price']['execution_price_source']})")
    print(f"Executed USDT: {row['executed_usdt']}")
    print(f"Portfolio value: {row['portfolio_value_usdt']} USDT")


if __name__ == "__main__":
    main()
