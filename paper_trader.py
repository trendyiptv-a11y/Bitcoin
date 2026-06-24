#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "btc-swing-strategy" / "coeziv_state.json"
PAPER_STATE_PATH = ROOT / "btc-swing-strategy" / "paper_trading_state.json"
PAPER_LOG_PATH = ROOT / "btc-swing-strategy" / "paper_trading_log.csv"
DECISION_PATH = ROOT / "btc-swing-strategy" / "paper_trader_decision.json"

STARTING_BALANCE_USDT = 1000.0
MAX_ENTRY_FRACTION = 0.10
MAX_BTC_EXPOSURE_FRACTION = 0.40
MIN_ENTRY_FRACTION = 0.025
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
COINBASE_SPOT_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
KRAKEN_TICKER_URL = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_utc() -> str:
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


def f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        n = float(value)
        return n if math.isfinite(n) else default
    except (TypeError, ValueError):
        return default


def i(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def state_date(state: dict[str, Any]) -> str:
    context = state.get("model_price_context") or {}
    return str(context.get("date") or str(state.get("timestamp") or "")[:10] or "")[:10]


def fetch_url_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "CohesivX-Paper-Trader/1.0"})
    with urllib.request.urlopen(req, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_binance_price() -> float:
    payload = fetch_url_json(BINANCE_TICKER_URL)
    price = f(payload.get("price"))
    if price <= 0:
        raise ValueError("invalid Binance price")
    return price


def fetch_coinbase_price() -> float:
    payload = fetch_url_json(COINBASE_SPOT_URL)
    price = f((payload.get("data") or {}).get("amount"))
    if price <= 0:
        raise ValueError("invalid Coinbase price")
    return price


def fetch_kraken_price() -> float:
    payload = fetch_url_json(KRAKEN_TICKER_URL)
    result = payload.get("result") or {}
    if not result:
        raise ValueError("empty Kraken result")
    first = next(iter(result.values()))
    price = f((first.get("c") or [None])[0])
    if price <= 0:
        raise ValueError("invalid Kraken price")
    return price


def fetch_live_btc_price() -> tuple[float | None, str, dict[str, Any]]:
    sources = [
        ("binance", fetch_binance_price),
        ("coinbase", fetch_coinbase_price),
        ("kraken", fetch_kraken_price),
    ]
    status: dict[str, Any] = {"checked_at": now_iso(), "selected_source": None, "selected_price": None, "sources": {}}
    valid_prices: list[tuple[str, float]] = []
    for name, fn in sources:
        try:
            price = fn()
            status["sources"][name] = {"ok": True, "price": price}
            valid_prices.append((name, price))
        except Exception as exc:
            status["sources"][name] = {"ok": False, "error_type": exc.__class__.__name__, "error": str(exc)[:240]}
    if not valid_prices:
        return None, "live_price_unavailable", status

    if len(valid_prices) == 1:
        selected_name, selected_price = valid_prices[0]
        status["selected_source"] = selected_name
        status["selected_price"] = selected_price
        status["dispersion_pct"] = 0.0
        return selected_price, selected_name

    prices = [p for _, p in valid_prices]
    median = sorted(prices)[len(prices) // 2]
    dispersion = (max(prices) - min(prices)) / median if median > 0 else 0.0
    selected_name, selected_price = min(valid_prices, key=lambda item: abs(item[1] - median))
    status["selected_source"] = selected_name
    status["selected_price"] = selected_price
    status["median_price"] = median
    status["dispersion_pct"] = dispersion * 100
    return selected_price, selected_name, status


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


def structural_confirmation_rate(state: dict[str, Any]) -> float:
    structural = state.get("structural_confirmation") or {}
    h7 = structural.get("horizon_7d") or {}
    h30 = structural.get("horizon_30d") or {}
    rates = [f(h7.get("directional_hit_rate")), f(h30.get("directional_hit_rate"))]
    rates = [x for x in rates if x > 0]
    return sum(rates) / len(rates) if rates else 0.0


def hydrate(state: dict[str, Any], paper: dict[str, Any], live_price: float | None, live_source: str, live_status: dict[str, Any]) -> dict[str, Any]:
    ctx = state.get("model_price_context") or {}
    market = state.get("market_regime") or {}
    costs = state.get("production_costs_usd") or {}
    bands = state.get("model_price_bands") or {}
    comps = state.get("model_price_components") or {}
    diag = state.get("model_price_diagnostics") or {}
    nearest = diag.get("v2a_nearest_contexts") or {}
    same = diag.get("v2b_same_regime") or {}

    snapshot_price = f(state.get("price_usd"))
    execution_price = live_price if live_price and live_price > 0 else snapshot_price
    model_price = f(state.get("model_price_usd"))
    deviation = (execution_price - model_price) / model_price if model_price > 0 and execution_price > 0 else f(state.get("model_price_deviation"))
    avg_cost = f(costs.get("average"))
    production_dev = (execution_price - avg_cost) / avg_cost if avg_cost > 0 and execution_price > 0 else f(state.get("deviation_from_production"))

    cash = f(paper.get("cash_usdt"), STARTING_BALANCE_USDT)
    btc = f(paper.get("btc_amount"))
    portfolio_value = cash + btc * execution_price if execution_price > 0 else cash
    btc_exposure = btc * execution_price / portfolio_value if portfolio_value > 0 and execution_price > 0 else 0.0

    return {
        "state_date": state_date(state),
        "run_date_utc": today_utc(),
        "is_fresh_for_today": state_date(state) == today_utc(),
        "live_price_status": live_status,
        "price": {
            "execution_price_usd": execution_price,
            "execution_price_source": live_source if live_price else "coeziv_state_snapshot_fallback",
            "snapshot_price_usd": snapshot_price,
            "live_price_usd": live_price,
            "cohesive_fair_price_usd": model_price,
            "cohesive_deviation": deviation,
            "cohesive_deviation_pct": deviation * 100,
            "snapshot_model_deviation": f(state.get("model_price_deviation")),
        },
        "production": {
            "cheap_usd": f(costs.get("cheap")),
            "average_usd": avg_cost,
            "expensive_usd": f(costs.get("expensive")),
            "deviation_from_production": production_dev,
            "snapshot_deviation_from_production": f(state.get("deviation_from_production")),
        },
        "bands": {"p10": f(bands.get("p10")), "p50": f(bands.get("p50")), "p90": f(bands.get("p90"))},
        "flow": {"score": f(state.get("flow_score")), "bias": str(state.get("flow_bias") or "").lower(), "strength": str(state.get("flow_strength") or "").lower()},
        "liquidity": {"score": f(state.get("liquidity_score")), "regime": str(state.get("liquidity_regime") or "").lower(), "strength": str(state.get("liquidity_strength") or "").lower()},
        "regime": {"market_label": market.get("label"), "market_code": str(market.get("code") or "").lower(), "structural_code": str(ctx.get("regime") or "").lower(), "signal": str(state.get("signal") or "flat").lower()},
        "ic_vector": {"ic_struct": f(ctx.get("ic_struct")), "ic_dir": f(ctx.get("ic_dir")), "ic_flux": f(ctx.get("ic_flux")), "ic_cycle": f(ctx.get("ic_cycle")), "vol30_index": f(ctx.get("vol30_index"))},
        "historical_memory": {
            "aligned_points": i(diag.get("aligned_historical_points")),
            "similar_context_samples": i(comps.get("similar_context_samples") or nearest.get("samples")),
            "similar_context_distance_median": f(comps.get("similar_context_distance_median") or nearest.get("distance_median")),
            "similar_price_p10": f(nearest.get("price_p10") or bands.get("p10")),
            "similar_price_p50": f(nearest.get("price_p50") or bands.get("p50")),
            "similar_price_p90": f(nearest.get("price_p90") or bands.get("p90")),
            "similar_multiplier_p50": f(nearest.get("multiplier_p50") or comps.get("historical_multiplier_p50")),
            "same_regime_samples": i(comps.get("same_regime_samples") or same.get("samples")),
            "same_regime_price_p10": f(same.get("price_p10")),
            "same_regime_price_p50": f(comps.get("same_regime_price_p50") or same.get("price_p50")),
            "same_regime_price_p90": f(same.get("price_p90")),
            "same_regime_spot_deviation": f(same.get("spot_deviation_from_p50")),
        },
        "structural_confirmation": {"combined_rate": structural_confirmation_rate(state), "raw": state.get("structural_confirmation") or {}},
        "paper_portfolio": {"cash_usdt": cash, "btc_amount": btc, "portfolio_value_usdt": portfolio_value, "btc_exposure": btc_exposure, "btc_exposure_pct": btc_exposure * 100},
    }


def estimate_memory_edge(s: dict[str, Any]) -> dict[str, Any]:
    p = s["price"]["execution_price_usd"]
    mem = s["historical_memory"]
    regime = s["regime"]["structural_code"]
    market_code = s["regime"]["market_code"]
    flow_bias = s["flow"]["bias"]
    flow_strength = s["flow"]["strength"]
    liq_regime = s["liquidity"]["regime"]
    liq_strength = s["liquidity"]["strength"]
    ic = s["ic_vector"]
    confirmation = s["structural_confirmation"]["combined_rate"] or 0.5
    similar_samples = mem["similar_context_samples"]
    same_samples = mem["same_regime_samples"]
    distance = mem["similar_context_distance_median"]
    similar_p10 = mem["similar_price_p10"]
    similar_p50 = mem["similar_price_p50"]
    similar_p90 = mem["similar_price_p90"]
    same_p10 = mem["same_regime_price_p10"] or similar_p10
    same_p50 = mem["same_regime_price_p50"] or similar_p50
    same_p90 = mem["same_regime_price_p90"] or similar_p90
    if p <= 0 or similar_p50 <= 0:
        return {"available": False, "reason": "missing price or memory distribution"}

    sample_conf = clamp(similar_samples / 250.0, 0.0, 1.0)
    same_regime_conf = clamp(same_samples / max(similar_samples, 1), 0.0, 1.0)
    distance_conf = clamp(1.0 - max(distance, 0.0) / 0.80, 0.0, 1.0) if distance else 0.50
    confirmation_conf = clamp(confirmation, 0.35, 0.75)
    context_confidence = clamp(0.35 * sample_conf + 0.20 * same_regime_conf + 0.25 * distance_conf + 0.20 * confirmation_conf, 0.0, 1.0)
    weighted_p10 = 0.60 * similar_p10 + 0.40 * same_p10
    weighted_p50 = 0.60 * similar_p50 + 0.40 * same_p50
    weighted_p90 = 0.60 * similar_p90 + 0.40 * same_p90
    expected_30d = (weighted_p50 - p) / p
    expected_7d = expected_30d * 0.35
    downside_to_p10 = min(0.0, (weighted_p10 - p) / p)
    upside_to_p90 = max(0.0, (weighted_p90 - p) / p)

    regime_adjust = 0.0
    if regime == "bear_late": regime_adjust += 0.06
    elif regime in {"range", "neutral"}: regime_adjust += 0.02
    elif regime.startswith("bull"): regime_adjust += 0.03
    elif regime.startswith("bear"): regime_adjust -= 0.04
    if "dev_extreme" in market_code: regime_adjust -= 0.03
    if flow_bias in {"neutru", "neutral"} and flow_strength in {"slab", "weak"}: regime_adjust += 0.02
    elif "neg" in flow_bias or "bear" in flow_bias: regime_adjust -= 0.04
    if "ridic" in liq_regime or "putern" in liq_strength or "strong" in liq_strength: regime_adjust += 0.02
    if ic["ic_struct"] >= 55 and ic["ic_flux"] >= 50: regime_adjust += 0.03
    raw_edge = context_confidence * expected_30d - (1 - context_confidence) * abs(downside_to_p10)
    decision_edge = raw_edge + regime_adjust
    drawdown_risk = abs(downside_to_p10)
    if decision_edge > 0.18 and context_confidence >= 0.58 and drawdown_risk < 0.28:
        action, fraction, confidence_label = "ACCUMULATE_SMALL", MAX_ENTRY_FRACTION, "memory_moderate"
    elif decision_edge > 0.07 and context_confidence >= 0.45 and drawdown_risk < 0.38:
        action, fraction, confidence_label = "OBSERVE_ACCUMULATE_SMALL", 0.05, "memory_moderate_low"
    elif expected_30d < -0.08 or decision_edge < -0.06:
        action, fraction, confidence_label = "REDUCE_RISK", 0.10, "memory_defensive"
    else:
        action, fraction, confidence_label = "OBSERVE", 0.0, "memory_low"
    if fraction > 0 and context_confidence < 0.50:
        fraction = min(fraction, MIN_ENTRY_FRACTION)
    return {"available": True, "weighted_price_p10": weighted_p10, "weighted_price_p50": weighted_p50, "weighted_price_p90": weighted_p90, "expected_7d_return": expected_7d, "expected_30d_return": expected_30d, "historical_drawdown_risk": drawdown_risk, "upside_to_p90": upside_to_p90, "memory_confidence": context_confidence, "sample_confidence": sample_conf, "same_regime_confidence": same_regime_conf, "distance_confidence": distance_conf, "confirmation_confidence": confirmation_conf, "regime_adjustment": regime_adjust, "decision_edge": decision_edge, "memory_action": action, "position_fraction": fraction, "confidence_label": confidence_label, "method": "memory_weighted_distribution_v0.3"}


def decide(state: dict[str, Any], paper: dict[str, Any], live_price: float | None, live_source: str, live_status: dict[str, Any]) -> tuple[str, str, list[str], float, dict[str, Any], dict[str, Any]]:
    s = hydrate(state, paper, live_price, live_source, live_status)
    if not s["is_fresh_for_today"]:
        edge = {"available": False, "memory_action": "OBSERVE_STALE_DATA", "position_fraction": 0.0, "decision_edge": 0.0, "reason": "stale state"}
        return "OBSERVE_STALE_DATA", "none", [f"State date {s['state_date']} is not UTC today {s['run_date_utc']}.", "No paper execution is allowed on stale data."], 0.0, s, edge
    edge = estimate_memory_edge(s)
    if not live_price:
        edge["execution_blocked"] = True
        edge["execution_block_reason"] = "all_live_sources_failed"
        return "OBSERVE_LIVE_PRICE_UNAVAILABLE", "none", ["Live execution price is unavailable in the runner.", "All live price sources failed or returned invalid data.", "Professional safety rule: no paper buy/sell is executed using snapshot fallback."], 0.0, s, edge

    price_ok = s["price"]["execution_price_usd"] > 0 and s["price"]["cohesive_fair_price_usd"] > 0
    if not price_ok or not edge.get("available"):
        return "OBSERVE", "low", ["Missing valid execution price, cohesive price or memory distribution."], 0.0, s, edge

    exposure = s["paper_portfolio"]["btc_exposure"]
    cash = s["paper_portfolio"]["cash_usdt"]
    btc = s["paper_portfolio"]["btc_amount"]
    action = str(edge["memory_action"])
    fraction = float(edge["position_fraction"])
    confidence = str(edge["confidence_label"])
    if action in {"ACCUMULATE_SMALL", "OBSERVE_ACCUMULATE_SMALL"}:
        if cash <= 10 or exposure >= MAX_BTC_EXPOSURE_FRACTION:
            action, fraction, confidence = "HOLD" if btc > 0 else "OBSERVE", 0.0, "risk_cap"
        else:
            fraction = min(fraction, MAX_BTC_EXPOSURE_FRACTION - exposure, MAX_ENTRY_FRACTION)
    elif action == "REDUCE_RISK" and btc <= 0:
        action, fraction = "OBSERVE", 0.0
    reasons = [f"Memory edge: {edge['decision_edge'] * 100:.2f}%.", f"Expected 7d/30d: {edge['expected_7d_return'] * 100:.2f}% / {edge['expected_30d_return'] * 100:.2f}%.", f"Historical drawdown risk to weighted p10: {edge['historical_drawdown_risk'] * 100:.2f}%.", f"Memory confidence: {edge['memory_confidence'] * 100:.1f}%.", f"Weighted memory p50: {edge['weighted_price_p50']:.2f} USD vs execution price {s['price']['execution_price_usd']:.2f} USD.", f"Live source: {live_source}.", f"Structural regime: {s['regime']['structural_code']}; market regime: {s['regime']['market_code']}.", f"Flow: {s['flow']['bias']}/{s['flow']['strength']}; liquidity: {s['liquidity']['regime']}/{s['liquidity']['strength']}."]
    return action, confidence, reasons, fraction, s, edge


def apply_action(paper: dict[str, Any], price: float, action: str, fraction: float) -> dict[str, Any]:
    cash = f(paper.get("cash_usdt"), STARTING_BALANCE_USDT)
    btc = f(paper.get("btc_amount"))
    before = cash + btc * price if price > 0 else cash
    executed_usdt = 0.0
    executed_btc = 0.0
    if price > 0 and action in {"ACCUMULATE_SMALL", "OBSERVE_ACCUMULATE_SMALL"} and fraction > 0:
        executed_usdt = min(cash, before * fraction)
        if executed_usdt >= 5:
            executed_btc = executed_usdt / price
            cash -= executed_usdt
            btc += executed_btc
        else:
            executed_usdt = 0.0
    elif price > 0 and action == "REDUCE_RISK" and fraction > 0 and btc > 0:
        executed_usdt = min(btc * price, before * fraction)
        if executed_usdt >= 5:
            executed_btc = -(executed_usdt / price)
            cash += executed_usdt
            btc += executed_btc
        else:
            executed_usdt = 0.0
    after = cash + btc * price if price > 0 else cash
    return {"cash_usdt": cash, "btc_amount": btc, "executed_usdt": executed_usdt, "executed_btc": executed_btc, "portfolio_value_before": before, "portfolio_value_after": after}


def append_log(row: dict[str, Any]) -> None:
    PAPER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_at", "state_date", "execution_price_source", "execution_price_usd", "snapshot_price_usd", "model_price_usd", "deviation_pct", "expected_7d_return_pct", "expected_30d_return_pct", "historical_drawdown_risk_pct", "memory_confidence_pct", "decision_edge_pct", "structural_regime", "market_regime_code", "action", "confidence", "executed_usdt", "executed_btc", "cash_usdt", "btc_amount", "btc_exposure_pct", "portfolio_value_usdt", "reason"]
    exists = PAPER_LOG_PATH.exists()
    with PAPER_LOG_PATH.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"Missing {STATE_PATH}")
    state = load_json(STATE_PATH, {})
    paper = load_json(PAPER_STATE_PATH, initial_paper_state()) or initial_paper_state()
    live_price, live_source, live_status = fetch_live_btc_price()
    action, confidence, reasons, fraction, snapshot, edge = decide(state, paper, live_price, live_source, live_status)
    price = snapshot["price"]["execution_price_usd"]
    execution = apply_action(paper, price, action, fraction)
    run_at = now_iso()
    after = execution["portfolio_value_after"]
    exposure = execution["btc_amount"] * price / after if after > 0 and price > 0 else 0.0

    updated = {**paper, "mode": "paper", "cash_usdt": round(execution["cash_usdt"], 8), "btc_amount": round(execution["btc_amount"], 12), "portfolio_value_usdt": round(after, 8), "btc_exposure_pct": round(exposure * 100, 4), "last_action": action, "last_confidence": confidence, "last_reason": reasons, "last_run_at": run_at, "last_execution_price_usd": price, "last_execution_price_source": snapshot["price"]["execution_price_source"], "last_snapshot_price_usd": snapshot["price"]["snapshot_price_usd"], "last_model_price_usd": snapshot["price"]["cohesive_fair_price_usd"], "last_deviation_pct": round(snapshot["price"]["cohesive_deviation_pct"], 4), "last_live_price_status": live_status, "last_memory_decision": edge, "decision_snapshot": {**snapshot, "memory_weighted_decision": edge}, "not_trading_advice": True, "rules": {"logic": "CohesivX memory-weighted paper trading v0.3. Historical memory distribution drives decision; fixed thresholds are safety rails only. Live price is mandatory for any paper execution.", "live_price_sources": ["binance", "coinbase", "kraken"], "max_entry_fraction": MAX_ENTRY_FRACTION, "max_btc_exposure_fraction": MAX_BTC_EXPOSURE_FRACTION, "min_entry_fraction": MIN_ENTRY_FRACTION, "requires_live_price_for_execution": True}}
    save_json(PAPER_STATE_PATH, updated)
    decision_doc = {"run_at": run_at, "action": action, "confidence": confidence, "position_fraction": fraction, "execution": execution, "live_price_status": live_status, "memory_weighted_decision": edge, "snapshot": snapshot, "reason": reasons, "not_trading_advice": True}
    save_json(DECISION_PATH, decision_doc)
    row = {"run_at": run_at, "state_date": snapshot["state_date"], "execution_price_source": snapshot["price"]["execution_price_source"], "execution_price_usd": round(price, 8), "snapshot_price_usd": snapshot["price"]["snapshot_price_usd"], "model_price_usd": snapshot["price"]["cohesive_fair_price_usd"], "deviation_pct": round(snapshot["price"]["cohesive_deviation_pct"], 4), "expected_7d_return_pct": round(f(edge.get("expected_7d_return")) * 100, 4), "expected_30d_return_pct": round(f(edge.get("expected_30d_return")) * 100, 4), "historical_drawdown_risk_pct": round(f(edge.get("historical_drawdown_risk")) * 100, 4), "memory_confidence_pct": round(f(edge.get("memory_confidence")) * 100, 4), "decision_edge_pct": round(f(edge.get("decision_edge")) * 100, 4), "structural_regime": snapshot["regime"]["structural_code"], "market_regime_code": snapshot["regime"]["market_code"], "action": action, "confidence": confidence, "executed_usdt": round(execution["executed_usdt"], 8), "executed_btc": round(execution["executed_btc"], 12), "cash_usdt": round(updated["cash_usdt"], 8), "btc_amount": round(updated["btc_amount"], 12), "btc_exposure_pct": round(updated["btc_exposure_pct"], 4), "portfolio_value_usdt": round(updated["portfolio_value_usdt"], 8), "reason": " | ".join(reasons)}
    append_log(row)
    print(f"Paper trader action: {action}")
    print(f"Confidence: {confidence}")
    print(f"Live source: {live_source}")
    print(f"Live price status: {json.dumps(live_status, ensure_ascii=False)}")
    print(f"Memory edge: {row['decision_edge_pct']}%")
    print(f"Expected 7d/30d: {row['expected_7d_return_pct']}% / {row['expected_30d_return_pct']}%")
    print(f"Executed USDT: {row['executed_usdt']}")
    print(f"Portfolio value: {row['portfolio_value_usdt']} USDT")


if __name__ == "__main__":
    main()
