#!/usr/bin/env python
from __future__ import annotations

import csv
import json
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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def initial_paper_state() -> dict[str, Any]:
    return {
        "mode": "paper",
        "starting_balance_usdt": STARTING_BALANCE_USDT,
        "cash_usdt": STARTING_BALANCE_USDT,
        "btc_amount": 0.0,
        "last_action": "INIT",
        "last_reason": ["Paper trader initialized."],
        "last_run_at": None,
        "not_trading_advice": True,
    }


def get_structural_confirmation(state: dict[str, Any]) -> float:
    structural = state.get("structural_confirmation") or {}
    h7 = structural.get("horizon_7d") or {}
    h30 = structural.get("horizon_30d") or {}
    rates = [as_float(h7.get("directional_hit_rate")), as_float(h30.get("directional_hit_rate"))]
    rates = [r for r in rates if r > 0]
    if not rates:
        return 0.0
    return sum(rates) / len(rates)


def decide(state: dict[str, Any], paper: dict[str, Any]) -> tuple[str, str, list[str], float]:
    price = as_float(state.get("price_usd"))
    model_price = as_float(state.get("model_price_usd"))
    deviation = as_float(state.get("model_price_deviation"))
    signal = str(state.get("signal") or "flat").lower()
    flow_bias = str(state.get("flow_bias") or "neutru").lower()
    flow_strength = str(state.get("flow_strength") or "slab").lower()
    liquidity_regime = str(state.get("liquidity_regime") or "").lower()
    liquidity_strength = str(state.get("liquidity_strength") or "").lower()
    context = state.get("model_price_context") or {}
    regime = str(context.get("regime") or "").lower()
    structural_confirmation = get_structural_confirmation(state)

    cash = as_float(paper.get("cash_usdt"))
    btc = as_float(paper.get("btc_amount"))
    portfolio_value = cash + btc * price if price > 0 else cash
    btc_exposure = (btc * price / portfolio_value) if portfolio_value > 0 and price > 0 else 0.0

    reasons: list[str] = []
    action = "OBSERVE"
    confidence = "low"
    fraction = 0.0

    if price <= 0 or model_price <= 0:
        return "OBSERVE", "low", ["Missing valid BTC price or cohesive model price."], 0.0

    if deviation <= -0.05:
        reasons.append(f"BTC is {deviation * 100:.1f}% below the cohesive reference.")
    else:
        reasons.append(f"Deviation is {deviation * 100:.1f}%, not deep enough for accumulation logic.")

    if regime in {"bear_late", "range", "neutral"}:
        reasons.append(f"Structural regime is {regime or 'unknown'}, acceptable for observation/conditional accumulation.")
    else:
        reasons.append(f"Structural regime is {regime or 'unknown'}, not ideal for accumulation.")

    if "ridic" in liquidity_regime or "strong" in liquidity_strength or "putern" in liquidity_strength:
        reasons.append("Liquidity is strong/high, reducing execution fragility in paper mode.")
    else:
        reasons.append("Liquidity is not clearly strong.")

    if structural_confirmation >= MIN_STRUCTURAL_CONFIRMATION:
        reasons.append(f"Structural confirmation is {structural_confirmation * 100:.1f}%, above the {MIN_STRUCTURAL_CONFIRMATION * 100:.0f}% threshold.")
    else:
        reasons.append("Structural confirmation is missing or below threshold.")

    if flow_bias in {"neutru", "neutral"} and flow_strength in {"slab", "weak"}:
        reasons.append("Flow is neutral/weak, so the bot avoids aggressive sizing.")
    elif "neg" in flow_bias or "bear" in flow_bias:
        reasons.append("Flow is negative, accumulation requires extra caution.")
    else:
        reasons.append(f"Flow bias is {flow_bias}, strength {flow_strength}.")

    accumulate_conditions = [
        deviation <= -0.05,
        regime in {"bear_late", "range", "neutral"},
        structural_confirmation >= MIN_STRUCTURAL_CONFIRMATION,
        btc_exposure < MAX_BTC_EXPOSURE_FRACTION,
        cash > 10.0,
    ]

    if all(accumulate_conditions):
        action = "ACCUMULATE_SMALL"
        confidence = "moderate_low"
        remaining_exposure_room = max(0.0, MAX_BTC_EXPOSURE_FRACTION - btc_exposure)
        fraction = min(MAX_ENTRY_FRACTION, remaining_exposure_room)
        reasons.append(f"Paper allocation allowed: {fraction * 100:.1f}% of current portfolio value.")
    elif btc > 0 and deviation > 0.10:
        action = "REDUCE_RISK"
        confidence = "moderate"
        fraction = min(0.10, btc_exposure)
        reasons.append("BTC is above cohesive reference by more than 10%; reduce paper exposure.")
    elif btc > 0:
        action = "HOLD"
        confidence = "low"
        reasons.append("Existing paper BTC position is maintained; no new strong action.")
    else:
        action = "OBSERVE"
        confidence = "low"
        reasons.append("No paper position opened because conditions are incomplete.")

    return action, confidence, reasons, fraction


def apply_paper_action(state: dict[str, Any], paper: dict[str, Any], action: str, fraction: float) -> dict[str, Any]:
    price = as_float(state.get("price_usd"))
    cash = as_float(paper.get("cash_usdt"))
    btc = as_float(paper.get("btc_amount"))
    portfolio_value_before = cash + btc * price if price > 0 else cash
    executed_usdt = 0.0
    executed_btc = 0.0

    if price <= 0:
        return {
            "cash_usdt": cash,
            "btc_amount": btc,
            "executed_usdt": 0.0,
            "executed_btc": 0.0,
            "portfolio_value_before": portfolio_value_before,
            "portfolio_value_after": portfolio_value_before,
        }

    if action == "ACCUMULATE_SMALL" and fraction > 0:
        target_usdt = min(cash, portfolio_value_before * fraction)
        if target_usdt >= 5.0:
            executed_usdt = target_usdt
            executed_btc = executed_usdt / price
            cash -= executed_usdt
            btc += executed_btc

    elif action == "REDUCE_RISK" and fraction > 0 and btc > 0:
        btc_value = btc * price
        target_usdt = min(btc_value, portfolio_value_before * fraction)
        if target_usdt >= 5.0:
            executed_usdt = target_usdt
            executed_btc = -(executed_usdt / price)
            cash += executed_usdt
            btc += executed_btc

    portfolio_value_after = cash + btc * price
    return {
        "cash_usdt": cash,
        "btc_amount": btc,
        "executed_usdt": executed_usdt,
        "executed_btc": executed_btc,
        "portfolio_value_before": portfolio_value_before,
        "portfolio_value_after": portfolio_value_after,
    }


def append_log(row: dict[str, Any]) -> None:
    PAPER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_at",
        "state_timestamp",
        "price_usd",
        "model_price_usd",
        "deviation_pct",
        "regime",
        "signal",
        "action",
        "confidence",
        "executed_usdt",
        "executed_btc",
        "cash_usdt",
        "btc_amount",
        "portfolio_value_usdt",
        "reason",
    ]
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
    paper = load_json(PAPER_STATE_PATH, initial_paper_state())
    if not paper:
        paper = initial_paper_state()

    action, confidence, reasons, fraction = decide(state, paper)
    execution = apply_paper_action(state, paper, action, fraction)

    price = as_float(state.get("price_usd"))
    context = state.get("model_price_context") or {}
    run_at = now_iso()
    updated = {
        **paper,
        "mode": "paper",
        "cash_usdt": round(execution["cash_usdt"], 8),
        "btc_amount": round(execution["btc_amount"], 12),
        "portfolio_value_usdt": round(execution["portfolio_value_after"], 8),
        "last_action": action,
        "last_confidence": confidence,
        "last_reason": reasons,
        "last_run_at": run_at,
        "last_price_usd": price,
        "last_model_price_usd": as_float(state.get("model_price_usd")),
        "last_deviation_pct": round(as_float(state.get("model_price_deviation")) * 100, 4),
        "not_trading_advice": True,
        "rules": {
            "starting_balance_usdt": STARTING_BALANCE_USDT,
            "max_entry_fraction": MAX_ENTRY_FRACTION,
            "max_btc_exposure_fraction": MAX_BTC_EXPOSURE_FRACTION,
            "min_structural_confirmation": MIN_STRUCTURAL_CONFIRMATION,
        },
    }
    save_json(PAPER_STATE_PATH, updated)

    row = {
        "run_at": run_at,
        "state_timestamp": state.get("timestamp"),
        "price_usd": price,
        "model_price_usd": as_float(state.get("model_price_usd")),
        "deviation_pct": round(as_float(state.get("model_price_deviation")) * 100, 4),
        "regime": context.get("regime"),
        "signal": state.get("signal"),
        "action": action,
        "confidence": confidence,
        "executed_usdt": round(execution["executed_usdt"], 8),
        "executed_btc": round(execution["executed_btc"], 12),
        "cash_usdt": round(updated["cash_usdt"], 8),
        "btc_amount": round(updated["btc_amount"], 12),
        "portfolio_value_usdt": round(updated["portfolio_value_usdt"], 8),
        "reason": " | ".join(reasons),
    }
    append_log(row)

    print(f"Paper trader action: {action}")
    print(f"Confidence: {confidence}")
    print(f"Executed USDT: {row['executed_usdt']}")
    print(f"Portfolio value: {row['portfolio_value_usdt']} USDT")


if __name__ == "__main__":
    main()
