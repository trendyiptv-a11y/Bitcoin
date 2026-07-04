#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "btc-swing-strategy"
DECISION = OUT_DIR / "paper_trader_decision.json"
SAFE = OUT_DIR / "bitget_safe_status.json"
AUDIT = OUT_DIR / "paper_decision_audit.json"

ALLOWED = {
    "HOLD", "OBSERVE", "HOLD_CASH_CORRIDOR",
    "OBSERVE_STALE_DATA", "OBSERVE_LIVE_PRICE_UNAVAILABLE",
    "ACCUMULATE_SMALL", "OBSERVE_ACCUMULATE_SMALL",
    "REDUCE_RISK", "TAKE_PROFIT_SMALL", "TAKE_PROFIT_MEDIUM",
}

def load(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def now():
    return datetime.now(timezone.utc).isoformat()

def main():
    decision = load(DECISION)
    safe = load(SAFE)

    action = decision.get("action")
    reasons = []

    public_ok = bool(safe.get("public_ticker", {}).get("available"))
    account_ok = bool(
        safe.get("account_assets", {}).get("available")
        or safe.get("account_assets", {}).get("ok")
    )

    status = "PASS"

    if not action:
        status = "BLOCKED"
        reasons.append("No action in decision file.")

    if action and action not in ALLOWED:
        status = "BLOCKED"
        reasons.append(f"Unsupported action: {action}")

    if not public_ok:
        status = "BLOCKED"
        reasons.append("Bitget public ticker unavailable.")

    if not account_ok:
        status = "BLOCKED"
        reasons.append("Bitget account assets unavailable.")

    out = {
        "timestamp_utc": now(),
        "status": status,
        "action": action,
        "decision_file": str(DECISION),
        "safe_file": str(SAFE),
        "checks": {
            "action_allowed": action in ALLOWED,
            "public_ticker_available": public_ok,
            "account_assets_available": account_ok,
        },
        "reasons": reasons,
        "not_trading_advice": True,
    }

    AUDIT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
