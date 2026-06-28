#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "btc-swing-strategy"

REAL_EXECUTOR = OUT_DIR / "bitget_micro_live_real_executor_report.json"
AUDIT = OUT_DIR / "paper_decision_audit.json"
SAFE = OUT_DIR / "bitget_safe_status.json"

PUBLIC_STATUS = ROOT / "cohesivx-public-status.json"


def load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def extract_price(safe):
    try:
        data = safe.get("public_ticker", {}).get("raw", {}).get("data")
        if isinstance(data, list) and data:
            return float(data[0].get("lastPr", 0))
        if isinstance(data, dict):
            return float(data.get("lastPr", 0))
    except Exception:
        return 0
    return 0


def extract_asset(safe, coin):
    try:
        data = safe.get("account_assets", {}).get("raw", {}).get("data", [])
        for item in data:
            if str(item.get("coin", "")).upper() == coin.upper():
                return float(item.get("available", 0))
    except Exception:
        return 0
    return 0


def main():
    real = load_json(REAL_EXECUTOR)
    audit = load_json(AUDIT)
    safe = load_json(SAFE)

    checks = real.get("checks", {})

    public_status = {
        "bot": "CohesivX Micro-Live",
        "symbol": real.get("symbol", "BTCUSDT"),
        "mode": "PUBLIC_STATUS_SAFE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),

        "live_mode": "ARMED" if checks.get("live_enabled") and checks.get("live_confirm") else "NOT_ARMED",
        "action": real.get("action"),
        "result": real.get("result"),
        "status": real.get("status"),
        "real_order_sent": bool(real.get("real_order_sent")),

        "audit": audit.get("status") or checks.get("audit_status"),
        "bitget_account_available": bool(checks.get("bitget_account_available")),
        "last_price": checks.get("last_price") or extract_price(safe),

        "available_usdt": checks.get("available_usdt", extract_asset(safe, "USDT")),
        "available_btc": checks.get("available_btc", extract_asset(safe, "BTC")),

        "cooldown_seconds": checks.get("last_order_age_seconds"),
        "min_seconds_between_real_orders": checks.get("min_seconds_between_real_orders"),

        "max_usdt": real.get("max_usdt"),

        "security": {
            "api_keys_exposed": False,
            "secret_exposed": False,
            "passphrase_exposed": False,
            "signature_exposed": False,
            "raw_headers_exposed": False
        }
    }

    PUBLIC_STATUS.write_text(json.dumps(public_status, indent=2, sort_keys=True))
    print(json.dumps(public_status, indent=2))


if __name__ == "__main__":
    main()
