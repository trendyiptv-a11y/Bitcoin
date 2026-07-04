#!/usr/bin/env python3
import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "btc-swing-strategy"

ENV_PATH = ROOT / ".env"
DECISION_PATH = OUT_DIR / "paper_trader_decision.json"
AUDIT_PATH = OUT_DIR / "paper_decision_audit.json"
BITGET_SAFE_PATH = OUT_DIR / "bitget_safe_status.json"

REPORT_PATH = OUT_DIR / "bitget_micro_live_real_executor_report.json"
TRADE_LOG_PATH = OUT_DIR / "bitget_micro_live_real_trade_log.jsonl"
LOCK_PATH = OUT_DIR / "bitget_micro_live_last_order.json"

BASE_URL = "https://api.bitget.com"
ORDER_PATH = "/api/v2/spot/trade/place-order"
METHOD = "POST"

SYMBOL = "BTCUSDT"

LIVE_ARM_VALUE = "YES_I_ACCEPT_REAL_ORDERS"
LIVE_CONFIRM_VALUE = "BTCUSDT_SPOT_5USDT_ONLY"

DEFAULT_MAX_USDT = 5.0
MIN_SECONDS_BETWEEN_REAL_ORDERS = 14 * 60  # aproape 15 minute

HOLD_ACTIONS = {
    "HOLD",
    "OBSERVE",
    "HOLD_CASH_CORRIDOR",
    "OBSERVE_STALE_DATA",
    "OBSERVE_LIVE_PRICE_UNAVAILABLE",
}

BUY_ACTIONS = {
    "ACCUMULATE_SMALL",
    "OBSERVE_ACCUMULATE_SMALL",
}

SELL_ACTIONS = {
    "REDUCE_RISK",
    "TAKE_PROFIT_SMALL",
    "TAKE_PROFIT_MEDIUM",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_local_env():
    if not ENV_PATH.exists():
        return

    for raw in ENV_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return {"_load_error": str(e)}


def write_json(path, data):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def extract_bitget_response_fields(http_result):
    response = {}
    if isinstance(http_result, dict):
        response = http_result.get("response") or {}

    data = response.get("data") if isinstance(response, dict) else {}
    if not isinstance(data, dict):
        data = {}

    return {
        "orderId": data.get("orderId"),
        "clientOid": data.get("clientOid"),
        "result_code": response.get("code") if isinstance(response, dict) else None,
        "result_msg": response.get("msg") if isinstance(response, dict) else None,
        "requestTime": response.get("requestTime") if isinstance(response, dict) else None,
    }


def enrich_order_record(report):
    """Add stable top-level execution fields to reports/log entries.

    Older logs kept last_price and balances only under checks, which made
    cost-basis reconstruction difficult. This keeps the full report intact
    while also exposing the fields directly in the JSONL trade log and the
    last-order lock file.
    """
    if not isinstance(report, dict):
        return report

    data = dict(report)
    checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    body = data.get("prepared_body") if isinstance(data.get("prepared_body"), dict) else {}
    response_fields = extract_bitget_response_fields(data.get("http_result"))

    data.setdefault("epoch_seconds", time.time())
    data.setdefault("last_price", checks.get("last_price"))
    data.setdefault("available_btc_before", checks.get("available_btc"))
    data.setdefault("available_usdt_before", checks.get("available_usdt"))
    data.setdefault("last_order_age_seconds", checks.get("last_order_age_seconds"))
    data.setdefault("order_type", body.get("orderType"))
    data.setdefault("order_size", body.get("size"))
    data.setdefault("clientOid", body.get("clientOid") or response_fields.get("clientOid"))
    data.setdefault("orderId", response_fields.get("orderId"))
    data.setdefault("result_code", response_fields.get("result_code"))
    data.setdefault("result_msg", response_fields.get("result_msg"))
    data.setdefault("requestTime", response_fields.get("requestTime"))

    return data


def append_trade_log(data):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enriched = enrich_order_record(data)
    with TRADE_LOG_PATH.open("a") as f:
        f.write(json.dumps(enriched, sort_keys=True) + "\n")


def sign_bitget(timestamp_ms, method, request_path, body_string, secret):
    prehash = f"{timestamp_ms}{method.upper()}{request_path}{body_string}"
    digest = hmac.new(
        secret.encode("utf-8"),
        prehash.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def extract_last_price(bitget_safe):
    try:
        raw = bitget_safe.get("public_ticker", {}).get("raw", {})
        data = raw.get("data")
        if isinstance(data, list) and data:
            return float(data[0].get("lastPr"))
        if isinstance(data, dict):
            return float(data.get("lastPr"))
    except Exception:
        return None
    return None


def extract_available_asset(bitget_safe, coin):
    try:
        raw = bitget_safe.get("account_assets", {}).get("raw", {})
        data = raw.get("data", [])
        for item in data:
            if str(item.get("coin", "")).upper() == coin.upper():
                return float(item.get("available", 0))
    except Exception:
        return 0.0
    return 0.0


def seconds_since_last_order():
    last = load_json(LOCK_PATH)
    if not last:
        return None

    try:
        ts = float(last.get("epoch_seconds", 0))
        return time.time() - ts
    except Exception:
        return None


def mark_last_order(report):
    enriched = enrich_order_record(report)
    data = {
        "timestamp_utc": enriched.get("timestamp_utc") or now_iso(),
        "epoch_seconds": enriched.get("epoch_seconds") or time.time(),
        "symbol": SYMBOL,
        "side": enriched.get("side"),
        "action": enriched.get("action"),
        "result": enriched.get("result"),
        "status": enriched.get("status"),
        "clientOid": enriched.get("clientOid"),
        "orderId": enriched.get("orderId"),
        "result_code": enriched.get("result_code"),
        "result_msg": enriched.get("result_msg"),
        "requestTime": enriched.get("requestTime"),
        "last_price": enriched.get("last_price"),
        "max_usdt": enriched.get("max_usdt"),
        "available_btc_before": enriched.get("available_btc_before"),
        "available_usdt_before": enriched.get("available_usdt_before"),
        "last_order_age_seconds": enriched.get("last_order_age_seconds"),
        "order_type": enriched.get("order_type"),
        "order_size": enriched.get("order_size"),
        "prepared_body": enriched.get("prepared_body"),
    }
    write_json(LOCK_PATH, data)


def make_market_buy_body(max_usdt):
    return {
        "symbol": SYMBOL,
        "side": "buy",
        "orderType": "market",
        "size": f"{max_usdt:.2f}",
        "clientOid": f"cohesivx-live-buy-{int(time.time())}",
    }


def make_market_sell_body(bitget_safe, max_usdt):
    last_price = extract_last_price(bitget_safe)
    available_btc = extract_available_asset(bitget_safe, "BTC")

    if not last_price or last_price <= 0:
        return None, "NO_VALID_LAST_PRICE"

    if available_btc <= 0:
        return None, "NO_BTC_AVAILABLE"

    btc_equivalent = max_usdt / last_price
    sell_btc = min(available_btc, btc_equivalent)

    if sell_btc <= 0:
        return None, "SELL_AMOUNT_ZERO"

    # Bitget BTCUSDT Spot accepts BTC size scale max 6 decimals.
    # Use ROUND_DOWN so we never sell more BTC than calculated/available.
    sell_btc_dec = Decimal(str(sell_btc)).quantize(Decimal("0.000001"), rounding=ROUND_DOWN)

    if sell_btc_dec <= 0:
        return None, "SELL_AMOUNT_BELOW_BITGET_SCALE"

    sell_btc_str = format(sell_btc_dec, "f").rstrip("0").rstrip(".")

    return {
        "symbol": SYMBOL,
        "side": "sell",
        "orderType": "market",
        "size": sell_btc_str,
        "clientOid": f"cohesivx-live-sell-{int(time.time())}",
    }, None


def post_bitget_order(api_key, api_secret, passphrase, body):
    body_string = json.dumps(body, separators=(",", ":"))
    timestamp_ms = str(int(time.time() * 1000))
    signature = sign_bitget(timestamp_ms, METHOD, ORDER_PATH, body_string, api_secret)

    req = urllib.request.Request(
        BASE_URL + ORDER_PATH,
        data=body_string.encode("utf-8"),
        method=METHOD,
        headers={
            "ACCESS-KEY": api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-PASSPHRASE": passphrase,
            "ACCESS-TIMESTAMP": timestamp_ms,
            "Content-Type": "application/json",
            "locale": "en-US",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"raw": raw}
            return {
                "http_ok": True,
                "http_status": resp.status,
                "response": parsed,
            }
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}
        return {
            "http_ok": False,
            "http_status": e.code,
            "response": parsed,
        }
    except Exception as e:
        return {
            "http_ok": False,
            "http_status": None,
            "response": {
                "error": str(e)
            },
        }


def block(report, result, reason):
    report["status"] = "BLOCKED"
    report["result"] = result
    report["reasons"].append(reason)
    write_json(REPORT_PATH, report)
    print(json.dumps(report, indent=2))
    return


def main():
    load_local_env()

    api_key = os.getenv("BITGET_API_KEY", "")
    api_secret = os.getenv("BITGET_API_SECRET", "")
    passphrase = os.getenv("BITGET_API_PASSPHRASE", "")

    live_enabled = os.getenv("COHESIVX_LIVE_ENABLED", "")
    live_confirm = os.getenv("COHESIVX_LIVE_CONFIRM", "")
    max_usdt_raw = os.getenv("COHESIVX_LIVE_MAX_USDT", str(DEFAULT_MAX_USDT))

    try:
        max_usdt = float(max_usdt_raw)
    except Exception:
        max_usdt = DEFAULT_MAX_USDT

    decision = load_json(DECISION_PATH)
    audit = load_json(AUDIT_PATH)
    bitget_safe = load_json(BITGET_SAFE_PATH)

    report = {
        "timestamp_utc": now_iso(),
        "mode": "BITGET_MICRO_LIVE_REAL_EXECUTOR",
        "symbol": SYMBOL,
        "status": "UNKNOWN",
        "result": None,
        "real_order_sent": False,
        "max_usdt": max_usdt,
        "action": None,
        "side": None,
        "prepared_body": None,
        "http_result": None,
        "checks": {},
        "reasons": [],
        "not_trading_advice": True,
    }

    if max_usdt <= 0 or max_usdt > 5.0:
        return block(report, "MAX_USDT_INVALID", "COHESIVX_LIVE_MAX_USDT must be > 0 and <= 5.")

    if not decision:
        return block(report, "NO_DECISION_FILE", "Decision file missing or unreadable.")

    if not audit:
        return block(report, "NO_AUDIT_FILE", "Audit file missing or unreadable.")

    if not bitget_safe:
        return block(report, "NO_BITGET_SAFE_FILE", "Bitget safe status file missing or unreadable.")

    action = decision.get("action")
    audit_status = audit.get("status")

    credentials_present = bool(api_key and api_secret and passphrase)
    account_available = bool(
        bitget_safe.get("account_assets", {}).get("available")
        or bitget_safe.get("account_assets", {}).get("ok")
    )

    available_usdt = extract_available_asset(bitget_safe, "USDT")
    available_btc = extract_available_asset(bitget_safe, "BTC")
    last_price = extract_last_price(bitget_safe)

    last_order_age = seconds_since_last_order()

    report["action"] = action
    report["checks"] = {
        "live_enabled": live_enabled == LIVE_ARM_VALUE,
        "live_confirm": live_confirm == LIVE_CONFIRM_VALUE,
        "required_live_enabled": LIVE_ARM_VALUE,
        "required_live_confirm": LIVE_CONFIRM_VALUE,
        "credentials_present": credentials_present,
        "audit_status": audit_status,
        "audit_pass": audit_status == "PASS",
        "bitget_account_available": account_available,
        "available_usdt": available_usdt,
        "available_btc": available_btc,
        "last_price": last_price,
        "last_order_age_seconds": last_order_age,
        "min_seconds_between_real_orders": MIN_SECONDS_BETWEEN_REAL_ORDERS,
    }

    if live_enabled != LIVE_ARM_VALUE:
        return block(report, "LIVE_NOT_ARMED", "COHESIVX_LIVE_ENABLED is not armed.")

    if live_confirm != LIVE_CONFIRM_VALUE:
        return block(report, "LIVE_CONFIRM_MISSING", "COHESIVX_LIVE_CONFIRM is missing or invalid.")

    if not credentials_present:
        return block(report, "CREDENTIALS_MISSING", "Bitget credentials missing.")

    if audit_status != "PASS":
        return block(report, "AUDIT_NOT_PASS", "Paper decision audit is not PASS.")

    if not account_available:
        return block(report, "BITGET_ACCOUNT_NOT_AVAILABLE", "Bitget account read check is not available.")

    if last_order_age is not None and last_order_age < MIN_SECONDS_BETWEEN_REAL_ORDERS:
        return block(report, "COOLDOWN_ACTIVE", "Last real order is too recent.")

    if action in HOLD_ACTIONS:
        report["status"] = "PASS"
        report["result"] = "NO_ACTION_HOLD"
        report["reasons"].append("Decision is HOLD/OBSERVE class. No real order sent.")
        write_json(REPORT_PATH, report)
        print(json.dumps(report, indent=2))
        return

    if action in BUY_ACTIONS:
        if available_usdt < max_usdt:
            return block(report, "INSUFFICIENT_USDT", "Available USDT is below max order amount.")

        body = make_market_buy_body(max_usdt)
        side = "buy"

    elif action in SELL_ACTIONS:
        body, error = make_market_sell_body(bitget_safe, max_usdt)
        if error:
            return block(report, error, "Could not build safe sell body.")
        side = "sell"

    else:
        return block(report, "UNKNOWN_ACTION", f"Unsupported action: {action}")

    report["side"] = side
    report["prepared_body"] = body
    report["reasons"].append("All gates passed. Sending one real Bitget spot market order.")

    http_result = post_bitget_order(api_key, api_secret, passphrase, body)

    report["http_result"] = http_result
    report["real_order_attempted"] = True
    report["real_order_sent"] = False
    report["real_order_confirmed"] = False

    response = http_result.get("response", {})
    success = http_result.get("http_ok") and response.get("code") == "00000"

    if success:
        report["status"] = "PASS"
        report["result"] = "REAL_ORDER_PLACED"
        report["real_order_sent"] = True
        report["real_order_confirmed"] = True
        report["reasons"].append("Bitget returned success code 00000.")
        mark_last_order(report)
    else:
        report["status"] = "ERROR"
        report["result"] = "REAL_ORDER_ATTEMPT_FAILED"
        report["real_order_sent"] = False
        report["real_order_confirmed"] = False
        report["reasons"].append("Bitget did not return success code 00000.")

    write_json(REPORT_PATH, report)
    append_trade_log(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
