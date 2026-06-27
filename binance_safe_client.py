#!/usr/bin/env python
from __future__ import annotations

"""
CohesivX Binance Safe Client

SAFE MODE ONLY.

This module is the first Binance integration layer. It is intentionally
read-only and cannot place orders. It can:
- fetch public BTCUSDT price;
- optionally call signed read-only account endpoints when API keys exist;
- produce a safety report for the future live-trading guard.

It must not be extended with order execution. Real order placement belongs in a
separate executor that is gated by explicit dry-run/testnet/live flags and a
hard risk guard.
"""

import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "btc-swing-strategy"
REPORT_PATH = OUT_DIR / "binance_safe_status.json"

BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
SYMBOL = os.getenv("BINANCE_SYMBOL", "BTCUSDT")
PUBLIC_TIMEOUT_SECONDS = 12
SIGNED_TIMEOUT_SECONDS = 12

# Hard safety defaults. This client never trades.
TRADING_ENABLED = False
WITHDRAWALS_ENABLED = False
ORDERS_ENABLED = False


@dataclass(frozen=True)
class BinanceCredentials:
    api_key: str | None
    api_secret: str | None

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.api_secret)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_credentials() -> BinanceCredentials:
    return BinanceCredentials(
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET"),
    )


def fetch_json(url: str, headers: dict[str, str] | None = None, timeout: int = PUBLIC_TIMEOUT_SECONDS) -> Any:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "CohesivX-Binance-Safe-Client/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def public_price(symbol: str = SYMBOL) -> dict[str, Any]:
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/price?symbol={urllib.parse.quote(symbol)}"
    payload = fetch_json(url)
    price = float(payload.get("price", 0.0))
    if price <= 0:
        raise ValueError("invalid Binance public price")
    return {"ok": True, "symbol": symbol, "price": price, "source": "binance_public_ticker"}


def sign_query(params: dict[str, Any], secret: str) -> str:
    encoded = urllib.parse.urlencode(params)
    signature = hmac.new(secret.encode("utf-8"), encoded.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{encoded}&signature={signature}"


def signed_get(path: str, params: dict[str, Any], credentials: BinanceCredentials) -> Any:
    if not credentials.available:
        raise PermissionError("missing Binance API credentials")
    assert credentials.api_key is not None
    assert credentials.api_secret is not None
    params = {**params, "timestamp": int(time.time() * 1000)}
    query = sign_query(params, credentials.api_secret)
    url = f"{BINANCE_BASE_URL}{path}?{query}"
    headers = {
        "User-Agent": "CohesivX-Binance-Safe-Client/1.0",
        "X-MBX-APIKEY": credentials.api_key,
    }
    return fetch_json(url, headers=headers, timeout=SIGNED_TIMEOUT_SECONDS)


def safe_account_snapshot(credentials: BinanceCredentials) -> dict[str, Any]:
    if not credentials.available:
        return {
            "ok": False,
            "available": False,
            "reason": "BINANCE_API_KEY/BINANCE_API_SECRET not configured. Public price check only.",
        }
    try:
        account = signed_get("/api/v3/account", {"recvWindow": 5000}, credentials)
        balances = account.get("balances") or []
        important_assets = {"BTC", "USDT", "FDUSD", "USDC", "EUR"}
        compact_balances = []
        for item in balances:
            asset = str(item.get("asset") or "")
            free = float(item.get("free") or 0.0)
            locked = float(item.get("locked") or 0.0)
            if asset in important_assets or free > 0 or locked > 0:
                compact_balances.append({"asset": asset, "free": free, "locked": locked})
        permissions = account.get("permissions") or []
        return {
            "ok": True,
            "available": True,
            "account_type": account.get("accountType"),
            "can_trade_flag_from_exchange": bool(account.get("canTrade")),
            "can_withdraw_flag_from_exchange": bool(account.get("canWithdraw")),
            "can_deposit_flag_from_exchange": bool(account.get("canDeposit")),
            "permissions": permissions,
            "balances": compact_balances,
        }
    except Exception as exc:
        return {
            "ok": False,
            "available": True,
            "error_type": exc.__class__.__name__,
            "error": str(exc)[:500],
        }


def safety_report() -> dict[str, Any]:
    credentials = load_credentials()
    report: dict[str, Any] = {
        "checked_at": now_iso(),
        "mode": "safe_read_only",
        "symbol": SYMBOL,
        "binance_base_url": BINANCE_BASE_URL,
        "trading_enabled": TRADING_ENABLED,
        "orders_enabled": ORDERS_ENABLED,
        "withdrawals_enabled": WITHDRAWALS_ENABLED,
        "credentials_present": credentials.available,
        "rules": {
            "this_client_can_place_orders": False,
            "this_client_can_withdraw": False,
            "required_next_step_before_trading": "dry_run_executor_with_live_trade_guard",
            "recommended_api_permissions": "read-only first; later spot trading only; never withdrawal",
        },
    }
    try:
        report["public_price"] = public_price(SYMBOL)
    except Exception as exc:
        report["public_price"] = {"ok": False, "error_type": exc.__class__.__name__, "error": str(exc)[:500]}
    report["account"] = safe_account_snapshot(credentials)
    return report


def save_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    report = safety_report()
    save_report(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report.get("trading_enabled") or report.get("orders_enabled") or report.get("withdrawals_enabled"):
        raise RuntimeError("Safety invariant failed: safe client must not enable trading/orders/withdrawals.")


if __name__ == "__main__":
    main()
