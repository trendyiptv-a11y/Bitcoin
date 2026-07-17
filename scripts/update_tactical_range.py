#!/usr/bin/env python3
"""Generate btc-swing-strategy/tactical_range.json.

Tactical range is a backend-only layer. The frontend should only display
BUY / SELL / WAIT and a short message. This script detects the recent BTCUSDT
range from market candles and writes a compact JSON payload.

Default data source: Binance public spot klines.
No trading action is executed. This is a structural/tactical monitor signal,
not financial advice.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = ROOT / "btc-swing-strategy" / "coeziv_state.json"
DEFAULT_OUTPUT_PATH = ROOT / "btc-swing-strategy" / "tactical_range.json"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fetch_klines(symbol: str, interval: str, limit: int, timeout: int = 20) -> List[Dict[str, float]]:
    url = f"{BINANCE_KLINES_URL}?symbol={symbol}&interval={interval}&limit={limit}"
    with urlopen(url, timeout=timeout) as response:
        raw = json.loads(response.read().decode("utf-8"))

    candles: List[Dict[str, float]] = []
    for row in raw:
        # Binance kline format:
        # [open_time, open, high, low, close, volume, close_time, ...]
        candles.append({
            "open_time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "close_time": int(row[6]),
        })
    return candles


def safe_slice(candles: List[Dict[str, float]], count: int, offset_from_end: int = 0) -> List[Dict[str, float]]:
    if count <= 0:
        return []
    end = len(candles) - offset_from_end if offset_from_end else len(candles)
    start = max(0, end - count)
    return candles[start:end]


def window_stats(candles: Iterable[Dict[str, float]]) -> Dict[str, Optional[float]]:
    rows = list(candles)
    if not rows:
        return {"low": None, "high": None, "width": None, "mid": None, "volume_avg": None}
    low = min(row["low"] for row in rows)
    high = max(row["high"] for row in rows)
    width = high - low
    mid = (high + low) / 2.0
    vols = [row["volume"] for row in rows if math.isfinite(row.get("volume", float("nan")))]
    return {
        "low": low,
        "high": high,
        "width": width,
        "mid": mid,
        "volume_avg": statistics.fmean(vols) if vols else None,
    }


def pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old in (None, 0):
        return None
    return (new - old) / old


def classify_range_state(current: Dict[str, Optional[float]], previous: Dict[str, Optional[float]], price: float) -> str:
    low = current.get("low")
    high = current.get("high")
    width = current.get("width")
    prev_low = previous.get("low")
    prev_high = previous.get("high")
    prev_width = previous.get("width")

    if low is None or high is None or width is None or width <= 0:
        return "INVALID"
    if price > high:
        return "BROKEN_UP"
    if price < low:
        return "BROKEN_DOWN"
    if prev_low is None or prev_high is None or prev_width is None or prev_width <= 0:
        return "STABLE"

    move_threshold = max(width * 0.08, price * 0.0025)
    width_change = pct_change(width, prev_width) or 0.0

    high_up = (high - prev_high) > move_threshold
    low_up = (low - prev_low) > move_threshold
    high_down = (prev_high - high) > move_threshold
    low_down = (prev_low - low) > move_threshold

    if high_up and low_up:
        return "SHIFTING_UP"
    if high_down and low_down:
        return "SHIFTING_DOWN"
    if width_change >= 0.18:
        return "EXPANDING"
    if width_change <= -0.18:
        return "COMPRESSING"
    return "STABLE"


def structural_signal(state: Dict[str, Any]) -> str:
    raw = str(state.get("signal") or state.get("current_signal") or "flat").lower().strip()
    if raw in {"long", "buy"}:
        return "LONG"
    if raw in {"short", "sell"}:
        return "SHORT"
    return "WAIT"


def decide_tactical_signal(position: Optional[float], range_state: str, structural: str) -> Tuple[str, str, str]:
    """Return (signal, confidence, reason_code)."""
    if structural != "WAIT":
        return "WAIT", "low", "STRUCTURAL_NOT_WAIT"
    if position is None or not math.isfinite(position):
        return "WAIT", "low", "INVALID_RANGE_POSITION"
    if range_state in {"INVALID", "BROKEN_UP", "BROKEN_DOWN"}:
        return "WAIT", "low", range_state

    if position <= 0.25:
        if range_state == "SHIFTING_DOWN":
            return "WAIT", "medium", "LOW_EDGE_BUT_RANGE_SHIFTING_DOWN"
        return "BUY", "medium" if range_state in {"STABLE", "COMPRESSING"} else "low", "LOW_EDGE"

    if position >= 0.75:
        if range_state == "SHIFTING_UP":
            return "WAIT", "medium", "HIGH_EDGE_BUT_RANGE_SHIFTING_UP"
        return "SELL", "medium" if range_state in {"STABLE", "COMPRESSING"} else "low", "HIGH_EDGE"

    return "WAIT", "medium", "MID_RANGE"


def message_ro(signal: str, reason: str) -> str:
    if signal == "BUY":
        return "Cumpărare tactică: prețul este aproape de baza range-ului, iar structural rămâne WAIT."
    if signal == "SELL":
        return "Vânzare tactică: prețul este aproape de vârful range-ului, iar structural rămâne WAIT."
    if reason == "LOW_EDGE_BUT_RANGE_SHIFTING_DOWN":
        return "Așteaptă: prețul este jos, dar range-ul se mută în jos."
    if reason == "HIGH_EDGE_BUT_RANGE_SHIFTING_UP":
        return "Așteaptă: prețul este sus, dar range-ul se mută în sus."
    if reason in {"BROKEN_UP", "BROKEN_DOWN"}:
        return "Așteaptă: range-ul este rupt, semnalul tactic se resetează."
    if reason == "STRUCTURAL_NOT_WAIT":
        return "Așteaptă: semnalul structural are prioritate față de range-ul tactic."
    return "Așteaptă: prețul este în zona mediană sau range-ul nu oferă avantaj tactic."


def message_en(signal: str, reason: str) -> str:
    if signal == "BUY":
        return "Tactical buy: price is near the lower range edge while the structural layer remains WAIT."
    if signal == "SELL":
        return "Tactical sell: price is near the upper range edge while the structural layer remains WAIT."
    if reason == "LOW_EDGE_BUT_RANGE_SHIFTING_DOWN":
        return "Wait: price is low, but the range is shifting down."
    if reason == "HIGH_EDGE_BUT_RANGE_SHIFTING_UP":
        return "Wait: price is high, but the range is shifting up."
    if reason in {"BROKEN_UP", "BROKEN_DOWN"}:
        return "Wait: the range has broken, so the tactical signal resets."
    if reason == "STRUCTURAL_NOT_WAIT":
        return "Wait: the structural signal has priority over tactical range."
    return "Wait: price is in the middle zone or the range offers no tactical edge."


def build_payload(candles: List[Dict[str, float]], state: Dict[str, Any], window_hours: int) -> Dict[str, Any]:
    if len(candles) < max(12, window_hours):
        return {
            "schema_version": 1,
            "updated_at": now_iso(),
            "data_status": "insufficient_candles",
            "tactical_signal": "WAIT",
            "confidence": "low",
            "message_ro": "Așteaptă: nu sunt suficiente lumânări pentru Tactical Range.",
            "message_en": "Wait: not enough candles for Tactical Range.",
        }

    current_window = safe_slice(candles, window_hours)
    previous_window = safe_slice(candles, window_hours, offset_from_end=window_hours)
    current = window_stats(current_window)
    previous = window_stats(previous_window)
    price = candles[-1]["close"]

    low = current.get("low")
    high = current.get("high")
    width = current.get("width")
    position: Optional[float] = None
    if low is not None and high is not None and width and width > 0:
        position = (price - low) / width

    range_state = classify_range_state(current, previous, price)
    structural = structural_signal(state)
    tactical, confidence, reason = decide_tactical_signal(position, range_state, structural)

    return {
        "schema_version": 1,
        "updated_at": now_iso(),
        "data_status": "live",
        "symbol": "BTCUSDT",
        "source": "binance_spot_klines",
        "timeframe": "1h",
        "window_hours": window_hours,
        "range_low": round(low, 2) if low is not None else None,
        "range_high": round(high, 2) if high is not None else None,
        "range_width": round(width, 2) if width is not None else None,
        "current_price": round(price, 2),
        "position": round(position, 4) if position is not None else None,
        "range_state": range_state,
        "structural_signal": structural,
        "tactical_signal": tactical,
        "confidence": confidence,
        "reason_code": reason,
        "message_ro": message_ro(tactical, reason),
        "message_en": message_en(tactical, reason),
        "frontend_hint": {
            "show_only": ["tactical_signal", "message_ro", "message_en"],
            "labels": {"BUY": "BUY", "SELL": "SELL", "WAIT": "WAIT"},
        },
        "debug": {
            "previous_range_low": round(previous["low"], 2) if previous.get("low") is not None else None,
            "previous_range_high": round(previous["high"], 2) if previous.get("high") is not None else None,
            "previous_range_width": round(previous["width"], 2) if previous.get("width") is not None else None,
            "candles_used": len(candles),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CohesivX BTC Tactical Range JSON")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--limit", type=int, default=168, help="Number of candles to fetch")
    parser.add_argument("--window-hours", type=int, default=72, help="Current range window in 1h candles")
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    state = load_json(args.state_path)
    try:
        candles = fetch_klines(args.symbol, args.interval, args.limit)
        payload = build_payload(candles, state, args.window_hours)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "schema_version": 1,
            "updated_at": now_iso(),
            "data_status": "error",
            "tactical_signal": "WAIT",
            "confidence": "low",
            "message_ro": "Așteaptă: Tactical Range nu a putut fi actualizat momentan.",
            "message_en": "Wait: Tactical Range could not be updated right now.",
            "error": str(exc),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "signal": payload.get("tactical_signal"), "status": payload.get("data_status")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
