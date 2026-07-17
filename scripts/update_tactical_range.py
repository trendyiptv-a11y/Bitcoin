#!/usr/bin/env python3
"""Generate btc-swing-strategy/tactical_range.json.

Tactical range is a backend-only layer. The frontend should only display
BUY / SELL / WAIT and a short message. This script detects the recent BTCUSDT
range from market candles and writes a compact JSON payload.

Data source order:
1) Bitget public spot candles
2) Kraken public OHLC
3) Binance public spot klines

No trading action is executed. This is a structural/tactical monitor signal,
not financial advice.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = ROOT / "btc-swing-strategy" / "coeziv_state.json"
DEFAULT_OUTPUT_PATH = ROOT / "btc-swing-strategy" / "tactical_range.json"

BITGET_CANDLES_URL = "https://api.bitget.com/api/v2/spot/market/candles"
KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fetch_json(url: str, timeout: int = 20) -> Any:
    req = Request(
        url,
        headers={
            "User-Agent": "CohesivX-TacticalRange/1.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_candle(open_time: Any, open_: Any, high: Any, low: Any, close: Any, volume: Any = 0) -> Dict[str, float]:
    ts = int(float(open_time))
    if ts < 10_000_000_000:
        ts *= 1000
    return {
        "open_time": ts,
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume or 0),
    }


def sorted_candles(candles: List[Dict[str, float]], limit: int) -> List[Dict[str, float]]:
    rows = sorted(candles, key=lambda x: x["open_time"])
    return rows[-limit:]


def fetch_bitget_candles(symbol: str, interval: str, limit: int, timeout: int = 20) -> Tuple[List[Dict[str, float]], str]:
    granularity = {"1h": "1h", "1H": "1h", "60m": "1h"}.get(interval, interval)
    url = f"{BITGET_CANDLES_URL}?symbol={symbol}&granularity={granularity}&limit={limit}"
    raw = fetch_json(url, timeout=timeout)
    data = raw.get("data") if isinstance(raw, dict) else raw
    if not isinstance(data, list) or not data:
        raise ValueError("Bitget returned no candle data")

    candles: List[Dict[str, float]] = []
    for row in data:
        if not isinstance(row, list) or len(row) < 6:
            continue
        # Bitget spot candles: [timestamp_ms, open, high, low, close, volume, ...]
        candles.append(normalize_candle(row[0], row[1], row[2], row[3], row[4], row[5]))

    if len(candles) < max(12, min(72, limit // 2)):
        raise ValueError(f"Bitget returned insufficient candles: {len(candles)}")
    return sorted_candles(candles, limit), "bitget_spot_candles"


def fetch_kraken_candles(symbol: str, interval: str, limit: int, timeout: int = 20) -> Tuple[List[Dict[str, float]], str]:
    pair = "XBTUSDT" if symbol.upper() in {"BTCUSDT", "BTC/USDT"} else symbol.upper()
    interval_minutes = {"1h": 60, "60m": 60, "1H": 60}.get(interval, 60)
    url = f"{KRAKEN_OHLC_URL}?pair={pair}&interval={interval_minutes}"
    raw = fetch_json(url, timeout=timeout)
    if not isinstance(raw, dict):
        raise ValueError("Kraken returned invalid payload")
    if raw.get("error"):
        raise ValueError("Kraken error: " + "; ".join(map(str, raw.get("error", []))))
    result = raw.get("result") or {}
    rows = None
    for key, value in result.items():
        if key != "last" and isinstance(value, list):
            rows = value
            break
    if not rows:
        raise ValueError("Kraken returned no OHLC rows")

    candles: List[Dict[str, float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 7:
            continue
        # Kraken OHLC: [time_s, open, high, low, close, vwap, volume, count]
        candles.append(normalize_candle(row[0], row[1], row[2], row[3], row[4], row[6]))

    if len(candles) < max(12, min(72, limit // 2)):
        raise ValueError(f"Kraken returned insufficient candles: {len(candles)}")
    return sorted_candles(candles, limit), "kraken_ohlc"


def fetch_binance_candles(symbol: str, interval: str, limit: int, timeout: int = 20) -> Tuple[List[Dict[str, float]], str]:
    url = f"{BINANCE_KLINES_URL}?symbol={symbol}&interval={interval}&limit={limit}"
    raw = fetch_json(url, timeout=timeout)
    if not isinstance(raw, list):
        raise ValueError("Binance returned invalid payload")

    candles: List[Dict[str, float]] = []
    for row in raw:
        if not isinstance(row, list) or len(row) < 7:
            continue
        # Binance kline: [open_time_ms, open, high, low, close, volume, close_time, ...]
        candles.append(normalize_candle(row[0], row[1], row[2], row[3], row[4], row[5]))

    if len(candles) < max(12, min(72, limit // 2)):
        raise ValueError(f"Binance returned insufficient candles: {len(candles)}")
    return sorted_candles(candles, limit), "binance_spot_klines"


def fetch_candles(symbol: str, interval: str, limit: int) -> Tuple[List[Dict[str, float]], str, List[str]]:
    errors: List[str] = []
    for name, fn in (
        ("bitget", fetch_bitget_candles),
        ("kraken", fetch_kraken_candles),
        ("binance", fetch_binance_candles),
    ):
        try:
            candles, source = fn(symbol, interval, limit)
            return candles, source, errors
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("All candle sources failed: " + " | ".join(errors))


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
    vols = [row["volume"] for row in rows if math.isfinite(row.get("volume", float("nan")))]
    return {
        "low": low,
        "high": high,
        "width": width,
        "mid": (high + low) / 2.0,
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


def build_payload(candles: List[Dict[str, float]], state: Dict[str, Any], window_hours: int, source: str, source_errors: List[str]) -> Dict[str, Any]:
    if len(candles) < max(12, window_hours):
        return {
            "schema_version": 1,
            "updated_at": now_iso(),
            "data_status": "insufficient_candles",
            "source": source,
            "tactical_signal": "WAIT",
            "confidence": "low",
            "message_ro": "Așteaptă: nu sunt suficiente lumânări pentru Tactical Range.",
            "message_en": "Wait: not enough candles for Tactical Range.",
            "debug": {"candles_used": len(candles), "source_errors": source_errors},
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
        "source": source,
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
            "source_errors": source_errors,
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
        candles, source, source_errors = fetch_candles(args.symbol, args.interval, args.limit)
        payload = build_payload(candles, state, args.window_hours, source, source_errors)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
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
    print(json.dumps({"output": str(args.output), "signal": payload.get("tactical_signal"), "status": payload.get("data_status"), "source": payload.get("source")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
