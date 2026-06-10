#!/usr/bin/env python3
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request, parse

OUT_PATH = Path("data") / "btc_daily.csv"
START_TS = 1293840000  # 2011-01-01 UTC aproximativ


def read_json_url(url: str, headers=None, timeout=20):
    req = request.Request(url, headers=headers or {"User-Agent": "CohesivX-BTC-Monitor/1.0"})
    with request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_cryptocompare_batch(to_ts: int):
    params = {
        "fsym": "BTC",
        "tsym": "USD",
        "limit": "2000",
        "toTs": str(to_ts),
    }
    api_key = os.getenv("CRYPTOCOMPARE_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key

    url = "https://min-api.cryptocompare.com/data/v2/histoday?" + parse.urlencode(params)
    data = read_json_url(url)
    if data.get("Response") != "Success":
        raise RuntimeError(data)
    return data["Data"]["Data"]


def fetch_from_cryptocompare():
    print("[INFO] Fetch CryptoCompare full history...")
    all_rows = []
    to_ts = int(time.time())

    while True:
        batch = fetch_cryptocompare_batch(to_ts)
        if not batch:
            break

        print(f"[INFO] CryptoCompare batch: {len(batch)} zile, până la {datetime.utcfromtimestamp(batch[0]['time']).strftime('%Y-%m-%d')}")

        for k in batch:
            if k.get("open", 0) == 0:
                continue
            all_rows.append({
                "date": datetime.utcfromtimestamp(k["time"]).strftime("%Y-%m-%d"),
                "open": float(k["open"]),
                "high": float(k["high"]),
                "low": float(k["low"]),
                "close": float(k["close"]),
                "volume": float(k.get("volumefrom", 0)),
            })

        oldest_ts = batch[0]["time"]
        if oldest_ts < START_TS:
            break

        to_ts = oldest_ts - 86400
        time.sleep(0.25)

    return normalize_rows(all_rows)


def fetch_binance_klines(end_ms: int):
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "limit": "1000",
        "endTime": str(end_ms),
    }
    url = "https://api.binance.com/api/v3/klines?" + parse.urlencode(params)
    return read_json_url(url)


def fetch_from_binance():
    print("[INFO] Fetch Binance public daily klines fallback...")
    all_rows = []
    end_ms = int(time.time() * 1000)
    start_ms = START_TS * 1000

    while True:
        batch = fetch_binance_klines(end_ms)
        if not batch:
            break

        print(f"[INFO] Binance batch: {len(batch)} zile, până la {datetime.fromtimestamp(batch[0][0] / 1000, timezone.utc).strftime('%Y-%m-%d')}")

        for k in batch:
            open_time_ms = int(k[0])
            if open_time_ms < start_ms:
                continue
            all_rows.append({
                "date": datetime.fromtimestamp(open_time_ms / 1000, timezone.utc).strftime("%Y-%m-%d"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        oldest_ms = int(batch[0][0])
        if oldest_ms <= start_ms:
            break

        end_ms = oldest_ms - 1
        time.sleep(0.25)

    return normalize_rows(all_rows)


def normalize_rows(rows):
    dedup = {}
    for row in rows:
        dedup[row["date"]] = row
    out = list(dedup.values())
    out.sort(key=lambda r: r["date"])
    return out


def write_csv(rows):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[INFO] Total zile: {len(rows)}")
    print(f"[INFO] Scris în {OUT_PATH}")


def existing_csv_is_usable():
    if not OUT_PATH.exists() or OUT_PATH.stat().st_size <= 32:
        return False
    try:
        with OUT_PATH.open("r", encoding="utf-8") as f:
            header = f.readline().strip()
        return header == "date,open,high,low,close,volume"
    except Exception:
        return False


def main():
    errors = []
    rows = []

    try:
        rows = fetch_from_cryptocompare()
    except Exception as e:
        errors.append(f"CryptoCompare failed: {e}")
        print(f"[WARN] {errors[-1]}")

    if not rows:
        try:
            rows = fetch_from_binance()
        except Exception as e:
            errors.append(f"Binance failed: {e}")
            print(f"[WARN] {errors[-1]}")

    if rows:
        write_csv(rows)
        return

    if existing_csv_is_usable():
        print("[WARN] All live sources failed. Keeping existing data/btc_daily.csv.")
        for err in errors:
            print(f"[WARN] {err}")
        return

    raise RuntimeError("All BTC daily data sources failed and no usable existing btc_daily.csv is available. " + " | ".join(errors))


if __name__ == "__main__":
    main()
