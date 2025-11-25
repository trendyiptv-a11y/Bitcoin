#!/usr/bin/env python3
import csv
import time
import json
from datetime import datetime
from urllib import request
from pathlib import Path

OUT_PATH = Path("data") / "btc_daily.csv"

def fetch_batch(to_ts: int):
    url = (
        f"https://min-api.cryptocompare.com/data/v2/histoday"
        f"?fsym=BTC&tsym=USD&limit=2000&toTs={to_ts}"
    )
    with request.urlopen(url) as r:
        data = json.loads(r.read().decode())
        if data["Response"] != "Success":
            raise RuntimeError(data)
        return data["Data"]["Data"]

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Fetch CryptoCompare full history...")
    all_rows = []

    # mergem spre trecut, batch cu batch
    to_ts = int(time.time())

    while True:
        batch = fetch_batch(to_ts)
        if not batch:
            break

        print(f"[INFO] Batch: {len(batch)} zile, până la {datetime.utcfromtimestamp(batch[0]['time']).strftime('%Y-%m-%d')}")

        for k in batch:
            # dacă CryptoCompare pune 0, ignorăm ziua
            if k["open"] == 0:
                continue

            all_rows.append({
                "date": datetime.utcfromtimestamp(k["time"]).strftime("%Y-%m-%d"),
                "open": k["open"],
                "high": k["high"],
                "low": k["low"],
                "close": k["close"],
                "volume": k["volumefrom"],   # "volumefrom" = volume BTC
            })

        # pregătim următorul batch
        oldest_ts = batch[0]["time"]
        # dacă suntem înainte de 2011, oprim
        if oldest_ts < 1293840000:  # 2011-01-01 aproximativ
            break

        to_ts = oldest_ts - 86400
        time.sleep(0.2)

    # ordonăm crescător
    all_rows.sort(key=lambda r: r["date"])

    print(f"[INFO] Total zile: {len(all_rows)}")

    # scriem CSV
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"[INFO] Scris în {OUT_PATH}")

if __name__ == "__main__":
    main()
