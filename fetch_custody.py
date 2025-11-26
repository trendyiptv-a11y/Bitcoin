#!/usr/bin/env python3
"""
fetch_custody.py

Ia date reale de la Glassnode (sau alt provider) și calculează p_cust real:
    p_cust = BTC_on_exchanges / BTC_circulating

Scrie:
    data/custody_snapshot.csv
    (un singur rând, cu timestamp și p_cust)

Ai nevoie de:
    - variabila de mediu GLASSNODE_API_KEY
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List

import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

GLASSNODE_API = "https://api.glassnode.com/v1/metrics"
API_KEY = os.environ.get("GLASSNODE_API_KEY")


def safe_get(url: str, params: dict) -> Any:
    print(f"[INFO] GET {url}")
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        return None


def latest_value_from_series(series: List[dict]) -> float:
    if not series:
        return 0.0
    # sortăm după t (timestamp)
    series_sorted = sorted(series, key=lambda x: x.get("t", 0))
    return float(series_sorted[-1].get("v", 0.0))


def fetch_custody_glassnode() -> None:
    if not API_KEY:
        raise SystemExit("GLASSNODE_API_KEY nu este setat.")

    # vom cere date zilnice pe ultimul an, doar ca să fim siguri că avem valori
    since = int((datetime.now(timezone.utc) - timedelta(days=365)).timestamp())

    common_params = {
        "api_key": API_KEY,
        "a": "BTC",
        "s": since,
        "i": "24h",
    }

    # 1) BTC pe exchange-uri
    # metrica clasică: balance on exchanges
    url_ex = f"{GLASSNODE_API}/distribution/balance_exchanges"
    ex_data = safe_get(url_ex, common_params)
    if ex_data is None:
        raise SystemExit("Nu am reușit să iau balance_exchanges de la Glassnode.")

    btc_on_exchanges = latest_value_from_series(ex_data)

    # 2) BTC supply circulant
    url_supply = f"{GLASSNODE_API}/supply/current"
    sup_data = safe_get(url_supply, common_params)
    if sup_data is None:
        raise SystemExit("Nu am reușit să iau supply/current de la Glassnode.")

    btc_supply = latest_value_from_series(sup_data)

    if btc_supply <= 0:
        raise SystemExit("btc_supply <= 0, ceva nu e ok cu datele.")

    p_cust = btc_on_exchanges / btc_supply

    out_path = DATA_DIR / "custody_snapshot.csv"
    print(f"[INFO] Writing custody snapshot -> {out_path}")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "btc_on_exchanges",
                "btc_supply",
                "p_cust",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                + "Z",
                "btc_on_exchanges": btc_on_exchanges,
                "btc_supply": btc_supply,
                "p_cust": p_cust,
            }
        )
    print(f"[INFO] p_cust (BTC pe exchange-uri / supply) = {p_cust:.4f}")


def main() -> None:
    fetch_custody_glassnode()


if __name__ == "__main__":
    main()
