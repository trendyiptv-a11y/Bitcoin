#!/usr/bin/env python3
"""
fetch_all_data.py

Script pentru a descărca automat dataset-urile relevante pentru J_BTC
și a le salva în CSV-uri în directorul `data/`.

Ce descarcă:
- mining_pools.csv      (mempool.space / mining hashrate per pool)
- nodes_bitnodes.csv    (Bitnodes snapshot - noduri + țări + versiune)
- mempool_snapshot.csv  (mempool.space /api/mempool + /fees/recommended)
- mempool_demand.csv    (Figshare "Demand CSV from Bitcoin mempool")
- mempool_supply.csv    (Figshare "Supply CSV from Bitcoin mempool")

NOTE:
- Custody (C_custody) și Governance (C_gov) nu au surse publice simple CSV;
  scriptul doar lasă hooks / TODO-uri în acest moment.
"""

import os
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any
import requests


# -------------------------
# Utilități generale
# -------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def safe_get(url: str, timeout: int = 20) -> Any:
    """Wrapper simplu peste requests.get cu mesaje de eroare clare."""
    print(f"[INFO] GET {url}")
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        print(f"[ERROR] Request failed for {url}: {e}", file=sys.stderr)
        return None


# -------------------------
# 1) Mining pools (C_hash)
# -------------------------

def fetch_mining_pools(time_period: str = "1m") -> None:
    """
    Folosește mempool.space REST API pentru a lua hashrate pe pool-uri:
    GET /api/v1/mining/hashrate/pools/:timePeriod

    Docs: https://mempool.space/docs/api/rest
    timePeriod: '1m', '3m', '6m', '1y', '2y', '3y'
    """
    url = f"https://mempool.space/api/v1/mining/hashrate/pools/{time_period}"
    r = safe_get(url, timeout=20)
    if r is None:
        return

    data = r.json()  # listă de obiecte

    out_path = DATA_DIR / f"mining_pools_{time_period}.csv"
    if not data:
        print(f"[WARN] No data returned for mining pools. Skipping {out_path}")
        return

    # extragem cheile comune
    fieldnames = sorted({k for item in data for k in item.keys()})

    print(f"[INFO] Writing mining pools CSV -> {out_path}")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            writer.writerow(item)


# -------------------------
# 2) Bitnodes snapshot (C_nodes)
# -------------------------

def fetch_bitnodes_snapshot() -> None:
    """
    Folosește Bitnodes API pentru snapshot-ul curent:
    GET https://bitnodes.io/api/v1/snapshots/latest/

    Docs: https://bitnodes.io/api/
    """
    url = "https://bitnodes.io/api/v1/snapshots/latest/"
    r = safe_get(url, timeout=30)
    if r is None:
        return

    data = r.json()
    nodes = data.get("nodes", {})

    out_path = DATA_DIR / "nodes_bitnodes.csv"
    print(f"[INFO] Writing Bitnodes snapshot -> {out_path}")

    # Structura tipică: "ip:port": [timestamp, user_agent, height, ... country]
    # Dar poate varia; tratăm generic.
    # Vom salva:
    # ip, port, timestamp, user_agent, height, country, raw
    fieldnames = [
        "ip",
        "port",
        "timestamp",
        "user_agent",
        "height",
        "country",
        "raw",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for addr, info in nodes.items():
            # addr = "ip:port"
            if ":" in addr:
                ip, port = addr.rsplit(":", 1)
            else:
                ip, port = addr, ""

            # info este o listă; încercăm să extragem câteva poziții
            ts = info[0] if len(info) > 0 else None
            ua = info[1] if len(info) > 1 else None
            height = info[2] if len(info) > 2 else None
            # țara e adesea la index 7, dar nu mereu
            country = info[7] if len(info) > 7 else None

            writer.writerow({
                "ip": ip,
                "port": port,
                "timestamp": ts,
                "user_agent": ua,
                "height": height,
                "country": country,
                "raw": json.dumps(info, ensure_ascii=False),
            })


# -------------------------
# 3) Mempool snapshot (C_mempool)
# -------------------------

def fetch_mempool_snapshot() -> None:
    """
    Ia un snapshot simplu de mempool de la mempool.space:
    - GET /api/mempool
    - GET /api/v1/fees/recommended

    și salvează într-un singur CSV cu un rând.
    """
    url_mempool = "https://mempool.space/api/mempool"
    url_fees = "https://mempool.space/api/v1/fees/recommended"

    r_mem = safe_get(url_mempool, timeout=10)
    r_fee = safe_get(url_fees, timeout=10)

    if r_mem is None or r_fee is None:
        print("[ERROR] Could not fetch mempool or fee data.")
        return

    mem = r_mem.json()       # { count, vsize, total_fee }
    fees = r_fee.json()      # { fastestFee, halfHourFee, hourFee, economyFee, minimumFee }

    out_path = DATA_DIR / "mempool_snapshot.csv"
    print(f"[INFO] Writing mempool snapshot -> {out_path}")

    # combinăm într-un singur rând
    row = {}
    for k, v in mem.items():
        row[f"mempool_{k}"] = v
    for k, v in fees.items():
        row[f"fee_{k}"] = v

    fieldnames = list(row.keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


# -------------------------
# 4) Mempool Demand/Supply CSV (Figshare)
# -------------------------

def download_figshare_file(figshare_file_id: str, out_filename: str) -> None:
    """
    Unele dataset-uri Figshare oferă direct 'ndownloader/files/<id>'.

    Exemplu (din articol):
    Demand CSV from Bitcoin mempool:
      https://figshare.com/articles/dataset/Demand_CSV_from_Bitcoin_mempool/24183288
    Are un fișier intern (id numeric) accesibil la:
      https://figshare.com/ndownloader/files/<file_id>

    Aici presupunem că știi file_id-ul. Pentru dataset-urile din articol:
    - Demand CSV: file id verificat în pagina Figshare.
    - Supply CSV: la fel.

    Dacă se schimbă, trebuie actualizate ID-urile.
    """
    url = f"https://figshare.com/ndownloader/files/{figshare_file_id}"
    r = safe_get(url, timeout=60)
    if r is None:
        return

    out_path = DATA_DIR / out_filename
    print(f"[INFO] Downloading Figshare file -> {out_path}")
    with out_path.open("wb") as f:
        f.write(r.content)


def fetch_mempool_figshare() -> None:
    """
    Descărcăm:
    - Demand CSV from Bitcoin mempool
    - Supply CSV from Bitcoin mempool

    NOTĂ: ID-urile fișierelor sunt luate din pagina Figshare
    ('Download' -> 'Direct link').
    Acestea pot fi actualizate dacă autorul schimbă fișierele.
    """
    # ID-urile din exemplul actual (pot fi diferite în timp; verifică pe figshare dacă apar erori)
    demand_file_id = "24183288"  # <- aici e ID dataset, dar pentru fișier e un alt ID numeric; vezi notă mai jos
    supply_file_id = "24190272"  # idem

    # Observație importantă:
    # De obicei, linkul direct de download are forma:
    #   https://figshare.com/ndownloader/files/<some_big_numeric_id>
    # iar acel ID NU este identic cu ID-ul dataset-ului.
    #
    # Așadar, cel mai robust mod este:
    # 1. Intri în paginile:
    #    - Demand: https://figshare.com/articles/dataset/Demand_CSV_from_Bitcoin_mempool/24183288
    #    - Supply: https://figshare.com/articles/dataset/Supply_CSV_from_Bitcoin_mempool/24190272
    # 2. Apeși "Download" și copiezi linkul direct (de forma /files/XXXXXXXX).
    # 3. Înlocuiești mai jos valorile cu ID-urile corecte.

    print("[WARN] ID-urile din fetch_mempool_figshare sunt placeholders.")
    print("       Verifică în browser linkurile directe de download și actualizează ID-urile dacă e nevoie.")

    # dacă vrei să le dezactivezi până actualizezi, comentează liniile de mai jos:
    # download_figshare_file(demand_file_id, "mempool_demand.csv")
    # download_figshare_file(supply_file_id, "mempool_supply.csv")


# -------------------------
# 5) Hooks pentru custody / governance
# -------------------------

def note_custody_governance():
    """
    Aici doar printăm mesaje de TODO, ca să fie clar ce lipsește.
    Custody & Governance nu au surse CSV publice simple la liber.
    """
    print("\n[INFO] Custody (C_custody) și Governance (C_gov) nu sunt descărcate automat.")
    print("       Pentru custody poți folosi, de exemplu, Glassnode / CryptoQuant / Coinglass (API cheie).")
    print("       Pentru governance poți construi un CSV din:")
    print("         - repo-ul bitcoin/bips (lista BIP + status)")
    print("         - lista de forks (Bitcoin Wiki)")
    print("         - timeline blocksize wars, etc.\n")


# -------------------------
# MAIN
# -------------------------

def main():
    print(f"[INFO] Data directory: {DATA_DIR}")

    # 1) Mining pools
    fetch_mining_pools(time_period="1m")

    # 2) Bitnodes snapshot
    fetch_bitnodes_snapshot()

    # 3) Mempool snapshot
    fetch_mempool_snapshot()

    # 4) Figshare mempool datasets (vezi notele despre ID-urile fișierelor)
    fetch_mempool_figshare()

    # 5) Custody & Governance hooks
    note_custody_governance()

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
