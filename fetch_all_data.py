#!/usr/bin/env python3
"""
fetch_all_data.py

Script pentru a descărca automat dataset-urile relevante pentru J_BTC
și a le salva în CSV-uri în directorul `data/`.

Ce descarcă acum:
- mining_pools.csv      (mempool.space /api/v1/mining/pools – distribuție blocuri pe pool-uri)
- nodes_bitnodes.csv    (Bitnodes snapshot - noduri + țări + versiune)
- mempool_snapshot.csv  (mempool.space /api/mempool + /fees/recommended)

Ce rămâne ca TODO:
- mempool_demand.csv / mempool_supply.csv de la Figshare (trebuie actualizate manual ID-urile)
- Custody (C_custody) și Governance (C_gov) – necesită surse cu API key sau construire manuală.
"""

import os
import csv
import json
import sys
from pathlib import Path
from typing import Any
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

def fetch_mining_pools() -> None:
    """
    Fallback stabil pentru mining pools:
    folosim /api/v1/mining/pools (nu /hashrate/pools/1m, care poate fi instabil
    sau filtrat de Cloudflare în GitHub Actions).

    Endpoint-ul întoarce o listă de pool-uri cu număr de blocuri minate
    într-o fereastră recentă. Din blockCount poți deriva share-ul fiecărui pool.
    """
    url = "https://mempool.space/api/v1/mining/pools"
    r = safe_get(url, timeout=20)
    if r is None:
        return

    try:
        data = r.json()  # listă de obiecte
    except ValueError:
        print("[ERROR] Response from mining/pools is not valid JSON.", file=sys.stderr)
        return

    if not data:
        print("[WARN] No data returned for mining pools. (fallback)")
        return

    out_path = DATA_DIR / "mining_pools.csv"
    print(f"[INFO] Writing mining pools CSV -> {out_path}")

    # câmpuri comune (poolId, poolName, blockCount, etc.)
    fieldnames = sorted({k for item in data for k in item.keys()})

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

    try:
        data = r.json()
    except ValueError:
        print("[ERROR] Bitnodes snapshot is not valid JSON.", file=sys.stderr)
        return

    nodes = data.get("nodes", {})

    out_path = DATA_DIR / "nodes_bitnodes.csv"
    print(f"[INFO] Writing Bitnodes snapshot -> {out_path}")

    # Structură tipică: "ip:port": [timestamp, user_agent, height, ... country]
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

    try:
        mem = r_mem.json()       # { count, vsize, total_fee }
        fees = r_fee.json()      # { fastestFee, halfHourFee, hourFee, economyFee, minimumFee }
    except ValueError:
        print("[ERROR] Mempool or fees response is not valid JSON.", file=sys.stderr)
        return

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
# 4) Mempool Figshare (TODO)
# -------------------------

def download_figshare_file(figshare_file_id: str, out_filename: str) -> None:
    """
    Unele dataset-uri Figshare oferă direct 'ndownloader/files/<id>'.

    Exemplu generic:
      https://figshare.com/ndownloader/files/<file_id>

    NOTĂ: file_id NU este același cu ID-ul dataset-ului.
    Trebuie copiat manual din linkul de download al fișierului.
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
    Placeholder pentru:
    - Demand CSV from Bitcoin mempool
    - Supply CSV from Bitcoin mempool

    Pentru a le activa:

    1. Deschizi în browser paginile dataset-urilor Figshare.
    2. Apeși "Download" și copiezi linkul direct (care conține /files/<ID>).
    3. Actualizezi variablele demand_file_id / supply_file_id de mai jos.
    4. Decomentezi apelurile la download_figshare_file(...).

    Până atunci, funcția doar afișează un WARNING.
    """
    print("[WARN] fetch_mempool_figshare: ID-urile fișierelor sunt placeholders.")
    print("       Verifică în browser linkurile directe de download și actualizează file_id-urile.")
    # Exemplu după ce ai ID-uri:
    # demand_file_id = "12345678"
    # supply_file_id = "12345679"
    # download_figshare_file(demand_file_id, "mempool_demand.csv")
    # download_figshare_file(supply_file_id, "mempool_supply.csv")


# -------------------------
# 5) Hooks pentru custody / governance
# -------------------------

def note_custody_governance() -> None:
    """
    Custody (C_custody) și Governance (C_gov) nu sunt descărcate automat,
    pentru că nu există surse CSV publice simple la liber.

    În practică:
    - pentru custody poți folosi Glassnode / CryptoQuant / Coinglass (API cu cheie),
      și să salvezi tu CSV-urile în `data/custody_*.csv`.
    - pentru governance poți construi un CSV din:
      * repo-ul bitcoin/bips (lista BIP + status)
      * lista de forks (Bitcoin Wiki)
      * timeline blocksize wars, etc.
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

def main() -> None:
    print(f"[INFO] Data directory: {DATA_DIR}")

    # 1) Mining pools (C_hash)
    fetch_mining_pools()

    # 2) Bitnodes snapshot (C_nodes)
    fetch_bitnodes_snapshot()

    # 3) Mempool snapshot (C_mempool)
    fetch_mempool_snapshot()

    # 4) Figshare mempool datasets (opțional, TODO)
    fetch_mempool_figshare()

    # 5) Custody & Governance hooks (doar info)
    note_custody_governance()

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
