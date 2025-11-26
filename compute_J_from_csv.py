#!/usr/bin/env python3
"""
compute_J_from_csv.py

Calculează invariantul structural J_BTC plecând de la CSV-urile din `data/`:

Tehnic (J_tech – date reale):
- btc_daily.csv      -> C_hash, C_nodes, C_mempool

Social (J_soc – date reale, doar guvernanță):
- governance_snapshot.csv   -> C_gov

Output:
- data/j_btc_latest.json  (snapshot complet)
- data/j_btc_series.csv   (istoric J_BTC în timp)
"""

import csv
import json
from datetime import datetime, timezone
import os

# 🔥 DEBUG: vrem să vedem exact de unde este rulat acest fișier
print(">>> RUNNING compute_J_from_csv.py FROM:", __file__)

DATA_DIR = "data"


def load_csv(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def main():
    # --- 1) METRICE TEHNICE ---
    btc_daily_path = os.path.join(DATA_DIR, "btc_daily.csv")
    tech_rows = load_csv(btc_daily_path)
    if not tech_rows:
        raise SystemExit(f"btc_daily.csv este gol sau lipsă la {btc_daily_path}")

    latest_tech = tech_rows[-1]

    C_hash = safe_float(latest_tech.get("C_hash", 0))
    C_nodes = safe_float(latest_tech.get("C_nodes", 0))
    C_mempool = safe_float(latest_tech.get("C_mempool", 0))

    J_tech = (C_hash + C_nodes + C_mempool) / 3.0

    print(f">>> TECH: C_hash={C_hash:.3f}, C_nodes={C_nodes:.3f}, C_mempool={C_mempool:.3f}, J_tech={J_tech:.3f}")

    # --- 2) METRICE DE GUVERNANȚĂ ---
    gov_path = os.path.join(DATA_DIR, "governance_snapshot.csv")
    gov_rows = load_csv(gov_path)
    if not gov_rows:
        raise SystemExit(f"governance_snapshot.csv este gol sau lipsă la {gov_path}")

    latest_gov = gov_rows[-1]
    C_gov = safe_float(latest_gov.get("C_gov", 0))

    print(f">>> GOV: C_gov={C_gov:.3f}")

    # --- 3) Componenta socială = DOAR C_gov ---
    C_custody = 0.0  # încă nu avem custody măsurat automat
    J_soc = C_gov    # tensiune socială = tensiune de guvernanță

    # --- 4) Tensiunea totală ---
    J_tot = (J_tech + J_soc) / 2.0

    print(f">>> RESULT: J_tot={J_tot:.3f}, J_tech={J_tech:.3f}, J_soc={J_soc:.3f}")
    print(f">>> RESULT: soc_C_custody={C_custody:.3f}, soc_C_gov={C_gov:.3f}")

    # --- 5) Scriem j_btc_latest.json ---
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),

        "J_tot": round(J_tot, 6),
        "J_tech": round(J_tech, 6),
        "J_soc": round(J_soc, 6),

        "tech_C_hash": C_hash,
        "tech_C_nodes": C_nodes,
        "tech_C_mempool": C_mempool,

        "soc_C_custody": C_custody,
        "soc_C_gov": C_gov,
    }

    latest_path = os.path.join(DATA_DIR, "j_btc_latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f">>> WROTE {latest_path}")

    # --- 6) Adăugăm linie în j_btc_series.csv ---
    series_path = os.path.join(DATA_DIR, "j_btc_series.csv")
    file_exists = os.path.isfile(series_path)

    with open(series_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp_utc", "J_tot", "J_tech", "J_soc",
                "tech_C_hash", "tech_C_nodes", "tech_C_mempool",
                "soc_C_custody", "soc_C_gov",
            ])
        writer.writerow([
            snapshot["timestamp_utc"], J_tot, J_tech, J_soc,
            C_hash, C_nodes, C_mempool,
            C_custody, C_gov,
        ])
    print(f">>> APPENDED {series_path}")


if __name__ == "__main__":
    main()
