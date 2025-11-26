import csv
import json
from datetime import datetime, timezone
import os

DATA_DIR = "data"

def load_csv(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def main():
    # --- 1) LOAD TECH METRICS ---
    tech_rows = load_csv(os.path.join(DATA_DIR, "btc_daily.csv"))
    latest_tech = tech_rows[-1]

    C_hash = safe_float(latest_tech.get("C_hash", 0))
    C_nodes = safe_float(latest_tech.get("C_nodes", 0))
    C_mempool = safe_float(latest_tech.get("C_mempool", 0))

    J_tech = (C_hash + C_nodes + C_mempool) / 3.0

    # --- 2) LOAD GOVERNANCE METRICS ---
    gov_rows = load_csv(os.path.join(DATA_DIR, "governance_snapshot.csv"))
    latest_gov = gov_rows[-1]

    C_gov = safe_float(latest_gov.get("C_gov", 0))

    # --- 3) social component = ONLY C_gov ---
    C_custody = 0.0  # no custody available yet
    J_soc = C_gov  # pure governance tension

    # --- 4) total J ---
    J_tot = (J_tech + J_soc) / 2.0

    # --- 5) write j_btc_latest.json ---
    out = {
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

    with open(os.path.join(DATA_DIR, "j_btc_latest.json"), "w") as f:
        json.dump(out, f, indent=2)

    # --- 6) append to j_btc_series.csv ---
    series_path = os.path.join(DATA_DIR, "j_btc_series.csv")
    file_exists = os.path.isfile(series_path)

    with open(series_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp_utc", "J_tot", "J_tech", "J_soc",
                "tech_C_hash", "tech_C_nodes", "tech_C_mempool",
                "soc_C_custody", "soc_C_gov"
            ])
        writer.writerow([
            out["timestamp_utc"], J_tot, J_tech, J_soc,
            C_hash, C_nodes, C_mempool,
            C_custody, C_gov
        ])

if __name__ == "__main__":
    main()
