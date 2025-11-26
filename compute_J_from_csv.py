#!/usr/bin/env python3
"""
compute_J_from_csv.py

Calculează invariantul structural J_BTC plecând de la CSV-urile din `data/`:

Tehnic (J_tech – date reale):
- mining_pools.csv      -> C_hash (concentrarea puterii de minare)
- nodes_bitnodes.csv    -> C_nodes (răspândirea nodurilor)
- mempool_snapshot.csv  -> C_mempool (presiunea din mempool)

Social (J_soc – date reale, doar guvernanță):
- governance_snapshot.csv   -> C_gov (tensiune de guvernanță)

Output:
- data/j_btc_latest.json  (snapshot complet)
- data/j_btc_series.csv   (istoric J_BTC în timp)
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path("data")


# ------------------ utilitare ------------------ #

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        print(f"[WARN] CSV missing: {path}")
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def normalized_hhi(shares: Dict[str, float]) -> float:
    """
    HHI normalizat în [0,1]:
    0 = distribuție perfect egală
    1 = complet concentrat (un singur actor)
    """
    if not shares:
        return 0.0
    vals = list(shares.values())
    total = sum(vals)
    if total <= 0:
        return 0.0
    norm_vals = [v / total for v in vals]
    hhi = sum(v * v for v in norm_vals)
    n = len(norm_vals)
    if n <= 1:
        return 1.0
    hhi_min = 1.0 / n
    hhi_max = 1.0
    return max(0.0, min(1.0, (hhi - hhi_min) / (hhi_max - hhi_min)))


# ------------------ C_hash din mining_pools.csv ------------------ #

def compute_C_hash(path: Path) -> float:
    rows = read_csv_rows(path)
    if not rows:
        print("[WARN] mining_pools.csv gol sau lipsă – C_hash=0.0")
        return 0.0

    counts: Dict[str, float] = {}
    for r in rows:
        name = r.get("poolName") or r.get("name") or "unknown"
        bc_raw = r.get("blockCount") or r.get("blocks") or r.get("block_count")
        bc = safe_float(bc_raw, 0.0)
        if bc > 0:
            counts[name] = counts.get(name, 0.0) + bc

    if not counts:
        print("[WARN] mining_pools.csv nu are blockCount valid – C_hash=0.0")
        return 0.0

    C_hash = normalized_hhi(counts)
    print(f"[INFO] C_hash={C_hash:.3f}")
    return C_hash


# ------------------ C_nodes din nodes_bitnodes.csv ------------------ #

def compute_C_nodes(path: Path, N_ref: int = 15000, alpha: float = 0.5) -> float:
    rows = read_csv_rows(path)
    total_nodes = len(rows)
    if total_nodes == 0:
        print("[WARN] nodes_bitnodes.csv gol sau lipsă – C_nodes=1.0 (tensiune maximă)")
        return 1.0

    # penalizare pentru puține noduri
    clamped = min(total_nodes, N_ref)
    P_count = 1.0 - clamped / float(N_ref)

    # concentrarea geografică
    country_counts: Dict[str, int] = {}
    for r in rows:
        c = r.get("country") or "NA"
        country_counts[c] = country_counts.get(c, 0) + 1

    total = sum(country_counts.values()) or 1
    shares = {c: v / total for c, v in country_counts.items()}
    P_geo = normalized_hhi(shares)

    C_nodes = alpha * P_count + (1.0 - alpha) * P_geo
    C_nodes = max(0.0, min(1.0, C_nodes))
    print(f"[INFO] C_nodes={C_nodes:.3f} (P_count={P_count:.3f}, P_geo={P_geo:.3f}, N={total_nodes})")
    return C_nodes


# ------------------ C_mempool din mempool_snapshot.csv ------------------ #

def compute_C_mempool(path: Path, count_ref: int = 50000, fee_ref: int = 20, beta: float = 0.6) -> float:
    rows = read_csv_rows(path)
    if not rows:
        print("[WARN] mempool_snapshot.csv gol sau lipsă – C_mempool=0.0")
        return 0.0

    r = rows[0]
    count = safe_float(r.get("mempool_count", 0))
    fastest = safe_float(r.get("fee_fastestFee", fee_ref))

    C_count = min(1.0, count / float(count_ref))
    C_fee = min(1.0, fastest / float(fee_ref))

    C_mem = beta * C_count + (1.0 - beta) * C_fee
    C_mem = max(0.0, min(1.0, C_mem))
    print(f"[INFO] C_mempool={C_mem:.3f} (C_count={C_count:.3f}, C_fee={C_fee:.3f})")
    return C_mem


# ------------------ C_gov din governance_snapshot.csv ------------------ #

def compute_C_gov(path: Path) -> float:
    rows = read_csv_rows(path)
    if not rows:
        print("[WARN] governance_snapshot.csv gol sau lipsă – C_gov=0.0")
        return 0.0

    r = rows[-1]  # ultima înregistrare
    C_gov = safe_float(r.get("C_gov", 0.0))
    print(f"[INFO] C_gov={C_gov:.3f}")
    return C_gov


# ------------------ calcul J ------------------ #

def compute_J() -> dict:
    mining_path = DATA_DIR / "mining_pools.csv"
    nodes_path = DATA_DIR / "nodes_bitnodes.csv"
    mempool_path = DATA_DIR / "mempool_snapshot.csv"
    gov_path = DATA_DIR / "governance_snapshot.csv"

    C_hash = compute_C_hash(mining_path)
    C_nodes = compute_C_nodes(nodes_path)
    C_mempool = compute_C_mempool(mempool_path)
    C_gov = compute_C_gov(gov_path)

    # ponderi tehnice (dacă C_hash=0 din lipsă de date, îl lăsăm dar sistemul îl vede ca „dispersie maximă”)
    w_hash, w_nodes, w_mem = 0.4, 0.3, 0.3
    s = w_hash + w_nodes + w_mem
    w_hash, w_nodes, w_mem = w_hash / s, w_nodes / s, w_mem / s

    J_tech = (
        w_hash * C_hash +
        w_nodes * C_nodes +
        w_mem * C_mempool
    )
    J_tech = max(0.0, min(1.0, J_tech))

    # social: doar guvernanță, custody=0
    C_custody = 0.0
    J_soc = C_gov
    J_soc = max(0.0, min(1.0, J_soc))

    # total
    alpha = 0.5
    J_tot = alpha * J_tech + (1.0 - alpha) * J_soc
    J_tot = max(0.0, min(1.0, J_tot))

    print(f"[INFO] J_tech={J_tech:.3f}, J_soc={J_soc:.3f}, J_tot={J_tot:.3f}")

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
    return snapshot


def write_outputs(snapshot: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    latest_path = DATA_DIR / "j_btc_latest.json"
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[INFO] Wrote {latest_path}")

    series_path = DATA_DIR / "j_btc_series.csv"
    fields = [
        "timestamp_utc", "J_tot", "J_tech", "J_soc",
        "tech_C_hash", "tech_C_nodes", "tech_C_mempool",
        "soc_C_custody", "soc_C_gov",
    ]
    file_exists = series_path.exists()
    with series_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        row = {k: snapshot.get(k, "") for k in fields}
        writer.writerow(row)
    print(f"[INFO] Appended {series_path}")


def main() -> None:
    print(">>> RUNNING compute_J_from_csv.py FROM:", Path(__file__).resolve())
    snapshot = compute_J()
    write_outputs(snapshot)


if __name__ == "__main__":
    main()
