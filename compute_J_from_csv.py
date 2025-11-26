#!/usr/bin/env python3
"""
compute_J_from_csv.py

Calculează invariantul structural J_BTC plecând de la CSV-urile din `data/`:

- mining_pools.csv      -> C_hash
- nodes_bitnodes.csv    -> C_nodes
- mempool_snapshot.csv  -> C_mempool
- j_soc_config.json     -> C_custody, C_gov (opțional)

Output:
- data/j_btc_latest.json  (snapshot complet)
- data/j_btc_series.csv   (istoric J_BTC în timp)

Model:
- J_tech = w_hash*C_hash + w_nodes*C_nodes + w_mempool*C_mempool
- J_soc  = w_custody*C_custody + w_gov*C_gov
- J_tot  = alpha*J_tech + (1-alpha)*J_soc

Valorile sunt în [0,1].
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MINING_CSV = DATA_DIR / "mining_pools.csv"
NODES_CSV = DATA_DIR / "nodes_bitnodes.csv"
MEMPOOL_CSV = DATA_DIR / "mempool_snapshot.csv"
J_SOC_CONFIG = DATA_DIR / "j_soc_config.json"

J_LATEST_JSON = DATA_DIR / "j_btc_latest.json"
J_SERIES_CSV = DATA_DIR / "j_btc_series.csv"


# ------------------------
# Utilitare generice
# ------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


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


# ------------------------
# C_hash din mining_pools.csv
# ------------------------

def compute_C_hash_from_csv(path: Path = MINING_CSV) -> float:
    """
    Folosește mining_pools.csv (mempool.space /api/v1/mining/pools).

    Așteptăm coloane de tip:
      - poolName
      - blockCount   (principal)
    """
    rows = read_csv_rows(path)
    if not rows:
        print("[WARN] mining_pools.csv gol – C_hash=0")
        return 0.0

    counts: Dict[str, float] = {}
    for row in rows:
        name = row.get("poolName") or row.get("name") or "unknown"
        # încercăm mai multe nume posibile pentru coloană
        bc_raw = (
            row.get("blockCount")
            or row.get("blocks")
            or row.get("block_count")
        )
        try:
            bc = float(bc_raw) if bc_raw is not None else 0.0
        except ValueError:
            bc = 0.0
        if bc > 0:
            counts[name] = counts.get(name, 0.0) + bc

    if not counts:
        print("[WARN] mining_pools.csv nu are blockCount valid – C_hash=0")
        return 0.0

    C_hash = normalized_hhi(counts)
    print(f"[INFO] C_hash = {C_hash:.3f}")
    return C_hash


# ------------------------
# C_nodes din nodes_bitnodes.csv
# ------------------------

def compute_C_nodes_from_csv(
    path: Path = NODES_CSV,
    N_ref: int = 10000,
    alpha: float = 0.5,
) -> float:
    """
    Folosește nodes_bitnodes.csv generat din Bitnodes snapshot.

    Așteptăm cel puțin coloanele:
      - country
    """
    rows = read_csv_rows(path)
    total_nodes = len(rows)
    if total_nodes == 0:
        print("[WARN] nodes_bitnodes.csv gol – C_nodes=1 (tensiune maximă)")
        return 1.0

    # P_count – penalizare pentru puține noduri
    clamped = min(total_nodes, N_ref)
    P_count = 1.0 - clamped / float(N_ref)

    # P_geo – concentrarea geografică
    country_counts: Dict[str, int] = {}
    for row in rows:
        country = row.get("country") or "NA"
        country_counts[country] = country_counts.get(country, 0) + 1

    total = sum(country_counts.values()) or 1
    shares = {c: v / total for c, v in country_counts.items()}
    P_geo = normalized_hhi(shares)

    C_nodes = alpha * P_count + (1.0 - alpha) * P_geo
    C_nodes = max(0.0, min(1.0, C_nodes))
    print(f"[INFO] C_nodes = {C_nodes:.3f} (P_count={P_count:.3f}, P_geo={P_geo:.3f}, N={total_nodes})")
    return C_nodes


# ------------------------
# C_mempool din mempool_snapshot.csv
# ------------------------

def compute_C_mempool_from_csv(
    path: Path = MEMPOOL_CSV,
    count_ref: int = 50000,
    fee_ref: int = 20,
    beta: float = 0.6,
) -> float:
    """
    mempool_snapshot.csv are un singur rând cu coloane de tip:
      - mempool_count
      - fee_fastestFee
    """
    rows = read_csv_rows(path)
    if not rows:
        print("[WARN] mempool_snapshot.csv gol – C_mempool=0")
        return 0.0

    row = rows[0]

    try:
        count = float(row.get("mempool_count", 0) or 0)
    except ValueError:
        count = 0.0
    try:
        fastest = float(row.get("fee_fastestFee", fee_ref) or fee_ref)
    except ValueError:
        fastest = float(fee_ref)

    C_count = min(1.0, count / float(count_ref))
    C_fee = min(1.0, fastest / float(fee_ref))

    C_mem = beta * C_count + (1.0 - beta) * C_fee
    C_mem = max(0.0, min(1.0, C_mem))
    print(f"[INFO] C_mempool = {C_mem:.3f} (C_count={C_count:.3f}, C_fee={C_fee:.3f})")
    return C_mem


# ------------------------
# C_custody / C_gov din j_soc_config.json
# ------------------------

def load_j_soc_config(path: Path = J_SOC_CONFIG) -> Dict[str, float]:
    """
    j_soc_config.json – fișier mic cu parametri sociali.

    Structură recomandată:

    {
      "p_cust": 0.12,
      "events_E": 2,
      "D_impl": 0.1,
      "w_custody": 0.5,
      "w_gov": 0.5,
      "alpha": 0.5
    }

    Dacă nu există, folosim valori default foarte conservatoare.
    """
    if not path.exists():
        print(f"[WARN] {path} nu există – folosim valori default pentru J_soc.")
        return {
            "p_cust": 0.12,
            "events_E": 2,
            "D_impl": 0.1,
            "w_custody": 0.5,
            "w_gov": 0.5,
            "alpha": 0.5,
        }

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # completăm lipsurile cu default-uri
    defaults = {
        "p_cust": 0.12,
        "events_E": 2,
        "D_impl": 0.1,
        "w_custody": 0.5,
        "w_gov": 0.5,
        "alpha": 0.5,
    }
    for k, v in defaults.items():
        data.setdefault(k, v)
    return data


def compute_C_custody(p_cust: float, p_min: float = 0.10, p_max: float = 0.60) -> float:
    if p_cust <= p_min:
        return 0.0
    if p_cust >= p_max:
        return 1.0
    return (p_cust - p_min) / (p_max - p_min)


def compute_C_gov(events_E: int, E_max: int = 10, D_impl: float = 0.0, gamma: float = 0.5) -> float:
    C_events = min(1.0, events_E / float(E_max))
    C_div = max(0.0, min(1.0, D_impl))
    C_g = gamma * C_events + (1.0 - gamma) * C_div
    return max(0.0, min(1.0, C_g))


# ------------------------
# J_tech, J_soc, J_tot
# ------------------------

def compute_J_tech_from_csv(
    w_hash: float = 0.4,
    w_nodes: float = 0.3,
    w_mempool: float = 0.3,
) -> Tuple[float, Dict[str, float]]:
    # normalizăm ponderile
    s = w_hash + w_nodes + w_mempool
    w_hash /= s
    w_nodes /= s
    w_mempool /= s

    C_hash = compute_C_hash_from_csv()
    C_nodes = compute_C_nodes_from_csv()
    C_mempool = compute_C_mempool_from_csv()

    J_tech = (
        w_hash * C_hash +
        w_nodes * C_nodes +
        w_mempool * C_mempool
    )
    J_tech = max(0.0, min(1.0, J_tech))
    print(f"[INFO] J_tech = {J_tech:.3f}")
    return J_tech, {
        "C_hash": C_hash,
        "C_nodes": C_nodes,
        "C_mempool": C_mempool,
    }


def compute_J_soc_from_config(cfg: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    p_cust = float(cfg["p_cust"])
    events_E = int(cfg["events_E"])
    D_impl = float(cfg["D_impl"])
    w_custody = float(cfg["w_custody"])
    w_gov = float(cfg["w_gov"])

    s = w_custody + w_gov
    w_custody /= s
    w_gov /= s

    C_custody = compute_C_custody(p_cust)
    C_gov = compute_C_gov(events_E, D_impl=D_impl)

    J_soc = w_custody * C_custody + w_gov * C_gov
    J_soc = max(0.0, min(1.0, J_soc))
    print(f"[INFO] J_soc = {J_soc:.3f} (C_custody={C_custody:.3f}, C_gov={C_gov:.3f})")

    return J_soc, {
        "C_custody": C_custody,
        "C_gov": C_gov,
    }


def compute_J_tot_from_csv() -> Dict[str, float]:
    cfg = load_j_soc_config()
    alpha = float(cfg.get("alpha", 0.5))

    J_tech, d_tech = compute_J_tech_from_csv()
    J_soc, d_soc = compute_J_soc_from_config(cfg)

    J_tot = alpha * J_tech + (1.0 - alpha) * J_soc
    J_tot = max(0.0, min(1.0, J_tot))

    snapshot = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "J_tot": J_tot,
        "J_tech": J_tech,
        "J_soc": J_soc,
    }
    for k, v in d_tech.items():
        snapshot[f"tech_{k}"] = v
    for k, v in d_soc.items():
        snapshot[f"soc_{k}"] = v

    return snapshot


# ------------------------
# Output: JSON + series CSV
# ------------------------

def write_latest_json(snapshot: Dict[str, float], path: Path = J_LATEST_JSON) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote latest snapshot -> {path}")


def append_series_row(snapshot: Dict[str, float], path: Path = J_SERIES_CSV) -> None:
    path.parent.mkdir(exist_ok=True)

    # definim coloanele în ordinea asta:
    fieldnames = [
        "timestamp_utc",
        "J_tot",
        "J_tech",
        "J_soc",
        "tech_C_hash",
        "tech_C_nodes",
        "tech_C_mempool",
        "soc_C_custody",
        "soc_C_gov",
    ]

    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()

        row = {k: snapshot.get(k, "") for k in fieldnames}
        writer.writerow(row)

    print(f"[INFO] Appended series row -> {path}")


# ------------------------
# MAIN
# ------------------------

def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")

    snapshot = compute_J_tot_from_csv()
    write_latest_json(snapshot)
    append_series_row(snapshot)


if __name__ == "__main__":
    main()
