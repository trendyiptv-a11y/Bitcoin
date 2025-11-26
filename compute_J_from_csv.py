#!/usr/bin/env python3
"""
compute_J_from_csv.py

Calculează invariantul structural J_BTC plecând de la CSV-urile din `data/`:

Tehnic (J_tech – date reale):
- mining_pools.csv      -> C_hash
- nodes_bitnodes.csv    -> C_nodes
- mempool_snapshot.csv  -> C_mempool

Social (J_soc – date reale, din guvernanță):
- governance_snapshot.csv   -> issues de consens + divergență implementări

Output:
- data/j_btc_latest.json  (snapshot complet)
- data/j_btc_series.csv   (istoric J_BTC în timp)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MINING_CSV = DATA_DIR / "mining_pools.csv"
NODES_CSV = DATA_DIR / "nodes_bitnodes.csv"
MEMPOOL_CSV = DATA_DIR / "mempool_snapshot.csv"
GOV_CSV = DATA_DIR / "governance_snapshot.csv"

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
    rows = read_csv_rows(path)
    if not rows:
        print("[WARN] mining_pools.csv gol – C_hash=0")
        return 0.0

    counts: Dict[str, float] = {}
    for row in rows:
        name = row.get("poolName") or row.get("name") or "unknown"
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
# C_gov din governance_snapshot.csv (J_soc 100% real)
# ------------------------

def read_governance_snapshot(path: Path = GOV_CSV) -> Dict[str, float]:
    if not path.exists():
        print(f"[WARN] {path} lipsește – C_gov=0.")
        return {
            "issues_governance": 0.0,
            "prs_open": 0.0,
            "impl_diversity": 0.0,
            "height_spread": 0.0,
        }
    rows = read_csv_rows(path)
    if not rows:
        return {
            "issues_governance": 0.0,
            "prs_open": 0.0,
            "impl_diversity": 0.0,
            "height_spread": 0.0,
        }
    row = rows[-1]
    out: Dict[str, float] = {}
    for key in ["issues_governance", "prs_open", "impl_diversity", "height_spread"]:
        try:
            out[key] = float(row.get(key, 0.0) or 0.0)
        except ValueError:
            out[key] = 0.0
    return out


def compute_C_gov_real(
    issues_governance: float,
    prs_open: float,
    impl_diversity: float,
    height_spread: float,
) -> float:
    """
    Construim un scor C_gov în [0,1] din patru componente reale:

    - issues_governance: câte issues sensibile sunt deschise (BIP/consensus)
    - prs_open: câte PR-uri sunt deschise
    - impl_diversity: câte implementări distincte / nod
    - height_spread: cât de mult diferă înălțimile blocurilor
    """
    C_issue = min(1.0, issues_governance / 50.0)
    C_pr = min(1.0, prs_open / 500.0)
    C_impl = min(1.0, impl_diversity / 0.2)
    C_height = max(0.0, min(1.0, height_spread))

    w_issue, w_pr, w_impl, w_height = 0.35, 0.15, 0.30, 0.20
    s = w_issue + w_pr + w_impl + w_height
    w_issue /= s
    w_pr /= s
    w_impl /= s
    w_height /= s

    C_g = (
        w_issue * C_issue +
        w_pr * C_pr +
        w_impl * C_impl +
        w_height * C_height
    )
    return max(0.0, min(1.0, C_g))


def compute_J_tech_from_csv(
    w_hash: float = 0.4,
    w_nodes: float = 0.3,
    w_mempool: float = 0.3,
) -> Tuple[float, Dict[str, float]]:
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


def compute_J_soc_from_csv() -> Tuple[float, Dict[str, float]]:
    """
    J_soc 100% real, dar doar pe componenta de guvernanță (C_gov).
    Componenta de custody lipsește deocamdată (nu avem sursă free/automatizabilă).
    """
    gov = read_governance_snapshot()

    C_custody = 0.0   # încă nemăsurat automat
    C_gov = compute_C_gov_real(
        issues_governance=gov["issues_governance"],
        prs_open=gov["prs_open"],
        impl_diversity=gov["impl_diversity"],
        height_spread=gov["height_spread"],
    )

    J_soc = C_gov   # w_gov = 1, w_custody = 0
    print(f"[INFO] J_soc(real, gov-only) = {J_soc:.3f} (C_gov={C_gov:.3f})")

    return J_soc, {
        "C_custody": C_custody,
        "C_gov": C_gov,
    }


def compute_J_tot_from_csv() -> Dict[str, float]:
    alpha = 0.5  # cât contează J_tech vs J_soc în J_tot

    J_tech, d_tech = compute_J_tech_from_csv()
    J_soc, d_soc = compute_J_soc_from_csv()

    J_tot = alpha * J_tech + (1.0 - alpha) * J_soc
    J_tot = max(0.0, min(1.0, J_tot))

    snapshot: Dict[str, float] = {
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
