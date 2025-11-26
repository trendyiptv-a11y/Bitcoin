#!/usr/bin/env python3
"""
fetch_governance.py

Calculează un snapshot de tensiune de guvernanță (C_gov_raw) pe baza:

1) GitHub API pentru https://github.com/bitcoin/bitcoin:
   - issues + PR-uri deschise cu label-uri de tip consens/softfork/etc.

2) Divergență de implementări din data/nodes_bitnodes.csv:
   - câte implementări diferite vs total noduri
   - cât de mult deviază înălțimile blocurilor între noduri

Scrie:
    data/governance_snapshot.csv
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

NODES_CSV = DATA_DIR / "nodes_bitnodes.csv"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # opțional, pentru rate limit mai generos


def gh_get(url: str, params: dict | None = None) -> list:
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    print(f"[INFO] GH GET {url}")
    r = requests.get(url, headers=headers, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def count_governance_issues() -> Dict[str, int]:
    """
    Numărăm issues & PR-uri cu label-uri sensibile.
    Pentru simplitate: căutăm în ultimele ~6 luni.
    """
    owner = "bitcoin"
    repo = "bitcoin"

    labels = [
        "consensus",
        "soft fork",
        "taproot",
        "segwit",
        "bip",
    ]

    total_issues = 0
    total_prs = 0

    for label in labels:
        # GitHub search issues API
        q = f"repo:{owner}/{repo} label:\"{label}\" state:open"
        url = "https://api.github.com/search/issues"
        data = gh_get(url, {"q": q, "per_page": 1})
        count = int(data.get("total_count", 0))
        total_issues += count

    # PR-uri deschise (indiferent de label) – tensiune generală
    q_pr = f"repo:{owner}/{repo} is:pr state:open"
    data_pr = gh_get("https://api.github.com/search/issues", {"q": q_pr, "per_page": 1})
    total_prs = int(data_pr.get("total_count", 0))

    return {
        "issues_governance": total_issues,
        "prs_open": total_prs,
    }


def compute_divergence_from_nodes() -> Dict[str, float]:
    """
    Din nodes_bitnodes.csv:
      - calculăm diversitatea implementărilor (user_agent)
      - dispersia înălțimii blockurilor
    """
    path = NODES_CSV
    if not path.exists():
        print("[WARN] nodes_bitnodes.csv lipsește – divergență=0")
        return {"impl_diversity": 0.0, "height_spread": 0.0}

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {"impl_diversity": 0.0, "height_spread": 0.0}

    # diversitatea implementărilor: câte user_agent distincte / total
    ua_counts: Dict[str, int] = {}
    heights: List[int] = []

    for r in rows:
        ua = r.get("user_agent") or "unknown"
        ua_counts[ua] = ua_counts.get(ua, 0) + 1

        try:
            h = int(float(r.get("height", 0) or 0))
            if h > 0:
                heights.append(h)
        except ValueError:
            continue

    total_nodes = len(rows)
    distinct_impl = len(ua_counts)

    impl_diversity = distinct_impl / total_nodes if total_nodes > 0 else 0.0

    if not heights:
        height_spread = 0.0
    else:
        h_min, h_max = min(heights), max(heights)
        # normalizăm pe o fereastră de 100 blocuri – mai mult este tensionat
        spread = h_max - h_min
        height_spread = min(1.0, spread / 100.0)

    return {
        "impl_diversity": impl_diversity,
        "height_spread": height_spread,
    }


def main() -> None:
    gh_counts = count_governance_issues()
    div = compute_divergence_from_nodes()

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"

    out_path = DATA_DIR / "governance_snapshot.csv"
    print(f"[INFO] Writing governance snapshot -> {out_path}")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "issues_governance",
                "prs_open",
                "impl_diversity",
                "height_spread",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": timestamp,
                **gh_counts,
                **div,
            }
        )


if __name__ == "__main__":
    main()
