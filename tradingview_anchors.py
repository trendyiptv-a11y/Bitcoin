from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from cohesive_fair_price import (
    DEFAULT_SAMPLES,
    _estimate_cost_usd_per_btc_from_difficulty,
    _fetch_hashrate_history,
    _find_ic_file,
    _finite,
    _load_ic_full,
    _sample_result,
)


def _prepare_historical_context_frame(
    base_dir: str | os.PathLike[str],
    ic_path: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Build the same historical IC + hashrate + production-cost frame used by
    compute_cohesive_fair_price_v2, without changing that production model.
    """
    ic_file = _find_ic_file(base_dir, ic_path)
    ic = _load_ic_full(ic_file)
    h = _fetch_hashrate_history()

    df = ic.merge(h, on="date", how="inner")
    if df.empty:
        raise ValueError("Nu s-au putut alinia seria IC și istoricul HashRate pentru TradingView.")

    df["prod_cost_usd"] = [
        _estimate_cost_usd_per_btc_from_difficulty(float(diff), date)
        for diff, date in zip(df["difficulty_implied"], df["date"])
    ]

    needed = ["close", "prod_cost_usd", "ic_struct", "ic_dir", "ic_flux", "ic_cycle", "regime"]
    if "vol30_index" in df.columns:
        needed.append("vol30_index")

    df = df.replace([math.inf, -math.inf], math.nan).dropna(subset=needed)
    df = df[(df["close"].astype(float) > 0.0) & (df["prod_cost_usd"].astype(float) > 0.0)].copy()
    if df.empty:
        raise ValueError("Nu există puncte istorice valide pentru ancorele TradingView.")

    df["historical_multiplier"] = df["close"].astype(float) / df["prod_cost_usd"].astype(float)
    return df.sort_values("date"), str(ic_file)


def _target_row_for_year(df: pd.DataFrame, year: int) -> Optional[pd.Series]:
    """
    Use the first valid model point from each year. Pine anchors are placed at
    Jan 1, so this gives a stable yearly starting anchor. For the active year,
    coeziv_state_v2.py overrides the value with the live monitor snapshot.
    """
    year_df = df[df["date"].dt.year == int(year)]
    if year_df.empty:
        return None
    return year_df.sort_values("date").iloc[0]


def _anchor_for_row(df: pd.DataFrame, target: pd.Series, samples: int) -> Optional[Dict[str, Any]]:
    target_date = pd.to_datetime(target["date"], utc=True).normalize()
    hist = df[df["date"] < target_date].copy()
    if hist.empty:
        return None

    feature_cols = ["ic_struct", "ic_dir", "ic_flux", "ic_cycle"]
    if "vol30_index" in df.columns and "vol30_index" in target.index and _finite(target["vol30_index"]):
        feature_cols.append("vol30_index")

    dist = pd.Series(0.0, index=hist.index)
    for col in feature_cols:
        if col not in hist.columns or col not in target.index or not _finite(target[col]):
            continue
        dist += ((hist[col].astype(float) - float(target[col])) / 100.0) ** 2
    hist["context_distance"] = dist.pow(0.5)

    n = max(25, min(int(samples), len(hist)))
    nearest = hist.sort_values("context_distance").head(n)
    if nearest.empty:
        return None

    cost = float(target["prod_cost_usd"])
    spot = float(target["close"])
    res = _sample_result(nearest, cost, spot)
    central = res.get("price_p50")
    if not (_finite(central) and float(central) > 0):
        return None

    return {
        "date": str(target_date.date()),
        "central": float(central),
        "miner": cost,
        "p10": res.get("price_p10"),
        "p50": res.get("price_p50"),
        "p90": res.get("price_p90"),
        "ic_close": spot,
        "samples": res.get("samples"),
        "multiplier_p10": res.get("multiplier_p10"),
        "multiplier_p50": res.get("multiplier_p50"),
        "multiplier_p90": res.get("multiplier_p90"),
        "regime": target.get("regime"),
        "source": "historical_context_similarity_v2_year_start",
    }


def compute_tradingview_yearly_anchors(
    base_dir: str | os.PathLike[str],
    samples: int = DEFAULT_SAMPLES,
    start_year: int = 2020,
    ic_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export real yearly anchors for the TradingView Pine snapshot.

    This does not affect the live monitor calculation. It derives historical
    yearly central/p10/p90/miner levels from the same IC + hashrate + production
    cost frame used by compute_cohesive_fair_price_v2.
    """
    df, ic_file = _prepare_historical_context_frame(base_dir, ic_path=ic_path)
    last_year = int(df["date"].dt.year.max())
    first_year = max(int(start_year), int(df["date"].dt.year.min()) + 1)

    yearly: Dict[str, Dict[str, Any]] = {}
    for year in range(first_year, last_year + 1):
        target = _target_row_for_year(df, year)
        if target is None:
            continue
        anchor = _anchor_for_row(df, target, samples=samples)
        if anchor is not None:
            yearly[str(year)] = anchor

    return {
        "method": "tradingview_yearly_anchors_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "ic_series": ic_file,
            "hashrate": "https://api.blockchain.info/charts/hash-rate?timespan=all&format=json",
            "notes": "Yearly anchors use first valid historical point in each year; the active year is overridden by the live monitor snapshot in coeziv_state_v2.py.",
        },
        "samples_requested": int(samples),
        "yearly": yearly,
    }
