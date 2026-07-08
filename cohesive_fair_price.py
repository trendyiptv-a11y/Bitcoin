from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


BLOCKCHAIN_HASHRATE_URL = "https://api.blockchain.info/charts/hash-rate?timespan=all&format=json"
ELECTRICITY_USD_PER_KWH_BASE = 0.05
# NOTA: markup-ul a fost scos (era 1.25x) ca sa fie consistent cu
# btc_production_auto.py, care calculeaza current_production_cost
# doar din costul electric pur, fara markup. Multiplicatorul istoric
# (historical_multiplier = close / prod_cost_usd) trebuie calculat pe
# aceeasi baza ca ancora "curenta", altfel model_price e sistematic
# deplasat (~25-30% observat empiric inainte de fix).
PRODUCTION_MARKUP = 1.0
DEFAULT_SAMPLES = 250
HTTP_TIMEOUT = 90


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _block_subsidy_btc(dt: pd.Timestamp) -> float:
    d = dt.to_pydatetime().replace(tzinfo=None)
    if d < datetime(2012, 11, 28):
        return 50.0
    if d < datetime(2016, 7, 9):
        return 25.0
    if d < datetime(2020, 5, 11):
        return 12.5
    if d < datetime(2024, 4, 20):
        return 6.25
    return 3.125


def _efficiency_j_per_th(dt: pd.Timestamp) -> float:
    """
    Same structural efficiency schedule used by the Coeziv BTC production-cost model.
    It is a hardware-era approximation, not an accounting claim.
    """
    d = dt.to_pydatetime().replace(tzinfo=None)
    if d < datetime(2013, 1, 1):
        return 5_000_000.0
    if d < datetime(2016, 1, 1):
        return 600.0
    if d < datetime(2018, 1, 1):
        return 120.0
    if d < datetime(2020, 1, 1):
        return 80.0
    if d < datetime(2023, 1, 1):
        return 45.0
    if d < datetime(2024, 6, 1):
        return 33.0
    if d < datetime(2025, 1, 1):
        return 28.2
    # Aliniat cu profilul "average" din btc_production_auto.py (22.0 J/TH),
    # folosit pentru current_production_cost. Anterior era 26.0, ceea ce
    # crea o mica inconsistenta (~18%) intre costul "curent" (anchor) si
    # costurile istorice folosite pentru multiplicator.
    return 22.0


def _estimate_cost_usd_per_btc_from_difficulty(
    difficulty: float,
    when: pd.Timestamp,
    fees_btc_per_block: float = 0.0,
    electricity_usd_per_kwh: float = ELECTRICITY_USD_PER_KWH_BASE,
    production_markup: float = PRODUCTION_MARKUP,
) -> float:
    if not (math.isfinite(difficulty) and difficulty > 0.0):
        return float("nan")

    subsidy = _block_subsidy_btc(when)
    btc_per_block = subsidy + max(float(fees_btc_per_block or 0.0), 0.0)
    if btc_per_block <= 0.0:
        return float("nan")

    eff_j_per_th = _efficiency_j_per_th(when)
    hashes_per_block = difficulty * (2.0 ** 32)
    joules_per_hash = eff_j_per_th / 1e12
    energy_joules = hashes_per_block * joules_per_hash
    energy_kwh = energy_joules / 3_600_000.0
    cost_block_electric = energy_kwh * electricity_usd_per_kwh
    cost_electric_per_btc = cost_block_electric / btc_per_block
    return float(cost_electric_per_btc * production_markup)


def _find_ic_file(base_dir: str | os.PathLike[str], explicit_path: Optional[str] = None) -> Path:
    base = Path(base_dir)
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            base / "ic_btc_series.json",
            base / "j-btc-coeziv" / "ic_btc_series.json",
            base / "data" / "ic_btc_series.json",
            base / "btc-swing-strategy" / "ic_btc_series.json",
        ]
    )
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError("Nu am găsit ic_btc_series.json pentru prețul coeziv V2.")


def _load_ic_full(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(raw.get("series", []))
    if df.empty:
        raise ValueError("ic_btc_series.json nu conține seria IC.")
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.normalize()
    return df.sort_values("date")


def _fetch_hashrate_history() -> pd.DataFrame:
    resp = requests.get(BLOCKCHAIN_HASHRATE_URL, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    raw = resp.json()
    rows = raw.get("values", [])
    if not rows:
        raise ValueError("Blockchain.com nu a întors valori de HashRate.")

    h = pd.DataFrame(rows)
    h["date"] = pd.to_datetime(h["x"], unit="s", utc=True).dt.normalize()
    h = h.rename(columns={"y": "hashrate_ths"})

    # Blockchain.com hash-rate chart is in TH/s.
    # difficulty ~= hashrate_hashes_per_second * 600 / 2^32
    h["difficulty_implied"] = h["hashrate_ths"].astype(float) * 1e12 * 600.0 / (2.0 ** 32)
    return h[["date", "hashrate_ths", "difficulty_implied"]].sort_values("date")


def _quantile(s: pd.Series, p: float) -> float:
    return float(s.astype(float).quantile(p))


def _sample_result(sample: pd.DataFrame, current_production_cost: float, spot_price: float) -> Dict[str, Any]:
    if sample.empty:
        return {"samples": 0, "error": "No similar contexts found"}

    mult = sample["historical_multiplier"].astype(float)
    p10 = _quantile(mult, 0.10)
    p50 = _quantile(mult, 0.50)
    p90 = _quantile(mult, 0.90)

    price_p10 = current_production_cost * p10
    price_p50 = current_production_cost * p50
    price_p90 = current_production_cost * p90

    return {
        "samples": int(len(sample)),
        "date_min": str(sample["date"].min().date()),
        "date_max": str(sample["date"].max().date()),
        "multiplier_p10": p10,
        "multiplier_p50": p50,
        "multiplier_p90": p90,
        "price_p10": float(price_p10),
        "price_p50": float(price_p50),
        "price_p90": float(price_p90),
        "spot_deviation_from_p50": float((spot_price - price_p50) / price_p50) if price_p50 else None,
        "distance_median": float(sample["context_distance"].median()),
        "regime_counts": sample["regime"].value_counts().to_dict() if "regime" in sample.columns else {},
    }


def compute_cohesive_fair_price_v2(
    base_dir: str | os.PathLike[str],
    current_production_cost: float,
    spot_price: float,
    samples: int = DEFAULT_SAMPLES,
    ic_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes the Coeziv V2 fair/model price.

    It does not hardcode a valuation multiplier. The multiplier is learned from
    historical market/cost ratios in historical IC contexts similar to today.
    """
    if not (_finite(current_production_cost) and float(current_production_cost) > 0):
        raise ValueError("current_production_cost invalid pentru model_price V2.")
    if not (_finite(spot_price) and float(spot_price) > 0):
        raise ValueError("spot_price invalid pentru model_price V2.")

    ic_file = _find_ic_file(base_dir, ic_path)
    ic = _load_ic_full(ic_file)
    h = _fetch_hashrate_history()

    df = ic.merge(h, on="date", how="inner")
    if df.empty:
        raise ValueError("Nu s-au putut alinia seria IC și istoricul HashRate.")

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
        raise ValueError("Nu există puncte istorice valide pentru costul de producție V2.")

    df["historical_multiplier"] = df["close"].astype(float) / df["prod_cost_usd"].astype(float)

    current = ic.iloc[-1].copy()
    current_date = pd.to_datetime(current["t"], unit="ms", utc=True).normalize()

    feature_cols = ["ic_struct", "ic_dir", "ic_flux", "ic_cycle"]
    if "vol30_index" in df.columns and "vol30_index" in current.index:
        feature_cols.append("vol30_index")

    dist = pd.Series(0.0, index=df.index)
    for col in feature_cols:
        dist += ((df[col].astype(float) - float(current[col])) / 100.0) ** 2
    df["context_distance"] = dist.pow(0.5)

    hist = df[df["date"] < current_date].copy()
    if hist.empty:
        raise ValueError("Nu există istoric anterior punctului curent pentru model_price V2.")

    n = max(25, min(int(samples), len(hist)))
    nearest_all = hist.sort_values("context_distance").head(n)

    current_regime = current.get("regime")
    same_regime_hist = hist[hist["regime"] == current_regime] if "regime" in hist.columns else pd.DataFrame()
    same_regime = same_regime_hist.sort_values("context_distance").head(n) if not same_regime_hist.empty else pd.DataFrame()

    v2a = _sample_result(nearest_all, float(current_production_cost), float(spot_price))
    v2b = _sample_result(same_regime, float(current_production_cost), float(spot_price))

    # Use p50 from nearest contexts as canonical model price.
    model_price = v2a.get("price_p50")
    if not (_finite(model_price) and float(model_price) > 0):
        raise ValueError("Nu s-a putut calcula price_p50 pentru model_price V2.")

    return {
        "method": "production_cost_context_similarity_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "ic_series": str(ic_file),
            "hashrate": BLOCKCHAIN_HASHRATE_URL,
            "difficulty": "implied from hashrate_ths * 1e12 * 600 / 2^32",
            "cost_model": {
                "electricity_usd_per_kwh": ELECTRICITY_USD_PER_KWH_BASE,
                "production_markup": PRODUCTION_MARKUP,
                "hardware_efficiency": "piecewise Coeziv schedule",
            },
        },
        "current": {
            "date": str(current_date.date()),
            "spot_price_usd": float(spot_price),
            "production_cost_usd": float(current_production_cost),
            "ic_close_usd": float(current["close"]),
            "ic_struct": float(current["ic_struct"]),
            "ic_dir": float(current["ic_dir"]),
            "ic_flux": float(current["ic_flux"]),
            "ic_cycle": float(current["ic_cycle"]),
            "vol30_index": float(current["vol30_index"]) if "vol30_index" in current.index and _finite(current["vol30_index"]) else None,
            "regime": current_regime,
        },
        "aligned_historical_points": int(len(df)),
        "model_price_usd": float(model_price),
        "bands": {
            "p10": v2a.get("price_p10"),
            "p50": v2a.get("price_p50"),
            "p90": v2a.get("price_p90"),
        },
        "components": {
            "production_cost_anchor_usd": float(current_production_cost),
            "historical_multiplier_p10": v2a.get("multiplier_p10"),
            "historical_multiplier_p50": v2a.get("multiplier_p50"),
            "historical_multiplier_p90": v2a.get("multiplier_p90"),
            "similar_context_samples": v2a.get("samples"),
            "similar_context_distance_median": v2a.get("distance_median"),
            "same_regime_price_p50": v2b.get("price_p50"),
            "same_regime_multiplier_p50": v2b.get("multiplier_p50"),
            "same_regime_samples": v2b.get("samples"),
        },
        "v2a_nearest_contexts": v2a,
        "v2b_same_regime": v2b,
    }
