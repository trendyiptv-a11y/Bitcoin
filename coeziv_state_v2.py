from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

import coeziv_state as base
from cohesive_fair_price import compute_cohesive_fair_price_v2
from trader_signal_card import build_trader_signal_card


BASE_DIR = base.BASE_DIR
STRATEGY_DIR = base.STRATEGY_DIR


def _safe_get_costs() -> tuple[Dict[str, Optional[float]], Optional[str]]:
    try:
        cheap_cost, as_of = base.estimate_production_cost(profile="cheap")
        avg_cost, as_of = base.estimate_production_cost(profile="average")
        exp_cost, as_of = base.estimate_production_cost(profile="expensive")
        return {"cheap": cheap_cost, "average": avg_cost, "expensive": exp_cost}, as_of
    except Exception as exc:
        print("Nu am putut estima costurile de producție BTC.", exc)
        return {}, None


def _safe_flow() -> Dict[str, Any]:
    try:
        return base.compute_flow_from_file()
    except Exception as exc:
        print("