import math
import requests
from datetime import datetime, timezone
from typing import Optional, Tuple

# ============================
#  CALCUL AUTOMAT COST PRODUCȚIE BTC
# ============================

def get_difficulty() -> Optional[float]:
    """Returnează difficulty-ul BTC (live)."""
    try:
        resp = requests.get("https://blockchain.info/q/getdifficulty", timeout=10)
        resp.raise_for_status()
        return float(resp.text.strip())
    except:
        return None


def get_reward() -> Optional[float]:
    """Returnează reward-ul per block (BTC)."""
    try:
        resp = requests.get("https://blockchain.info/q/bcperblock", timeout=10)
        resp.raise_for_status()
        return float(resp.text.strip())
    except:
        return None


def estimate_production_cost(
    electricity_price_usd_per_kwh: float = 0.07,
    hw_efficiency_j_per_th: float = 25.0,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculează costul de producție BTC:
    difficulty → hashrate → consum energie → cost → cost/BTC
    """
    difficulty = get_difficulty()
    reward = get_reward()

    if difficulty is None or reward is None:
        return None, None

    # hashing power global (hashes per second)
    network_hashrate_hps = difficulty * (2 ** 32) / 600

    # Joule per hash = J/TH / 1e12
    energy_per_hash_j = hw_efficiency_j_per_th / 1e12

    # Power (W)
    power_watts = network_hashrate_hps * energy_per_hash_j

    # Energie pe zi (kWh) – putere [W] * 24 ore / 1000 (W -> kW)
    energy_kwh_day = power_watts * 24 / 1000

    # Cost energetic pe zi
    daily_cost_usd = energy_kwh_day * electricity_price_usd_per_kwh

    # BTC minați pe zi
    btc_per_day = reward * 144.0

    if btc_per_day <= 0 or daily_cost_usd <= 0:
        return None, None

    cost_per_btc = daily_cost_usd / btc_per_day
    timestamp = datetime.now(timezone.utc).isoformat()

    return float(cost_per_btc), timestamp
