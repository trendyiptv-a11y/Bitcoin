import math
import requests
from datetime import datetime, timezone
from typing import Optional, Tuple

PROFILE_PRESETS = {
    "cheap": {      # miner eficient, curent ieftin
        "electricity_price_usd_per_kwh": 0.03,
        "hw_efficiency_j_per_th": 20.0,
    },
    "average": {    # profil mediu
        "electricity_price_usd_per_kwh": 0.05,
        "hw_efficiency_j_per_th": 22.0,
    },
    "expensive": {  # profil scump (similar cu ce aveai)
        "electricity_price_usd_per_kwh": 0.07,
        "hw_efficiency_j_per_th": 25.0,
    },
}

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
    profile: Optional[str] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculează costul de producție BTC:
    difficulty → hashrate → consum energie → cost → cost/BTC
    """
    difficulty = get_difficulty()
    reward = get_reward()

    # Dacă s-a specificat un profil, suprascriem parametrii cu preseturile lui
    if profile is not None:
        preset = PROFILE_PRESETS.get(profile.lower())
        if preset is not None:
            electricity_price_usd_per_kwh = preset["electricity_price_usd_per_kwh"]
            hw_efficiency_j_per_th = preset["hw_efficiency_j_per_th"]
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
