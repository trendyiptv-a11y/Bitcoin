import json
import os
import sys
import math
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, List, Any

import pandas as pd
import requests

# ============================
#  SETĂRI DE PATH
# ============================

BASE_DIR = os.path.dirname(__file__)
STRATEGY_DIR = os.path.join(BASE_DIR, "btc-swing-strategy")

if STRATEGY_DIR not in sys.path:
    sys.path.append(STRATEGY_DIR)

from btc_swing_strategy import generate_signals

# Flow & Liquidity – folosesc btc_daily.csv prin modulele dedicate
try:
    from btc_flow_score import compute_flow_from_daily_csv as _compute_flow_from_daily_csv

    def compute_flow_from_file(path: Optional[str] = None) -> Dict[str, Any]:
        raw = _compute_flow_from_daily_csv(path or os.path.join("data", "btc_daily.csv"))
        return {
            "flow_score": raw.get("flow_value"),
            "flow_bias": raw.get("bias"),
            "flow_strength": raw.get("strength"),
            "components": {
                "open": raw.get("open"),
                "close": raw.get("close"),
            },
        }

except ImportError:
    def compute_flow_from_file(path: Optional[str] = None) -> Dict[str, Any]:
        return {
            "flow_score": None,
            "flow_bias": None,
            "flow_strength": None,
            "components": {},
        }

try:
    from btc_liquidity_score import compute_liquidity_from_daily_csv as _compute_liquidity_from_daily_csv

    def compute_liquidity_from_file(path: Optional[str] = None) -> Dict[str, Any]:
        raw = _compute_liquidity_from_daily_csv(path or os.path.join("data", "btc_daily.csv"))
        return {
            "liquidity_score": raw.get("liquidity_value"),
            "liquidity_regime": raw.get("regime"),
            "liquidity_strength": raw.get("strength"),
            "components": {
                "avg7": raw.get("avg7"),
            },
        }

except ImportError:
    def compute_liquidity_from_file(path: Optional[str] = None) -> Dict[str, Any]:
        return {
            "liquidity_score": None,
            "liquidity_regime": None,
            "liquidity_strength": None,
            "components": {},
        }

# cost de producție – versiunea automată; fallback la None dacă nu există modulul
try:
    from btc_production_auto import estimate_production_cost
except ImportError:
    def estimate_production_cost(
        electricity_price_usd_per_kwh: float = 0.07,
        hw_efficiency_j_per_th: float = 25.0,
    ) -> Tuple[Optional[float], Optional[str]]:
        return None, None


# ============================
#  ÎNCĂRCARE IC SERIES (ic_btc_series.json)
# ============================


def load_ic_series(path: Optional[str] = None) -> pd.DataFrame:
    candidates = []

    if path:
        candidates.append(path)
    else:
        candidates.extend([
            os.path.join(BASE_DIR, "ic_btc_series.json"),
            os.path.join(BASE_DIR, "data", "ic_btc_series.json"),
            os.path.join(STRATEGY_DIR, "ic_btc_series.json"),
        ])

    chosen = next((p for p in candidates if os.path.exists(p)), None)
    if not chosen:
        raise FileNotFoundError("Nu am găsit ic_btc_series.json")

    with open(chosen, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []

    for ts, payload in raw.items():
        if not isinstance(payload, dict):
            continue

        row = payload.copy()
        row["timestamp"] = ts
        rows.append(row)

    if not rows:
        raise ValueError("ic_btc_series.json este gol sau invalid")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    return df


# ============================
#  UTILITARE
# ============================

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _safe_int(x: Any) -> Optional[int]:
    try:
        v = int(x)
    except Exception:
        return None
    return v


# ============================
#  ISTORIC CONTEXTE PENTRU UI
# ============================

def build_signal_history(df_signals: pd.DataFrame, limit: int = 30) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    if df_signals is None or df_signals.empty:
        return history

    tail = df_signals.tail(limit)

    for ts, row in tail.iterrows():
        sig = str(row.get("signal", "")).lower()
        if sig not in ("long", "short", "flat", "neutral"):
            sig = "neutral"

        close_val = _safe_float(row.get("close"))

        history.append(
            {
                "timestamp": ts.isoformat(),
                "signal": sig,
                "model_price_usd": close_val,
            }
        )

    return history


# ============================
#  PREȚ LIVE BTC (API GRATUIT)
# ============================

def get_live_btc_price() -> float:
    """
    Ia prețul BTC/USD din CoinGecko (API gratuit, fără cheie).
    Dacă apare o eroare, excepția va fi prinsă în main() și
    vom cădea pe prețul din model.
    """
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin", "vs_currencies": "usd"}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return float(data["bitcoin"]["usd"])


# ============================
#  STATISTICĂ ISTORICĂ PENTRU SEMNAL
# ============================

def compute_signal_stats(
    df: pd.DataFrame,
    horizon_hours: int = 24,
    move_threshold: float = 0.005,
    min_samples: int = 100,
) -> Dict[str, Any]:
    """
    Calculează, pe baza seriei IC + semnalelor:
      - probabilitatea ca mișcarea pe următorul interval (ex. 24h)
        să fie în direcția semnalului curent;
      - o distribuție simplă a randamentelor (p10/p50/p90) pentru
        scenariile similare.
    """

    out: Dict[str, Any] = {
        "probability": None,
        "samples": 0,
        "horizon_hours": horizon_hours,
        "source": "ohlc",
        "breakdown": None,
        "expected_drift": None,
    }

    if df is None or df.empty:
        return out

    if "close" not in df.columns or "signal" not in df.columns:
        return out

    # ne asigurăm că indexul e datetime tz-aware
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame-ul trebuie să aibă index de tip DatetimeIndex.")
    df = df.sort_index()
    if df.index.tz is None:
        df = df.tz_localize("UTC")

    closes = df["close"].astype(float)
    signals = df["signal"].astype(str).str.lower().replace({"nan": ""})

    idx = closes.index
    if len(idx) < 3:
        return out

    step_seconds = (idx[1] - idx[0]).total_seconds()
    if step_seconds <= 0:
        return out

    horizon_steps = max(1, round(horizon_hours * 3600 / step_seconds))

    # construim triplete (timestamp, signal, return_horizon)
    triplets: List[Tuple[pd.Timestamp, str, float]] = []
    for i in range(len(closes) - horizon_steps):
        p0 = closes.iloc[i]
        pH = closes.iloc[i + horizon_steps]
        sig = signals.iloc[i]

        if sig not in ("long", "short", "flat", "neutral"):
            continue

        if not (math.isfinite(p0) and math.isfinite(pH) and p0 > 0):
            continue

        ret = (pH - p0) / p0
        ts0 = closes.index[i]
        triplets.append((ts0, sig, ret))

    if not triplets:
        return out

    last_sig = signals.iloc[-1]
    if last_sig not in ("long", "short", "flat", "neutral"):
        return out

    # normalizăm "neutral" -> "flat"
    if last_sig == "neutral":
        last_sig = "flat"

    same = [
        (ts, ret) for (ts, sig, ret) in triplets if sig == last_sig
    ]
    n = len(same)
    out["samples"] = n

    if n < 1:
        return out

    # distribuția randamentelor în contexte similare (pentru expected_drift)
    rets = [ret for _, ret in same if isinstance(ret, (int, float)) and math.isfinite(ret)]
    if rets:
        s = pd.Series(rets)
        try:
            p10 = float(s.quantile(0.10))
            p50 = float(s.quantile(0.50))
            p90 = float(s.quantile(0.90))
            out["expected_drift"] = {
                "horizon_hours": horizon_hours,
                "p10": p10,
                "p50": p50,
                "p90": p90,
            }
        except Exception:
            pass

    # probabilitate în direcția semnalului
    in_dir = 0
    opp = 0
    flat = 0

    for _, ret in same:
        if not isinstance(ret, (int, float)) or not math.isfinite(ret):
            continue

        if last_sig == "long":
            if ret > move_threshold:
                in_dir += 1
            elif ret < -move_threshold:
                opp += 1
            else:
                flat += 1
        elif last_sig == "short":
            if ret < -move_threshold:
                in_dir += 1
            elif ret > move_threshold:
                opp += 1
            else:
                flat += 1
        else:  # flat
            if abs(ret) <= move_threshold:
                in_dir += 1
            elif ret > move_threshold:
                opp += 1
            else:
                flat += 1

    total = in_dir + opp + flat
    if total == 0:
        return out

    p_in_dir = in_dir / total
    p_opp = opp / total
    p_flat = flat / total

    out["probability"] = p_in_dir
    out["breakdown"] = {
        "in_direction": p_in_dir,
        "opposite": p_opp,
        "flat": p_flat,
    }

    return out


# ============================
#  CLASIFICATOR DE REGIM DE PIAȚĂ
# ============================

def classify_market_regime(row: pd.Series, dev_pct: Optional[float] = None) -> Optional[Dict[str, str]]:
    """
    Construiește o etichetă textuală de regim de piață pe baza coloanelor
    din ultimul snapshot (regime, ic_flux, ic_dir) și, opțional, a deviației
    prețului față de model (dev_pct).

    Întoarce un dict de forma:
      {
        "label": "Trend descendent cu deviație moderată față de model",
        "code": "down_trend_moderate_dev"
      }
    sau None, dacă nu avem destule informații.
    """
    if row is None or not isinstance(row, pd.Series):
        return None

    try:
        regime_val = float(row.get("regime", 0.0))
    except Exception:
        regime_val = 0.0

    try:
        flux_val = float(row.get("ic_flux", 0.0))
    except Exception:
        flux_val = 0.0

    trend_label = "Regim neutru"
    trend_code = "neutral"

    if regime_val >= 0.4:
        if flux_val >= 0.2:
            trend_label = "Trend ascendent susținut"
            trend_code = "up_trend_strong"
        else:
            trend_label = "Trend ascendent moderat"
            trend_code = "up_trend_moderate"
    elif regime_val <= -0.4:
        if flux_val <= -0.2:
            trend_label = "Trend descendent susținut"
            trend_code = "down_trend_strong"
        else:
            trend_label = "Trend descendent moderat"
            trend_code = "down_trend_moderate"
    else:
        if abs(flux_val) < 0.1:
            trend_label = "Range neutru"
            trend_code = "range_neutral"
        elif flux_val > 0:
            trend_label = "Range cu bias pozitiv"
            trend_code = "range_bias_up"
        else:
            trend_label = "Range cu bias negativ"
            trend_code = "range_bias_down"

    dev_label = "cu deviație nedefinită față de model"
    dev_code = "dev_unknown"

    if dev_pct is not None and math.isfinite(dev_pct):
        dev_abs = abs(dev_pct)
        if dev_abs < 0.005:
            dev_label = "cu deviație normală față de model"
            dev_code = "dev_normal"
        elif dev_abs < 0.02:
            dev_label = "cu deviație moderată față de model"
            dev_code = "dev_moderate"
        elif dev_abs < 0.04:
            dev_label = "cu deviație tensionată față de model"
            dev_code = "dev_tension"
        else:
            dev_label = "cu deviație extremă față de model"
            dev_code = "dev_extreme"

    label = f"{trend_label} {dev_label}"
    code = f"{trend_code}_{dev_code}"

    return {"label": label, "code": code}


# ============================
#  MESAJ COEZIV PENTRU UI
# ============================

def build_message(
    signal: str,
    price: Optional[float],
    flow_bias: Optional[str],
    flow_strength: Optional[str],
    liquidity_regime: Optional[str],
    stats: Dict[str, Any],
    deviation_from_production: Optional[float],
) -> str:
    parts: List[str] = []

    # 1. interpretare pe baza semnalului
    if signal == "long":
        parts.append(
            "Context structural ușor pozitiv, dar cu probabilitate istorică relativ modestă. "
            "Pentru utilizatori obișnuiți: atenție la dimensiunea pozițiilor long; "
            "nu este un moment de euforie, ci mai degrabă de reacție controlată."
        )
    elif signal == "short":
        parts.append(
            "Context structural ușor negativ, dar cu probabilitate istorică slabă pe downside. "
            "Pentru utilizatori obișnuiți: prudență cu pozițiile mari, în special cele long, "
            "dar nu este un moment de acțiune agresivă de vânzare."
        )
    else:
        parts.append(
            "Context structural mai degrabă neutru; mecanismul nu vede un avantaj clar "
            "nici pentru upside, nici pentru downside în acest snapshot."
        )

    # 2. flux de piață (Flow Score)
    if flow_bias in ("pozitiv", "negativ", "neutru"):
        if flow_bias == "pozitiv":
            txt = "Fluxul actual de piață este orientat spre cumpărare"
        elif flow_bias == "negativ":
            txt = "Fluxul actual de piață este orientat spre vânzare"
        else:
            txt = "Fluxul actual de piață este relativ echilibrat"

        if flow_strength in ("slab", "moderat", "puternic"):
            txt += f" ({flow_strength})."

        parts.append(txt)

    # 3. regim de lichiditate
    if liquidity_regime in ("scăzută", "normală", "ridicată", "moderată"):
        if liquidity_regime == "ridicată":
            txt = "Lichiditatea este ridicată, iar mișcările de preț tind să fie mai stabile."
        elif liquidity_regime == "scăzută":
            txt = "Lichiditatea este scăzută, iar mișcările de preț pot fi mai bruște decât de obicei."
        else:
            txt = "Lichiditatea este într-o zonă normală pentru acest regim de piață."
        parts.append(txt)

    # 4. cost de producție (dacă avem deviație)
    if deviation_from_production is not None and math.isfinite(deviation_from_production):
        pct = deviation_from_production * 100.0
        if pct < -20:
            parts.append(
                "Prețul este semnificativ sub costul estimat de producție al rețelei, "
                "zonă asociată istoric cu stres ridicat asupra minerilor."
            )
        elif pct < -5:
            parts.append(
                "Prețul este ușor sub costul estimat de producție, ceea ce indică "
                "presiune ridicată pe minerii mai puțin eficienți."
            )
        elif pct <= 20:
            parts.append(
                "Prețul este într-o zonă de echilibru față de costul estimat de producție al rețelei."
            )
        elif pct <= 50:
            parts.append(
                "Prețul este semnificativ peste costul estimat de producție, "
                "zonă specifică fazelor speculative moderate."
            )
        else:
            parts.append(
                "Prețul este mult peste costul estimat de producție, "
                "ceea ce sugerează o fază speculativă ridicată."
            )

    # 5. statistică istorică (drift așteptat)
    if stats:
        drift = stats.get("expected_drift") or {}
        p10 = drift.get("p10")
        p50 = drift.get("p50")
        p90 = drift.get("p90")
        horizon = drift.get("horizon_hours", 24)

        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (p10, p50, p90)):
            parts.append(
                f"Istoric, în contexte similare, mișcarea pe următoarele ~{horizon} ore "
                f"a fost de aproximativ {p10 * 100:.1f}% în scenariile mai pesimiste, "
                f"{p50 * 100:.1f}% în scenariile mediene și până la {p90 * 100:.1f}% "
                f"în scenariile mai favorabile."
            )

    parts.append("Nu este o recomandare de tranzacționare, ci o interpretare structurată a riscului.")
    return " ".join(parts)


# ============================
#  MAIN – GENERAREA STĂRII COEZIVE
# ============================

def main() -> None:
    # 1. încărcăm datele IC (structură, direcție, flux, regim)
    df = load_ic_series()

    # 2. generăm semnalele coezive
    df = generate_signals(df)
    if df is None or df.empty:
        raise RuntimeError("DataFrame-ul cu semnale este gol după generate_signals().")

    # 3. extragem ultimul punct din serie
    last = df.iloc[-1]
    ts = last.name  # index datetime

    # semnal brut
    signal = str(last.get("signal", "neutral")).lower()
    if signal not in ("long", "short", "flat", "neutral"):
        signal = "neutral"

    # preț model
    model_price = _safe_float(last.get("close"))

    # preț pentru UI (ideal: spot live)
    price_from_series = _safe_float(last.get("close"))
    price_for_text: Optional[float] = None
    price_source = "spot"

    try:
        price_for_text = get_live_btc_price()
        if not (isinstance(price_for_text, (int, float)) and math.isfinite(price_for_text)):
            price_for_text = None
    except Exception as e:
        print("Nu am putut lua prețul BTC live, folosesc seria locală / modelul.", e)
        price_for_text = None

    if price_for_text is None:
        if price_from_series is not None:
            price_for_text = price_from_series
            price_source = "spot_series"
        elif model_price is not None:
            price_for_text = model_price
            price_source = "model"
        else:
            price_for_text = None
            price_source = "unknown"

    # 4. cost de producție (dacă avem modulul)
    production_cost: Optional[float] = None
    production_as_of: Optional[str] = None
    try:
        production_cost, production_as_of = estimate_production_cost()
    except Exception as e:
        print("Nu am putut estima costul de producție BTC.", e)
        production_cost, production_as_of = None, None

    deviation_from_production: Optional[float] = None
    try:
        if production_cost is not None and math.isfinite(production_cost) and production_cost > 0:
            ref_price = (
                price_for_text
                if (isinstance(price_for_text, (int, float)) and math.isfinite(price_for_text) and price_for_text > 0)
                else model_price
            )
            if ref_price is not None and math.isfinite(ref_price) and ref_price > 0:
                deviation_from_production = (ref_price - production_cost) / production_cost
    except Exception:
        deviation_from_production = None

    # 5. statistică istorică pentru semnal
    stats = compute_signal_stats(df, horizon_hours=24, move_threshold=0.005, min_samples=100)

    # 6. deviație față de model
    dev_pct_model: Optional[float] = None
    try:
        if math.isfinite(model_price) and model_price > 0 and math.isfinite(price_for_text):
            dev_pct_model = (price_for_text - model_price) / model_price
    except Exception:
        dev_pct_model = None

    # 7. clasificarea regimului de piață
    market_regime = classify_market_regime(last, dev_pct_model)

    # 8. Flow Score și Liquidity Score
    try:
        flow = compute_flow_from_file()
    except Exception as e:
        print("Nu am putut calcula Flow Score:", e)
        flow = {
            "flow_score": None,
            "flow_bias": None,
            "flow_strength": None,
            "components": {},
        }

    try:
        liq = compute_liquidity_from_file()
    except Exception as e:
        print("Nu am putut calcula Liquidity Score:", e)
        liq = {
            "liquidity_score": None,
            "liquidity_regime": None,
            "liquidity_strength": None,
            "components": {},
        }

    # 9. istoric pentru UI (doar ultimii 30 de pași)
    signal_history = build_signal_history(df, limit=30)

    # 10. mesaj narativ
    message = build_message(
        signal=signal,
        price=price_for_text,
        flow_bias=flow.get("flow_bias"),
        flow_strength=flow.get("flow_strength"),
        liquidity_regime=liq.get("liquidity_regime"),
        stats=stats,
        deviation_from_production=deviation_from_production,
    )

    # 11. structură finală pentru JSON
    state: Dict[str, Any] = {
        "timestamp": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
        "price_usd": price_for_text,
        "model_price_usd": model_price,
        "price_source": price_source,
        "signal": signal,
        "message": message,
        "generated_at": datetime.now(timezone.utc).isoformat(),

        "signal_history": signal_history,

        "signal_probability": stats.get("probability"),
        "signal_prob_samples": stats.get("samples"),
        "signal_prob_horizon_hours": stats.get("horizon_hours"),
        "signal_prob_source": stats.get("source"),
        "signal_prob_breakdown": stats.get("breakdown"),
        "signal_expected_drift": stats.get("expected_drift"),

        "market_regime": market_regime,

        "production_cost_usd": production_cost,
        "production_cost_as_of": production_as_of,
        "deviation_from_production": deviation_from_production,

        "flow_score": flow.get("flow_score"),
        "flow_bias": flow.get("flow_bias"),
        "flow_strength": flow.get("flow_strength"),
        "flow_components": flow.get("components", {}),

        "liquidity_score": liq.get("liquidity_score"),
        "liquidity_regime": liq.get("liquidity_regime"),
        "liquidity_strength": liq.get("liquidity_strength"),
        "liquidity_components": liq.get("components", {}),
    }

    # 13. scriem JSON în folderul frontend-ului
    os.makedirs(STRATEGY_DIR, exist_ok=True)
    output_path = os.path.join(STRATEGY_DIR, "coeziv_state.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    # 14. log simplu în consolă
    print("Stare coezivă generată:", output_path)
    print(
        "Semnal:", signal,
        "| Sursă preț:", price_source,
        "| Preț mesaj:", f"{price_for_text:,.2f} USD" if price_for_text is not None else "n/a",
        "| Preț model:", f"{model_price:,.2f} USD" if model_price is not None else "n/a",
    )

    prob = state.get("signal_probability")
    samples = state.get("signal_prob_samples")
    drift = state.get("signal_expected_drift") or {}

    if prob is not None and samples:
        try:
            print(
                f"Probabilitate istorică (în contextul actual, "
                f"{state['signal_prob_horizon_hours']}h): "
                f"~{prob * 100:.1f}% (bazat pe {samples} situații similare)."
            )
        except Exception:
            pass

    if drift:
        try:
            p10 = drift.get("p10")
            p50 = drift.get("p50")
            p90 = drift.get("p90")
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (p10, p50, p90)):
                print(
                    "Expected drift (24h) aproximativ: "
                    f"{p10 * 100:.1f}% ... {p50 * 100:.1f}% ... {p90 * 100:.1f}%."
                )
        except Exception:
            pass


# ============================
#  ENTRY POINT
# ============================

if __name__ == "__main__":
    main()
