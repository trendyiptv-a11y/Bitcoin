import json
import os
import sys
import math
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, List, Any

import pandas as pd
import requests

from btc_flow_score import compute_flow_from_daily_csv
from btc_liquidity_score import compute_liquidity_from_daily_csv

from market_regime_classifier import classify_market_regime as canonical_classify_market_regime

# ============================
#  SETĂRI DE PATH
# ============================

BASE_DIR = os.path.dirname(__file__)
STRATEGY_DIR = os.path.join(BASE_DIR, "btc-swing-strategy")

if STRATEGY_DIR not in sys.path:
    sys.path.append(STRATEGY_DIR)

from btc_swing_strategy import generate_signals

# Flow & Liquidity – pot fi opționale, tratăm robust
try:
    from btc_flow_score import compute_flow_from_file
except ImportError:
    def compute_flow_from_file(path: Optional[str] = None) -> Dict[str, Any]:
        return {
            "flow_score": None,
            "flow_bias": None,
            "flow_strength": None,
            "components": {},
        }


try:
    from btc_liquidity_score import compute_liquidity_from_file
except ImportError:
    def compute_liquidity_from_file(path: Optional[str] = None) -> Dict[str, Any]:
        return {
            "liquidity_score": None,
            "liquidity_regime": None,
            "liquidity_strength": None,
            "components": {},
        }


# ============================
#  ÎNCĂRCARE IC SERIES
# ============================

def load_ic_series(path: Optional[str] = None) -> pd.DataFrame:
    """
    Încarcă seria IC BTC din ic_btc_series.json.

    Caută fișierul în mai multe locații posibile:
      - ./ic_btc_series.json
      - ./j-btc-coeziv/ic_btc_series.json
      - ./data/ic_btc_series.json
      - ./btc-swing-strategy/ic_btc_series.json
    """

    candidates: List[str] = []

    if path is not None:
        candidates.append(path)
    else:
        candidates.extend(
            [
                os.path.join(BASE_DIR, "ic_btc_series.json"),
                os.path.join(BASE_DIR, "j-btc-coeziv", "ic_btc_series.json"),
                os.path.join(BASE_DIR, "data", "ic_btc_series.json"),
                os.path.join(STRATEGY_DIR, "ic_btc_series.json"),
            ]
        )

    chosen: Optional[str] = None
    for c in candidates:
        if os.path.exists(c):
            chosen = c
            break

    if chosen is None:
        raise FileNotFoundError(
            f"Nu am găsit ic_btc_series.json în niciuna dintre locațiile candidate: {candidates}"
        )

    with open(chosen, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # raw poate fi dict cu cheie "data" sau direct listă de înregistrări
    if isinstance(raw, dict) and "data" in raw:
        data = raw["data"]
    else:
        data = raw

    df = pd.DataFrame(data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

    return df


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
    Calculează statistică istorică pentru semnalul curent.

    Întoarce un dict de forma:

    {
      "probability": 0.37,        # ≈ 37% șanse ca semnalul să se confirme
      "samples": 1462,
      "horizon_hours": 24,
      "source": "ohlc",
      "breakdown": {
          "horizon_hours": horizon_hours,
          "p10": ...,
          "p50": ...,
          "p90": ...,
      }
    }
    """

    out: Dict[str, Any] = {
        "probability": None,
        "samples": None,
        "horizon_hours": horizon_hours,
        "source": None,
        "breakdown": None,
        "expected_drift": None,
    }

    if df is None or df.empty:
        return out

    if "signal" not in df.columns or "close" not in df.columns:
        return out

    df = df.copy()

    df["return_24h"] = (
        df["close"].shift(-horizon_hours) / df["close"] - 1.0
    )

    valid = df.dropna(subset=["return_24h"])
    if valid.empty:
        return out

    last_signal = str(df["signal"].iloc[-1]).lower()
    if last_signal not in ("long", "short"):
        return out

    if last_signal == "long":
        mask = valid["return_24h"] > move_threshold
    else:
        mask = valid["return_24h"] < -move_threshold

    samples = int(mask.sum())
    total = int(mask.count())

    if samples < max(1, min_samples) or total <= 0:
        return out

    prob = samples / total

    out["probability"] = prob
    out["samples"] = total
    out["source"] = "ohlc"

    rets = valid["return_24h"].values
    p10 = float(pd.Series(rets).quantile(0.10))
    p50 = float(pd.Series(rets).quantile(0.50))
    p90 = float(pd.Series(rets).quantile(0.90))

    out["expected_drift"] = {
        "p10": p10,
        "p50": p50,
        "p90": p90,
    }
    out["breakdown"] = {
        "horizon_hours": horizon_hours,
        "p10": p10,
        "p50": p50,
        "p90": p90,
    }

    return out


# ============================
#  COST DE PRODUCȚIE (PLACEHOLDER ROBUST)
# ============================

def load_production_cost() -> Tuple[Optional[float], Optional[str]]:
    """
    Încarcă un cost de producție pentru BTC, dacă e disponibil
    (de ex. dintr-un fișier separat). Deocamdată implementăm
    o versiune safe care poate fi extinsă ulterior.
    """
    try:
        path = os.path.join(BASE_DIR, "btc_production_cost.json")
        if not os.path.exists(path):
            return None, None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cost = float(data.get("production_cost_usd"))
        as_of = str(data.get("as_of"))
        return cost, as_of
    except Exception:
        return None, None


# ============================
#  CLASIFICATOR DE REGIM DE PIAȚĂ (VECHI - TEXTUAL, NEFOLOSIT ÎN UI)
# ============================

def classify_market_regime(row: pd.Series, dev_pct: Optional[float] = None) -> Optional[Dict[str, str]]:
    """
    Vechea versiune textuală – păstrată pentru compatibilitate internă,
    dar *NU* mai este scrisă în coeziv_state.json. UI folosește acum
    clasificatorul canonic din market_regime_classifier.py.

    Întoarce un dict de forma:
      {
        "label": "...",
        "code": "..."
      }
    sau None.
    """
    regime = str(row.get("regime", "")).lower()
    ic_flux = float(row.get("ic_flux", 0.0) or 0.0)
    ic_dir = float(row.get("ic_dir", 0.0) or 0.0)

    label = None
    code = None

    if "bear" in regime or ic_dir < -0.5:
        label = "Context cu presiune de vânzare"
        code = "sell_pressure"
    elif "bull" in regime or ic_dir > 0.5:
        label = "Context cu presiune de cumpărare"
        code = "buy_pressure"
    elif "range" in regime:
        label = "Context lateral cu volum echilibrat"
        code = "sideways"
    else:
        label = "Context mixt / neclar"
        code = "unclear"

    dev_label = None
    dev_code = None
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

    if label is None:
        return None

    full_label = label
    full_code = code
    if dev_label and dev_code:
        full_label = f"{label} {dev_label}"
        full_code = f"{code}_{dev_code}"

    return {"label": full_label, "code": full_code}


# ============================
#  MESAJ COEZIV PENTRU UTILIZATOR
# ============================

def build_message(
    signal: str,
    price: float,
    flow_bias: Optional[str],
    flow_strength: Optional[str],
    liquidity_regime: Optional[str],
    stats: Dict[str, Any],
    deviation_from_production: Optional[float],
) -> str:
    """
    Construiește un mesaj narativ scurt, în română,
    pentru a explica contextul actual.
    """
    parts: List[str] = []

    if math.isfinite(price):
        parts.append(f"În acest snapshot, Bitcoin este în jur de ~{price:,.0f} USD,")

    if signal == "long":
        parts.append("iar mecanismul identifică un bias pozitiv al contextului de piață.")
    elif signal == "short":
        parts.append("iar mecanismul identifică un bias negativ al contextului de piață.")
    else:
        parts.append("iar mecanismul indică un context mai degrabă neutru / mixt.")

    if flow_bias:
        fb = flow_bias.lower()
        if fb == "pozitiv":
            parts.append("Fluxul recent de piață indică presiune de cumpărare,")
        elif fb == "negativ":
            parts.append("Fluxul recent de piață indică presiune de vânzare,")
        elif fb == "neutru":
            parts.append("Fluxul de piață este relativ echilibrat,")
        else:
            parts.append("Fluxul de piață este atipic,")

        if flow_strength:
            parts.append(f"cu intensitate {flow_strength}.")

    if liquidity_regime:
        lr = liquidity_regime.lower()
        if "scăzut" in lr or "scazut" in lr:
            parts.append("Lichiditatea este relativ scăzută, ceea ce poate amplifica mișcările pe termen scurt.")
        elif "ridicat" in lr or "ridicata" in lr:
            parts.append("Lichiditatea este ridicată, iar piața absoarbe mai ușor ordinele mari.")
        elif "moderată" in lr or "moderata" in lr:
            parts.append("Lichiditatea este moderată, fără condiții extreme.")

    prob = stats.get("probability")
    samples = stats.get("samples")
    if prob is not None and samples:
        try:
            p_pct = prob * 100.0
            parts.append(
                f"Istoric, pentru contexte similare, aproximativ {p_pct:.1f}% dintre episoade au evoluat în "
                f"direcția acestui context pe un orizont de circa 24 de ore (pe baza a {samples} observații)."
            )
        except Exception:
            pass

    drift = stats.get("expected_drift") or {}
    p10 = drift.get("p10")
    p50 = drift.get("p50")
    p90 = drift.get("p90")
    if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (p10, p50, p90)):
        try:
            p10p = p10 * 100.0
            p50p = p50 * 100.0
            p90p = p90 * 100.0
            parts.append(
                "În aceleași tipuri de contexte, pe un orizont de ~24h, randamentele relative au avut de regulă "
                f"un interval aproximativ între {p10p:.1f}% ... {p50p:.1f}% ... {p90p:.1f}%."
            )
        except Exception:
            pass

    if deviation_from_production is not None and math.isfinite(deviation_from_production):
        dev_pct = deviation_from_production * 100.0
        if abs(dev_pct) > 5.0:
            if dev_pct > 0:
                parts.append(
                    f"Prețul este semnificativ peste un nivel estimativ de cost de producție (≈{dev_pct:.1f}% peste), "
                    "ceea ce poate implica o zonă mai speculativă."
                )
            else:
                parts.append(
                    f"Prețul este semnificativ sub un nivel estimativ de cost de producție (≈{abs(dev_pct):.1f}% sub), "
                    "ceea ce poate semnala stres pe partea de ofertă sau oportunități pentru cumpărători."
                )

    parts.append(
        "Acest mesaj descrie contextul structural al pieței pe baza datelor istorice și a modelului intern și nu "
        "reprezintă o recomandare de tranzacționare."
    )

    return " ".join(parts)


# ============================
#  PRINT FLOW & LIQUIDITY (LOG)
# ============================

def print_flow_and_liquidity(flow: Dict[str, Any], liq: Dict[str, Any]) -> None:
    print(
        "Flow Score:",
        flow.get("flow_score"),
        "| Bias flux:",
        flow.get("flow_bias"),
        "| Flow strength:",
        flow.get("flow_strength"),
    )
    print(
        "Liquidity Score:",
        liq.get("liquidity_score"),
        "| Regim lichiditate:",
        liq.get("liquidity_regime"),
        "| Forță lichiditate:",
        liq.get("liquidity_strength"),
    )


# ============================
#  MAIN
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

    # prețul din snapshot-ul modelului
    try:
        model_price = float(last["close"])
    except Exception:
        raise RuntimeError("Coloana 'close' lipsește sau nu poate fi convertită la float.")

    # 4. preț spot live (cu fallback la model_price)
    price_source = "model"
    price_for_text = model_price

    try:
        live_price = get_live_btc_price()
        if math.isfinite(live_price) and live_price > 0:
            price_for_text = live_price
            price_source = "spot"
    except Exception as e:
        print("Nu am putut lua prețul live BTC:", e)

    # 5. cost de producție (opțional)
    production_cost, production_as_of = load_production_cost()
    deviation_from_production: Optional[float] = None
    if production_cost is not None and math.isfinite(production_cost) and production_cost > 0:
        try:
            deviation_from_production = (price_for_text - production_cost) / production_cost
        except Exception:
            deviation_from_production = None

    # 6. deviație față de model
    try:
        dev_pct_model = (price_for_text - model_price) / model_price
    except Exception:
        dev_pct_model = None

    # 7. clasificarea regimului de piață
    # (varianta veche, textuală, a fost înlocuită cu clasificatorul canonic din market_regime_classifier.py)
    # Aici doar calculăm deviația față de model; regimul final va fi calculat MAI JOS,
    # după ce avem și Flow & Liquidity în state.
    market_regime = None

    # 8. Flow Score și Liquidity Score (versiunea "veche" – folosită doar pentru mesaj)
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

    # 9. istoric semnale
    history_cols = ["signal", "close", "ic_dir", "ic_flux", "regime"]
    history_df = df[history_cols].copy()
    history_df = history_df.tail(120)
    signal_history = []
    for idx, row in history_df.iterrows():
        try:
            item = {
                "timestamp": idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx),
                "signal": str(row.get("signal")),
                "model_price_usd": float(row.get("close")),
            }
            signal_history.append(item)
        except Exception:
            continue

    # 10. statistică istorică pentru semnalul curent
    stats = compute_signal_stats(
        df,
        horizon_hours=24,
        move_threshold=0.005,
        min_samples=100,
    )

    # 11. generăm mesajul coeziv pentru utilizator (ce se întâmplă + ce se poate întâmpla)
    message = build_message(
        signal=signal,
        price=price_for_text,
        flow_bias=flow.get("flow_bias"),
        flow_strength=flow.get("flow_strength"),
        liquidity_regime=liq.get("liquidity_regime"),
        stats=stats,
        deviation_from_production=deviation_from_production,
    )

    # --- Flow Score (flux de piață) – versiunea nouă, pe daily_csv, folosită în JSON ---
    try:
        flow_raw = compute_flow_from_daily_csv()  # folosește data/btc_daily.csv
        flow_score = float(flow_raw.get("flow_value", 0.0) or 0.0)

        flow_state = {
            "flow_score": flow_score,
            "flow_bias": flow_raw.get("bias"),
            "flow_strength": flow_raw.get("strength"),
            "flow_components": {
                "open": flow_raw.get("open"),
                "close": flow_raw.get("close"),
                "flow_value": flow_score,
                "bias": flow_raw.get("bias"),
                "strength": flow_raw.get("strength"),
            },
        }
    except Exception as e:
        print("Nu am putut calcula Flow Score (daily_csv):", e)
        flow_state = {
            "flow_score": None,
            "flow_bias": None,
            "flow_strength": None,
            "flow_components": {"error": str(e)},
        }

    # --- Liquidity Score (lichiditate piață) – versiunea nouă, pe daily_csv, folosită în JSON ---
    try:
        liq_raw = compute_liquidity_from_daily_csv()
        liq_score = float(liq_raw.get("liquidity_value", 0.0) or 0.0)

        liq_state = {
            "liquidity_score": liq_score,
            "liquidity_regime": liq_raw.get("regime"),
            "liquidity_strength": liq_raw.get("strength"),
            "liquidity_components": {
                "liquidity_value": liq_score,
                "regime": liq_raw.get("regime"),
                "strength": liq_raw.get("strength"),
                "avg7": liq_raw.get("avg7"),
            },
        }
    except Exception as e:
        print("Nu am putut calcula Liquidity Score:", e)
        liq_state = {
            "liquidity_score": None,
            "liquidity_regime": None,
            "liquidity_strength": None,
            "liquidity_components": {"error": str(e)},
        }

    # 12. construim starea finală
    state: Dict[str, Any] = {
        "timestamp": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),

        # prețuri
        "price_usd": price_for_text,      # ce vede utilizatorul ca preț "acum"
        "model_price_usd": model_price,   # prețul intern al mecanismului
        "price_source": price_source,     # "spot" sau "model"

        # context principal
        "signal": signal,
        "message": message,
        "generated_at": datetime.now(timezone.utc).isoformat(),

        # istoric contexte
        "signal_history": signal_history,

        # statistică istorică
        "signal_probability": stats.get("probability"),
        "signal_prob_samples": stats.get("samples"),
        "signal_prob_horizon_hours": stats.get("horizon_hours"),
        "signal_prob_source": stats.get("source"),
        "signal_prob_breakdown": stats.get("breakdown"),
        "signal_expected_drift": stats.get("expected_drift"),

        # regim de piață (este setat mai jos, după clasificarea canonică)
        "market_regime": None,

        # cost de producție
        "production_cost_usd": production_cost,
        "production_cost_as_of": production_as_of,
        "deviation_from_production": deviation_from_production,

        # Flow Score (nou)
        "flow_score": flow_state["flow_score"],
        "flow_bias": flow_state["flow_bias"],
        "flow_strength": flow_state["flow_strength"],
        "flow_components": flow_state["flow_components"],

        # Liquidity Score (nou)
        "liquidity_score": liq_state["liquidity_score"],
        "liquidity_regime": liq_state["liquidity_regime"],
        "liquidity_strength": liq_state["liquidity_strength"],
        "liquidity_components": liq_state["liquidity_components"],
    }

    # 12b. clasificăm regimul de piață cu clasificatorul CANONIC
    try:
        regime_obj = canonical_classify_market_regime(state)
    except Exception as e:
        print("Nu am putut clasifica regimul de piață (canonic):", e)
        regime_obj = None

    if regime_obj is not None:
        state["market_regime"] = {
            "id": regime_obj.id,
            "key": regime_obj.key,
            "label": regime_obj.label,
            "description": regime_obj.description,
        }
    else:
        state["market_regime"] = None

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
        "| Preț mesaj:", f"{price_for_text:,.2f} USD",
        "| Preț model:", f"{model_price:,.2f} USD",
    )

    prob = state.get("signal_probability")
    samples = state.get("signal_prob_samples")
    drift = state.get("signal_expected_drift") or {}
    regime_label = (state.get("market_regime") or {}).get("label", "n/a")

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
                    "Drift istoric (24h) aproximativ: "
                    f"{p10 * 100:.1f}% ... {p50 * 100:.1f}% ... {p90 * 100:.1f}%"
                )
        except Exception:
            pass

    print("Regim de piață (canonic):", regime_label)
    print_flow_and_liquidity(flow_state, liq_state)


# ============================
#  ENTRY POINT
# ============================

if __name__ == "__main__":
    main()
