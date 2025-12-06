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
            "Nu am găsit ic_btc_series.json în niciuna din locațiile așteptate:\n"
            + "\n".join(candidates)
        )

    with open(chosen, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw["series"])
    # presupunem timestamp în ms în coloana "t"
    if "t" in df.columns:
        df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.set_index("date").sort_index()
    else:
        # fallback: căutăm coloană de timp generică
        time_col = None
        for cand in ["timestamp", "time", "date", "datetime"]:
            if cand in df.columns:
                time_col = cand
                break
        if time_col is None:
            raise ValueError("Nu am găsit coloană de timp în ic_btc_series.json.")
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.set_index(time_col).sort_index()

    return df[
        [
            "close",
            "ic_struct",
            "ic_dir",
            "ic_flux",
            "ic_cycle",
            "regime",
        ]
    ]


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

    - horizon_hours: orizontul de timp pe care măsurăm randamentul (ex: 24h)
    - move_threshold: prag relativ (ex: 0.005 = 0.5%) pentru a considera mișcarea "relevantă"
    - min_samples: număr minim de cazuri similare pentru o probabilitate robustă

    Întoarce un dict cu:
      {
        "probability": p_in_direction (sau None),
        "samples": n,
        "horizon_hours": horizon_hours,
        "source": "ohlc",
        "breakdown": {
            "in_direction": p_in_dir,
            "opposite": p_opp,
            "flat": p_flat,
        },
        "expected_drift": {
            "horizon_hours": horizon_hours,
            "p10": ...,
            "p50": ...,
            "p90": ...,
        }
      }
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

    triplets: List[Tuple[pd.Timestamp, str, float]] = []
    for i in range(0, len(closes) - horizon_steps):
        sig = signals.iloc[i]
        if sig not in ("long", "short", "flat", "neutral"):
            continue

        p0 = closes.iloc[i]
        pH = closes.iloc[i + horizon_steps]
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

    # filtrăm doar cazurile cu semnal identic cu cel actual
    same: List[Tuple[pd.Timestamp, float]] = [
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

    # dacă nu avem suficiente cazuri, păstrăm doar expected_drift (dacă s-a putut calcula)
    if n < max(1, min_samples):
        return out

    in_dir = 0
    opp = 0
    flat = 0

    for _, ret in same:
        if not (isinstance(ret, (int, float)) and math.isfinite(ret)):
            continue

        if last_sig == "long":
            if ret >= move_threshold:
                in_dir += 1
            elif ret <= -move_threshold:
                opp += 1
            else:
                flat += 1
        elif last_sig == "short":
            if ret <= -move_threshold:
                in_dir += 1
            elif ret >= move_threshold:
                opp += 1
            else:
                flat += 1
        else:
            # semnal flat/neutral: "în direcție" = rămâne într-un range mic
            if abs(ret) < move_threshold:
                in_dir += 1
            elif ret > 0:
                opp += 1
            else:
                flat += 1

    total = in_dir + opp + flat
    if total <= 0:
        return out

    prob_dir = in_dir / total

    out["probability"] = prob_dir
    out["breakdown"] = {
        "in_direction": prob_dir,
        "opposite": opp / total,
        "flat": flat / total,
    }

    return out


# ============================
#  ISTORIC SEMNALE / CONTEXTE
# ============================

def build_signal_history(df_signals: pd.DataFrame, limit: int = 30) -> List[Dict[str, Any]]:
    """
    Construiește un istoric simplificat al contextelor (semnalelor) pentru UI.
    Păstrăm doar ultimele `limit` snapshot-uri (implicit 30).
    """

    history: List[Dict[str, Any]] = []
    if df_signals is None or df_signals.empty:
        return history

    tail = df_signals.tail(limit)

    for ts, row in tail.iterrows():
        sig = str(row.get("signal", "")).lower()
        if sig not in ("long", "short", "flat", "neutral"):
            sig = "neutral"

        close_val = row.get("close", None)
        try:
            close_val = float(close_val) if close_val is not None else None
        except Exception:
            close_val = None

        history.append(
            {
                "timestamp": ts.isoformat(),
                "signal": sig,
                "model_price_usd": close_val,
            }
        )

    return history


# ============================
#  MESAJUL COEZIV PENTRU UTILIZATOR
# ============================

def build_message(
    signal: str,
    price: float,
    flow_bias: Optional[str] = None,
    flow_strength: Optional[str] = None,
    liquidity_regime: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    deviation_from_production: Optional[float] = None,  # păstrat în semnătură, nu îl mai folosim în text
) -> str:
    """
    Construiește un mesaj text instituțional care explică:
      - contextul de piață identificat de mecanism (Market Context)
      - Flow Regime și Liquidity Regime pentru snapshotul curent
      - câteva repere istorice (frecvență, drift) pentru contexte similare

    Mesajul descrie contextul, nu dă recomandări de tranzacționare.
    """
    signal_norm = (signal or "").lower()
    if signal_norm == "long":
        ctx_label = "bias pozitiv al contextului de piață"
    elif signal_norm == "short":
        ctx_label = "bias negativ al contextului de piață"
    else:
        ctx_label = "context de piață neutru"

    parts: List[str] = []

    # 1. Context principal + preț de referință
    if isinstance(price, (int, float)) and math.isfinite(price) and price > 0:
        parts.append(
            f"Contextul curent indică un {ctx_label}, "
            f"cu un preț de referință BTC de aproximativ {price:,.0f} USD."
        )
    else:
        parts.append(
            f"Contextul curent indică un {ctx_label}, "
            f"dar prețul de referință nu este disponibil în mod robust."
        )

    # 2. Flow Regime
    if flow_bias or flow_strength:
        bias_txt = (flow_bias or "").strip()
        strength_txt = (flow_strength or "").strip()

        if bias_txt and strength_txt:
            parts.append(
                f"Structura fluxului de piață este caracterizată printr-un bias {bias_txt.lower()}, "
                f"cu intensitate {strength_txt.lower()}."
            )
        elif bias_txt:
            parts.append(
                f"Structura fluxului de piață arată un bias {bias_txt.lower()}, "
                f"fără o intensitate clar definită."
            )
        elif strength_txt:
            parts.append(
                f"Intensitatea fluxului de piață este {strength_txt.lower()}, "
                f"în absența unui bias directional clar."
            )

    # 3. Liquidity Regime
    if liquidity_regime:
        liq_txt = liquidity_regime.lower().strip()

        if liq_txt in ("low", "scăzută", "scazuta", "scăzut"):
            parts.append(
                "Lichiditatea de piață este redusă, ceea ce poate amplifica "
                "mișcările de preț la apariția unor ordine mari."
            )
        elif liq_txt in ("high", "ridicată", "ridicata", "ridicat"):
            parts.append(
                "Lichiditatea de piață este confortabilă, permițând absorbția "
                "ordinelor fără mișcări bruște ale prețului."
            )
        elif liq_txt in ("mid", "moderată", "moderata", "medie", "normală", "normala"):
            parts.append(
                "Lichiditatea de piață este moderată, în linie cu condițiile "
                "specific condițiilor obișnuite de tranzacționare."
            )
        else:
            parts.append(
                "Regimul de lichiditate este definit, dar într-o categorie "
                "care nu se încadrează clar în tiparele clasice low/mid/high."
            )

    # 4. Statistică istorică – frecvența contextelor similare
    if stats:
        try:
            prob = stats.get("probability")
            samples = stats.get("samples") or 0
            horizon = stats.get("horizon_hours", 24)

            if isinstance(prob, (int, float)) and math.isfinite(prob) and samples > 0:
                pct = prob * 100.0
                parts.append(
                    f"Istoric, pentru contexte similare, în orizontul de aproximativ {horizon}h, "
                    f"mişcarea în direcția semnalului actual a apărut în circa {pct:.1f}% "
                    f"din cazuri, pe baza a {samples} observații."
                )
        except Exception:
            pass

    # 5. Expected drift – interval probabil de mișcare
    if stats:
        try:
            drift = stats.get("expected_drift") or {}
            p10 = drift.get("p10", None)
            p50 = drift.get("p50", None)
            p90 = drift.get("p90", None)
            horizon = drift.get("horizon_hours", None)

            if all(isinstance(x, (int, float)) and math.isfinite(x) for x in [p10, p50, p90]) and horizon:
                parts.append(
                    f"Pe baza istoricului, într-un orizont de {horizon}h, "
                    f"randamentele au avut, în contexte similare, un interval tipic "
                    f"între aproximativ {p10*100:.1f}% și {p90*100:.1f}%, "
                    f"cu o valoare mediană în jur de {p50*100:.1f}%."
                )
        except Exception:
            pass

    # 6. Închidere instituțională (fără recomandare)
    parts.append(
        "Acest mesaj descrie exclusiv contextul de piață identificat de mecanismul de analiză "
        "și nu reprezintă o recomandare de tranzacționare sau de investiție."
    )

    return " ".join(parts)


# ============================
#  FUNCȚIA PRINCIPALĂ
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

    # prețul modelului (din seria IC)
    try:
        model_price = float(last.get("close", float("nan")))
        if not (math.isfinite(model_price) and model_price > 0):
            raise ValueError
    except Exception:
        raise RuntimeError("Coloana 'close' lipsește sau nu poate fi convertită la float.")

    # 4. preț spot live (cu fallback la model_price)
    price_source = "model"
    price_for_text = model_price

    try:
        live_price = get_live_btc_price()
        if math.isfinite(live_price) and live_price > 0:
            price_for_text = float(live_price)
            price_source = "spot"
    except Exception as e:  # log, dar nu cădem
        print("Nu am putut obține prețul live BTC. Folosesc prețul din model.", e)

    # 5. cost de producție BTC (ancoră fundamentală, opțională)
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
                if (math.isfinite(price_for_text) and price_for_text > 0)
                else model_price
            )
            if math.isfinite(ref_price) and ref_price > 0:
                deviation_from_production = (ref_price - production_cost) / production_cost
    except Exception:
        deviation_from_production = None

    # 6. deviația curentă față de prețul modelului (pentru regim de piață)
    dev_pct_model: Optional[float] = None
    try:
        if math.isfinite(model_price) and model_price > 0 and math.isfinite(price_for_text):
            dev_pct_model = (price_for_text - model_price) / model_price
    except Exception:
        dev_pct_model = None


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

    # 8. Regim de piață – clasificator CANONIC
    try:
        regime_input = {
            "model_price_usd": model_price,
            "price_usd": price_for_text,
            "flow_bias": flow.get("flow_bias"),
            "flow_strength": flow.get("flow_strength"),
            "liquidity_regime": liq.get("liquidity_regime"),
            "liquidity_strength": liq.get("liquidity_strength"),
        }

        regime_obj = canonical_classify_market_regime(regime_input)
        if regime_obj is not None:
            market_regime = {
                "id": regime_obj.id,
                "key": regime_obj.key,
                "label": regime_obj.label,
                "description": regime_obj.description,
            }
        else:
            market_regime = None
    except Exception as e:
        print("Nu am putut clasifica regimul de piață (canonic):", e)
        market_regime = None

    # 9. istoric de contexte (ultimele N puncte)
    signal_history = build_signal_history(df, limit=30)

    # 10. statistică istorică pentru semnalul curent
    stats = compute_signal_stats(df)

    # 11. construim mesajul coeziv
    message = build_message(
        signal=signal,
        price=price_for_text,
        flow_bias=flow.get("flow_bias"),
        flow_strength=flow.get("flow_strength"),
        liquidity_regime=liq.get("liquidity_regime"),
        stats=stats,
        deviation_from_production=deviation_from_production,
    )

    # --- Flow Score (flux de piață) ---
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
        print("Nu am putut calcula Flow Score:", e)
        flow_state = {
            "flow_score": None,
            "flow_bias": None,
            "flow_strength": None,
            "flow_components": {"error": str(e)},
        }

    # --- Liquidity Score (lichiditate piață) ---
    try:
        liq_raw = compute_liquidity_from_daily_csv()
        liq_score = float(liq_raw.get("liquidity_value", 0.0) or 0.0)

        liq_state = {
            "liquidity_score": liq_score,
            "liquidity_regime": liq_raw.get("regime"),
            "liquidity_strength": liq_raw.get("strength"),
            "liquidity_components": {
                "open": liq_raw.get("open"),
                "close": liq_raw.get("close"),
                "liquidity_value": liq_score,
                "regime": liq_raw.get("regime"),
                "strength": liq_raw.get("strength"),
            },
        }
    except Exception as e:
        print("Nu am putut calcula Liquidity Score zilnic:", e)
        liq_state = {
            "liquidity_score": None,
            "liquidity_regime": None,
            "liquidity_strength": None,
            "liquidity_components": {"error": str(e)},
        }

    # construim obiectul 'state' care va fi serializat în JSON

    state: Dict[str, Any] = {
        # timestamp & context intern
        "timestamp": ts.isoformat(),
        "signal_raw": signal,
        "ic_struct": float(last.get("ic_struct", 0.0) or 0.0),
        "ic_dir": float(last.get("ic_dir", 0.0) or 0.0),
        "ic_flux": float(last.get("ic_flux", 0.0) or 0.0),
        "ic_cycle": float(last.get("ic_cycle", 0.0) or 0.0),
        "regime_score": last.get("regime", None),
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

        # regim de piață
        "market_regime": market_regime,

        # cost de producție
        "production_cost_usd": production_cost,
        "production_cost_as_of": production_as_of,
        "deviation_from_production": deviation_from_production,

         # Flow Score
        "flow_state": flow_state,

        # Liquidity Score
        "liquidity_state": liq_state,
    }

    # scriem în fișier JSON
    out_path = os.path.join(BASE_DIR, "coeziv_state.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    # mic log în consolă
    print(f"Snapshot coeziv generat la {state['generated_at']}")
    print(f"Preț BTC (source={price_source}): {price_for_text:,.0f} USD")
    print(f"Preț model: {model_price:,.0f} USD")
    if production_cost is not None:
        print(f"Cost de producție BTC (approx): {production_cost:,.0f} USD")
        if deviation_from_production is not None:
            try:
                print(f"Deviație față de costul de producție: {deviation_from_production*100:.1f}%")
            except Exception:
                pass

    prob = state.get("signal_probability")
    samples = state.get("signal_prob_samples") or 0
    drift = state.get("signal_expected_drift") or {}
    regime_label = (market_regime or {}).get("label", "n/a")

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
            if all(isinstance(x, (int, float)) and math.isfinite(x) for x in [p10, p50, p90]):
                print(
                    f"Drift istoric (context actual): "
                    f"interval aproximativ [{p10*100:.1f}%, {p90*100:.1f}%], "
                    f"mediană ~{p50*100:.1f}%."
                )
        except Exception:
            pass

    print("Regim de piață (canonic):", regime_label)


if __name__ == "__main__":
    main()
