from __future__ import annotations

import math
from typing import Any, Dict, Optional


def _safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _clean(value: Any, fallback: str = "n/a") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _flow_label(flow_bias: Any, flow_strength: Any, lang: str) -> str:
    bias = _clean(flow_bias, "n/a").lower()
    strength = _clean(flow_strength, "").lower()
    if bias in ("n/a", "none", "null"):
        return "n/a"
    ro_bias = {"neutral": "neutru", "negative": "negativ", "positive": "pozitiv", "neutru": "neutru", "negativ": "negativ", "pozitiv": "pozitiv"}
    en_bias = {"neutru": "neutral", "negativ": "negative", "pozitiv": "positive", "neutral": "neutral", "negative": "negative", "positive": "positive"}
    ro_strength = {"weak": "slab", "moderate": "moderat", "strong": "puternic", "slab": "slab", "moderat": "moderat", "puternic": "puternic"}
    en_strength = {"slab": "weak", "moderat": "moderate", "puternic": "strong", "weak": "weak", "moderate": "moderate", "strong": "strong"}
    if lang == "en":
        return f"{en_bias.get(bias, bias)} {en_strength.get(strength, strength)}".strip()
    return f"{ro_bias.get(bias, bias)} {ro_strength.get(strength, strength)}".strip()


def _liquidity_label(liquidity_regime: Any, liquidity_strength: Any, lang: str) -> str:
    regime = _clean(liquidity_regime, "n/a").lower()
    strength = _clean(liquidity_strength, "").lower()
    if regime in ("n/a", "none", "null"):
        return "n/a"
    high = regime in ("ridicată", "ridicata", "high") or strength in ("puternică", "puternica", "puternic", "strong")
    moderate = regime in ("moderată", "moderata", "moderate")
    low = regime in ("scăzută", "scazuta", "low")
    if lang == "en":
        if high:
            return "good / high"
        if moderate:
            return "moderate"
        if low:
            return "low"
        return regime
    if high:
        return "bună / ridicată"
    if moderate:
        return "moderată"
    if low:
        return "scăzută"
    return regime


def _participation_label(signal: str, flow_bias: Any, liquidity_regime: Any, lang: str) -> str:
    bias = _clean(flow_bias, "").lower()
    liq = _clean(liquidity_regime, "").lower()
    if signal == "long" and bias in ("pozitiv", "positive"):
        return "cohesive participation" if lang == "en" else "participare coezivă"
    if bias in ("negativ", "negative") or liq in ("ridicată", "ridicata", "high"):
        return "tense participation" if lang == "en" else "participare tensionată"
    return "neutral participation" if lang == "en" else "participare neutră"


def build_trader_signal_card(
    *,
    signal: str,
    flow: Dict[str, Any],
    liq: Dict[str, Any],
    stats: Dict[str, Any],
    market_regime: Dict[str, Any],
    dev_pct_model: Optional[float],
    deviation_from_production: Optional[float],
) -> Dict[str, Any]:
    """Audited source-of-truth for the UI trader signal card.

    Uses structured model fields only. The frontend should render this directly and
    keep text matching only as a legacy fallback.
    """
    signal_raw = signal if signal in ("long", "short", "flat", "neutral") else "neutral"
    flow_bias = flow.get("flow_bias")
    flow_strength = flow.get("flow_strength")
    liquidity_regime = liq.get("liquidity_regime")
    liquidity_strength = liq.get("liquidity_strength")
    probability = _safe_float(stats.get("probability"))
    samples = _safe_float(stats.get("samples"))
    expected_drift = _safe_float(stats.get("expected_drift"))
    regime = market_regime or {}
    regime_code = _clean(regime.get("code"), "")
    regime_label = _clean(regime.get("label"), "")

    missing = any(v is None for v in (flow_bias, flow_strength, liquidity_regime, liquidity_strength))
    flow_positive = _clean(flow_bias, "").lower() in ("pozitiv", "positive")
    flow_negative = _clean(flow_bias, "").lower() in ("negativ", "negative")
    liquidity_good = _liquidity_label(liquidity_regime, liquidity_strength, "en") == "good / high"
    prob_ok = probability is not None and probability >= 0.55 and (samples or 0) >= 30
    growth_confirmed = bool(signal_raw == "long" and flow_positive and liquidity_good and prob_ok)
    deep_or_extreme = "dev_extreme" in regime_code or "degradare" in regime_label.lower()
    under_model = dev_pct_model is not None and dev_pct_model < -0.25

    if missing:
        key = "no_data"
    elif signal_raw == "long" and growth_confirmed and not deep_or_extreme:
        key = "buy"
    elif signal_raw == "long" and flow_positive and liquidity_good:
        key = "accumulate"
    elif signal_raw == "short" and flow_negative and not liquidity_good:
        key = "sell"
    elif signal_raw == "short" or (deep_or_extreme and flow_negative):
        key = "risk"
    elif flow_negative and not growth_confirmed:
        key = "attention"
    else:
        key = "wait"

    copy = {
        "buy": ("CUMPĂRARE", "BUY", "Piața confirmă direcția. Forța internă susține urcarea.", "The market confirms direction. Internal strength supports upside.", "Atitudine: cumpărare controlată / acumulare", "Attitude: controlled buying / accumulation"),
        "accumulate": ("ACUMULARE", "ACCUMULATE", "Piața arată refacere, dar confirmarea completă lipsește.", "The market shows recovery, but full confirmation is still missing.", "Atitudine: intrări mici / fără agresivitate", "Attitude: small entries / no aggression"),
        "wait": ("AȘTEAPTĂ", "WAIT", "Piața nu arată panică, dar refacerea nu este confirmată.", "The market does not show panic, but recovery is not confirmed.", "Atitudine: prudență / fără intrări agresive", "Attitude: prudence / no aggressive entries"),
        "attention": ("ATENȚIE", "ATTENTION", "Piața stă în picioare, dar presiunea internă este prezentă.", "The market is standing, but internal pressure is present.", "Atitudine: reducere risc / fără poziții noi mari", "Attitude: reduce risk / no large new positions"),
        "sell": ("VÂNZARE", "SELL", "Presiunea internă domină. Zona curentă nu este susținută.", "Internal pressure dominates. The current zone is not supported.", "Atitudine: reducere expunere", "Attitude: reduce exposure"),
        "risk": ("RISC", "RISK", "Structura este fragilă, iar presiunea internă crește.", "The structure is fragile and internal pressure is rising.", "Atitudine: protecție capital / cash", "Attitude: capital protection / cash"),
        "no_data": ("AȘTEAPTĂ DATE", "WAITING DATA", "Cardul nu are încă toate câmpurile structurale necesare.", "The card does not yet have all required structural fields.", "Atitudine: fără interpretare până la următorul snapshot", "Attitude: no interpretation until the next snapshot"),
    }[key]

    reason_codes = [f"signal_{signal_raw}"]
    reason_codes.append("flow_missing" if flow_bias is None else f"flow_{_clean(flow_bias).lower()}_{_clean(flow_strength).lower()}".replace(" ", "_"))
    reason_codes.append("liquidity_missing" if liquidity_regime is None else f"liquidity_{_clean(liquidity_regime).lower()}_{_clean(liquidity_strength).lower()}".replace(" ", "_"))
    reason_codes.append("growth_confirmed" if growth_confirmed else "growth_not_confirmed")
    reason_codes.append("probability_supported" if prob_ok else "probability_missing_or_weak")
    if deep_or_extreme:
        reason_codes.append("structural_tension_or_extreme_deviation")
    if under_model:
        reason_codes.append("price_under_cohesive_model")

    title_ro, title_en, subtitle_ro, subtitle_en, action_ro, action_en = copy
    return {
        "version": "v1_observed_fields",
        "source": "trader_signal_card.build_trader_signal_card",
        "data_status": "missing_fields" if missing else "ok",
        "key": key,
        "signal_raw": signal_raw,
        "title_ro": title_ro,
        "title_en": title_en,
        "subtitle_ro": subtitle_ro,
        "subtitle_en": subtitle_en,
        "action_ro": action_ro,
        "action_en": action_en,
        "flow_label_ro": _flow_label(flow_bias, flow_strength, "ro"),
        "flow_label_en": _flow_label(flow_bias, flow_strength, "en"),
        "participation_label_ro": _participation_label(signal_raw, flow_bias, liquidity_regime, "ro"),
        "participation_label_en": _participation_label(signal_raw, flow_bias, liquidity_regime, "en"),
        "liquidity_label_ro": _liquidity_label(liquidity_regime, liquidity_strength, "ro"),
        "liquidity_label_en": _liquidity_label(liquidity_regime, liquidity_strength, "en"),
        "growth_confirmed": growth_confirmed,
        "growth_label_ro": "da" if growth_confirmed else "nu",
        "growth_label_en": "yes" if growth_confirmed else "no",
        "reason_codes": reason_codes,
        "inputs": {
            "flow_bias": flow_bias,
            "flow_strength": flow_strength,
            "liquidity_regime": liquidity_regime,
            "liquidity_strength": liquidity_strength,
            "signal_probability": probability,
            "signal_prob_samples": samples,
            "signal_expected_drift": expected_drift,
            "market_regime_code": regime_code,
            "market_regime_label": regime_label,
            "model_price_deviation": dev_pct_model,
            "deviation_from_production": deviation_from_production,
        },
    }
