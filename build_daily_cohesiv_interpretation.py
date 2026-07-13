#!/usr/bin/env python
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent
STRATEGY = ROOT / "btc-swing-strategy"
STATE_PATH = STRATEGY / "coeziv_state.json"
RISK_PATH = STRATEGY / "risk_window.json"
PARTICIPATION_PATH = STRATEGY / "participation_cohesion.json"
OUT_PATH = STRATEGY / "daily_cohesiv_interpretation.json"


def load(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def n(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def pct(value: Any, digits: int = 1) -> str:
    return f"{n(value) * 100:.{digits}f}".replace(".", ",")


def usd(value: Any) -> str:
    return f"{n(value):,.0f}".replace(",", ".")


def date_from_state(state: Dict[str, Any]) -> str:
    raw = str(state.get("timestamp") or state.get("generated_at") or "")[:10]
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        return raw
    return datetime.now(timezone.utc).date().isoformat()


def ro_date(ymd: str) -> str:
    y, m, d = ymd.split("-")
    return f"{d}.{m}.{y}"


def flow_label(state: Dict[str, Any]) -> str:
    bias = str(state.get("flow_bias") or "neutru").strip().lower()
    strength = str(state.get("flow_strength") or "").strip().lower()
    if bias == "neutru" and strength:
        return f"neutru {strength}"
    if bias and strength:
        return f"{bias} {strength}"
    return bias or "neutru"


def flow_label_en(state: Dict[str, Any]) -> str:
    ro = flow_label(state)
    return (ro.replace("neutru", "neutral")
              .replace("pozitiv", "positive")
              .replace("negativ", "negative")
              .replace("slab", "weak")
              .replace("moderat", "moderate")
              .replace("puternic", "strong")
              .replace("puternică", "strong"))


def liquidity_label(state: Dict[str, Any]) -> str:
    reg = str(state.get("liquidity_regime") or "").strip().lower()
    strength = str(state.get("liquidity_strength") or "").strip().lower()
    if "ridicat" in reg or "ridicată" in reg or "ridicata" in reg:
        return "bună / ridicată"
    if reg:
        return reg
    return strength or "–"


def liquidity_label_en(state: Dict[str, Any]) -> str:
    ro = liquidity_label(state)
    return (ro.replace("bună / ridicată", "good / high")
              .replace("ridicată", "high")
              .replace("ridicata", "high")
              .replace("moderată", "moderate")
              .replace("moderata", "moderate")
              .replace("slabă", "weak")
              .replace("slaba", "weak"))


def trader_phrase(state: Dict[str, Any]) -> str:
    trader = state.get("trader_signal") or {}
    label = trader.get("label") or "AȘTEAPTĂ"
    attitude = trader.get("attitude") or "prudență"
    return f"{label} / {attitude}"


def trader_phrase_en(state: Dict[str, Any]) -> str:
    trader = state.get("trader_signal") or {}
    label = trader.get("label_en") or "WAIT"
    attitude = trader.get("attitude_en") or "prudence"
    return f"{label} / {attitude}"


def main() -> None:
    state = load(STATE_PATH)
    risk = load(RISK_PATH)
    participation = load(PARTICIPATION_PATH)

    ymd = date_from_state(state)
    rd = ro_date(ymd)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    next_expected = (now + timedelta(days=1)).replace(hour=5, minute=30, second=0)

    price = n(state.get("price_usd"))
    model = n(state.get("model_price_usd"))
    prod = n((state.get("production_costs_usd") or {}).get("average"))
    dev_model = state.get("model_price_deviation")
    dev_prod = state.get("deviation_from_production")
    flow = flow_label(state)
    flow_en = flow_label_en(state)
    liquidity = liquidity_label(state)
    liquidity_en = liquidity_label_en(state)

    part_score = n(participation.get("score"))
    part_label = participation.get("label") or "participare tensionată"
    part_level = participation.get("level") or "tense"
    recent = participation.get("inputs") or {}
    recent_short = int(n(recent.get("recent_short_days")))
    recent_long = int(n(recent.get("recent_long_days")))
    recent_flat = int(n(recent.get("recent_flat_days")))

    risk_regime = risk.get("current_regime") or "Degradare profundă · zona istoric modelată neatinsă"
    risk_days = int(n(risk.get("consecutive_degradation_days") or risk.get("streak_days")))
    bottom_low = risk.get("bottom_risk_zone_low")
    bottom_high = risk.get("bottom_risk_zone_high")
    bottom_range = f"{usd(bottom_low)}–{usd(bottom_high)}"

    fg = state.get("fg") or {}
    fg_score = n(fg.get("combined"), 50)
    fg_label = fg.get("cohesive_label") or fg.get("combined_zone") or "Neutru"

    trader = state.get("trader_signal") or {}
    trader_key = trader.get("key") or "wait"
    trader_label = trader.get("label") or "AȘTEAPTĂ"
    trader_label_en = trader.get("label_en") or "WAIT"
    trader_att = trader.get("attitude") or "prudență / fără intrări agresive"
    trader_att_en = trader.get("attitude_en") or "prudence / no aggressive entries"
    trader_reason = trader.get("reason") or "sinteză coezivă a mecanismului"

    formula = (
        f"{str(risk_regime).lower()}, preț peste costul mediu și Fear & Greed {str(fg_label).lower()}, "
        f"dar flux {flow}, lichiditate {liquidity}, participare tensionată și structură încă nereparată."
    )
    formula_en = (
        f"{str(risk_regime).replace('Degradare profundă', 'deep degradation').replace('zona istoric modelată neatinsă', 'historically modeled zone not reached').lower()}, "
        f"price above average cost and Fear & Greed {str(fg_label).lower()}, but {flow_en} flow, {liquidity_en} liquidity, tense participation and structure still unrepaired."
    )

    summary = (
        f"Bitcoin este astăzi în jur de {usd(price)} USD. Valoarea internă estimată de mecanism este în jur de {usd(model)} USD, "
        f"deci prețul stă cu aproximativ {pct(dev_model)}% sub reperul intern. Față de costul mediu de producție, "
        f"prețul este cu aproximativ {pct(dev_prod)}% peste. Fluxul este {flow}, lichiditatea este {liquidity}, "
        f"iar participarea este {part_score:.2f} și rămâne tensionată. Radarul structural indică: {risk_regime}. "
        f"Zona istoric modelată de mecanism, aproximativ {bottom_range}, nu este atinsă, dar structura rămâne sub efectul rupturii și nu este reparată. "
        f"Semnalul trader este {trader_phrase(state)}."
    ).replace(".", ",", 1) if False else None

    summary = (
        f"Bitcoin este astăzi în jur de {usd(price)} USD. Valoarea internă estimată de mecanism este în jur de {usd(model)} USD, "
        f"deci prețul stă cu aproximativ {pct(dev_model)}% sub reperul intern. Față de costul mediu de producție, "
        f"prețul este cu aproximativ {pct(dev_prod)}% peste. Fluxul este {flow}, lichiditatea este {liquidity}, "
        f"iar participarea este {part_score:.2f} și rămâne tensionată. Radarul structural indică: {risk_regime}. "
        f"Zona istoric modelată de mecanism, aproximativ {bottom_range}, nu este atinsă, dar structura rămâne sub efectul rupturii și nu este reparată. "
        f"Semnalul trader este {trader_phrase(state)}."
    )

    summary_en = (
        f"Bitcoin is around {usd(price)} USD today. The mechanism's estimated internal value is around {usd(model)} USD, "
        f"so price sits about {abs(n(dev_model)*100):.1f}% below the internal reference. Versus average production cost, "
        f"price is about {n(dev_prod)*100:.1f}% above. Flow is {flow_en}, liquidity is {liquidity_en}, "
        f"while participation is {part_score:.2f} and remains tense. The structural radar indicates: Deep degradation · historically modeled zone not reached. "
        f"The mechanism's modeled zone, roughly {bottom_range}, is not reached, but the structure remains under the effect of the rupture and is not repaired. "
        f"The trader signal is {trader_phrase_en(state)}."
    )

    full = (
        f"## Interpretare coezivă BTC — {rd}\n\n"
        f"Bitcoin este astăzi în jur de **{usd(price)} USD**, iar valoarea internă estimată de mecanism este în jur de **{usd(model)} USD**. BTC stă cu aproximativ **{pct(dev_model)}% sub** reperul intern.\n\n"
        f"Față de costul mediu de producție, prețul este cu aproximativ **{pct(dev_prod)}% peste**.\n\n"
        f"Expresia curentă a radarului este **{risk_regime}**. Zona modelată de mecanism, aproximativ **{bottom_range} USD**, nu este atinsă. Structura rămâne sub efectul rupturii și nu este reparată complet.\n\n"
        f"Fluxul este **{flow}**, lichiditatea este **{liquidity}**, iar participarea este **{part_score:.2f} / 100** și rămâne **tensionată**.\n\n"
        f"Fear & Greed coeziv este **{fg_label}**, cu scor în jur de **{fg_score:.2f}**.\n\n"
        f"Semnalul trader scris de mecanism este **{trader_label}**, cu atitudine de **{trader_att}**. Motivul sintetic este: **{trader_reason}**.\n\n"
        f"Formula zilei este:\n\n**{formula}**"
    )

    full_en = (
        f"## BTC cohesive interpretation — {rd}\n\n"
        f"Bitcoin is around **{usd(price)} USD** today, while the mechanism's estimated internal value is around **{usd(model)} USD**. BTC sits about **{abs(n(dev_model)*100):.1f}% below** the internal reference.\n\n"
        f"Versus average production cost, price is about **{n(dev_prod)*100:.1f}% above**.\n\n"
        f"The current radar expression is **Deep degradation · historically modeled zone not reached**. The modeled zone, roughly **{bottom_range} USD**, is not reached. The structure remains under the effect of the rupture and is not fully repaired.\n\n"
        f"Flow is **{flow_en}**, liquidity is **{liquidity_en}**, and participation is **{part_score:.2f} / 100** and remains **tense**.\n\n"
        f"Cohesive Fear & Greed is **{fg_label}**, with a score around **{fg_score:.2f}**.\n\n"
        f"The trader signal written by the mechanism is **{trader_label_en}**, with an attitude of **{trader_att_en}**.\n\n"
        f"The daily formula is:\n\n**{formula_en}**"
    )

    data = {
        "date": ymd,
        "generated_at": now.isoformat(),
        "last_analysis_at": now.isoformat(),
        "next_analysis_expected_at": next_expected.isoformat(),
        "analysis_status": "auto_updated",
        "analysis_status_message": "Analiza zilei este generată automat după actualizarea mecanismului.",
        "source": "automatic_pipeline_after_mechanism_update",
        "status": "approved",
        "general_state": f"BTC rămâne în {str(risk_regime).lower()}. Fluxul este {flow}, lichiditatea este {liquidity}, participarea rămâne tensionată, iar structura rămâne nereparată. Semnalul trader este {trader_phrase(state)}.",
        "summary": summary,
        "participation": f"Participarea este {part_score:.2f} din 100 și rămâne în zona tensionată. Indicatorul arată {recent_flat} zile neutre, {recent_short} zile defensive și {recent_long} zile de creștere.",
        "risk_window": f"Radarul structural indică: {risk_regime}. Cele aproximativ {risk_days} zile măsoară durata în care structura a rămas sub efectul rupturii, fără reparare completă. Zona istoric modelată, aproximativ {bottom_range}, este încă neatinsă.",
        "fear_greed": f"Fear & Greed coeziv este {fg_label}, cu scor combinat în jur de {fg_score:.2f}.",
        "market_regime": f"Prețul este cu aproximativ {pct(dev_model)}% sub valoarea internă estimată și cu aproximativ {pct(dev_prod)}% peste costul mediu de producție.",
        "watch_next": "Urmărim schimbarea fluxului, revenirea participării peste 70–75, apariția contextelor reale de creștere și eventualele apropieri de zona istoric modelată.",
        "plain_language": f"BTC nu este sub costul mediu de producție, dar mecanismul vede flux {flow}, participare tensionată și structură nereparată. Mesajul trader este: {trader_phrase(state)}.",
        "radar_info": {
            "date": rd,
            "context": "neutru" if str(state.get("signal", "flat")).lower() == "flat" else str(state.get("signal")),
            "price": "degradare profundă" if "Degradare profundă" in str(risk_regime) else str(risk_regime),
            "price_detail": "zona istoric modelată este neatinsă, iar structura nu este reparată",
            "participation": "tensionată" if part_level == "tense" else str(part_label),
            "flow": flow,
            "liquidity": liquidity,
            "growth_context": "prezent" if recent_long > 0 else "absent",
            "structural_risk": f"degradare profundă · {risk_days}z" if risk_days else str(risk_regime).lower(),
            "formula": formula,
            "visual": {
                "growth_present": recent_long > 0,
                "risk_reduced": False,
                "participation_level": part_level,
                "flow_level": "negative" if "negativ" in flow else ("positive" if "pozitiv" in flow else "weak_neutral"),
                "liquidity_level": "high" if "ridicat" in liquidity else "moderate",
            },
        },
        "general_state_en": f"BTC remains in deep degradation with the historically modeled zone not reached. Flow is {flow_en}, liquidity is {liquidity_en}, participation remains tense and structure remains unrepaired. The trader signal is {trader_phrase_en(state)}.",
        "summary_en": summary_en,
        "participation_en": f"Participation is {part_score:.2f} out of 100 and remains tense. The indicator shows {recent_flat} neutral days, {recent_short} defensive days and {recent_long} growth days.",
        "risk_window_en": f"The structural radar indicates: Deep degradation · historically modeled zone not reached. The roughly {risk_days} days measure the duration under the rupture effect, without full repair. The historically modeled zone, roughly {bottom_range}, is still not reached.",
        "fear_greed_en": f"Cohesive Fear & Greed is {fg_label}, with a combined score around {fg_score:.2f}.",
        "market_regime_en": f"Price is about {abs(n(dev_model)*100):.1f}% below estimated internal value and about {n(dev_prod)*100:.1f}% above average production cost.",
        "watch_next_en": "We watch flow changes, participation returning above 70–75, real growth contexts appearing and any move toward the historically modeled zone.",
        "plain_language_en": f"BTC is not below average production cost, but the mechanism sees {flow_en} flow, tense participation and unrepaired structure. The trader message is: {trader_phrase_en(state)}.",
        "radar_info_en": {
            "date": rd,
            "context": "neutral",
            "price": "deep degradation",
            "price_detail": "the historically modeled zone is not reached, and the structure is not repaired",
            "participation": "tense",
            "flow": flow_en,
            "liquidity": liquidity_en,
            "growth_context": "present" if recent_long > 0 else "absent",
            "structural_risk": f"deep degradation · {risk_days}d" if risk_days else "deep degradation",
            "formula": formula_en,
            "visual": {
                "growth_present": recent_long > 0,
                "risk_reduced": False,
                "participation_level": part_level,
                "flow_level": "negative" if "negative" in flow_en else ("positive" if "positive" in flow_en else "weak_neutral"),
                "liquidity_level": "high" if "high" in liquidity_en else "moderate",
            },
        },
        "full_interpretation": full,
        "full_interpretation_en": full_en,
        "confidence": "moderată",
        "disclaimer": f"Interpretare actualizată automat pe {rd}. Interpretare structurală experimentală, nu recomandare financiară.",
        "disclaimer_en": f"Interpretation automatically updated on {rd}. Experimental structural interpretation, not financial advice.",
    }

    OUT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Daily cohesive interpretation written to {OUT_PATH}")


if __name__ == "__main__":
    main()
