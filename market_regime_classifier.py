# market_regime_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

Position = Literal["below", "around", "above"]
FlowDir = Literal["buy", "sell", "neutral"]
LiqLevel = Literal["low", "mid", "high"]


@dataclass
class MarketRegime:
    id: int
    key: str
    label: str
    description: str


def _normalize_flow_bias(flow_bias: Optional[str]) -> Optional[FlowDir]:
    if not flow_bias:
        return None
    fb = flow_bias.lower()
    if fb.startswith("pozit"):
        return "buy"
    if fb.startswith("negat"):
        return "sell"
    if fb.startswith("neutr"):
        return "neutral"
    return None


def _normalize_liquidity_regime(liq_regime: Optional[str]) -> Optional[LiqLevel]:
    if not liq_regime:
        return None
    lr = liq_regime.lower()
    if "scaz" in lr or "scăz" in lr:
        return "low"
    if "ridic" in lr:
        return "high"
    if "moder" in lr or "normal" in lr:
        return "mid"
    return None


def _compute_position(model_price: Optional[float], spot_price: Optional[float]) -> tuple[Optional[Position], Optional[float]]:
    if model_price is None or spot_price is None:
        return None, None
    if model_price <= 0:
        return None, None

    diff = spot_price - model_price
    pct = diff / model_price  # ex: 0.01 = +1%

    # praguri: ±0.5% = "la model"
    if pct < -0.005:
        pos: Position = "below"
    elif pct > 0.005:
        pos = "above"
    else:
        pos = "around"

    return pos, pct


def classify_market_regime(state: Dict[str, Any]) -> Optional[MarketRegime]:
    """
    Primește dict-ul complet 'state' folosit pentru coeziv_state.json
    și întoarce un MarketRegime sau None dacă nu putem clasifica sigur.
    """

    model_price = state.get("model_price_usd")
    spot_price = state.get("price_usd") or state.get("spot")

    pos, pct = _compute_position(model_price, spot_price)
    flow = _normalize_flow_bias(state.get("flow_bias"))
    liq = _normalize_liquidity_regime(state.get("liquidity_regime"))

    flow_strength = (state.get("flow_strength") or "").lower()
    liq_strength = (state.get("liquidity_strength") or "").lower()

    if not pos or not flow or not liq:
        return None

    # === Reguli canonice (8 situații) ===
    # 1️⃣ Fragilitate maximă
    if (
        pos == "below"
        and flow == "sell"
        and liq == "low"
        and flow_strength in ("medie", "puternică")
    ):
        return MarketRegime(
            id=1,
            key="fragility_max",
            label="Fragilitate maximă",
            description="Sub model, presiune de vânzare și lichiditate scăzută – mișcările sunt amplificate de absența market makerilor."
        )

    # 2️⃣ Bearish structural
    if (
        pos == "below"
        and flow == "sell"
        and liq in ("mid", "high")
    ):
        return MarketRegime(
            id=2,
            key="bearish_structural",
            label="Bearish structural",
            description="Sub model, vânzare în condiții de lichiditate cel puțin moderată – piața acceptă niveluri mai joase."
        )

    # 3️⃣ Piață inertă
    if (
        pos == "below"
        and flow == "neutral"
        and liq in ("low", "mid")
    ):
        return MarketRegime(
            id=3,
            key="inert_below",
            label="Piață inertă",
            description="Sub model, flux echilibrat și lichiditate redusă – lipsă de apetit clar pentru direcție."
        )

    # 4️⃣ Echilibru sănătos
    if (
        pos == "around"
        and flow in ("neutral",)
        and liq in ("mid", "high")
    ):
        return MarketRegime(
            id=4,
            key="equilibrium",
            label="Echilibru sănătos",
            description="BTC este la nivelul modelului, fluxul este relativ echilibrat, iar lichiditatea este confortabilă."
        )

    # 5️⃣ Accumulare ordonată
    if (
        pos == "around"
        and flow == "buy"
        and liq == "high"
    ):
        return MarketRegime(
            id=5,
            key="accumulation",
            label="Accumulare ordonată",
            description="La model, cumpărare în regim de lichiditate ridicată – interes real, dar fără grabă."
        )

    # 6️⃣ Expansiune sănătoasă
    if (
        pos == "above"
        and flow == "buy"
        and liq == "high"
    ):
        return MarketRegime(
            id=6,
            key="expansion",
            label="Expansiune sănătoasă",
            description="Peste model, cumpărare în condiții de lichiditate ridicată – momentum susținut într-o piață elastică."
        )

    # 7️⃣ Fragilitate bullish
    if (
        pos == "above"
        and flow == "buy"
        and liq == "low"
    ):
        return MarketRegime(
            id=7,
            key="fragility_bullish",
            label="Fragilitate bullish",
            description="Peste model, cumpărare într-o piață subțire – raliu rapid, dar instabil."
        )

    # 8️⃣ Tranziție de regim
    if (
        pos == "above"
        and flow == "sell"
        and liq == "low"
    ):
        return MarketRegime(
            id=8,
            key="regime_transition",
            label="Tranziție de regim",
            description="Peste model, vânzare în lichiditate scăzută – distribuție sau schimbare de comportament al pieței."
        )

    # Niciun regim standard nu se potrivește clar
    return None
