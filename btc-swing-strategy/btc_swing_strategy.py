"""
btc_swing_strategy.py

Framework pentru strategia de swing trading pe BTC bazată pe:
- regimuri macro (A/B/C/D)
- bias direcțional (LONG_ONLY / SHORT_TACTIC / etc.)
- setup-uri de intrare (impuls -> retragere ordonată -> reconfirmare)
- management poziții (TP-uri parțiale + trailing stop)

Notă:
- Funcțiile care țin de "indici coezivi" (CT, CC, CS, CN) și de date brute (OHLC) sunt lăsate ca TODO.
- Folosește pandas sau orice alt framework vrei pentru a alimenta aceste funcții.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple


# ==========================
#  ENUMURI ȘI MODELE DE DATE
# ==========================

class Regime(str, Enum):
    A = "A"           # Bull coerent early-cycle
    B = "B"           # Bull matur
    C = "C"           # Distribuție / late-cycle
    D = "D"           # Bear / re-acumulare
    NEUTRU = "NEUTRU"


class Bias(str, Enum):
    LONG_ONLY = "LONG_ONLY"
    SHORT_TACTIC = "SHORT_TACTIC"
    LONG_SELECTIVE = "LONG_SELECTIVE"
    DEFENSIVE = "DEFENSIVE"
    WAIT = "WAIT"


class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    side: TradeSide
    entry_price: float
    stop_loss: float
    size: float
    tp1_hit: bool = False
    tp2_hit: bool = False
    is_closed: bool = False

    # pentru trailing: ultimul HL/LH structural
    last_structural_level: Optional[float] = None


# ==========================
#  STRATEGY CLASS
# ==========================

@dataclass
class BTCSwingStrategy:
    account_equity: float
    risk_per_trade: float = 0.01   # 1% risc pe trade
    max_risk: float = 0.05         # 5% risc total
    regime: Regime = Regime.NEUTRU
    bias: Bias = Bias.WAIT
    open_trades: List[Trade] = field(default_factory=list)

    # ------------- MACRO / REGIM ----------------

    def calc_regim_macro(self, CT: float, CC: float, CS: float, CN: float) -> Regime:
        """
        Logica de clasificare în regimuri, pe baza indicilor coezivi.
        Pragurile numerice sunt doar exemple; ajustează după nevoile tale.
        """

        # Exemplu de praguri: customizează-le
        ct_high = 0.7
        ct_mid = 0.4
        cc_low = 0.3
        cc_high = 0.7
        cs_good = 0.6
        cn_high = 0.7
        cn_very_high = 0.85

        # Regim A: CT mare, CC mic, CS bun, CN moderat
        if CT >= ct_high and CC <= cc_low and CS >= cs_good and CN < cn_high:
            return Regime.A

        # Regim B: CT mare/mediu, CC mediu-mare, CS bun, CN mare
        if CT >= ct_mid and cc_low < CC <= cc_high and CS >= cs_good and CN >= cn_high:
            return Regime.B

        # Regim C: CT scade, CC mare, CN foarte mare, CS începe să slăbească
        # Aici presupunem că:
        # - CT între ct_mid și ct_high, dar în scădere (info externă)
        # - CC > cc_high
        # - CN foarte mare
        # - CS ceva mai slab (exemplu)
        if CC > cc_high and CN >= cn_very_high:
            return Regime.C

        # Regim D: CT mic, CN mic/confuz, CS se stabilizează/crește
        if CT < ct_mid and CN < cn_high and CS >= cs_good:
            return Regime.D

        return Regime.NEUTRU

    def calc_bias_directional(
        self,
        regime: Regime,
        uptrend_1d: bool,
        downtrend_1d: bool,
        golden_cross: bool,
        death_cross: bool,
    ) -> Bias:
        """
        Determină bias-ul direcțional în funcție de regim și de structura 1D + MA50/MA200.
        """

        if regime in (Regime.A, Regime.B):
            if uptrend_1d and golden_cross:
                return Bias.LONG_ONLY
            else:
                return Bias.WAIT

        if regime == Regime.C:
            if downtrend_1d or not uptrend_1d:
                return Bias.SHORT_TACTIC
            else:
                return Bias.DEFENSIVE

        if regime == Regime.D:
            if downtrend_1d:
                return Bias.LONG_SELECTIVE
            else:
                return Bias.WAIT

        return Bias.WAIT

    # ------------- DETECTARE SETUP-URI ----------------

    def detect_long_setup(
        self,
        impuls_up: bool,
        retracement_percent: float,
        orderly_pullback: bool,
        breakout: bool,
        entry_price: float,
        pullback_low: float,
        buffer: float,
    ) -> Optional[Trade]:
        """
        Detectează setup de intrare LONG.
        Parametrii (impuls_up, retracement_percent, orderly_pullback, breakout etc.)
        ar trebui calculați din datele 4h/1D (OHLC + volum).
        """

        # 1. Impuls valid?
        if not impuls_up:
            return None

        # 2. Retragere ordonată, în interval rezonabil
        if retracement_percent < 5 or retracement_percent > 40:
            return None

        if not orderly_pullback:
            return None

        # 3. Reconfirmare prin breakout
        if not breakout:
            return None

        stop_loss = pullback_low - buffer

        # size se calculează ulterior cu calc_position_size
        trade = Trade(
            side=TradeSide.LONG,
            entry_price=entry_price,
            stop_loss=stop_loss,
            size=0.0,  # se completează după calc_position_size
        )
        return trade

    def detect_short_setup(
        self,
        impuls_down: bool,
        retracement_percent_up: float,
        has_lower_high: bool,
        breakdown: bool,
        entry_price: float,
        recent_swing_high: float,
        buffer: float,
    ) -> Optional[Trade]:
        """
        Detectează setup de intrare SHORT (Regim C).
        """

        if not impuls_down:
            return None

        if retracement_percent_up < 5 or retracement_percent_up > 40:
            return None

        if not has_lower_high:
            return None

        if not breakdown:
            return None

        stop_loss = recent_swing_high + buffer

        trade = Trade(
            side=TradeSide.SHORT,
            entry_price=entry_price,
            stop_loss=stop_loss,
            size=0.0,
        )
        return trade

    # ------------- POSITION SIZING ----------------

    def calc_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        regime: Regime,
    ) -> float:
        """
        Calculează mărimea poziției în funcție de risc și de regim.
        """

        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0:
            return 0.0

        risk_amount = self.risk_per_trade * self.account_equity
        base_size = risk_amount / stop_distance

        # multipliers în funcție de regim
        if regime == Regime.A:
            multiplier = 1.0
        elif regime == Regime.B:
            multiplier = 0.7
        elif regime == Regime.C:
            multiplier = 0.4
        elif regime == Regime.D:
            multiplier = 0.5
        else:
            multiplier = 0.0

        position_size = base_size * multiplier
        return max(position_size, 0.0)

    def total_risk_exposed(self) -> float:
        """
        Estimează riscul total curent (foarte simplificat).
        În practică, poți considera riscul = sum(risc_per_trade) pentru fiecare poziție.
        Aici îl aproximăm ca nr. de trades * risk_per_trade.
        """
        return len([t for t in self.open_trades if not t.is_closed]) * self.risk_per_trade

    # ------------- MANAGEMENTUL POZIȚIILOR ----------------

    def manage_long_trade(
        self,
        trade: Trade,
        current_price: float,
        last_higher_low: Optional[float],
        buffer: float,
    ) -> None:
        """
        Management pentru poziții LONG:
        - TP1 la R>=1.5
        - TP2 la R>=3
        - trailing SL sub ultimele higher lows
        """

        if trade.is_closed:
            return

        r_per_trade = (trade.entry_price - trade.stop_loss)
        if r_per_trade == 0:
            return

        R = (current_price - trade.entry_price) / r_per_trade

        # TP1
        if (not trade.tp1_hit) and R >= 1.5:
            self.close_partial(trade, 0.30)
            trade.tp1_hit = True

        # TP2
        if (not trade.tp2_hit) and R >= 3.0:
            self.close_partial(trade, 0.30)
            trade.tp2_hit = True

        # Trailing stop pe HL
        if last_higher_low is not None:
            new_trailing_sl = last_higher_low - buffer
            if new_trailing_sl > trade.stop_loss:
                trade.stop_loss = new_trailing_sl

        # Stop-out
        if current_price <= trade.stop_loss:
            self.close_trade(trade)

    def manage_short_trade(
        self,
        trade: Trade,
        current_price: float,
        last_lower_high: Optional[float],
        buffer: float,
    ) -> None:
        """
        Management pentru poziții SHORT:
        - TP1 la R>=1.5
        - TP2 la R>=3
        - trailing SL peste ultimele lower highs
        """

        if trade.is_closed:
            return

        r_per_trade = (trade.stop_loss - trade.entry_price)
        if r_per_trade == 0:
            return

        R = (trade.entry_price - current_price) / r_per_trade

        # TP1
        if (not trade.tp1_hit) and R >= 1.5:
            self.close_partial(trade, 0.30)
            trade.tp1_hit = True

        # TP2
        if (not trade.tp2_hit) and R >= 3.0:
            self.close_partial(trade, 0.30)
            trade.tp2_hit = True

        # Trailing stop pe LH
        if last_lower_high is not None:
            new_trailing_sl = last_lower_high + buffer
            if new_trailing_sl < trade.stop_loss:
                trade.stop_loss = new_trailing_sl

        # Stop-out
        if current_price >= trade.stop_loss:
            self.close_trade(trade)

    # ------------- HELPERI PENTRU EXECUȚIE ----------------

    def open_trade(self, trade: Trade) -> None:
        """
        Adaugă trade-ul în lista de poziții deschise.
        În practică: aici trimiți și ordinul către exchange/broker.
        """
        if self.total_risk_exposed() >= self.max_risk:
            # Nu mai deschide noi poziții dacă ai atins riscul maxim
            return
        self.open_trades.append(trade)

    def close_trade(self, trade: Trade) -> None:
        """
        Închide complet o poziție.
        În practică: trimite ordin de închidere.
        """
        trade.is_closed = True
        trade.size = 0.0

    def close_partial(self, trade: Trade, fraction: float) -> None:
        """
        Închidere parțială a poziției (ex. 30%).
        """
        if trade.is_closed:
            return
        if fraction <= 0:
            return
        trade.size *= max(0.0, (1.0 - fraction))
        if trade.size <= 0:
            self.close_trade(trade)

    # ------------- BUCLE EXTERNE (EXEMPLU) ----------------

    def daily_update_macro(
        self,
        CT: float,
        CC: float,
        CS: float,
        CN: float,
        uptrend_1d: bool,
        downtrend_1d: bool,
        golden_cross: bool,
        death_cross: bool,
    ) -> None:
        """
        Exemplu de apel zilnic / săptămânal pentru recalcularea regimului și bias-ului.
        """

        self.regime = self.calc_regim_macro(CT, CC, CS, CN)
        self.bias = self.calc_bias_directional(
            self.regime, uptrend_1d, downtrend_1d, golden_cross, death_cross
        )

    def on_new_bar(
        self,
        current_price: float,
        long_signal_data: dict,
        short_signal_data: dict,
        last_higher_low: Optional[float],
        last_lower_high: Optional[float],
        buffer: float,
    ) -> None:
        """
        Exemplu de funcție chemată la fiecare lumânare nouă (4h/1D).
        - current_price: prețul curent
        - long_signal_data: dict cu info relevante pentru detect_long_setup
        - short_signal_data: dict cu info relevante pentru detect_short_setup
        - last_higher_low / last_lower_high: nivele structurale calculate separat
        """

        # 1. Management poziții existente
        for trade in self.open_trades:
            if trade.is_closed:
                continue

            if trade.side == TradeSide.LONG:
                self.manage_long_trade(trade, current_price, last_higher_low, buffer)
            else:
                self.manage_short_trade(trade, current_price, last_lower_high, buffer)

        # 2. Căutare noi setup-uri, dacă încă nu ai depășit MaxRisk
        if self.total_risk_exposed() >= self.max_risk:
            return

        # LONG setups
        if self.bias in (Bias.LONG_ONLY, Bias.LONG_SELECTIVE):
            trade = self.detect_long_setup(
                impuls_up=long_signal_data.get("impuls_up", False),
                retracement_percent=long_signal_data.get("retracement_percent", 0.0),
                orderly_pullback=long_signal_data.get("orderly_pullback", False),
                breakout=long_signal_data.get("breakout", False),
                entry_price=long_signal_data.get("entry_price", current_price),
                pullback_low=long_signal_data.get("pullback_low", current_price),
                buffer=buffer,
            )
            if trade is not None:
                trade.size = self.calc_position_size(trade.entry_price, trade.stop_loss, self.regime)
                if trade.size > 0:
                    self.open_trade(trade)

        # SHORT setups
        if self.bias == Bias.SHORT_TACTIC:
            trade = self.detect_short_setup(
                impuls_down=short_signal_data.get("impuls_down", False),
                retracement_percent_up=short_signal_data.get("retracement_percent_up", 0.0),
                has_lower_high=short_signal_data.get("has_lower_high", False),
                breakdown=short_signal_data.get("breakdown", False),
                entry_price=short_signal_data.get("entry_price", current_price),
                recent_swing_high=short_signal_data.get("recent_swing_high", current_price),
                buffer=buffer,
            )
            if trade is not None:
                trade.size = self.calc_position_size(trade.entry_price, trade.stop_loss, self.regime)
                if trade.size > 0:
                    self.open_trade(trade)
