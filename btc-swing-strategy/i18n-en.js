/* CohesivX BTC — professional RO/EN language + accessibility module */
(function () {
  "use strict";

  const LANG_KEY = "coeziv_btc_lang";
  const SCALE_KEY = "coeziv_btc_scale";
  const SCALE_STEPS = [1, 1.15, 1.30];
  const USD = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
  let collapseTimer = null;
  let refreshTimer = null;

  const I18N = {
    ro: {
      appTitle: "MECANISM COEZIV BTC",
      snapshotLine: "Actualizat aproximativ la 24h.",
      livePrice: "Preț live:",
      dailyTitle: "INTERPRETARE COEZIVĂ ZILNICĂ",
      dailySubtitle: "Explicație naturală a stării structurale observate de mecanism.",
      dailyLabels: ["Participare", "Fereastră risc", "Fear & Greed / Regim", "Ce urmărim"],
      participationTitle: "COEZIUNE PARTICIPATIVĂ",
      participationSubtitle: "Starea interesului participanților în ecosistem.",
      participationMetricLabels: ["Flux", "Lichiditate", "Istoric mediu", "Minim istoric"],
      riskTitle: "FEREASTRĂ DE RISC",
      riskSubtitle: "Context istoric derivat din backtestul coeziv.",
      riskMetricLabels: ["Confirmare istorică", "Timp median", "Regim curent", "Persistență"],
      fgTitle: "INDICE FEAR & GREED COEZIV",
      fgSubtitle: "Sentiment de piață derivat din mecanismul coeziv (structură + semnal + volatilitate), fără date sociale sau externe.",
      fgLabels: ["Structural", "Tactic (24h)", "Tensiune"],
      historyTitle: "ISTORIC CONTEXTE MECANISM",
      historySubtitle: "Ultimele snapshot-uri ale mecanismului. Nu sunt semnale de intrare automată, ci context.",
      regimesTitle: "REGIMURI STANDARD DE PIAȚĂ",
      regimesSubtitle: "Mecanismul traduce structura internă a pieței (regim + flux) în câteva contexte standard. Eticheta de sus, „Regim de piață”, va fi întotdeauna una dintre situațiile de mai jos.",
      viewFull: "Vezi interpretarea completă",
      hideFull: "Ascunde interpretarea completă"
    },
    en: {
      appTitle: "COHESIVE BTC MECHANISM",
      snapshotLine: "Mechanism snapshot: updated approximately every 24h.",
      livePrice: "Live price:",
      dailyTitle: "DAILY COHESIVE INTERPRETATION",
      dailySubtitle: "Natural-language explanation of the structural state observed by the mechanism.",
      dailyLabels: ["Participation", "Risk window", "Fear & Greed / Regime", "What to watch"],
      participationTitle: "PARTICIPATION COHESION",
      participationSubtitle: "State of participant interest across the ecosystem.",
      participationMetricLabels: ["Flow", "Liquidity", "Historical average", "Historical minimum"],
      riskTitle: "RISK WINDOW",
      riskSubtitle: "Historical context derived from the cohesive backtest.",
      riskMetricLabels: ["Historical confirmation", "Median time", "Current regime", "Persistence"],
      fgTitle: "COHESIVE FEAR & GREED INDEX",
      fgSubtitle: "Market sentiment derived from the cohesive mechanism: structure, signal and volatility, without social or external data.",
      fgLabels: ["Structural", "Tactical (24h)", "Tension"],
      historyTitle: "MECHANISM CONTEXT HISTORY",
      historySubtitle: "Recent mechanism snapshots. These are not automatic entry signals, but structural context.",
      regimesTitle: "STANDARD MARKET REGIMES",
      regimesSubtitle: "The mechanism translates the market’s internal structure, regime and flow into a small set of standard contexts. The top regime label will always match one of the situations below.",
      viewFull: "View full interpretation",
      hideFull: "Hide full interpretation"
    }
  };

  const STANDARD_REGIMES = {
    up_trend_strong: ["1. Trend ascendent susținut", "1. Sustained upward trend", "Cumpărătorii domină clar piața, iar corecțiile tind să fie rapide și superficiale. Context tipic pentru faze de expansiune, cu risc crescut de supra-extindere pe termen scurt.", "Buyers clearly dominate the market, while pullbacks tend to be quick and shallow. This is typical of expansion phases, with increased short-term overextension risk."],
    up_trend_moderate: ["2. Trend ascendent moderat", "2. Moderate upward trend", "Trend pozitiv, dar cu corecții regulate. Presiunea de creștere există, însă mișcările sunt mai echilibrate între impulsuri și pullback-uri.", "Positive trend, but with regular corrections. Upward pressure exists, while movement is more balanced between impulses and pullbacks."],
    range_bias_up: ["3. Range cu bias pozitiv", "3. Range with positive bias", "Piața consolidează într-un interval, dar maximele sunt împinse treptat în sus. Context favorabil pentru acumulare graduală, nu pentru intrări agresive.", "The market consolidates within a range, but highs are gradually pushed upward. This supports gradual accumulation rather than aggressive entries."],
    range_neutral: ["4. Range neutru", "4. Neutral range", "Echilibru relativ între cumpărători și vânzători, fără direcție clară. Mișcările laterale domină, iar semnalele direcționale au robustețe redusă.", "Relative balance between buyers and sellers, with no clear direction. Sideways movement dominates, and directional signals have reduced robustness."],
    range_bias_down: ["5. Range cu bias negativ", "5. Range with negative bias", "Piața consolidează, dar minimele sunt coborâte treptat. Context în care presiunea de vânzare se acumulează, iar raliurile sunt fragile.", "The market is consolidating, but lows are gradually pushed lower. Selling pressure is accumulating, and rallies remain fragile."],
    down_trend_moderate: ["6. Trend descendent moderat", "6. Moderate downward trend", "Presiune de scădere prezentă, însă cu pauze și contra-mișcări regulate. Raliurile tind să fie folosite pentru reducerea expunerii, nu pentru acumulare.", "Downward pressure is present, but with pauses and regular counter-moves. Rallies tend to be used for reducing exposure rather than accumulation."],
    down_trend_strong: ["7. Trend descendent susținut", "7. Sustained downward trend", "Vânzătorii controlează clar piața, iar bounce-urile sunt de obicei scurte și fragile. Context specific fazelor de capitulare sau de descărcare forțată de poziții.", "Sellers clearly control the market, and bounces are usually short and fragile. This is typical of capitulation or forced-position unwinding phases."],
    transition_accumulation: ["8. Regim neutru / tranziție", "8. Neutral / transition regime", "Zone de schimbare, în care modelul nu vede încă un avantaj structural clar nici pentru cumpărători, nici pentru vânzători. De obicei premerg fazelor în care piața alege o nouă direcție: trend sau range.", "Change zones where the model does not yet see a clear structural advantage for buyers or sellers. They often precede phases where the market chooses a new direction: trend or range."]
  };

  function lang() { return localStorage.getItem(LANG_KEY) === "en" ? "en" : "ro"; }
  function en() { return lang() === "en"; }
  function copy() { return I18N[lang()]; }
  function setEl(el, text) { if (el && typeof text === "string") el.textContent = text; }
  function setText(id, text) { setEl(document.getElementById(id), text); }
  function num(v, d = 0) { return Number.isFinite(Number(v)) ? Number(v).toFixed(d) : "n/a"; }
  function pct(v, d = 1) { return Number.isFinite(Number(v)) ? (Number(v) * 100).toFixed(d) : "n/a"; }
  function usd(v) { return Number.isFinite(Number(v)) ? USD.format(Number(v)) : "n/a"; }

  function fetchJson(url) {
    return fetch(`${url}?t=${Date.now()}`, { cache: "no-store" }).then(r => {
      if (!r.ok) throw new Error(url);
      return r.json();
    });
  }

  function signalLabel(signal) {
    const s = String(signal || "flat").toLowerCase();
    if (en()) return s === "long" ? "Context: upward pressure" : s === "short" ? "Context: downside risk" : "Context: neutral";
    return s === "long" ? "Context: presiune de creștere" : s === "short" ? "Context: risc de scădere" : "Context: neutru";
  }

  function signalShort(signal) {
    const s = String(signal || "flat").toLowerCase();
    if (en()) return s === "long" ? "Upward pressure" : s === "short" ? "Downside risk" : "Neutral context";
    return s === "long" ? "Presiune de creștere" : s === "short" ? "Risc de scădere" : "Context neutru";
  }

  function regimeCompact(fullText, code) {
    const t = `${fullText || ""} ${code || ""}`.toLowerCase();
    if (t.includes("range") && (t.includes("pozitiv") || t.includes("bias_up"))) return "RANGE+";
    if (t.includes("range") && (t.includes("negativ") || t.includes("bias_down"))) return "RANGE−";
    if (t.includes("range")) return "RANGE";
    if (t.includes("ascendent") || t.includes("up_trend")) return "TREND+";
    if (t.includes("descendent") || t.includes("down_trend")) return "TREND−";
    if (t.includes("tranzi") || t.includes("transition")) return "TRANZ";
    if (t.includes("neutru") || t.includes("neutral")) return "NEUTRU";
    if (t.includes("n/a")) return "N/A";
    return "REGIM";
  }

  function flowText(bias, strength) {
    const s = strength === "slab" ? "weak" : strength === "puternică" || strength === "puternic" ? "strong" : strength || "";
    if (en()) {
      if (bias === "pozitiv") return `Market flow: tilted toward buying${s ? ` (${s})` : ""}.`;
      if (bias === "negativ") return `Market flow: tilted toward selling${s ? ` (${s})` : ""}.`;
      if (bias === "neutru") return `Market flow: relatively balanced between buyers and sellers${s ? ` (${s})` : ""}.`;
      return "Market flow: not available yet.";
    }
    let text = "Flux de piață: n/a (așteptăm date despre flux).";
    if (bias === "pozitiv") text = "Flux de piață: orientat spre cumpărare";
    else if (bias === "negativ") text = "Flux de piață: orientat spre vânzare";
    else if (bias === "neutru") text = "Flux de piață: relativ echilibrat între cumpărători și vânzători";
    return bias && strength ? `${text} (${strength}).` : bias ? `${text}.` : text;
  }

  function liquidityText(regime, strength) {
    const r = String(regime || "").trim().toLowerCase();
    const s = strength === "puternică" || strength === "puternic" ? "strong" : strength === "slab" ? "weak" : strength || "";
    if (en()) {
      if (r === "ridicată") return `Market liquidity: high; movement is above usual levels${s ? `, with a ${s} deviation from normal activity` : ""}.`;
      if (r === "scăzută") return `Market liquidity: low; movement is below usual levels${s ? `, with a ${s} deviation from normal activity` : ""}.`;
      if (r === "normală" || r === "moderată") return "Market liquidity: moderate for this context.";
      return "Market liquidity: not available yet.";
    }
    let text = "Lichiditate piață: n/a (așteptăm date suplimentare)";
    if (r === "ridicată") text = "Lichiditate piață: ridicată; mișcări peste nivelurile obișnuite.";
    else if (r === "scăzută") text = "Lichiditate piață: scăzută; mișcări sub nivelurile obișnuite.";
    else if (r === "normală" || r === "moderată") text = "Lichiditate piață: moderată pentru acest context.";
    if (regime && strength && r !== "normală" && r !== "moderată") text += ` (deviație ${strength} față de nivelurile obișnuite.)`;
    return text;
  }

  function renderEnergy(costs) {
    const el = document.getElementById("prod-cost-line");
    if (!el) return;
    const t = en();
    const c = costs || {};
    const fmt = v => Number.isFinite(Number(v)) ? `~${usd(v)} USD` : "n/a";
    el.className = "energy-card" + (!costs ? " loading" : "");
    el.innerHTML = `<div class="energy-card-header"><div><div class="energy-title">${t ? "BTC ENERGY THRESHOLD" : "Prag energetic BTC"}</div><div class="energy-subtitle">${t ? "Estimated energy cost of production" : "Cost energetic estimat de producție"}</div></div><div class="energy-badge">${t ? "estimate" : "estimare"}</div></div><div class="energy-values"><div class="energy-item"><span class="energy-label">${t ? "Efficient miner" : "Miner eficient"}</span><span class="energy-value">${fmt(c.cheap)}</span></div><div class="energy-item main"><span class="energy-label">${t ? "Average miner" : "Miner mediu"}</span><span class="energy-value">${fmt(c.average)}</span></div><div class="energy-item"><span class="energy-label">${t ? "Expensive miner" : "Miner scump"}</span><span class="energy-value">${fmt(c.expensive)}</span></div></div><div class="energy-note">${t ? "Calculation based on network difficulty, block reward, equipment efficiency and energy price. It does not represent miners’ full accounting cost." : "Calcul bazat pe dificultatea rețelei, recompensa pe bloc, eficiența echipamentelor și prețul energiei. Nu reprezintă costul total contabil al minerilor."}</div>`;
  }

  function riskRegime(regime) {
    const r = String(regime || "").toLowerCase();
    const ro = { bear_struct:"Degradare structurală", bear_late:"Degradare avansată", accum_bear:"Acumulare fragilă", bull_struct:"Structură pozitivă", bull_late:"Expansiune matură", accum_bull:"Acumulare pozitivă", range_pos:"Range cu bias pozitiv", range_neg:"Range cu bias negativ", range_neutral:"Range neutru", neutral:"Tranziție neutră" };
    const enm = { bear_struct:"Structural degradation", bear_late:"Advanced degradation", accum_bear:"Fragile accumulation", bull_struct:"Positive structure", bull_late:"Mature expansion", accum_bull:"Positive accumulation", range_pos:"Range with positive bias", range_neg:"Range with negative bias", range_neutral:"Neutral range", neutral:"Neutral transition" };
    return (en() ? enm[r] : ro[r]) || regime || "n/a";
  }

  function riskLevel(level, active) {
    if (en()) {
      if (!active) return "Normal window";
      if (level === "high") return "High structural risk";
      if (level === "moderate") return "Moderate structural risk";
      if (level === "low") return "Low structural risk";
      return "Structural risk n/a";
    }
    if (!active) return "Fereastră normală";
    if (level === "high") return "Risc structural ridicat";
    if (level === "moderate") return "Risc structural moderat";
    if (level === "low") return "Risc structural scăzut";
    return "Risc structural n/a";
  }

  function fgZone(zone) {
    return ({ extreme_fear:"Extreme Fear", fear:"Fear", neutral:"Neutral", greed:"Greed", extreme_greed:"Extreme Greed" }[String(zone || "neutral").toLowerCase()] || "Neutral");
  }

  function fgDescription(fg) {
    if (!fg) return en() ? "The indicator will be populated automatically after the next mechanism update." : "Indicatorul va fi populat automat după următorul update al mecanismului.";
    const z = String(fg.combined_zone || "neutral").toLowerCase();
    const tension = Number.isFinite(Number(fg.tension)) ? Number(fg.tension) : null;
    if (en()) {
      let base = z === "extreme_fear" ? "Panic or structural capitulation; elevated risk of fast and disorderly movement." : z === "fear" ? "Risk aversion and defensive positioning; useful context for prudence and position-size management." : z === "greed" ? "High confidence and greater risk-taking; supportive of continuation, but with attention to exhaustion." : z === "extreme_greed" ? "Euphoria and possible overextension; historically, such zones often precede cooling or correction phases." : "Relative balance between risk and opportunity; the mechanism does not see a decisive emotional advantage.";
      if (tension === null) return base;
      return base + (tension < 15 ? " Structure and emotion are well aligned." : tension < 30 ? " There is moderate tension between structure and market emotion." : " Strong misalignment between structure and emotion; the context is sensitive to regime change.");
    }
    let base = z === "extreme_fear" ? "Panică sau capitulare structurală; risc ridicat de mișcări rapide și dezordonate." : z === "fear" ? "Aversiune la risc și poziționare defensivă; context util pentru prudență și management al dimensiunii pozițiilor." : z === "greed" ? "Încredere ridicată și asumare mai mare de risc; context favorabil continuării trendului, dar cu atenție la epuizare." : z === "extreme_greed" ? "Euforie și potențial de supra-extindere; istoric, astfel de zone preced frecvent faze de răcire sau corecție." : "Echilibru relativ între risc și oportunitate; mecanismul nu vede un avantaj emoțional decisiv.";
    if (tension === null) return base;
    return base + (tension < 15 ? " Aliniere bună între structură și emoție." : tension < 30 ? " Există o tensiune moderată între structură și emoția de piață." : " Dezaliniere puternică între structură și emoție; context sensibil la schimbări de regim.");
  }

  async function renderState() {
    try {
      const state = await fetchJson("coeziv_state.json");
      const signal = String(state.signal || "flat").toLowerCase();
      const drift = state.signal_expected_drift || {};
      const bd = state.signal_prob_breakdown || {};
      const horizon = state.signal_prob_horizon_hours || 24;
      const regime = state.market_regime || {};
      const regimeLabel = typeof regime === "string" ? regime : regime.label || "n/a";
      const regimeCode = typeof regime === "object" ? regime.code || "" : "";

      setText("signal-label", signalLabel(signal));
      const regEl = document.getElementById("regime-line");
      if (regEl) {
        const full = en() ? `Market regime: ${regimeLabel}.` : `Regim de piață: ${regimeLabel}.`;
        regEl.setAttribute("data-full-regime", full);
        regEl.title = full;
        regEl.textContent = regimeCompact(full, regimeCode);
      }

      if (en()) {
        setText("message", `At the mechanism update, BTC was trading around ${usd(state.price_usd)} USD. The mechanism reads the current context as neutral. Market flow is relatively balanced and weak. Liquidity is high, so price movement is more easily absorbed. Price remains close to the network’s estimated production-cost equilibrium. In similar historical contexts, the next ~24 hours typically ranged from about ${pct(drift.p10)}% in weaker scenarios to ${pct(drift.p90)}% in stronger scenarios, with a median near ${pct(drift.p50)}%. This is a structural risk interpretation, not financial advice.`);
        setText("signal-prob", `Historical probability: about ${pct(state.signal_probability, 0)}% for price to move with the mechanism’s signal over the next ${horizon}h, based on ${state.signal_prob_samples || 0} similar historical contexts.`);
        setText("signal-prob-breakdown", `Historical distribution over ${horizon}h: about ${pct(bd.in_direction, 0)}% with the signal, ${pct(bd.opposite, 0)}% against it, and ${pct(bd.flat, 0)}% neutral movement or noise.`);
        setText("drift-range", `Typical movement range (${horizon}h): between ~${pct(drift.p10)}% and ~${pct(drift.p90)}%, with a median near ~${pct(drift.p50)}%, based on similar historical contexts.`);
      } else {
        setText("message", state.message || "Nu există mesaj coeziv pentru acest moment.");
        setText("signal-prob", `Probabilitate istorică: ~${pct(state.signal_probability, 0)}% ca prețul să se miște în direcția semnalului în următoarele ${horizon}h (bazat pe ${state.signal_prob_samples || 0} situații similare).`);
        setText("signal-prob-breakdown", `Distribuție istorică (în ${horizon}h): ~${pct(bd.in_direction, 0)}% în direcția semnalului, ~${pct(bd.opposite, 0)}% contra semnalului și ~${pct(bd.flat, 0)}% mișcare neutră / zgomot.`);
        setText("drift-range", `Interval tipic de mișcare (${horizon}h): între ~${pct(drift.p10)}% și ~${pct(drift.p90)}% (mediană ~${pct(drift.p50)}%), bazat pe contexte similare din istoric.`);
      }
      setText("flow-line", flowText(state.flow_bias, state.flow_strength));
      setText("liquidity-line", liquidityText(state.liquidity_regime, state.liquidity_strength));
      renderEnergy(state.production_costs_usd || null);
      renderFearGreed(state.fg || null);
      renderHistory(state);
    } catch (_) {}
  }

  async function renderDaily() {
    try {
      const data = await fetchJson("daily_cohesiv_interpretation.json");
      const t = en();
      setText("daily-ai-state", t ? data.general_state_en || data.general_state || "structural state pending" : data.general_state || "stare în curs de interpretare");
      setText("daily-ai-summary", t ? data.plain_language_en || data.summary_en || data.plain_language || data.summary || "Daily interpretation pending." : data.plain_language || data.summary || "Interpretarea nu este încă disponibilă.");
      setText("daily-ai-participation", t ? data.participation_en || data.participation || "–" : data.participation || "–");
      setText("daily-ai-risk", t ? data.risk_window_en || data.risk_window || "–" : data.risk_window || "–");
      setText("daily-ai-fg-regime", t ? [data.fear_greed_en || data.fear_greed, data.market_regime_en || data.market_regime].filter(Boolean).join(" • ") : [data.fear_greed, data.market_regime].filter(Boolean).join(" • "));
      setText("daily-ai-watch", t ? data.watch_next_en || data.watch_next || "–" : data.watch_next || "–");
      setText("daily-ai-footer", t ? data.disclaimer_en || "Experimental structural interpretation, not financial advice." : data.disclaimer || "Interpretare structurală experimentală, nu recomandare financiară.");
      document.querySelectorAll("#daily-ai-card .daily-ai-label").forEach((el, i) => setEl(el, copy().dailyLabels[i] || el.textContent));
      const panel = document.getElementById("daily-ai-full-panel");
      const btn = document.getElementById("daily-ai-full-toggle");
      const full = t ? data.full_interpretation_en || data.full_interpretation || "" : data.full_interpretation || "";
      if (panel && full) panel.textContent = full.replace(/^##\s+/gm, "").replace(/^###\s+/gm, "").replace(/\*\*(.*?)\*\*/g, "$1").trim();
      if (btn) btn.textContent = panel && panel.classList.contains("open") ? copy().hideFull : copy().viewFull;
    } catch (_) {}
  }

  async function renderParticipation() {
    try {
      const data = await fetchJson("participation_cohesion.json");
      const pill = document.getElementById("participation-cohesion-pill");
      if (pill) { pill.className = `participation-pill participation-${data.level || "tense"}`; pill.textContent = en() ? (data.level === "cohesive" ? "Cohesive participation" : "Tense participation") : ((data.label || "participare necunoscută").replace(/^./, c => c.toUpperCase())); }
      setText("participation-cohesion-score", `${num(data.score)} / 100`);
      const comps = data.components || {}, hist = data.history || {};
      setText("participation-flow", num(comps.flow_component));
      setText("participation-liquidity", num(comps.liquidity_component));
      setText("participation-history-mean", num(hist.mean_score));
      setText("participation-history-min", num(hist.min_score));
      setText("participation-cohesion-text", en() ? "Participants remain active, but behavior is tense. Interest persists, while the dominant flow may be defensive or oriented toward exit." : data.main_text || "Indicatorul de participare nu este disponibil momentan.");
      setText("participation-cohesion-footer", en() ? "Experimental indicator derived from flow, liquidity, persistence and tension against the model." : data.footer || "Indicator experimental, nu recomandare de tranzacționare.");
      document.querySelectorAll("#participation-cohesion-card .participation-metric-label").forEach((el, i) => setEl(el, copy().participationMetricLabels[i] || el.textContent));
    } catch (_) {}
  }

  async function renderRisk() {
    try {
      const data = await fetchJson("risk_window.json");
      const active = !!data.active;
      const level = data.level || (active ? "unknown" : "normal");
      const pill = document.getElementById("risk-window-pill");
      if (pill) { pill.className = `risk-pill risk-${active ? level : "normal"}`; pill.textContent = riskLevel(level, active); }
      setText("risk-window-rate", `${num((data.historical_confirmation_rate || 0) * 100)}%`);
      setText("risk-window-days", en() ? `~${num(data.median_days_to_confirmation)} days` : `~${num(data.median_days_to_confirmation)} zile`);
      setText("risk-window-regime", riskRegime(data.current_regime || "n/a"));
      setText("risk-window-streak", en() ? `${data.consecutive_degradation_days || 0} days` : `${data.consecutive_degradation_days || 0} zile`);
      setText("risk-window-text", en() ? `The current context belongs to a historical family that produced major weakening in about ${num((data.historical_confirmation_rate || 0) * 100)}% of cases. When those cases confirmed, confirmation appeared after roughly ${num(data.median_days_to_confirmation)} days on average.` : data.main_text || "Fereastra de risc nu este disponibilă momentan.");
      setText("risk-window-footer", en() ? "Statistical interpretation of structural degradation, not a trading recommendation." : data.footer || "Interpretare statistică, nu recomandare de tranzacționare.");
      document.querySelectorAll("#risk-window-card .risk-metric-label").forEach((el, i) => setEl(el, copy().riskMetricLabels[i] || el.textContent));
    } catch (_) {}
  }

  function renderFearGreed(fg) {
    if (!fg) return;
    setText("fg-score-label", en() ? "Combined index: structure + tactical state" : "Indice combinat (structură + tactică)");
    setText("fg-score-zone", fgZone(fg.combined_zone));
    setText("fg-description", fgDescription(fg));
    document.querySelectorAll(".fg-detail-label").forEach((el, i) => setEl(el, copy().fgLabels[i] || el.textContent));
  }

  function renderHistory(state) {
    document.querySelectorAll("#history-list .history-row").forEach(row => {
      const chip = row.querySelector(".history-signal-chip");
      const tag = row.querySelector(".history-tag");
      if (chip) {
        const cls = chip.className || "";
        const sig = cls.includes("history-signal-long") ? "long" : cls.includes("history-signal-short") ? "short" : "flat";
        const nodes = Array.from(chip.childNodes).filter(n => n.nodeType === Node.TEXT_NODE);
        if (nodes.length) nodes[nodes.length - 1].nodeValue = " " + signalShort(sig);
      }
      if (tag && /Preț model|Model price/.test(tag.textContent)) tag.textContent = en() ? "Model price at snapshot" : "Preț model la snapshot";
    });
    const meta = document.getElementById("history-meta");
    if (!meta || !state) return;
    const h = Array.isArray(state.signal_history) ? state.signal_history : [];
    const up = h.filter(x => x.signal === "long").length;
    const down = h.filter(x => x.signal === "short").length;
    const flat = h.filter(x => x.signal === "flat").length;
    const top = en() ? `<span>Total contexts: ${h.length}</span><span class="history-meta-dot">•</span><span>Upward pressure: ${up}</span><span class="history-meta-dot">•</span><span>Downside risk: ${down}</span><span class="history-meta-dot">•</span><span>Neutral context: ${flat}</span>` : `<span>Total contexte: ${h.length}</span><span class="history-meta-dot">•</span><span>Presiune de creștere: ${up}</span><span class="history-meta-dot">•</span><span>Risc de scădere: ${down}</span><span class="history-meta-dot">•</span><span>Context neutru: ${flat}</span>`;
    const line = [flowText(state.flow_bias, "").replace(en() ? "Market flow:" : "Flux de piață:", en() ? "Current flow:" : "Flux actual:"), liquidityText(state.liquidity_regime, "")].filter(Boolean).join(" ");
    meta.innerHTML = `<div>${top}</div>${line ? `<div>${line}</div>` : ""}`;
  }

  function renderStandardRegimes() {
    Object.entries(STANDARD_REGIMES).forEach(([key, copyData]) => {
      const row = document.querySelector(`[data-standard-regime="${key}"]`);
      if (!row) return;
      setEl(row.querySelector(".history-date"), en() ? copyData[1] : copyData[0]);
      setEl(row.querySelector(".history-tag"), en() ? copyData[3] : copyData[2]);
    });
  }

  function renderHeadings() {
    const c = copy();
    setEl(document.querySelector(".title-bar"), c.appTitle);
    setEl(document.querySelector(".asset-text span"), c.snapshotLine);
    setEl(document.querySelector(".live-label"), c.livePrice);
    setEl(document.querySelector("#daily-ai-card .daily-ai-title"), c.dailyTitle);
    setEl(document.querySelector("#daily-ai-card .daily-ai-subtitle"), c.dailySubtitle);
    setEl(document.querySelector("#participation-cohesion-card .participation-title"), c.participationTitle);
    setEl(document.querySelector("#participation-cohesion-card .participation-subtitle"), c.participationSubtitle);
    setEl(document.querySelector("#risk-window-card .risk-title"), c.riskTitle);
    setEl(document.querySelector("#risk-window-card .risk-subtitle"), c.riskSubtitle);
    setEl(document.querySelector(".fg-title"), c.fgTitle);
    setEl(document.querySelector(".fg-subtitle"), c.fgSubtitle);
    const headers = Array.from(document.querySelectorAll(".history-header"));
    const history = headers.find(h => /ISTORIC|HISTORY|CONTEXT HISTORY/.test(h.textContent));
    const regimes = headers.find(h => /REGIMURI|STANDARD MARKET/.test(h.textContent));
    if (history) { setEl(history.querySelector(".history-title"), c.historyTitle); setEl(history.querySelector(".history-subtitle"), c.historySubtitle); }
    if (regimes) { setEl(regimes.querySelector(".history-title"), c.regimesTitle); setEl(regimes.querySelector(".history-subtitle"), c.regimesSubtitle); }
  }

  function injectStyles() {
    if (document.getElementById("coeziv-i18n-style")) return;
    const style = document.createElement("style");
    style.id = "coeziv-i18n-style";
    style.textContent = `
      body[data-lang="en"] .standard-regime-row.active-standard-regime::after{content:"DETECTED NOW" !important;}
      body[data-lang="ro"] .standard-regime-row.active-standard-regime::after{content:"DETECTAT ACUM" !important;}
      #coeziv-accessibility-panel{position:fixed;right:10px;bottom:14px;z-index:2147483647;display:flex;align-items:center;gap:6px;padding:6px;border-radius:999px;border:1px solid rgba(56,189,248,.42);background:linear-gradient(135deg,rgba(15,23,42,.94),rgba(12,74,110,.90));color:#e5e7eb;box-shadow:0 14px 42px rgba(0,0,0,.38),0 0 0 1px rgba(125,211,252,.12),0 0 22px rgba(56,189,248,.18);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:11px;line-height:1;transition:opacity .18s ease,transform .18s ease,box-shadow .18s ease}
      #coeziv-accessibility-panel.collapsed{padding:5px 8px;box-shadow:0 10px 30px rgba(0,0,0,.36),0 0 24px rgba(56,189,248,.26)}
      #coeziv-accessibility-panel .coeziv-compact-toggle{border:0;border-radius:999px;background:rgba(56,189,248,.18);color:#bae6fd;min-width:96px;height:30px;padding:0 10px;cursor:pointer;font-size:11px;font-weight:900;letter-spacing:.03em;line-height:30px;text-align:center;white-space:nowrap}
      #coeziv-accessibility-panel .coeziv-panel-controls{display:inline-flex;align-items:center;gap:5px}
      #coeziv-accessibility-panel.collapsed .coeziv-panel-controls{display:none}
      #coeziv-accessibility-panel button:not(.coeziv-compact-toggle){border:0;border-radius:999px;background:transparent;color:inherit;min-width:34px;height:30px;padding:0 9px;cursor:pointer;font-size:12px;font-weight:800;line-height:30px;text-align:center}
      #coeziv-accessibility-panel button.active{background:rgba(56,189,248,.25);color:#7dd3fc}
      #coeziv-accessibility-panel .sep{width:1px;height:20px;background:rgba(148,163,184,.28)}
      body.light-mode #coeziv-accessibility-panel{background:linear-gradient(135deg,rgba(255,255,255,.96),rgba(224,242,254,.92));color:#0f172a}
      @media(max-width:420px){#coeziv-accessibility-panel{right:8px;bottom:10px;gap:4px;padding:5px}#coeziv-accessibility-panel .coeziv-compact-toggle{min-width:92px;height:28px;padding:0 8px;font-size:10px;line-height:28px}#coeziv-accessibility-panel button:not(.coeziv-compact-toggle){min-width:30px;height:28px;padding:0 7px;font-size:11px;line-height:28px}}
    `;
    document.head.appendChild(style);
  }

  function applyScale() {
    const scale = SCALE_STEPS.includes(parseFloat(localStorage.getItem(SCALE_KEY) || "1")) ? parseFloat(localStorage.getItem(SCALE_KEY) || "1") : 1;
    document.documentElement.style.setProperty("--coeziv-ui-scale", String(scale));
    document.body.style.zoom = String(scale);
    document.body.style.transformOrigin = "top center";
    document.body.setAttribute("data-scale", scale === 1 ? "normal" : scale === 1.15 ? "large" : "xlarge");
  }

  function refreshCompactLabel() {
    const btn = document.querySelector("#coeziv-accessibility-panel .coeziv-compact-toggle");
    if (!btn) return;
    const scale = parseFloat(localStorage.getItem(SCALE_KEY) || "1");
    btn.textContent = `RO/EN · ${scale === 1.15 ? "A×1.15" : scale === 1.30 ? "A×1.30" : "A+"}`;
  }

  function updateButtons() {
    const l = lang();
    document.querySelectorAll("#coeziv-accessibility-panel button[data-lang]").forEach(btn => btn.classList.toggle("active", btn.dataset.lang === l));
    document.querySelectorAll("#coeziv-accessibility-panel button[data-scale-action]").forEach(btn => btn.classList.remove("active"));
    const up = document.querySelector("#coeziv-accessibility-panel button[data-scale-action='up']");
    if (up && parseFloat(localStorage.getItem(SCALE_KEY) || "1") > 1) up.classList.add("active");
  }

  function setLang(next) {
    localStorage.setItem(LANG_KEY, next === "en" ? "en" : "ro");
    document.documentElement.setAttribute("lang", lang());
    if (document.body) document.body.setAttribute("data-lang", lang());
    applyAll();
  }

  function setScale(scale) {
    const next = SCALE_STEPS.includes(Number(scale)) ? Number(scale) : 1;
    localStorage.setItem(SCALE_KEY, String(next));
    applyScale();
    updateButtons();
    refreshCompactLabel();
  }

  function bumpScale(dir) {
    const current = parseFloat(localStorage.getItem(SCALE_KEY) || "1");
    const idx = Math.max(0, SCALE_STEPS.indexOf(current));
    setScale(SCALE_STEPS[Math.min(SCALE_STEPS.length - 1, Math.max(0, idx + dir))]);
  }

  function injectPanel() {
    if (document.getElementById("coeziv-accessibility-panel")) return;
    const panel = document.createElement("div");
    panel.id = "coeziv-accessibility-panel";
    panel.className = "collapsed";
    panel.setAttribute("aria-label", "Language and text size controls");
    panel.innerHTML = `<button type="button" class="coeziv-compact-toggle" title="Language / text size">RO/EN · A+</button><div class="coeziv-panel-controls"><button type="button" data-lang="ro" title="Română">RO</button><button type="button" data-lang="en" title="English">EN</button><span class="sep" aria-hidden="true"></span><button type="button" data-scale-action="down" title="Micșorează textul">A−</button><button type="button" data-scale-action="up" title="Mărește textul">A+</button></div>`;
    panel.addEventListener("click", ev => {
      const compact = ev.target.closest(".coeziv-compact-toggle");
      if (compact) { panel.classList.toggle("collapsed"); if (!panel.classList.contains("collapsed")) scheduleCollapse(); return; }
      const langBtn = ev.target.closest("button[data-lang]");
      if (langBtn) { setLang(langBtn.dataset.lang); scheduleCollapse(); return; }
      const scaleBtn = ev.target.closest("button[data-scale-action]");
      if (scaleBtn) { bumpScale(scaleBtn.dataset.scaleAction === "up" ? 1 : -1); scheduleCollapse(); }
    });
    document.body.appendChild(panel);
  }

  function scheduleCollapse() {
    clearTimeout(collapseTimer);
    collapseTimer = setTimeout(() => { const p = document.getElementById("coeziv-accessibility-panel"); if (p) p.classList.add("collapsed"); }, 3500);
  }

  function applyAll() {
    if (!document.body) return;
    document.documentElement.setAttribute("lang", lang());
    document.body.setAttribute("data-lang", lang());
    renderHeadings();
    renderStandardRegimes();
    renderState();
    renderDaily();
    renderParticipation();
    renderRisk();
    updateButtons();
    refreshCompactLabel();
  }

  function scheduleRefresh() { clearTimeout(refreshTimer); refreshTimer = setTimeout(applyAll, 150); }

  function observeChanges() {
    const observer = new MutationObserver(mutations => {
      for (const m of mutations) {
        if (m.target && m.target.closest && m.target.closest("#coeziv-accessibility-panel")) continue;
        scheduleRefresh();
        break;
      }
    });
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
  }

  function boot() {
    injectStyles();
    injectPanel();
    applyScale();
    setLang(lang());
    observeChanges();
    setInterval(applyAll, 5000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.CoezivI18n = { setLang, getLang: lang, setScale, getScale: () => parseFloat(localStorage.getItem(SCALE_KEY) || "1"), applyAll };
})();