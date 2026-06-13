/* CohesivX BTC — professional RO/EN language + accessibility module */
(function () {
  "use strict";

  const LANG_KEY = "coeziv_btc_lang";
  const SCALE_KEY = "coeziv_btc_scale";
  const SCALE_STEPS = [1, 1.15, 1.30];
  let collapseTimer = null;
  let refreshTimer = null;

  const USD = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

  const RO = {
    appTitle: "MECANISM COEZIV BTC",
    snapshotLine: "Actualizat aproximativ la 24h.",
    livePrice: "Preț live:",
    dailyTitle: "INTERPRETARE COEZIVĂ ZILNICĂ",
    dailySubtitle: "Explicație naturală a stării structurale observate de mecanism.",
    participationTitle: "COEZIUNE PARTICIPATIVĂ",
    participationSubtitle: "Starea interesului participanților în ecosistem.",
    riskTitle: "FEREASTRĂ DE RISC",
    riskSubtitle: "Context istoric derivat din backtestul coeziv.",
    fgTitle: "INDICE FEAR & GREED COEZIV",
    fgSubtitle: "Sentiment de piață derivat din mecanismul coeziv (structură + semnal + volatilitate), fără date sociale sau externe.",
    historyTitle: "ISTORIC CONTEXTE MECANISM",
    historySubtitle: "Ultimele snapshot-uri ale mecanismului. Nu sunt semnale de intrare automată, ci context.",
    regimesTitle: "REGIMURI STANDARD DE PIAȚĂ",
    regimesSubtitle: "Mecanismul traduce structura internă a pieței (regim + flux) în câteva contexte standard. Eticheta de sus, „Regim de piață”, va fi întotdeauna una dintre situațiile de mai jos.",
    viewFull: "Vezi interpretarea completă",
    hideFull: "Ascunde interpretarea completă"
  };

  const EN = {
    appTitle: "COHESIVE BTC MECHANISM",
    snapshotLine: "Mechanism snapshot: updated approximately every 24h.",
    livePrice: "Live price:",
    dailyTitle: "DAILY COHESIVE INTERPRETATION",
    dailySubtitle: "Natural-language explanation of the structural state observed by the mechanism.",
    participationTitle: "PARTICIPATION COHESION",
    participationSubtitle: "State of participant interest across the ecosystem.",
    riskTitle: "RISK WINDOW",
    riskSubtitle: "Historical context derived from the cohesive backtest.",
    fgTitle: "COHESIVE FEAR & GREED INDEX",
    fgSubtitle: "Market sentiment derived from the cohesive mechanism: structure, signal and volatility, without social or external data.",
    historyTitle: "MECHANISM CONTEXT HISTORY",
    historySubtitle: "Recent mechanism snapshots. These are not automatic entry signals, but structural context.",
    regimesTitle: "STANDARD MARKET REGIMES",
    regimesSubtitle: "The mechanism translates the market’s internal structure, regime and flow into a small set of standard contexts. The top regime label will always match one of the situations below.",
    viewFull: "View full interpretation",
    hideFull: "Hide full interpretation"
  };

  const STATIC_EN = new Map([
    ["MECANISM COEZIV BTC", EN.appTitle],
    ["Actualizat aproximativ la 24h.", EN.snapshotLine],
    ["Preț live:", EN.livePrice],
    ["INTERPRETARE COEZIVĂ ZILNICĂ", EN.dailyTitle],
    ["Explicație naturală a stării structurale observate de mecanism.", EN.dailySubtitle],
    ["COEZIUNE PARTICIPATIVĂ", EN.participationTitle],
    ["Starea interesului participanților în ecosistem.", EN.participationSubtitle],
    ["FEREASTRĂ DE RISC", EN.riskTitle],
    ["Context istoric derivat din backtestul coeziv.", EN.riskSubtitle],
    ["INDICE FEAR & GREED COEZIV", EN.fgTitle],
    ["ISTORIC CONTEXTE MECANISM", EN.historyTitle],
    ["REGIMURI STANDARD DE PIAȚĂ", EN.regimesTitle],
    ["Vezi interpretarea completă", EN.viewFull],
    ["Ascunde interpretarea completă", EN.hideFull]
  ]);

  const COMPLETE_EN_RULES = [
    [/Prețul live este egal cu prețul analizat de mecanism\./g, "The live price is equal to the price analyzed by the mechanism."],
    [/Preț live încărcat\. Așteptăm un snapshot de preț valid din mecanism\./g, "Live price loaded. Waiting for a valid price snapshot from the mechanism."],
    [/Nu am putut încărca snapshotul de preț\. Așteptăm ca mecanismul să revină\./g, "Could not load the price snapshot. Waiting for the mechanism to recover."],
    [/Diferență foarte mică \(zgomot normal de piață\)\./g, "Very small difference, normal market noise."],
    [/Mișcare mică; semnalul mecanismului rămâne reperul principal\./g, "Small movement; the mechanism signal remains the main reference."],
    [/Mișcare relevantă; poți ajusta timing-ul intrării sau ieșirii\./g, "Relevant movement; entry or exit timing can be adjusted."],
    [/Mișcare puternică; verifică contextul și lichiditatea\./g, "Strong movement; check context and liquidity."],
    [/Prețul live este ([+−\-]?[0-9.,]+)% \(([+−\-]?[0-9.,]+) USD\) peste prețul mecanismului\./g, "The live price is $1% ($2 USD) above the mechanism price."],
    [/Prețul live este ([+−\-]?[0-9.,]+)% \(([+−\-]?[0-9.,]+) USD\) sub prețul mecanismului\./g, "The live price is $1% ($2 USD) below the mechanism price."],
    [/Status deviație: Normală \(fără abatere față de model\)\./g, "Deviation status: Normal, with no meaningful deviation from the model."],
    [/Status deviație: Controlată\./g, "Deviation status: Controlled."],
    [/Status deviație: Tensionată\./g, "Deviation status: Tense."],
    [/Status deviație: Extremă\./g, "Deviation status: Extreme."]
  ];

  const STANDARD_REGIME_COPY = {
    up_trend_strong: {
      roTitle: "1. Trend ascendent susținut",
      enTitle: "1. Sustained upward trend",
      roText: "Cumpărătorii domină clar piața, iar corecțiile tind să fie rapide și superficiale. Context tipic pentru faze de expansiune, cu risc crescut de supra-extindere pe termen scurt.",
      enText: "Buyers clearly dominate the market, while pullbacks tend to be quick and shallow. This is typical of expansion phases, with increased short-term overextension risk."
    },
    up_trend_moderate: {
      roTitle: "2. Trend ascendent moderat",
      enTitle: "2. Moderate upward trend",
      roText: "Trend pozitiv, dar cu corecții regulate. Presiunea de creștere există, însă mișcările sunt mai echilibrate între impulsuri și pullback-uri.",
      enText: "Positive trend, but with regular corrections. Upward pressure exists, while movement is more balanced between impulses and pullbacks."
    },
    range_bias_up: {
      roTitle: "3. Range cu bias pozitiv",
      enTitle: "3. Range with positive bias",
      roText: "Piața consolidează într-un interval, dar maximele sunt împinse treptat în sus. Context favorabil pentru acumulare graduală, nu pentru intrări agresive.",
      enText: "The market consolidates within a range, but highs are gradually pushed upward. This supports gradual accumulation rather than aggressive entries."
    },
    range_neutral: {
      roTitle: "4. Range neutru",
      enTitle: "4. Neutral range",
      roText: "Echilibru relativ între cumpărători și vânzători, fără direcție clară. Mișcările laterale domină, iar semnalele direcționale au robustețe redusă.",
      enText: "Relative balance between buyers and sellers, with no clear direction. Sideways movement dominates, and directional signals have reduced robustness."
    },
    range_bias_down: {
      roTitle: "5. Range cu bias negativ",
      enTitle: "5. Range with negative bias",
      roText: "Piața consolidează, dar minimele sunt coborâte treptat. Context în care presiunea de vânzare se acumulează, iar raliurile sunt fragile.",
      enText: "The market is consolidating, but lows are gradually pushed lower. Selling pressure is accumulating, and rallies remain fragile."
    },
    down_trend_moderate: {
      roTitle: "6. Trend descendent moderat",
      enTitle: "6. Moderate downward trend",
      roText: "Presiune de scădere prezentă, însă cu pauze și contra-mișcări regulate. Raliurile tind să fie folosite pentru reducerea expunerii, nu pentru acumulare.",
      enText: "Downward pressure is present, but with pauses and regular counter-moves. Rallies tend to be used for reducing exposure rather than accumulation."
    },
    down_trend_strong: {
      roTitle: "7. Trend descendent susținut",
      enTitle: "7. Sustained downward trend",
      roText: "Vânzătorii controlează clar piața, iar bounce-urile sunt de obicei scurte și fragile. Context specific fazelor de capitulare sau de descărcare forțată de poziții.",
      enText: "Sellers clearly control the market, and bounces are usually short and fragile. This is typical of capitulation or forced-position unwinding phases."
    },
    transition_accumulation: {
      roTitle: "8. Regim neutru / tranziție",
      enTitle: "8. Neutral / transition regime",
      roText: "Zone de schimbare, în care modelul nu vede încă un avantaj structural clar nici pentru cumpărători, nici pentru vânzători. De obicei premerg fazelor în care piața alege o nouă direcție: trend sau range.",
      enText: "Change zones where the model does not yet see a clear structural advantage for buyers or sellers. They often precede phases where the market chooses a new direction: trend or range."
    }
  };

  function getLang() { return localStorage.getItem(LANG_KEY) || window.COEZIV_DEFAULT_LANG || "ro"; }
  function isEn() { return getLang() === "en"; }
  function getScale() { const raw = parseFloat(localStorage.getItem(SCALE_KEY) || "1"); return SCALE_STEPS.includes(raw) ? raw : 1; }
  function setText(id, value) { const el = document.getElementById(id); if (el && typeof value === "string") el.textContent = value; }
  function setEl(el, value) { if (el && typeof value === "string") el.textContent = value; }
  function fmtUsd(v) { return Number.isFinite(Number(v)) ? USD.format(Number(v)) : "n/a"; }
  function pct(v, decimals = 1) { return Number.isFinite(Number(v)) ? (Number(v) * 100).toFixed(decimals) : "n/a"; }
  function nr(v, decimals = 0) { return Number.isFinite(Number(v)) ? Number(v).toFixed(decimals) : "n/a"; }

  function setLang(lang) {
    const next = lang === "en" ? "en" : "ro";
    localStorage.setItem(LANG_KEY, next);
    document.documentElement.setAttribute("lang", next);
    if (document.body) document.body.setAttribute("data-lang", next);
    applyAll();
  }

  function setScale(scale) {
    let next = Number(scale);
    if (!SCALE_STEPS.includes(next)) next = 1;
    localStorage.setItem(SCALE_KEY, String(next));
    applyScale();
    updateScaleButtons();
    refreshCompactLabel();
  }

  function bumpScale(direction) {
    const current = getScale();
    const index = Math.max(0, SCALE_STEPS.indexOf(current));
    setScale(SCALE_STEPS[Math.min(SCALE_STEPS.length - 1, Math.max(0, index + direction))]);
  }

  function applyScale() {
    const scale = getScale();
    document.documentElement.style.setProperty("--coeziv-ui-scale", String(scale));
    document.body.style.zoom = String(scale);
    document.body.style.transformOrigin = "top center";
    document.body.setAttribute("data-scale", scale === 1 ? "normal" : scale === 1.15 ? "large" : "xlarge");
  }

  function translateText(value) {
    if (!value || typeof value !== "string") return value;
    if (!isEn()) return value;
    const trimmed = value.trim();
    if (STATIC_EN.has(trimmed)) return value.replace(trimmed, STATIC_EN.get(trimmed));
    let out = value;
    for (const [pattern, replacement] of COMPLETE_EN_RULES) out = out.replace(pattern, replacement);
    return out;
  }

  function translateNode(node) {
    if (!node || node.nodeType !== Node.TEXT_NODE) return;
    const original = node.__coezivRoText || node.nodeValue;
    if (!node.__coezivRoText) node.__coezivRoText = original;
    node.nodeValue = isEn() ? translateText(original) : original;
  }

  function walk(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent) return NodeFilter.FILTER_REJECT;
        if (parent.closest && parent.closest("#coeziv-accessibility-panel")) return NodeFilter.FILTER_REJECT;
        if (parent.closest && parent.closest("#daily-ai-card")) return NodeFilter.FILTER_REJECT;
        if (["SCRIPT", "STYLE", "NOSCRIPT"].includes(parent.tagName)) return NodeFilter.FILTER_REJECT;
        if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(translateNode);
  }

  function applyStaticTranslation() {
    if (!document.body) return;
    walk(document.body);
    document.querySelectorAll("[title], [aria-label]").forEach((el) => {
      if (el.closest && el.closest("#coeziv-accessibility-panel")) return;
      ["title", "aria-label"].forEach((attr) => {
        if (!el.hasAttribute(attr)) return;
        const key = `__coezivRo_${attr}`;
        const original = el[key] || el.getAttribute(attr);
        if (!el[key]) el[key] = original;
        el.setAttribute(attr, isEn() ? translateText(original) : original);
      });
    });
  }

  async function fetchJson(url) {
    const res = await fetch(`${url}?t=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) throw new Error(url);
    return await res.json();
  }

  function regimeCompactLabel(fullText, code) {
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

  function signalLabel(signal) {
    const s = String(signal || "flat").toLowerCase();
    if (isEn()) return s === "long" ? "Context: upward pressure" : s === "short" ? "Context: downside risk" : "Context: neutral";
    return s === "long" ? "Context: presiune de creștere" : s === "short" ? "Context: risc de scădere" : "Context: neutru";
  }

  function signalShortLabel(signal) {
    const s = String(signal || "flat").toLowerCase();
    if (isEn()) return s === "long" ? "Upward pressure" : s === "short" ? "Downside risk" : "Neutral context";
    return s === "long" ? "Presiune de creștere" : s === "short" ? "Risc de scădere" : "Context neutru";
  }

  function flowText(bias, strength) {
    if (isEn()) {
      const s = strength === "slab" ? "weak" : strength === "puternică" || strength === "puternic" ? "strong" : strength || "";
      if (bias === "pozitiv") return `Market flow: tilted toward buying${s ? ` (${s})` : ""}.`;
      if (bias === "negativ") return `Market flow: tilted toward selling${s ? ` (${s})` : ""}.`;
      if (bias === "neutru") return `Market flow: relatively balanced between buyers and sellers${s ? ` (${s})` : ""}.`;
      return "Market flow: not available yet.";
    }
    let text = "Flux de piață: n/a (așteptăm date despre flux).";
    if (bias === "pozitiv") text = "Flux de piață: orientat spre cumpărare";
    else if (bias === "negativ") text = "Flux de piață: orientat spre vânzare";
    else if (bias === "neutru") text = "Flux de piață: relativ echilibrat între cumpărători și vânzători";
    if (bias && strength) text += ` (${strength}).`; else if (bias) text += ".";
    return text;
  }

  function liquidityText(regime, strength) {
    const r = String(regime || "").trim().toLowerCase();
    if (isEn()) {
      const s = strength === "puternică" || strength === "puternic" ? "strong" : strength === "slab" ? "weak" : strength || "";
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

  function riskLevelLabel(level, active) {
    if (isEn()) {
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

  function riskRegimeLabel(regime) {
    const r = String(regime || "").toLowerCase();
    const ro = { bear_struct:"Degradare structurală", bear_late:"Degradare avansată", accum_bear:"Acumulare fragilă", bull_struct:"Structură pozitivă", bull_late:"Expansiune matură", accum_bull:"Acumulare pozitivă", range_pos:"Range cu bias pozitiv", range_neg:"Range cu bias negativ", range_neutral:"Range neutru", neutral:"Tranziție neutră" };
    const en = { bear_struct:"Structural degradation", bear_late:"Advanced degradation", accum_bear:"Fragile accumulation", bull_struct:"Positive structure", bull_late:"Mature expansion", accum_bull:"Positive accumulation", range_pos:"Range with positive bias", range_neg:"Range with negative bias", range_neutral:"Neutral range", neutral:"Neutral transition" };
    return (isEn() ? en[r] : ro[r]) || regime || "n/a";
  }

  function renderEnergyCard(costs) {
    const el = document.getElementById("prod-cost-line");
    if (!el) return;
    const fmt = (v) => Number.isFinite(Number(v)) ? "~" + fmtUsd(v) + " USD" : "n/a";
    const c = costs || {};
    const t = isEn();
    el.className = "energy-card" + (!costs ? " loading" : "");
    el.innerHTML = `
      <div class="energy-card-header">
        <div>
          <div class="energy-title">${t ? "BTC ENERGY THRESHOLD" : "Prag energetic BTC"}</div>
          <div class="energy-subtitle">${t ? "Estimated energy cost of production" : "Cost energetic estimat de producție"}</div>
        </div>
        <div class="energy-badge">${t ? "estimate" : "estimare"}</div>
      </div>
      <div class="energy-values">
        <div class="energy-item"><span class="energy-label">${t ? "Efficient miner" : "Miner eficient"}</span><span class="energy-value">${fmt(c.cheap)}</span></div>
        <div class="energy-item main"><span class="energy-label">${t ? "Average miner" : "Miner mediu"}</span><span class="energy-value">${fmt(c.average)}</span></div>
        <div class="energy-item"><span class="energy-label">${t ? "Expensive miner" : "Miner scump"}</span><span class="energy-value">${fmt(c.expensive)}</span></div>
      </div>
      <div class="energy-note">${t ? "Calculation based on network difficulty, block reward, equipment efficiency and energy price. It does not represent miners’ full accounting cost." : "Calcul bazat pe dificultatea rețelei, recompensa pe bloc, eficiența echipamentelor și prețul energiei. Nu reprezintă costul total contabil al minerilor."}</div>`;
  }

  async function renderStateLanguage() {
    try {
      const state = await fetchJson("coeziv_state.json");
      const t = isEn();
      const price = state.price_usd;
      const signal = String(state.signal || "flat").toLowerCase();
      const prob = state.signal_probability;
      const samples = state.signal_prob_samples || 0;
      const horizon = state.signal_prob_horizon_hours || 24;
      const drift = state.signal_expected_drift || {};
      const bd = state.signal_prob_breakdown || {};
      const regime = state.market_regime;
      let regimeLabel = "n/a", regimeCode = "";
      if (typeof regime === "string") regimeLabel = regime;
      else if (regime && typeof regime === "object") { regimeLabel = regime.label || regimeLabel; regimeCode = regime.code || ""; }

      setText("signal-label", signalLabel(signal));
      const regimeEl = document.getElementById("regime-line");
      const fullRegime = t ? `Market regime: ${regimeLabel}.` : `Regim de piață: ${regimeLabel}.`;
      if (regimeEl) { regimeEl.setAttribute("data-full-regime", fullRegime); regimeEl.setAttribute("title", fullRegime); regimeEl.textContent = regimeCompactLabel(fullRegime, regimeCode); }

      if (t) {
        setText("message", `At the mechanism update, BTC was trading around ${fmtUsd(price)} USD. The mechanism reads the current context as neutral. Market flow is relatively balanced and weak. Liquidity is high, so price movement is more easily absorbed. Price remains close to the network’s estimated production-cost equilibrium. In similar historical contexts, the next ~24 hours typically ranged from about ${pct(drift.p10)}% in weaker scenarios to ${pct(drift.p90)}% in stronger scenarios, with a median near ${pct(drift.p50)}%. This is a structural risk interpretation, not financial advice.`);
        setText("signal-prob", `Historical probability: about ${pct(prob, 0)}% for price to move with the mechanism’s signal over the next ${horizon}h, based on ${samples} similar historical contexts.`);
        setText("signal-prob-breakdown", `Historical distribution over ${horizon}h: about ${pct(bd.in_direction, 0)}% with the signal, ${pct(bd.opposite, 0)}% against it, and ${pct(bd.flat, 0)}% neutral movement or noise.`);
        setText("drift-range", `Typical movement range (${horizon}h): between ~${pct(drift.p10)}% and ~${pct(drift.p90)}%, with a median near ~${pct(drift.p50)}%, based on similar historical contexts.`);
      } else {
        setText("message", state.message || "Nu există mesaj coeziv pentru acest moment.");
        setText("signal-prob", `Probabilitate istorică: ~${pct(prob, 0)}% ca prețul să se miște în direcția semnalului în următoarele ${horizon}h (bazat pe ${samples} situații similare).`);
        setText("signal-prob-breakdown", `Distribuție istorică (în ${horizon}h): ~${pct(bd.in_direction, 0)}% în direcția semnalului, ~${pct(bd.opposite, 0)}% contra semnalului și ~${pct(bd.flat, 0)}% mișcare neutră / zgomot.`);
        setText("drift-range", `Interval tipic de mișcare (${horizon}h): între ~${pct(drift.p10)}% și ~${pct(drift.p90)}% (mediană ~${pct(drift.p50)}%), bazat pe contexte similare din istoric.`);
      }
      setText("flow-line", flowText(state.flow_bias, state.flow_strength));
      setText("liquidity-line", liquidityText(state.liquidity_regime, state.liquidity_strength));
      renderEnergyCard(state.production_costs_usd || null);
      renderFearGreedLanguage(state.fg || null);
      renderHistoryLanguage(state);
    } catch (e) {}
  }

  async function renderDailyLanguage() {
    try {
      const data = await fetchJson("daily_cohesiv_interpretation.json");
      const t = isEn();
      setText("daily-ai-state", t ? (data.general_state_en || data.general_state || "structural state pending") : (data.general_state || "stare în curs de interpretare"));
      setText("daily-ai-summary", t ? (data.plain_language_en || data.summary_en || data.plain_language || data.summary || "Daily interpretation pending.") : (data.plain_language || data.summary || "Interpretarea nu este încă disponibilă."));
      setText("daily-ai-participation", t ? (data.participation_en || data.participation || "–") : (data.participation || "–"));
      setText("daily-ai-risk", t ? (data.risk_window_en || data.risk_window || "–") : (data.risk_window || "–"));
      const fgRegime = t ? [data.fear_greed_en || data.fear_greed, data.market_regime_en || data.market_regime].filter(Boolean).join(" • ") : [data.fear_greed, data.market_regime].filter(Boolean).join(" • ");
      setText("daily-ai-fg-regime", fgRegime || "–");
      setText("daily-ai-watch", t ? (data.watch_next_en || data.watch_next || "–") : (data.watch_next || "–"));
      setText("daily-ai-footer", t ? (data.disclaimer_en || "Experimental structural interpretation, not financial advice.") : (data.disclaimer || "Interpretare structurală experimentală, nu recomandare financiară."));
      const panel = document.getElementById("daily-ai-full-panel");
      const btn = document.getElementById("daily-ai-full-toggle");
      const full = t ? (data.full_interpretation_en || data.full_interpretation || "") : (data.full_interpretation || "");
      if (panel && full) panel.textContent = full.replace(/^##\s+/gm, "").replace(/^###\s+/gm, "").replace(/\*\*(.*?)\*\*/g, "$1").trim();
      if (btn) btn.textContent = t ? (panel && panel.classList.contains("open") ? EN.hideFull : EN.viewFull) : (panel && panel.classList.contains("open") ? RO.hideFull : RO.viewFull);
    } catch (e) {}
  }

  function participationLabel(data) {
    const lvl = data && data.level;
    if (isEn()) return lvl === "cohesive" ? "Cohesive participation" : lvl === "tense" ? "Tense participation" : "Participation pending";
    const label = data && data.label ? data.label : "participare necunoscută";
    return label.charAt(0).toUpperCase() + label.slice(1);
  }

  async function renderParticipationLanguage() {
    try {
      const data = await fetchJson("participation_cohesion.json");
      const comps = data.components || {};
      const history = data.history || {};
      const card = document.getElementById("participation-cohesion-card");
      const pill = document.getElementById("participation-cohesion-pill");
      if (pill) { pill.className = `participation-pill participation-${data.level || "tense"}`; pill.textContent = participationLabel(data); }
      setText("participation-cohesion-score", `${nr(data.score, 0)} / 100`);
      setText("participation-flow", nr(comps.flow_component, 0));
      setText("participation-liquidity", nr(comps.liquidity_component, 0));
      setText("participation-history-mean", nr(history.mean_score, 0));
      setText("participation-history-min", nr(history.min_score, 0));
      if (isEn()) {
        setText("participation-cohesion-text", "Participants remain active, but behavior is tense. Interest persists, while the dominant flow may be defensive or oriented toward exit.");
        setText("participation-cohesion-footer", "Experimental indicator derived from flow, liquidity, persistence and tension against the model. It does not yet measure direct peer-to-peer on-chain usage.");
        if (card) card.querySelectorAll(".participation-metric-label").forEach((el, i) => setEl(el, ["Flow", "Liquidity", "Historical average", "Historical minimum"][i] || el.textContent));
      } else {
        setText("participation-cohesion-text", data.main_text || "Indicatorul de participare nu este disponibil momentan.");
        setText("participation-cohesion-footer", data.footer || "Indicator experimental, nu recomandare de tranzacționare.");
        if (card) card.querySelectorAll(".participation-metric-label").forEach((el, i) => setEl(el, ["Flux", "Lichiditate", "Istoric mediu", "Minim istoric"][i] || el.textContent));
      }
    } catch (e) {}
  }

  async function renderRiskWindowLanguage() {
    try {
      const data = await fetchJson("risk_window.json");
      const active = !!data.active;
      const level = data.level || (active ? "unknown" : "normal");
      const pill = document.getElementById("risk-window-pill");
      if (pill) { pill.className = `risk-pill risk-${active ? level : "normal"}`; pill.textContent = riskLevelLabel(level, active); }
      setText("risk-window-rate", `${nr((data.historical_confirmation_rate || 0) * 100, 0)}%`);
      setText("risk-window-days", isEn() ? `~${nr(data.median_days_to_confirmation, 0)} days` : `~${nr(data.median_days_to_confirmation, 0)} zile`);
      setText("risk-window-regime", riskRegimeLabel(data.current_regime || "n/a"));
      setText("risk-window-streak", isEn() ? `${data.consecutive_degradation_days || 0} days` : `${data.consecutive_degradation_days || 0} zile`);
      if (isEn()) {
        setText("risk-window-text", `The current context belongs to a historical family that produced major weakening in about ${nr((data.historical_confirmation_rate || 0) * 100, 0)}% of cases. When those cases confirmed, confirmation appeared after roughly ${nr(data.median_days_to_confirmation, 0)} days on average.`);
        setText("risk-window-footer", "Statistical interpretation of structural degradation, not a trading recommendation.");
        document.querySelectorAll("#risk-window-card .risk-metric-label").forEach((el, i) => setEl(el, ["Historical confirmation", "Median time", "Current regime", "Persistence"][i] || el.textContent));
      } else {
        setText("risk-window-text", data.main_text || "Fereastra de risc nu este disponibilă momentan.");
        setText("risk-window-footer", data.footer || "Interpretare statistică, nu recomandare de tranzacționare.");
        document.querySelectorAll("#risk-window-card .risk-metric-label").forEach((el, i) => setEl(el, ["Confirmare istorică", "Timp median", "Regim curent", "Persistență"][i] || el.textContent));
      }
    } catch (e) {}
  }

  function fgZoneTitle(zone) {
    const z = String(zone || "neutral").toLowerCase();
    const label = { extreme_fear:"Extreme Fear", fear:"Fear", neutral:"Neutral", greed:"Greed", extreme_greed:"Extreme Greed" }[z] || "Neutral";
    return label;
  }

  function fgDescription(fg) {
    if (!fg) return isEn() ? "The indicator will be populated automatically after the next mechanism update." : "Indicatorul va fi populat automat după următorul update al mecanismului.";
    const z = String(fg.combined_zone || "neutral").toLowerCase();
    const tension = Number.isFinite(Number(fg.tension)) ? Number(fg.tension) : null;
    let base;
    if (isEn()) {
      if (z === "extreme_fear") base = "Panic or structural capitulation; elevated risk of fast and disorderly movement.";
      else if (z === "fear") base = "Risk aversion and defensive positioning; useful context for prudence and position-size management.";
      else if (z === "greed") base = "High confidence and greater risk-taking; supportive of continuation, but with attention to exhaustion.";
      else if (z === "extreme_greed") base = "Euphoria and possible overextension; historically, such zones often precede cooling or correction phases.";
      else base = "Relative balance between risk and opportunity; the mechanism does not see a decisive emotional advantage.";
      if (tension === null) return base;
      if (tension < 15) return base + " Structure and emotion are well aligned.";
      if (tension < 30) return base + " There is moderate tension between structure and market emotion.";
      return base + " Strong misalignment between structure and emotion; the context is sensitive to regime change.";
    }
    if (z === "extreme_fear") base = "Panică sau capitulare structurală; risc ridicat de mișcări rapide și dezordonate.";
    else if (z === "fear") base = "Aversiune la risc și poziționare defensivă; context util pentru prudență și management al dimensiunii pozițiilor.";
    else if (z === "greed") base = "Încredere ridicată și asumare mai mare de risc; context favorabil continuării trendului, dar cu atenție la epuizare.";
    else if (z === "extreme_greed") base = "Euforie și potențial de supra-extindere; istoric, astfel de zone preced frecvent faze de răcire sau corecție.";
    else base = "Echilibru relativ între risc și oportunitate; mecanismul nu vede un avantaj emoțional decisiv.";
    if (tension === null) return base;
    if (tension < 15) return base + " Aliniere bună între structură și emoție.";
    if (tension < 30) return base + " Există o tensiune moderată între structură și emoția de piață.";
    return base + " Dezaliniere puternică între structură și emoție; context sensibil la schimbări de regim.";
  }

  function renderFearGreedLanguage(fg) {
    if (!fg) return;
    setText("fg-score-label", isEn() ? "Combined index: structure + tactical state" : "Indice combinat (structură + tactică)");
    setText("fg-score-zone", fgZoneTitle(fg.combined_zone));
    setText("fg-description", fgDescription(fg));
    document.querySelectorAll(".fg-detail-label").forEach((el, i) => setEl(el, isEn() ? ["Structural", "Tactical (24h)", "Tension"][i] : ["Structural", "Tactic (24h)", "Tensiune"][i]));
  }

  function renderHistoryLanguage(state) {
    const historyList = document.getElementById("history-list");
    const meta = document.getElementById("history-meta");
    if (historyList) {
      historyList.querySelectorAll(".history-row").forEach((row) => {
        const chip = row.querySelector(".history-signal-chip");
        const tag = row.querySelector(".history-tag");
        if (chip) {
          const cls = chip.className || "";
          const sig = cls.includes("history-signal-long") ? "long" : cls.includes("history-signal-short") ? "short" : "flat";
          chip.lastChild && (chip.lastChild.textContent = signalShortLabel(sig));
        }
        if (tag && tag.textContent && tag.textContent.includes("Preț model")) tag.textContent = isEn() ? "Model price at snapshot" : "Preț model la snapshot";
      });
    }
    if (meta && state) {
      const h = Array.isArray(state.signal_history) ? state.signal_history : [];
      const longN = h.filter(x => x.signal === "long").length;
      const shortN = h.filter(x => x.signal === "short").length;
      const flatN = h.filter(x => x.signal === "flat").length;
      const flowPart = state.flow_bias ? (isEn() ? flowText(state.flow_bias, "").replace("Market flow: ", "Current flow: ") : flowText(state.flow_bias, "").replace("Flux de piață: ", "Flux actual: ")) : "";
      const liqPart = state.liquidity_regime ? (isEn() ? liquidityText(state.liquidity_regime, "").replace("Market liquidity: ", "Liquidity ") : liquidityText(state.liquidity_regime, "").replace("Lichiditate piață: ", "Lichiditate ")) : "";
      const top = isEn()
        ? `<span>Total contexts: ${h.length}</span><span class="history-meta-dot">•</span><span>Upward pressure: ${longN}</span><span class="history-meta-dot">•</span><span>Downside risk: ${shortN}</span><span class="history-meta-dot">•</span><span>Neutral context: ${flatN}</span>`
        : `<span>Total contexte: ${h.length}</span><span class="history-meta-dot">•</span><span>Presiune de creștere: ${longN}</span><span class="history-meta-dot">•</span><span>Risc de scădere: ${shortN}</span><span class="history-meta-dot">•</span><span>Context neutru: ${flatN}</span>`;
      const line = [flowPart, liqPart].filter(Boolean).join(" ");
      meta.innerHTML = line ? `<div>${top}</div><div>${line}</div>` : `<div>${top}</div>`;
    }
  }

  function renderStandardRegimesLanguage() {
    Object.entries(STANDARD_REGIME_COPY).forEach(([key, copy]) => {
      const row = document.querySelector(`[data-standard-regime="${key}"]`);
      if (!row) return;
      const title = row.querySelector(".history-date");
      const tag = row.querySelector(".history-tag");
      setEl(title, isEn() ? copy.enTitle : copy.roTitle);
      setEl(tag, isEn() ? copy.enText : copy.roText);
    });
  }

  function renderHeadings() {
    const t = isEn() ? EN : RO;
    setEl(document.querySelector(".title-bar"), t.appTitle);
    setEl(document.querySelector(".asset-text span"), t.snapshotLine);
    setEl(document.querySelector(".live-label"), t.livePrice);
    setEl(document.querySelector("#daily-ai-card .daily-ai-title"), t.dailyTitle);
    setEl(document.querySelector("#daily-ai-card .daily-ai-subtitle"), t.dailySubtitle);
    setEl(document.querySelector("#participation-cohesion-card .participation-title"), t.participationTitle);
    setEl(document.querySelector("#participation-cohesion-card .participation-subtitle"), t.participationSubtitle);
    setEl(document.querySelector("#risk-window-card .risk-title"), t.riskTitle);
    setEl(document.querySelector("#risk-window-card .risk-subtitle"), t.riskSubtitle);
    setEl(document.querySelector(".fg-title"), t.fgTitle);
    setEl(document.querySelector(".fg-subtitle"), t.fgSubtitle);
    const historyCards = Array.from(document.querySelectorAll(".history-header"));
    const historyHeader = historyCards.find(h => h.textContent.includes("ISTORIC") || h.textContent.includes("HISTORY"));
    const regimeHeader = historyCards.find(h => h.textContent.includes("REGIMURI") || h.textContent.includes("STANDARD MARKET"));
    if (historyHeader) { setEl(historyHeader.querySelector(".history-title"), t.historyTitle); setEl(historyHeader.querySelector(".history-subtitle"), t.historySubtitle); }
    if (regimeHeader) { setEl(regimeHeader.querySelector(".history-title"), t.regimesTitle); setEl(regimeHeader.querySelector(".history-subtitle"), t.regimesSubtitle); }
  }

  function injectStyles() {
    if (document.getElementById("coeziv-i18n-style")) return;
    const style = document.createElement("style");
    style.id = "coeziv-i18n-style";
    style.textContent = `
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

  function scheduleCollapse() {
    clearTimeout(collapseTimer);
    collapseTimer = setTimeout(() => { const panel = document.getElementById("coeziv-accessibility-panel"); if (panel) panel.classList.add("collapsed"); }, 3500);
  }

  function refreshCompactLabel() {
    const btn = document.querySelector("#coeziv-accessibility-panel .coeziv-compact-toggle");
    if (!btn) return;
    const scale = getScale();
    const scaleLabel = scale === 1 ? "A+" : scale === 1.15 ? "A×1.15" : "A×1.30";
    btn.textContent = `RO/EN · ${scaleLabel}`;
  }

  function injectPanel() {
    if (document.getElementById("coeziv-accessibility-panel")) return;
    const panel = document.createElement("div");
    panel.id = "coeziv-accessibility-panel";
    panel.className = "collapsed";
    panel.setAttribute("aria-label", "Language and text size controls");
    panel.innerHTML = `<button type="button" class="coeziv-compact-toggle" title="Language / text size">RO/EN · A+</button><div class="coeziv-panel-controls"><button type="button" data-lang="ro" title="Română">RO</button><button type="button" data-lang="en" title="English">EN</button><span class="sep" aria-hidden="true"></span><button type="button" data-scale-action="down" title="Micșorează textul">A−</button><button type="button" data-scale-action="up" title="Mărește textul">A+</button></div>`;
    panel.addEventListener("click", (event) => {
      const compact = event.target.closest(".coeziv-compact-toggle");
      if (compact) { panel.classList.toggle("collapsed"); if (!panel.classList.contains("collapsed")) scheduleCollapse(); return; }
      const langBtn = event.target.closest("button[data-lang]");
      if (langBtn) { setLang(langBtn.dataset.lang); scheduleCollapse(); return; }
      const scaleBtn = event.target.closest("button[data-scale-action]");
      if (scaleBtn) { bumpScale(scaleBtn.dataset.scaleAction === "up" ? 1 : -1); scheduleCollapse(); }
    });
    document.body.appendChild(panel);
    updateButtonState();
    updateScaleButtons();
    refreshCompactLabel();
  }

  function updateButtonState() {
    const lang = getLang();
    document.querySelectorAll("#coeziv-accessibility-panel button[data-lang]").forEach((btn) => btn.classList.toggle("active", btn.dataset.lang === lang));
  }

  function updateScaleButtons() {
    const scale = getScale();
    document.querySelectorAll("#coeziv-accessibility-panel button[data-scale-action]").forEach((btn) => btn.classList.remove("active"));
    const up = document.querySelector("#coeziv-accessibility-panel button[data-scale-action='up']");
    if (up && scale > 1) up.classList.add("active");
  }

  function scheduleRefresh() { clearTimeout(refreshTimer); refreshTimer = setTimeout(applyAll, 150); }

  function observeChanges() {
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.target && mutation.target.closest && mutation.target.closest("#coeziv-accessibility-panel")) continue;
        scheduleRefresh();
        break;
      }
    });
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
  }

  function applyAll() {
    if (!document.body) return;
    applyStaticTranslation();
    renderHeadings();
    renderStandardRegimesLanguage();
    renderStateLanguage();
    renderDailyLanguage();
    renderParticipationLanguage();
    renderRiskWindowLanguage();
    updateButtonState();
    updateScaleButtons();
    refreshCompactLabel();
  }

  function boot() {
    injectStyles();
    injectPanel();
    applyScale();
    setLang(getLang());
    observeChanges();
    setInterval(applyAll, 5000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.CoezivI18n = { setLang, getLang, setScale, getScale, applyScale, applyAll, translateText };
})();