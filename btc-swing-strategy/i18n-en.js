/* CohesivX BTC — professional RO/EN language + accessibility module */
(function () {
  "use strict";

  const LANG_KEY = "coeziv_btc_lang";
  const SCALE_KEY = "coeziv_btc_scale";
  const SCALE_STEPS = [1, 1.15, 1.30];
  let collapseTimer = null;
  let refreshTimer = null;

  const USD = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

  const STATIC_EN = new Map([
    ["MECANISM COEZIV BTC", "COHESIVE BTC MECHANISM"],
    ["Actualizat aproximativ la 24h.", "Mechanism snapshot: updated approximately every 24h."],
    ["Context: neutru", "Context: neutral"],
    ["Context: presiune de creștere", "Context: upward pressure"],
    ["Context: risc de scădere", "Context: downside risk"],
    ["Preț live:", "Live price:"],
    ["Prag energetic BTC", "BTC ENERGY THRESHOLD"],
    ["Cost energetic estimat de producție", "Estimated energy cost of production"],
    ["estimare", "estimate"],
    ["Miner eficient", "Efficient miner"],
    ["Miner mediu", "Average miner"],
    ["Miner scump", "Expensive miner"],
    ["Calcul bazat pe dificultatea rețelei, recompensa pe bloc, eficiența echipamentelor și prețul energiei. Nu reprezintă costul total contabil al minerilor.", "Calculation based on network difficulty, block reward, equipment efficiency and energy price. It does not represent miners’ full accounting cost."],
    ["INTERPRETARE COEZIVĂ ZILNICĂ", "DAILY COHESIVE INTERPRETATION"],
    ["Explicație naturală a stării structurale observate de mecanism.", "Natural-language explanation of the structural state observed by the mechanism."],
    ["Participare", "Participation"],
    ["Fereastră risc", "Risk window"],
    ["Fear & Greed / Regim", "Fear & Greed / Regime"],
    ["Ce urmărim", "What to watch"],
    ["Vezi interpretarea completă", "View full interpretation"],
    ["Ascunde interpretarea completă", "Hide full interpretation"],
    ["ISTORIC SEMNALE", "SIGNAL HISTORY"],
    ["Preț model la snapshot", "Model price at snapshot"],
    ["COEZIUNE PARTICIPATIVĂ", "PARTICIPATION COHESION"],
    ["Starea interesului participanților în ecosistem.", "State of participant interest across the ecosystem."],
    ["FEREASTRA DE RISC", "RISK WINDOW"],
    ["Nu este o recomandare de tranzacționare.", "This is not a trading recommendation."],
    ["Nu este o recomandare financiară.", "This is not financial advice."]
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

  function getLang() { return localStorage.getItem(LANG_KEY) || window.COEZIV_DEFAULT_LANG || "ro"; }

  function getScale() {
    const raw = parseFloat(localStorage.getItem(SCALE_KEY) || "1");
    return SCALE_STEPS.includes(raw) ? raw : 1;
  }

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el && typeof value === "string") el.textContent = value;
  }

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
    if (getLang() !== "en") return value;
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
    node.nodeValue = getLang() === "en" ? translateText(original) : original;
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
    walk(document.body);
    document.querySelectorAll("[title], [aria-label]").forEach((el) => {
      if (el.closest && el.closest("#coeziv-accessibility-panel")) return;
      ["title", "aria-label"].forEach((attr) => {
        if (!el.hasAttribute(attr)) return;
        const key = `__coezivRo_${attr}`;
        const original = el[key] || el.getAttribute(attr);
        if (!el[key]) el[key] = original;
        el.setAttribute(attr, getLang() === "en" ? translateText(original) : original);
      });
    });
  }

  function fmtUsd(v) { return Number.isFinite(Number(v)) ? USD.format(Number(v)) : "n/a"; }
  function pct(v, decimals = 1) { return Number.isFinite(Number(v)) ? (Number(v) * 100).toFixed(decimals) : "n/a"; }

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

  function flowRo(bias, strength) {
    let text = "Flux de piață: n/a (așteptăm date despre flux).";
    if (bias === "pozitiv") text = "Flux de piață: orientat spre cumpărare";
    else if (bias === "negativ") text = "Flux de piață: orientat spre vânzare";
    else if (bias === "neutru") text = "Flux de piață: relativ echilibrat între cumpărători și vânzători";
    if (bias && strength) text += ` (${strength}).`; else if (bias) text += ".";
    return text;
  }

  function flowEn(bias, strength) {
    const s = strength === "slab" ? "weak" : strength === "puternică" || strength === "puternic" ? "strong" : strength || "";
    if (bias === "pozitiv") return `Market flow: tilted toward buying${s ? ` (${s})` : ""}.`;
    if (bias === "negativ") return `Market flow: tilted toward selling${s ? ` (${s})` : ""}.`;
    if (bias === "neutru") return `Market flow: relatively balanced between buyers and sellers${s ? ` (${s})` : ""}.`;
    return "Market flow: not available yet.";
  }

  function liquidityRo(regime, strength) {
    const r = String(regime || "").trim().toLowerCase();
    let text = "Lichiditate piață: n/a (așteptăm date suplimentare)";
    if (r === "ridicată") text = "Lichiditate piață: ridicată; mișcări peste nivelurile obișnuite.";
    else if (r === "scăzută") text = "Lichiditate piață: scăzută; mișcări sub nivelurile obișnuite.";
    else if (r === "normală" || r === "moderată") text = "Lichiditate piață: moderată pentru acest context.";
    if (regime && strength && r !== "normală" && r !== "moderată") text += ` (deviație ${strength} față de nivelurile obișnuite.)`;
    return text;
  }

  function liquidityEn(regime, strength) {
    const r = String(regime || "").trim().toLowerCase();
    const s = strength === "puternică" || strength === "puternic" ? "strong" : strength === "slab" ? "weak" : strength || "";
    if (r === "ridicată") return `Market liquidity: high; movement is above usual levels${s ? `, with a ${s} deviation from normal activity` : ""}.`;
    if (r === "scăzută") return `Market liquidity: low; movement is below usual levels${s ? `, with a ${s} deviation from normal activity` : ""}.`;
    if (r === "normală" || r === "moderată") return "Market liquidity: moderate for this context.";
    return "Market liquidity: not available yet.";
  }

  async function fetchJson(url) {
    const res = await fetch(`${url}?t=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) throw new Error(url);
    return await res.json();
  }

  async function renderStateLanguage() {
    try {
      const state = await fetchJson("coeziv_state.json");
      const en = getLang() === "en";
      const price = state.price_usd;
      const model = state.model_price_usd;
      const signal = String(state.signal || "flat").toLowerCase();
      const prob = state.signal_probability;
      const samples = state.signal_prob_samples || 0;
      const horizon = state.signal_prob_horizon_hours || 24;
      const drift = state.signal_expected_drift || {};
      const bd = state.signal_prob_breakdown || {};
      const costs = state.production_costs_usd || {};
      const avgCost = costs.average;
      const deviation = state.deviation_from_production;
      const fg = state.fg || {};

      const signalLabel = en
        ? signal === "long" ? "Context: upward pressure" : signal === "short" ? "Context: downside risk" : "Context: neutral"
        : signal === "long" ? "Context: presiune de creștere" : signal === "short" ? "Context: risc de scădere" : "Context: neutru";
      setText("signal-label", signalLabel);

      const regime = state.market_regime;
      let regimeLabel = "n/a", regimeCode = "";
      if (typeof regime === "string") regimeLabel = regime;
      else if (regime && typeof regime === "object") { regimeLabel = regime.label || regimeLabel; regimeCode = regime.code || ""; }
      const fullRegime = en ? `Market regime: ${regimeLabel}.` : `Regim de piață: ${regimeLabel}.`;
      const regimeEl = document.getElementById("regime-line");
      if (regimeEl) {
        regimeEl.setAttribute("data-full-regime", fullRegime);
        regimeEl.setAttribute("title", fullRegime);
        regimeEl.textContent = regimeCompactLabel(fullRegime, regimeCode);
      }

      if (en) {
        setText("message", `At the mechanism update, BTC was trading around ${fmtUsd(price)} USD. The mechanism reads the current context as neutral. Market flow is relatively balanced and weak. Liquidity is high, so price movement is more easily absorbed. Price remains close to the network’s estimated production-cost equilibrium. In similar historical contexts, the next ~24 hours typically ranged from about ${pct(drift.p10)}% in weaker scenarios to ${pct(drift.p90)}% in stronger scenarios, with a median near ${pct(drift.p50)}%. This is a structural risk interpretation, not financial advice.`);
        setText("signal-prob", `Historical probability: about ${pct(prob, 0)}% for price to move with the mechanism’s signal over the next ${horizon}h, based on ${samples} similar historical contexts.`);
        setText("signal-prob-breakdown", `Historical distribution over ${horizon}h: about ${pct(bd.in_direction, 0)}% with the signal, ${pct(bd.opposite, 0)}% against it, and ${pct(bd.flat, 0)}% neutral movement or noise.`);
        setText("drift-range", `Typical movement range (${horizon}h): between ~${pct(drift.p10)}% and ~${pct(drift.p90)}%, with a median near ~${pct(drift.p50)}%, based on similar historical contexts.`);
        setText("flow-line", flowEn(state.flow_bias, state.flow_strength));
        setText("liquidity-line", liquidityEn(state.liquidity_regime, state.liquidity_strength));
      } else {
        setText("message", state.message || "Nu există mesaj coeziv pentru acest moment.");
        setText("signal-prob", `Probabilitate istorică: ~${pct(prob, 0)}% ca prețul să se miște în direcția semnalului în următoarele ${horizon}h (bazat pe ${samples} situații similare).`);
        setText("signal-prob-breakdown", `Distribuție istorică (în ${horizon}h): ~${pct(bd.in_direction, 0)}% în direcția semnalului, ~${pct(bd.opposite, 0)}% contra semnalului și ~${pct(bd.flat, 0)}% mișcare neutră / zgomot.`);
        setText("drift-range", `Interval tipic de mișcare (${horizon}h): între ~${pct(drift.p10)}% și ~${pct(drift.p90)}% (mediană ~${pct(drift.p50)}%), bazat pe contexte similare din istoric.`);
        setText("flow-line", flowRo(state.flow_bias, state.flow_strength));
        setText("liquidity-line", liquidityRo(state.liquidity_regime, state.liquidity_strength));
      }
    } catch (e) {}
  }

  async function renderDailyLanguage() {
    try {
      const data = await fetchJson("daily_cohesiv_interpretation.json");
      const en = getLang() === "en";
      setText("daily-ai-state", en ? (data.general_state_en || data.general_state || "structural state pending") : (data.general_state || "stare în curs de interpretare"));
      setText("daily-ai-summary", en ? (data.plain_language_en || data.summary_en || data.plain_language || data.summary || "Daily interpretation pending.") : (data.plain_language || data.summary || "Interpretarea nu este încă disponibilă."));
      setText("daily-ai-participation", en ? (data.participation_en || data.participation || "–") : (data.participation || "–"));
      setText("daily-ai-risk", en ? (data.risk_window_en || data.risk_window || "–") : (data.risk_window || "–"));
      const fgRegime = en
        ? [data.fear_greed_en || data.fear_greed, data.market_regime_en || data.market_regime].filter(Boolean).join(" • ")
        : [data.fear_greed, data.market_regime].filter(Boolean).join(" • ");
      setText("daily-ai-fg-regime", fgRegime || "–");
      setText("daily-ai-watch", en ? (data.watch_next_en || data.watch_next || "–") : (data.watch_next || "–"));
      setText("daily-ai-footer", en ? (data.disclaimer_en || "Experimental structural interpretation, not financial advice.") : (data.disclaimer || "Interpretare structurală experimentală, nu recomandare financiară."));

      const panel = document.getElementById("daily-ai-full-panel");
      const btn = document.getElementById("daily-ai-full-toggle");
      const full = en ? (data.full_interpretation_en || data.full_interpretation || "") : (data.full_interpretation || "");
      if (panel && full) panel.textContent = full.replace(/^##\s+/gm, "").replace(/^###\s+/gm, "").replace(/\*\*(.*?)\*\*/g, "$1").trim();
      if (btn) btn.textContent = en ? (panel && panel.classList.contains("open") ? "Hide full interpretation" : "View full interpretation") : (panel && panel.classList.contains("open") ? "Ascunde interpretarea completă" : "Vezi interpretarea completă");
    } catch (e) {}
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
    collapseTimer = setTimeout(() => {
      const panel = document.getElementById("coeziv-accessibility-panel");
      if (panel) panel.classList.add("collapsed");
    }, 3500);
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
    panel.innerHTML = `
      <button type="button" class="coeziv-compact-toggle" title="Language / text size">RO/EN · A+</button>
      <div class="coeziv-panel-controls">
        <button type="button" data-lang="ro" title="Română">RO</button>
        <button type="button" data-lang="en" title="English">EN</button>
        <span class="sep" aria-hidden="true"></span>
        <button type="button" data-scale-action="down" title="Micșorează textul">A−</button>
        <button type="button" data-scale-action="up" title="Mărește textul">A+</button>
      </div>`;
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

  function scheduleRefresh() {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(applyAll, 120);
  }

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
    renderStateLanguage();
    renderDailyLanguage();
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