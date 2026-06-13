/*
  Mecanism Coeziv BTC — language + accessibility module
  Author: Sergiu Bulboacă / Coeziv 3.14 project

  Purpose:
  - Adds RO/EN language switching without changing the data model.
  - Adds A-/A+ screen/text scaling for browser and Android WebView APK.
  - Keeps JSON values, prices, timestamps, scores and calculations unchanged.

  Usage inside mecanism.html:
    <script src="./i18n-en.js" defer></script>
*/
(function () {
  "use strict";

  const LANG_KEY = "coeziv_btc_lang";
  const SCALE_KEY = "coeziv_btc_scale";
  const SCALE_STEPS = [1, 1.15, 1.30];

  const PHRASE_MAP = new Map([
    ["MECANISM COEZIV BTC", "COHESIVE BTC MECHANISM"],
    ["Schimba tema aplicatiei", "Change app theme"],
    ["Actualizat aproximativ la 24h.", "Mechanism snapshot: updated approximately every 24h."],
    ["Context: neutru", "Context: neutral"],
    ["Context: presiune de creștere", "Context: upward pressure"],
    ["Context: risc de scădere", "Context: downside risk"],
    ["Regim de piață: n/a (așteptăm date suficiente din mecanism).", "Market regime: n/a (waiting for enough mechanism data)."],
    ["Preț live:", "Live price:"],
    ["Se compară prețul live cu prețul analizat de mecanism.", "Live price is compared with the mechanism snapshot price."],
    ["Status deviație: n/a (așteptăm un snapshot al modelului).", "Deviation status: n/a (waiting for a valid mechanism snapshot)."],
    ["Status deviație: Normală.", "Deviation status: Normal."],
    ["Status deviație: Controlată.", "Deviation status: Controlled."],
    ["Status deviație: Tensionată.", "Deviation status: Tense."],
    ["Status deviație: Extremă.", "Deviation status: Extreme."],
    ["Probabilitate istorică: n/a (așteptăm date suficiente din mecanism).", "Historical probability: n/a (waiting for enough mechanism data)."],
    ["Distribuție istorică: n/a (mecanismul nu are încă o descompunere robustă a probabilității).", "Historical distribution: n/a (the mechanism does not yet have a robust probability breakdown)."],
    ["Interval tipic de mișcare: n/a (așteptăm statistici suficiente din mecanism).", "Typical movement range: n/a (waiting for enough statistics from the mechanism)."],
    ["Flux de piață: n/a (așteptăm date despre flux).", "Market flow: n/a (waiting for flow data)."],
    ["Lichiditate piață: n/a (așteptăm date despre lichiditate).", "Market liquidity: n/a (waiting for liquidity data)."],
    ["Prag energetic BTC", "BTC energy threshold"],
    ["Cost energetic estimat de producție", "Estimated energy cost of production"],
    ["estimare", "estimate"],
    ["Miner eficient", "Efficient miner"],
    ["Miner mediu", "Average miner"],
    ["Miner scump", "Expensive miner"],
    ["Calcul bazat pe dificultatea rețelei, recompensa pe bloc, eficiența echipamentelor și prețul energiei. Nu reprezintă costul total contabil al minerilor.", "Calculation based on network difficulty, block reward, equipment efficiency and energy price. It does not represent miners’ full accounting cost."],
    ["Se încarcă graficul BTC. Dacă nu apare, poate fi o limitare temporară sau o extensie de browser.", "Loading the BTC chart. If it does not appear, this may be a temporary limitation or a browser extension issue."],
    ["Se încarcă mesajul coeziv...", "Loading cohesive message..."],
    ["Snapshot: n/a", "Snapshot: n/a"],
    ["Generat: n/a", "Generated: n/a"],
    ["INTERPRETARE COEZIVĂ ZILNICĂ", "DAILY COHESIVE INTERPRETATION"],
    ["Explicație naturală a stării structurale observate de mecanism.", "Natural-language explanation of the structural state observed by the mechanism."],
    ["Se încarcă...", "Loading..."],
    ["Interpretarea zilnică va apărea după prima analiză aprobată.", "The daily interpretation will appear after the first approved analysis."],
    ["Participare", "Participation"],
    ["Fereastră risc", "Risk window"],
    ["Fear & Greed / Regim", "Fear & Greed / Regime"],
    ["Ce urmărim", "What to watch"],
    ["Interpretare structurală experimentală, nu recomandare financiară.", "Experimental structural interpretation, not financial advice."],
    ["Vezi interpretarea completă", "View full interpretation"],
    ["Ascunde interpretarea completă", "Hide full interpretation"],
    ["ISTORIC SEMNALE", "SIGNAL HISTORY"],
    ["Istoricul nu este disponibil încă. Va apărea după următorul update al mecanismului.", "History is not available yet. It will appear after the next mechanism update."],
    ["Preț model la snapshot", "Model price at snapshot"],
    ["COEZIUNE_PARTICIPATIVA", "PARTICIPATION_COHESION"],
    ["FEREASTRA_DE_RISC", "RISK_WINDOW"],
    ["Nu este o recomandare de tranzacționare.", "This is not a trading recommendation."],
    ["Nu este o recomandare financiară.", "This is not financial advice."]
  ]);

  const REGEX_RULES = [
    [/Prețul live este egal cu prețul analizat de mecanism\./g, "The live price is equal to the price analyzed by the mechanism."],
    [/Preț live încărcat\. Așteptăm un snapshot de preț valid din mecanism\./g, "Live price loaded. Waiting for a valid price snapshot from the mechanism."],
    [/Nu am putut încărca snapshotul de preț\. Așteptăm ca mecanismul să revină\./g, "Could not load the price snapshot. Waiting for the mechanism to recover."],
    [/Nu am reușit să obținem mesajul coeziv\./g, "Could not retrieve the cohesive message."],
    [/Nu am putut încărca coeziv_state\.json\. Verifică workflow-ul de backend\./g, "Could not load coeziv_state.json. Check the backend workflow."],
    [/Diferență foarte mică \(zgomot normal de piață\)\./g, "Very small difference (normal market noise)."],
    [/Mișcare mică; semnalul mecanismului rămâne reperul principal\./g, "Small move; the mechanism signal remains the main reference."],
    [/Mișcare relevantă; poți ajusta timing-ul intrării sau ieșirii\./g, "Relevant move; entry or exit timing may be adjusted."],
    [/Mișcare puternică; verifică contextul și lichiditatea\./g, "Strong move; check context and liquidity."],
    [/Prețul live este ([+−\-]?[0-9.,]+)% \(([+−\-]?[0-9.,]+) USD\) peste prețul mecanismului\./g, "The live price is $1% ($2 USD) above the mechanism price."],
    [/Prețul live este ([+−\-]?[0-9.,]+)% \(([+−\-]?[0-9.,]+) USD\) sub prețul mecanismului\./g, "The live price is $1% ($2 USD) below the mechanism price."],
    [/Status deviație: Normală \(fără abatere față de model\)\./g, "Deviation status: Normal (no deviation from the model)."],
    [/Status deviație: n\/a \(nu avem date suficiente de la mecanism\)\./g, "Deviation status: n/a (not enough mechanism data)."],
    [/Probabilitate istorică:/g, "Historical probability:"],
    [/Distribuție istorică:/g, "Historical distribution:"],
    [/Interval tipic de mișcare:/g, "Typical movement range:"],
    [/Flux de piață:/g, "Market flow:"],
    [/Lichiditate piață:/g, "Market liquidity:"],
    [/Context neutru/g, "Neutral context"],
    [/Presiune de creștere/g, "Upward pressure"],
    [/Risc de scădere/g, "Downside risk"],
    [/neutru/g, "neutral"],
    [/scădere/g, "decline"],
    [/creștere/g, "growth"],
    [/ridicată/g, "high"],
    [/scăzută/g, "low"],
    [/normală/g, "normal"],
    [/moderată/g, "moderate"],
    [/puternică/g, "strong"],
    [/slab/g, "weak"],
    [/participare tensionată/g, "tense participation"],
    [/participare coezivă/g, "cohesive participation"],
    [/Regim neutru/g, "Neutral regime"],
    [/tranziție/g, "transition"],
    [/deviație normală față de model/g, "normal deviation from the model"],
    [/flux tensionat/g, "tense flow"],
    [/Nu recomandare de tranzacționare/g, "Not a trading recommendation"],
    [/nu recomandare de tranzacționare/g, "not a trading recommendation"]
  ];

  function getLang() {
    return localStorage.getItem(LANG_KEY) || window.COEZIV_DEFAULT_LANG || "ro";
  }

  function setLang(lang) {
    const next = lang === "en" ? "en" : "ro";
    localStorage.setItem(LANG_KEY, next);
    document.documentElement.setAttribute("lang", next);
    document.body.setAttribute("data-lang", next);
    applyTranslation();
    updateButtonState();
  }

  function getScale() {
    const raw = parseFloat(localStorage.getItem(SCALE_KEY) || "1");
    return SCALE_STEPS.includes(raw) ? raw : 1;
  }

  function setScale(scale) {
    let next = Number(scale);
    if (!SCALE_STEPS.includes(next)) next = 1;
    localStorage.setItem(SCALE_KEY, String(next));
    applyScale();
    updateScaleButtons();
  }

  function bumpScale(direction) {
    const current = getScale();
    const index = Math.max(0, SCALE_STEPS.indexOf(current));
    const nextIndex = Math.min(SCALE_STEPS.length - 1, Math.max(0, index + direction));
    setScale(SCALE_STEPS[nextIndex]);
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
    const trimmed = value.trim();
    if (PHRASE_MAP.has(trimmed)) return value.replace(trimmed, PHRASE_MAP.get(trimmed));
    let out = value;
    for (const [pattern, replacement] of REGEX_RULES) out = out.replace(pattern, replacement);
    return out;
  }

  function translateNode(node) {
    if (!node || node.nodeType !== Node.TEXT_NODE) return;
    const original = node.__coezivRoText || node.nodeValue;
    if (!node.__coezivRoText) node.__coezivRoText = original;
    node.nodeValue = getLang() === "en" ? translateText(original) : original;
  }

  function walk(root) {
    if (!root) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent) return NodeFilter.FILTER_REJECT;
        if (parent.closest && parent.closest("#coeziv-accessibility-panel")) return NodeFilter.FILTER_REJECT;
        const tag = parent.tagName;
        if (tag === "SCRIPT" || tag === "STYLE" || tag === "NOSCRIPT") return NodeFilter.FILTER_REJECT;
        if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(translateNode);
  }

  function translateAttributes() {
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

  function applyTranslation() {
    if (!document.body) return;
    walk(document.body);
    translateAttributes();
  }

  function injectStyles() {
    if (document.getElementById("coeziv-i18n-style")) return;
    const style = document.createElement("style");
    style.id = "coeziv-i18n-style";
    style.textContent = `
      #coeziv-accessibility-panel {
        position: fixed;
        right: 12px;
        bottom: 14px;
        z-index: 2147483647;
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,.35);
        background: rgba(15,23,42,.92);
        color: #e5e7eb;
        box-shadow: 0 14px 42px rgba(0,0,0,.38);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 11px;
        line-height: 1;
      }
      #coeziv-accessibility-panel button {
        border: 0;
        border-radius: 999px;
        background: transparent;
        color: inherit;
        min-width: 34px;
        height: 30px;
        padding: 0 9px;
        cursor: pointer;
        font-size: 12px;
        font-weight: 800;
        line-height: 30px;
        text-align: center;
      }
      #coeziv-accessibility-panel button.active {
        background: rgba(56,189,248,.22);
        color: #7dd3fc;
      }
      #coeziv-accessibility-panel .sep {
        width: 1px;
        height: 20px;
        background: rgba(148,163,184,.28);
      }
      body.light-mode #coeziv-accessibility-panel {
        background: rgba(255,255,255,.92);
        color: #0f172a;
      }
      @media (max-width: 420px) {
        #coeziv-accessibility-panel {
          right: 8px;
          bottom: 10px;
          gap: 3px;
          padding: 5px;
        }
        #coeziv-accessibility-panel button {
          min-width: 30px;
          height: 28px;
          padding: 0 7px;
          font-size: 11px;
          line-height: 28px;
        }
      }
    `;
    document.head.appendChild(style);
  }

  function injectPanel() {
    if (document.getElementById("coeziv-accessibility-panel")) return;
    const panel = document.createElement("div");
    panel.id = "coeziv-accessibility-panel";
    panel.setAttribute("aria-label", "Language and text size controls");
    panel.innerHTML = `
      <button type="button" data-lang="ro" title="Română">RO</button>
      <button type="button" data-lang="en" title="English">EN</button>
      <span class="sep" aria-hidden="true"></span>
      <button type="button" data-scale-action="down" title="Micșorează textul">A−</button>
      <button type="button" data-scale-action="up" title="Mărește textul">A+</button>
    `;
    panel.addEventListener("click", (event) => {
      const langBtn = event.target.closest("button[data-lang]");
      if (langBtn) {
        setLang(langBtn.dataset.lang);
        return;
      }
      const scaleBtn = event.target.closest("button[data-scale-action]");
      if (scaleBtn) {
        bumpScale(scaleBtn.dataset.scaleAction === "up" ? 1 : -1);
      }
    });
    document.body.appendChild(panel);
    updateButtonState();
    updateScaleButtons();
  }

  function updateButtonState() {
    const lang = getLang();
    document.querySelectorAll("#coeziv-accessibility-panel button[data-lang]").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.lang === lang);
    });
  }

  function updateScaleButtons() {
    const scale = getScale();
    document.querySelectorAll("#coeziv-accessibility-panel button[data-scale-action]").forEach((btn) => {
      btn.classList.remove("active");
    });
    if (scale > 1) {
      const up = document.querySelector("#coeziv-accessibility-panel button[data-scale-action='up']");
      if (up) up.classList.add("active");
    }
  }

  function observeChanges() {
    const observer = new MutationObserver((mutations) => {
      let shouldTranslate = false;
      for (const mutation of mutations) {
        if (mutation.target && mutation.target.closest && mutation.target.closest("#coeziv-accessibility-panel")) continue;
        if (mutation.type === "childList" || mutation.type === "characterData") {
          shouldTranslate = true;
          break;
        }
      }
      if (shouldTranslate && getLang() === "en") requestAnimationFrame(applyTranslation);
    });
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
  }

  function boot() {
    injectStyles();
    injectPanel();
    applyScale();
    setLang(getLang());
    observeChanges();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

  window.CoezivI18n = {
    setLang,
    getLang,
    setScale,
    getScale,
    applyScale,
    applyTranslation,
    translateText
  };
})();
