/* CohesivX BTC — hydrated visual model summary, RO/EN
   Safe drop-in module. It does not modify the energy card.
   Load after mecanism.html main script and after i18n-en.js:
   <script src="./visual-model-summary.js" defer></script>
*/
(function () {
  "use strict";

  const LANG_KEY = "coeziv_btc_lang";
  const STATE_URLS = [
    "coeziv_state.json",
    "./coeziv_state.json",
    "/btc-swing-strategy/coeziv_state.json",
    "/coeziv_state.json"
  ];

  const USD = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  });

  const COPY = {
    ro: {
      livePrice: "Preț live",
      modelPrice: "Reper model",
      deviation: "Deviație",
      normal: "Normală",
      strong: "Puternică",
      extreme: "Extremă",
      intensityTitle: "Deviație față de reperul coeziv",
      modelBand: "Bandă statistică model",
      upperLimit: "Limită superioară",
      centralModel: "Preț coeziv central",
      liveMarker: "Preț live",
      lowerLimit: "Limită inferioară",
      confirmation: "Confirmare structurală istorică",
      sevenDays: "Orizont 7 zile",
      thirtyDays: "Orizont 30 zile",
      base: "Bază",
      events: "evenimente istorice similare",
      readFull: "Citește interpretarea completă",
      hideFull: "Ascunde interpretarea completă",
      readContext: "Citește contextul mecanismului",
      above: "peste",
      below: "sub",
      against: "față de reperul modelului",
      noteNormal: "Mișcare normală față de reper",
      noteStrong: "Mișcare controlată față de reper",
      noteExtreme: "Mișcare puternică — verifică lichiditatea",
      disclaimer: "Interpretare structurală, nu recomandare de tranzacționare.",
      unavailable: "Așteptăm date suficiente din mecanism."
    },
    en: {
      livePrice: "Live price",
      modelPrice: "Model reference",
      deviation: "Deviation",
      normal: "Normal",
      strong: "Strong",
      extreme: "Extreme",
      intensityTitle: "Deviation from cohesive reference",
      modelBand: "Statistical model band",
      upperLimit: "Upper limit",
      centralModel: "Central cohesive price",
      liveMarker: "Live price",
      lowerLimit: "Lower limit",
      confirmation: "Historical structural confirmation",
      sevenDays: "7-day horizon",
      thirtyDays: "30-day horizon",
      base: "Base",
      events: "similar historical events",
      readFull: "Read full interpretation",
      hideFull: "Hide full interpretation",
      readContext: "Read mechanism context",
      above: "above",
      below: "below",
      against: "versus the model reference",
      noteNormal: "Normal movement versus the reference",
      noteStrong: "Controlled movement versus the reference",
      noteExtreme: "Strong movement — check liquidity",
      disclaimer: "Structural interpretation, not a trading recommendation.",
      unavailable: "Waiting for sufficient mechanism data."
    }
  };

  function getLang() {
    try {
      return localStorage.getItem(LANG_KEY) === "en" ? "en" : "ro";
    } catch (_) {
      return "ro";
    }
  }

  function tx(key) {
    const lang = getLang();
    return (COPY[lang] && COPY[lang][key]) || COPY.ro[key] || key;
  }

  function $(id) {
    return document.getElementById(id);
  }

  function numberOrNull(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function formatUsd(value) {
    const n = numberOrNull(value);
    return n === null ? "n/a" : `${USD.format(n)} USD`;
  }

  function formatNumber(value) {
    const n = numberOrNull(value);
    return n === null ? "n/a" : USD.format(n);
  }

  function parseDisplayedLivePrice() {
    const el = $("live-price");
    if (!el) return null;
    const raw = String(el.textContent || "")
      .replace(/[^0-9.,-]/g, "")
      .replace(/,/g, "");
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : null;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function getDeviationStatus(absPct) {
    if (absPct < 3) {
      return {
        cls: "normal",
        label: tx("normal"),
        note: tx("noteNormal"),
        needle: clamp(absPct * 8, 0, 28)
      };
    }

    if (absPct < 10) {
      return {
        cls: "strong",
        label: tx("strong"),
        note: tx("noteStrong"),
        needle: clamp(35 + (absPct - 3) * 5, 35, 70)
      };
    }

    return {
      cls: "extreme",
      label: tx("extreme"),
      note: tx("noteExtreme"),
      needle: clamp(70 + (absPct - 10) * 1.5, 70, 96)
    };
  }

  async function fetchJsonFallback(urls) {
    let lastError = null;
    for (const url of urls) {
      try {
        const res = await fetch(`${url}?t=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) {
          lastError = new Error(`HTTP ${res.status} ${url}`);
          continue;
        }
        return await res.json();
      } catch (err) {
        lastError = err;
      }
    }
    throw lastError || new Error("coeziv_state.json unavailable");
  }

  function injectStyles() {
    if ($("cxv-style")) return;

    const style = document.createElement("style");
    style.id = "cxv-style";
    style.textContent = `
      .cx-visual-summary { margin: 12px 0 16px; text-align: left; }
      .cxv-metrics { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1px; background: rgba(30,44,66,.95); border-radius: 14px; overflow: hidden; margin-bottom: 12px; border: 1px solid rgba(56,189,248,.16); }
      .cxv-metric { background: rgba(16,25,43,.86); padding: 12px 7px; text-align: center; min-width: 0; }
      .cxv-label { font-size: 9px; text-transform: uppercase; letter-spacing: .08em; color: var(--text-soft, #8492a6); margin-bottom: 5px; }
      .cxv-value { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 14px; font-weight: 800; color: var(--text-main, #e8edf5); }
      .cxv-dev.extreme .cxv-value { color: #fb7185; }
      .cxv-dev.strong .cxv-value { color: #fbbf24; }
      .cxv-dev.normal .cxv-value { color: #67e8f9; }
      .cxv-intensity, .cxv-band, .cxv-confirm, .cxv-narrative { background: rgba(16,25,43,.72); border: 1px solid rgba(56,189,248,.22); border-radius: 17px; padding: 15px 16px; margin-bottom: 12px; box-shadow: inset 0 1px 0 rgba(255,255,255,.035); }
      .cxv-intensity.extreme { border-color: rgba(248,113,113,.36); }
      .cxv-section-title { font-size: 10px; color: var(--text-soft, #8492a6); text-transform: uppercase; letter-spacing: .10em; margin-bottom: 8px; }
      .cxv-headline { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 24px; font-weight: 850; color: #fb7185; line-height: 1; }
      .cxv-intensity.strong .cxv-headline { color: #fbbf24; }
      .cxv-intensity.normal .cxv-headline { color: #67e8f9; }
      .cxv-sub { font-size: 11px; color: var(--text-soft, #8492a6); margin-top: 5px; margin-bottom: 14px; line-height: 1.35; }
      .cxv-therm { height: 10px; border-radius: 999px; background: linear-gradient(90deg, #1c7a68 0%, #ffb84d 45%, #ff5d6c 78%, #ff5d6c 100%); position: relative; margin-bottom: 7px; }
      .cxv-therm span { position: absolute; top: -6px; width: 3px; height: 22px; background: #e8edf5; border-radius: 2px; box-shadow: 0 0 8px rgba(255,255,255,.45); transform: none !important; transition: none !important; animation: none !important; }
      .cxv-therm-labels { display: flex; justify-content: space-between; font-size: 9px; color: var(--text-soft, #8492a6); text-transform: uppercase; letter-spacing: .06em; }
      .cxv-status { margin-top: 12px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
      .cxv-status b { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; border: 1px solid rgba(248,113,113,.48); color: #fecaca; padding: 4px 9px; border-radius: 999px; }
      .cxv-status span { font-size: 11px; color: var(--text-soft, #8492a6); }
      .cxv-band-layout { display: flex; gap: 18px; align-items: stretch; }
      .cxv-band-track { position: relative; width: 14px; height: 220px; border-radius: 8px; background: linear-gradient(to top, #ff5d6c 0%, #ffb84d 38%, #1c7a68 58%, #35e0c2 100%); flex: 0 0 14px; margin-left: 8px; }
      .cxv-model-line { position: absolute; left: -7px; right: -7px; height: 2px; background: rgba(226,232,240,.65); border-radius: 2px; }
      .cxv-live-marker { position: absolute; left: 50%; transform: translate(-50%, 50%) !important; width: 24px; height: 24px; border-radius: 50%; background: #070b14; border: 3px solid #e8edf5; box-shadow: 0 0 0 2px rgba(15,23,42,.65); transition: none !important; animation: none !important; }
      .cxv-live-marker::after { content: ""; position: absolute; top: 50%; left: 50%; width: 7px; height: 7px; border-radius: 50%; background: #e8edf5; transform: translate(-50%, -50%); }
      .cxv-band-labels { flex: 1; display: flex; flex-direction: column; justify-content: space-between; padding: 1px 0; }
      .cxv-band-labels div { display: flex; flex-direction: column; gap: 2px; }
      .cxv-band-labels span { font-size: 10px; color: var(--text-soft, #8492a6); text-transform: uppercase; letter-spacing: .06em; }
      .cxv-band-labels b { font-size: 13px; color: var(--text-main, #e8edf5); }
      .cxv-confirm-sub { font-size: 11px; color: var(--text-soft, #8492a6); margin-bottom: 12px; }
      .cxv-confirm-row { margin-top: 10px; }
      .cxv-confirm-row div { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 12px; }
      .cxv-confirm-row b { color: #67e8f9; }
      .cxv-confirm-row i { display: block; height: 8px; background: rgba(255,255,255,.07); border-radius: 999px; overflow: hidden; }
      .cxv-confirm-row em { display: block; height: 100%; background: #1c7a68; border-radius: 999px; position: relative; }
      .cxv-confirm-row em::after { content: ""; position: absolute; right: 0; top: 0; width: 3px; height: 100%; background: #35e0c2; }
      .cxv-narrative { padding: 0; overflow: hidden; }
      .cxv-narrative summary { list-style: none; cursor: pointer; padding: 13px 15px; color: #67e8f9; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; display: flex; justify-content: space-between; }
      .cxv-narrative summary::-webkit-details-marker { display: none; }
      .cxv-narrative summary::after { content: "+"; color: var(--text-soft, #8492a6); font-size: 15px; }
      .cxv-narrative[open] summary::after { content: "−"; }
      .cxv-narrative div { padding: 0 15px 15px; color: var(--text-soft, #8492a6); font-size: 12px; line-height: 1.55; }
      body.light-mode .cxv-metric, body.light-mode .cxv-intensity, body.light-mode .cxv-band, body.light-mode .cxv-confirm, body.light-mode .cxv-narrative { background: rgba(255,255,255,.78); border-color: rgba(14,165,233,.25); }
      body.light-mode .cxv-metrics { background: rgba(148,163,184,.38); }
      body.light-mode .cxv-live-marker { background: #f8fafc; border-color: #0f172a; }
      body.light-mode .cxv-live-marker::after { background: #0f172a; }
      @media (max-width: 380px) { .cxv-value { font-size: 12px; } .cxv-metric { padding: 10px 5px; } .cxv-band-track { height: 190px; } }
    `;
    document.head.appendChild(style);
  }

  function ensureContainer() {
    let el = $("cx-visual-summary");
    if (el) return el;

    el = document.createElement("div");
    el.id = "cx-visual-summary";
    el.className = "cx-visual-summary";
    el.setAttribute("aria-live", "polite");

    const anchor = $("model-price-explanation") || $("prod-cost-line") || $("message");
    if (anchor && anchor.parentNode) {
      anchor.parentNode.insertBefore(el, anchor.nextSibling);
    }
    return el;
  }

  function buildNarrative(args) {
    const { diffPct, diffUsd, model, low, high, production, multiplier } = args;
    const sign = diffPct >= 0 ? "+" : "−";

    if (getLang() === "en") {
      return [
        `The live price is ${sign}${Math.abs(diffPct).toFixed(2)}% (${sign}${USD.format(Math.abs(diffUsd))} USD) ${tx("against")}.`,
        `The central cohesive price is ~${formatUsd(model)}${production ? `, anchored in the current production-cost area (~${formatUsd(production)})` : ""}${multiplier ? ` and a median historical price/cost multiplier of ${multiplier.toFixed(3)}×` : ""}.`,
        `The statistical model band is ~${formatUsd(low)} – ~${formatUsd(high)}.`,
        tx("disclaimer")
      ].join(" ");
    }

    return [
      `Prețul live este ${sign}${Math.abs(diffPct).toFixed(2)}% (${sign}${USD.format(Math.abs(diffUsd))} USD) ${tx("against")}.`,
      `Prețul coeziv central este ~${formatUsd(model)}${production ? `, ancorat în zona costului de producție actual (~${formatUsd(production)})` : ""}${multiplier ? ` și într-un multiplicator median istoric preț/cost de ${multiplier.toFixed(3)}×` : ""}.`,
      `Banda statistică a modelului este ~${formatUsd(low)} – ~${formatUsd(high)}.`,
      tx("disclaimer")
    ].join(" ");
  }

  function renderVisualSummary(state) {
    injectStyles();

    const el = ensureContainer();
    if (!el || !state) return;

    const model = numberOrNull(state.model_price_usd) || numberOrNull(state.price_usd);
    const live = parseDisplayedLivePrice() || numberOrNull(state.live_price_usd) || numberOrNull(state.price_usd);

    if (!model || !live) {
      el.innerHTML = `<div class="cxv-intensity"><div class="cxv-section-title">${tx("unavailable")}</div></div>`;
      return;
    }

    const bands = state.model_price_bands || {};
    const components = state.model_price_components || {};
    const structural = state.structural_confirmation || {};
    const h7 = structural.horizon_7d || {};
    const h30 = structural.horizon_30d || {};

    const diffUsd = live - model;
    const diffPct = (diffUsd / model) * 100;
    const absPct = Math.abs(diffPct);
    const st = getDeviationStatus(absPct);
    const sign = diffPct >= 0 ? "+" : "−";

    const production =
      numberOrNull(components.production_cost_anchor_usd) ||
      (state.production_costs_usd && numberOrNull(state.production_costs_usd.average));

    const multiplier = numberOrNull(components.historical_multiplier_p50);
    const low = numberOrNull(bands.p10) || Math.min(live, model) * 0.75;
    const high = numberOrNull(bands.p90) || Math.max(live, model) * 1.35;
    const span = Math.max(1, high - low);
    const livePos = clamp(((live - low) / span) * 100, 0, 100);
    const modelPos = clamp(((model - low) / span) * 100, 0, 100);

    const c7 = numberOrNull(h7.directional_hit_rate);
    const c30 = numberOrNull(h30.directional_hit_rate);
    const c7Pct = c7 !== null ? clamp(c7 * 100, 0, 100) : 0;
    const c30Pct = c30 !== null ? clamp(c30 * 100, 0, 100) : 0;

    const events =
      numberOrNull(h30.events) ||
      numberOrNull(structural.similar_context_samples) ||
      numberOrNull(components.similar_context_samples);

    const directionWord = diffPct >= 0 ? tx("above") : tx("below");
    const contextTextFromDom = $("message") ? String($("message").textContent || "").trim() : "";
    const contextText = contextTextFromDom && !/se încarcă|loading/i.test(contextTextFromDom)
      ? contextTextFromDom
      : (typeof state.message === "string" ? state.message.trim() : "");
    const contextBlock = contextText
      ? `<details class="cxv-narrative"><summary>${tx("readContext")}</summary><div>${contextText}</div></details>`
      : "";
    const narrative = buildNarrative({ diffPct, diffUsd, model, low, high, production, multiplier });

    const nextHtml = `
      <div class="cxv-metrics">
        <div class="cxv-metric"><div class="cxv-label">${tx("livePrice")}</div><div class="cxv-value">${formatNumber(live)}</div></div>
        <div class="cxv-metric"><div class="cxv-label">${tx("modelPrice")}</div><div class="cxv-value">${formatNumber(model)}</div></div>
        <div class="cxv-metric cxv-dev ${st.cls}"><div class="cxv-label">${tx("deviation")}</div><div class="cxv-value">${sign}${Math.abs(diffPct).toFixed(1)}%</div></div>
      </div>

      <div class="cxv-intensity ${st.cls}">
        <div class="cxv-section-title">${tx("intensityTitle")}</div>
        <div class="cxv-headline">${sign}${Math.abs(diffPct).toFixed(2)}%</div>
        <div class="cxv-sub">${sign}${USD.format(Math.abs(diffUsd))} USD ${directionWord} ${tx("against")}</div>
        <div class="cxv-therm"><span style="left:${st.needle}%"></span></div>
        <div class="cxv-therm-labels"><span>${tx("normal")}</span><span>${tx("strong")}</span><span>${tx("extreme")}</span></div>
        <div class="cxv-status"><b>${st.label}</b><span>${st.note}</span></div>
      </div>

      <div class="cxv-band">
        <div class="cxv-section-title">${tx("modelBand")}</div>
        <div class="cxv-band-layout">
          <div class="cxv-band-track"><i class="cxv-model-line" style="bottom:${modelPos}%"></i><i class="cxv-live-marker" style="bottom:${livePos}%"></i></div>
          <div class="cxv-band-labels">
            <div><span>${tx("upperLimit")}</span><b>${formatUsd(high)}</b></div>
            <div><span>${tx("centralModel")}</span><b>${formatUsd(model)}</b></div>
            <div><span>${tx("liveMarker")}</span><b>${formatUsd(live)}</b></div>
            <div><span>${tx("lowerLimit")}</span><b>${formatUsd(low)}</b></div>
          </div>
        </div>
      </div>

      <div class="cxv-confirm">
        <div class="cxv-section-title">${tx("confirmation")}</div>
        <div class="cxv-confirm-sub">${tx("base")}: ${events ? USD.format(events) : "n/a"} ${tx("events")}</div>
        <div class="cxv-confirm-row"><div><span>${tx("sevenDays")}</span><b>${c7 !== null ? Math.round(c7Pct) + "%" : "n/a"}</b></div><i><em style="width:${c7Pct}%"></em></i></div>
        <div class="cxv-confirm-row"><div><span>${tx("thirtyDays")}</span><b>${c30 !== null ? Math.round(c30Pct) + "%" : "n/a"}</b></div><i><em style="width:${c30Pct}%"></em></i></div>
      </div>

      ${contextBlock}
      <details class="cxv-narrative"><summary>${tx("readFull")}</summary><div>${narrative}</div></details>
    `;

    if (el.dataset.lastHtml !== nextHtml || el.innerHTML !== nextHtml) {
      el.dataset.lastHtml = nextHtml;
      el.innerHTML = nextHtml;
    }

    const oldText = $("model-price-explanation");
    if (oldText) oldText.style.display = "none";
    const oldMessage = $("message");
    if (oldMessage) oldMessage.style.display = "none";
  }

  async function renderFromStateFile() {
    try {
      const state = await fetchJsonFallback(STATE_URLS);
      window.COHESIVX_LAST_STATE = state;
      renderVisualSummary(state);
    } catch (err) {
      console.warn("CohesivX visual summary unavailable", err);
      if (window.COHESIVX_LAST_STATE) renderVisualSummary(window.COHESIVX_LAST_STATE);
    }
  }

  window.COHESIVX_RENDER_VISUAL_MODEL_SUMMARY = renderVisualSummary;
  window.COHESIVX_REFRESH_VISUAL_MODEL_SUMMARY = renderFromStateFile;

  function start() {
    injectStyles();
    renderFromStateFile();

    document.addEventListener("click", function (event) {
      const target = event.target;
      if (target && target.closest && target.closest("#coeziv-accessibility-panel")) {
        setTimeout(renderFromStateFile, 80);
        setTimeout(renderFromStateFile, 250);
        setTimeout(renderFromStateFile, 700);
      }
    }, true);

    setInterval(renderFromStateFile, 2500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
