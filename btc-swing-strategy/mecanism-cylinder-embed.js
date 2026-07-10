(function () {
  const STYLE_ID = "cohesivx-daily-cylinder-style";
  const LANG_KEY = "coeziv_btc_lang";
  let LAST_DATA = null;

  function lang() {
    try { return localStorage.getItem(LANG_KEY) === "en" ? "en" : "ro"; }
    catch (_) { return "ro"; }
  }

  const T = {
    ro: {
      context: "CONTEXT", price: "PREȚ", participation: "PARTICIPARE", flow: "FLUX",
      liquidity: "LICHIDITATE", growth: "CONTEXT CREȘTERE", risk: "RISC STRUCTURAL",
      daily: "Interpretarea zilei", formula: "FORMULA ZILEI", full: "VEZI INTERPRETAREA COMPLETĂ",
      hide: "ASCUNDE INTERPRETAREA COMPLETĂ", waiting: "așteptăm date",
      footer: "Interpretare structurală experimentală, nu recomandare financiară."
    },
    en: {
      context: "CONTEXT", price: "PRICE", participation: "PARTICIPATION", flow: "FLOW",
      liquidity: "LIQUIDITY", growth: "GROWTH CONTEXT", risk: "STRUCTURAL RISK",
      daily: "Daily interpretation", formula: "DAILY FORMULA", full: "SEE FULL INTERPRETATION",
      hide: "HIDE FULL INTERPRETATION", waiting: "waiting data",
      footer: "Experimental structural interpretation, not financial advice."
    }
  };

  function pick(obj, roKey, enKey) {
    if (!obj) return "";
    return lang() === "en" ? (obj[enKey] || obj[roKey] || "") : (obj[roKey] || obj[enKey] || "");
  }

  function radarBlock(data) {
    if (!data) return {};
    return lang() === "en" ? (data.radar_info_en || data.radar_info || {}) : (data.radar_info || data.radar_info_en || {});
  }

  function fmtDate(data, radar) {
    const raw = radar?.date || data?.date || data?.generated_at || data?.last_analysis_at;
    if (!raw) return "–";
    if (/^\d{4}-\d{2}-\d{2}$/.test(String(raw))) {
      const [y, m, d] = raw.split("-");
      return `${d}.${m}.${y}`;
    }
    if (/^\d{2}\.\d{2}\.\d{4}$/.test(String(raw))) return raw;
    const d = new Date(raw);
    return Number.isNaN(d.getTime()) ? String(raw) : d.toLocaleDateString(lang() === "en" ? "en-GB" : "ro-RO");
  }

  function compact(text, max = 175) {
    const s = String(text || "").replace(/\s+/g, " ").trim();
    if (!s) return "–";
    return s.length > max ? s.slice(0, max - 1).trim() + "…" : s;
  }

  function contains(t, arr) { t = String(t || "").toLowerCase(); return arr.some(x => t.includes(x)); }

  function deriveFromText(text, type) {
    const en = lang() === "en";
    const t = String(text || "").toLowerCase();
    if (type === "context") return contains(t, ["creștere confirmată", "growth confirmed"]) ? (en ? "growth" : "creștere") : (en ? "neutral" : "neutru");
    if (type === "price") {
      if (contains(t, ["stabilizare joasă", "low stabilization"])) return en ? "low stabilization" : "stabilizare joasă";
      if (contains(t, ["recuperare lentă", "slow recovery"])) return en ? "slow recovery" : "recuperare lentă";
      if (contains(t, ["recuperare", "recovery"])) return en ? "recovery" : "recuperare";
      return en ? "low zone" : "zonă joasă";
    }
    if (type === "participation") {
      if (contains(t, ["coeziv", "cohesive"])) return en ? "cohesive" : "coezivă";
      if (contains(t, ["tension", "tense"])) return en ? "tense" : "tensionată";
      return "–";
    }
    if (type === "flow") {
      if (contains(t, ["neutru slab", "weak neutral"])) return en ? "weak neutral" : "neutru slab";
      if (contains(t, ["pozitiv", "positive"])) return en ? "positive" : "pozitiv";
      if (contains(t, ["negativ", "negative"])) return en ? "negative" : "negativ";
      return "–";
    }
    if (type === "liquidity") {
      if (contains(t, ["ridicată", "puternică", "high", "strong", "bună", "good"])) return en ? "good / high" : "bună / ridicată";
      if (contains(t, ["moderată", "moderate"])) return en ? "moderate" : "moderată";
      return "–";
    }
    if (type === "growth") {
      if (contains(t, ["fără contexte de creștere", "no growth contexts", "absent"])) return "absent";
      if (contains(t, ["prezent", "present"])) return en ? "present" : "prezent";
      return "–";
    }
    return "–";
  }

  function daysFrom(text) {
    const m = String(text || "").match(/(\d+)\s+de\s+zile|(\d+)\s+zile|(\d+)\s+days?/i);
    return m ? Number(m[1] || m[2] || m[3]) : null;
  }

  function derive(data) {
    const en = lang() === "en";
    const r = radarBlock(data);
    const all = [
      pick(data, "summary", "summary_en"), pick(data, "plain_language", "plain_language_en"),
      pick(data, "participation", "participation_en"), pick(data, "risk_window", "risk_window_en"),
      pick(data, "market_regime", "market_regime_en"), pick(data, "watch_next", "watch_next_en")
    ].join(" | ");

    const riskText = pick(data, "risk_window", "risk_window_en");
    const d = daysFrom(riskText || all);
    const reduced = /reduc|decreasing/i.test(riskText);
    const riskShort = d ? `${reduced ? (en ? "decreasing" : "în reducere") : (en ? "active" : "activ")} · ${d} ${en ? "days" : "zile"}` : "–";
    const formula = r.formula || pick(data, "formula", "formula_en") || pick(data, "plain_language", "plain_language_en") || pick(data, "summary", "summary_en");

    const growthText = r.growth_context || r.growth || deriveFromText(all, "growth");
    const participationText = r.participation || deriveFromText(all, "participation");
    const flowText = r.flow || deriveFromText(all, "flow");

    return {
      date: fmtDate(data, r),
      context: r.context || deriveFromText(all, "context"),
      price: r.price || deriveFromText(all, "price"),
      participation: participationText,
      flow: flowText,
      liquidity: r.liquidity || deriveFromText(all, "liquidity"),
      growth: growthText,
      risk: r.structural_risk || riskShort,
      formula: compact(formula, 185),
      footer: pick(data, "disclaimer", "disclaimer_en") || T[lang()].footer,
      full: pick(data, "full_interpretation", "full_interpretation_en"),
      visual: {
        growthPresent: /prezent|present/i.test(growthText),
        riskReduced: reduced || !!r.visual?.risk_reduced,
        participationCohesive: /coeziv|cohesive/i.test(participationText),
        flowPositive: /pozitiv|positive/i.test(flowText),
        flowNegative: /negativ|negative/i.test(flowText)
      }
    };
  }

  function fallbackInfo() {
    const l = lang();
    return {
      date: "–", context: "–", price: T[l].waiting, participation: "–", flow: "–", liquidity: "–", growth: "–", risk: "–",
      formula: l === "en" ? "The card will be completed after the first approved daily analysis." : "Cardul va fi completat după prima analiză zilnică aprobată.",
      footer: T[l].footer, full: "", visual: { growthPresent:false, riskReduced:false, participationCohesive:false, flowPositive:false, flowNegative:false }
    };
  }

  function css() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      #daily-ai-card{border-color:rgba(34,238,255,.38);box-shadow:0 14px 36px rgba(0,0,0,.42),0 0 34px rgba(0,210,255,.10)!important;background:#020617!important;contain:layout paint style;isolation:isolate;}
      #daily-cylinder-root{padding:0!important;background:#020617!important;}
      .cxdc,.cxdc *{box-sizing:border-box;backface-visibility:hidden;-webkit-backface-visibility:hidden;-webkit-tap-highlight-color:transparent;}
      .cxdc{position:relative;border-radius:22px;border:1px solid rgba(34,238,255,.24);overflow:hidden;padding:12px;background:#031222;contain:layout paint style;isolation:isolate;}
      .cxdc-core{position:relative;min-height:570px;border-radius:22px;border:1px solid rgba(34,238,255,.24);overflow:hidden;background:#03111f;box-shadow:inset 0 0 34px rgba(0,229,255,.08);contain:layout paint style;}
      .cxdc-core:before{content:"";position:absolute;inset:0;background:radial-gradient(circle at 74% 16%,rgba(0,245,255,.18),transparent 32%),linear-gradient(180deg,rgba(2,18,34,.86),rgba(0,10,22,.92));z-index:0;pointer-events:none;}
      .cxdc-labels{position:absolute;left:12px;top:72px;width:39%;max-width:160px;display:flex;flex-direction:column;gap:23px;z-index:5;}
      .cxdc-label{position:relative;min-height:43px;--c:#fff}.cxdc-label .k{font-size:clamp(15px,4.1vw,19px);font-weight:850;line-height:1.05;color:#fff}.cxdc-label .v{margin-top:4px;font-size:clamp(15px,4vw,19px);font-weight:760;line-height:1.08;color:var(--c);text-shadow:0 0 10px rgba(255,255,255,.25)}
      .cxdc-label .wire{position:absolute;left:calc(100% - 2px);top:14px;width:clamp(30px,8.8vw,46px);height:2px;background:linear-gradient(90deg,var(--c),rgba(255,255,255,.22));box-shadow:0 0 8px var(--c)}.cxdc-label .pin{position:absolute;left:calc(100% + clamp(26px,8.8vw,42px));top:8px;width:13px;height:13px;border-radius:50%;background:var(--c);box-shadow:0 0 12px var(--c)}
      .cxdc-label.context{--c:#bffcff}.cxdc-label.price{--c:#ffe600}.cxdc-label.part{--c:#ff9800}.cxdc-label.flow{--c:#14f4ff}.cxdc-label.liq{--c:#19a7ff}.cxdc-label.growth{--c:#ff3156}.cxdc-label.risk{--c:#ff6542}
      .cxdc-cylinder{position:absolute;right:8px;top:20px;width:57%;max-width:246px;height:520px;border-radius:40px 40px 24px 24px;border:1px solid rgba(175,245,255,.42);overflow:hidden;background:#062033;box-shadow:inset 0 0 42px rgba(0,240,255,.14),0 0 24px rgba(0,180,255,.10);z-index:2;contain:layout paint style;}
      .cxdc-cylinder:before,.cxdc-cylinder:after{content:"";position:absolute;left:18px;right:18px;height:34px;border-radius:50%;border:2px solid rgba(210,250,255,.72);box-shadow:0 0 14px rgba(210,250,255,.32),inset 0 0 10px rgba(120,230,255,.18);z-index:6}.cxdc-cylinder:before{top:26px}.cxdc-cylinder:after{bottom:34px;border-color:rgba(35,238,255,.54)}
      .cxdc-particles{position:absolute;inset:0;opacity:.18;background-image:radial-gradient(circle,rgba(125,250,255,.74) 1px,transparent 1.8px);background-size:22px 22px;animation:cxdcParticles 14s linear infinite}.cxdc-grid{position:absolute;inset:18px 10px;border-radius:26px;background:linear-gradient(90deg,rgba(200,240,255,.055) 1px,transparent 1px),linear-gradient(180deg,rgba(200,240,255,.045) 1px,transparent 1px);background-size:22px 22px;opacity:.38}
      .cxdc-beam{position:absolute;left:50%;top:34px;bottom:34px;width:8px;transform:translateX(-50%);background:linear-gradient(180deg,transparent,rgba(0,247,255,.55) 18%,rgba(170,250,255,.64) 50%,rgba(0,247,255,.55) 82%,transparent);box-shadow:0 0 18px rgba(0,240,255,.55),0 0 34px rgba(0,120,255,.22);animation:cxdcBeam 5.5s ease-in-out infinite;z-index:2}
      .cxdc-layer{position:absolute;left:17px;right:17px;height:46px;border-radius:50%;border:2px solid var(--c);background:rgba(255,255,255,.025);box-shadow:0 0 16px var(--c),inset 0 0 12px rgba(255,255,255,.08);animation:cxdcFloat 6s ease-in-out infinite;z-index:4}.cxdc-layer:before{content:"";position:absolute;left:12%;right:12%;top:50%;height:2px;background:var(--c);box-shadow:0 0 8px var(--c)}
      .cxdc-layer.context{top:66px;--c:#f8feff}.cxdc-layer.price{top:132px;--c:#ffe600}.cxdc-layer.part{top:198px;--c:#ff9800}.cxdc-layer.flow{top:264px;--c:#14f4ff}.cxdc-layer.liq{top:330px;--c:#19a7ff}.cxdc-layer.growth{top:396px;--c:#ff4968}.cxdc-risk{position:absolute;left:30px;right:30px;bottom:34px;height:54px;border-radius:50%;border:2px solid #ff4c25;background:rgba(255,75,32,.12);box-shadow:0 0 22px rgba(255,88,40,.58);z-index:5;animation:cxdcRisk 5s ease-in-out infinite}.cxdc-off{position:absolute;left:50%;top:412px;transform:translateX(-50%);font-size:40px;color:#ff3156;text-shadow:0 0 12px #ff3156;z-index:7}
      .cxdc-info{border-top:1px solid rgba(35,238,255,.18);margin-top:12px;padding-top:13px}.cxdc-head{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;margin-bottom:10px}.cxdc-date{font-size:18px;color:#a9bbd2}.cxdc-title{font-size:clamp(20px,6vw,26px);font-weight:700;color:#fff;text-align:right}.cxdc-formula{border-top:1px solid rgba(35,238,255,.14);padding-top:12px;margin-top:10px;font-size:14px;line-height:1.45;color:#d9e8f7}.cxdc-formula b{display:block;color:#00f7ff;letter-spacing:.08em;margin-bottom:5px}.cxdc-footer{margin-top:10px;font-size:12px;color:#8fa6bd}.cxdc-full{margin-top:12px;width:100%;border:1px solid rgba(0,247,255,.58);border-radius:16px;padding:13px 14px;color:#00f7ff;background:rgba(0,247,255,.08);font-weight:800;letter-spacing:.03em}.cxdc-panel{display:none;margin-top:10px;border-radius:16px;border:1px solid rgba(56,189,248,.26);background:rgba(15,23,42,.62);padding:12px 13px;font-size:13px;line-height:1.58;color:var(--text-main);white-space:pre-wrap}.cxdc-panel.open{display:block}
      @keyframes cxdcBeam{0%,100%{opacity:.84}50%{opacity:.58}}@keyframes cxdcParticles{to{background-position:0 -240px}}@keyframes cxdcFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-3px)}}@keyframes cxdcRisk{0%,100%{opacity:.78;transform:scale(1)}50%{opacity:.94;transform:scale(1.012)}}
      @media(max-width:760px),(pointer:coarse){#daily-ai-card,.cxdc,.cxdc-core,.cxdc-cylinder{background-color:#020617!important;box-shadow:none!important;transform:none!important;filter:none!important;will-change:auto!important}.cxdc-core:before{background:linear-gradient(180deg,rgba(3,18,34,.92),rgba(0,10,22,.96))!important}.cxdc-particles,.cxdc-beam,.cxdc-layer,.cxdc-risk{animation:none!important;transition:none!important;will-change:auto!important}.cxdc-particles{opacity:.08!important}.cxdc-beam{opacity:.34!important;background:linear-gradient(180deg,transparent,rgba(0,247,255,.30) 22%,rgba(125,245,255,.36) 50%,rgba(0,247,255,.30) 78%,transparent)!important;box-shadow:0 0 10px rgba(0,240,255,.28)!important}.cxdc-layer{box-shadow:0 0 8px rgba(120,240,255,.18),inset 0 0 8px rgba(255,255,255,.05)!important}.cxdc-risk{box-shadow:0 0 10px rgba(255,88,40,.28)!important}.cxdc-cylinder:before,.cxdc-cylinder:after{box-shadow:none!important;border-color:rgba(125,245,255,.42)!important}}
      @media(max-width:380px){.cxdc-core{min-height:555px}.cxdc-labels{left:10px;width:40%;gap:20px}.cxdc-label .wire,.cxdc-label .pin{display:none}.cxdc-cylinder{right:5px;width:56%;height:505px}}
    `;
    document.head.appendChild(style);
  }

  function render(info) {
    css();
    const root = document.getElementById("daily-cylinder-root") || document.getElementById("daily-ai-card")?.querySelector(".card-inner");
    if (!root) return;
    const l = lang();
    root.innerHTML = `
      <div class="cxdc">
        <div class="cxdc-core">
          <div class="cxdc-labels">
            <div class="cxdc-label context"><div class="k">${T[l].context}</div><div class="v">${info.context}</div><div class="wire"></div><div class="pin"></div></div>
            <div class="cxdc-label price"><div class="k">${T[l].price}</div><div class="v">${info.price}</div><div class="wire"></div><div class="pin"></div></div>
            <div class="cxdc-label part"><div class="k">${T[l].participation}</div><div class="v">${info.participation}</div><div class="wire"></div><div class="pin"></div></div>
            <div class="cxdc-label flow"><div class="k">${T[l].flow}</div><div class="v">${info.flow}</div><div class="wire"></div><div class="pin"></div></div>
            <div class="cxdc-label liq"><div class="k">${T[l].liquidity}</div><div class="v">${info.liquidity}</div><div class="wire"></div><div class="pin"></div></div>
            <div class="cxdc-label growth"><div class="k">${T[l].growth}</div><div class="v">${info.growth}</div><div class="wire"></div><div class="pin"></div></div>
            <div class="cxdc-label risk"><div class="k">${T[l].risk}</div><div class="v">${info.risk}</div><div class="wire"></div><div class="pin"></div></div>
          </div>
          <div class="cxdc-cylinder">
            <div class="cxdc-particles"></div><div class="cxdc-grid"></div><div class="cxdc-beam"></div>
            <div class="cxdc-layer context"></div><div class="cxdc-layer price"></div><div class="cxdc-layer part"></div><div class="cxdc-layer flow"></div><div class="cxdc-layer liq"></div><div class="cxdc-layer growth"></div>
            <div class="cxdc-off" style="display:${info.visual.growthPresent ? "none" : "block"}">⊘</div><div class="cxdc-risk" style="opacity:${info.visual.riskReduced ? ".55" : ".98"}"></div>
          </div>
        </div>
        <div class="cxdc-info">
          <div class="cxdc-head"><div class="cxdc-date">${info.date}</div><div class="cxdc-title">${T[l].daily}</div></div>
          <div class="cxdc-formula"><b>${T[l].formula}</b><span class="cxdc-summary">${info.formula}</span></div>
          <div class="cxdc-footer">${info.footer}</div>
          <button class="cxdc-full" type="button" aria-expanded="false" style="display:${info.full ? "block" : "none"}">${T[l].full}</button>
          <div class="cxdc-panel" aria-hidden="true"></div>
        </div>
      </div>`;
    const btn = root.querySelector(".cxdc-full");
    const panel = root.querySelector(".cxdc-panel");
    if (btn && panel && info.full) {
      panel.textContent = info.full;
      btn.addEventListener("click", function () {
        const open = panel.classList.toggle("open");
        btn.setAttribute("aria-expanded", open ? "true" : "false");
        panel.setAttribute("aria-hidden", open ? "false" : "true");
        btn.textContent = open ? T[lang()].hide : T[lang()].full;
      });
    }
  }

  function rerender() {
    render(LAST_DATA ? derive(LAST_DATA) : fallbackInfo());
  }

  window.COHESIVX_RENDER_DAILY_CYLINDER = function (data) {
    LAST_DATA = data || null;
    rerender();
  };

  document.addEventListener("click", function (ev) {
    if (ev.target && ev.target.closest && ev.target.closest("#coeziv-accessibility-panel")) {
      setTimeout(rerender, 180);
      setTimeout(rerender, 650);
    }
  }, true);

  window.addEventListener("storage", function (ev) {
    if (ev.key === LANG_KEY) rerender();
  });
})();
