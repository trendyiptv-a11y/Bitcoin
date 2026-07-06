(function () {
  const STYLE_ID = "cohesivx-daily-cylinder-style";

  function lang() {
    try { return localStorage.getItem("coeziv_btc_lang") === "en" ? "en" : "ro"; }
    catch (_) { return "ro"; }
  }

  const T = {
    ro: {
      context: "CONTEXT", price: "PREȚ", participation: "PARTICIPARE", flow: "FLUX",
      liquidity: "LICHIDITATE", growth: "CONTEXT CREȘTERE", risk: "RISC STRUCTURAL",
      daily: "Interpretarea zilei", formula: "FORMULA ZILEI", full: "VEZI INTERPRETAREA COMPLETĂ",
      hide: "ASCUNDE INTERPRETAREA COMPLETĂ", waiting: "așteptăm date", unavailable: "–",
      footer: "Interpretare structurală experimentală, nu recomandare financiară."
    },
    en: {
      context: "CONTEXT", price: "PRICE", participation: "PARTICIPATION", flow: "FLOW",
      liquidity: "LIQUIDITY", growth: "GROWTH CONTEXT", risk: "STRUCTURAL RISK",
      daily: "Daily interpretation", formula: "DAILY FORMULA", full: "SEE FULL INTERPRETATION",
      hide: "HIDE FULL INTERPRETATION", waiting: "waiting data", unavailable: "–",
      footer: "Experimental structural interpretation, not financial advice."
    }
  };

  function pick(obj, roKey, enKey) {
    const l = lang();
    if (!obj) return "";
    if (l === "en") return obj[enKey] || obj[roKey] || "";
    return obj[roKey] || obj[enKey] || "";
  }

  function fmtDate(data) {
    const raw = data && (data.date || data.generated_at || data.last_analysis_at);
    if (!raw) return "–";
    if (/^\d{4}-\d{2}-\d{2}$/.test(String(raw))) {
      const [y,m,d] = raw.split("-");
      return `${d}.${m}.${y}`;
    }
    if (/^\d{2}\.\d{2}\.\d{4}$/.test(String(raw))) return raw;
    const d = new Date(raw);
    if (Number.isNaN(d.getTime())) return String(raw);
    return d.toLocaleDateString(lang() === "en" ? "en-GB" : "ro-RO");
  }

  function firstSentence(text, max = 120) {
    const s = String(text || "").replace(/\s+/g, " ").trim();
    if (!s) return "–";
    const cut = s.match(/^(.{20,}?[.!?])\s/);
    const out = cut ? cut[1] : s;
    return out.length > max ? out.slice(0, max - 1).trim() + "…" : out;
  }

  function includesAny(t, arr) { return arr.some(x => t.includes(x)); }

  function valFromText(text, type) {
    const en = lang() === "en";
    const t = String(text || "").toLowerCase();
    if (type === "context") {
      if (includesAny(t, ["creștere confirmată", "ridicare confirmată", "growth confirmed"])) return en ? "growth" : "creștere";
      return en ? "neutral" : "neutru";
    }
    if (type === "price") {
      if (includesAny(t, ["recuperare lentă", "slow recovery"])) return en ? "slow recovery" : "recuperare lentă";
      if (includesAny(t, ["recuperare", "recovery"])) return en ? "recovery" : "recuperare";
      if (includesAny(t, ["ridicare", "lifting", "lift"])) return en ? "lifting" : "ridicare";
      if (includesAny(t, ["slăbire", "weakening"])) return en ? "weakening" : "slăbire";
      return en ? "low zone" : "zonă joasă";
    }
    if (type === "participation") {
      if (includesAny(t, ["coeziv", "cohesive"])) return en ? "cohesive" : "coezivă";
      if (includesAny(t, ["tension", "tense"])) return en ? "tense" : "tensionată";
      return "–";
    }
    if (type === "flow") {
      if (includesAny(t, ["neutru slab", "weak neutral"])) return en ? "weak neutral" : "neutru slab";
      if (includesAny(t, ["pozitiv moderat", "moderately positive", "moderate positive"])) return en ? "moderate positive" : "pozitiv moderat";
      if (includesAny(t, ["negativ moderat", "moderately negative", "moderate negative"])) return en ? "moderate negative" : "negativ moderat";
      if (includesAny(t, ["pozitiv", "positive"])) return en ? "positive" : "pozitiv";
      if (includesAny(t, ["negativ", "negative"])) return en ? "negative" : "negativ";
      return "–";
    }
    if (type === "liquidity") {
      if (includesAny(t, ["ridicată", "puternică", "high", "strong", "bună", "good"])) return en ? "good / high" : "bună / ridicată";
      if (includesAny(t, ["moderată", "moderate"])) return en ? "moderate" : "moderată";
      return "–";
    }
    if (type === "growth") {
      if (includesAny(t, ["0 contexte de creștere", "fără contexte de creștere", "no growth contexts", "absent"])) return en ? "absent" : "absent";
      if (includesAny(t, ["prezent", "present"])) return en ? "present" : "prezent";
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
    const all = [
      pick(data, "summary", "summary_en"), pick(data, "plain_language", "plain_language_en"),
      pick(data, "participation", "participation_en"), pick(data, "risk_window", "risk_window_en"),
      pick(data, "market_regime", "market_regime_en"), pick(data, "watch_next", "watch_next_en")
    ].join(" | ");

    const riskText = pick(data, "risk_window", "risk_window_en");
    const d = daysFrom(riskText || all);
    const reduced = /reduc|decreasing/i.test(riskText);
    const riskShort = d ? `${reduced ? (en ? "decreasing" : "în reducere") : (en ? "active" : "activ")} · ${d} ${en ? "days" : "zile"}` : "–";

    const formula = pick(data, "formula", "short_formula") || pick(data, "summary", "summary_en") || pick(data, "plain_language", "plain_language_en");

    return {
      date: fmtDate(data),
      context: data?.radar_info?.context || valFromText(all, "context"),
      price: data?.radar_info?.price || valFromText(all, "price"),
      participation: data?.radar_info?.participation || valFromText(all, "participation"),
      flow: data?.radar_info?.flow || valFromText(all, "flow"),
      liquidity: data?.radar_info?.liquidity || valFromText(all, "liquidity"),
      growth: data?.radar_info?.growth_context || data?.radar_info?.growth || valFromText(all, "growth"),
      risk: data?.radar_info?.structural_risk || riskShort,
      formula: firstSentence(formula, 170),
      footer: pick(data, "disclaimer", "disclaimer_en") || T[lang()].footer,
      full: pick(data, "full_interpretation", "full_interpretation_en"),
      visual: {
        growthPresent: /prezent|present/i.test(data?.radar_info?.growth_context || data?.radar_info?.growth || valFromText(all, "growth")),
        riskReduced: reduced,
        participationCohesive: /coeziv|cohesive/i.test(data?.radar_info?.participation || valFromText(all, "participation")),
        flowPositive: /pozitiv|positive/i.test(data?.radar_info?.flow || valFromText(all, "flow")),
        flowNegative: /negativ|negative/i.test(data?.radar_info?.flow || valFromText(all, "flow"))
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
      #daily-ai-card{border-color:rgba(34,238,255,.38);box-shadow:0 14px 36px rgba(0,0,0,.42),0 0 34px rgba(0,210,255,.10)!important;}
      #daily-cylinder-root{padding:0!important;}
      .cxdc{position:relative;border-radius:22px;border:1px solid rgba(34,238,255,.24);overflow:hidden;padding:12px;background:linear-gradient(180deg,rgba(3,18,34,.96),rgba(1,10,22,.98));}
      .cxdc-core{position:relative;min-height:570px;border-radius:22px;border:1px solid rgba(34,238,255,.24);overflow:hidden;background:radial-gradient(circle at 74% 16%,rgba(0,245,255,.22),transparent 32%),linear-gradient(180deg,rgba(2,18,34,.86),rgba(0,10,22,.92));box-shadow:inset 0 0 44px rgba(0,229,255,.09);}
      .cxdc-labels{position:absolute;left:12px;top:72px;width:39%;max-width:160px;display:flex;flex-direction:column;gap:23px;z-index:5;}
      .cxdc-label{position:relative;min-height:43px;--c:#fff}.cxdc-label .k{font-size:clamp(15px,4.1vw,19px);font-weight:850;line-height:1.05;color:#fff}.cxdc-label .v{margin-top:4px;font-size:clamp(15px,4vw,19px);font-weight:760;line-height:1.08;color:var(--c);text-shadow:0 0 14px color-mix(in srgb,var(--c) 70%,transparent)}
      .cxdc-label .wire{position:absolute;left:calc(100% - 2px);top:14px;width:clamp(30px,8.8vw,46px);height:2px;background:linear-gradient(90deg,var(--c),rgba(255,255,255,.22));box-shadow:0 0 12px var(--c)}.cxdc-label .pin{position:absolute;left:calc(100% + clamp(26px,8.8vw,42px));top:8px;width:13px;height:13px;border-radius:50%;background:var(--c);box-shadow:0 0 18px var(--c),0 0 34px color-mix(in srgb,var(--c) 56%,transparent)}
      .cxdc-label.context{--c:#bffcff}.cxdc-label.price{--c:#ffe600}.cxdc-label.part{--c:#ff9800}.cxdc-label.flow{--c:#14f4ff}.cxdc-label.liq{--c:#19a7ff}.cxdc-label.growth{--c:#ff3156}.cxdc-label.risk{--c:#ff6542}
      .cxdc-cylinder{position:absolute;right:8px;top:20px;width:57%;max-width:246px;height:520px;border-radius:40px 40px 24px 24px;border:1px solid rgba(175,245,255,.42);overflow:hidden;background:linear-gradient(90deg,rgba(255,255,255,.07),rgba(0,220,255,.14) 48%,rgba(255,255,255,.03));box-shadow:inset 0 0 62px rgba(0,240,255,.20),0 0 44px rgba(0,180,255,.16)}
      .cxdc-cylinder:before,.cxdc-cylinder:after{content:"";position:absolute;left:18px;right:18px;height:34px;border-radius:50%;border:2px solid rgba(242,255,255,.88);box-shadow:0 0 24px rgba(210,250,255,.60),inset 0 0 16px rgba(120,230,255,.24);z-index:6}.cxdc-cylinder:before{top:26px}.cxdc-cylinder:after{bottom:34px;border-color:rgba(35,238,255,.60)}
      .cxdc-particles{position:absolute;inset:0;opacity:.30;background-image:radial-gradient(circle,rgba(125,250,255,.92) 1px,transparent 1.8px);background-size:22px 22px;animation:cxdcParticles 10s linear infinite}.cxdc-grid{position:absolute;inset:18px 10px 18px 10px;border-radius:26px;background:linear-gradient(90deg,rgba(200,240,255,.055) 1px,transparent 1px),linear-gradient(180deg,rgba(200,240,255,.045) 1px,transparent 1px);background-size:22px 22px;opacity:.45}
      .cxdc-beam{position:absolute;left:50%;top:34px;bottom:34px;width:10px;transform:translateX(-50%);background:linear-gradient(180deg,transparent,#00f7ff 18%,#fff 50%,#00f7ff 82%,transparent);box-shadow:0 0 24px #00f7ff,0 0 56px rgba(0,240,255,.95),0 0 88px rgba(0,120,255,.28);animation:cxdcBeam 3.8s ease-in-out infinite;z-index:2}
      .cxdc-layer{position:absolute;left:17px;right:17px;height:46px;border-radius:50%;border:2px solid var(--c);background:radial-gradient(ellipse at center,color-mix(in srgb,var(--c) 22%,transparent),transparent 70%);box-shadow:0 0 22px color-mix(in srgb,var(--c) 80%,transparent),0 0 48px color-mix(in srgb,var(--c) 30%,transparent),inset 0 0 18px color-mix(in srgb,var(--c) 32%,transparent);animation:cxdcFloat 4.4s ease-in-out infinite}.cxdc-layer:before{content:"";position:absolute;left:12%;right:12%;top:50%;height:2px;transform:translateY(-50%);background:var(--c);box-shadow:0 0 10px var(--c)}
      .cxdc-layer.context{top:66px;--c:#f8feff}.cxdc-layer.price{top:132px;--c:#ffe600}.cxdc-layer.part{top:198px;--c:#ff9800}.cxdc-layer.flow{top:264px;--c:#14f4ff}.cxdc-layer.liq{top:330px;--c:#19a7ff}.cxdc-layer.growth{top:396px;--c:#ff4968}.cxdc-risk{position:absolute;left:30px;right:30px;bottom:34px;height:54px;border-radius:50%;border:2px solid #ff4c25;background:radial-gradient(ellipse at center,rgba(255,75,32,.62),transparent 72%);box-shadow:0 0 34px rgba(255,88,40,.85),0 0 64px rgba(255,50,35,.26);z-index:5;animation:cxdcRisk 3.2s ease-in-out infinite}.cxdc-off{position:absolute;left:50%;top:412px;transform:translateX(-50%);font-size:40px;color:#ff3156;text-shadow:0 0 16px #ff3156,0 0 34px rgba(255,49,86,.72);z-index:7;animation:cxdcBlink 2.8s ease-in-out infinite}
      .cxdc-wave{position:absolute;left:31px;right:31px;height:36px;z-index:7}.cxdc-wave svg{width:100%;height:100%}.cxdc-wave path{fill:none;stroke-width:2.5;filter:drop-shadow(0 0 8px currentColor);stroke-dasharray:170;animation:cxdcDash 5s linear infinite}.cxdc-wave.price{top:139px;color:#ffe600}.cxdc-wave.part{top:205px;color:#ff9800}.cxdc-wave.flow{top:271px;color:#14f4ff}.cxdc-wave.price path{stroke:#ffe600}.cxdc-wave.part path{stroke:#ff9800}.cxdc-wave.flow path{stroke:#14f4ff}
      .cxdc-info{border-top:1px solid rgba(35,238,255,.18);margin-top:12px;padding-top:13px}.cxdc-head{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;margin-bottom:10px}.cxdc-date{font-size:18px;color:#a9bbd2}.cxdc-title{font-size:clamp(20px,6vw,26px);font-weight:700;color:#fff;text-align:right}.cxdc-formula{border-top:1px solid rgba(35,238,255,.14);padding-top:12px;margin-top:10px;font-size:14px;line-height:1.45;color:#d9e8f7}.cxdc-formula b{display:block;color:#00f7ff;letter-spacing:.08em;margin-bottom:5px;text-shadow:0 0 12px rgba(0,247,255,.34)}.cxdc-footer{margin-top:10px;font-size:12px;color:#8fa6bd}
      @keyframes cxdcBeam{0%,100%{opacity:.92}50%{opacity:.66}}@keyframes cxdcParticles{to{background-position:0 -240px}}@keyframes cxdcFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}@keyframes cxdcRisk{0%,100%{opacity:.82;transform:scale(1)}50%{opacity:1;transform:scale(1.025)}}@keyframes cxdcDash{to{stroke-dashoffset:-340}}@keyframes cxdcBlink{0%,100%{opacity:.64}50%{opacity:1}}
      @media(max-width:380px){.cxdc-core{min-height:555px}.cxdc-labels{left:10px;width:40%;gap:20px}.cxdc-label .wire,.cxdc-label .pin{display:none}.cxdc-cylinder{right:5px;width:56%;height:505px}.cxdc-title{font-size:22px}.cxdc-date{font-size:16px}}
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
            <div class="cxdc-wave price"><svg viewBox="0 0 160 36" preserveAspectRatio="none"><path d="M0 24 L18 24 L30 25 L42 20 L56 21 L70 16 L84 17 L98 11 L112 13 L126 8 L142 10 L160 4"/></svg></div>
            <div class="cxdc-wave part"><svg viewBox="0 0 160 36" preserveAspectRatio="none"><path d="M0 18 L10 20 L18 13 L26 22 L34 16 L42 19 L50 15 L58 24 L66 17 L74 20 L82 14 L90 22 L98 16 L106 21 L114 15 L122 19 L130 16 L138 23 L146 14 L154 19 L160 17"/></svg></div>
            <div class="cxdc-wave flow"><svg viewBox="0 0 160 36" preserveAspectRatio="none"><path d="M0 20 L18 20 L34 16 L50 20 L66 19 L82 15 L98 21 L114 20 L130 16 L146 20 L160 20"/></svg></div>
            <div class="cxdc-off" style="display:${info.visual.growthPresent ? "none" : "block"}">⊘</div><div class="cxdc-risk" style="opacity:${info.visual.riskReduced ? ".55" : ".98"}"></div>
          </div>
        </div>
        <div class="cxdc-info">
          <div class="cxdc-head"><div class="cxdc-date">${info.date}</div><div class="cxdc-title">${T[l].daily}</div></div>
          <div class="cxdc-formula"><b>${T[l].formula}</b><span id="daily-ai-summary">${info.formula}</span></div>
          <div id="daily-ai-footer" class="daily-ai-footer cxdc-footer">${info.footer}</div>
          <button id="daily-ai-full-toggle" class="daily-full-toggle" type="button" aria-expanded="false">${T[l].full}</button>
          <div id="daily-ai-full-panel" class="daily-full-panel" aria-hidden="true"></div>
        </div>
      </div>`;
  }

  window.COHESIVX_RENDER_DAILY_CYLINDER = function (data) {
    render(data ? derive(data) : fallbackInfo());
  };
})();
