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

  const TRADER_T = {
    ro: {
      signal: "SEMNAL",
      asset: "BTC",
      flow: "Flux",
      participation: "Participare",
      liquidity: "Lichiditate",
      growth: "Confirmare creștere",
      yes: "da",
      no: "nu",
      legendTitle: "LEGENDĂ SEMNALE",
      legendSubtitle: "Ce poate afișa cardul în funcție de context.",
      legend: {
        buy: ["CUMPĂRARE", "Forță confirmată"],
        accumulate: ["ACUMULARE", "Refacere în formare"],
        wait: ["AȘTEAPTĂ", "Direcție neconfirmată"],
        attention: ["ATENȚIE", "Presiune internă"],
        sell: ["VÂNZARE", "Reducere expunere"],
        risk: ["RISC", "Protecție capital"]
      },
      buy: {
        icon: "⬆",
        title: "CUMPĂRARE",
        subtitle: "Piața confirmă direcția. Forța internă susține urcarea.",
        action: "Atitudine: cumpărare controlată / acumulare"
      },
      accumulate: {
        icon: "↗",
        title: "ACUMULARE",
        subtitle: "Piața arată refacere, dar confirmarea completă lipsește.",
        action: "Atitudine: intrări mici / fără agresivitate"
      },
      wait: {
        icon: "Ⅱ",
        title: "AȘTEAPTĂ",
        subtitle: "Piața nu arată panică, dar refacerea nu este confirmată.",
        action: "Atitudine: prudență / fără intrări agresive"
      },
      attention: {
        icon: "⚠",
        title: "ATENȚIE",
        subtitle: "Piața stă în picioare, dar presiunea internă este prezentă.",
        action: "Atitudine: reducere risc / fără poziții noi mari"
      },
      sell: {
        icon: "↓",
        title: "VÂNZARE",
        subtitle: "Presiunea internă domină. Zona curentă nu este susținută.",
        action: "Atitudine: reducere expunere"
      },
      risk: {
        icon: "!",
        title: "RISC",
        subtitle: "Structura este fragilă, iar presiunea internă crește.",
        action: "Atitudine: protecție capital / cash"
      }
    },
    en: {
      signal: "SIGNAL",
      asset: "BTC",
      flow: "Flow",
      participation: "Participation",
      liquidity: "Liquidity",
      growth: "Growth confirmation",
      yes: "yes",
      no: "no",
      legendTitle: "SIGNAL LEGEND",
      legendSubtitle: "What the card can show depending on context.",
      legend: {
        buy: ["BUY", "Confirmed strength"],
        accumulate: ["ACCUMULATE", "Recovery forming"],
        wait: ["WAIT", "Direction unconfirmed"],
        attention: ["ATTENTION", "Internal pressure"],
        sell: ["SELL", "Reduce exposure"],
        risk: ["RISK", "Protect capital"]
      },
      buy: {
        icon: "⬆",
        title: "BUY",
        subtitle: "The market confirms direction. Internal strength supports upside.",
        action: "Attitude: controlled buying / accumulation"
      },
      accumulate: {
        icon: "↗",
        title: "ACCUMULATE",
        subtitle: "The market shows recovery, but full confirmation is still missing.",
        action: "Attitude: small entries / no aggression"
      },
      wait: {
        icon: "Ⅱ",
        title: "WAIT",
        subtitle: "The market does not show panic, but recovery is not confirmed.",
        action: "Attitude: prudence / no aggressive entries"
      },
      attention: {
        icon: "⚠",
        title: "ATTENTION",
        subtitle: "The market is standing, but internal pressure is present.",
        action: "Attitude: reduce risk / no large new positions"
      },
      sell: {
        icon: "↓",
        title: "SELL",
        subtitle: "Internal pressure dominates. The current zone is not supported.",
        action: "Attitude: reduce exposure"
      },
      risk: {
        icon: "!",
        title: "RISK",
        subtitle: "The structure is fragile and internal pressure is rising.",
        action: "Attitude: capital protection / cash"
      }
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
      .cxtr{margin:0 0 12px;padding:13px;border-radius:20px;border:1px solid rgba(250,204,21,.58);background:radial-gradient(circle at 0 0,rgba(250,204,21,.16),transparent 42%),linear-gradient(180deg,rgba(15,23,42,.90),rgba(2,6,23,.82));box-shadow:0 0 24px rgba(250,204,21,.12);contain:layout paint style}
      .cxtr-top{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:9px;font-size:11px;color:#9fb1c7}.cxtr-asset{font-weight:950;letter-spacing:.16em;color:#facc15}.cxtr-main{display:flex;align-items:center;gap:12px}.cxtr-icon{width:54px;height:54px;border-radius:17px;display:flex;align-items:center;justify-content:center;font-size:30px;font-weight:950;background:rgba(250,204,21,.14);color:#facc15;border:1px solid rgba(250,204,21,.50);box-shadow:inset 0 1px 0 rgba(255,255,255,.06),0 0 18px rgba(250,204,21,.12)}
      .cxtr-k{font-size:10px;letter-spacing:.20em;text-transform:uppercase;color:#8fa6bd;font-weight:800}.cxtr-title{font-size:clamp(30px,9vw,46px);line-height:.96;font-weight:950;letter-spacing:.02em;color:#facc15;text-shadow:0 0 22px rgba(250,204,21,.20)}.cxtr-sub{margin-top:10px;font-size:14px;line-height:1.42;color:#f8fafc}.cxtr-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:12px}.cxtr-cell{border-radius:13px;padding:8px 9px;border:1px solid rgba(148,163,184,.18);background:rgba(15,23,42,.66)}.cxtr-cell span{display:block;font-size:10px;color:#8fa6bd;text-transform:uppercase;letter-spacing:.08em}.cxtr-cell strong{display:block;margin-top:3px;font-size:13px;color:#f8fafc}.cxtr-action{margin-top:12px;padding:10px 12px;border-radius:15px;font-size:14px;font-weight:900;text-align:center;border:1px solid rgba(250,204,21,.46);color:#facc15;background:rgba(250,204,21,.08)}
      .cxlg{margin:0 0 12px;padding:12px;border-radius:18px;border:1px solid rgba(148,163,184,.20);background:linear-gradient(180deg,rgba(15,23,42,.78),rgba(2,6,23,.64));contain:layout paint style}
      .cxlg-head{display:flex;justify-content:space-between;align-items:flex-end;gap:10px;margin-bottom:9px}.cxlg-title{font-size:12px;font-weight:950;letter-spacing:.14em;color:#dbeafe}.cxlg-sub{font-size:10px;color:#8fa6bd;text-align:right;line-height:1.25}.cxlg-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.cxlg-item{display:flex;align-items:center;gap:8px;padding:8px;border-radius:12px;border:1px solid rgba(148,163,184,.16);background:rgba(15,23,42,.56);opacity:.78}.cxlg-item.active{opacity:1;border-width:2px;background:rgba(15,23,42,.82)}.cxlg-dot{width:10px;height:10px;border-radius:999px;flex:0 0 auto;background:#94a3b8;box-shadow:0 0 10px rgba(148,163,184,.45)}.cxlg-text{min-width:0}.cxlg-name{font-size:11px;font-weight:900;line-height:1.1;color:#f8fafc}.cxlg-desc{margin-top:2px;font-size:10px;color:#8fa6bd;line-height:1.12}.cxlg-item.buy .cxlg-dot{background:#22c55e;box-shadow:0 0 10px rgba(34,197,94,.55)}.cxlg-item.accumulate .cxlg-dot{background:#2dd4bf;box-shadow:0 0 10px rgba(45,212,191,.55)}.cxlg-item.wait .cxlg-dot{background:#facc15;box-shadow:0 0 10px rgba(250,204,21,.55)}.cxlg-item.attention .cxlg-dot{background:#fb923c;box-shadow:0 0 10px rgba(251,146,60,.55)}.cxlg-item.sell .cxlg-dot,.cxlg-item.risk .cxlg-dot{background:#ef4444;box-shadow:0 0 10px rgba(239,68,68,.58)}.cxlg-item.buy.active{border-color:rgba(34,197,94,.70)}.cxlg-item.accumulate.active{border-color:rgba(45,212,191,.70)}.cxlg-item.wait.active{border-color:rgba(250,204,21,.70)}.cxlg-item.attention.active{border-color:rgba(251,146,60,.76)}.cxlg-item.sell.active,.cxlg-item.risk.active{border-color:rgba(239,68,68,.78)}
      .cxtr.signal-buy{border-color:rgba(34,197,94,.70);box-shadow:0 0 28px rgba(34,197,94,.16)}.cxtr.signal-buy .cxtr-icon,.cxtr.signal-buy .cxtr-title,.cxtr.signal-buy .cxtr-action{color:#22c55e;border-color:rgba(34,197,94,.58)}.cxtr.signal-buy .cxtr-icon,.cxtr.signal-buy .cxtr-action{background:rgba(34,197,94,.10)}
      .cxtr.signal-accumulate{border-color:rgba(45,212,191,.70);box-shadow:0 0 28px rgba(45,212,191,.14)}.cxtr.signal-accumulate .cxtr-icon,.cxtr.signal-accumulate .cxtr-title,.cxtr.signal-accumulate .cxtr-action{color:#2dd4bf;border-color:rgba(45,212,191,.58)}.cxtr.signal-accumulate .cxtr-icon,.cxtr.signal-accumulate .cxtr-action{background:rgba(45,212,191,.10)}
      .cxtr.signal-attention{border-color:rgba(249,115,22,.75);box-shadow:0 0 28px rgba(249,115,22,.16)}.cxtr.signal-attention .cxtr-icon,.cxtr.signal-attention .cxtr-title,.cxtr.signal-attention .cxtr-action{color:#fb923c;border-color:rgba(249,115,22,.60)}.cxtr.signal-attention .cxtr-icon,.cxtr.signal-attention .cxtr-action{background:rgba(249,115,22,.10)}
      .cxtr.signal-sell,.cxtr.signal-risk{border-color:rgba(239,68,68,.78);box-shadow:0 0 30px rgba(239,68,68,.18)}.cxtr.signal-sell .cxtr-icon,.cxtr.signal-sell .cxtr-title,.cxtr.signal-sell .cxtr-action,.cxtr.signal-risk .cxtr-icon,.cxtr.signal-risk .cxtr-title,.cxtr.signal-risk .cxtr-action{color:#ef4444;border-color:rgba(239,68,68,.62)}.cxtr.signal-sell .cxtr-icon,.cxtr.signal-sell .cxtr-action,.cxtr.signal-risk .cxtr-icon,.cxtr.signal-risk .cxtr-action{background:rgba(239,68,68,.10)}
      @media(max-width:380px){.cxtr{padding:12px}.cxtr-icon{width:48px;height:48px;font-size:26px}.cxtr-grid{grid-template-columns:1fr}.cxtr-sub,.cxtr-action{font-size:13px}.cxlg-grid{grid-template-columns:1fr}.cxlg-head{align-items:flex-start;flex-direction:column}.cxlg-sub{text-align:left}}
      @keyframes cxdcBeam{0%,100%{opacity:.84}50%{opacity:.58}}@keyframes cxdcParticles{to{background-position:0 -240px}}@keyframes cxdcFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-3px)}}@keyframes cxdcRisk{0%,100%{opacity:.78;transform:scale(1)}50%{opacity:.94;transform:scale(1.012)}}
      @media(max-width:760px),(pointer:coarse){#daily-ai-card,.cxdc,.cxdc-core,.cxdc-cylinder{background-color:#020617!important;box-shadow:none!important;transform:none!important;filter:none!important;will-change:auto!important}.cxdc-core:before{background:linear-gradient(180deg,rgba(3,18,34,.92),rgba(0,10,22,.96))!important}.cxdc-particles,.cxdc-beam,.cxdc-layer,.cxdc-risk{animation:none!important;transition:none!important;will-change:auto!important}.cxdc-particles{opacity:.08!important}.cxdc-beam{opacity:.34!important;background:linear-gradient(180deg,transparent,rgba(0,247,255,.30) 22%,rgba(125,245,255,.36) 50%,rgba(0,247,255,.30) 78%,transparent)!important;box-shadow:0 0 10px rgba(0,240,255,.28)!important}.cxdc-layer{box-shadow:0 0 8px rgba(120,240,255,.18),inset 0 0 8px rgba(255,255,255,.05)!important}.cxdc-risk{box-shadow:0 0 10px rgba(255,88,40,.28)!important}.cxdc-cylinder:before,.cxdc-cylinder:after{box-shadow:none!important;border-color:rgba(125,245,255,.42)!important}}
      @media(max-width:380px){.cxdc-core{min-height:555px}.cxdc-labels{left:10px;width:40%;gap:20px}.cxdc-label .wire,.cxdc-label .pin{display:none}.cxdc-cylinder{right:5px;width:56%;height:505px}}
    `;
    document.head.appendChild(style);
  }

  function normSignalText(v) {
    return String(v || "")
      .toLowerCase()
      .normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  }

  function escHtml(v) {
    return String(v ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function deriveTraderSignal(info) {
    const t = TRADER_T[lang()];
    const price = normSignalText(info.price);
    const participation = normSignalText(info.participation);
    const flow = normSignalText(info.flow);
    const liquidity = normSignalText(info.liquidity);
    const growth = normSignalText(info.growth);
    const risk = normSignalText(info.risk);
    const formula = normSignalText(info.formula);

    const hasGrowth = info.visual?.growthPresent || growth.includes("prezent") || growth.includes("present") || growth === "da" || growth === "yes";
    const isCohesive = info.visual?.participationCohesive || participation.includes("coeziv") || participation.includes("cohesive");
    const isTense = participation.includes("tension") || participation.includes("tense");
    const flowPositive = info.visual?.flowPositive || flow.includes("pozitiv") || flow.includes("positive");
    const flowNegative = info.visual?.flowNegative || flow.includes("negativ") || flow.includes("negative");
    const liquidityGood = liquidity.includes("ridicat") || liquidity.includes("buna") || liquidity.includes("buna") || liquidity.includes("bun") || liquidity.includes("good") || liquidity.includes("high");
    const deep = price.includes("degradare profunda") || price.includes("deep degradation") || risk.includes("degradare profunda") || risk.includes("deep degradation") || formula.includes("degradare profunda") || formula.includes("deep degradation");
    const unrepaired = formula.includes("nereparat") || formula.includes("unrepaired") || formula.includes("not repaired") || risk.includes("ruptur");

    let key = "wait";
    if (hasGrowth && isCohesive && flowPositive && liquidityGood && !deep && !unrepaired) key = "buy";
    else if (isCohesive && flowPositive && liquidityGood && !hasGrowth) key = "accumulate";
    else if (flowNegative && isTense && !liquidityGood) key = "sell";
    else if (deep && flowNegative) key = "risk";
    else if (flowNegative && isTense && !hasGrowth) key = "attention";

    const copy = t[key] || t.wait;
    return {
      key,
      icon: copy.icon,
      title: copy.title,
      subtitle: copy.subtitle,
      action: copy.action,
      growth: hasGrowth ? t.yes : t.no,
      flow: info.flow || "–",
      participation: info.participation || "–",
      liquidity: info.liquidity || "–"
    };
  }

  function traderLegendHtml(activeKey) {
    const tt = TRADER_T[lang()];
    const order = ["buy", "accumulate", "wait", "attention", "sell", "risk"];
    const rows = order.map((key) => {
      const item = tt.legend && tt.legend[key] ? tt.legend[key] : [key, ""];
      return `<div class="cxlg-item ${key}${key === activeKey ? " active" : ""}"><span class="cxlg-dot"></span><span class="cxlg-text"><span class="cxlg-name">${escHtml(item[0])}</span><span class="cxlg-desc">${escHtml(item[1])}</span></span></div>`;
    }).join("");
    return `<div class="cxlg"><div class="cxlg-head"><div class="cxlg-title">${escHtml(tt.legendTitle)}</div><div class="cxlg-sub">${escHtml(tt.legendSubtitle)}</div></div><div class="cxlg-grid">${rows}</div></div>`;
  }

  function render(info) {
    css();
    const root = document.getElementById("daily-cylinder-root") || document.getElementById("daily-ai-card")?.querySelector(".card-inner");
    if (!root) return;
    const l = lang();
    const tr = deriveTraderSignal(info);
    const tt = TRADER_T[l];
    root.innerHTML = `
      <div class="cxtr signal-${tr.key}">
        <div class="cxtr-top"><div class="cxtr-asset">${tt.asset}</div><div class="cxtr-date">${escHtml(info.date)}</div></div>
        <div class="cxtr-main">
          <div class="cxtr-icon">${escHtml(tr.icon)}</div>
          <div><div class="cxtr-k">${tt.signal}</div><div class="cxtr-title">${escHtml(tr.title)}</div></div>
        </div>
        <div class="cxtr-sub">${escHtml(tr.subtitle)}</div>
        <div class="cxtr-grid">
          <div class="cxtr-cell"><span>${tt.flow}</span><strong>${escHtml(tr.flow)}</strong></div>
          <div class="cxtr-cell"><span>${tt.participation}</span><strong>${escHtml(tr.participation)}</strong></div>
          <div class="cxtr-cell"><span>${tt.liquidity}</span><strong>${escHtml(tr.liquidity)}</strong></div>
          <div class="cxtr-cell"><span>${tt.growth}</span><strong>${escHtml(tr.growth)}</strong></div>
        </div>
        <div class="cxtr-action">${escHtml(tr.action)}</div>
      </div>
      ${traderLegendHtml(tr.key)}
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
