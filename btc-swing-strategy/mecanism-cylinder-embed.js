(function () {
  const STYLE_ID = "cohesivx-daily-cylinder-style";
  const LANG_KEY = "coeziv_btc_lang";
  const STATE_URLS = [
    "coeziv_state.json",
    "./coeziv_state.json",
    "/btc-swing-strategy/coeziv_state.json",
    "/coeziv_state.json"
  ];
  let LAST_DATA = null;

  function lang() {
    try { return localStorage.getItem(LANG_KEY) === "en" ? "en" : "ro"; }
    catch (_) { return "ro"; }
  }

  const T = {
    ro: {
      signal: "SEMNAL",
      asset: "BTC",
      flow: "FLUX",
      participation: "PARTICIPARE",
      liquidity: "LICHIDITATE",
      growth: "CONTEXT CREȘTERE",
      legendTitle: "LEGENDĂ SEMNALE",
      legendSubtitle: "Semnal auditat din mecanism; fallback text doar când câmpul lipsește.",
      footer: "Interpretare structurală experimentală, nu recomandare financiară."
    },
    en: {
      signal: "SIGNAL",
      asset: "BTC",
      flow: "FLOW",
      participation: "PARTICIPATION",
      liquidity: "LIQUIDITY",
      growth: "GROWTH CONTEXT",
      legendTitle: "SIGNAL LEGEND",
      legendSubtitle: "Audited mechanism signal; text fallback only when the field is missing.",
      footer: "Experimental structural interpretation, not financial advice."
    }
  };

  const SIGNAL_META = {
    buy: { icon: "⬆", cls: "buy", ro: ["CUMPĂRARE", "Piața confirmă direcția. Forța internă susține urcarea.", "Atitudine: cumpărare controlată / acumulare"], en: ["BUY", "The market confirms direction. Internal strength supports upside.", "Attitude: controlled buying / accumulation"] },
    accumulate: { icon: "↗", cls: "accumulate", ro: ["ACUMULARE", "Piața arată refacere, dar confirmarea completă lipsește.", "Atitudine: intrări mici / fără agresivitate"], en: ["ACCUMULATE", "The market shows recovery, but full confirmation is still missing.", "Attitude: small entries / no aggression"] },
    wait: { icon: "Ⅱ", cls: "wait", ro: ["AȘTEAPTĂ", "Piața nu arată panică, dar refacerea nu este confirmată.", "Atitudine: prudență / fără intrări agresive"], en: ["WAIT", "The market does not show panic, but recovery is not confirmed.", "Attitude: prudence / no aggressive entries"] },
    attention: { icon: "⚠", cls: "attention", ro: ["ATENȚIE", "Piața stă în picioare, dar presiunea internă este prezentă.", "Atitudine: reducere risc / fără poziții noi mari"], en: ["ATTENTION", "The market is standing, but internal pressure is present.", "Attitude: reduce risk / no large new positions"] },
    sell: { icon: "↓", cls: "sell", ro: ["VÂNZARE", "Presiunea internă domină. Zona curentă nu este susținută.", "Atitudine: reducere expunere"], en: ["SELL", "Internal pressure dominates. The current zone is not supported.", "Attitude: reduce exposure"] },
    risk: { icon: "!", cls: "risk", ro: ["RISC", "Structura este fragilă, iar presiunea internă crește.", "Atitudine: protecție capital / cash"], en: ["RISK", "The structure is fragile and internal pressure is rising.", "Attitude: capital protection / cash"] },
    no_data: { icon: "…", cls: "no-data", ro: ["AȘTEAPTĂ DATE", "Cardul nu are încă toate câmpurile structurale necesare.", "Atitudine: fără interpretare până la următorul snapshot"], en: ["WAITING DATA", "The card does not yet have all required structural fields.", "Attitude: no interpretation until the next snapshot"] }
  };

  function escHtml(v) {
    return String(v ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function fetchJson(url) {
    const res = await fetch(url + "?t=" + Date.now(), { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status + " " + url);
    return await res.json();
  }

  function radarBlock(data) {
    if (!data) return {};
    return lang() === "en" ? (data.radar_info_en || data.radar_info || {}) : (data.radar_info || data.radar_info_en || {});
  }

  function traderCard(data) {
    return data?.trader_signal_card || data?.state?.trader_signal_card || data?.coeziv_state?.trader_signal_card || null;
  }

  async function hydrateWithState(data) {
    const out = data ? { ...data } : {};
    if (traderCard(out)) return out;
    for (const url of STATE_URLS) {
      try {
        const state = await fetchJson(url);
        if (state && state.trader_signal_card) {
          out.trader_signal_card = state.trader_signal_card;
          out.coeziv_state = state;
          return out;
        }
      } catch (_) {}
    }
    return out;
  }

  function fmtDate(data, radar) {
    const raw = radar?.date || data?.date || data?.timestamp || data?.generated_at || data?.last_analysis_at;
    if (!raw) return "–";
    if (/^\d{4}-\d{2}-\d{2}/.test(String(raw))) {
      const d = new Date(raw);
      return Number.isNaN(d.getTime()) ? String(raw).slice(0, 10) : d.toLocaleDateString(lang() === "en" ? "en-GB" : "ro-RO");
    }
    if (/^\d{2}\.\d{2}\.\d{4}$/.test(String(raw))) return raw;
    const d = new Date(raw);
    return Number.isNaN(d.getTime()) ? String(raw) : d.toLocaleDateString(lang() === "en" ? "en-GB" : "ro-RO");
  }

  function deriveFallbackText(data, type) {
    const r = radarBlock(data);
    if (type === "flow") return r.flow || "–";
    if (type === "participation") return r.participation || "–";
    if (type === "liquidity") return r.liquidity || "–";
    if (type === "growth") return r.growth_context || r.growth || "–";
    return "–";
  }

  function legacyTraderSignal(info) {
    const flow = String(info.flow || "").toLowerCase();
    const liq = String(info.liquidity || "").toLowerCase();
    const growth = String(info.growth || "").toLowerCase();
    const hasGrowth = /prezent|present|da|yes/i.test(growth);
    const flowNeg = /negativ|negative/i.test(flow);
    const liqGood = /ridicat|bun|good|high/i.test(liq);
    if (flowNeg && !hasGrowth && liqGood) return { key: "attention", audited: false };
    if (flowNeg && !liqGood) return { key: "sell", audited: false };
    return { key: "wait", audited: false };
  }

  function traderFromAudited(card) {
    if (!card) return null;
    const l = lang();
    const key = SIGNAL_META[card.key] ? card.key : "wait";
    const meta = SIGNAL_META[key];
    const tuple = l === "en" ? meta.en : meta.ro;
    return {
      key,
      icon: meta.icon,
      title: l === "en" ? (card.title_en || tuple[0]) : (card.title_ro || tuple[0]),
      subtitle: l === "en" ? (card.subtitle_en || tuple[1]) : (card.subtitle_ro || tuple[1]),
      action: l === "en" ? (card.action_en || tuple[2]) : (card.action_ro || tuple[2]),
      flow: l === "en" ? (card.flow_label_en || "–") : (card.flow_label_ro || "–"),
      participation: l === "en" ? (card.participation_label_en || "–") : (card.participation_label_ro || "–"),
      liquidity: l === "en" ? (card.liquidity_label_en || "–") : (card.liquidity_label_ro || "–"),
      growth: l === "en" ? (card.growth_label_en || (card.growth_confirmed ? "yes" : "no")) : (card.growth_label_ro || (card.growth_confirmed ? "da" : "nu")),
      audited: true
    };
  }

  function derive(data) {
    const r = radarBlock(data);
    const card = traderCard(data);
    const audited = traderFromAudited(card);
    const info = {
      date: fmtDate(data, r),
      flow: audited ? audited.flow : deriveFallbackText(data, "flow"),
      participation: audited ? audited.participation : deriveFallbackText(data, "participation"),
      liquidity: audited ? audited.liquidity : deriveFallbackText(data, "liquidity"),
      growth: audited ? audited.growth : deriveFallbackText(data, "growth")
    };
    info.trader = audited || Object.assign(legacyTraderSignal(info), { audited: false });
    return info;
  }

  function fallbackInfo() {
    return {
      date: "–",
      flow: "–",
      participation: "–",
      liquidity: "–",
      growth: "–",
      trader: { key: "no_data", audited: false }
    };
  }

  function css() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      #daily-ai-card{border-color:rgba(34,238,255,.38);box-shadow:0 14px 36px rgba(0,0,0,.42),0 0 34px rgba(0,210,255,.10)!important;background:#020617!important;}
      #daily-cylinder-root{padding:0!important;background:#020617!important;}
      .cxtr,.cxlg,.cxtr *,.cxlg *{box-sizing:border-box;-webkit-tap-highlight-color:transparent;}
      .cxtr{margin:0 0 12px;padding:14px;border-radius:21px;border:1px solid rgba(250,204,21,.58);background:radial-gradient(circle at 0 0,rgba(250,204,21,.16),transparent 42%),linear-gradient(180deg,rgba(15,23,42,.92),rgba(2,6,23,.84));box-shadow:0 0 24px rgba(250,204,21,.13);}
      .cxtr-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;font-size:11px;color:#9fb1c7}.cxtr-asset{font-weight:950;letter-spacing:.16em;color:#facc15}.cxtr-main{display:flex;align-items:center;gap:12px}.cxtr-icon{width:54px;height:54px;border-radius:17px;display:flex;align-items:center;justify-content:center;font-size:30px;font-weight:950;background:rgba(250,204,21,.14);color:#facc15;border:1px solid rgba(250,204,21,.50)}
      .cxtr-k{font-size:10px;letter-spacing:.20em;text-transform:uppercase;color:#8fa6bd;font-weight:800}.cxtr-title{font-size:clamp(30px,9vw,46px);line-height:.96;font-weight:950;color:#facc15}.cxtr-sub{margin-top:10px;font-size:14px;line-height:1.45;color:#f8fafc}.cxtr-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:12px}.cxtr-cell{border-radius:13px;padding:8px 9px;border:1px solid rgba(148,163,184,.18);background:rgba(15,23,42,.66)}.cxtr-cell span{display:block;font-size:10px;color:#8fa6bd;text-transform:uppercase;letter-spacing:.08em}.cxtr-cell strong{display:block;margin-top:3px;font-size:13px;color:#f8fafc}.cxtr-action{margin-top:12px;padding:10px 12px;border-radius:15px;font-size:14px;font-weight:900;text-align:center;border:1px solid rgba(250,204,21,.46);color:#facc15;background:rgba(250,204,21,.08)}.cxtr-audit{margin-top:8px;font-size:10px;color:#8fa6bd;text-align:center;}
      .cxtr.signal-buy{border-color:rgba(34,197,94,.70)}.cxtr.signal-buy .cxtr-icon,.cxtr.signal-buy .cxtr-title,.cxtr.signal-buy .cxtr-action{color:#22c55e;border-color:rgba(34,197,94,.58)}.cxtr.signal-accumulate{border-color:rgba(45,212,191,.70)}.cxtr.signal-accumulate .cxtr-icon,.cxtr.signal-accumulate .cxtr-title,.cxtr.signal-accumulate .cxtr-action{color:#2dd4bf;border-color:rgba(45,212,191,.58)}.cxtr.signal-attention{border-color:rgba(249,115,22,.75)}.cxtr.signal-attention .cxtr-icon,.cxtr.signal-attention .cxtr-title,.cxtr.signal-attention .cxtr-action{color:#fb923c;border-color:rgba(249,115,22,.60)}.cxtr.signal-sell,.cxtr.signal-risk{border-color:rgba(239,68,68,.78)}.cxtr.signal-sell .cxtr-icon,.cxtr.signal-sell .cxtr-title,.cxtr.signal-sell .cxtr-action,.cxtr.signal-risk .cxtr-icon,.cxtr.signal-risk .cxtr-title,.cxtr.signal-risk .cxtr-action{color:#ef4444;border-color:rgba(239,68,68,.62)}.cxtr.signal-no-data{border-color:rgba(148,163,184,.45)}
      .cxlg{margin:0;padding:12px;border-radius:18px;border:1px solid rgba(148,163,184,.20);background:linear-gradient(180deg,rgba(15,23,42,.78),rgba(2,6,23,.64));}.cxlg-head{display:flex;justify-content:space-between;align-items:flex-end;gap:10px;margin-bottom:9px}.cxlg-title{font-size:12px;font-weight:950;letter-spacing:.14em;color:#dbeafe}.cxlg-sub{font-size:10px;color:#8fa6bd;text-align:right;line-height:1.25}.cxlg-grid{display:grid;grid-template-columns:1fr;gap:8px}.cxlg-item{display:flex;align-items:center;gap:10px;padding:9px 10px;border-radius:12px;border:1px solid rgba(148,163,184,.16);background:rgba(15,23,42,.56);opacity:.78}.cxlg-item.active{opacity:1;border-width:2px;background:rgba(15,23,42,.82)}.cxlg-dot{width:10px;height:10px;border-radius:999px;flex:0 0 auto;background:#94a3b8}.cxlg-name{display:block;font-size:12px;font-weight:900;line-height:1.15;color:#f8fafc}.cxlg-desc{display:block;margin-top:2px;font-size:10px;color:#8fa6bd;line-height:1.22}.cxlg-item.buy .cxlg-dot{background:#22c55e}.cxlg-item.accumulate .cxlg-dot{background:#2dd4bf}.cxlg-item.wait .cxlg-dot{background:#facc15}.cxlg-item.attention .cxlg-dot{background:#fb923c}.cxlg-item.sell .cxlg-dot,.cxlg-item.risk .cxlg-dot{background:#ef4444}.cxlg-item.no-data .cxlg-dot{background:#94a3b8}
      @media(max-width:380px){.cxtr{padding:12px}.cxtr-icon{width:48px;height:48px;font-size:26px}.cxtr-grid{grid-template-columns:1fr}.cxlg-head{display:block}.cxlg-sub{text-align:left;margin-top:6px}}
    `;
    document.head.appendChild(style);
  }

  function traderLegendHtml(activeKey) {
    const l = lang();
    const tt = T[l];
    const order = ["buy", "accumulate", "wait", "attention", "sell", "risk", "no_data"];
    const rows = order.map((key) => {
      const meta = SIGNAL_META[key];
      const tuple = l === "en" ? meta.en : meta.ro;
      return `<div class="cxlg-item ${meta.cls}${key === activeKey ? " active" : ""}"><span class="cxlg-dot"></span><span><span class="cxlg-name">${escHtml(tuple[0])}</span><span class="cxlg-desc">${escHtml(tuple[1])}</span></span></div>`;
    }).join("");
    return `<div class="cxlg"><div class="cxlg-head"><div class="cxlg-title">${escHtml(tt.legendTitle)}</div><div class="cxlg-sub">${escHtml(tt.legendSubtitle)}</div></div><div class="cxlg-grid">${rows}</div></div>`;
  }

  function render(info) {
    css();
    const root = document.getElementById("daily-cylinder-root") || document.getElementById("daily-ai-card")?.querySelector(".card-inner");
    if (!root) return;
    const l = lang();
    const tt = T[l];
    const tr = info.trader || { key: "no_data", audited: false };
    const meta = SIGNAL_META[tr.key] || SIGNAL_META.wait;
    const tuple = l === "en" ? meta.en : meta.ro;
    const title = tr.title || tuple[0];
    const subtitle = tr.subtitle || tuple[1];
    const action = tr.action || tuple[2];
    const auditText = tr.audited ? "audit: trader_signal_card · backend" : "fallback: text-derived UI";
    root.innerHTML = `
      <div class="cxtr signal-${escHtml(meta.cls)}">
        <div class="cxtr-top"><div class="cxtr-asset">${tt.asset}</div><div>${escHtml(info.date)}</div></div>
        <div class="cxtr-main"><div class="cxtr-icon">${escHtml(tr.icon || meta.icon)}</div><div><div class="cxtr-k">${tt.signal}</div><div class="cxtr-title">${escHtml(title)}</div></div></div>
        <div class="cxtr-sub">${escHtml(subtitle)}</div>
        <div class="cxtr-grid">
          <div class="cxtr-cell"><span>${tt.flow}</span><strong>${escHtml(tr.flow || info.flow || "–")}</strong></div>
          <div class="cxtr-cell"><span>${tt.participation}</span><strong>${escHtml(tr.participation || info.participation || "–")}</strong></div>
          <div class="cxtr-cell"><span>${tt.liquidity}</span><strong>${escHtml(tr.liquidity || info.liquidity || "–")}</strong></div>
          <div class="cxtr-cell"><span>${tt.growth}</span><strong>${escHtml(tr.growth || info.growth || "–")}</strong></div>
        </div>
        <div class="cxtr-action">${escHtml(action)}</div>
        <div class="cxtr-audit">${escHtml(auditText)}</div>
      </div>
      ${traderLegendHtml(tr.key)}
    `;
  }

  function rerender() {
    render(LAST_DATA ? derive(LAST_DATA) : fallbackInfo());
  }

  window.COHESIVX_RENDER_DAILY_CYLINDER = async function (data) {
    LAST_DATA = await hydrateWithState(data || {});
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
