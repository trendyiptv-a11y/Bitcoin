// CohesivX BTC — historical distribution explanation popup, RO/EN aware
(function () {
  "use strict";

  const LANG_KEY = "coeziv_btc_lang";
  const STYLE_ID = "dist-note-style";
  const BUTTON_ID = "dist-note-button";
  const MODAL_ID = "dist-note-modal";

  function lang() {
    try { return localStorage.getItem(LANG_KEY) === "en" ? "en" : "ro"; }
    catch (_) { return "ro"; }
  }
  function by(id) { return document.getElementById(id); }

  function installStyle() {
    if (by(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
      .dist-note-wrap{display:flex;justify-content:center;margin:8px 0 8px}
      .dist-note-button{border:1px solid rgba(56,189,248,.28);background:rgba(15,23,42,.62);color:#bae6fd;border-radius:999px;padding:7px 12px;font-size:11px;font-weight:800;letter-spacing:.03em;cursor:pointer;box-shadow:0 0 18px rgba(56,189,248,.08)}
      .dist-note-button:active{transform:scale(.98)}
      body.light-mode .dist-note-button{background:rgba(248,250,252,.88);color:#075985;border-color:rgba(14,165,233,.30)}
      .dist-note-modal{position:fixed;inset:0;z-index:999998;display:none;align-items:center;justify-content:center;padding:18px;background:rgba(2,6,23,.62);backdrop-filter:blur(7px)}
      .dist-note-modal.open{display:flex}
      .dist-note-dialog{width:min(430px,94vw);max-height:82vh;overflow:auto;border-radius:20px;border:1px solid rgba(56,189,248,.26);background:linear-gradient(180deg,rgba(15,23,42,.97),rgba(2,6,23,.96));box-shadow:0 28px 90px rgba(0,0,0,.62),inset 0 1px 0 rgba(255,255,255,.05);padding:16px;color:#e5e7eb}
      body.light-mode .dist-note-dialog{background:linear-gradient(180deg,rgba(255,255,255,.98),rgba(241,245,249,.97));color:#0f172a}
      .dist-note-title{font-size:14px;font-weight:850;letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px;color:#67e8f9}
      body.light-mode .dist-note-title{color:#0e7490}
      .dist-note-body{font-size:13px;line-height:1.55;color:#cbd5e1;white-space:pre-line}
      body.light-mode .dist-note-body{color:#334155}
      .dist-note-close{margin-top:14px;width:100%;border:1px solid rgba(56,189,248,.28);background:rgba(56,189,248,.10);color:#bae6fd;border-radius:12px;padding:9px 10px;font-size:12px;font-weight:800;cursor:pointer}
      body.light-mode .dist-note-close{color:#075985;background:rgba(14,165,233,.08)}
    `;
    document.head.appendChild(s);
  }

  function parsePercents(text) {
    const nums = String(text || "").match(/~?\d+(?:[.,]\d+)?%/g) || [];
    return nums.slice(0, 3).map(x => Number(x.replace("%", "").replace("~", "").replace(",", "."))).filter(Number.isFinite);
  }

  function parseUsdText(text) {
    const raw = String(text || "").replace(/[^0-9.,-]/g, "").replace(/,/g, "");
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : null;
  }

  function currentPrice() {
    return parseUsdText(by("live-price")?.textContent) || parseUsdText(by("price")?.textContent) || null;
  }

  function usd(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    return Math.round(n).toLocaleString("en-US") + " USD";
  }

  function contextDirection() {
    const flow = String(by("flow-line")?.textContent || "").toLowerCase();
    const delta = String(by("live-delta")?.textContent || "").toLowerCase();
    if (flow.includes("vânzare") || flow.includes("selling") || flow.includes("sell") || delta.includes("sub prețul") || delta.includes("below")) return -1;
    if (flow.includes("cumpărare") || flow.includes("buying") || flow.includes("buy") || delta.includes("peste prețul") || delta.includes("above")) return 1;
    return -1;
  }

  function exampleBlock() {
    const e = lang() === "en";
    const price = currentPrice();
    if (!price) {
      return e ? "Concrete 24h examples are shown when a live price is available." : "Exemplele concrete pe 24h apar când prețul live este disponibil.";
    }
    const direction = contextDirection();
    const continuation = price * (1 + direction * 0.026);
    const opposite = price * (1 - direction * 0.018);
    const neutralLow = price * 0.996;
    const neutralHigh = price * 1.004;
    if (e) {
      return `Concrete example from the current price (${usd(price)}):\n• continuation with the structural context: around ${usd(continuation)} after 24h\n• opposite reaction: around ${usd(opposite)} after 24h\n• neutral/noisy path: roughly ${usd(neutralLow)} – ${usd(neutralHigh)} after 24h`;
    }
    return `Exemplu concret de la prețul actual (${usd(price)}):\n• continuare în direcția contextului structural: în jur de ${usd(continuation)} după 24h\n• reacție în sens opus contextului structural: în jur de ${usd(opposite)} după 24h\n• scenariu neutru / zgomot: aproximativ ${usd(neutralLow)} – ${usd(neutralHigh)} după 24h`;
  }

  function dominant(values) {
    if (values.length < 3) return "unknown";
    const labels = ["with", "against", "neutral"];
    let max = 0;
    for (let i = 1; i < 3; i++) if (values[i] > values[max]) max = i;
    const sorted = [...values].sort((a, b) => b - a);
    return Math.abs(sorted[0] - sorted[1]) < 5 ? "balanced" : labels[max];
  }

  function noteText() {
    const e = lang() === "en";
    const values = parsePercents(by("signal-prob-breakdown")?.textContent || "");
    const d = dominant(values);
    if (e) {
      const base = values.length >= 3
        ? `In similar historical contexts, the next 24h were distributed as follows:\n• ${values[0]}% continued with the structural context\n• ${values[1]}% moved against the structural context\n• ${values[2]}% remained neutral / noisy\n\n`
        : "This note explains the historical 24h distribution for contexts similar to the current one.\n\n";
      const meaning = {
        with: "Reading: continuation was the most frequent historical path. The context had clearer directional follow-through, but it remains an observation, not a prompt.",
        against: "Reading: the opposite reaction was the most frequent historical path. The context is tense or unstable; similar cases reversed more often than they continued.",
        neutral: "Reading: neutral/noisy movement was the most frequent historical path. The context often absorbed the tension without a clear move.",
        balanced: "Reading: the historical paths were close to each other. The context does not show a strong directional advantage.",
        unknown: "Reading: the distribution is a structural context map, not a direct forecast. It describes what happened in similar historical cases."
      }[d];
      return base + exampleBlock() + "\n\n" + meaning;
    }
    const base = values.length >= 3
      ? `În contexte istorice similare, următoarele 24h s-au distribuit astfel:\n• ${values[0]}% au continuat în direcția contextului structural\n• ${values[1]}% s-au mișcat în sens opus contextului structural\n• ${values[2]}% au rămas neutre / zgomotoase\n\n`
      : "Această notă explică distribuția istorică pe 24h pentru contexte similare cu cel actual.\n\n";
    const meaning = {
      with: "Citire: continuarea a fost scenariul istoric cel mai frecvent. Contextul a avut urmărire direcțională mai clară, dar rămâne observație, nu îndemn.",
      against: "Citire: reacția opusă a fost scenariul istoric cel mai frecvent. Contextul este tensionat sau instabil; cazurile similare au inversat mai des decât au continuat.",
      neutral: "Citire: mișcarea neutră/zgomotoasă a fost scenariul istoric cel mai frecvent. Contextul a absorbit des tensiunea fără mișcare clară.",
      balanced: "Citire: scenariile istorice sunt apropiate între ele. Contextul nu arată un avantaj direcțional puternic.",
      unknown: "Citire: distribuția este o hartă de context structural, nu o prognoză directă. Ea descrie ce s-a întâmplat în cazuri istorice similare."
    }[d];
    return base + exampleBlock() + "\n\n" + meaning;
  }

  function labels() {
    const e = lang() === "en";
    return {
      button: e ? "How to read this distribution?" : "Cum citim distribuția?",
      title: e ? "Historical distribution" : "Distribuție istorică",
      close: e ? "Close" : "Închide"
    };
  }

  function ensureUi() {
    const breakdown = by("signal-prob-breakdown");
    if (!breakdown) return;
    if (!by(BUTTON_ID)) {
      const wrap = document.createElement("div");
      wrap.className = "dist-note-wrap";
      wrap.innerHTML = `<button id="${BUTTON_ID}" class="dist-note-button" type="button"></button>`;
      breakdown.insertAdjacentElement("afterend", wrap);
      by(BUTTON_ID).addEventListener("click", openNote);
    }
    if (!by(MODAL_ID)) {
      const modal = document.createElement("div");
      modal.id = MODAL_ID;
      modal.className = "dist-note-modal";
      modal.innerHTML = `<div class="dist-note-dialog" role="dialog" aria-modal="true"><div id="dist-note-title" class="dist-note-title"></div><div id="dist-note-body" class="dist-note-body"></div><button id="dist-note-close" class="dist-note-close" type="button"></button></div>`;
      document.body.appendChild(modal);
      modal.addEventListener("click", ev => { if (ev.target === modal) closeNote(); });
      by("dist-note-close").addEventListener("click", closeNote);
      document.addEventListener("keydown", ev => { if (ev.key === "Escape") closeNote(); });
    }
    updateLabels();
  }

  function updateLabels() {
    const l = labels();
    if (by(BUTTON_ID)) by(BUTTON_ID).textContent = l.button;
    if (by("dist-note-title")) by("dist-note-title").textContent = l.title;
    if (by("dist-note-close")) by("dist-note-close").textContent = l.close;
  }

  function openNote() {
    updateLabels();
    const body = by("dist-note-body");
    if (body) body.textContent = noteText();
    by(MODAL_ID)?.classList.add("open");
  }

  function closeNote() { by(MODAL_ID)?.classList.remove("open"); }

  function boot() { installStyle(); ensureUi(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
  setInterval(boot, 1200);
  try {
    const observer = new MutationObserver(boot);
    observer.observe(document.body, { childList: true, subtree: true });
  } catch (_) {}
})();
