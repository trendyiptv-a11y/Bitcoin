/* CohesivX BTC — animated radar insert for mecanism.html, RO/EN aware
   Main display is anchored to the historical bottom-zone investigation.
   Trading logic is not modified. */
(function () {
  "use strict";

  const ID = "coeziv-mini-radar";
  const STRUCT_ID = "coeziv-structural-map";
  const LANG_KEY = "coeziv_btc_lang";
  let refreshTimer = null;
  let liveSyncTimer = null;
  let lastState = null;
  let lastRisk = null;
  let lastSummary = null;
  let lastLang = null;
  let lastToneClass = null;

  function by(id) { return document.getElementById(id); }
  function lang() { return localStorage.getItem(LANG_KEY) === "en" ? "en" : "ro"; }
  function tr(ro, en) { return lang() === "en" ? en : ro; }

  function usdK(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "—";
    return (n / 1000).toFixed(1).replace(".0", "") + "K USD";
  }

  function parseUsdText(text) {
    const raw = String(text || "").replace(/[^0-9.,-]/g, "").replace(/,/g, "");
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : null;
  }

  function currentLivePrice() {
    const liveEl = by("live-price");
    const live = liveEl ? parseUsdText(liveEl.textContent) : null;
    return live || Number(lastState && lastState.price_usd) || null;
  }

  async function json(path) {
    const r = await fetch(path + "?t=" + Date.now(), { cache: "no-store" });
    if (!r.ok) throw new Error(path);
    return r.json();
  }

  async function jsonOrNull(path) {
    try { return await json(path); }
    catch (_) { return null; }
  }

  function deriveStart(state) {
    const hist = Array.isArray(state && state.signal_history) ? state.signal_history : [];
    const firstShort = hist.find(x => x.signal === "short");
    return firstShort ? Number(firstShort.model_price_usd) : Number((state && (state.model_price_usd || state.price_usd)) || 0);
  }

  function liveWindowThreshold(state, risk) {
    const start = deriveStart(state);
    const dd = Number((risk && risk.major_drawdown_threshold) ?? -0.2);
    if (!Number.isFinite(start) || start <= 0 || !Number.isFinite(dd)) return null;
    return start * (1 + dd);
  }

  function candidate(summary) {
    return summary && summary.radar_candidate ? summary.radar_candidate : null;
  }

  function flowDirection(state, risk) {
    const signal = String((state && state.signal) || (risk && risk.current_signal) || "").toLowerCase();
    const flow = String((state && state.flow_bias) || "").toLowerCase();
    const strength = String((state && state.flow_strength) || "").toLowerCase();
    if (signal === "long" || flow.includes("pozitiv") || flow.includes("buy") || flow.includes("cump")) {
      return { code: "up", ro: strength ? `revenire urmărită (${strength})` : "revenire urmărită", en: strength ? `recovery watched (${strength})` : "recovery watched" };
    }
    if (signal === "short" || flow.includes("negativ") || flow.includes("sell") || flow.includes("vânz")) {
      return { code: "down", ro: strength ? `scădere activă (${strength})` : "scădere activă", en: strength ? `active decline (${strength})` : "active decline" };
    }
    return { code: "flat", ro: "structură laterală", en: "sideways structure" };
  }

  function structuralZone(price, summary, state, risk) {
    const c = candidate(summary);
    if (!c || !Number.isFinite(Number(price))) return null;

    const ath = Number(c.ath_price);
    const bear = Number(c.bear_warning_threshold);
    const bottomLow = Number(c.bottom_risk_zone_low);
    const bottomMid = Number(c.bottom_risk_zone_mid);
    const bottomHigh = Number(c.bottom_risk_zone_high);
    const hard = Number(c.hard_capitulation_below);
    const liveInternal = liveWindowThreshold(state, risk);
    const dir = flowDirection(state, risk);
    const p = Number(price);

    if (p >= ath) {
      return { key: "expansion", cls: "tone-green", icon: "🟢", color: "#6dffb0", title: tr("Expansiune peste ATH", "Expansion above ATH"), direction: dir, ath, bear, bottomLow, bottomMid, bottomHigh, hard, liveInternal };
    }
    if (p >= bear) {
      return { key: "repaired", cls: "tone-green", icon: "🟢", color: "#6dffb0", title: tr("Structură refăcută", "Structure repaired"), direction: dir, ath, bear, bottomLow, bottomMid, bottomHigh, hard, liveInternal };
    }
    if (Number.isFinite(liveInternal) && p >= liveInternal) {
      return { key: "fragility", cls: "tone-orange", icon: "🟠", color: "#ffb454", title: tr("Ruptură de fragilitate", "Fragility rupture"), direction: dir, ath, bear, bottomLow, bottomMid, bottomHigh, hard, liveInternal };
    }
    if (p > bottomHigh) {
      return { key: "deep", cls: "tone-red", icon: "🔴", color: "#ff5d6c", title: tr("Degradare profundă", "Deep degradation"), direction: dir, ath, bear, bottomLow, bottomMid, bottomHigh, hard, liveInternal };
    }
    if (p >= bottomLow) {
      return { key: "bottom", cls: "tone-orange", icon: "🟠", color: "#ffb454", title: tr("Bottom final activ", "Final bottom zone active"), direction: dir, ath, bear, bottomLow, bottomMid, bottomHigh, hard, liveInternal };
    }
    return { key: "capitulation", cls: "tone-red", icon: "🔴", color: "#ff5d6c", title: tr("Capitulare sub bottom absolut", "Capitulation below absolute bottom"), direction: dir, ath, bear, bottomLow, bottomMid, bottomHigh, hard, liveInternal };
  }

  function fallbackToneFor(risk) {
    const active = !!(risk && risk.active);
    const days = Number((risk && risk.consecutive_degradation_days) || 0);
    const median = Number((risk && risk.median_days_to_confirmation) || 27);
    const level = String((risk && risk.level) || "");
    if (!active) return { cls: "tone-green", icon: "🟢", title: tr("Structură refăcută", "Structure repaired"), color: "#6dffb0" };
    if (days >= median || level === "high") {
      return { cls: "tone-red", icon: "🔴", title: days >= median ? tr("Degradare persistentă", "Persistent degradation") : tr("Risc structural", "Structural risk"), color: "#ff5d6c" };
    }
    return { cls: "tone-orange", icon: "🟠", title: tr("Degradare activă", "Active degradation"), color: "#ffb454" };
  }

  function installStyle() {
    if (by("coeziv-mini-radar-style")) return;
    const s = document.createElement("style");
    s.id = "coeziv-mini-radar-style";
    s.textContent = `
      #${ID}{margin:14px auto 12px;padding:10px 10px 12px;border-radius:18px;border:1px solid rgba(109,255,176,.20);background:radial-gradient(circle at 50% -18%,rgba(109,255,176,.10),transparent 55%),linear-gradient(180deg,rgba(12,17,24,.82),rgba(12,17,24,.50));box-shadow:inset 0 1px 0 rgba(255,255,255,.04),0 14px 30px rgba(0,0,0,.24);text-align:center;overflow:hidden;position:relative;transition:border-color .25s ease,box-shadow .25s ease,transform .25s ease}
      #${ID} .radar-stage{position:relative;width:172px;height:172px;margin:0 auto 8px;transition:transform .25s ease,filter .25s ease}
      #${ID} svg{width:100%;height:100%;display:block}
      #${ID} .radar-tick{stroke:rgba(148,163,184,.20);stroke-width:1}
      #${ID} .radar-sweep{transform-origin:86px 86px;animation:coezivRadarSpin 4.8s linear infinite}
      #${ID} .radar-pulse{opacity:0;transform-origin:86px 86px;animation:coezivRadarPulse 2.8s ease-out infinite}
      #${ID} .radar-pulse.p2{animation-delay:1.4s}
      #${ID} .radar-dot{transform-origin:86px 86px;animation:coezivRadarBreath 2.8s ease-in-out infinite}
      @keyframes coezivRadarSpin{to{transform:rotate(360deg)}}
      @keyframes coezivRadarSpinFast{to{transform:rotate(360deg)}}
      @keyframes coezivRadarPulse{0%{opacity:.55;transform:scale(.45)}70%,100%{opacity:0;transform:scale(1)}}
      @keyframes coezivRadarBreath{0%,100%{r:5}50%{r:6.6}}
      @keyframes coezivRadarAlert{0%{transform:scale(1);box-shadow:inset 0 1px 0 rgba(255,255,255,.04),0 14px 30px rgba(0,0,0,.24)}18%{transform:scale(1.018);box-shadow:0 0 0 2px rgba(255,255,255,.10),0 0 34px var(--radar-alert,rgba(255,180,84,.45)),0 18px 42px rgba(0,0,0,.30)}42%{transform:scale(1)}62%{transform:scale(1.012);box-shadow:0 0 0 1px rgba(255,255,255,.08),0 0 24px var(--radar-alert,rgba(255,180,84,.35)),0 18px 42px rgba(0,0,0,.30)}100%{transform:scale(1)}}
      @keyframes coezivIconPop{0%{transform:scale(1)}20%{transform:scale(1.22)}45%{transform:scale(.96)}70%{transform:scale(1.08)}100%{transform:scale(1)}}
      #${ID}.state-changed{animation:coezivRadarAlert 2.6s ease-out 1}
      #${ID}.state-changed .radar-stage{filter:drop-shadow(0 0 20px var(--radar-alert,rgba(255,180,84,.45)))}
      #${ID}.state-changed .radar-sweep{animation:coezivRadarSpinFast .72s linear infinite}
      #${ID}.state-changed .radar-pulse{animation-duration:1.15s;opacity:.8}
      #${ID}.state-changed .radar-icon{animation:coezivIconPop 1.1s ease-out 1}
      #${ID} .radar-read{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none}
      #${ID} .radar-icon{font-size:27px;line-height:1;filter:drop-shadow(0 0 14px currentColor);transform-origin:center;display:inline-block}
      #${ID} .radar-title{font-size:13px;font-weight:850;color:#e5e7eb;line-height:1.12;margin-top:5px}
      #${ID} .radar-price{font-size:10.5px;color:#94a3b8;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;margin-top:5px}
      #${ID} .radar-badges{display:flex;justify-content:center;flex-wrap:wrap;gap:7px;margin-top:2px}
      #${ID} .radar-badge{border-radius:11px;padding:6px 9px;font-size:10.5px;font-weight:850;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;border:1px solid rgba(148,163,184,.22);background:rgba(2,6,23,.36);color:#cbd5e1;white-space:nowrap}
      #${ID} .radar-badge.day{border-color:rgba(255,180,84,.36);color:#ffd29c}
      #${ID} .radar-badge.threshold{border-color:rgba(255,93,108,.38);color:#ffb2ba}
      #${ID}.tone-green{border-color:rgba(109,255,176,.24);--radar-alert:rgba(109,255,176,.46)}
      #${ID}.tone-orange{border-color:rgba(255,180,84,.24);--radar-alert:rgba(255,180,84,.46)}
      #${ID}.tone-red{border-color:rgba(255,93,108,.26);--radar-alert:rgba(255,93,108,.50)}
      #${STRUCT_ID}{margin:10px auto 2px;padding:9px 9px 10px;border-radius:14px;border:1px solid rgba(148,163,184,.18);background:linear-gradient(180deg,rgba(2,6,23,.34),rgba(2,6,23,.18));text-align:left}
      #${STRUCT_ID} .sd-head{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:7px;font-size:10.5px;font-weight:900;color:#e5e7eb}
      #${STRUCT_ID} .sd-mode{color:#94a3b8;font-size:9.5px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:nowrap}
      #${STRUCT_ID} .sd-row{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;padding:6px 7px;border-radius:10px;border:1px solid rgba(148,163,184,.13);background:rgba(15,23,42,.20);margin-top:5px}
      #${STRUCT_ID} .sd-row.active{border-color:rgba(255,180,84,.55);box-shadow:0 0 0 1px rgba(255,180,84,.12),0 0 20px rgba(255,180,84,.12);background:rgba(255,180,84,.08)}
      #${STRUCT_ID} .sd-name{color:#cbd5e1;font-size:10.2px;font-weight:800;line-height:1.15}
      #${STRUCT_ID} .sd-value{color:#e5e7eb;font-size:10.2px;font-weight:900;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:nowrap}
      #${STRUCT_ID} .sd-note{margin-top:7px;color:#94a3b8;font-size:9.5px;line-height:1.25;text-align:center}
      body.light-mode #${ID}{background:radial-gradient(circle at 50% -18%,rgba(14,165,233,.10),transparent 55%),linear-gradient(180deg,rgba(255,255,255,.94),rgba(248,250,252,.84));border-color:rgba(14,165,233,.25);box-shadow:0 10px 24px rgba(15,23,42,.10)}
      body.light-mode #${ID} .radar-title{color:#0f172a}
      body.light-mode #${STRUCT_ID}{background:linear-gradient(180deg,rgba(255,255,255,.88),rgba(248,250,252,.76));border-color:rgba(15,23,42,.12)}
      body.light-mode #${STRUCT_ID} .sd-head{color:#0f172a}
      body.light-mode #${STRUCT_ID} .sd-name{color:#334155}
      body.light-mode #${STRUCT_ID} .sd-value{color:#0f172a}
      @media(max-width:380px){#${ID} .radar-stage{width:156px;height:156px}}
      @media(max-width:768px){#${ID},#${STRUCT_ID},#${ID} .radar-stage,#${ID} svg{backface-visibility:hidden;-webkit-backface-visibility:hidden;transform:translate3d(0,0,0);-webkit-transform:translate3d(0,0,0)}#${ID} .radar-pulse,#${ID} .radar-sweep,#${ID} .radar-dot{filter:none!important}}
    `;
    document.head.appendChild(s);
  }

  function tickMarks() {
    let out = "";
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      const r1 = 75;
      const r2 = i % 3 === 0 ? 66 : 70;
      const x1 = 86 + r1 * Math.cos(a);
      const y1 = 86 + r1 * Math.sin(a);
      const x2 = 86 + r2 * Math.cos(a);
      const y2 = 86 + r2 * Math.sin(a);
      out += `<line class="radar-tick" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"/>`;
    }
    return out;
  }

  function ensureCard() {
    if (by(ID)) return by(ID);
    const anchor = by("prod-cost-line");
    if (!anchor || !anchor.parentNode) return null;
    const card = document.createElement("div");
    card.id = ID;
    card.setAttribute("aria-live", "polite");
    card.innerHTML = `
      <div class="radar-stage">
        <svg viewBox="0 0 172 172" aria-hidden="true">
          <circle cx="86" cy="86" r="80" fill="none" stroke="rgba(148,163,184,.18)" stroke-width="1"/>
          <circle cx="86" cy="86" r="55" fill="none" stroke="rgba(148,163,184,.16)" stroke-width="1"/>
          <g>${tickMarks()}</g>
          <circle class="radar-pulse p1" cx="86" cy="86" r="40" fill="none" stroke="#ffb454" stroke-width="1.4"/>
          <circle class="radar-pulse p2" cx="86" cy="86" r="40" fill="none" stroke="#ffb454" stroke-width="1.4"/>
          <g class="radar-sweep"><path d="M86 86 L86 7 A79 79 0 0 1 138 28 Z" fill="rgba(109,255,176,.32)"/></g>
          <circle class="radar-dot" cx="86" cy="86" r="5" fill="#ffb454"/>
        </svg>
        <div class="radar-read">
          <div class="radar-icon" id="mini-radar-icon">🟠</div>
          <div class="radar-title" id="mini-radar-title">Radar</div>
          <div class="radar-price" id="mini-radar-price">—</div>
        </div>
      </div>
      <div class="radar-badges">
        <span id="mini-radar-day" class="radar-badge day">⏳ Ziua —</span>
        <span id="mini-radar-threshold" class="radar-badge threshold">🎯 Bottom —</span>
      </div>
    `;
    anchor.parentNode.insertBefore(card, anchor);
    return card;
  }

  function ensureMap(card) {
    let box = by(STRUCT_ID);
    if (box) return box;
    box = document.createElement("div");
    box.id = STRUCT_ID;
    const badges = card && card.querySelector(".radar-badges");
    if (badges) badges.insertAdjacentElement("afterend", box);
    else if (card) card.appendChild(box);
    return box;
  }

  function set(id, value) {
    const el = by(id);
    if (el) el.textContent = value;
  }

  function triggerStateChange(card, toneClass) {
    if (!card || !lastToneClass || lastToneClass === toneClass) return;
    card.classList.remove("state-changed");
    void card.offsetWidth;
    card.classList.add("state-changed");
    setTimeout(() => card.classList.remove("state-changed"), 2800);
  }

  function row(key, name, value, active) {
    return `
      <div class="sd-row ${active === key ? "active" : ""}">
        <div class="sd-name">${name}</div>
        <div class="sd-value">${value}</div>
      </div>
    `;
  }

  function renderMap(card, z) {
    const box = ensureMap(card);
    if (!box) return;

    if (!z) {
      box.innerHTML = `
        <div class="sd-head">
          <span>${tr("Hartă istorică BTC", "BTC historical map")}</span>
          <span class="sd-mode">${tr("din istoric", "historical")}</span>
        </div>
        <div class="sd-note">${tr("Lipsește adaptive_bottom_zone_summary.json.", "adaptive_bottom_zone_summary.json is missing.")}</div>
      `;
      return;
    }

    const deepTop = Number.isFinite(z.liveInternal) ? z.liveInternal : z.bear;
    const directionText = tr(z.direction.ro, z.direction.en);

    box.innerHTML = `
      <div class="sd-head">
        <span>${tr("Hartă istorică structurală BTC", "BTC historical structural map")}</span>
        <span class="sd-mode">${tr("din istoric", "historical")}</span>
      </div>
      ${row("ath", tr("ATH ciclu / ancoră superioară", "Cycle ATH / upper anchor"), usdK(z.ath), z.key)}
      ${row("repaired", tr("Structură refăcută peste", "Structure repaired above"), usdK(z.bear), z.key)}
      ${row("fragility", tr("Ruptură de fragilitate", "Fragility rupture"), usdK(z.bear), z.key)}
      ${Number.isFinite(z.liveInternal) ? row("internal", tr("Prag fereastră live", "Live window threshold"), usdK(z.liveInternal), z.key) : ""}
      ${row("deep", tr("Degradare profundă", "Deep degradation"), `${usdK(deepTop)} → ${usdK(z.bottomHigh)}`, z.key)}
      ${row("bottom", tr("Bottom final istoric", "Historical final bottom"), `${usdK(z.bottomLow)} – ${usdK(z.bottomHigh)}`, z.key)}
      ${row("bottom-mid", tr("Bottom median", "Median bottom"), usdK(z.bottomMid), z.key)}
      ${row("bottom-absolute", tr("Bottom absolut", "Absolute bottom"), usdK(z.bottomLow), z.key)}
      ${row("capitulation", tr("Capitulare sub", "Capitulation below"), usdK(z.hard), z.key)}
      <div class="sd-note">
        ${tr("Stare", "State")}: ${z.title} · ${tr("Direcție", "Direction")}: ${directionText} · ${tr("trading neschimbat", "trading unchanged")}
      </div>
    `;
  }

  function render(state, risk, summary) {
    lastState = state || lastState;
    lastRisk = risk || lastRisk;
    lastSummary = summary || lastSummary;
    const card = ensureCard();
    if (!card || !lastState) return;

    const live = currentLivePrice() || Number(lastState.price_usd);
    const zone = structuralZone(live, lastSummary, lastState, lastRisk);
    const tone = zone || fallbackToneFor(lastRisk || {});
    const oldTone = lastToneClass;

    card.className = tone.cls;
    if (oldTone && oldTone !== tone.cls) triggerStateChange(card, tone.cls);

    set("mini-radar-icon", tone.icon);
    set("mini-radar-title", tone.title);
    set("mini-radar-price", usdK(live || lastState.price_usd));
    set("mini-radar-day", `⏳ ${tr("Ziua", "Day")} ${(lastRisk && lastRisk.consecutive_degradation_days) ?? "—"} / ~${(lastRisk && lastRisk.median_days_to_confirmation) ?? "—"}`);

    if (zone) {
      set("mini-radar-threshold", `🎯 ${tr("Bottom final", "Final bottom")} ${usdK(zone.bottomLow)}–${usdK(zone.bottomHigh)}`);
    } else {
      const threshold = liveWindowThreshold(lastState, lastRisk);
      set("mini-radar-threshold", `⚙️ ${tr("Prag fereastră", "Window threshold")} ${usdK(threshold)}`);
    }

    card.querySelectorAll(".radar-pulse").forEach(el => el.setAttribute("stroke", tone.color));
    const dot = card.querySelector(".radar-dot");
    if (dot) dot.setAttribute("fill", tone.color);

    renderMap(card, zone);

    lastToneClass = tone.cls;
    lastLang = lang();
  }

  async function load() {
    installStyle();
    if (!ensureCard()) return;
    try {
      const [state, risk, summary] = await Promise.all([
        json("coeziv_state.json"),
        json("risk_window.json"),
        jsonOrNull("adaptive_bottom_zone_summary.json")
      ]);
      render(state, risk, summary);
    } catch (_) {
      set("mini-radar-title", tr("Radar indisponibil", "Radar unavailable"));
    }
  }

  function boot() {
    load();
    clearInterval(refreshTimer);
    clearInterval(liveSyncTimer);
    refreshTimer = setInterval(() => {
      if (lastLang !== lang()) render(lastState, lastRisk, lastSummary);
      else load();
    }, 60 * 1000);
    liveSyncTimer = setInterval(() => render(lastState, lastRisk, lastSummary), 1000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
