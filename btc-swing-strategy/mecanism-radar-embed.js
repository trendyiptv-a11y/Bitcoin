/* CohesivX BTC — compact animated radar insert for mecanism.html, RO/EN aware
   Display is anchored to the historical bottom-zone investigation.
   Trading logic is not modified. */
(function () {
  "use strict";

  const ID = "coeziv-mini-radar";
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
    return (n / 1000).toFixed(1).replace(".0", "") + "K";
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

  function candidate(summary) { return summary && summary.radar_candidate ? summary.radar_candidate : null; }
  function timing(summary) { return summary && summary.structural_timing ? summary.structural_timing : null; }

  function ymdDate(value) {
    const m = String(value || "").match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (!m) return null;
    return new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
  }

  function daysBetween(start, end) {
    const a = ymdDate(start);
    const b = ymdDate(end);
    if (!a || !b) return null;
    return Math.max(0, Math.floor((b.getTime() - a.getTime()) / 86400000));
  }

  function stateDate(state, summary) {
    const s = String((state && (state.timestamp || state.generated_at)) || "").slice(0, 10);
    if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;
    const d = String((summary && summary.date_max) || "");
    if (/^\d{4}-\d{2}-\d{2}$/.test(d)) return d;
    return new Date().toISOString().slice(0, 10);
  }

  function flowDirection(state, risk) {
    const signal = String((state && state.signal) || (risk && risk.current_signal) || "").toLowerCase();
    const flow = String((state && state.flow_bias) || "").toLowerCase();
    if (signal === "long" || flow.includes("pozitiv") || flow.includes("buy") || flow.includes("cump")) return "up";
    if (signal === "short" || flow.includes("negativ") || flow.includes("sell") || flow.includes("vânz") || flow.includes("vanz")) return "down";
    return "flat";
  }

  function pctBetween(value, min, max) {
    const v = Number(value), a = Number(min), b = Number(max);
    if (!Number.isFinite(v) || !Number.isFinite(a) || !Number.isFinite(b) || b <= a) return 0;
    return Math.max(0, Math.min(100, ((v - a) / (b - a)) * 100));
  }

  function timingSnapshot(summary, state) {
    const t = timing(summary);
    if (!t) return null;
    const asOf = stateDate(state, summary);
    const touch = t.fragility_touch_date;
    const confirm = t.close_confirmation_date;
    const touchDays = daysBetween(touch, asOf);
    const confirmDays = daysBetween(confirm, asOf);
    const confirmLag = Number(t.days_from_fragility_touch_to_confirmation);
    const confirmed = !!(confirm && confirmDays !== null && confirmDays >= 0);
    return {
      asOf,
      touch,
      confirm,
      touchDays,
      confirmDays,
      confirmLag: Number.isFinite(confirmLag) ? confirmLag : null,
      confirmed,
      reference: Number(t.fragility_reference_price),
      adaptive: Number(t.adaptive_bear_warning_threshold)
    };
  }

  function zonePack(key, dir, bounds, titles) {
    const isUp = dir === "up";
    const isDown = dir === "down";
    const title = isDown ? titles.down : (isUp ? titles.up : titles.flat);
    const cls = isDown ? titles.downCls : (isUp ? titles.upCls : titles.flatCls);
    const icon = isDown ? titles.downIcon : (isUp ? titles.upIcon : titles.flatIcon);
    const color = cls === "tone-green" ? "#6dffb0" : (cls === "tone-red" ? "#ff5d6c" : "#ffb454");
    return Object.assign({ key, dir, title, cls, icon, color }, bounds);
  }

  function structuralTone(price, summary, state, risk) {
    const c = candidate(summary);
    if (!c || !Number.isFinite(Number(price))) return null;
    const p = Number(price);
    const bounds = {
      ath: Number(c.ath_price),
      bear: Number(c.bear_warning_threshold),
      bottomLow: Number(c.bottom_risk_zone_low),
      bottomMid: Number(c.bottom_risk_zone_mid),
      bottomHigh: Number(c.bottom_risk_zone_high),
      hard: Number(c.hard_capitulation_below),
      timing: timingSnapshot(summary, state)
    };
    const dir = flowDirection(state, risk);

    if (p >= bounds.ath) return zonePack("expansion", dir, bounds, {
      down: tr("Expansiune sub presiune", "Expansion under pressure"), flat: tr("Expansiune structurală", "Structural expansion"), up: tr("Expansiune activă", "Active expansion"),
      downCls: "tone-orange", flatCls: "tone-green", upCls: "tone-green", downIcon: "🟠", flatIcon: "🟢", upIcon: "🟢"
    });

    if (p >= bounds.bear) return zonePack("repaired", dir, bounds, {
      down: tr("Retest de ruptură", "Rupture retest"), flat: tr("Structură refăcută", "Structure repaired"), up: tr("Creștere structurală", "Structural growth"),
      downCls: "tone-orange", flatCls: "tone-green", upCls: "tone-green", downIcon: "🟠", flatIcon: "🟢", upIcon: "🟢"
    });

    if (p > bounds.bottomHigh) return zonePack("deep", dir, bounds, {
      down: tr("Risc de adâncire", "Deepening risk"), flat: tr("Degradare profundă", "Deep degradation"), up: tr("Revenire parțială", "Partial recovery"),
      downCls: "tone-red", flatCls: "tone-red", upCls: "tone-orange", downIcon: "🔴", flatIcon: "🔴", upIcon: "🟠"
    });

    if (p >= bounds.bottomLow) return zonePack("bottom", dir, bounds, {
      down: tr("Risc în bottom final", "Final bottom risk"), flat: tr("Bottom final în test", "Final bottom testing"), up: tr("Ieșire din bottom final", "Leaving final bottom"),
      downCls: "tone-red", flatCls: "tone-orange", upCls: "tone-orange", downIcon: "🔴", flatIcon: "🟠", upIcon: "🟠"
    });

    return zonePack("capitulation", dir, bounds, {
      down: tr("Risc de capitulare", "Capitulation risk"), flat: tr("Capitulare în test", "Capitulation testing"), up: tr("Revenire din capitulare", "Recovery from capitulation"),
      downCls: "tone-red", flatCls: "tone-red", upCls: "tone-orange", downIcon: "🔴", flatIcon: "🔴", upIcon: "🟠"
    });
  }

  function fallbackToneFor(risk) {
    const active = !!(risk && risk.active);
    const days = Number((risk && risk.consecutive_degradation_days) || 0);
    const median = Number((risk && risk.median_days_to_confirmation) || 27);
    const level = String((risk && risk.level) || "");
    if (!active) return { cls: "tone-green", icon: "🟢", title: tr("Structură refăcută", "Structure repaired"), color: "#6dffb0" };
    if (days >= median || level === "high") return { cls: "tone-red", icon: "🔴", title: tr("Degradare persistentă", "Persistent degradation"), color: "#ff5d6c" };
    return { cls: "tone-orange", icon: "🟠", title: tr("Degradare activă", "Active degradation"), color: "#ffb454" };
  }

  function momentBadge(zone, risk) {
    const oldDays = Number((risk && risk.consecutive_degradation_days) || 0);
    const median = Number((risk && risk.median_days_to_confirmation) || 27);
    const ts = zone && zone.timing;
    if (ts && ts.touchDays !== null) {
      if (ts.confirmed) {
        if (zone.dir === "up") return `↗ ${tr("Revenire", "Recovery")} · ${tr("confirmat", "confirmed")} · ${ts.touchDays}z`;
        if (zone.dir === "down") return `⚠️ ${tr("Risc", "Risk")} · ${tr("confirmat", "confirmed")} · ${ts.touchDays}z`;
        return `✓ ${tr("Structură", "Structure")} · ${tr("confirmată", "confirmed")} · ${ts.touchDays}z`;
      }
      return `⏳ ${tr("În test", "Testing")} · ${ts.touchDays}z`;
    }
    if (zone && zone.dir === "up") return `↗ ${tr("Revenire", "Recovery")} · ${oldDays || "—"}z`;
    if (zone && zone.dir === "down") return `⚠️ ${tr("Risc", "Risk")} · ${oldDays || "—"} / ~${median || "—"}`;
    return `⏳ ${tr("Fereastră", "Window")} ${oldDays || "—"} / ~${median || "—"}`;
  }

  function rangeBadgeText(zone) {
    if (!zone) return null;
    if (zone.key === "capitulation") return `🧨 ${tr("Capitulare sub", "Capitulation below")} ${usdK(zone.hard)}`;
    if (zone.key === "bottom") return `🎯 ${tr("Bottom în test", "Bottom testing")} ${usdK(zone.bottomLow)}–${usdK(zone.bottomHigh)}`;
    return `🎯 ${tr("Bottom neatins", "Bottom not reached")} ${usdK(zone.bottomLow)}–${usdK(zone.bottomHigh)}`;
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
      @keyframes coezivRadarSpin{to{transform:rotate(360deg)}}@keyframes coezivRadarSpinFast{to{transform:rotate(360deg)}}@keyframes coezivRadarPulse{0%{opacity:.55;transform:scale(.45)}70%,100%{opacity:0;transform:scale(1)}}@keyframes coezivRadarBreath{0%,100%{r:5}50%{r:6.6}}
      @keyframes coezivRadarAlert{0%{transform:scale(1);box-shadow:inset 0 1px 0 rgba(255,255,255,.04),0 14px 30px rgba(0,0,0,.24)}18%{transform:scale(1.018);box-shadow:0 0 0 2px rgba(255,255,255,.10),0 0 34px var(--radar-alert,rgba(255,180,84,.45)),0 18px 42px rgba(0,0,0,.30)}42%{transform:scale(1)}62%{transform:scale(1.012);box-shadow:0 0 0 1px rgba(255,255,255,.08),0 0 24px var(--radar-alert,rgba(255,180,84,.35)),0 18px 42px rgba(0,0,0,.30)}100%{transform:scale(1)}}
      @keyframes coezivIconPop{0%{transform:scale(1)}20%{transform:scale(1.22)}45%{transform:scale(.96)}70%{transform:scale(1.08)}100%{transform:scale(1)}}
      #${ID}.state-changed{animation:coezivRadarAlert 2.6s ease-out 1}#${ID}.state-changed .radar-stage{filter:drop-shadow(0 0 20px var(--radar-alert,rgba(255,180,84,.45)))}#${ID}.state-changed .radar-sweep{animation:coezivRadarSpinFast .72s linear infinite}#${ID}.state-changed .radar-pulse{animation-duration:1.15s;opacity:.8}#${ID}.state-changed .radar-icon{animation:coezivIconPop 1.1s ease-out 1}
      #${ID} .radar-read{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none}#${ID} .radar-icon{font-size:27px;line-height:1;filter:drop-shadow(0 0 14px currentColor);transform-origin:center;display:inline-block}#${ID} .radar-title{font-size:13px;font-weight:850;color:#e5e7eb;line-height:1.12;margin-top:5px;padding:0 10px}#${ID} .radar-price{font-size:10.5px;color:#94a3b8;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;margin-top:5px}
      #${ID} .radar-badges{display:flex;justify-content:center;align-items:stretch;flex-direction:column;gap:7px;margin-top:2px;padding:0 4px}#${ID} .radar-badge{box-sizing:border-box;border-radius:11px;padding:6px 9px;font-size:10.2px;font-weight:850;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;border:1px solid rgba(148,163,184,.22);background:rgba(2,6,23,.36);color:#cbd5e1;white-space:normal;max-width:100%;overflow-wrap:anywhere;word-break:normal;line-height:1.22;text-align:center}#${ID} .radar-badge.day{border-color:rgba(255,180,84,.36);color:#ffd29c}#${ID} .radar-badge.threshold{border-color:rgba(255,93,108,.38);color:#ffb2ba}
      #${ID} .radar-range{width:min(94%,480px);margin:9px auto 0;padding:9px 10px;border-radius:14px;border:1px solid rgba(148,163,184,.18);background:rgba(2,6,23,.36);text-align:left}#${ID} .range-head{display:flex;justify-content:space-between;gap:8px;align-items:flex-start;color:#e5e7eb;font-size:10.1px;font-weight:900;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;margin-bottom:8px;line-height:1.15}#${ID} .range-head span{min-width:0;overflow-wrap:anywhere}#${ID} .range-head span:last-child{color:#94a3b8;font-weight:800;text-align:right}
      #${ID} .range-track{position:relative;height:10px;border-radius:999px;background:linear-gradient(90deg,rgba(255,93,108,.90) 0%,rgba(255,93,108,.90) var(--bottom-start,8%),rgba(255,180,84,.92) var(--bottom-start,8%),rgba(255,180,84,.92) var(--bottom-end,18%),rgba(45,212,191,.74) var(--bottom-end,18%),rgba(45,212,191,.74) var(--live-pos,58%),rgba(109,255,176,.32) var(--live-pos,58%),rgba(109,255,176,.32) 100%);box-shadow:inset 0 0 0 1px rgba(255,255,255,.07);overflow:visible}#${ID} .range-marker{position:absolute;top:50%;left:var(--live-pos,50%);width:16px;height:16px;border-radius:999px;background:#e5e7eb;border:2px solid currentColor;transform:translate(-50%,-50%);box-shadow:0 0 18px currentColor;color:#ffb454}#${ID} .range-labels{display:flex;justify-content:space-between;gap:6px;color:#94a3b8;font-size:8.7px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;margin-top:6px;line-height:1.15}#${ID} .range-labels span{min-width:0;overflow-wrap:anywhere}
      #${ID}.tone-green{border-color:rgba(109,255,176,.24);--radar-alert:rgba(109,255,176,.46)}#${ID}.tone-orange{border-color:rgba(255,180,84,.24);--radar-alert:rgba(255,180,84,.46)}#${ID}.tone-red{border-color:rgba(255,93,108,.26);--radar-alert:rgba(255,93,108,.50)}
      body.light-mode #${ID}{background:radial-gradient(circle at 50% -18%,rgba(14,165,233,.10),transparent 55%),linear-gradient(180deg,rgba(255,255,255,.94),rgba(248,250,252,.84));border-color:rgba(14,165,233,.25);box-shadow:0 10px 24px rgba(15,23,42,.10)}body.light-mode #${ID} .radar-title{color:#0f172a}body.light-mode #${ID} .radar-range{background:rgba(255,255,255,.78);border-color:rgba(15,23,42,.12)}body.light-mode #${ID} .range-head{color:#0f172a}
      body.radar-scroll-paused #${ID} .radar-sweep,body.radar-scroll-paused #${ID} .radar-pulse,body.radar-scroll-paused #${ID} .radar-dot,body.radar-scroll-paused #${ID}.state-changed,body.radar-scroll-paused #${ID}.state-changed .radar-sweep,body.radar-scroll-paused #${ID}.state-changed .radar-pulse,body.radar-scroll-paused #${ID}.state-changed .radar-icon{animation-play-state:paused!important}body.radar-scroll-paused #${ID}.state-changed .radar-stage{filter:none!important}
      @media(max-width:380px){#${ID} .radar-stage{width:156px;height:156px}#${ID} .radar-badge{font-size:9.5px;padding:6px 7px}#${ID} .range-head{font-size:9.2px}#${ID} .range-labels{font-size:8px}}@media(max-width:768px){#${ID},#${ID} .radar-stage,#${ID} svg{backface-visibility:hidden;-webkit-backface-visibility:hidden;transform:translate3d(0,0,0);-webkit-transform:translate3d(0,0,0)}#${ID} .radar-pulse,#${ID} .radar-sweep,#${ID} .radar-dot{filter:none!important}}
    `;
    document.head.appendChild(s);
  }

  function tickMarks() {
    let out = "";
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      const r1 = 75;
      const r2 = i % 3 === 0 ? 66 : 70;
      out += `<line class="radar-tick" x1="${(86 + r1 * Math.cos(a)).toFixed(1)}" y1="${(86 + r1 * Math.sin(a)).toFixed(1)}" x2="${(86 + r2 * Math.cos(a)).toFixed(1)}" y2="${(86 + r2 * Math.sin(a)).toFixed(1)}"/>`;
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
      <div class="radar-stage"><svg viewBox="0 0 172 172" aria-hidden="true"><circle cx="86" cy="86" r="80" fill="none" stroke="rgba(148,163,184,.18)" stroke-width="1"/><circle cx="86" cy="86" r="55" fill="none" stroke="rgba(148,163,184,.16)" stroke-width="1"/><g>${tickMarks()}</g><circle class="radar-pulse p1" cx="86" cy="86" r="40" fill="none" stroke="#ffb454" stroke-width="1.4"/><circle class="radar-pulse p2" cx="86" cy="86" r="40" fill="none" stroke="#ffb454" stroke-width="1.4"/><g class="radar-sweep"><path d="M86 86 L86 7 A79 79 0 0 1 138 28 Z" fill="rgba(109,255,176,.32)"/></g><circle class="radar-dot" cx="86" cy="86" r="5" fill="#ffb454"/></svg><div class="radar-read"><div class="radar-icon" id="mini-radar-icon">🟠</div><div class="radar-title" id="mini-radar-title">Radar</div><div class="radar-price" id="mini-radar-price">—</div></div></div>
      <div class="radar-badges"><span id="mini-radar-day" class="radar-badge day">⏳ Fereastră —</span><span id="mini-radar-threshold" class="radar-badge threshold">🎯 Bottom —</span></div>
      <div class="radar-range" id="mini-radar-range" aria-label="BTC structural range"><div class="range-head"><span id="mini-range-title">—</span><span id="mini-range-live">—</span></div><div class="range-track" id="mini-range-track"><span class="range-marker" id="mini-range-marker"></span></div><div class="range-labels"><span id="mini-range-low">—</span><span id="mini-range-mid">—</span><span id="mini-range-high">—</span></div></div>
    `;
    anchor.parentNode.insertBefore(card, anchor);
    return card;
  }

  function set(id, value) { const el = by(id); if (el) el.textContent = value; }

  function triggerStateChange(card, toneClass) {
    if (!card || !lastToneClass || lastToneClass === toneClass) return;
    card.classList.remove("state-changed");
    void card.offsetWidth;
    card.classList.add("state-changed");
    setTimeout(() => card.classList.remove("state-changed"), 2800);
  }

  function renderRange(zone, live) {
    const track = by("mini-range-track"), marker = by("mini-range-marker"), box = by("mini-radar-range");
    if (!track || !marker || !box) return;
    if (!zone) { box.style.display = "none"; return; }
    box.style.display = "block";
    const min = zone.bottomLow;
    const max = zone.bear;
    track.style.setProperty("--live-pos", pctBetween(live, min, max).toFixed(2) + "%");
    track.style.setProperty("--bottom-start", pctBetween(zone.bottomLow, min, max).toFixed(2) + "%");
    track.style.setProperty("--bottom-end", pctBetween(zone.bottomHigh, min, max).toFixed(2) + "%");
    marker.style.color = zone.color;
    const confirmed = zone.timing && zone.timing.confirmed ? tr("confirmat", "confirmed") : tr("neconfirmat", "unconfirmed");
    set("mini-range-title", zone.dir === "up" ? `${tr("Traseu revenire", "Recovery path")} · ${confirmed}` : (zone.dir === "down" ? `${tr("Traseu risc", "Risk path")} · ${confirmed}` : `${tr("Traseu structural", "Structural path")} · ${confirmed}`));
    set("mini-range-live", `${tr("Live", "Live")} ${usdK(live)}`);
    set("mini-range-low", `${tr("Cap", "Cap")} ${usdK(zone.bottomLow)}`);
    set("mini-range-mid", `${tr("Mid", "Mid")} ${usdK(zone.bottomMid)}`);
    set("mini-range-high", `${tr("Rupt", "Break")} ${usdK(zone.bear)}`);
  }

  function render(state, risk, summary) {
    lastState = state || lastState;
    lastRisk = risk || lastRisk;
    lastSummary = summary || lastSummary;
    const card = ensureCard();
    if (!card || !lastState) return;
    const live = currentLivePrice() || Number(lastState.price_usd);
    const zone = structuralTone(live, lastSummary, lastState, lastRisk);
    const tone = zone || fallbackToneFor(lastRisk || {});
    const oldTone = lastToneClass;
    card.className = tone.cls;
    if (oldTone && oldTone !== tone.cls) triggerStateChange(card, tone.cls);
    set("mini-radar-icon", tone.icon);
    set("mini-radar-title", tone.title);
    set("mini-radar-price", usdK(live || lastState.price_usd));
    set("mini-radar-day", momentBadge(zone, lastRisk));
    const rangeText = rangeBadgeText(zone);
    if (rangeText) set("mini-radar-threshold", rangeText);
    else set("mini-radar-threshold", `⚙️ ${tr("Prag fereastră", "Window threshold")} —`);
    card.querySelectorAll(".radar-pulse").forEach(el => el.setAttribute("stroke", tone.color));
    const dot = card.querySelector(".radar-dot");
    if (dot) dot.setAttribute("fill", tone.color);
    renderRange(zone, live);
    lastToneClass = tone.cls;
    lastLang = lang();
  }

  async function load() {
    installStyle();
    if (!ensureCard()) return;
    try {
      const [state, risk, summary] = await Promise.all([json("coeziv_state.json"), json("risk_window.json"), jsonOrNull("adaptive_bottom_zone_summary.json")]);
      render(state, risk, summary);
    } catch (_) {
      set("mini-radar-title", tr("Radar indisponibil", "Radar unavailable"));
    }
  }


  let scrollPauseTimer = null;
  function bindScrollPause() {
    if (document.body && document.body.dataset.radarScrollPauseBound === "1") return;
    if (document.body) document.body.dataset.radarScrollPauseBound = "1";
    const pause = () => {
      if (!document.body) return;
      document.body.classList.add("radar-scroll-paused");
      clearTimeout(scrollPauseTimer);
      scrollPauseTimer = setTimeout(() => {
        if (document.body) document.body.classList.remove("radar-scroll-paused");
      }, 480);
    };
    window.addEventListener("scroll", pause, { passive: true });
    window.addEventListener("touchmove", pause, { passive: true });
  }

  function boot() {
    bindScrollPause();
    load();
    clearInterval(refreshTimer);
    clearInterval(liveSyncTimer);
    refreshTimer = setInterval(() => { if (lastLang !== lang()) render(lastState, lastRisk, lastSummary); else load(); }, 60 * 1000);
    // Mobile-safe: rotation stays CSS-based; 1s JS re-render disabled.
    liveSyncTimer = null;
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
