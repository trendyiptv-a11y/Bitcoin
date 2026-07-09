/* CohesivX BTC — RO/EN UI layer. Energy card is owned by mecanism.html. */
(function(){
  "use strict";

  const LANG_KEY = "coeziv_btc_lang";
  const SCALE_KEY = "coeziv_btc_scale";
  const SCALE_STEPS = [1, 1.15, 1.30];
  const USD = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

  const L = {
    ro: {
      app: "MECANISM COEZIV BTC",
      snap: "Actualizat aproximativ la 24h.",
      live: "Preț live:",
      promo: ["CohesivX BTC Monitor pe Android", "Pentru notificări, acces rapid și experiență mai stabilă, descarcă aplicația din Google Play.", "Google Play"],
      daily: ["INTERPRETARE COEZIVĂ ZILNICĂ", "Explicație naturală a stării structurale observate de mecanism.", "Participare", "Fereastră risc", "Fear & Greed / Regim", "Ce urmărim", "Vezi interpretarea completă", "Ascunde interpretarea completă"],
      part: ["COEZIUNE PARTICIPATIVĂ", "Starea interesului participanților în ecosistem.", "Flux", "Lichiditate", "Istoric mediu", "Minim istoric"],
      risk: ["FEREASTRĂ DE RISC", "Context istoric derivat din backtestul coeziv.", "Confirmare istorică", "Timp median", "Regim curent", "Persistență"],
      fg: ["INDICE FEAR & GREED COEZIV", "Sentiment de piață derivat din mecanismul coeziv (structură + semnal + volatilitate), fără date sociale sau externe.", "Structural", "Tactic (24h)", "Tensiune"],
      hist: ["ISTORIC CONTEXTE MECANISM", "Ultimele snapshot-uri ale mecanismului. Nu sunt îndemnuri de intrare automată, ci context structural."],
      regimes: ["REGIMURI STANDARD DE PIAȚĂ", "Mecanismul traduce structura internă a pieței (regim + flux) în câteva contexte standard. Eticheta de sus, „Regim de piață”, va fi întotdeauna una dintre situațiile de mai jos."]
    },
    en: {
      app: "COHESIVE BTC MECHANISM",
      snap: "Mechanism snapshot: updated approximately every 24h.",
      live: "Live price:",
      promo: ["CohesivX BTC Monitor for Android", "For notifications, quick access and a more stable experience, download the app from Google Play.", "Google Play"],
      daily: ["DAILY COHESIVE INTERPRETATION", "Natural-language explanation of the structural state observed by the mechanism.", "Participation", "Risk window", "Fear & Greed / Regime", "What to watch", "View full interpretation", "Hide full interpretation"],
      part: ["PARTICIPATION COHESION", "State of participant interest across the ecosystem.", "Flow", "Liquidity", "Historical average", "Historical minimum"],
      risk: ["RISK WINDOW", "Historical context derived from the cohesive backtest.", "Historical confirmation", "Median time", "Current regime", "Persistence"],
      fg: ["COHESIVE FEAR & GREED INDEX", "Market sentiment derived from the cohesive mechanism: structure, signal and volatility, without social or external data.", "Structural", "Tactical (24h)", "Tension"],
      hist: ["MECHANISM CONTEXT HISTORY", "Recent mechanism snapshots. These are not entry prompts, but structural context."],
      regimes: ["STANDARD MARKET REGIMES", "The mechanism translates the market’s internal structure, regime and flow into a small set of standard contexts. The top regime label will always match one of the situations below."]
    }
  };

  const STD = {
    up_trend_strong:["1. Trend ascendent susținut","1. Sustained upward trend","Cumpărătorii domină clar piața, iar corecțiile tind să fie rapide și superficiale. Context tipic pentru faze de expansiune, cu risc crescut de supra-extindere pe termen scurt.","Buyers clearly dominate the market, while pullbacks tend to be quick and shallow. This is typical of expansion phases, with increased short-term overextension risk."],
    up_trend_moderate:["2. Trend ascendent moderat","2. Moderate upward trend","Trend pozitiv, dar cu corecții regulate. Presiunea de creștere există, însă mișcările sunt mai echilibrate între impulsuri și pullback-uri.","Positive trend, but with regular corrections. Upward pressure exists, while movement is more balanced between impulses and pullbacks."],
    range_bias_up:["3. Range cu bias pozitiv","3. Range with positive bias","Piața consolidează într-un interval, dar maximele sunt împinse treptat în sus. Context favorabil pentru acumulare graduală.","The market consolidates within a range, but highs are gradually pushed upward. This supports gradual accumulation."],
    range_neutral:["4. Range neutru","4. Neutral range","Echilibru relativ între cumpărători și vânzători, fără direcție clară. Mișcările laterale domină.","Relative balance between buyers and sellers, with no clear direction. Sideways movement dominates."],
    range_bias_down:["5. Range cu bias negativ","5. Range with negative bias","Piața consolidează, dar minimele sunt coborâte treptat. Presiunea de vânzare se acumulează.","The market is consolidating, but lows are gradually pushed lower. Selling pressure is accumulating."],
    down_trend_moderate:["6. Trend descendent moderat","6. Moderate downward trend","Presiune de scădere prezentă, însă cu pauze și contra-mișcări regulate.","Downward pressure is present, but with pauses and regular counter-moves."],
    down_trend_strong:["7. Trend descendent susținut","7. Sustained downward trend","Vânzătorii controlează clar piața, iar revenirile sunt de obicei scurte și fragile.","Sellers clearly control the market, and rebounds are usually short and fragile."],
    transition_accumulation:["8. Regim neutru / tranziție","8. Neutral / transition regime","Zone de schimbare, în care modelul nu vede încă un avantaj structural clar nici pentru cumpărători, nici pentru vânzători.","Change zones where the model does not yet see a clear structural advantage for buyers or sellers."]
  };

  function lang(){ return localStorage.getItem(LANG_KEY) === "en" ? "en" : "ro"; }
  function en(){ return lang() === "en"; }
  function t(){ return L[lang()]; }
  function by(id){ return document.getElementById(id); }
  function set(el, text){ if (el && typeof text === "string") el.textContent = text; }
  function n(v, d=0){ return Number.isFinite(Number(v)) ? Number(v).toFixed(d) : "n/a"; }
  function p(v, d=1){ return Number.isFinite(Number(v)) ? (Number(v)*100).toFixed(d) : "n/a"; }
  function u(v){ return Number.isFinite(Number(v)) ? USD.format(Number(v)) : "n/a"; }
  async function json(url){ const r = await fetch(`${url}?t=${Date.now()}`, { cache: "no-store" }); if(!r.ok) throw new Error(url); return r.json(); }

  function signalLabel(s){ s=String(s||"flat").toLowerCase(); return en() ? (s==="long"?"Context: upward pressure":s==="short"?"Context: downside risk":"Context: neutral") : (s==="long"?"Context: presiune de creștere":s==="short"?"Context: risc de scădere":"Context: neutru"); }
  function signalShort(s){ s=String(s||"flat").toLowerCase(); return en() ? (s==="long"?"Upward pressure":s==="short"?"Downside risk":"Neutral context") : (s==="long"?"Presiune de creștere":s==="short"?"Risc de scădere":"Context neutru"); }
  function regimeChip(full,code){ const x=`${full||""} ${code||""}`.toLowerCase(); if(x.includes("range")&&x.includes("bias_up"))return"RANGE+"; if(x.includes("range")&&x.includes("bias_down"))return"RANGE−"; if(x.includes("range"))return"RANGE"; if(x.includes("up_trend")||x.includes("ascendent"))return"TREND+"; if(x.includes("down_trend")||x.includes("descendent"))return"TREND−"; if(x.includes("transition")||x.includes("tranzi"))return"TRANZ"; if(x.includes("neutral")||x.includes("neutru"))return"NEUTRU"; return x.includes("n/a")?"N/A":"REGIM"; }

  function renderEnergy(){
    // IMPORTANT: mecanism.html owns the dynamic energy card.
    // This i18n layer must not rebuild the old static card, because that caused
    // “Miner mediu / Miner scump” to overwrite the new live-status card.
    if (typeof window.COHESIVX_RENDER_DYNAMIC_ENERGY_CARD === "function") {
      window.COHESIVX_RENDER_DYNAMIC_ENERGY_CARD();
    }
  }

  function translateLiveText(){
    const live = by("live-delta"), dev = by("deviation-status");
    if (live && en()) {
      live.textContent = live.textContent
        .replace("Prețul live este", "The live price is")
        .replace("peste prețul mecanismului", "above the mechanism price")
        .replace("sub prețul mecanismului", "below the mechanism price")
        .replace("Diferență foarte mică (zgomot normal de piață).", "Very small difference (normal market noise).")
        .replace("Diferență foarte mică; poate fi zgomot normal de piață.", "Very small difference; this may be normal market noise.")
        .replace("Mișcare mică față de mecanism; reperul structural rămâne principal.", "Small movement versus the mechanism; the structural reference remains primary.")
        .replace("Mișcare relevantă față de mecanism; urmărește contextul și nivelurile de risc.", "Relevant movement versus the mechanism; watch context and risk levels.")
        .replace("Mișcare relevantă față de reper; merită urmărit contextul.", "Relevant movement versus the reference; context should be watched.")
        .replace("Mișcare puternică; verifică contextul și lichiditatea.", "Strong movement; check context and liquidity.")
        .replace("Mișcare puternică față de reper; verifică regimul și riscul.", "Strong movement versus the reference; check regime and risk.")
        .replace("Prețul live este egal cu prețul analizat de mecanism.", "The live price is equal to the mechanism price.");
    }
    if (dev && en()) {
      dev.textContent = dev.textContent
        .replace("Status deviație:", "Deviation status:")
        .replace("Normală", "Normal")
        .replace("Controlată", "Controlled")
        .replace("Tensionată", "Tense")
        .replace("Extremă", "Extreme")
        .replace("mișcare de zgomot față de mecanism", "noise-level movement versus the mechanism")
        .replace("diferență mică față de reper", "small difference from the reference")
        .replace("prețul se îndepărtează vizibil de mecanism", "price is visibly moving away from the mechanism")
        .replace("prețul este mult decuplat de reperul coeziv", "price is strongly decoupled from the cohesive reference")
        .replace("fără abatere față de model", "no deviation from the model")
        .replace("așteptăm un snapshot al modelului", "waiting for a model snapshot");
    }
  }

  function flowText(bias,str){ const s=str==="slab"?"weak":str==="puternică"||str==="puternic"?"strong":str||""; if(en()){ if(bias==="pozitiv")return`Market flow: tilted toward buying${s?` (${s})`:""}.`; if(bias==="negativ")return`Market flow: tilted toward selling${s?` (${s})`:""}.`; if(bias==="neutru")return`Market flow: relatively balanced between buyers and sellers${s?` (${s})`:""}.`; return"Market flow: not available yet."; } let r="Flux de piață: n/a (așteptăm date despre flux)."; if(bias==="pozitiv")r="Flux de piață: orientat spre cumpărare"; else if(bias==="negativ")r="Flux de piață: orientat spre vânzare"; else if(bias==="neutru")r="Flux de piață: relativ echilibrat între cumpărători și vânzători"; return bias&&str?`${r} (${str}).`:bias?`${r}.`:r; }
  function liqText(reg,str){ const r=String(reg||"").trim().toLowerCase(),s=str==="puternică"||str==="puternic"?"strong":str==="slab"?"weak":str||""; if(en()){ if(r==="ridicată")return`Market liquidity: high; movement is above usual levels${s?`, with a ${s} deviation from normal activity`:""}.`; if(r==="scăzută")return`Market liquidity: low; movement is below usual levels${s?`, with a ${s} deviation from normal activity`:""}.`; if(r==="normală"||r==="moderată")return"Market liquidity: moderate for this context."; return"Market liquidity: not available yet."; } let out="Lichiditate piață: n/a (așteptăm date suplimentare)"; if(r==="ridicată")out="Lichiditate piață: ridicată; mișcări peste nivelurile obișnuite."; else if(r==="scăzută")out="Lichiditate piață: scăzută; mișcări sub nivelurile obișnuite."; else if(r==="normală"||r==="moderată")out="Lichiditate piață: moderată pentru acest context."; if(reg&&str&&r!=="normală"&&r!=="moderată")out+=` (deviație ${str} față de nivelurile obișnuite.)`; return out; }
  function riskRegime(x){ const k=String(x||"").toLowerCase(),ro={bear_struct:"Degradare structurală",bear_late:"Degradare avansată",accum_bear:"Acumulare fragilă",bull_struct:"Structură pozitivă",bull_late:"Expansiune matură",accum_bull:"Acumulare pozitivă",range_pos:"Range cu bias pozitiv",range_neg:"Range cu bias negativ",range_neutral:"Range neutru",neutral:"Tranziție neutră"},em={bear_struct:"Structural degradation",bear_late:"Advanced degradation",accum_bear:"Fragile accumulation",bull_struct:"Positive structure",bull_late:"Mature expansion",accum_bull:"Positive accumulation",range_pos:"Range with positive bias",range_neg:"Range with negative bias",range_neutral:"Neutral range",neutral:"Neutral transition"}; return (en()?em[k]:ro[k])||x||"n/a"; }
  function riskLevel(level,active){ return en()?(!active?"Normal window":level==="high"?"High structural risk":level==="moderate"?"Moderate structural risk":level==="low"?"Low structural risk":"Structural risk n/a"):(!active?"Fereastră normală":level==="high"?"Risc structural ridicat":level==="moderate"?"Risc structural moderat":level==="low"?"Risc structural scăzut":"Risc structural n/a"); }

  async function renderState(){
    try{
      const s=await json("coeziv_state.json"), d=s.signal_expected_drift||{}, b=s.signal_prob_breakdown||{}, h=s.signal_prob_horizon_hours||24, r=s.market_regime||{}, rl=typeof r==="string"?r:r.label||"n/a", rc=typeof r==="object"?r.code||"":"";
      set(by("signal-label"), signalLabel(s.signal));
      const re=by("regime-line"); if(re){ const full=en()?`Market regime: ${rl}.`:`Regim de piață: ${rl}.`; re.dataset.fullRegime=full; re.title=full; re.textContent=regimeChip(full,rc); }
      if(en()){
        set(by("message"), `At the mechanism update, BTC was trading around ${u(s.price_usd)} USD. The mechanism reads the current context as neutral. Market flow is relatively balanced and weak. Liquidity is high, so movement is more easily absorbed. Price remains close to the network’s estimated production-cost equilibrium.`);
        set(by("signal-prob"), `Historical probability: about ${p(s.signal_probability,0)}% for price to move with the structural context over the next ${h}h, based on ${s.signal_prob_samples||0} similar historical contexts.`);
        set(by("signal-prob-breakdown"), `Historical distribution over ${h}h: about ${p(b.in_direction,0)}% with the structural context, ${p(b.opposite,0)}% against the structural context, and ${p(b.flat,0)}% neutral movement or noise.`);
        set(by("drift-range"), `Typical movement range (${h}h): between ~${p(d.p10)}% and ~${p(d.p90)}%, with a median near ~${p(d.p50)}%.`);
      }
      set(by("flow-line"), flowText(s.flow_bias,s.flow_strength));
      set(by("liquidity-line"), liqText(s.liquidity_regime,s.liquidity_strength));
      translateLiveText();
      renderEnergy();
      renderHistory(s);
    }catch(_){ translateLiveText(); renderEnergy(); }
  }

  async function renderDaily(){ try{ const d=await json("daily_cohesiv_interpretation.json"), e=en(); set(by("daily-ai-state"), e?d.general_state_en||d.general_state:d.general_state); set(by("daily-ai-summary"), e?d.plain_language_en||d.summary_en||d.plain_language||d.summary:d.plain_language||d.summary); set(by("daily-ai-participation"), e?d.participation_en||d.participation:d.participation); set(by("daily-ai-risk"), e?d.risk_window_en||d.risk_window:d.risk_window); set(by("daily-ai-fg-regime"), e?[d.fear_greed_en||d.fear_greed,d.market_regime_en||d.market_regime].filter(Boolean).join(" • "):[d.fear_greed,d.market_regime].filter(Boolean).join(" • ")); set(by("daily-ai-watch"), e?d.watch_next_en||d.watch_next:d.watch_next); set(by("daily-ai-footer"), e?d.disclaimer_en||"Experimental structural interpretation, not financial advice.":d.disclaimer||"Interpretare structurală experimentală, nu recomandare financiară."); document.querySelectorAll("#daily-ai-card .daily-ai-label").forEach((x,i)=>set(x,t().daily[i+2])); const panel=by("daily-ai-full-panel"),btn=by("daily-ai-full-toggle"),full=e?d.full_interpretation_en||d.full_interpretation:d.full_interpretation; if(panel&&full)panel.textContent=full.replace(/^##\s+/gm,"").replace(/^###\s+/gm,"").replace(/\*\*(.*?)\*\*/g,"$1").trim(); if(btn)btn.textContent=panel&&panel.classList.contains("open")?t().daily[7]:t().daily[6]; }catch(_){} }
  async function renderParticipation(){ try{ const d=await json("participation_cohesion.json"),c=d.components||{},h=d.history||{},pill=by("participation-cohesion-pill"); if(pill){pill.className=`participation-pill participation-${d.level||"tense"}`;pill.textContent=en()?(d.level==="cohesive"?"Cohesive participation":"Tense participation"):((d.label||"participare necunoscută").replace(/^./,m=>m.toUpperCase()));} set(by("participation-cohesion-score"),`${n(d.score)} / 100`); set(by("participation-flow"),n(c.flow_component)); set(by("participation-liquidity"),n(c.liquidity_component)); set(by("participation-history-mean"),n(h.mean_score)); set(by("participation-history-min"),n(h.min_score)); set(by("participation-cohesion-text"),en()?"Participants remain active, but behavior is tense. Interest persists, while the dominant flow may be defensive or oriented toward exit.":d.main_text); set(by("participation-cohesion-footer"),en()?"Experimental indicator derived from flow, liquidity, persistence and tension against the model.":d.footer); document.querySelectorAll("#participation-cohesion-card .participation-metric-label").forEach((x,i)=>set(x,t().part[i+2])); }catch(_){} }
  async function renderRisk(){ try{ const d=await json("risk_window.json"),active=!!d.active,level=d.level||(active?"unknown":"normal"),pill=by("risk-window-pill"); if(pill){pill.className=`risk-pill risk-${active?level:"normal"}`;pill.textContent=riskLevel(level,active);} set(by("risk-window-rate"),`${n((d.historical_confirmation_rate||0)*100)}%`); set(by("risk-window-days"),en()?`~${n(d.median_days_to_confirmation)} days`:`~${n(d.median_days_to_confirmation)} zile`); set(by("risk-window-regime"),riskRegime(d.current_regime)); set(by("risk-window-streak"),en()?`${d.consecutive_degradation_days||0} days`:`${d.consecutive_degradation_days||0} zile`); set(by("risk-window-text"),en()?`The current context belongs to a historical family that produced major weakening in about ${n((d.historical_confirmation_rate||0)*100)}% of cases. Confirmation appeared after roughly ${n(d.median_days_to_confirmation)} days on average.`:d.main_text); set(by("risk-window-footer"),en()?"Statistical interpretation of structural degradation, not a trading recommendation.":d.footer); document.querySelectorAll("#risk-window-card .risk-metric-label").forEach((x,i)=>set(x,t().risk[i+2])); }catch(_){} }
  function renderHistory(s){ document.querySelectorAll("#history-list .history-row").forEach(row=>{ const chip=row.querySelector(".history-signal-chip"),tag=row.querySelector(".history-tag"); if(chip){ const cls=chip.className||"",sig=cls.includes("history-signal-long")?"long":cls.includes("history-signal-short")?"short":"flat",nodes=Array.from(chip.childNodes).filter(n=>n.nodeType===Node.TEXT_NODE); if(nodes.length)nodes[nodes.length-1].nodeValue=" "+signalShort(sig); } if(tag&&/Preț model|Model price/.test(tag.textContent))tag.textContent=en()?"Model price at snapshot":"Preț model la snapshot"; }); const meta=by("history-meta"); if(!meta||!s)return; const h=Array.isArray(s.signal_history)?s.signal_history:[],up=h.filter(x=>x.signal==="long").length,dn=h.filter(x=>x.signal==="short").length,fl=h.filter(x=>x.signal==="flat").length; meta.innerHTML=en()?`<div class="history-meta-summary"><span>Contexts: ${h.length}</span><span class="history-meta-dot">•</span><span>Up: ${up}</span><span class="history-meta-dot">•</span><span>Down: ${dn}</span><span class="history-meta-dot">•</span><span>Neutral: ${fl}</span></div>`:`<div class="history-meta-summary"><span>Total contexte: ${h.length}</span><span class="history-meta-dot">•</span><span>Creștere: ${up}</span><span class="history-meta-dot">•</span><span>Scădere: ${dn}</span><span class="history-meta-dot">•</span><span>Neutru: ${fl}</span></div>`; }
  function renderStd(){ Object.entries(STD).forEach(([k,v])=>{ const row=document.querySelector(`[data-standard-regime="${k}"]`); if(row){ set(row.querySelector(".history-date"),en()?v[1]:v[0]); set(row.querySelector(".history-tag"),en()?v[3]:v[2]); }}); }
  function renderPromo(){ const c=t().promo; set(document.querySelector(".app-download-title"),c[0]); set(document.querySelector(".app-download-subtitle"),c[1]); set(document.querySelector(".app-download-btn"),c[2]); }
  function renderHeadings(){ const c=t(); set(document.querySelector(".title-bar"),c.app); set(document.querySelector(".asset-text span"),c.snap); set(document.querySelector(".live-label"),c.live); renderPromo(); set(document.querySelector("#daily-ai-card .daily-ai-title"),c.daily[0]); set(document.querySelector("#daily-ai-card .daily-ai-subtitle"),c.daily[1]); set(document.querySelector("#participation-cohesion-card .participation-title"),c.part[0]); set(document.querySelector("#participation-cohesion-card .participation-subtitle"),c.part[1]); set(document.querySelector("#risk-window-card .risk-title"),c.risk[0]); set(document.querySelector("#risk-window-card .risk-subtitle"),c.risk[1]); set(document.querySelector(".fg-title"),c.fg[0]); set(document.querySelector(".fg-subtitle"),c.fg[1]); const heads=Array.from(document.querySelectorAll(".history-header")),h=heads.find(x=>/ISTORIC|HISTORY|CONTEXT HISTORY/.test(x.textContent)),r=heads.find(x=>/REGIMURI|STANDARD MARKET/.test(x.textContent)); if(h){set(h.querySelector(".history-title"),c.hist[0]);set(h.querySelector(".history-subtitle"),c.hist[1]);} if(r){set(r.querySelector(".history-title"),c.regimes[0]);set(r.querySelector(".history-subtitle"),c.regimes[1]);} }

  function injectStyles(){ if(by("coeziv-i18n-style"))return; const s=document.createElement("style"); s.id="coeziv-i18n-style"; s.textContent=`body[data-lang="en"] .standard-regime-row.active-standard-regime::after{content:"DETECTED NOW" !important;}body[data-lang="ro"] .standard-regime-row.active-standard-regime::after{content:"DETECTAT ACUM" !important;}#coeziv-accessibility-panel{position:fixed;right:10px;bottom:14px;z-index:2147483647;display:flex;align-items:center;gap:6px;padding:6px;border-radius:999px;border:1px solid rgba(56,189,248,.42);background:linear-gradient(135deg,rgba(15,23,42,.94),rgba(12,74,110,.90));color:#e5e7eb;box-shadow:0 14px 42px rgba(0,0,0,.38);font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:11px;line-height:1}#coeziv-accessibility-panel.collapsed{padding:5px 8px}#coeziv-accessibility-panel .coeziv-compact-toggle{border:0;border-radius:999px;background:rgba(56,189,248,.18);color:#bae6fd;min-width:96px;height:30px;padding:0 10px;cursor:pointer;font-size:11px;font-weight:900;letter-spacing:.03em;line-height:30px;text-align:center;white-space:nowrap}#coeziv-accessibility-panel .coeziv-panel-controls{display:inline-flex;align-items:center;gap:5px}#coeziv-accessibility-panel.collapsed .coeziv-panel-controls{display:none}#coeziv-accessibility-panel button:not(.coeziv-compact-toggle){border:0;border-radius:999px;background:transparent;color:inherit;min-width:34px;height:30px;padding:0 9px;cursor:pointer;font-size:12px;font-weight:800;line-height:30px;text-align:center}#coeziv-accessibility-panel button.active{background:rgba(56,189,248,.25);color:#7dd3fc}#coeziv-accessibility-panel .sep{width:1px;height:20px;background:rgba(148,163,184,.28)}body.light-mode #coeziv-accessibility-panel{background:linear-gradient(135deg,rgba(255,255,255,.96),rgba(224,242,254,.92));color:#0f172a}`; document.head.appendChild(s); }
  function applyScale(){ const v=SCALE_STEPS.includes(parseFloat(localStorage.getItem(SCALE_KEY)||"1"))?parseFloat(localStorage.getItem(SCALE_KEY)||"1"):1; document.documentElement.style.setProperty("--coeziv-ui-scale",String(v)); document.body.style.zoom=String(v); document.body.style.transformOrigin="top center"; }
  function compactLabel(){ const b=document.querySelector("#coeziv-accessibility-panel .coeziv-compact-toggle"); if(!b)return; const v=parseFloat(localStorage.getItem(SCALE_KEY)||"1"); b.textContent=`RO/EN · ${v===1.15?"A×1.15":v===1.30?"A×1.30":"A+"}`; }
  function buttons(){ document.querySelectorAll("#coeziv-accessibility-panel button[data-lang]").forEach(b=>b.classList.toggle("active",b.dataset.lang===lang())); }
  function setLang(v){ localStorage.setItem(LANG_KEY,v==="en"?"en":"ro"); document.documentElement.setAttribute("lang",lang()); if(document.body)document.body.setAttribute("data-lang",lang()); applyAll(); }
  function setScale(v){ v=SCALE_STEPS.includes(Number(v))?Number(v):1; localStorage.setItem(SCALE_KEY,String(v)); applyScale(); buttons(); compactLabel(); }
  function bump(dir){ const cur=parseFloat(localStorage.getItem(SCALE_KEY)||"1"),i=Math.max(0,SCALE_STEPS.indexOf(cur)); setScale(SCALE_STEPS[Math.min(SCALE_STEPS.length-1,Math.max(0,i+dir))]); }
  function panel(){ if(by("coeziv-accessibility-panel"))return; const p=document.createElement("div"); p.id="coeziv-accessibility-panel"; p.className="collapsed"; p.innerHTML=`<button type="button" class="coeziv-compact-toggle" aria-label="RO/EN and text size">RO/EN · A+</button><div class="coeziv-panel-controls"><button type="button" data-lang="ro">RO</button><button type="button" data-lang="en">EN</button><span class="sep"></span><button type="button" data-scale-action="down">A−</button><button type="button" data-scale-action="up">A+</button></div>`; document.body.appendChild(p); p.querySelector(".coeziv-compact-toggle").addEventListener("click",()=>p.classList.toggle("collapsed")); p.querySelectorAll("button[data-lang]").forEach(b=>b.addEventListener("click",()=>setLang(b.dataset.lang))); p.querySelector("[data-scale-action='up']").addEventListener("click",()=>bump(1)); p.querySelector("[data-scale-action='down']").addEventListener("click",()=>bump(-1)); }
  function applyAll(){ renderHeadings(); translateLiveText(); renderEnergy(); renderStd(); renderState(); renderDaily(); renderParticipation(); renderRisk(); buttons(); compactLabel(); }
  function init(){ injectStyles(); panel(); applyScale(); document.documentElement.setAttribute("lang",lang()); document.body.setAttribute("data-lang",lang()); applyAll(); setInterval(()=>{ renderHeadings(); translateLiveText(); renderEnergy(); }, 1500); }
  if(document.readyState === "loading") document.addEventListener("DOMContentLoaded",init); else init();
})();
