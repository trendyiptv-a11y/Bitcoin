(function(){
  'use strict';

  var LANG_KEY = 'coeziv_btc_lang';

  function num(v, fallback){
    var n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  }

  function lang(){
    try{
      return localStorage.getItem(LANG_KEY) === 'en' ||
        document.documentElement.lang === 'en' ||
        (document.body && document.body.getAttribute('data-lang') === 'en') ? 'en' : 'ro';
    }catch(e){ return 'ro'; }
  }

  function pick(ro, en){ return lang() === 'en' ? en : ro; }

  function signalLabel(signal){
    var s = String(signal || 'flat').toLowerCase();
    if (lang() === 'en') {
      if (s === 'long') return 'UPWARD PRESSURE';
      if (s === 'short') return 'DOWNSIDE RISK';
      return 'WAIT';
    }
    if (s === 'long') return 'PRESIUNE DE CREȘTERE';
    if (s === 'short') return 'RISC DE SCĂDERE';
    return 'AȘTEAPTĂ';
  }

  function currentStreak(risk){
    var a = risk && risk.since_2025_12_summary ? risk.since_2025_12_summary.current_streak_days : null;
    var b = risk && risk.legacy_risk_window ? risk.legacy_risk_window.consecutive_degradation_days : null;
    var c = risk ? risk.consecutive_degradation_days : null;
    return num(a, num(b, num(c, 0)));
  }

  function avgDays(risk){
    var a = risk && risk.legacy_risk_window ? risk.legacy_risk_window.average_days_to_confirmation : null;
    var b = risk ? risk.average_days_to_confirmation : null;
    return Math.round(num(a, num(b, 41)));
  }

  function precedentDays(risk){
    var arr = risk && risk.since_2025_12_summary && Array.isArray(risk.since_2025_12_summary.top_streaks) ? risk.since_2025_12_summary.top_streaks : [];
    var max = 0;
    arr.forEach(function(x){ if (!x.is_current && num(x.days, 0) > max) max = num(x.days, 0); });
    return max || num(risk && risk.since_2025_12_summary && risk.since_2025_12_summary.max_streak_days, 74);
  }

  function participationScore(state){
    return num(state && (state.participation_score || state.participation_cohesion_score), null);
  }

  function flowBias(state){
    return String((state && (state.flow_bias || (state.flow && state.flow.bias))) || '').toLowerCase();
  }

  function flowStrength(state){
    return String((state && (state.flow_strength || (state.flow && state.flow.strength))) || '').toLowerCase();
  }

  function liquidityRegime(state){
    return String((state && (state.liquidity_regime || (state.liquidity && state.liquidity.regime))) || '').toLowerCase();
  }

  function priceChangeHint(state){
    var h = state && state.signal_history;
    if (!Array.isArray(h) || h.length < 2) return 0;
    var last = h[h.length - 1] || {};
    var prev = h[h.length - 2] || {};
    var p1 = num(last.model_price_usd || last.price_usd || last.ic_close_usd, null);
    var p0 = num(prev.model_price_usd || prev.price_usd || prev.ic_close_usd, null);
    if (!Number.isFinite(p1) || !Number.isFinite(p0) || p0 === 0) return 0;
    return (p1 - p0) / p0;
  }

  function scenario(risk, state){
    risk = risk || {};
    state = state || {};
    var day = currentStreak(risk);
    var avg = avgDays(risk);
    var precedent = precedentDays(risk);
    var signal = String(state.signal || risk.current_signal || 'flat').toLowerCase();
    var ps = participationScore(state);
    var fb = flowBias(state);
    var fs = flowStrength(state);
    var liq = liquidityRegime(state);
    var nearPrecedent = precedent > 0 && day >= Math.round(precedent * 0.80);
    var aboveAvg = day > avg;
    var priceUp = priceChangeHint(state) > 0.008;
    var weakFlow = fb.indexOf('neutru') !== -1 || fb.indexOf('neutral') !== -1 || fs.indexOf('slab') !== -1 || fs.indexOf('weak') !== -1;
    var highLiquidity = liq.indexOf('ridicat') !== -1 || liq.indexOf('high') !== -1 || liq.indexOf('bun') !== -1;

    var id = 'daily_default';
    if (signal === 'long') id = 'structure_confirming';
    else if (signal === 'short') id = 'structural_pressure';
    else if (nearPrecedent) id = 'testing_limit';
    else if (aboveAvg) id = 'tactic_changed';
    else if (priceUp && signal === 'flat') id = 'price_up_no_confirm';
    else if (highLiquidity && weakFlow) id = 'liquidity_without_direction';
    else if (Number.isFinite(ps) && ps >= 72) id = 'participation_repair';

    var data = {
      id:id,
      day:day || '–',
      avg:avg,
      precedent:precedent,
      signal:signalLabel(signal),
      signalRaw:signal,
      priceChange:priceChangeHint(state),
      flowBias:fb,
      flowStrength:fs,
      liquidityRegime:liq,
      participationScore:ps
    };

    var ro = {
      structure_confirming:{kicker:'INFORMAȚIA ZILEI',title:'STRUCTURA ÎNCEPE SĂ CONFIRME',badge:'CONFIRMARE STRUCTURALĂ',subtitle:'Semnalul mecanismului s-a schimbat. Nu mai privim doar prețul, ci coeziunea din spatele mișcării.',warning1:'DIRECȚIA NU MAI ESTE DOAR NARAȚIUNE.',warning2:'CONFIRMAREA TREBUIE URMĂRITĂ ÎN CONTEXT.',meaning:['Semnalul mecanismului a trecut în regim pozitiv.','Participarea și fluxul trebuie să susțină mișcarea.','Tactica: confirmare, disciplină și risc controlat.'],footer:'CÂND STRUCTURA CONFIRMĂ, PREȚUL NU MAI ESTE SINGURUL ARGUMENT.'},
      structural_pressure:{kicker:'INFORMAȚIA ZILEI',title:'PRESIUNE STRUCTURALĂ ACTIVĂ',badge:'RISC STRUCTURAL',subtitle:'Mecanismul vede presiune, nu doar volatilitate. Zona trebuie tratată cu prudență.',warning1:'RISCUL NU MAI ESTE DOAR LOCAL.',warning2:'STRUCTURA CERE PRUDENȚĂ.',meaning:['Semnalul indică presiune descendentă.','Lichiditatea poate accelera mișcările.','Tactica: reducerea expunerii și observare.'],footer:'CÂND STRUCTURA APASĂ, REACȚIA TREBUIE SĂ FIE MAI LENTĂ DECÂT EMOȚIA.'},
      testing_limit:{kicker:'INFORMAȚIA ZILEI',title:'PIAȚA TESTEAZĂ LIMITA',badge:'APROAPE DE PRECEDENT',subtitle:'Range-ul curent se apropie de precedentul major. Timpul devine la fel de important ca prețul.',warning1:'PRECEDENTUL 2026 DEVINE REPER TACTIC.',warning2:'FIECARE ZI ÎN PLUS CONTEAZĂ.',meaning:['Durata curentă se apropie de recordul recent.','Modelele medii nu mai sunt suficiente.','Tactica: răbdare, context și confirmare.'],footer:'PIAȚA TESTEAZĂ LIMITA DE TIMP A MECANISMULUI.'},
      tactic_changed:{kicker:'INFORMAȚIA ZILEI',title:'TACTICA S-A SCHIMBAT',badge:'RANGE PRELUNGIT',subtitle:'Piața stă peste media istorică în mecanism. Semnalul principal rămâne prudența.',warning1:'MEDIA ISTORICĂ A FOST DEPĂȘITĂ CA PRAG TACTIC.',warning2:'PRECEDENTUL 2026 RĂMÂNE REPERUL MAJOR.',meaning:['Structura s-a prelungit peste media istorică.','Modelele vechi pot da semnale înșelătoare.','Noua tactică: răbdare + adaptare + disciplină.'],footer:'PIAȚA A ÎNVĂȚAT SĂ STEA MAI MULT ÎN MECANISM.'},
      price_up_no_confirm:{kicker:'INFORMAȚIA ZILEI',title:'PREȚUL A URCAT, STRUCTURA NU A CONFIRMAT',badge:'POMPĂ FĂRĂ CONFIRMARE',subtitle:'Mișcarea de preț există, dar mecanismul nu vede încă o reparație structurală completă.',warning1:'PREȚUL SINGUR NU ESTE CONFIRMARE.',warning2:'SEMNALUL RĂMÂNE AȘTEAPTĂ.',meaning:['Prețul poate urca fără reparație structurală.','Fluxul și participarea trebuie să confirme.','Tactica: nu confunda impulsul cu schimbarea de regim.'],footer:'CÂND PREȚUL FUGE ÎNAINTEA STRUCTURII, MECANISMUL AȘTEAPTĂ.'},
      liquidity_without_direction:{kicker:'INFORMAȚIA ZILEI',title:'LICHIDITATE MULTĂ, DIRECȚIE PUȚINĂ',badge:'ABSORBȚIE DE PIAȚĂ',subtitle:'Lichiditatea este prezentă, dar direcția structurală nu este încă suficient confirmată.',warning1:'PIAȚA POATE ABSORBI MIȘCĂRI FĂRĂ SĂ SCHIMBE REGIMUL.',warning2:'DIRECȚIA ARE NEVOIE DE FLUX.',meaning:['Lichiditatea poate susține mișcări scurte.','Fără flux, mișcarea rămâne incompletă.','Tactica: urmărește confirmarea, nu doar volumul.'],footer:'LICHIDITATEA DESCHIDE UȘA, DAR FLUXUL DECIDE DIRECȚIA.'},
      participation_repair:{kicker:'INFORMAȚIA ZILEI',title:'PARTICIPAREA REVINE',badge:'REPARAȚIE ÎN CURS',subtitle:'Mecanismul vede prime semne de reconectare structurală prin participare.',warning1:'REPARAȚIA ÎNCEPE PRIN PARTICIPARE.',warning2:'CONFIRMAREA FINALĂ ARE NEVOIE DE FLUX.',meaning:['Participarea urcă spre o zonă mai sănătoasă.','Structura începe să se reconecteze.','Tactica: observă dacă fluxul confirmă.'],footer:'CÂND PARTICIPAREA REVINE, MECANISMUL ÎNCEPE SĂ RESPIRE.'},
      daily_default:{kicker:'INFORMAȚIA ZILEI',title:'MECANISMUL RĂMÂNE ÎN OBSERVAȚIE',badge:'MONITORIZARE',subtitle:'Nu există încă o schimbare structurală majoră. Contextul rămâne mai important decât zgomotul.',warning1:'FĂRĂ CONFIRMARE STRUCTURALĂ MAJORĂ.',warning2:'CONTEXTUL RĂMÂNE CHEIA.',meaning:['Semnalul curent nu cere grabă.','Prețul trebuie citit împreună cu fluxul și participarea.','Tactica: observare și disciplină.'],footer:'MECANISMUL NU ALEARGĂ DUPĂ PREȚ. ÎL AȘTEAPTĂ SĂ CONFIRME.'}
    };

    var en = {
      structure_confirming:{kicker:'TODAY\'S IMPACT',title:'THE STRUCTURE STARTS TO CONFIRM',badge:'STRUCTURAL CONFIRMATION',subtitle:'The mechanism signal has shifted. Price is no longer the only argument; cohesion behind the move matters.',warning1:'DIRECTION IS NO LONGER JUST NARRATIVE.',warning2:'CONFIRMATION MUST BE READ IN CONTEXT.',meaning:['The mechanism has shifted into a positive regime.','Participation and flow must support the move.','Tactic: confirmation, discipline and controlled risk.'],footer:'WHEN STRUCTURE CONFIRMS, PRICE IS NO LONGER THE ONLY ARGUMENT.'},
      structural_pressure:{kicker:'TODAY\'S IMPACT',title:'ACTIVE STRUCTURAL PRESSURE',badge:'STRUCTURAL RISK',subtitle:'The mechanism sees pressure, not just volatility. This zone requires caution.',warning1:'RISK IS NO LONGER ONLY LOCAL.',warning2:'STRUCTURE REQUIRES CAUTION.',meaning:['The signal indicates downside pressure.','Liquidity can accelerate moves.','Tactic: reduce exposure and observe.'],footer:'WHEN STRUCTURE PRESSES, REACTION MUST BE SLOWER THAN EMOTION.'},
      testing_limit:{kicker:'TODAY\'S IMPACT',title:'THE MARKET IS TESTING THE LIMIT',badge:'NEAR PRECEDENT',subtitle:'The current range is approaching the major precedent. Time becomes as important as price.',warning1:'THE 2026 PRECEDENT BECOMES THE TACTICAL REFERENCE.',warning2:'EVERY EXTRA DAY MATTERS.',meaning:['The current duration is approaching the recent record.','Average models are no longer enough.','Tactic: patience, context and confirmation.'],footer:'THE MARKET IS TESTING THE TIME LIMIT OF THE MECHANISM.'},
      tactic_changed:{kicker:'TODAY\'S IMPACT',title:'THE TACTIC HAS CHANGED',badge:'EXTENDED RANGE',subtitle:'The market is staying above the historical average inside the mechanism. The main signal remains caution.',warning1:'THE HISTORICAL AVERAGE HAS BEEN EXCEEDED AS A TACTICAL THRESHOLD.',warning2:'THE 2026 PRECEDENT REMAINS THE MAJOR REFERENCE.',meaning:['The structure has extended beyond the historical average.','Old models can create misleading signals.','New tactic: patience + adaptation + discipline.'],footer:'THE MARKET HAS LEARNED TO STAY LONGER INSIDE THE MECHANISM.'},
      price_up_no_confirm:{kicker:'TODAY\'S IMPACT',title:'PRICE ROSE, STRUCTURE DID NOT CONFIRM',badge:'MOVE WITHOUT CONFIRMATION',subtitle:'The price move exists, but the mechanism does not yet see a full structural repair.',warning1:'PRICE ALONE IS NOT CONFIRMATION.',warning2:'THE SIGNAL REMAINS WAIT.',meaning:['Price can rise without structural repair.','Flow and participation must confirm.','Tactic: do not confuse impulse with regime change.'],footer:'WHEN PRICE RUNS AHEAD OF STRUCTURE, THE MECHANISM WAITS.'},
      liquidity_without_direction:{kicker:'TODAY\'S IMPACT',title:'HIGH LIQUIDITY, LOW DIRECTION',badge:'MARKET ABSORPTION',subtitle:'Liquidity is present, but structural direction is not yet sufficiently confirmed.',warning1:'THE MARKET CAN ABSORB MOVES WITHOUT CHANGING REGIME.',warning2:'DIRECTION NEEDS FLOW.',meaning:['Liquidity can support short moves.','Without flow, the move remains incomplete.','Tactic: watch confirmation, not only volume.'],footer:'LIQUIDITY OPENS THE DOOR, BUT FLOW DECIDES DIRECTION.'},
      participation_repair:{kicker:'TODAY\'S IMPACT',title:'PARTICIPATION IS RETURNING',badge:'REPAIR IN PROGRESS',subtitle:'The mechanism sees early signs of structural reconnection through participation.',warning1:'REPAIR STARTS THROUGH PARTICIPATION.',warning2:'FINAL CONFIRMATION NEEDS FLOW.',meaning:['Participation is moving toward a healthier zone.','The structure begins to reconnect.','Tactic: watch whether flow confirms.'],footer:'WHEN PARTICIPATION RETURNS, THE MECHANISM STARTS TO BREATHE.'},
      daily_default:{kicker:'TODAY\'S IMPACT',title:'THE MECHANISM REMAINS UNDER OBSERVATION',badge:'MONITORING',subtitle:'There is no major structural shift yet. Context remains more important than noise.',warning1:'NO MAJOR STRUCTURAL CONFIRMATION YET.',warning2:'CONTEXT REMAINS THE KEY.',meaning:['The current signal does not require urgency.','Price must be read together with flow and participation.','Tactic: observation and discipline.'],footer:'THE MECHANISM DOES NOT CHASE PRICE. IT WAITS FOR CONFIRMATION.'}
    };

    var pack = (lang() === 'en' ? en : ro)[id] || (lang() === 'en' ? en.daily_default : ro.daily_default);
    Object.keys(pack).forEach(function(k){ data[k] = pack[k]; });
    data.metaLine = pick('Ziua '+data.day+' · media istorică ~'+data.avg+' zile · precedent 2026: '+data.precedent+' zile · semnal: '+data.signal,
                         'Day '+data.day+' · historical average ~'+data.avg+' days · 2026 precedent: '+data.precedent+' days · signal: '+data.signal);
    data.cta = pick('Vezi explicația vizuală →','View visual explanation →');
    data.dayLabel = pick('Ziua','Day');
    data.contextTitle = pick('CONTEXT VS. ISTORIC','CONTEXT VS. HISTORY');
    data.meaningTitle = pick('CE ÎNSEAMNĂ?','WHAT DOES IT MEAN?');
    data.backText = pick('← Înapoi la BTC Monitor','← Back to BTC Monitor');
    return data;
  }

  function fetchJson(url){
    return fetch(url + (url.indexOf('?') === -1 ? '?' : '&') + 't=' + Date.now(), {cache:'no-store'}).then(function(r){ return r.json(); }).catch(function(){ return null; });
  }

  function load(basePath){
    basePath = basePath || './';
    return Promise.all([
      fetchJson(basePath + 'risk_window.json'),
      fetchJson(basePath + 'coeziv_state.json')
    ]).then(function(res){ return scenario(res[0], res[1]); });
  }

  window.CohesivXImpactEngine = { scenario:scenario, load:load, lang:lang };
})();
