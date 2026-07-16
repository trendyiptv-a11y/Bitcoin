(function(){
  'use strict';

  var KEY = 'coeziv_btc_lang';
  var lastStructuralState = null;
  var impactData = null;
  var topScrollTimersStarted = false;
  var enginePromise = null;

  function isEn(){
    try{
      return localStorage.getItem(KEY) === 'en' ||
        document.documentElement.lang === 'en' ||
        (document.body && document.body.getAttribute('data-lang') === 'en');
    }catch(e){ return false; }
  }

  function isMonitorPage(){ return /mecanism\.html/i.test(location.pathname); }
  function tx(ro, en){ return isEn() ? en : ro; }

  function disableScrollRestoration(){
    if (!isMonitorPage()) return;
    try{ if ('scrollRestoration' in history) history.scrollRestoration = 'manual'; }catch(e){}
  }

  function forceTopOnInitialLoad(){
    if (!isMonitorPage() || location.hash) return;
    var navType = '';
    try{
      var nav = performance.getEntriesByType && performance.getEntriesByType('navigation')[0];
      navType = nav && nav.type ? nav.type : '';
    }catch(e){}
    if (navType === 'back_forward') return;
    if (topScrollTimersStarted) return;
    topScrollTimersStarted = true;
    [0,80,240,700,1300].forEach(function(ms){
      setTimeout(function(){ try{ window.scrollTo(0,0); }catch(e){} }, ms);
    });
  }

  function loadImpactEngine(){
    if (window.CohesivXImpactEngine) return Promise.resolve(window.CohesivXImpactEngine);
    if (enginePromise) return enginePromise;
    enginePromise = new Promise(function(resolve, reject){
      var existing = document.querySelector('script[data-cohesivx-impact-engine]');
      if (existing) {
        existing.addEventListener('load', function(){ resolve(window.CohesivXImpactEngine); });
        existing.addEventListener('error', reject);
        return;
      }
      var s = document.createElement('script');
      s.src = './impact/impact-engine.js?v=1';
      s.defer = true;
      s.setAttribute('data-cohesivx-impact-engine','1');
      s.onload = function(){ resolve(window.CohesivXImpactEngine); };
      s.onerror = reject;
      document.head.appendChild(s);
    });
    return enginePromise;
  }

  function samePageGuideLink(a){
    var href = (a.getAttribute('href') || '').toLowerCase();
    return href.indexOf('ghid-mecanism.html') !== -1;
  }

  function samePageBackLink(a){
    var href = (a.getAttribute('href') || '').toLowerCase();
    if (href.indexOf('ghid-mecanism.html') !== -1) return false;
    return href.indexOf('mecanism.html') !== -1;
  }

  function applyGuideText(){
    var english = isEn();
    var guide = document.getElementById('guide-link') || Array.prototype.slice.call(document.querySelectorAll('a')).find(samePageGuideLink);
    if (guide) {
      guide.textContent = english ? 'Mechanism guide' : 'Ghid mecanism';
      guide.setAttribute('aria-label', english ? 'Open mechanism guide' : 'Deschide ghidul mecanismului');
    }
    Array.prototype.slice.call(document.querySelectorAll('a')).forEach(function(a){
      if (samePageBackLink(a)) {
        a.textContent = english ? '← Back to mechanism' : '← Înapoi la mecanism';
        a.setAttribute('aria-label', english ? 'Back to mechanism' : 'Înapoi la mecanism');
      }
    });
    document.documentElement.lang = english ? 'en' : 'ro';
    if (lastStructuralState) renderStructural(lastStructuralState);
    if (impactData) renderImpactBanner(impactData);
    reverseSignalLegend();
  }

  function reverseSignalLegend(){
    var grid = document.querySelector('.cxlg-grid');
    if (!grid) return;
    ['risk','sell','attention','wait','accumulate','buy'].forEach(function(key){
      var item = grid.querySelector('.cxlg-item.' + key);
      if (item) grid.appendChild(item);
    });
  }

  function addLegendReverseObserver(){
    reverseSignalLegend();
    setTimeout(reverseSignalLegend, 250);
    setTimeout(reverseSignalLegend, 800);
    setTimeout(reverseSignalLegend, 1500);
    setInterval(reverseSignalLegend, 900);
    if (!window.MutationObserver) return;
    var root = document.getElementById('daily-cylinder-root') || document.body;
    if (!root) return;
    var obs = new MutationObserver(function(){ window.requestAnimationFrame(reverseSignalLegend); });
    obs.observe(root, { childList:true, subtree:true });
  }

  function addImpactStyle(){
    if (document.getElementById('cohesivx-impact-banner-style')) return;
    var style = document.createElement('style');
    style.id = 'cohesivx-impact-banner-style';
    style.textContent = ''+
      '#cohesivx-impact-banner{display:block;text-decoration:none;margin:0 0 12px;padding:13px 13px 14px;border-radius:20px;border:1px solid rgba(56,189,248,.42);background:radial-gradient(circle at 0 0,rgba(56,189,248,.18),transparent 40%),radial-gradient(circle at 100% 0,rgba(249,115,22,.18),transparent 42%),linear-gradient(180deg,rgba(3,18,34,.96),rgba(2,6,23,.92));box-shadow:0 14px 36px rgba(0,0,0,.34),inset 0 1px 0 rgba(255,255,255,.045);color:var(--text-main);position:relative;overflow:hidden;contain:layout paint style;}'+
      '#cohesivx-impact-banner::before{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(56,189,248,.08),transparent);opacity:.55;pointer-events:none;}'+
      '#cohesivx-impact-banner .impact-inner{position:relative;z-index:1;display:flex;align-items:center;justify-content:space-between;gap:12px;}'+
      '#cohesivx-impact-banner .impact-kicker{font-size:10px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#67e8f9;margin-bottom:4px;}'+
      '#cohesivx-impact-banner .impact-title{font-size:18px;line-height:1.05;font-weight:950;text-transform:uppercase;color:#f8fafc;}'+
      '#cohesivx-impact-banner .impact-sub{margin-top:6px;font-size:11px;line-height:1.35;color:#cbd5e1;}'+
      '#cohesivx-impact-banner .impact-sub b{color:#fbbf24;}'+
      '#cohesivx-impact-banner .impact-day{min-width:78px;border-radius:16px;border:1px solid rgba(56,189,248,.38);background:rgba(15,23,42,.58);padding:8px 9px;text-align:center;}'+
      '#cohesivx-impact-banner .impact-day span{display:block;color:#cbd5e1;font-size:10px;font-weight:800;text-transform:uppercase;}'+
      '#cohesivx-impact-banner .impact-day strong{display:block;color:#facc15;font-size:34px;line-height:.95;text-shadow:0 0 16px rgba(250,204,21,.34);}'+
      '#cohesivx-impact-banner .impact-cta{margin-top:7px;display:inline-flex;align-items:center;gap:5px;color:#7dd3fc;font-size:11px;font-weight:850;text-transform:uppercase;letter-spacing:.06em;}'+
      'body.light-mode #cohesivx-impact-banner{background:radial-gradient(circle at 0 0,rgba(14,165,233,.14),transparent 42%),linear-gradient(180deg,rgba(255,255,255,.98),rgba(248,250,252,.88));border-color:rgba(14,165,233,.32);box-shadow:0 12px 30px rgba(15,23,42,.12);}'+
      'body.light-mode #cohesivx-impact-banner .impact-title{color:#0f172a;}body.light-mode #cohesivx-impact-banner .impact-sub{color:#475569;}'+
      '@media(max-width:430px){#cohesivx-impact-banner{padding:12px;}#cohesivx-impact-banner .impact-inner{gap:9px;}#cohesivx-impact-banner .impact-title{font-size:16px;}#cohesivx-impact-banner .impact-day{min-width:66px;}#cohesivx-impact-banner .impact-day strong{font-size:29px;}}';
    document.head.appendChild(style);
  }

  function createImpactBanner(){
    if (!isMonitorPage()) return null;
    addImpactStyle();
    var banner = document.getElementById('cohesivx-impact-banner');
    var anchor = document.querySelector('.top-controls') || document.querySelector('.title-bar');
    if (!anchor || !anchor.parentNode) return banner || null;
    if (!banner) {
      banner = document.createElement('a');
      banner.id = 'cohesivx-impact-banner';
      banner.href = './impact/tactica-s-a-schimbat.html';
    }
    if (banner.parentNode !== anchor.parentNode || banner.previousSibling !== anchor) anchor.parentNode.insertBefore(banner, anchor.nextSibling);
    return banner;
  }

  function renderImpactBanner(data){
    var banner = createImpactBanner();
    if (!banner || !data) return;
    banner.setAttribute('aria-label', data.title || tx('Deschide informația zilei','Open today\'s impact note'));
    banner.innerHTML = '<div class="impact-inner">'+
      '<div class="impact-copy">'+
        '<div class="impact-kicker">'+data.kicker+'</div>'+
        '<div class="impact-title">'+data.title+'</div>'+
        '<div class="impact-sub">'+data.metaLine.replace('semnal: '+data.signal, 'semnal: <b>'+data.signal+'</b>').replace('signal: '+data.signal, 'signal: <b>'+data.signal+'</b>')+'</div>'+
        '<div class="impact-cta">'+data.cta+'</div>'+
      '</div>'+
      '<div class="impact-day"><span>'+data.dayLabel+'</span><strong>'+data.day+'</strong></div>'+
    '</div>';
  }

  function fetchImpact(){
    if (!isMonitorPage()) return;
    createImpactBanner();
    loadImpactEngine().then(function(engine){ return engine.load('./'); }).then(function(data){
      impactData = data;
      renderImpactBanner(data);
      forceTopOnInitialLoad();
    }).catch(function(){
      renderImpactBanner({kicker:tx('INFORMAȚIA ZILEI','TODAY\'S IMPACT'),title:tx('MECANISMUL RĂMÂNE ÎN OBSERVAȚIE','THE MECHANISM REMAINS UNDER OBSERVATION'),metaLine:tx('Datele se actualizează.','Data is updating.'),cta:tx('Vezi explicația vizuală →','View visual explanation →'),dayLabel:tx('Ziua','Day'),day:'–',signal:''});
    });
    setTimeout(createImpactBanner, 400);
  }

  function addStructuralStyle(){
    if (document.getElementById('structural-confirmation-style')) return;
    var style = document.createElement('style');
    style.id = 'structural-confirmation-style';
    style.textContent = ''+
      '#structural-confirmation-card{margin:12px auto 12px;padding:12px 12px 13px;border-radius:18px;border:1px solid rgba(56,189,248,.23);background:radial-gradient(circle at 50% -18%,rgba(56,189,248,.10),transparent 55%),linear-gradient(180deg,rgba(15,23,42,.68),rgba(15,23,42,.42));box-shadow:inset 0 1px 0 rgba(255,255,255,.035);text-align:left;}'+
      '#structural-confirmation-card .structural-confirmation-title{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--text-soft);margin-bottom:8px;}'+
      '#structural-confirmation-card .structural-confirmation-main{padding:10px 11px;border-radius:14px;border:1px solid rgba(56,189,248,.25);background:rgba(56,189,248,.065);font-size:12.5px;line-height:1.55;font-weight:600;color:var(--text-main);}'+
      '#structural-confirmation-card .structural-confirmation-base{margin-top:9px;font-size:10.5px;line-height:1.42;color:var(--text-soft);}'+
      '#structural-confirmation-card .structural-confirmation-context{margin-top:9px;font-size:12.5px;line-height:1.55;color:var(--text-main);}'+
      'body.light-mode #structural-confirmation-card{background:linear-gradient(180deg,rgba(255,255,255,.88),rgba(248,250,252,.72));border-color:rgba(14,165,233,.25);}'+
      'body.light-mode #structural-confirmation-card .structural-confirmation-main{background:rgba(14,165,233,.075);border-color:rgba(14,165,233,.26);}';
    document.head.appendChild(style);
  }

  function createStructuralCard(){
    var card = document.getElementById('structural-confirmation-card');
    var anchor = document.getElementById('coeziv-mini-radar') || document.getElementById('prod-cost-line');
    if (!anchor || !anchor.parentNode) return card || null;
    if (!card) {
      card = document.createElement('div');
      card.id = 'structural-confirmation-card';
      card.innerHTML = '<div id="structural-confirmation-title" class="structural-confirmation-title"></div><div id="structural-confirmation-main" class="structural-confirmation-main"></div><div id="structural-confirmation-base" class="structural-confirmation-base"></div><div id="structural-confirmation-context" class="structural-confirmation-context"></div>';
    }
    if (card.parentNode !== anchor.parentNode || card.nextSibling !== anchor) anchor.parentNode.insertBefore(card, anchor);
    return card;
  }

  function pct(value){ var n = Number(value); if (!Number.isFinite(n)) return null; if (Math.abs(n) <= 1) n *= 100; return '~' + n.toFixed(0) + '%'; }
  function whole(value){ var n = Number(value); return Number.isFinite(n) ? n.toFixed(0) : null; }
  function contextLabel(regime){
    var key = String(regime || '').toLowerCase();
    var ro = {bear_late:'presiune descendentă aflată în fază matură, cu semne de stabilizare',bear_early:'presiune descendentă aflată în fază activă',bull_early:'reluare pozitivă timpurie, încă în confirmare',bull_late:'impuls pozitiv matur, cu risc de epuizare',range:'zonă de echilibru relativ, fără direcție structurală dominantă',neutral:'zonă neutră, cu direcție structurală insuficient confirmată'};
    var en = {bear_late:'mature downside pressure with signs of stabilization',bear_early:'active downside pressure',bull_early:'early positive recovery, still awaiting confirmation',bull_late:'mature positive impulse with exhaustion risk',range:'relative equilibrium zone without a dominant structural direction',neutral:'neutral zone with insufficient structural direction confirmation'};
    return (isEn() ? en : ro)[key] || tx('context coeziv curent în evaluare structurală','current cohesive context under structural evaluation');
  }

  function renderStructural(state){
    lastStructuralState = state || lastStructuralState;
    createStructuralCard();
    var title = document.getElementById('structural-confirmation-title');
    var main = document.getElementById('structural-confirmation-main');
    var base = document.getElementById('structural-confirmation-base');
    var ctx = document.getElementById('structural-confirmation-context');
    if (!main || !base || !ctx) return;
    if (title) title.textContent = tx('Confirmare structurală','Structural confirmation');
    var structural = state && state.structural_confirmation ? state.structural_confirmation : null;
    if (!structural) {
      main.textContent = tx('Confirmarea structurală va apărea după următorul snapshot coeziv.','Structural confirmation will appear after the next cohesive snapshot.');
      base.textContent = tx('Bază statistică: așteptăm state.structural_confirmation din coeziv_state.json.','Statistical base: waiting for state.structural_confirmation from coeziv_state.json.');
      ctx.textContent = tx('Context structural curent: indisponibil momentan. Semnalul este un reper de structură, nu o recomandare de tranzacționare.','Current structural context: temporarily unavailable. The signal is a structural reference, not a trading recommendation.');
      return;
    }
    var h7 = structural.horizon_7d || {}, h30 = structural.horizon_30d || {};
    var p7 = pct(h7.directional_hit_rate), p30 = pct(h30.directional_hit_rate);
    var events = whole(h30.events != null ? h30.events : h7.events);
    var samples = whole(structural.similar_context_samples || (state.model_price_components && state.model_price_components.similar_context_samples));
    var regime = structural.regime || (state.model_price_context && state.model_price_context.regime);
    var label = structural.context_label && !isEn() ? structural.context_label : contextLabel(regime);
    var threshold = isEn() ? 'wide deviations from the cohesive reference' : 'deviații ample față de reperul coeziv';
    main.textContent = (p7 && p30) ? tx('În contexte istorice similare, mecanismul a confirmat direcția structurală în '+p7+' din cazuri pe 7 zile și '+p30+' din cazuri pe 30 zile.','In similar historical contexts, the mechanism confirmed the structural direction in '+p7+' of cases over 7 days and '+p30+' of cases over 30 days.') : tx('Confirmarea structurală se actualizează după următorul backtest al mecanismului.','Structural confirmation will update after the next mechanism backtest.');
    base.textContent = events ? tx('Bază statistică: '+events+' evenimente istorice cu '+threshold+'.','Statistical base: '+events+' historical events with '+threshold+'.') : tx('Bază statistică: evenimentele istorice se actualizează.','Statistical base: historical events are updating.');
    ctx.textContent = tx('Context structural curent: '+label+'. '+(samples ? 'Analiza folosește '+samples+' contexte istorice similare.' : 'Analiza folosește contexte istorice similare.')+' Semnalul este un reper de structură, nu o recomandare de tranzacționare.','Current structural context: '+label+'. '+(samples ? 'The analysis uses '+samples+' similar historical contexts.' : 'The analysis uses similar historical contexts.')+' The signal is a structural reference, not a trading recommendation.');
  }

  function fetchStructural(){
    if (!isMonitorPage()) return;
    addStructuralStyle();
    createStructuralCard();
    fetch('./coeziv_state.json', {cache:'no-store'}).then(function(r){ return r.json(); }).then(renderStructural).catch(function(){ renderStructural(null); });
    setTimeout(createStructuralCard, 400);
    setTimeout(createStructuralCard, 1200);
  }

  function start(){
    disableScrollRestoration();
    applyGuideText();
    fetchImpact();
    fetchStructural();
    addLegendReverseObserver();
    forceTopOnInitialLoad();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
  setInterval(applyGuideText, 700);
})();
