(function(){
  'use strict';

  var KEY = 'coeziv_btc_lang';
  var lastStructuralState = null;

  function isEn(){
    try{
      return localStorage.getItem(KEY) === 'en' ||
        document.documentElement.lang === 'en' ||
        (document.body && document.body.getAttribute('data-lang') === 'en');
    }catch(e){ return false; }
  }

  function tx(ro, en){ return isEn() ? en : ro; }

  function samePageGuideLink(a){
    var href = (a.getAttribute('href') || '').toLowerCase();
    return href.indexOf('ghid-mecanism.html') !== -1;
  }

  function samePageBackLink(a){
    var href = (a.getAttribute('href') || '').toLowerCase();
    if (href.indexOf('ghid-mecanism.html') !== -1) return false;
    return href.indexOf('mecanism.html') !== -1;
  }

  function restoreAccidentallyHiddenContent(){
    if (!/mecanism\.html/i.test(location.pathname)) return;
    var selectors = ['.shell', '.card', '.card-secondary', '#daily-cylinder-root', '#coeziv-mini-radar', '#prod-cost-line'];
    selectors.forEach(function(sel){
      Array.prototype.slice.call(document.querySelectorAll(sel)).forEach(function(el){
        if (!el) return;
        if (el.getAttribute('data-cohesivx-hidden-daily-visual') === '1') {
          el.removeAttribute('data-cohesivx-hidden-daily-visual');
          el.removeAttribute('aria-hidden');
        }
        if (el.style && el.style.display === 'none') {
          el.style.display = '';
        }
      });
    });
    var tactical = document.getElementById('cohesivx-tactical-range-card');
    if (tactical && tactical.parentNode) tactical.parentNode.removeChild(tactical);
  }

  function applyGuideText(){
    restoreAccidentallyHiddenContent();
    var english = isEn();
    var guide = document.getElementById('guide-link') ||
      Array.prototype.slice.call(document.querySelectorAll('a')).find(samePageGuideLink);

    if (guide) {
      guide.textContent = english ? 'Mechanism guide' : 'Ghid mecanism';
      guide.setAttribute('aria-label', english ? 'Open mechanism guide' : 'Deschide ghidul mecanismului');
    }

    var links = document.querySelectorAll('a');
    for (var i = 0; i < links.length; i++) {
      if (samePageBackLink(links[i])) {
        links[i].textContent = english ? '← Back to mechanism' : '← Înapoi la mecanism';
        links[i].setAttribute('aria-label', english ? 'Back to mechanism' : 'Înapoi la mecanism');
      }
    }

    document.documentElement.lang = english ? 'en' : 'ro';
    if (lastStructuralState) renderStructural(lastStructuralState);
    reverseSignalLegend();
  }

  function reverseSignalLegend(){
    var grid = document.querySelector('.cxlg-grid');
    if (!grid) return;

    var order = ['risk', 'sell', 'attention', 'wait', 'accumulate', 'buy'];
    order.forEach(function(key){
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
    var obs = new MutationObserver(function(){
      window.requestAnimationFrame(function(){
        restoreAccidentallyHiddenContent();
        reverseSignalLegend();
      });
    });
    obs.observe(root, { childList:true, subtree:true });
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
    var radar = document.getElementById('coeziv-mini-radar');
    var energy = document.getElementById('prod-cost-line');
    var anchor = radar || energy;
    if (!anchor || !anchor.parentNode) return card || null;

    if (!card) {
      card = document.createElement('div');
      card.id = 'structural-confirmation-card';
      card.innerHTML = '<div id="structural-confirmation-title" class="structural-confirmation-title"></div>'+
        '<div id="structural-confirmation-main" class="structural-confirmation-main"></div>'+
        '<div id="structural-confirmation-base" class="structural-confirmation-base"></div>'+
        '<div id="structural-confirmation-context" class="structural-confirmation-context"></div>';
    }

    if (card.parentNode !== anchor.parentNode || card.nextSibling !== anchor) {
      anchor.parentNode.insertBefore(card, anchor);
    }
    return card;
  }

  function pct(value){
    var n = Number(value);
    if (!Number.isFinite(n)) return null;
    if (Math.abs(n) <= 1) n = n * 100;
    return '~' + n.toFixed(0) + '%';
  }

  function whole(value){
    var n = Number(value);
    if (!Number.isFinite(n)) return null;
    return n.toFixed(0);
  }

  function contextLabel(regime){
    var key = String(regime || '').toLowerCase();
    var ro = {
      bear_late:'presiune descendentă aflată în fază matură, cu semne de stabilizare',
      bear_early:'presiune descendentă aflată în fază activă',
      bull_early:'reluare pozitivă timpurie, încă în confirmare',
      bull_late:'impuls pozitiv matur, cu risc de epuizare',
      range:'zonă de echilibru relativ, fără direcție structurală dominantă',
      neutral:'zonă neutră, cu direcție structurală insuficient confirmată'
    };
    var en = {
      bear_late:'mature downside pressure with signs of stabilization',
      bear_early:'active downside pressure',
      bull_early:'early positive recovery, still awaiting confirmation',
      bull_late:'mature positive impulse with exhaustion risk',
      range:'relative equilibrium zone without a dominant structural direction',
      neutral:'neutral zone with insufficient structural direction confirmation'
    };
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

    var h7 = structural.horizon_7d || {};
    var h30 = structural.horizon_30d || {};
    var p7 = pct(h7.directional_hit_rate);
    var p30 = pct(h30.directional_hit_rate);
    var events = whole(h30.events != null ? h30.events : h7.events);
    var samples = whole(structural.similar_context_samples || (state.model_price_components && state.model_price_components.similar_context_samples));
    var regime = structural.regime || (state.model_price_context && state.model_price_context.regime);
    var label = structural.context_label && !isEn() ? structural.context_label : contextLabel(regime);
    var threshold = isEn() ? 'wide deviations from the cohesive reference' : 'deviații ample față de reperul coeziv';

    main.textContent = (p7 && p30)
      ? tx('În contexte istorice similare, mecanismul a confirmat direcția structurală în '+p7+' din cazuri pe 7 zile și '+p30+' din cazuri pe 30 zile.',
           'In similar historical contexts, the mechanism confirmed the structural direction in '+p7+' of cases over 7 days and '+p30+' of cases over 30 days.')
      : tx('Confirmarea structurală se actualizează după următorul backtest al mecanismului.',
           'Structural confirmation will update after the next mechanism backtest.');

    base.textContent = events
      ? tx('Bază statistică: '+events+' evenimente istorice cu '+threshold+'.',
           'Statistical base: '+events+' historical events with '+threshold+'.')
      : tx('Bază statistică: evenimentele istorice se actualizează.',
           'Statistical base: historical events are updating.');

    ctx.textContent = tx(
      'Context structural curent: '+label+'. '+(samples ? 'Analiza folosește '+samples+' contexte istorice similare.' : 'Analiza folosește contexte istorice similare.')+' Semnalul este un reper de structură, nu o recomandare de tranzacționare.',
      'Current structural context: '+label+'. '+(samples ? 'The analysis uses '+samples+' similar contexts.' : 'The analysis uses similar contexts.')+' The signal is a structural reference, not a trading recommendation.'
    );
  }

  function fetchStructural(){
    if (!/mecanism\.html/i.test(location.pathname)) return;
    addStructuralStyle();
    createStructuralCard();
    fetch('./coeziv_state.json', {cache:'no-store'})
      .then(function(r){ return r.json(); })
      .then(renderStructural)
      .catch(function(){ renderStructural(null); });
    setTimeout(createStructuralCard, 400);
    setTimeout(createStructuralCard, 1200);
  }

  function start(){
    restoreAccidentallyHiddenContent();
    applyGuideText();
    fetchStructural();
    addLegendReverseObserver();
    setTimeout(restoreAccidentallyHiddenContent, 200);
    setTimeout(restoreAccidentallyHiddenContent, 800);
    setTimeout(restoreAccidentallyHiddenContent, 1800);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }

  setInterval(applyGuideText, 700);
})();