(function(){
  'use strict';
  var KEY='coeziv_btc_lang';

  function en(){
    try{
      return localStorage.getItem(KEY)==='en'||
        document.documentElement.lang==='en'||
        (document.body&&document.body.getAttribute('data-lang')==='en');
    }catch(e){return false;}
  }

  function samePageGuideLink(a){
    var href=(a.getAttribute('href')||'').toLowerCase();
    return href.indexOf('ghid-mecanism.html')!==-1;
  }

  function samePageBackLink(a){
    var href=(a.getAttribute('href')||'').toLowerCase();
    if(href.indexOf('ghid-mecanism.html')!==-1)return false;
    return href.indexOf('mecanism.html')!==-1;
  }

  function apply(){
    var isEnglish=en();

    var guide=document.getElementById('guide-link')||
      Array.prototype.slice.call(document.querySelectorAll('a')).find(samePageGuideLink);
    if(guide){
      guide.textContent=isEnglish?'Mechanism guide':'Ghid mecanism';
      guide.setAttribute('aria-label',isEnglish?'Open mechanism guide':'Deschide ghidul mecanismului');
    }

    var links=document.querySelectorAll('a');
    for(var i=0;i<links.length;i++){
      if(samePageBackLink(links[i])){
        links[i].textContent=isEnglish?'← Back to mechanism':'← Înapoi la mecanism';
        links[i].setAttribute('aria-label',isEnglish?'Back to mechanism':'Înapoi la mecanism');
      }
    }

    document.documentElement.lang=isEnglish?'en':'ro';
  }

  function addStructuralStyle(){
    if(document.getElementById('structural-confirmation-style'))return;
    var style=document.createElement('style');
    style.id='structural-confirmation-style';
    style.textContent='\n'+
      '#structural-confirmation-card{margin:12px auto 12px;padding:12px 12px 13px;border-radius:18px;border:1px solid rgba(56,189,248,.23);background:radial-gradient(circle at 50% -18%,rgba(56,189,248,.10),transparent 55%),linear-gradient(180deg,rgba(15,23,42,.68),rgba(15,23,42,.42));box-shadow:inset 0 1px 0 rgba(255,255,255,.035);text-align:left;}\n'+
      '#structural-confirmation-card .structural-confirmation-title{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--text-soft);margin-bottom:8px;}\n'+
      '#structural-confirmation-card .structural-confirmation-main{padding:10px 11px;border-radius:14px;border:1px solid rgba(56,189,248,.25);background:rgba(56,189,248,.065);font-size:12.5px;line-height:1.55;font-weight:600;color:var(--text-main);}\n'+
      '#structural-confirmation-card .structural-confirmation-base{margin-top:9px;font-size:10.5px;line-height:1.42;color:var(--text-soft);}\n'+
      '#structural-confirmation-card .structural-confirmation-context{margin-top:9px;font-size:12.5px;line-height:1.55;color:var(--text-main);}\n'+
      'body.light-mode #structural-confirmation-card{background:linear-gradient(180deg,rgba(255,255,255,.88),rgba(248,250,252,.72));border-color:rgba(14,165,233,.25);}\n'+
      'body.light-mode #structural-confirmation-card .structural-confirmation-main{background:rgba(14,165,233,.075);border-color:rgba(14,165,233,.26);}\n';
    document.head.appendChild(style);
  }

  function createStructuralCard(){
    var card=document.getElementById('structural-confirmation-card');
    var radar=document.getElementById('coeziv-mini-radar');
    var energy=document.getElementById('prod-cost-line');
    var anchor=radar||energy;

    if(!anchor||!anchor.parentNode)return card||null;

    if(!card){
      card=document.createElement('div');
      card.id='structural-confirmation-card';
      card.innerHTML=''+
        '<div class="structural-confirmation-title">Confirmare structurală</div>'+
        '<div id="structural-confirmation-main" class="structural-confirmation-main">Se încarcă confirmarea structurală...</div>'+
        '<div id="structural-confirmation-base" class="structural-confirmation-base">Bază statistică: se actualizează.</div>'+
        '<div id="structural-confirmation-context" class="structural-confirmation-context">Context structural curent: se actualizează.</div>';
    }

    if(card.parentNode!==anchor.parentNode||card.nextSibling!==anchor){
      anchor.parentNode.insertBefore(card, anchor);
    }
    return card;
  }

  function pct(value){
    var n=Number(value);
    if(!Number.isFinite(n))return null;
    if(Math.abs(n)<=1)n=n*100;
    return '~'+n.toFixed(0)+'%';
  }

  function whole(value){
    var n=Number(value);
    if(!Number.isFinite(n))return null;
    return n.toFixed(0);
  }

  function contextLabel(regime){
    var key=String(regime||'').toLowerCase();
    var labels={
      bear_late:'presiune descendentă aflată în fază matură, cu semne de stabilizare',
      bear_early:'presiune descendentă aflată în fază activă',
      bull_early:'reluare pozitivă timpurie, încă în confirmare',
      bull_late:'impuls pozitiv matur, cu risc de epuizare',
      range:'zonă de echilibru relativ, fără direcție structurală dominantă',
      neutral:'zonă neutră, cu direcție structurală insuficient confirmată'
    };
    return labels[key]||'context coeziv curent în evaluare structurală';
  }

  function renderStructural(state){
    createStructuralCard();
    var main=document.getElementById('structural-confirmation-main');
    var base=document.getElementById('structural-confirmation-base');
    var ctx=document.getElementById('structural-confirmation-context');
    if(!main||!base||!ctx)return;

    var structural=state&&state.structural_confirmation?state.structural_confirmation:null;
    if(!structural){
      main.textContent='Confirmarea structurală va apărea după următorul snapshot coeziv.';
      base.textContent='Bază statistică: așteptăm state.structural_confirmation din coeziv_state.json.';
      ctx.textContent='Context structural curent: indisponibil momentan. Semnalul este un reper de structură, nu o recomandare de tranzacționare.';
      return;
    }

    var h7=structural.horizon_7d||{};
    var h30=structural.horizon_30d||{};
    var p7=pct(h7.directional_hit_rate);
    var p30=pct(h30.directional_hit_rate);
    var events=whole(h30.events!=null?h30.events:h7.events);
    var samples=whole(structural.similar_context_samples||
      (state.model_price_components&&state.model_price_components.similar_context_samples));
    var threshold=structural.threshold_label||'deviații ample față de reperul coeziv';
    var regime=structural.regime||(state.model_price_context&&state.model_price_context.regime);
    var label=structural.context_label||contextLabel(regime);

    main.textContent=(p7&&p30)
      ? 'În contexte istorice similare, mecanismul a confirmat direcția structurală în '+p7+' din cazuri pe 7 zile și '+p30+' din cazuri pe 30 zile.'
      : 'Confirmarea structurală se actualizează după următorul backtest al mecanismului.';
    base.textContent=events
      ? 'Bază statistică: '+events+' evenimente istorice cu '+threshold+'.'
      : 'Bază statistică: evenimentele istorice se actualizează.';
    ctx.textContent='Context structural curent: '+label+'. '+(samples?'Analiza folosește '+samples+' contexte istorice similare.':'Analiza folosește contexte istorice similare.')+' Semnalul este un reper de structură, nu o recomandare de tranzacționare.';
  }

  function fetchStructural(){
    if(!/mecanism\.html/i.test(location.pathname))return;
    addStructuralStyle();
    createStructuralCard();
    fetch('./coeziv_state.json',{cache:'no-store'})
      .then(function(r){return r.json();})
      .then(renderStructural)
      .catch(function(){renderStructural(null);});
    setTimeout(createStructuralCard,400);
    setTimeout(createStructuralCard,1200);
  }

  if(document.readyState==='loading'){
    document.addEventListener('DOMContentLoaded',function(){apply();fetchStructural();});
  }else{
    apply();
    fetchStructural();
  }
  setInterval(apply,700);
})();
