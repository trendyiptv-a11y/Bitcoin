(function(){
  'use strict';

  var LANG_KEY = 'coeziv_btc_lang';
  var PROD_BASE = 'https://coezivx.vercel.app/btc-swing-strategy/';

  function lang(){
    try{
      return localStorage.getItem(LANG_KEY)==='en' ||
        document.documentElement.lang==='en' ||
        (document.body && document.body.getAttribute('data-lang')==='en') ? 'en' : 'ro';
    }catch(e){ return 'ro'; }
  }

  function pick(ro,en){ return lang()==='en' ? en : ro; }
  function num(v,fallback){ var n=Number(v); return Number.isFinite(n)?n:fallback; }

  function signalLabel(signal){
    var s=String(signal||'flat').toLowerCase();
    if(lang()==='en'){
      if(s==='long') return 'UPWARD PRESSURE';
      if(s==='short') return 'DOWNSIDE RISK';
      return 'WAIT';
    }
    if(s==='long') return 'PRESIUNE DE CREȘTERE';
    if(s==='short') return 'RISC DE SCĂDERE';
   