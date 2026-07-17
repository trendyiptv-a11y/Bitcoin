(function(){
  'use strict';

  var KEY = 'coeziv_btc_lang';
  var lastStructuralState = null;
  var lastTacticalState = null;

  function isEn(){
    try{
      return localStorage.getItem(KEY) === 'en' ||
        document.documentElement.lang === 'en' ||
        (document.body && document.body.getAttribute('data-lang') === 'en');
    }catch(e){ return false; }
  }

  function tx(ro, en){ return isEn() ? en : ro; }

  function samePageGuideLink(a){
    var href = (a