(function(){
  'use strict';

  var KEY = 'coeziv_btc_lang';
  var lastStructuralState = null;
  var lastImpactData = { risk:null, state:null };

  function isEn(){
    try{
      return localStorage.getItem(KEY) === 'en' ||
        document.documentElement.lang === 'en' ||
        (document.body && document.body.getAttribute('data-lang') === 'en');
    }catch(e){ return