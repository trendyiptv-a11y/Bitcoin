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

  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',apply);else apply();
  setInterval(apply,700);
})();
