// Mounts the story's animated figures and, when present, the word explainers.
(function () {
  function start() {
    if (window.OTPFigures) window.OTPFigures.mountAll();
    if (window.OTPExplain) window.OTPExplain.init(document);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
