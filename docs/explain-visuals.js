/* Mini-diagrams for the explainer pop-ups. This site has none yet; explain.js checks has(id) and shows text only.
   (Same interface as gpu_stack's explain-visuals.js, so visuals can be added later without touching explain.js.) */
(function () {
  "use strict";
  var defs = {};
  window.OTPExplainVisuals = {
    ids: [],
    has: function (id) { return Object.prototype.hasOwnProperty.call(defs, id); },
    svg: function () { return ""; },
    mount: function () {},
    unmount: function () {},
    css: function () { return ""; },
  };
})();
