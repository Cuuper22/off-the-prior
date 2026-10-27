/* selfval: the training-free fix. Error after the switch rule divided by Timestral's error, on fresh samples. */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  F.define("selfval", {
    title: "A fix with no training",
    label: "Table of error with the context-decides rule divided by Timestral's error, per family and model size",
    summary: "Below 1 means the rule helped. The rule was frozen before these fresh samples were made.",
    build: function (ctx) {
      root.OTPData.get("selfval").then(function (d) {
        var rows = root.OTPRows(d);
        var fams = ["chaos", "regime", "tails", "in_prior"];
        var names = { chaos: "chaos", regime: "regime switches", tails: "heavy tails", in_prior: "home turf (in-prior)" };
        C.table(ctx.el, ["family", "5M", "28M", "123M", "209M"], fams.map(function (f) {
          return [names[f]].concat(C.SIZES.map(function (s) { var r = rows.filter(function (x) { return x.family === f && x.size === s; })[0]; return r ? r.mase_fix_over_timestral : null; }));
        }));
      });
      return root.OTPStatic(ctx);
    },
  });
})(window);
