/* realdata: real series that stand in for the failure families, compared with the synthetic families
   (ratio of Timestral's error to the best old-school method; no referee exists for real data). */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  var NAMES = { delay_feedback: "delay feedback (sunspots, lynx, El Niño...)", coupled_oscillators: "coupled oscillators (grid frequency, activity...)",
                heavy_tails: "heavy tails (rivers, quakes, crypto...)", regime_switches: "regime switches (macro breaks, traffic, COVID...)" };
  F.define("realdata", {
    title: "Real look-alikes vs the synthetic families",
    label: "Table comparing Timestral's error relative to the best classical model on real analogue series and on the synthetic families",
    summary: "Below 1 means Timestral beats the best old-school method. There is no referee for real data, so this compares against old-school methods only.",
    build: function (ctx) {
      root.OTPData.get("realdata").then(function (d) {
        var fams = Object.keys(NAMES);
        var rows = fams.map(function (f) {
          var g = d.analogues.filter(function (r) { return r[0] === f; });
          function v(size, k) { var r = g.filter(function (x) { return x[1] === size; })[0]; return r ? r[k] : null; }
          return [NAMES[f], v("5M", 2), v("5M", 3), v("5M", 4), v("209M", 3), v("209M", 4)];
        });
        C.table(ctx.el, ["family (real examples)", "real windows", "real, 5M", "synthetic, 5M", "real, 209M", "synthetic, 209M"], rows);
      });
      return root.OTPStatic(ctx);
    },
  });
})(window);
