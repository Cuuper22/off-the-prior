/* noise: adding noise to the generator hides the gap (the referee loses its edge too). */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  var NAMES = { mackey_glass_tau17: "Mackey-Glass", rossler_c5: "Rössler", "rossler_c5.7": "Rössler", lorenz63_rho28: "Lorenz" };
  F.define("noise", {
    title: "Noise hides the gap",
    label: "Line chart of Timestral's error divided by the referee's against the noise level of the generator",
    summary: "Pick a system. Left is nearly noise-free, right is noisy. With more noise even the referee can't see far, so the ratio falls toward 1.",
    build: function (ctx) {
      var bar = C.h("div", { class: "otp-controls" }), plot = C.h("div");
      ctx.el.appendChild(bar); ctx.el.appendChild(plot);
      root.OTPData.get("noise").then(function (d) {
        var keys = Object.keys(d.systems);
        function show(k) {
          C.line(plot, { series: C.SIZES.map(function (s) { return { name: s, color: C.SIZE_COLORS[s], points: d.systems[k][s] || [] }; }),
                         xlog: true, ylog: true, ref1: true, xLabel: "noise (% of the signal's scale)", yLabel: "error ÷ referee's error" });
        }
        C.segmented(bar, keys.map(function (k) { return [k, NAMES[k] || k]; }), keys[0], show, "system");
        show(keys[0]);
      });
      return root.OTPStatic(ctx);
    },
  });
})(window);
