/* regime: error relative to the referee against the number of steps since a regime switch. */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  var KINDS = [
    ["regime/mean_jump/context/delta4/", "Level jumps (4 sd)"],
    ["regime/frequency_jump/context/ratio2/", "Season doubles its period"],
    ["regime/variance_jump/context/factor0.25/", "Noise drops to a quarter"],
  ];
  F.define("regime", {
    title: "Fresh switches: the first steps hurt",
    label: "Line chart of Timestral's error divided by the referee's against steps since a switch, per model size",
    summary: "Pick a kind of switch. The x-axis is how long ago the switch happened, on a log scale; history length 8,192 steps, horizon 128.",
    build: function (ctx) {
      var bar = C.h("div", { class: "otp-controls" }), plot = C.h("div");
      ctx.el.appendChild(bar); ctx.el.appendChild(plot);
      root.OTPMap.rows().then(function (rows) {
        var reg = rows.filter(function (r) { return r.family === "regime" && r.T === 8192 && r.H === 128; });
        function show(prefix) {
          var rr = reg.filter(function (r) { return r.config_id.indexOf(prefix) === 0; });
          root.OTPMap.draw(plot, rr, { x: function (r) { var m = /\/d(\d+)$/.exec(r.config_id); return m ? +m[1] : null; },
                                       xlog: true, xLabel: "steps since the switch" });
        }
        C.segmented(bar, KINDS, KINDS[0][0], show, "kind of switch");
        show(KINDS[0][0]);
      });
      return root.OTPStatic(ctx);
    },
  });
})(window);
