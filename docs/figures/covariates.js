/* covariates: how much a helper series (covariate) improves the forecast, against what a perfect user of it would gain. */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  var LAGS = [0, 8, 32, 128, 512];
  F.define("covariates", {
    title: "Helper series: used only when the link is immediate",
    label: "Line chart of the gain from a covariate against the lag between the driver and the main series, Timestral solid, referee dashed",
    summary: "The driver's future is handed over as a known-future covariate. Gain = error without it ÷ error with it. Solid: Timestral. Dashed: the referee using the same information.",
    build: function (ctx) {
      var bar = C.h("div", { class: "otp-controls" }), plot = C.h("div");
      ctx.el.appendChild(bar); ctx.el.appendChild(plot);
      root.OTPData.get("covariates").then(function (d) {
        var rows = root.OTPRows(d).filter(function (r) { return r.system === "LaggedDriver"; });
        function show(cond) {
          var series = [];
          ["5M", "28M"].forEach(function (size) {
            var g = rows.filter(function (r) { return r.size === size && r.H === 16 && r.condition === cond; });
            function pts(key) { return LAGS.map(function (L, i) { var r = g.filter(function (x) { return x.knob_value === L; })[0]; var p = [i, r ? r[key] : null]; p.label = "lag " + L; return p; }); }
            series.push({ name: size + " Timestral", color: C.SIZE_COLORS[size], points: pts("covariate_gain") });
            series.push({ name: size + " referee", color: C.SIZE_COLORS[size], dash: "6 4", points: pts("oracle_gain") });
          });
          C.line(plot, { series: series, ylog: true, ref1: true, xTicks: LAGS.map(function (L, i) { return [i, String(L)]; }),
                         xLabel: "lag between driver and main series (steps)", yLabel: "gain from the helper series" });
        }
        C.segmented(bar, [["known_cov", "future of the driver given"], ["past_cov", "only its past given"]], "known_cov", show, "what is given");
        show("known_cov");
      });
      return root.OTPStatic(ctx);
    },
  });
})(window);
