/* readouts: after a level jump, a linear readout of the forecast token predicts the new level far better than the forecast. */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  F.define("readouts", {
    title: "It knows, but doesn't use it",
    label: "Line chart of R² against steps since a level jump, for a linear readout of the model's internals (dashed) and for the forecast itself (solid)",
    summary: "R² of 1 means a perfect prediction of where the level went. Dashed: a straight-line readout of what the model computes inside. Solid: what the model actually forecasts.",
    build: function (ctx) {
      var plot = C.h("div"); ctx.el.appendChild(plot);
      root.OTPData.get("readouts").then(function (d) {
        var series = [];
        ["5M", "209M"].filter(function (s) { return d.mean_jump[s]; }).forEach(function (size) {
          var g = d.mean_jump[size], col = C.SIZE_COLORS[size] || "#3987e5";
          series.push({ name: size + " inside (readout)", color: col, dash: "6 4", points: g.steps.map(function (x, i) { return [x, g.readout_r2[i]]; }) });
          series.push({ name: size + " forecast", color: col, points: g.steps.map(function (x, i) { return [x, g.forecast_r2[i]]; }) });
        });
        C.line(plot, { series: series, xlog: true, xLabel: "steps since the level jumped", yLabel: "R²", yMin: 0, yMax: 1.02 });
      });
      return root.OTPStatic(ctx);
    },
  });
})(window);
