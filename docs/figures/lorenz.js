/* lorenz: the Lorenz gap against forecast reach measured in Lyapunov times. */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  F.define("lorenz", {
    title: "Lorenz: the gap lives in the first few Lyapunov times",
    label: "Line chart of Timestral's error divided by the referee's against forecast reach in Lyapunov times, one line per size",
    summary: "Four sampling speeds and three horizons put on one axis: how many Lyapunov times ahead the forecast reaches. Far enough out, nobody can forecast Lorenz, so the referee and Timestral meet at 1.",
    build: function (ctx) {
      var plot = C.h("div"); ctx.el.appendChild(plot);
      root.OTPData.get("lorenz").then(function (d) {
        C.line(plot, { series: C.SIZES.map(function (s) { return { name: s, color: C.SIZE_COLORS[s], points: d.series[s] }; }),
                       xlog: true, ylog: true, ref1: true, xLabel: "forecast reach (Lyapunov times)", yLabel: "error ÷ referee's error" });
      });
      return root.OTPStatic(ctx);
    },
  });
})(window);
