/* Shared drawing for failure-map figures: lines per model size from data/failure_map.json. */
(function (root) {
  "use strict";
  var C = root.OTPChart;
  root.OTPMap = {
    rows: function () {
      return root.OTPData.get("failure_map").then(function (d) { return root.OTPRows(d); });
    },
    draw: function (host, rows, opts) {
      var series = C.SIZES.map(function (size) {
        var r = rows.filter(function (x) { return x.size === size; }).map(function (x) {
          var kv = opts.x(x);
          var y = opts.vsClassical ? x.ratio_classical : x.ratio_oracle;
          var p = [kv, y]; p.label = opts.xLabelOf ? opts.xLabelOf(x) : undefined;
          return { p: p, lo: x.lo95, hi: x.hi95 };
        }).filter(function (o) { return o.p[0] !== null && isFinite(o.p[0]); }).sort(function (a, b) { return a.p[0] - b.p[0]; });
        return { name: size, color: C.SIZE_COLORS[size], points: r.map(function (o) { return o.p; }),
                 band: opts.vsClassical ? [] : r.map(function (o) { return [o.p[0], o.lo, o.hi]; }) };
      });
      return C.line(host, { series: series, ylog: true, xlog: !!opts.xlog, ref1: true, xLabel: opts.xLabel,
                            yLabel: opts.vsClassical ? "error ÷ best old-school method" : "error ÷ referee's error",
                            xTicks: opts.xTicks, label: opts.label });
    },
  };
})(window);
