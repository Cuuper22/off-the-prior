/* bench: baseline scores on the two released benchmarks. */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  var ORDER = ["oracle", "timestral-209M", "timestral-123M", "timestral-28M", "timestral-5M", "ets", "arima", "seasonal_naive"];
  var NICE = { oracle: "referee (oracle)", ets: "ETS", arima: "ARIMA", seasonal_naive: "repeat last season" };
  F.define("bench", {
    title: "The benchmarks",
    label: "Tables of baseline scores on the diagnostic and leaderboard benchmark sets",
    summary: "Error ÷ the referee's error (1 = perfect), error ÷ 'repeat last season' (the GIFT-Eval convention), and how often the truth lands inside the 80% band.",
    build: function (ctx) {
      var bar = C.h("div", { class: "otp-controls" }), out = C.h("div");
      ctx.el.appendChild(bar); ctx.el.appendChild(out);
      root.OTPData.get("bench").then(function (d) {
        var rows = root.OTPRows(d);
        function show(set) {
          out.textContent = "";
          var rr = ORDER.map(function (m) { return rows.filter(function (r) { return r.set === set && r.model === m; })[0]; }).filter(Boolean);
          C.table(out, ["model", "series", "error ÷ referee", "error ÷ repeat-season", "inside 80% band"],
                  rr.map(function (r) { return [NICE[r.model] || r.model, r.series, r.mase_rel_oracle, r.mase_rel_snaive, r.cov80]; }));
          if (set === "diag") {
            var ft = d.diag_by_failure_type_mase_rel_oracle["timestral-209M"] || {};
            out.appendChild(C.h("p", { class: "otp-note", text: "Timestral-209M by failure type (error ÷ referee):" }));
            C.table(out, ["failure type", "209M"], Object.keys(ft).sort().map(function (k) { return [k.replace(/_/g, " "), ft[k]]; }));
          }
        }
        C.segmented(bar, [["diag", "Diagnostic set"], ["lb", "Leaderboard set"]], "diag", show, "benchmark");
        show("diag");
      });
      return root.OTPStatic(ctx);
    },
  });
})(window);
