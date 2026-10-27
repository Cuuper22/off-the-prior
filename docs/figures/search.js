/* search: three ways to search for failures at the same budget (slot; fills from results/search/comparison.csv). */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  F.define("search", {
    title: "Letting a computer hunt for failures",
    label: "Comparison of three search methods for failure-causing series",
    summary: "CMA-ES, MAP-Elites and gradient search each get the same number of Timestral calls.",
    build: root.OTPSlot("search", function (ctx, d) {
      C.table(ctx.el, d.columns, d.rows);
      if (d.transfer) { ctx.el.appendChild(C.h("p", { class: "otp-note", text: "Do the failures found on 5M carry over to bigger sizes?" })); C.table(ctx.el, d.transfer.columns, d.transfer.rows); }
    }),
  });
})(window);
