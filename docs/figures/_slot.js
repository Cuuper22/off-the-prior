/* Shared slot figure: reads data/<name>.json; while "status" is "pending" it says so, otherwise it lists the
   result tables as they come from the result files (the site worker replaces these with proper charts). */
(function (root) {
  "use strict";
  var C = root.OTPChart;
  root.OTPSlot = function (name, render) {
    return function (ctx) {
      root.OTPData.get(name).then(function (d) {
        if (!d || d.status !== "ready") return C.pending(ctx.el, d);
        render(ctx, d);
        var src = d.sources || [], shown = src.slice(0, 3).join(", ");
        ctx.el.appendChild(C.h("p", { class: "otp-note", text: "From: " + shown + (src.length > 3 ? " and " + (src.length - 3) + " more files" : "") }));
      }).catch(function () { C.pending(ctx.el, null); });
      return root.OTPStatic(ctx);
    };
  };
})(window);
