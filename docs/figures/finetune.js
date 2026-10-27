/* finetune: what cheap fine-tuning buys and costs (slot; fills from results/finetune/<local|colab>/summary.csv). */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  F.define("finetune", {
    title: "Teaching it the hard cases",
    label: "Fine-tuning gain and cost per method and protocol",
    summary: "Full fine-tuning, LoRA and data mixing, tested on families and knob ranges the training never saw.",
    build: root.OTPSlot("finetune", function (ctx, d) {
      Object.keys(d.runs).forEach(function (k) {
        ctx.el.appendChild(C.h("p", { class: "otp-note", text: k }));
        C.table(ctx.el, d.runs[k].columns, d.runs[k].rows);
      });
    }),
  });
})(window);
