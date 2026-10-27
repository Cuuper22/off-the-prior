/* Helper for figures that do not animate: hide the Replay button and return a no-op timeline. */
(function (root) {
  "use strict";
  root.OTPStatic = function (ctx) {
    var b = ctx.fig.querySelector(".gsf-replay");
    if (b) b.hidden = true;
    return { duration: 0.01, render: function () {}, autoplay: false };
  };
  root.OTPRows = function (d) {
    var idx = {};
    d.columns.forEach(function (c, i) { idx[c] = i; });
    return d.rows.map(function (r) { var o = {}; for (var k in idx) o[k] = r[idx[k]]; return o; });
  };
})(window);
