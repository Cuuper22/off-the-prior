/* Mini-diagrams for the explainer pop-ups.
 *
 * Every visual is a 260 x 120 SVG whose loop (2 to 5 seconds) shows one
 * mechanism. The markup is plain SVG, the motion is CSS keyframes, and the
 * base style of each element is the finished frame, so removing the
 * animation (prefers-reduced-motion) leaves a readable still picture.
 *
 *   OTPExplainVisuals.ids            list of visual ids
 *   OTPExplainVisuals.has(id)        true if the id exists
 *   OTPExplainVisuals.svg(id, alt)   SVG markup string
 *   OTPExplainVisuals.mount(host, id, { alt })  put it in an element
 *   OTPExplainVisuals.css()          the stylesheet text
 *
 * Works in a browser (window.OTPExplainVisuals) and in Node (require).
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.OTPExplainVisuals = api;
})(this, function () {
  "use strict";

  var W = 260;
  var H = 120;
  var defs = {};
  var order = [];
  var cur = null;
  var counter = 0;

  /* ---------- small helpers ---------- */

  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function r1(n) {
    return Math.round(n * 10) / 10;
  }

  function kf(name, stops) {
    return (
      "@keyframes " +
      name +
      "{" +
      stops
        .map(function (s) {
          return r1(s[0]) + "%{" + s[1] + "}";
        })
        .join("") +
      "}"
    );
  }

  function scope(sel) {
    return sel
      .split(",")
      .map(function (s) {
        return ".v-" + cur.id + " " + s.trim();
      })
      .join(",");
  }

  /* Hold `from`, move to `to` between a% and b% of the loop, hold, reset.
     The element's own style must already equal `to`. */
  function an(sel, dur, a, b, from, to, ease) {
    var n = "xv" + ++counter;
    cur.css.push(
      kf(n, [
        [0, from],
        [a, from],
        [b, to],
        [93, to],
        [100, from],
      ]) +
        scope(sel) +
        "{animation:" +
        n +
        " " +
        dur +
        "s " +
        (ease || "ease-out") +
        " infinite}"
    );
  }

  /* Free-form keyframes. */
  function loop(sel, dur, stops, ease, extra) {
    var n = "xv" + ++counter;
    cur.css.push(
      kf(n, stops) +
        scope(sel) +
        "{animation:" +
        n +
        " " +
        dur +
        "s " +
        (ease || "linear") +
        " infinite;" +
        (extra || "") +
        "}"
    );
  }

  /* Reveal a path (pathLength 1) between a% and b%. */
  function draw(sel, dur, a, b) {
    cur.css.push(scope(sel) + "{stroke-dasharray:1 1}");
    an(sel, dur, a, b, "stroke-dashoffset:1", "stroke-dashoffset:0", "linear");
  }

  /* Move a dot along sample points between a% and b%. */
  function follow(sel, dur, pts, a, b) {
    var stops = [[0, tr(pts[0])]];
    pts.forEach(function (p, i) {
      stops.push([a + ((b - a) * i) / (pts.length - 1), tr(p)]);
    });
    stops.push([93, tr(pts[pts.length - 1])]);
    stops.push([100, tr(pts[0])]);
    loop(sel, dur, stops, "linear");
  }

  function tr(p) {
    return "transform:translate(" + r1(p[0]) + "px," + r1(p[1]) + "px)";
  }

  function T(x, y, s, cls, extra) {
    return (
      '<text x="' +
      x +
      '" y="' +
      y +
      '" class="t ' +
      (cls || "") +
      '"' +
      (extra ? " " + extra : "") +
      ">" +
      s +
      "</text>"
    );
  }

  function rect(x, y, w, h, cls, extra) {
    return (
      '<rect x="' +
      x +
      '" y="' +
      y +
      '" width="' +
      w +
      '" height="' +
      h +
      '" class="' +
      (cls || "") +
      '"' +
      (extra ? " " + extra : "") +
      "/>"
    );
  }

  function line(x1, y1, x2, y2, cls, extra) {
    return (
      '<line x1="' +
      x1 +
      '" y1="' +
      y1 +
      '" x2="' +
      x2 +
      '" y2="' +
      y2 +
      '" class="' +
      (cls || "") +
      '"' +
      (extra ? " " + extra : "") +
      "/>"
    );
  }

  function pathD(pts) {
    return pts
      .map(function (p, i) {
        return (i ? "L" : "M") + r1(p[0]) + " " + r1(p[1]);
      })
      .join("");
  }

  function poly(pts, cls, extra) {
    return (
      '<path d="' +
      pathD(pts) +
      '" pathLength="1" class="' +
      (cls || "") +
      '"' +
      (extra ? " " + extra : "") +
      "/>"
    );
  }

  function at(x, y, inner) {
    return '<g transform="translate(' + x + " " + y + ')">' + inner + "</g>";
  }

  function arc(cx, cy, r, p0, p1) {
    function pt(p) {
      var a = Math.PI * (1 - p);
      return [cx + r * Math.cos(a), cy - r * Math.sin(a)];
    }
    var a = pt(p0);
    var b = pt(p1);
    return (
      "M" +
      r1(a[0]) +
      " " +
      r1(a[1]) +
      "A" +
      r +
      " " +
      r +
      " 0 0 1 " +
      r1(b[0]) +
      " " +
      r1(b[1])
    );
  }

  function arrowHead(x, y, dir, cls) {
    var d = dir;
    return (
      '<path d="M' +
      x +
      " " +
      (y - 5) +
      "L" +
      (x + 8 * d) +
      " " +
      y +
      "L" +
      x +
      " " +
      (y + 5) +
      'Z" class="' +
      cls +
      '"/>'
    );
  }

  /* A line with a small triangle head at (x2, y2). cls carries stroke and fill colours. */
  function arrowTo(x1, y1, x2, y2, cls) {
    var ang = (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI;
    var len = Math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    var ux = (x2 - x1) / len,
      uy = (y2 - y1) / len;
    return (
      '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + r1(x2 - ux * 6) + '" y2="' + r1(y2 - uy * 6) + '" class="' + cls + '"/>' +
      '<path d="M0 0L-8 -4L-8 4Z" transform="translate(' + x2 + " " + y2 + ") rotate(" + r1(ang) + ')" class="' + cls + ' w1"/>'
    );
  }

  function def(id, alt, build) {
    cur = { id: id, css: [] };
    var svg = build();
    defs[id] = { id: id, alt: alt, svg: svg, css: cur.css.join("") };
    order.push(id);
    cur = null;
  }

  /* ---------- shared stylesheet ---------- */

  var BASE_CSS =
    ".gsx-vis{" +
    "--k-ink:var(--pixel-ink,#14151a);" +
    "--k-cy:var(--accent-cyan,#2d8f9a);" +
    "--k-cd:var(--accent-cyan-deep,#1f6f79);" +
    "--k-go:var(--accent-gold,#f2c230);" +
    "--k-gn:var(--accent-green,#5cc14a);" +
    "--k-rd:var(--danger-red,#d0473a);" +
    "--k-bl:var(--title-bar,#2d3d9a);" +
    "--k-pa:var(--pane-bright,#fbfbf8);" +
    "--k-so:var(--chrome-well,#e4e6ea);" +
    "--k-ln:var(--chrome-border,#9a9ea6);" +
    "--k-tx:var(--text-soft,#4a4d55);" +
    "--k-wd:var(--well-dark,#272a33);" +
    "--k-t1:oklch(0.88 0.07 195);--k-t2:oklch(0.92 0.10 88);" +
    "--k-t3:oklch(0.89 0.11 140);--k-t4:oklch(0.87 0.08 28);--k-t5:oklch(0.87 0.06 260);" +
    "}" +
    ".gsx-vis svg{display:block;width:100%;height:auto;overflow:hidden}" +
    ".gsx-vis svg *{transform-box:fill-box}" +
    ".gsx-vis .t{font:500 10px var(--font-mono,ui-monospace,monospace);fill:var(--k-ink);white-space:pre}" +
    ".gsx-vis .tb{font-weight:600}.gsx-vis .tm{text-anchor:middle}.gsx-vis .te{text-anchor:end}" +
    ".gsx-vis .ts{fill:var(--k-tx)}.gsx-vis .tw{fill:var(--k-pa)}.gsx-vis .tr{fill:var(--k-rd)}" +
    ".gsx-vis .z{font-size:15px}.gsx-vis .zz{font-size:26px}" +
    ".gsx-vis .f-ink{fill:var(--k-ink)}.gsx-vis .f-cy{fill:var(--k-cy)}.gsx-vis .f-cd{fill:var(--k-cd)}" +
    ".gsx-vis .f-go{fill:var(--k-go)}.gsx-vis .f-gn{fill:var(--k-gn)}.gsx-vis .f-rd{fill:var(--k-rd)}" +
    ".gsx-vis .f-bl{fill:var(--k-bl)}.gsx-vis .f-pa{fill:var(--k-pa)}.gsx-vis .f-so{fill:var(--k-so)}" +
    ".gsx-vis .f-ln{fill:var(--k-ln)}.gsx-vis .f-wd{fill:var(--k-wd)}.gsx-vis .f-no{fill:none}" +
    ".gsx-vis .f-t1{fill:var(--k-t1)}.gsx-vis .f-t2{fill:var(--k-t2)}.gsx-vis .f-t3{fill:var(--k-t3)}" +
    ".gsx-vis .f-t4{fill:var(--k-t4)}.gsx-vis .f-t5{fill:var(--k-t5)}.gsx-vis .f-ha{fill:url(#gsx-hatch)}" +
    ".gsx-vis .s-ink{stroke:var(--k-ink)}.gsx-vis .s-cy{stroke:var(--k-cy)}.gsx-vis .s-cd{stroke:var(--k-cd)}" +
    ".gsx-vis .s-go{stroke:var(--k-go)}.gsx-vis .s-gn{stroke:var(--k-gn)}.gsx-vis .s-rd{stroke:var(--k-rd)}" +
    ".gsx-vis .s-bl{stroke:var(--k-bl)}.gsx-vis .s-ln{stroke:var(--k-ln)}.gsx-vis .s-so{stroke:var(--k-so)}" +
    ".gsx-vis .w1{stroke-width:1}.gsx-vis .w15{stroke-width:1.5}.gsx-vis .w2{stroke-width:2}.gsx-vis .w3{stroke-width:3}" +
    ".gsx-vis .ln{fill:none;stroke-linejoin:round;stroke-linecap:round}" +
    ".gsx-vis .dr{stroke-dasharray:1 1;stroke-dashoffset:0}" +
    ".gsx-vis .hv{transform-box:view-box}" +
    ".gsx-vis .o0{transform-origin:0 50%}.gsx-vis .ob{transform-origin:50% 100%}.gsx-vis .oc{transform-origin:50% 50%}" +
    ".gsx-vis.is-still *{animation:none!important;transition:none!important}" +
    "@media (prefers-reduced-motion:reduce){.gsx-vis *{animation:none!important;transition:none!important}}";

  /* colours used by this site's charts: Timestral blue, referee orange */
  BASE_CSS +=
    ".gsx-vis{--k-or:#b4531c;--k-bw:oklch(0.9 0.05 260);--k-ow:oklch(0.92 0.06 55)}" +
    ".gsx-vis .s-or{stroke:var(--k-or)}.gsx-vis .f-or{fill:var(--k-or)}.gsx-vis .f-bw{fill:var(--k-bw)}.gsx-vis .f-ow{fill:var(--k-ow)}" +
    ".gsx-vis .dd{stroke-dasharray:3 3}.gsx-vis .dp{stroke-dasharray:.025 .025}.gsx-vis .s-no{stroke:none}";

  /* ---------- sample curves (fixed, so every frame is the same every time) ---------- */
  function pts(x0, x1, n, fn) {
    var out = [];
    for (var i = 0; i < n; i++) {
      var x = x0 + ((x1 - x0) * i) / (n - 1);
      out.push([x, fn(x, i / (n - 1))]);
    }
    return out;
  }
  function jag(seed) {
    var r = rng(seed);
    return function () {
      return r() - 0.5;
    };
  }
  var PI2 = Math.PI * 2;

  /* =========================================================
     timestral: history in, model, forecast fan out
     ========================================================= */
  def(
    "timestral",
    "A line of history goes into a box labelled model, and a forecast fan of a median line with widening bands comes out.",
    function () {
      var h = jag(3);
      var hist = pts(8, 84, 40, function (x, t) {
        return 62 - 8 * Math.sin(t * 9) - t * 6 + h() * 4;
      });
      var out = poly(hist, "ln s-ink w15 dr hs");
      out += rect(94, 36, 52, 40, "f-so s-ink w2");
      out += T(120, 60, "model", "tm tb");
      out += line(86, 56, 94, 56, "s-ink w2") + line(146, 56, 156, 56, "s-ink w2");
      var med = pts(156, 252, 24, function (x, t) {
        return 56 - 7 * Math.sin(t * 6 + 0.6);
      });
      function band(k) {
        return med
          .map(function (p, i) {
            return [p[0], p[1] - (3 + i * 0.9) * k];
          })
          .concat(
            med
              .map(function (p, i) {
                return [p[0], p[1] + (3 + i * 0.9) * k];
              })
              .reverse()
          );
      }
      out += '<g class="fn">' + poly(band(1), "f-bw s-no") + poly(band(0.5), "f-t5 s-no") + poly(med, "ln s-bl w2") + "</g>";
      out += T(40, 92, "history", "tm ts") + T(120, 92, "reads patches", "tm ts") + T(208, 100, "forecast", "tm ts");
      out += T(130, 116, "21 guesses per future step", "tm tb");
      draw(".hs", 4, 4, 34);
      an(".fn", 4, 46, 78, "transform:scaleX(0)", "transform:none", "ease-out");
      cur.css.push(scope(".fn") + "{transform-origin:0 50%}");
      return out;
    }
  );

  /* forecast */
  def(
    "forecast",
    "A solid line for the past, a dotted line for what happened next, and a blue forecast line drawn beside the dotted one.",
    function () {
      var h = jag(5);
      var past = pts(10, 140, 60, function (x, t) {
        return 60 - 14 * Math.sin(t * 10) + h() * 5;
      });
      var fut = pts(140, 250, 40, function (x, t) {
        return 60 - 14 * Math.sin(10 + t * 7.5) + h() * 3;
      });
      var fc = pts(140, 250, 40, function (x, t) {
        return 60 - 13 * Math.sin(10 + t * 7.2);
      });
      var out = line(140, 12, 140, 100, "s-ln w1 dd");
      out += poly(past, "ln s-ink w15 dr pa") + poly(fut, "ln s-ink w1 dp fu") + poly(fc, "ln s-bl w2 dr fo");
      out += T(75, 112, "the past", "tm ts") + T(195, 112, "the future", "tm ts") + T(140, 9, "now", "tm tb");
      draw(".pa", 4, 2, 34);
      an(".fu", 4, 38, 50, "opacity:0", "opacity:1", "linear");
      draw(".fo", 4, 52, 84);
      return out;
    }
  );

  /* quantiles */
  def(
    "quantiles",
    "A forecast funnel: a median line in the middle with nested bands that widen with time.",
    function () {
      var med = pts(24, 236, 30, function (x, t) {
        return 58 - 8 * Math.sin(t * 5);
      });
      function band(k) {
        var up = med.map(function (p, i) {
          return [p[0], p[1] - (2 + i * k)];
        });
        var dn = med
          .map(function (p, i) {
            return [p[0], p[1] + (2 + i * k)];
          })
          .reverse();
        return up.concat(dn);
      }
      var out = poly(band(1.1), "f-bw s-no b1 oc") + poly(band(0.6), "f-t5 s-no b2 oc") + poly(band(0.25), "f-t1 s-no b3 oc") + poly(med, "ln s-bl w2");
      out += line(236, 24, 236, 92, "s-ln w1 dd");
      out += T(256, 28, "high", "te ts") + T(256, 94, "low", "te ts");
      out += T(130, 112, "middle guess, and how unsure it is", "tm tb");
      an(".b1", 4, 8, 40, "transform:scaleY(0.05)", "transform:none", "ease-out");
      an(".b2", 4, 22, 54, "transform:scaleY(0.05)", "transform:none", "ease-out");
      an(".b3", 4, 36, 68, "transform:scaleY(0.05)", "transform:none", "ease-out");
      return out;
    }
  );

  /* prior */
  def(
    "prior",
    "Many ordinary smooth curves inside a dashed region labelled what it saw in training, and one spiky curve outside it.",
    function () {
      var out = rect(8, 12, 150, 88, "f-so s-ln w1 dd");
      var r = rng(11);
      for (var k = 0; k < 6; k++) {
        var ph = r() * 6,
          fr = 4 + r() * 5,
          base = 28 + (k % 3) * 28,
          sl = (r() - 0.5) * 8;
        out += poly(
          pts(14, 152, 40, function (x, t) {
            return base - 8 * Math.sin(ph + t * fr) + t * sl;
          }),
          "ln s-cd w1"
        );
      }
      var sp = jag(9);
      out += poly(
        pts(176, 252, 34, function (x, t) {
          return 56 + sp() * 56;
        }),
        "ln s-or w15 odd"
      );
      out += T(83, 111, "what it saw in training", "tm ts") + T(214, 111, "off the prior", "tm tb tr");
      cur.css.push(scope(".odd") + "{animation:xvodd 2.4s ease-in-out infinite}" + kf("xvodd", [[0, "opacity:.35"], [50, "opacity:1"], [100, "opacity:.35"]]));
      return out;
    }
  );

  /* synthetic */
  def(
    "synthetic",
    "A straight rising line plus a repeating wave plus jagged noise add up to one made-up series.",
    function () {
      var out = "";
      var j = jag(21);
      var tr = pts(8, 56, 12, function (x, t) {
        return 70 - t * 30;
      });
      var se = pts(70, 118, 30, function (x, t) {
        return 50 - 12 * Math.sin(t * 14);
      });
      var no = pts(132, 180, 30, function () {
        return 50 + j() * 24;
      });
      var tot = pts(196, 252, 40, function (x, t) {
        return 60 - t * 20 - 10 * Math.sin(t * 14) + (j() * 8);
      });
      out += poly(tr, "ln s-bl w2 a1") + T(32, 96, "trend", "tm ts") + poly(se, "ln s-cd w2 a2") + T(94, 96, "season", "tm ts") + poly(no, "ln s-ln w2 a3") + T(156, 96, "noise", "tm ts");
      out += T(63, 54, "+", "tm zz") + T(125, 54, "+", "tm zz") + T(187, 54, "=", "tm zz");
      out += poly(tot, "ln s-ink w15 a4") + T(224, 96, "series", "tm tb");
      out += T(130, 114, "made-up, in any amount", "tm ts");
      draw(".a1", 4, 2, 14);
      draw(".a2", 4, 14, 30);
      draw(".a3", 4, 30, 46);
      draw(".a4", 4, 50, 80);
      return out;
    }
  );

  /* context */
  def(
    "context",
    "A series with its left part shaded as the history the model reads, a line marking now, and a dotted unknown part to the right.",
    function () {
      var h = jag(31);
      var all = pts(10, 170, 70, function (x, t) {
        return 62 - 12 * Math.sin(t * 12) - t * 8 + h() * 4;
      });
      var out = rect(10, 14, 160, 78, "f-bw s-no sw");
      out += poly(all, "ln s-ink w15");
      out += line(170, 10, 170, 96, "s-bl w2");
      out += poly(
        pts(170, 250, 20, function (x, t) {
          return 54 - 10 * Math.sin(12 + t * 8);
        }),
        "ln s-ln w1 dp"
      );
      out += T(90, 108, "history the model reads", "tm tb") + T(210, 108, "unknown", "tm ts") + T(170, 9, "now", "tm tb");
      an(".sw", 3.6, 5, 40, "transform:scaleX(0)", "transform:none", "ease-out");
      cur.css.push(scope(".sw") + "{transform-origin:0 50%}");
      return out;
    }
  );

  /* horizon */
  def(
    "horizon",
    "A series that stops at now, with a bracket that grows to the right to show how many steps ahead the forecast reaches.",
    function () {
      var h = jag(41);
      var past = pts(10, 120, 50, function (x, t) {
        return 56 - 10 * Math.sin(t * 9) + h() * 3;
      });
      var out = poly(past, "ln s-ink w15") + line(120, 14, 120, 90, "s-ln w1 dd");
      out += poly(
        pts(120, 250, 30, function (x, t) {
          return 56 - 10 * Math.sin(9 + t * 9);
        }),
        "ln s-ln w1 dp"
      );
      out += '<g class="br">' + line(120, 84, 250, 84, "s-bl w2") + line(120, 78, 120, 90, "s-bl w2") + line(250, 78, 250, 90, "s-bl w2") + "</g>";
      out += T(185, 104, "horizon: steps ahead", "tm tb") + T(120, 9, "now", "tm tb");
      an(".br", 3.6, 8, 50, "transform:scaleX(0)", "transform:none", "ease-out");
      cur.css.push(scope(".br") + "{transform-origin:0 50%}");
      return out;
    }
  );

  /* patch */
  def(
    "patch",
    "A line cut into equal slices, each slice becoming one small token card.",
    function () {
      var h = jag(51);
      var out = "";
      var n = 6;
      var all = pts(10, 250, 120, function (x, t) {
        return 28 - 10 * Math.sin(t * 14) - t * 6 + h() * 3;
      });
      out += poly(all, "ln s-ink w15");
      for (var i = 1; i < n; i++) out += line(10 + (240 * i) / n, 10, 10 + (240 * i) / n, 46, "s-rd w1 dd");
      for (var k = 0; k < n; k++) {
        var x0 = 14 + k * 40;
        var seg = all.slice(Math.round((k * 119) / n), Math.round(((k + 1) * 119) / n) + 1);
        var mn = Math.min.apply(
          null,
          seg.map(function (p) {
            return p[1];
          })
        );
        var mx = Math.max.apply(
          null,
          seg.map(function (p) {
            return p[1];
          })
        );
        var piece = seg.map(function (p, i) {
          return [x0 + 3 + (i * 28) / (seg.length - 1), 66 + ((p[1] - mn) / Math.max(1, mx - mn)) * 26];
        });
        out += '<g class="tc" style="--i:' + k + '">' + rect(x0, 60, 34, 36, "f-t" + ((k % 5) + 1) + " s-ink w1") + poly(piece, "ln s-ink w1") + "</g>";
      }
      out += T(130, 112, "each slice of 32 steps is one patch", "tm tb");
      cur.css.push(
        scope(".tc") + "{animation:xvpc 4s ease-in-out infinite;animation-delay:calc(var(--i)*.12s)}" +
          kf("xvpc", [
            [0, "transform:translateY(-34px);opacity:0"],
            [22, "transform:translateY(-34px);opacity:0"],
            [48, "transform:none;opacity:1"],
            [92, "transform:none;opacity:1"],
            [100, "transform:translateY(-34px);opacity:0"],
          ])
      );
      return out;
    }
  );

  /* oracle */
  def(
    "oracle",
    "A box holding the true equations and the exact hidden state, from which many thin paths run forward with a thick median path through them.",
    function () {
      var out = rect(8, 30, 78, 52, "f-so s-ink w2");
      out += T(47, 50, "true", "tm tb") + T(47, 62, "equations +", "tm") + T(47, 74, "exact state", "tm");
      out += line(86, 56, 112, 56, "s-ink w2") + arrowHead(112, 56, 1, "f-ink");
      var r = rng(61);
      for (var k = 0; k < 14; k++) {
        var s = (r() - 0.5) * 2.2;
        out += poly(
          pts(124, 252, 24, function (x, t) {
            return 56 + s * 34 * t + (r() - 0.5) * 3 - 6 * Math.sin(t * 6);
          }),
          "ln s-or w1 pth"
        );
      }
      out += poly(
        pts(124, 252, 24, function (x, t) {
          return 56 - 6 * Math.sin(t * 6);
        }),
        "ln s-or w3"
      );
      out += T(140, 104, "run it forward, many times", "tm tb") + T(140, 116, "the middle path is the referee", "tm ts");
      cur.css.push(scope(".pth") + "{animation:xvpth 3.6s ease-out infinite}" + kf("xvpth", [[0, "stroke-dashoffset:1"], [60, "stroke-dashoffset:0"], [100, "stroke-dashoffset:0"]]));
      cur.css.push(scope(".pth") + "{stroke-dasharray:1 1}");
      return out;
    }
  );

  /* ratio */
  def(
    "ratio",
    "Two blocks side by side, Timestral's error taller than the referee's, with the first divided by the second.",
    function () {
      var out = rect(40, 28, 44, 62, "f-bw s-bl w2 rb") + rect(150, 56, 44, 34, "f-ow s-or w2 rb2");
      out += T(62, 104, "Timestral", "tm") + T(172, 104, "referee", "tm");
      out += T(62, 22, "error", "tm ts") + T(172, 22, "error", "tm ts");
      out += T(112, 66, "÷", "tm zz");
      out += T(222, 66, "=", "tm zz") + T(222, 82, "ratio", "tm tb");
      out += T(130, 117, "1 means as good as the referee", "tm ts");
      an(".rb", 4, 6, 30, "transform:scaleY(0.2)", "transform:none", "ease-out");
      an(".rb2", 4, 6, 30, "transform:scaleY(0.2)", "transform:none", "ease-out");
      cur.css.push(scope(".rb,.rb2") + "{transform-origin:50% 100%}");
      return out;
    }
  );

  /* mase */
  def(
    "mase",
    "The truth and a forecast with the gap between them shaded, and a flat dashed line showing a repeat-the-last-value guess.",
    function () {
      var truth = pts(14, 246, 50, function (x, t) {
        return 56 - 16 * Math.sin(t * 8);
      });
      var fc = pts(14, 246, 50, function (x, t) {
        return 58 - 13 * Math.sin(t * 8 + 0.35);
      });
      var out = poly(truth.concat(fc.slice().reverse()), "f-t4 s-no ga");
      out += poly(truth, "ln s-ink w2") + poly(fc, "ln s-bl w2");
      out += line(14, 56, 246, 56, "s-ln w1 dd");
      out += T(130, 100, "average gap (shaded)", "tm tb") + T(130, 112, "÷ gap of a simple repeat-the-last guess", "tm ts");
      an(".ga", 3.6, 8, 40, "opacity:0", "opacity:1", "linear");
      return out;
    }
  );

  /* classical */
  def(
    "classical",
    "Three small cards, each with a different simple forecasting rule drawn as a tiny curve: repeat last season, smoothing, and ARIMA.",
    function () {
      var out = "";
      var names = ["repeat season", "smoothing", "ARIMA"];
      for (var k = 0; k < 3; k++) {
        var x0 = 8 + k * 84;
        out += rect(x0, 14, 76, 70, "f-so s-ink w1");
        var f;
        if (k === 0)
          f = function (x, t) {
            return 54 - 14 * Math.sin(t * 12.5);
          };
        else if (k === 1)
          f = function (x, t) {
            return t < 0.5 ? 54 - 14 * Math.sin(t * 12.5) : 46 + (t - 0.5) * 8;
          };
        else
          f = function (x, t) {
            return t < 0.5 ? 54 - 14 * Math.sin(t * 12.5) : 54 - 12 * Math.sin(t * 12.5) * (1.2 - t);
          };
        out += poly(pts(x0 + 4, x0 + 72, 40, f), "ln s-bl w15 cc" + k);
        out += line(x0 + 38, 18, x0 + 38, 80, "s-ln w1 dd");
        out += T(x0 + 38, 98, names[k], "tm tb");
        draw(".cc" + k, 4, 4 + k * 12, 30 + k * 12);
      }
      out += T(130, 114, "the old-school yardsticks", "tm ts");
      return out;
    }
  );

  /* chaos */
  def(
    "chaos",
    "Two curves that start almost on top of each other and then drift far apart.",
    function () {
      var a = pts(12, 248, 90, function (x, t) {
        return 56 - 20 * Math.sin(t * 22) * (0.5 + t * 0.5);
      });
      var b = pts(12, 248, 90, function (x, t) {
        var d = t * t * 2.2;
        return 56 - 20 * Math.sin(t * 22 + d * 1.4) * (0.5 + t * 0.5);
      });
      var out = poly(a, "ln s-bl w2 c1") + poly(b, "ln s-or w2 c2");
      out += line(50, 14, 50, 96, "s-ln w1 dd") + line(210, 14, 210, 96, "s-ln w1 dd");
      out += T(52, 110, "tiny difference", "tm ts") + T(206, 110, "huge difference", "tm tb");
      draw(".c1", 4, 4, 60);
      draw(".c2", 4, 4, 60);
      return out;
    }
  );

  /* lyapunov */
  def(
    "lyapunov",
    "Four blocks of growing height, each about two and a half times the last, showing a tiny gap growing at a steady multiplying rate.",
    function () {
      var out = "";
      var hts = [5, 13, 34, 88];
      for (var k = 0; k < 4; k++) {
        var x = 28 + k * 54;
        out += rect(x, 100 - hts[k], 30, hts[k], "f-go s-ink w1 ob lb" + k);
        an(".lb" + k, 4.4, 6 + k * 16, 20 + k * 16, "transform:scaleY(0)", "transform:none", "ease-out");
        out += T(x + 15, 112, "×e", "tm ts");
      }
      out += line(14, 100, 252, 100, "s-ink w1");
      out += T(120, 14, "gap between two runs", "tm tb") + T(120, 26, "×e every Lyapunov time", "tm ts");
      return out;
    }
  );

  /* mackey-glass */
  def(
    "mackey-glass",
    "A wavy curve with an arrow from an earlier point to a later one, showing that now depends on the value a fixed delay ago.",
    function () {
      var c = pts(10, 250, 120, function (x, t) {
        return 52 - 18 * Math.sin(t * 17) * (0.9 + 0.3 * Math.sin(t * 5));
      });
      var out = poly(c, "ln s-ink w15 mg");
      var x1 = 78,
        x2 = 168;
      var y1 = 52 - 18 * Math.sin(((x1 - 10) / 240) * 17) * (0.9 + 0.3 * Math.sin(((x1 - 10) / 240) * 5));
      var y2 = 52 - 18 * Math.sin(((x2 - 10) / 240) * 17) * (0.9 + 0.3 * Math.sin(((x2 - 10) / 240) * 5));
      out += '<g class="dl">' + line(x1, 88, x2, 88, "s-bl w2") + line(x1, 82, x1, 94, "s-bl w2") + arrowHead(x2 - 8, 88, 1, "f-bl") + line(x1, y1, x1, 88, "s-bl w1 dd") + line(x2, y2, x2, 88, "s-bl w1 dd") + "</g>";
      out += T(123, 82, "τ", "tm tb z") + T(123, 108, "today leans on the value τ ago", "tm tb");
      draw(".mg", 4, 2, 40);
      an(".dl", 4, 44, 66, "opacity:0", "opacity:1", "linear");
      return out;
    }
  );

  /* kuramoto */
  def(
    "kuramoto",
    "Eight dots on a ring that start spread out and then bunch together as the clocks they stand for lock in step.",
    function () {
      var out = "";
      var cx = 130,
        cy = 54,
        R = 40;
      out += '<circle cx="' + cx + '" cy="' + cy + '" r="' + R + '" class="f-no s-ln w1"/>';
      var r = rng(71);
      for (var k = 0; k < 8; k++) {
        var a0 = (k / 8) * PI2;
        var a1 = 4.3 + (r() - 0.5) * 0.5;
        var p0 = [cx + R * Math.cos(a0), cy + R * Math.sin(a0)],
          p1 = [cx + R * Math.cos(a1), cy + R * Math.sin(a1)];
        out += '<circle cx="' + r1(p1[0]) + '" cy="' + r1(p1[1]) + '" r="5" class="f-go s-ink w1 kd kd' + k + '"/>';
        var d = [p0[0] - p1[0], p0[1] - p1[1]];
        loop(".kd" + k, 4.4, [[0, tr(d)], [20, tr(d)], [60, "transform:none"], [93, "transform:none"], [100, tr(d)]], "ease-in-out");
      }
      out += T(130, 108, "weak links drift, strong links sync", "tm tb");
      return out;
    }
  );

  /* regime */
  def(
    "regime",
    "A series that runs at one level, then jumps to a new level at a dashed vertical line marked switch.",
    function () {
      var h = jag(81);
      var a = pts(10, 150, 50, function () {
        return 74 + h() * 12;
      });
      var b = pts(150, 250, 36, function () {
        return 38 + h() * 12;
      });
      var out = poly(a.concat(b), "ln s-ink w15 rg");
      out += line(150, 10, 150, 96, "s-rd w1 dd") + T(150, 8, "switch", "tm tb tr");
      out += T(80, 106, "old rules", "tm ts") + T(200, 106, "new rules", "tm ts");
      draw(".rg", 4, 2, 60);
      return out;
    }
  );

  /* heavy-tails */
  def(
    "heavy-tails",
    "A bell curve with a thin tail drawn dashed, and a second curve with a long fat tail on the right, with an arrow to a rare huge jump.",
    function () {
      function bell(s, tail) {
        return pts(14, 246, 80, function (x) {
          var z = (x - 80) / s;
          return 96 - 70 * (tail ? 1 / (1 + z * z * 0.5) : Math.exp(-z * z / 2));
        });
      }
      var out = poly(bell(24, false), "ln s-ln w2 dp") + poly(bell(24, true), "ln s-bl w2 ht");
      out += line(14, 96, 246, 96, "s-ink w1");
      out += arrowTo(214, 60, 214, 90, "s-rd f-rd w2");
      out += T(214, 54, "rare, huge", "tm tb tr") + T(130, 112, "dashed: thin tails. solid: fat tails", "tm ts");
      draw(".ht", 4, 2, 40);
      return out;
    }
  );

  /* readout */
  def(
    "readout",
    "Scattered points inside a box labelled inside the model, and a straight line that fits through them like a ruler.",
    function () {
      var out = rect(10, 14, 120, 80, "f-so s-ink w1") + T(70, 9, "inside the model", "tm ts");
      var r = rng(91);
      var P = [];
      for (var k = 0; k < 16; k++) {
        var x = 18 + r() * 104;
        var y = 86 - ((x - 18) / 104) * 56 + (r() - 0.5) * 16;
        P.push([x, y]);
        out += rect(r1(x - 2.5), r1(y - 2.5), 5, 5, "f-go s-ink w1");
      }
      out += arrowHead(138, 54, 1, "f-ink") + line(130, 54, 138, 54, "s-ink w2");
      out += T(204, 30, "a straight ruler", "tm ts") + T(204, 42, "laid over what the", "tm ts") + T(204, 54, "model holds inside", "tm ts");
      out += poly([[18, 90], [122, 30]], "ln s-rd w3 rl");
      out += T(130, 111, "if the ruler reads it, it is in there", "tm tb");
      draw(".rl", 3.6, 14, 44);
      return out;
    }
  );

  /* attention */
  def(
    "attention",
    "A row of token boxes with arcs from the last one to earlier ones, some thick and some thin, showing where it looks.",
    function () {
      var out = "";
      var n = 8,
        w = 22,
        gap = 8,
        x0 = 14;
      var ws = [1, 2.5, 1, 4.5, 1, 1.5, 3.5];
      for (var k = 0; k < n; k++) out += rect(x0 + k * (w + gap), 80, w, 20, k === n - 1 ? "f-go s-ink w2" : "f-so s-ink w1");
      var xl = x0 + (n - 1) * (w + gap) + w / 2;
      for (var j = 0; j < n - 1; j++) {
        var xj = x0 + j * (w + gap) + w / 2;
        var mid = (xl + xj) / 2,
          hgt = 70 - (xl - xj) * 0.22;
        out += '<path d="M' + r1(xj) + " 80Q" + r1(mid) + " " + r1(hgt) + " " + r1(xl) + ' 80" class="f-no s-bl at" style="stroke-width:' + ws[j] + ';animation-delay:' + r1(j * 0.1) + 's"/>';
      }
      out += T(130, 14, "each piece decides which others matter", "tm tb") + T(130, 114, "thicker arc = more attention", "tm ts");
      cur.css.push(scope(".at") + "{animation:xvat 3s ease-in-out infinite}" + kf("xvat", [[0, "opacity:.25"], [45, "opacity:1"], [100, "opacity:.25"]]));
      return out;
    }
  );

  /* covariate */
  def(
    "covariate",
    "Two stacked series: a helper series on top and the main series below, with the main series repeating the helper's bump a little later.",
    function () {
      var h = jag(101);
      var top = pts(14, 246, 60, function (x, t) {
        return 28 - 12 * Math.exp(-Math.pow((t - 0.35) * 7, 2)) + h() * 2;
      });
      var bot = pts(14, 246, 60, function (x, t) {
        return 84 - 14 * Math.exp(-Math.pow((t - 0.55) * 7, 2)) + h() * 2;
      });
      var out = poly(top, "ln s-cd w2 cv1") + poly(bot, "ln s-ink w2 cv2");
      out += T(14, 12, "helper series", "ts") + T(14, 104, "main series", "tb");
      out += '<g class="cva">' + arrowTo(98, 36, 140, 68, "s-rd f-rd w2") + "</g>";
      out += T(180, 52, "pushes it later", "tm tr tb");
      draw(".cv1", 4, 2, 30);
      draw(".cv2", 4, 20, 50);
      an(".cva", 4, 52, 64, "opacity:0", "opacity:1", "linear");
      return out;
    }
  );

  /* analogues */
  def(
    "analogues",
    "A series whose latest stretch is highlighted, an earlier stretch that looks the same outlined, and what followed it copied forward as the forecast.",
    function () {
      var h = jag(111);
      var s = pts(10, 170, 100, function (x, t) {
        return 56 - 14 * Math.sin(t * 24) * (0.8 + 0.3 * Math.sin(t * 5)) + h() * 2;
      });
      var out = poly(s, "ln s-ink w15");
      out += rect(146, 24, 24, 70, "f-go s-no wnow") + rect(70, 24, 24, 70, "f-no s-bl w2 dd wold");
      var fut = pts(170, 250, 30, function (x, t) {
        return 56 - 14 * Math.sin(((70 + 24 - 10) / 160 + t * 0.5) * 24) * 0.9;
      });
      out += poly(fut, "ln s-bl w2 dr fu2");
      out += T(158, 106, "now", "tm tb") + T(82, 106, "looked like now", "tm ts") + T(210, 116, "what came next", "tm tb");
      out += line(94, 30, 146, 30, "s-bl w1 dd");
      an(".wold", 4, 6, 20, "opacity:0", "opacity:1", "linear");
      draw(".fu2", 4, 40, 76);
      return out;
    }
  );

  /* benchmark */
  def(
    "benchmark",
    "A stack of test cards on the left, an arrow, and a score sheet on the right giving one score per model.",
    function () {
      var out = "";
      for (var k = 0; k < 4; k++) out += rect(12 + k * 4, 22 + k * 4, 70, 46, k === 3 ? "f-pa s-ink w2" : "f-so s-ink w1");
      out += poly([[28, 62], [38, 50], [46, 58], [58, 40], [70, 52]], "ln s-bl w15");
      out += T(50, 98, "fixed test series", "tm ts");
      out += line(96, 46, 124, 46, "s-ink w2") + arrowHead(124, 46, 1, "f-ink");
      out += rect(140, 18, 108, 70, "f-pa s-ink w2");
      var rows = ["model A", "model B", "model C"];
      for (var j = 0; j < rows.length; j++) {
        out += T(148, 38 + j * 17, rows[j], "ts") + rect(196, 29 + j * 17, 40 - j * 8, 8, "f-bw s-bl w1 sc sc" + j + " o0");
        an(".sc" + j, 4, 20 + j * 12, 40 + j * 12, "transform:scaleX(0)", "transform:none", "ease-out");
      }
      out += T(194, 100, "one scoring rule", "tm tb");
      out += T(130, 114, "same tests, same rule, fair comparison", "tm ts");
      return out;
    }
  );

  /* finetune */
  def(
    "finetune",
    "A small network of nodes and lines whose line thicknesses are nudged a little as a stack of new examples feeds in from the left.",
    function () {
      var cols = [3, 4, 4, 3];
      var xs = [90, 130, 170, 210];
      var rand = rng(121);
      var out = "";
      var nodes = cols.map(function (n, c) {
        var arr = [];
        for (var i = 0; i < n; i++) arr.push([xs[c], 52 + (i - (n - 1) / 2) * 15]);
        return arr;
      });
      for (var c = 0; c < cols.length - 1; c++)
        nodes[c].forEach(function (a) {
          nodes[c + 1].forEach(function (b) {
            var w = 0.5 + rand() * 1.8;
            out += '<line class="nl s-cd" style="stroke-width:' + r1(w) + ';animation-delay:' + r1(rand() * 2) + 's" x1="' + a[0] + '" y1="' + r1(a[1]) + '" x2="' + b[0] + '" y2="' + r1(b[1]) + '"/>';
          });
        });
      nodes.forEach(function (col) {
        col.forEach(function (p) {
          out += rect(p[0] - 4, r1(p[1] - 4), 8, 8, "f-go s-ink w1");
        });
      });
      for (var k = 0; k < 3; k++) out += rect(12 + k * 3, 34 + k * 3, 38, 30, k === 2 ? "f-pa s-ink w1" : "f-so s-ink w1");
      out += poly([[20, 56], [28, 48], [34, 54], [42, 44]], "ln s-or w15");
      out += arrowHead(66, 52, 1, "f-ink") + line(56, 52, 66, 52, "s-ink w2");
      out += T(32, 82, "new cases", "tm ts") + T(150, 100, "nudge the weights a little", "tm tb");
      cur.css.push(scope(".nl") + "{animation:xvnl 2.6s ease-in-out infinite}" + kf("xvnl", [[0, "stroke:var(--k-cd)"], [50, "stroke:var(--k-or)"], [100, "stroke:var(--k-cd)"]]));
      return out;
    }
  );

  /* search */
  def(
    "search",
    "A hilly curve with a dot that hops up the hill step by step, showing a search tweaking a generator to find where the model misses most.",
    function () {
      var land = pts(10, 250, 80, function (x, t) {
        return 92 - 40 * Math.exp(-Math.pow((t - 0.72) * 4.2, 2)) - 18 * Math.exp(-Math.pow((t - 0.28) * 5, 2));
      });
      var out = poly(land.concat([[250, 100], [10, 100]]), "f-so s-no") + poly(land, "ln s-ink w2");
      var path = [0.08, 0.18, 0.3, 0.2, 0.42, 0.55, 0.66, 0.74].map(function (t) {
        var x = 10 + t * 240;
        var y = 92 - 40 * Math.exp(-Math.pow((t - 0.72) * 4.2, 2)) - 18 * Math.exp(-Math.pow((t - 0.28) * 5, 2)) - 7;
        return [x, y];
      });
      out += '<rect x="-4" y="-4" width="8" height="8" class="f-rd s-ink w1 sd" style="' + tr(path[path.length - 1]) + '"/>';
      loop(".sd", 4.4, path.map(function (p, i) { return [4 + (i * 80) / (path.length - 1), tr(p)]; }).concat([[93, tr(path[path.length - 1])], [100, tr(path[0])]]), "ease-in-out");
      out += T(130, 14, "tweak a generator, find the misses", "tm tb") + T(130, 112, "higher = worse for the model", "tm ts");
      return out;
    }
  );

  /* neuron (and pole) */
  def(
    "neuron",
    "A wave that goes above and below a zero line, with the part above shaded as the positive pole and the part below shaded as the negative pole.",
    function () {
      var w = pts(10, 250, 120, function (x, t) {
        return 56 - 26 * Math.sin(t * 17) * (0.6 + 0.4 * Math.sin(t * 3.2 + 1));
      });
      var up = w.map(function (p) { return [p[0], Math.min(p[1], 56)]; });
      var dn = w.map(function (p) { return [p[0], Math.max(p[1], 56)]; });
      var out = poly(up.concat([[250, 56], [10, 56]]), "f-bw s-no") + poly(dn.concat([[250, 56], [10, 56]]), "f-ow s-no");
      out += line(10, 56, 250, 56, "s-ink w1") + poly(w, "ln s-ink w15 nw");
      out += T(14, 20, "positive pole", "tb") + T(14, 104, "negative pole", "tb tr");
      out += T(246, 20, "fires up", "te ts") + T(246, 104, "fires down", "te ts");
      out += T(130, 117, "one neuron = two poles", "tm ts");
      draw(".nw", 4, 2, 60);
      return out;
    }
  );

  /* parameters */
  def(
    "parameters",
    "Layers of nodes joined by lines of different thickness. Each line is one number the model has learned.",
    function () {
      var cols = [3, 5, 5, 3];
      var xs = [34, 94, 154, 214];
      var rand = rng(7);
      var out = "";
      var nodes = cols.map(function (n, c) {
        var arr = [];
        for (var i = 0; i < n; i++) arr.push([xs[c], 50 + (i - (n - 1) / 2) * 15]);
        return arr;
      });
      for (var c = 0; c < cols.length - 1; c++)
        nodes[c].forEach(function (a) {
          nodes[c + 1].forEach(function (b) {
            var w = 0.5 + rand() * 2.5;
            out += '<line class="pl s-cd" style="stroke-width:' + r1(w) + ";animation-delay:" + r1(c * 0.3) + 's" x1="' + a[0] + '" y1="' + r1(a[1]) + '" x2="' + b[0] + '" y2="' + r1(b[1]) + '"/>';
          });
        });
      nodes.forEach(function (col) {
        col.forEach(function (p) {
          out += rect(p[0] - 4, r1(p[1] - 4), 8, 8, "f-go s-ink w1");
        });
      });
      out += T(130, 108, "each line is one learned number", "tm tb");
      cur.css.push(scope(".pl") + "{opacity:.6;animation:xvpl 2.4s ease-in-out infinite}" + kf("xvpl", [[0, "opacity:.18"], [40, "opacity:.95"], [100, "opacity:.18"]]));
      return out;
    }
  );

  /* geometric mean */
  def(
    "geomean",
    "A scale where twice as bad and half as bad sit equally far on either side of 1, so their average lands on 1.",
    function () {
      var out = line(24, 62, 236, 62, "s-ink w2");
      [[40, "÷4"], [82, "÷2"], [130, "1"], [178, "×2"], [220, "×4"]].forEach(function (t) {
        out += line(t[0], 54, t[0], 70, "s-ink w2") + T(t[0], 86, t[1], "tm tb");
      });
      out += rect(76, 40, 12, 12, "f-go s-ink w1 gl") + rect(172, 40, 12, 12, "f-or s-ink w1 gr");
      out += '<path d="M124 36L130 44L136 36" class="s-rd w2 f-no"/>';
      out += T(130, 20, "average of ×2 and ÷2 is 1", "tm tb") + T(130, 108, "for ratios, multiply and take the root", "tm ts");
      an(".gl", 3.6, 10, 50, "transform:translateX(-30px)", "transform:none", "ease-out");
      an(".gr", 3.6, 10, 50, "transform:translateX(30px)", "transform:none", "ease-out");
      return out;
    }
  );

  /* crps */
  def(
    "crps",
    "A bell-shaped spread of guesses with a vertical line for what really happened, the distance between them shaded.",
    function () {
      var out = poly(
        pts(14, 246, 80, function (x) {
          var z = (x - 110) / 28;
          return 94 - 70 * Math.exp(-z * z / 2);
        }),
        "ln s-bl w2"
      );
      out += line(14, 94, 246, 94, "s-ink w1") + line(176, 20, 176, 94, "s-rd w3 tl");
      out += rect(110, 86, 66, 8, "f-t4 s-no sh");
      out += T(110, 110, "guesses", "tm ts") + T(176, 14, "what happened", "tm tb tr");
      an(".sh", 3.6, 8, 40, "transform:scaleX(0)", "transform:none", "ease-out");
      cur.css.push(scope(".sh") + "{transform-origin:0 50%}");
      return out;
    }
  );

  /* ---------- public API ---------- */

  var HATCH =
    '<defs><pattern id="gsx-hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">' +
    '<rect width="4" height="4" class="f-so"/><line x1="0" y1="0" x2="0" y2="4" class="s-ln w15"/></pattern></defs>';

  function css() {
    return (
      BASE_CSS +
      order
        .map(function (id) {
          return defs[id].css;
        })
        .join("")
    );
  }

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  }

  function svg(id, alt) {
    var d = defs[id];
    if (!d) return "";
    return (
      '<svg viewBox="0 0 ' +
      W +
      " " +
      H +
      '" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="' +
      esc(alt || d.alt) +
      '" focusable="false">' +
      HATCH +
      '<g class="v-' +
      id +
      '">' +
      d.svg +
      "</g></svg>"
    );
  }

  var styleEl = null;
  function ensureStyle() {
    if (styleEl || typeof document === "undefined") return;
    styleEl = document.createElement("style");
    styleEl.setAttribute("data-gsx-visuals", "");
    styleEl.textContent = css();
    document.head.appendChild(styleEl);
  }

  function mount(host, id, opts) {
    opts = opts || {};
    ensureStyle();
    host.classList.add("gsx-vis");
    host.classList.toggle("is-still", !!opts.still);
    host.innerHTML = svg(id, opts.alt);
  }

  function unmount(host) {
    host.innerHTML = "";
  }

  return {
    ids: order.slice(),
    has: function (id) {
      return Object.prototype.hasOwnProperty.call(defs, id);
    },
    alt: function (id) {
      return defs[id] ? defs[id].alt : "";
    },
    svg: svg,
    css: css,
    mount: mount,
    unmount: unmount,
    width: W,
    height: H,
  };
});
