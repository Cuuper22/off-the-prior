/* Story figures (copied from gpu_stack's docs/figures/index.js; namespace renamed). window.OTPFigures.mount(el, figureId) builds one figure
   inside el. Each figure plays when it scrolls into view, pauses off screen,
   has a Replay button, a text alternative, and a still final frame when the
   reader prefers reduced motion. Figure files register themselves with
   OTPFigures.define(id, definition) and are loaded on demand from the
   folder this file lives in. No libraries. */
(function (root) {
  "use strict";
  var F = (root.OTPFigures = root.OTPFigures || {});
  var defs = (F.defs = F.defs || {});
  var ALIASES = {};
  var IDS = [
    "fan", "chaos-map", "lorenz", "noise", "regime", "readouts", "tails", "covariates", "realdata",
    "selfval", "bench", "search", "finetune", "atlas",
  ];
  var NS = "http://www.w3.org/2000/svg";
  var hasDom = typeof document !== "undefined";
  var base = "";
  if (hasDom && document.currentScript && document.currentScript.src) {
    base = document.currentScript.src.replace(/[^/]*$/, "");
  }

  function clamp(x, a, b) {
    return Math.max(a === undefined ? 0 : a, Math.min(b === undefined ? 1 : b, x));
  }
  function lerp(a, b, t) {
    return a + (b - a) * t;
  }
  function ease(t) {
    t = clamp(t);
    return t * t * (3 - 2 * t);
  }
  function fmt(n, digits) {
    return Number(n).toLocaleString("en-US", { maximumFractionDigits: digits || 0 });
  }
  /* Small seeded random numbers so frames are the same every run. */
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
  function apply(node, attrs) {
    for (var k in attrs) {
      if (attrs[k] === null || attrs[k] === undefined || attrs[k] === false) continue;
      if (k === "text") node.textContent = attrs[k];
      else if (k === "html") node.innerHTML = attrs[k];
      else if (k === "class") node.setAttribute("class", attrs[k]);
      else if (k.slice(0, 2) === "on" && typeof attrs[k] === "function")
        node.addEventListener(k.slice(2), attrs[k]);
      else node.setAttribute(k, attrs[k] === true ? "" : attrs[k]);
    }
  }
  function build(make, tag, attrs, kids) {
    var n = make(tag);
    apply(n, attrs || {});
    (kids || []).forEach(function (c) {
      if (c) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return n;
  }
  function h(tag, attrs, kids) {
    return build(function (t) { return document.createElement(t); }, tag, attrs, kids);
  }
  function s(tag, attrs, kids) {
    return build(function (t) { return document.createElementNS(NS, t); }, tag, attrs, kids);
  }

  function reducedMotion() {
    try {
      return !!(root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches);
    } catch (e) {
      return false;
    }
  }

  /* A figure is a function of time: render(t) draws the frame for t seconds.
     That keeps the animation, the still frame and the screenshots identical. */
  function Timeline(cfg, onState) {
    var t = 0, playing = false, last = 0, raf = 0;
    var dur = cfg.duration;
    function frame(now) {
      if (!playing) return;
      t += clamp((now - last) / 1000, 0, 0.1);
      last = now;
      if (t >= dur) {
        if (cfg.loop) t = t % dur;
        else {
          t = dur;
          playing = false;
        }
      }
      cfg.render(t);
      if (playing) raf = root.requestAnimationFrame(frame);
      else if (onState) onState(false);
    }
    var api = {
      get t() { return t; },
      get playing() { return playing; },
      duration: dur,
      play: function () {
        if (playing || (!cfg.loop && t >= dur)) return;
        playing = true;
        last = root.performance.now();
        raf = root.requestAnimationFrame(frame);
      },
      pause: function () {
        playing = false;
        if (raf) root.cancelAnimationFrame(raf);
      },
      seek: function (x) {
        t = clamp(x, 0, dur);
        cfg.render(t);
      },
      end: function () {
        api.pause();
        t = cfg.still !== undefined ? cfg.still : dur;
        cfg.render(t);
      },
      replay: function () {
        api.pause();
        t = 0;
        if (cfg.onReplay) cfg.onReplay();
        cfg.render(0);
        api.play();
      },
    };
    return api;
  }

  F.define = function (id, def) {
    defs[id] = def;
  };
  F.ids = IDS.slice();
  F.util = { clamp: clamp, lerp: lerp, ease: ease, fmt: fmt, rng: rng, h: h, s: s };

  function load(id, then) {
    if (!hasDom || !base) return false;
    var tag = document.createElement("script");
    tag.src = base + id + ".js";
    tag.onload = then;
    tag.onerror = function () {
      if (root.console) root.console.info("figure script not found: " + id);
    };
    document.head.appendChild(tag);
    return true;
  }

  F.mount = function (el, figureId) {
    var id = ALIASES[figureId] || figureId;
    if (el.__gsf) return el.__gsf;
    var def = defs[id];
    if (!def) {
      load(id, function () { if (defs[id]) F.mount(el, figureId); });
      return null;
    }
    var reduced = reducedMotion();
    var summaryEl = h("figcaption", { class: "gsf-sr", text: def.summary });
    var stage = h("div", { class: "gsf-stage" });
    var replay = h("button", { type: "button", class: "gsf-replay", "aria-label": "Replay the animation", text: "Replay" });
    var fig = h("figure", { class: "gsf gsf--" + id, role: "group", "aria-label": def.label || def.title }, [
      h("div", { class: "gsf-bar" }, [h("span", { class: "gsf-title", text: def.title }), replay]),
      stage,
      summaryEl,
    ]);
    el.textContent = "";
    el.appendChild(fig);

    var visibleHooks = [], resizeHooks = [], visible = false, started = false, ro = null;
    var ctx = {
      el: stage,
      root: el,
      fig: fig,
      h: h,
      s: s,
      util: F.util,
      get reduced() { return reduced; },
      setSummary: function (txt) { summaryEl.textContent = txt; },
      onVisible: function (fn) { visibleHooks.push(fn); },
      onResize: function (fn) {
        resizeHooks.push(fn);
        if (root.ResizeObserver && !ro) {
          ro = new root.ResizeObserver(function () { resizeHooks.forEach(function (f) { f(); }); });
          ro.observe(stage);
        }
      },
      css: function (name, fallback) {
        try {
          var v = root.getComputedStyle(fig).getPropertyValue(name).trim();
          return v || fallback;
        } catch (e) {
          return fallback;
        }
      },
    };

    var cfg = def.build(ctx) || { duration: 1, render: function () {} };
    var tl = Timeline(cfg, null);
    replay.hidden = reduced || cfg.autoplay === false;
    replay.addEventListener("click", function () { tl.replay(); });

    function showStill() {
      tl.end();
    }
    if (reduced) showStill();
    else tl.seek(0);

    var ctrl = {
      id: id,
      timeline: tl,
      seek: function (x) { tl.pause(); tl.seek(x); },
      end: showStill,
      replay: function () { tl.replay(); },
      get visible() { return visible; },
    };
    el.__gsf = ctrl;

    function setVisible(v) {
      if (v === visible) return;
      visible = v;
      visibleHooks.forEach(function (fn) { fn(v); });
      if (reduced || cfg.autoplay === false) return;
      if (v) {
        if (!started) {
          started = true;
          tl.replay();
        } else tl.play();
      } else tl.pause();
    }
    if (root.IntersectionObserver) {
      var io = new root.IntersectionObserver(
        function (entries) {
          entries.forEach(function (e) { setVisible(e.isIntersecting); });
        },
        { threshold: 0.2 }
      );
      io.observe(fig);
    } else setVisible(true);

    return ctrl;
  };

  F.mountAll = function (scope) {
    var out = [];
    (scope || document).querySelectorAll("[data-figure]").forEach(function (n) {
      out.push(F.mount(n, n.getAttribute("data-figure")));
    });
    return out;
  };
})(typeof window !== "undefined" ? window : globalThis);
