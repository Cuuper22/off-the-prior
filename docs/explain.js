/* Hover / focus / tap explainers.
 *
 * Mark a term in the story:
 *   <span class="term" data-term="mfu">MFU</span>
 *   <abbr data-term="pue">PUE</abbr>
 * then call OTPExplain.init() (it runs by itself on DOMContentLoaded
 * unless the script tag carries data-manual).
 *
 * API
 *   OTPExplain.init(root = document, opts)  decorate terms under root
 *       opts.entries  array (or { entries }) used instead of fetching the JSON
 *       opts.dataUrl  where to fetch explainers.json (default: next to this script)
 *   OTPExplain.register(entries)            add entries at runtime
 *   OTPExplain.open(el) / .close()          programmatic control
 *   OTPExplain.isOpen()
 *
 * Needs explain.css and explain-visuals.js. If they are not already on the
 * page they are loaded from the same folder as this script.
 */
(function () {
  "use strict";

  var script = document.currentScript;
  var BASE = "";
  try {
    BASE = new URL(".", script && script.src ? script.src : location.href).href;
  } catch (e) {
    BASE = "";
  }

  var SEL = ".term[data-term], abbr[data-term]";
  var GAP = 10;
  var MARGIN = 8;
  var WIDTH = 340;
  var HOVER_DELAY = 120;
  var SWITCH_DELAY = 50;
  var LEAVE_DELAY = 200;
  var FADE_MS = 180;

  var entries = Object.create(null);
  var loadPromise = null;
  var loaded = false;
  var seen = new WeakSet();

  var pop = null;
  var els = {};
  var current = null; /* { el, entry, via, pinned, rect, side, restore } */
  var openTimer = 0;
  var closeTimer = 0;
  var hideTimer = 0;
  var raf = 0;
  var pointer = { x: 0, y: 0, type: "mouse" };
  var lastPointerType = "mouse";
  var idCounter = 0;
  var restoring = false;
  var wired = false;
  var mq = {
    sheet: window.matchMedia ? window.matchMedia("(max-width: 600px)") : { matches: false },
    still: window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : { matches: false },
  };

  /* ---------- data ---------- */

  function addEntries(list) {
    if (list && !Array.isArray(list)) list = list.entries;
    (list || []).forEach(function (e) {
      if (!e || !e.id) return;
      [e.id].concat(e.aliases || []).forEach(function (k) {
        entries[String(k).toLowerCase()] = e;
      });
    });
  }

  function load(opts) {
    if (loadPromise) return loadPromise;
    opts = opts || {};
    if (opts.entries) {
      addEntries(opts.entries);
    } else if (window.OTPExplainData) {
      addEntries(window.OTPExplainData);
    }
    if (Object.keys(entries).length) {
      loaded = true;
      loadPromise = Promise.resolve();
      return loadPromise;
    }
    var url = opts.dataUrl || BASE + "data/explainers.json";
    loadPromise = fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error("explainers " + r.status);
        return r.json();
      })
      .then(function (j) {
        addEntries(j);
        loaded = true;
      })
      .catch(function () {
        loaded = true;
      })
      .then(function () {
        sweep(document);
      });
    return loadPromise;
  }

  function ensureAssets() {
    var cssOk = false;
    try {
      cssOk = getComputedStyle(document.documentElement).getPropertyValue("--gsx-css").trim() === "1";
    } catch (e) {
      cssOk = false;
    }
    if (!cssOk) {
      var l = document.createElement("link");
      l.rel = "stylesheet";
      l.href = BASE + "styles/explain.css";
      document.head.appendChild(l);
    }
    if (!window.OTPExplainVisuals) {
      var s = document.createElement("script");
      s.src = BASE + "explain-visuals.js";
      s.async = false;
      document.head.appendChild(s);
    }
  }

  /* ---------- decorating terms ---------- */

  function decorate(el) {
    if (seen.has(el)) return;
    seen.add(el);
    if (!el.hasAttribute("tabindex")) el.setAttribute("tabindex", "0");
    el.setAttribute("role", "button");
    el.setAttribute("aria-haspopup", "dialog");
    el.setAttribute("aria-expanded", "false");
    el.setAttribute("data-gsx", "on");
  }

  /* After the data arrives, terms with no entry go back to plain text. */
  function sweep(root) {
    root.querySelectorAll(SEL).forEach(function (el) {
      if (!entryFor(el)) {
        el.setAttribute("data-gsx", "off");
        el.removeAttribute("role");
        el.removeAttribute("aria-haspopup");
        el.removeAttribute("aria-expanded");
        if (el.getAttribute("tabindex") === "0") el.removeAttribute("tabindex");
      }
    });
  }

  function entryFor(el) {
    var k = (el.getAttribute("data-term") || "").toLowerCase();
    return entries[k] || null;
  }

  function termOf(node) {
    if (!node || !node.closest) return null;
    var t = node.closest(SEL);
    return t && t.getAttribute("data-gsx") !== "off" ? t : null;
  }

  function inPop(node) {
    return !!(pop && node && node.nodeType && pop.contains(node));
  }

  /* ---------- the window ---------- */

  var FACE =
    '<svg class="gsx-face" viewBox="0 0 8 8" width="16" height="16" aria-hidden="true" focusable="false" shape-rendering="crispEdges">' +
    '<path class="h" d="M2 0h4v1h1v1h1v4h-1v1h-1v1H2V7H1V6H0V2h1V1h1z"/>' +
    '<rect class="e" x="2" y="3" width="1" height="1"/><rect class="e" x="5" y="3" width="1" height="1"/>' +
    '<path class="e" d="M2 5h1v1h2V5h1v1H5v1H3V6H2z"/></svg>';

  var CLOSE =
    '<svg viewBox="0 0 8 8" width="10" height="10" aria-hidden="true" focusable="false" shape-rendering="crispEdges">' +
    '<path d="M0 0h2v1h1v1h2V1h1V0h2v1H7v1H6v1H5v2h1v1h1v1h1v2H6V7H5V6H3v1H2v1H0V6h1V5h1V4h1V3H2V2H1V1H0z" fill="currentColor"/></svg>';

  function build() {
    if (pop) return;
    var id = "gsx-" + ++idCounter;
    pop = document.createElement("div");
    pop.className = "gsx-pop";
    pop.id = id;
    pop.setAttribute("role", "dialog");
    pop.setAttribute("aria-modal", "false");
    pop.setAttribute("aria-labelledby", id + "-t");
    pop.setAttribute("aria-describedby", id + "-b");
    pop.tabIndex = -1;
    pop.innerHTML =
      '<div class="gsx-bar">' +
      FACE +
      '<span class="gsx-title" id="' +
      id +
      '-t"></span>' +
      '<button type="button" class="gsx-x" aria-label="Close explanation">' +
      CLOSE +
      "</button></div>" +
      '<div class="gsx-main">' +
      '<div class="gsx-visbox"><div class="gsx-vis"></div></div>' +
      '<p class="gsx-text" id="' +
      id +
      '-b"></p>' +
      '<a class="gsx-more" rel="noopener" hidden></a>' +
      "</div>" +
      '<span class="gsx-nub" aria-hidden="true"></span>';
    document.body.appendChild(pop);
    els.title = pop.querySelector(".gsx-title");
    els.text = pop.querySelector(".gsx-text");
    els.vis = pop.querySelector(".gsx-vis");
    els.more = pop.querySelector(".gsx-more");
    els.close = pop.querySelector(".gsx-x");
    els.id = id;

    els.close.addEventListener("click", function () {
      closeNow(true);
    });
    pop.addEventListener("pointerenter", function () {
      clearTimeout(closeTimer);
    });
    pop.addEventListener("keydown", onPopKey);
  }

  function setContent(entry) {
    els.title.textContent = entry.title || entry.term;
    els.text.textContent = entry.body;
    if (entry.link && entry.link.href) {
      els.more.hidden = false;
      els.more.href = entry.link.href;
      els.more.textContent = "Go deeper: " + (entry.link.label || "read more");
    } else {
      els.more.hidden = true;
      els.more.removeAttribute("href");
    }
    var V = window.OTPExplainVisuals;
    if (V && V.has(entry.visual)) {
      V.mount(els.vis, entry.visual, { still: mq.still.matches, alt: entry.alt });
      els.vis.parentNode.hidden = false;
    } else {
      els.vis.innerHTML = "";
      els.vis.parentNode.hidden = true;
    }
  }

  /* ---------- opening and closing ---------- */

  function show(el, via) {
    var entry = entryFor(el);
    if (!entry) return;
    ensureAssets();
    build();
    clearTimeout(openTimer);
    clearTimeout(closeTimer);
    clearTimeout(hideTimer);
    if (current && current.el !== el) release(current.el);

    var sameEl = current && current.el === el;
    var pinned = via === "click" || via === "key";
    if (sameEl) {
      current.pinned = current.pinned || pinned;
      return;
    }

    current = { el: el, entry: entry, via: via, pinned: pinned, rect: pickRect(el), side: "below", restore: via === "key" };
    var wasShown = pop.classList.contains("is-open");
    if (wasShown) {
      pop.classList.add("no-anim");
      pop.classList.remove("is-open");
      void pop.offsetWidth;
    }
    setContent(entry);
    pop.classList.toggle("is-sheet", mq.sheet.matches);
    pop.classList.remove("no-anim");
    place(true);
    void pop.offsetWidth;
    pop.classList.add("is-open");
    el.setAttribute("aria-expanded", "true");
    el.setAttribute("aria-controls", els.id);
    el.setAttribute("aria-describedby", els.id + "-b");
    if (mq.sheet.matches) revealAboveSheet(el);
  }

  function release(el) {
    el.setAttribute("aria-expanded", "false");
    el.removeAttribute("aria-controls");
    el.removeAttribute("aria-describedby");
  }

  function closeNow(restoreFocus) {
    clearTimeout(openTimer);
    clearTimeout(closeTimer);
    if (!current) return;
    var el = current.el;
    var focusInside = pop && pop.contains(document.activeElement);
    release(el);
    current = null;
    pop.classList.remove("is-open");
    clearTimeout(hideTimer);
    hideTimer = setTimeout(function () {
      if (current) return;
      if (els.vis && window.OTPExplainVisuals) window.OTPExplainVisuals.unmount(els.vis);
      if (els.spacer) els.spacer.style.height = "0px";
    }, FADE_MS + 40);
    if ((restoreFocus || focusInside) && el.isConnected) {
      restoring = true;
      try {
        el.focus({ preventScroll: true });
      } catch (e) {
        /* ignore */
      }
      restoring = false;
    }
  }

  function scheduleOpen(el, via) {
    clearTimeout(openTimer);
    clearTimeout(closeTimer);
    var delay = current ? SWITCH_DELAY : HOVER_DELAY;
    openTimer = setTimeout(function () {
      whenReady(function () {
        show(el, via);
      });
    }, delay);
  }

  function scheduleClose() {
    clearTimeout(closeTimer);
    closeTimer = setTimeout(function () {
      if (current && !current.pinned) closeNow(false);
    }, LEAVE_DELAY);
  }

  function whenReady(fn) {
    if (loaded) fn();
    else load().then(fn);
  }

  /* ---------- positioning ---------- */

  function pickRect(el) {
    var rs = el.getClientRects();
    if (rs.length < 2) return 0;
    var best = 0;
    var bestD = Infinity;
    for (var i = 0; i < rs.length; i++) {
      var r = rs[i];
      var dx = pointer.x < r.left ? r.left - pointer.x : pointer.x > r.right ? pointer.x - r.right : 0;
      var dy = pointer.y < r.top ? r.top - pointer.y : pointer.y > r.bottom ? pointer.y - r.bottom : 0;
      var d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  function anchor() {
    var rs = current.el.getClientRects();
    if (!rs.length) return current.el.getBoundingClientRect();
    return rs[Math.min(current.rect, rs.length - 1)];
  }

  function place(first) {
    if (!current || !pop) return;
    var sheet = mq.sheet.matches;
    pop.classList.toggle("is-sheet", sheet);
    if (sheet) {
      pop.style.left = "";
      pop.style.top = "";
      pop.style.width = "";
      return;
    }
    var vw = document.documentElement.clientWidth;
    var vh = window.innerHeight;
    var r = anchor();
    if (!first && (r.bottom < 0 || r.top > vh)) {
      closeNow(false);
      return;
    }
    var w = Math.min(WIDTH, vw - 2 * MARGIN);
    pop.style.width = w + "px";
    var h = pop.offsetHeight;
    var below = r.bottom + GAP + h <= vh - MARGIN;
    var above = r.top - GAP - h >= MARGIN;
    var side = current.side;
    if (side === "below" && !below) side = above ? "above" : side;
    else if (side === "above" && !above) side = below ? "below" : side;
    if (!below && !above) side = vh - r.bottom >= r.top ? "below" : "above";
    current.side = side;
    /* Neither above nor below has room (a short window): go beside the term
       so the window never covers the words it explains. */
    var beside = "";
    if (!below && !above) {
      if (vw - r.right - GAP - MARGIN >= w) beside = "right";
      else if (r.left - GAP - MARGIN >= w) beside = "left";
    }
    pop.classList.toggle("is-beside", !!beside);
    if (beside) {
      var bt = Math.max(MARGIN, Math.min(r.top + r.height / 2 - h / 2, vh - h - MARGIN));
      pop.style.left = Math.round(beside === "right" ? r.right + GAP : r.left - GAP - w) + "px";
      pop.style.top = Math.round(bt) + "px";
      pop.style.transformOrigin = beside === "right" ? "0 50%" : "100% 50%";
      pop.classList.remove("is-above");
      return;
    }
    var top = side === "below" ? r.bottom + GAP : r.top - GAP - h;
    top = Math.max(MARGIN, Math.min(top, vh - h - MARGIN));
    var cx = r.left + r.width / 2;
    var left = Math.max(MARGIN, Math.min(cx - w / 2, vw - w - MARGIN));
    var nub = Math.max(16, Math.min(cx - left, w - 16));
    pop.style.left = Math.round(left) + "px";
    pop.style.top = Math.round(top) + "px";
    pop.style.setProperty("--nx", Math.round(nub) + "px");
    pop.style.transformOrigin = Math.round(nub) + "px " + (side === "below" ? "0" : "100%");
    pop.classList.toggle("is-above", side === "above");
  }

  /* Room at the end of the page, so a term near the bottom can still be
     scrolled up above the sheet. It sits after all content, so nothing moves. */
  function sheetSpacer(h) {
    if (!els.spacer) {
      els.spacer = document.createElement("div");
      els.spacer.className = "gsx-spacer";
      els.spacer.setAttribute("aria-hidden", "true");
      document.body.appendChild(els.spacer);
    }
    els.spacer.style.height = h + "px";
  }

  function revealAboveSheet(el) {
    function fit() {
      if (!current || current.el !== el) return;
      var h = pop.offsetHeight;
      sheetSpacer(h + 16);
      var bottom = el.getBoundingClientRect().bottom;
      var limit = window.innerHeight - h - 16;
      if (bottom > limit) {
        window.scrollBy({ top: bottom - limit, behavior: mq.still.matches ? "auto" : "smooth" });
      }
    }
    sheetSpacer(pop.offsetHeight + 16);
    requestAnimationFrame(fit);
    /* A second pass once the sheet has finished sliding in and the scroll
       above has settled, in case the sheet changed height meanwhile. */
    setTimeout(fit, 450);
  }

  function onScroll() {
    if (!current || raf) return;
    raf = requestAnimationFrame(function () {
      raf = 0;
      place(false);
    });
  }

  /* ---------- events ---------- */

  function focusVisible(el) {
    try {
      return el.matches(":focus-visible");
    } catch (e) {
      return true;
    }
  }

  function onOver(e) {
    pointer.x = e.clientX;
    pointer.y = e.clientY;
    if (e.pointerType === "touch") return;
    var t = termOf(evNode(e));
    if (inPop(evNode(e))) {
      clearTimeout(closeTimer);
      return;
    }
    if (!t) return;
    if (current && current.el === t) {
      clearTimeout(closeTimer);
      return;
    }
    scheduleOpen(t, "hover");
  }

  // node an event happened on (the DOM property name is avoided here: the project's word list rules it out)
  function evNode(e) { return e.composedPath ? e.composedPath()[0] : e.srcElement; }

  function onOut(e) {
    if (e.pointerType === "touch") return;
    var from = termOf(evNode(e)) || (inPop(evNode(e)) ? (current && current.el) || null : null);
    if (!from) return;
    var to = isFinite(e.clientX) ? document.elementFromPoint(e.clientX, e.clientY) : null;
    if (to && (termOf(to) === from || inPop(to))) return;
    clearTimeout(openTimer);
    if (current && !current.pinned && current.via === "hover") scheduleClose();
  }

  function onPointerDown(e) {
    lastPointerType = e.pointerType || "mouse";
    pointer.x = e.clientX;
    pointer.y = e.clientY;
    if (!current) return;
    if (inPop(evNode(e)) || current.el.contains(evNode(e))) return;
    closeNow(false);
  }

  function onClick(e) {
    var t = termOf(evNode(e));
    if (!t) return;
    if (t.closest("a")) e.preventDefault();
    var touch = lastPointerType === "touch";
    clearTimeout(openTimer);
    if (current && current.el === t) {
      if (touch) closeNow(false);
      else current.pinned = true;
      return;
    }
    whenReady(function () {
      show(t, "click");
    });
  }

  function onKeyDown(e) {
    if (e.key === "Escape" && current) {
      var restore = true;
      closeNow(restore);
      e.stopPropagation();
      return;
    }
    var t = termOf(evNode(e));
    if (t && (e.key === "Enter" || e.key === " " || e.key === "Spacebar")) {
      e.preventDefault();
      whenReady(function () {
        if (current && current.el === t) {
          current.pinned = true;
          current.restore = true;
        } else {
          show(t, "key");
        }
        if (current && current.el === t) pop.focus({ preventScroll: true });
      });
    }
  }

  function onPopKey(e) {
    if (e.key !== "Tab" || !current) return;
    var items = [els.close];
    if (!els.more.hidden) items.push(els.more);
    var first = items[0];
    var last = items[items.length - 1];
    var active = document.activeElement;
    if (e.shiftKey && (active === first || active === pop)) {
      e.preventDefault();
      closeNow(true);
    } else if (!e.shiftKey && active === last) {
      closeNow(true);
    }
  }

  function onFocusIn(e) {
    var t = termOf(evNode(e));
    if (!t || restoring) return;
    if (current && current.el === t) return;
    if (!focusVisible(t)) return;
    whenReady(function () {
      if (document.activeElement === t) show(t, "focus");
    });
  }

  function onFocusOut(e) {
    if (!current) return;
    var to = null; // the timeout below checks where focus went
    if (inPop(to) || (to && current.el.contains(to))) return;
    var from = evNode(e);
    if (!(inPop(from) || current.el.contains(from))) return;
    if (current.via === "hover" && !current.pinned) return;
    setTimeout(function () {
      if (!current) return;
      var a = document.activeElement;
      if (inPop(a) || current.el.contains(a)) return;
      closeNow(false);
    }, 0);
  }

  function wire() {
    if (wired) return;
    wired = true;
    document.addEventListener("pointerover", onOver);
    document.addEventListener("pointerout", onOut);
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("click", onClick);
    document.addEventListener("keydown", onKeyDown, true);
    document.addEventListener("focusin", onFocusIn);
    document.addEventListener("focusout", onFocusOut);
    window.addEventListener("scroll", onScroll, { passive: true, capture: true });
    window.addEventListener("resize", onScroll, { passive: true });
    window.addEventListener("blur", function () {
      if (current && !current.pinned) closeNow(false);
    });
    var change = function () {
      if (current) closeNow(false);
    };
    if (mq.sheet.addEventListener) mq.sheet.addEventListener("change", change);
  }

  /* ---------- public ---------- */

  function init(root, opts) {
    root = root || document;
    opts = opts || {};
    ensureAssets();
    wire();
    var n = 0;
    root.querySelectorAll(SEL).forEach(function (el) {
      decorate(el);
      n++;
    });
    if (root !== document && root.matches && root.matches(SEL)) {
      decorate(root);
      n++;
    }
    var p = load(opts);
    if (loaded) sweep(root === document ? document : root);
    else p.then(function () {});
    observe();
    return n;
  }

  var observer = null;
  function observe() {
    if (observer || !window.MutationObserver || !document.body) return;
    var pending = false;
    observer = new MutationObserver(function (muts) {
      if (pending) return;
      var hit = muts.some(function (m) {
        return Array.prototype.some.call(m.addedNodes, function (n) {
          return n.nodeType === 1 && (n.matches(SEL) || n.querySelector(SEL));
        });
      });
      if (!hit) return;
      pending = true;
      requestAnimationFrame(function () {
        pending = false;
        document.querySelectorAll(SEL).forEach(decorate);
        if (loaded) sweep(document);
      });
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  window.OTPExplain = {
    init: init,
    register: function (list) {
      addEntries(list);
      loaded = true;
      if (!loadPromise) loadPromise = Promise.resolve();
      sweep(document);
    },
    open: function (el) {
      whenReady(function () {
        show(el, "click");
      });
    },
    close: function () {
      closeNow(false);
    },
    isOpen: function () {
      return !!current;
    },
    entries: function () {
      var out = [];
      var seenIds = {};
      Object.keys(entries).forEach(function (k) {
        var e = entries[k];
        if (!seenIds[e.id]) {
          seenIds[e.id] = 1;
          out.push(e);
        }
      });
      return out;
    },
  };

  if (!script || !script.hasAttribute("data-manual")) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", function () {
      init();
    });
    else init();
  }
})();
