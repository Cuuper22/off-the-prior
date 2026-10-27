/* The toy's series generators in plain JavaScript, a port of otp/generators (mixture.py, primitives.py, chaos.py,
   regime.py, tails.py) for the pieces the toy offers: a season, AR noise, and one wild piece.

     y_t = season_w * z_season + 0.3 * z_ar + strength * z_wild + noise * e_t

   where each z is the piece standardized by its pilot mean and sd (exported by scripts/export_site_weights.py into
   live_spec.json, so the browser uses the numbers Python measured). The referee (oracle) is the same as in
   otp/generators/base.py: copy the exact state at the end of the context into many paths, advance each with fresh
   noise, and take quantiles.

   Random streams cannot match numpy's, so one seed gives a different series than the same seed in Python.
   tests/test_site_parity.py compares statistics instead of values.
   Works in a browser (self.OTPLiveGen) and in node (require). */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.OTPLiveGen = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  var TWO_PI = 2 * Math.PI;
  var LEVELS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99];
  var MEDIAN = 10;

  /* ---------- random numbers: xoshiro128** seeded by splitmix32 ---------- */
  function Rng(seed) {
    var s = (seed >>> 0) || 1, st = new Uint32Array(4);
    function sm() {
      s = (s + 0x9e3779b9) >>> 0;
      var z = s;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      return (z ^ (z >>> 16)) >>> 0;
    }
    for (var i = 0; i < 4; i++) st[i] = sm();
    this.st = st;
    this.spare = null;
  }
  function rotl(x, k) { return ((x << k) | (x >>> (32 - k))) >>> 0; }
  Rng.prototype.u32 = function () {
    var s = this.st, r = Math.imul(rotl(Math.imul(s[1], 5) >>> 0, 7), 9) >>> 0, t = (s[1] << 9) >>> 0;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 11);
    return r;
  };
  Rng.prototype.u = function () { return (this.u32() + 0.5) / 4294967296; };       // open interval (0, 1)
  Rng.prototype.normal = function () {
    if (this.spare !== null) { var v = this.spare; this.spare = null; return v; }
    var a, b, r;
    do { a = 2 * this.u() - 1; b = 2 * this.u() - 1; r = a * a + b * b; } while (r >= 1 || r === 0);
    var f = Math.sqrt((-2 * Math.log(r)) / r);
    this.spare = b * f;
    return a * f;
  };
  Rng.prototype.exp = function () { return -Math.log(this.u()); };
  function lgamma(x) {
    var c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    var y = x, t = x + 5.5, ser = 1.000000000190015;
    t -= (x + 0.5) * Math.log(t);
    for (var j = 0; j < 6; j++) ser += c[j] / ++y;
    return -t + Math.log((2.5066282746310005 * ser) / x);
  }
  Rng.prototype.poisson = function (lam) {
    if (!(lam > 0)) return 0;
    if (lam < 30) {
      var L = Math.exp(-lam), k = 0, p = 1;
      do { k++; p *= this.u(); } while (p > L);
      return k - 1;
    }
    // transformed rejection (Hormann, PTRS)
    var slam = Math.sqrt(lam), loglam = Math.log(lam), b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b;
    var invalpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
    for (;;) {
      var U = this.u() - 0.5, V = this.u(), us = 0.5 - Math.abs(U);
      var kk = Math.floor(((2 * a) / us + b) * U + lam + 0.43);
      if (us >= 0.07 && V <= vr) return kk;
      if (kk < 0 || (us < 0.013 && V > us)) continue;
      if (Math.log(V) + Math.log(invalpha) - Math.log(a / (us * us) + b) <= -lam + kk * loglam - lgamma(kk + 1)) return kk;
    }
  };

  /* ---------- pieces. Each: { warmup, init(rng) -> state, step(state, n, rng, out), clone(state) } ---------- */
  function cloneState(s) {
    var o = {};
    for (var k in s) o[k] = s[k] && s[k].slice ? s[k].slice() : s[k];
    return o;
  }

  function seasonPiece(P) {
    var per = P.params.period, amps = P.amps, ph = P.phases;
    return { warmup: 0,
      init: function () { return { t: 0 }; },
      step: function (st, n, rng, out) {
        for (var i = 0; i < n; i++) {
          var tt = st.t + i, y = 0;
          for (var j = 0; j < amps.length; j++) y += amps[j] * Math.sin((TWO_PI * (j + 1) * tt) / per + ph[j]);
          out[i] = y;
        }
        st.t += n;
      }, clone: cloneState };
  }

  function arPiece(P, nPaths1) {
    var rho = P.params.rho, sig = P.params.sigma;
    return { warmup: 0,
      init: function (rng) { return { x: rng.normal() * (sig / Math.sqrt(Math.max(1e-6, 1 - rho * rho))) }; },
      step: function (st, n, rng, out) {
        var x = st.x;
        for (var i = 0; i < n; i++) { x = rho * x + rng.normal() * sig; out[i] = x; }
        st.x = x;
      }, clone: cloneState };
  }

  /* RK4 + Euler-Maruyama noise on every state variable, observed every k internal steps (chaos._Integrated) */
  function integratedPiece(P, kind) {
    var p = P.params, dt = p.dt, k = p.obs_every, sig = p.dyn_rel * P.sd_ref * Math.sqrt(dt), obsSd = p.obs_rel * P.sd_ref;
    var D = kind === "kuramoto" ? p.N : 3, rhs, observe, init, wrap = null;
    if (kind === "lorenz63") {
      rhs = function (u, o) { o[0] = p.sigma * (u[1] - u[0]); o[1] = u[0] * (p.rho - u[2]) - u[1]; o[2] = u[0] * u[1] - p.beta * u[2]; };
      observe = function (u) { return u[0]; };
      init = function (rng) { return Float64Array.from([1 + rng.normal(), 1 + rng.normal(), 20 + rng.normal()]); };
    } else if (kind === "rossler") {
      rhs = function (u, o) { o[0] = -u[1] - u[2]; o[1] = u[0] + p.a * u[1]; o[2] = p.b + u[2] * (u[0] - p.c); };
      observe = function (u) { return u[0]; };
      init = function (rng) { return Float64Array.from([1 + 0.1 * rng.normal(), 1 + 0.1 * rng.normal(), 0.5 + 0.1 * rng.normal()]); };
    } else {
      var om = P.omega, K = p.K, N = p.N;
      rhs = function (u, o) {
        var ms = 0, mc = 0, i;
        for (i = 0; i < N; i++) { ms += Math.sin(u[i]); mc += Math.cos(u[i]); }
        ms /= N; mc /= N;
        for (i = 0; i < N; i++) o[i] = om[i] + K * (ms * Math.cos(u[i]) - mc * Math.sin(u[i]));
      };
      observe = function (u) { var s = 0; for (var i = 0; i < N; i++) s += Math.sin(u[i]); return s / N; };
      init = function (rng) { var u = new Float64Array(N); for (var i = 0; i < N; i++) u[i] = (2 * rng.u() - 1) * Math.PI; return u; };
      wrap = function (u) { for (var i = 0; i < N; i++) u[i] = ((((u[i] + Math.PI) % TWO_PI) + TWO_PI) % TWO_PI) - Math.PI; };
    }
    var k1 = new Float64Array(D), k2 = new Float64Array(D), k3 = new Float64Array(D), k4 = new Float64Array(D), tmp = new Float64Array(D);
    function rk4(u) {
      var i;
      rhs(u, k1);
      for (i = 0; i < D; i++) tmp[i] = u[i] + 0.5 * dt * k1[i];
      rhs(tmp, k2);
      for (i = 0; i < D; i++) tmp[i] = u[i] + 0.5 * dt * k2[i];
      rhs(tmp, k3);
      for (i = 0; i < D; i++) tmp[i] = u[i] + dt * k3[i];
      rhs(tmp, k4);
      for (i = 0; i < D; i++) u[i] += (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }
    return { warmup: P.warmup_steps,
      init: function (rng) { return { u: init(rng) }; },
      step: function (st, n, rng, out) {
        var u = st.u, total = n * k, i;
        for (var j = 0; j < total; j++) {
          rk4(u);
          for (i = 0; i < D; i++) u[i] += sig * rng.normal();
          if ((j + 1) % k === 0) out[(j + 1) / k - 1] = observe(u);
        }
        if (wrap) wrap(u);
        for (var o = 0; o < n; o++) out[o] += obsSd * rng.normal();
      }, clone: cloneState };
  }

  /* Heun's method on the delay equation, history of the last tau/h + 1 internal values (chaos.MackeyGlass) */
  function mackeyGlassPiece(P) {
    var p = P.params, h = p.h, k = p.obs_every, ls = Math.round(p.tau / h) + 1;
    var sig = p.dyn_rel * P.sd_ref * Math.sqrt(h), obsSd = p.obs_rel * P.sd_ref;
    return { warmup: P.warmup_steps,
      init: function (rng) { var hist = new Float64Array(ls); for (var i = 0; i < ls; i++) hist[i] = 1.2 + 0.05 * rng.normal(); return { hist: hist }; },
      step: function (st, n, rng, out) {
        var total = n * k, work = new Float64Array(ls + total);
        work.set(st.hist, 0);
        for (var j = 0; j < total; j++) {
          var x = work[ls - 1 + j], d0 = work[j], d1 = work[j + 1];
          var f0 = (p.beta * d0) / (1 + Math.pow(d0, p.n_exp)) - p.gamma * x;
          var f1 = (p.beta * d1) / (1 + Math.pow(d1, p.n_exp)) - p.gamma * (x + h * f0);
          work[ls + j] = x + 0.5 * h * (f0 + f1) + sig * rng.normal();
        }
        for (var i = 0; i < n; i++) out[i] = work[ls - 1 + k * (i + 1)] + obsSd * rng.normal();
        st.hist = work.slice(work.length - ls);
      }, clone: cloneState };
  }

  /* level or variance jump with the switch d steps before the end of the context (regime.py, placement "context") */
  function regimePiece(P) {
    var p = P.params, amps = P.amps, ph = P.phases, rho = p.rho, d = p.d, inc = TWO_PI / p.period;
    if (p.placement !== "context") throw new Error("only the context placement is exported");
    return { warmup: 0,
      init: function (rng) { return { t: 0, origin: -1, x: 0, phase: 0, u0: rng.normal() }; },
      step: function (st, n, rng, out) {
        var first = false;
        if (st.origin < 0) { st.origin = n; first = true; }
        var cut = st.origin - d;
        if (first) {
          var r0 = st.t >= cut ? 1 : 0;
          st.x = (st.u0 * p.sigma * (r0 ? P.sigma_mult : 1)) / Math.sqrt(1 - rho * rho);
        }
        var x = st.x, phase = st.phase;
        for (var i = 0; i < n; i++) {
          var R = st.t + i >= cut ? 1 : 0;
          x = rho * x + rng.normal() * p.sigma * (R ? P.sigma_mult : 1);
          phase += inc;
          var s = 0;
          for (var j = 0; j < amps.length; j++) s += amps[j] * Math.sin((j + 1) * phase + ph[j]);
          out[i] = p.level + P.level_shift * R + s + x;
        }
        st.t += n; st.x = x; st.phase = phase;
      }, clone: cloneState };
  }

  /* symmetric alpha-stable noise by Chambers-Mallows-Stuck, AR(1) filter (tails.StableAR) */
  function stableNoise(rng, alpha) {
    var u = (rng.u() - 0.5) * Math.PI;
    if (alpha === 1) return Math.tan(u);
    var w = rng.exp();
    return (Math.sin(alpha * u) / Math.pow(Math.cos(u), 1 / alpha)) * Math.pow(Math.cos((1 - alpha) * u) / w, (1 - alpha) / alpha);
  }
  function stableArPiece(P) {
    var p = P.params;
    return { warmup: P.warmup_steps,
      init: function () { return { x: 0 }; },
      step: function (st, n, rng, out) {
        var x = st.x;
        for (var i = 0; i < n; i++) { x = p.rho * x + p.sigma * stableNoise(rng, p.alpha); out[i] = x; }
        st.x = x;
      }, clone: cloneState };
  }

  /* discrete-time Hawkes counts (tails.Hawkes) */
  function hawkesPiece(P) {
    var p = P.params, d = Math.exp(-1 / p.tau), mu = p.rate * (1 - p.eta);
    return { warmup: P.warmup_steps,
      init: function () { return { s: p.eta * p.rate }; },
      step: function (st, n, rng, out) {
        var s = st.s;
        for (var i = 0; i < n; i++) { var c = rng.poisson(mu + s); out[i] = c; s = d * s + p.eta * (1 - d) * c; }
        st.s = s;
      }, clone: cloneState };
  }

  function wildPiece(name, P) {
    if (name === "lorenz63" || name === "rossler" || name === "kuramoto") return integratedPiece(P, name);
    if (name === "mackey_glass") return mackeyGlassPiece(P);
    if (name === "mean_jump" || name === "variance_jump") return regimePiece(P);
    if (name === "stable_ar") return stableArPiece(P);
    if (name === "hawkes") return hawkesPiece(P);
    throw new Error("unknown wild piece " + name);
  }

  /* ---------- mixture ---------- */
  /* controls: { season, wild, strength, noise } as in the toy; spec: live_spec.json */
  function build(spec, c) {
    var slots = [
      { w: c.season, P: spec.pieces.season, proc: seasonPiece(spec.pieces.season) },
      { w: spec.ar_noise_weight, P: spec.pieces.ar_noise, proc: arPiece(spec.pieces.ar_noise) },
    ];
    if (c.wild && c.wild !== "none") slots.push({ w: c.strength, P: spec.wild[c.wild], proc: wildPiece(c.wild, spec.wild[c.wild]) });
    var best = 1, bestW = 0.1;           // MixtureProcess.period
    slots.forEach(function (s) { if (s.w > bestW && s.P.period > 1) { best = s.P.period; bestW = s.w; } });
    return { slots: slots, noiseSd: c.noise, period: best };
  }

  function initStates(mix, rng) {
    return mix.slots.map(function (s) {
      var st = s.proc.init(rng);
      if (s.proc.warmup) { var tmp = new Float64Array(s.proc.warmup); s.proc.step(st, s.proc.warmup, rng, tmp); }
      return st;
    });
  }

  /* advance every slot n steps and combine: returns Float64Array(n) */
  function stepMix(mix, states, n, rng) {
    var out = new Float64Array(n), buf = new Float64Array(n);
    mix.slots.forEach(function (s, i) {
      s.proc.step(states[i], n, rng, buf);
      var mu = s.P.pilot.mean, sd = s.P.pilot.sd;
      for (var t = 0; t < n; t++) out[t] += (s.w * (buf[t] - mu)) / sd;
    });
    for (var t2 = 0; t2 < n; t2++) out[t2] += mix.noiseSd * rng.normal();
    return out;
  }

  function quantile(sorted, q) {
    var pos = q * (sorted.length - 1), lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  /* one sample: context (T), the realized future (H) and the referee's 21 quantiles per step (H x 21, row-major) */
  function makeSample(spec, controls, seed, opts) {
    opts = opts || {};
    var T = opts.T || spec.T, H = opts.H || spec.H, nPaths = opts.nPaths || 200;
    var rng = new Rng(seed), mix = build(spec, controls);
    var states = initStates(mix, rng);
    var ctx = stepMix(mix, states, T, rng);
    var paths = [], bad = 0;
    var real = null;
    for (var p = 0; p <= nPaths; p++) {
      var st = mix.slots.map(function (s, i) { return s.proc.clone(states[i]); });
      var f = stepMix(mix, st, H, rng);
      if (p === 0) { real = f; continue; }
      var ok = true;
      for (var t = 0; t < H; t++) if (!isFinite(f[t])) { ok = false; break; }
      if (ok) paths.push(f); else bad++;
    }
    var oracle = new Float64Array(H * 21), col = new Float64Array(paths.length);
    for (var s = 0; s < H; s++) {
      for (var i = 0; i < paths.length; i++) col[i] = paths[i][s];
      var sorted = Float64Array.from(col).sort();
      for (var q = 0; q < 21; q++) oracle[s * 21 + q] = quantile(sorted, LEVELS[q]);
    }
    return { context: ctx, future: real, oracle: oracle, period: mix.period, nonfiniteOracle: bad, T: T, H: H };
  }

  /* ---------- metrics.py: MASE ---------- */
  function naiveScale(ctx, m) {
    var T = ctx.length;
    if (m < 1 || m >= Math.floor(T / 2)) m = 1;
    var s = 0;
    for (var i = m; i < T; i++) s += Math.abs(ctx[i] - ctx[i - m]);
    return Math.max(s / (T - m), 1e-8);
  }
  function mase(ctx, fut, q, period) {
    var H = fut.length, e = 0;
    for (var t = 0; t < H; t++) e += Math.abs(fut[t] - q[t * 21 + MEDIAN]);
    return e / H / naiveScale(ctx, period);
  }

  return { Rng: Rng, build: build, makeSample: makeSample, mase: mase, naiveScale: naiveScale, LEVELS: LEVELS,
           pieces: { seasonPiece: seasonPiece, arPiece: arPiece, wildPiece: wildPiece, initStates: initStates, stepMix: stepMix } };
});
