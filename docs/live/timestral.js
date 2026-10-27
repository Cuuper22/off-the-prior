/* Timestral-5M forward pass in plain JavaScript (no libraries), for the live toy.

   It mirrors the Python package line by line for the case the toy needs: one series, no covariates, no missing
   values, a context that is a whole number of 32-step patches, a horizon of up to 1,024 steps (one decode block).
     scaler.py      causal per-patch mean/std, asinh squash, inverse with sinh
     model.py       patch embedding, forecast-query tokens, monotone quantile head
     layers.py      RMSNorm, gated attention with partial RoPE, SwiGLU, the time / variate layer schedule
     forecaster.py  refine frame for horizons over one patch, symmetric averaging, non-negative clip, context cap
   tests/test_site_parity.py runs this file under node against otp.model.forecast.

   Weights come from scripts/export_site_weights.py (timestral5m.json + timestral5m.f16 or .f32).
   Works in a browser (self.OTPTimestral) and in node (require). */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.OTPTimestral = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  var PS = 32;
  var EPS32 = 1.1920928955078125e-7;
  var FMAX32 = 3.4028234663852886e38;
  var fr = Math.fround;

  /* ---------- weights ---------- */
  function halfToFloat(h) {
    var s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * f * Math.pow(2, -24);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * (1 + f / 1024) * Math.pow(2, e - 15);
  }

  /* table: parsed timestral5m.json; buf: ArrayBuffer of the weights file */
  function load(table, buf) {
    var n = table.elements, all;
    if (table.dtype === "f32") all = new Float32Array(buf);
    else {
      var u = new Uint16Array(buf), lut = new Float32Array(65536);
      for (var k = 0; k < 65536; k++) lut[k] = halfToFloat(k);
      all = new Float32Array(n);
      for (var i = 0; i < n; i++) all[i] = lut[u[i]];
    }
    if (all.length !== n) throw new Error("weights file has " + all.length + " values, expected " + n);
    var W = {};
    table.tensors.forEach(function (t) {
      var size = 1;
      t.shape.forEach(function (s) { size *= s; });
      W[t.name] = all.subarray(t.offset, t.offset + size);
    });
    return { cfg: table.config, W: W, name: table.model, dtype: table.dtype };
  }

  /* ---------- small ops (token-major arrays, Float64 activations) ---------- */
  function linear(x, n, din, w, dout, b) {
    var y = new Float64Array(n * dout);
    for (var t = 0; t < n; t++) {
      var xo = t * din, yo = t * dout;
      for (var o = 0; o < dout; o++) {
        var wo = o * din, acc = b ? b[o] : 0;
        for (var i = 0; i < din; i++) acc += w[wo + i] * x[xo + i];
        y[yo + o] = acc;
      }
    }
    return y;
  }
  function silu(v) { return v / (1 + Math.exp(-v)); }
  function sigmoid(v) { return 1 / (1 + Math.exp(-v)); }
  function rmsnorm(x, n, d, scale, eps) {
    var y = new Float64Array(n * d);
    for (var t = 0; t < n; t++) {
      var s = 0, o = t * d;
      for (var i = 0; i < d; i++) s += x[o + i] * x[o + i];
      var r = 1 / Math.sqrt(s / d + eps);
      for (var j = 0; j < d; j++) y[o + j] = x[o + j] * r * scale[j];
    }
    return y;
  }
  /* output(silu(hidden(x))) + residual(x) */
  function residualBlock(x, n, din, dhid, dout, W, prefix, bias) {
    var h = linear(x, n, din, W[prefix + ".hidden_layer.weight"], dhid, bias ? W[prefix + ".hidden_layer.bias"] : null);
    for (var i = 0; i < h.length; i++) h[i] = silu(h[i]);
    var o = linear(h, n, dhid, W[prefix + ".output_layer.weight"], dout, bias ? W[prefix + ".output_layer.bias"] : null);
    var r = linear(x, n, din, W[prefix + ".residual_layer.weight"], dout, bias ? W[prefix + ".residual_layer.bias"] : null);
    for (var k = 0; k < o.length; k++) o[k] += r[k];
    return o;
  }

  /* ---------- scaler.py ---------- */
  /* causal running mean and std (float64), read at the last step of every patch, then stored as float32 */
  function patchStats(x, minScale) {
    var T = x.length, L = T / PS, loc = new Float64Array(L), scale = new Float64Array(L);
    var cumX = 0, cumW = 0, m2 = 0, prevMean = 0;
    for (var t = 0; t < T; t++) {
      cumW += 1; cumX += x[t];
      var mean = cumX / Math.max(cumW, 1);
      m2 += (x[t] - prevMean) * (x[t] - mean);
      prevMean = mean;
      if ((t + 1) % PS === 0) {
        var p = (t + 1) / PS - 1;
        loc[p] = fr(mean);
        scale[p] = fr(Math.max(Math.sqrt(m2 / Math.max(Math.max(cumW, 1) - 1, 1)), minScale));
      }
    }
    return { loc: loc, scale: scale };
  }

  /* ---------- rope ---------- */
  function ropeTables(cfg, n) {
    var hd = cfg.d_model / cfg.num_heads, rot = Math.floor(hd * cfg.rope_partial_factor), half = hd / 2;
    var inv = new Float64Array(half);
    for (var k = 0; k < rot / 2; k++) inv[k] = 1 / Math.pow(cfg.rope_theta, (2 * k) / rot);
    var cos = new Float64Array(n * hd), sin = new Float64Array(n * hd);
    for (var p = 0; p < n; p++)
      for (var d = 0; d < hd; d++) {
        var f = fr(p * inv[d % half]);
        cos[p * hd + d] = Math.cos(f);
        sin[p * hd + d] = Math.sin(f);
      }
    return { cos: cos, sin: sin, hd: hd };
  }

  /* ---------- attention (layers.py) ---------- */
  function gateAndOut(att, xn, n, D, A) {
    var g = linear(xn, n, D, A.gate, D, null);
    for (var i = 0; i < att.length; i++) att[i] *= sigmoid(g[i]);
    return linear(att, n, D, A.wO, D, null);
  }

  function timeAttention(xn, n, cfg, A, rope) {
    var D = cfg.d_model, nh = cfg.num_heads, hd = D / nh;
    var P = linear(xn, n, D, A.wQKV, 3 * D, null);
    // layout of the fused projection: (qkv, head_dim, n_heads)
    var q = new Float64Array(n * nh * hd), k = new Float64Array(n * nh * hd), v = new Float64Array(n * nh * hd);
    var half = hd / 2;
    for (var t = 0; t < n; t++)
      for (var h = 0; h < nh; h++) {
        var base = (t * nh + h) * hd;
        for (var d = 0; d < hd; d++) {
          var c = rope.cos[t * hd + d], s = rope.sin[t * hd + d];
          var d2 = d < half ? d + half : d - half, sign = d < half ? -1 : 1;
          var qd = P[t * 3 * D + d * nh + h], qr = P[t * 3 * D + d2 * nh + h];
          var kd = P[t * 3 * D + D + d * nh + h], kr = P[t * 3 * D + D + d2 * nh + h];
          q[base + d] = qd * c + sign * qr * s;
          k[base + d] = kd * c + sign * kr * s;
          v[base + d] = P[t * 3 * D + 2 * D + d * nh + h];
        }
      }
    var att = new Float64Array(n * D), sc = 1 / Math.sqrt(hd), w = new Float64Array(n);
    for (var t2 = 0; t2 < n; t2++)
      for (var h2 = 0; h2 < nh; h2++) {
        // every token reads the tokens at or before it: causal over context, then the decoder chain
        var mx = -Infinity, qb = (t2 * nh + h2) * hd;
        for (var j = 0; j <= t2; j++) {
          var kb = (j * nh + h2) * hd, dot = 0;
          for (var d3 = 0; d3 < hd; d3++) dot += q[qb + d3] * k[kb + d3];
          w[j] = dot * sc;
          if (w[j] > mx) mx = w[j];
        }
        var z = 0;
        for (var j2 = 0; j2 <= t2; j2++) { w[j2] = Math.exp(w[j2] - mx); z += w[j2]; }
        for (var d4 = 0; d4 < hd; d4++) {
          var acc = 0;
          for (var j3 = 0; j3 <= t2; j3++) acc += w[j3] * v[(j3 * nh + h2) * hd + d4];
          att[t2 * D + h2 * hd + d4] = acc / z;
        }
      }
    return gateAndOut(att, xn, n, D, A);
  }

  /* one row per series: the variate axis has a single entry, so softmax gives weight 1 and the output is v */
  function variateAttention(xn, n, cfg, A) {
    var D = cfg.d_model, nh = cfg.num_heads, hd = D / nh;
    var P = linear(xn, n, D, A.wQKV, 3 * D, null);
    var att = new Float64Array(n * D);
    for (var t = 0; t < n; t++)
      for (var h = 0; h < nh; h++)
        for (var d = 0; d < hd; d++) att[t * D + h * hd + d] = P[t * 3 * D + 2 * D + d * nh + h];
    return gateAndOut(att, xn, n, D, A);
  }

  var AXES = { time: [true], variate: [false], time_variate: [true, false], variate_time: [false, true] };

  function transformer(M, x, n) {
    var cfg = M.cfg, W = M.W, D = cfg.d_model;
    var rope = ropeTables(cfg, n);
    cfg.block_schedule.forEach(function (kind, li) {
      var pre = "transformer.layers." + li;
      AXES[kind].forEach(function (isTime, ai) {
        var A = { wQKV: W[pre + ".attns." + ai + ".wQKV.weight"], wO: W[pre + ".attns." + ai + ".wO.weight"],
                  gate: W[pre + ".attns." + ai + "._gate.weight"] };
        var xn = rmsnorm(x, n, D, W[pre + ".attn_norms." + ai + ".scale"], cfg.norm_eps);
        var a = isTime ? timeAttention(xn, n, cfg, A, rope) : variateAttention(xn, n, cfg, A);
        for (var i = 0; i < x.length; i++) x[i] += a[i];
      });
      var xm = rmsnorm(x, n, D, W[pre + ".mlp_norm.scale"], cfg.norm_eps);
      var hid = cfg.mlp_hidden_dim;
      var f = linear(xm, n, D, W[pre + ".mlp.w12.weight"], 2 * hid, null);
      var g = new Float64Array(n * hid);
      for (var t = 0; t < n; t++)
        for (var j = 0; j < hid; j++) g[t * hid + j] = f[t * 2 * hid + j] * silu(f[t * 2 * hid + hid + j]);
      var m = linear(g, n, hid, W[pre + ".mlp.w3.weight"], D, null);
      for (var k = 0; k < x.length; k++) x[k] += m[k];
    });
    return rmsnorm(x, n, D, W["transformer.out_norm.scale"], cfg.norm_eps);
  }

  /* ---------- model.py: one decoded block ---------- */
  function softplus(v) { return v > 20 ? v : Math.log1p(Math.exp(v)); }

  function monotone(raw, levels, widthFactor) {
    var Q = levels.length, m = levels.indexOf(0.5), inc = new Float64Array(Q), out = new Float64Array(Q);
    for (var i = m + 1; i < Q; i++) inc[i] = widthFactor * (levels[i] - levels[i - 1]) * softplus(raw[i]);
    for (var j = 0; j < m; j++) inc[j] = widthFactor * (levels[j + 1] - levels[j]) * softplus(raw[j]);
    out[m] = raw[m];
    var up = 0;
    for (var a = m + 1; a < Q; a++) { up += inc[a]; out[a] = raw[m] + up; }
    var down = 0;
    for (var b = m - 1; b >= 0; b--) { down += inc[b]; out[b] = raw[m] - down; }
    return out;
  }

  function clampNonfinite(row) {
    var hi = -Infinity, lo = Infinity, bad = false;
    for (var i = 0; i < row.length; i++) {
      if (isFinite(row[i])) { if (row[i] > hi) hi = row[i]; if (row[i] < lo) lo = row[i]; } else bad = true;
    }
    if (!bad) return;
    for (var j = 0; j < row.length; j++) {
      if (row[j] === Infinity) row[j] = isFinite(hi) ? hi : 0;
      else if (row[j] === -Infinity) row[j] = isFinite(lo) ? lo : 0;
      else if (!isFinite(row[j]) && row[j] !== row[j]) continue;
    }
  }

  /* quantiles (H*32 x Q, row-major) in the anchor frame of the last context patch, raw units */
  function predictBlock(M, x, hPatches, levels) {
    var cfg = M.cfg, W = M.W, D = cfg.d_model, Q = levels.length;
    var T = x.length, L = T / PS, n = L + hPatches;
    var st = patchStats(x, cfg.minimum_scale);
    // embedding of [scaled values, missing flags]
    var emb = new Float64Array(L * 2 * PS);
    for (var p = 0; p < L; p++)
      for (var s = 0; s < PS; s++) {
        var xv = x[p * PS + s];
        emb[p * 2 * PS + s] = Math.asinh((xv - st.loc[p]) / Math.max(st.scale[p], EPS32));
      }
    var enc = residualBlock(emb, L, 2 * PS, 4 * D, D, W, "representation.patch_proj", true);
    var seq = new Float64Array(n * D);
    seq.set(enc, 0);
    var fq = W["representation.forecast_query"];
    for (var h = 0; h < hPatches; h++) for (var d = 0; d < D; d++) seq[(L + h) * D + d] = fq[d];
    var out = transformer(M, seq, n);
    var dec = out.subarray(L * D);
    var raw = residualBlock(dec, hPatches, D, 4 * D, PS * Q, W, "head.projection", false);
    var aLoc = st.loc[L - 1], aScale = st.scale[L - 1];
    var res = new Float64Array(hPatches * PS * Q), row = new Float64Array(Q);
    for (var step = 0; step < hPatches * PS; step++) {
      var mq = monotone(raw.subarray(step * Q, step * Q + Q), levels, cfg.quantile_width_factor);
      for (var q = 0; q < Q; q++) row[q] = Math.sinh(fr(mq[q])) * aScale + aLoc;
      clampNonfinite(row);
      for (var q2 = 0; q2 < Q; q2++) res[step * Q + q2] = fr(row[q2]);
    }
    return { q: res, loc0: aLoc, scale0: aScale };
  }

  /* ---------- forecaster.py: refine frame for blocks of more than one patch ---------- */
  function refine(M, pred, x, levels, nPred) {
    var cfg = M.cfg, Q = levels.length, mi = levels.indexOf(0.5);
    if (nPred <= 1) return pred.q;
    var L = x.length / PS, loc0 = pred.loc0, scale0 = pred.scale0;
    // sinh of the median in the anchor frame, per horizon patch
    var sinhMed = new Float64Array(nPred * PS);
    for (var i = 0; i < nPred * PS; i++) sinhMed[i] = (pred.q[i * Q + mi] - loc0) / scale0;
    // Welford state of the observed context (count, mean, m2), patch by patch
    var cnt = 0, mean = 0, m2 = 0;
    for (var p = 0; p < L; p++) {
      var pm = 0;
      for (var s = 0; s < PS; s++) pm += x[p * PS + s];
      pm /= PS;
      var pm2 = 0;
      for (var s2 = 0; s2 < PS; s2++) pm2 += (x[p * PS + s2] - pm) * (x[p * PS + s2] - pm);
      var w = cnt + PS, ws = Math.max(w, Number.MIN_VALUE), delta = pm - mean;
      var nm = mean + (delta * PS) / ws;
      m2 = m2 + pm2 + (delta * delta * cnt * PS) / ws;
      mean = nm; cnt = w;
    }
    var offset = mean, clip = Math.sqrt(FMAX32) / 1e3;
    var locs = new Float64Array(nPred), scales = new Float64Array(nPred);
    for (var h = 0; h < nPred; h++) {
      var variance = Math.max(m2 / Math.max(cnt - 1, 1), 0);
      var scale = Math.max(Math.sqrt(variance), cfg.minimum_scale);
      locs[h] = mean; scales[h] = scale;
      var bm = 0, vals = new Float64Array(PS);
      for (var k = 0; k < PS; k++) {
        var f = sinhMed[h * PS + k] * scale + mean - offset;
        f = Math.min(Math.max(f, -clip), clip) + offset;
        vals[k] = f; bm += f;
      }
      bm /= PS;
      var bm2 = 0;
      for (var k2 = 0; k2 < PS; k2++) bm2 += (vals[k2] - bm) * (vals[k2] - bm);
      var wt = cnt + PS, wts = Math.max(wt, Number.MIN_VALUE), dl = bm - mean;
      var mean2 = mean + (dl * PS) / wts;
      m2 = m2 + bm2 + (dl * dl * cnt * PS) / wts;
      mean = mean2; cnt = wt;
    }
    var out = new Float64Array(pred.q.length), row = new Float64Array(Q);
    for (var step = 0; step < nPred * PS; step++) {
      var hp = Math.floor(step / PS);
      var lj = hp === 0 ? loc0 : fr(locs[hp]), sj = hp === 0 ? scale0 : fr(scales[hp]);
      for (var q = 0; q < Q; q++) row[q] = ((pred.q[step * Q + q] - loc0) / scale0) * sj + lj;
      clampNonfinite(row);
      for (var q2 = 0; q2 < Q; q2++) out[step * Q + q2] = fr(row[q2]);
    }
    return out;
  }

  /* one forecast pass: (H x Q) quantiles in raw units, with the cap of forecaster._cap_quantiles */
  function forecastRaw(M, x, horizon, levels) {
    var nPatches = Math.ceil(horizon / PS);
    if (nPatches > M.cfg.block_size) throw new Error("live forecast covers one decode block (at most " + M.cfg.block_size * PS + " steps)");
    var pred = predictBlock(M, x, nPatches, levels);
    var q = refine(M, pred, x, levels, nPatches);
    var Q = levels.length, k = 1e4, mn = Infinity, mx = -Infinity;
    for (var i = 0; i < x.length; i++) { if (x[i] < mn) mn = x[i]; if (x[i] > mx) mx = x[i]; }
    var lastScale = patchStats(x, M.cfg.minimum_scale).scale;
    var cap = lastScale[lastScale.length - 1] * k;
    var out = new Float64Array(horizon * Q);
    for (var j = 0; j < horizon * Q; j++) out[j] = Math.min(Math.max(q[j], mn - cap), mx + cap);
    return out;
  }

  /* public: series (array-like, length a multiple of 32, finite), horizon, options.
     Returns { q: Float64Array(horizon * 21) row-major, levels }. */
  function forecast(M, series, horizon, opts) {
    opts = opts || {};
    var levels = M.cfg.quantile_knots, Q = levels.length;
    var sym = opts.symmetric !== false, clipNonneg = opts.clipNonnegative !== false;
    if (series.length % PS !== 0) throw new Error("context length must be a multiple of " + PS);
    var x = new Float64Array(series.length), anyNeg = false;
    for (var i = 0; i < x.length; i++) {
      if (!isFinite(series[i])) throw new Error("context holds a value that is not finite");
      x[i] = fr(series[i]);
      if (x[i] < 0) anyNeg = true;
    }
    var q = forecastRaw(M, x, horizon, levels);
    if (sym) {
      var xm = new Float64Array(x.length);
      for (var j = 0; j < x.length; j++) xm[j] = -x[j];
      var m = forecastRaw(M, xm, horizon, levels), avg = new Float64Array(q.length);
      for (var s = 0; s < horizon; s++)
        for (var k = 0; k < Q; k++) avg[s * Q + k] = fr(0.5 * (q[s * Q + k] + -m[s * Q + (Q - 1 - k)]));
      q = avg;
    }
    if (clipNonneg && !anyNeg) for (var z = 0; z < q.length; z++) if (q[z] < 0) q[z] = 0;
    return { q: q, levels: levels.slice() };
  }

  return { load: load, forecast: forecast, PATCH: PS };
});
