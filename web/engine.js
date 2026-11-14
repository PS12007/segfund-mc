/* segfund-mc, browser engine.
 *
 * A line-by-line port of the NumPy fp64 reference (src/segfund/reference/engine.py)
 * for a single-deposit contract without resets: Hull-White rates fitted to a
 * Nelson-Siegel curve, a correlated equity fund, Gompertz-Makeham mortality and
 * dynamic lapse. Random numbers are the same Philox4x32-10 stream as the Python
 * and Triton engines, so path 17 at step 40 is the same future in all three.
 * tests/test_web_engine.py checks this file against the Python reference.
 *
 * Educational model; every assumption is illustrative.
 *
 * The whole engine is one self-contained function, so the page can also start a
 * Web Worker from segfundEngine.toString() (that works even from file://).
 */
function segfundEngine() {
  'use strict';

  // ---------------------------------------------------------------- Philox4x32-10
  const M_A = 0xD2511F53, M_B = 0xCD9E8D57, W_A = 0x9E3779B9, W_B = 0xBB67AE85;
  const U23 = Math.pow(2, -23);
  const TWO32 = 4294967296;

  // high 32 bits of m * b for m, b < 2^32, from 16-bit limbs (every partial product < 2^32)
  const MA0 = M_A & 0xffff, MA1 = M_A >>> 16, MB0 = M_B & 0xffff, MB1 = M_B >>> 16;
  function mulhi(m0, m1, b) {
    const b0 = b & 0xffff, b1 = b >>> 16;
    const p00 = m0 * b0, p01 = m0 * b1, p10 = m1 * b0;
    const mid = (p00 >>> 16) + (p01 & 0xffff) + (p10 & 0xffff);
    return (m1 * b1 + (p01 >>> 16) + (p10 >>> 16) + (mid >>> 16)) >>> 0;
  }

  const _w = new Float64Array(4);
  function philox(c0, c1, c2, c3, k0, k1) {
    for (let i = 0; i < 10; i++) {
      const hiB = mulhi(MB0, MB1, c2), loB = Math.imul(M_B, c2) >>> 0;
      const hiA = mulhi(MA0, MA1, c0), loA = Math.imul(M_A, c0) >>> 0;
      c0 = (hiB ^ c1 ^ k0) >>> 0;
      c1 = loB;
      c2 = (hiA ^ c3 ^ k1) >>> 0;
      c3 = loA;
      k0 = (k0 + W_A) >>> 0;
      k1 = (k1 + W_B) >>> 0;
    }
    _w[0] = c0; _w[1] = c1; _w[2] = c2; _w[3] = c3;
    return _w;
  }

  const _z = new Float64Array(3);
  // three standard normals for (path counter, step, stream), Box-Muller exactly as rng/philox.py
  function normals3(ctr, step, stream, k0, k1) {
    const w = philox(ctr >>> 0, step >>> 0, stream >>> 0, 0, k0, k1);
    const u0 = ((w[0] >>> 9) + 0.5) * U23, u1 = ((w[1] >>> 9) + 0.5) * U23;
    const u2 = ((w[2] >>> 9) + 0.5) * U23, u3 = ((w[3] >>> 9) + 0.5) * U23;
    const rad = Math.sqrt(-2.0 * Math.log(u0)), ang = Math.PI * (2.0 * u1 - 1.0);
    const rad2 = Math.sqrt(-2.0 * Math.log(u2)), ang2 = Math.PI * (2.0 * u3 - 1.0);
    _z[0] = -rad * Math.cos(ang);
    _z[1] = -rad * Math.sin(ang);
    _z[2] = -rad2 * Math.cos(ang2);
    return _z;
  }

  function seedKey(seed) {
    return [seed % TWO32 >>> 0, Math.floor(seed / TWO32) >>> 0];
  }

  // ---------------------------------------------------------------- model pieces
  const BASE = Object.freeze({
    age: 55, T: 10, dt: 1 / 12, deposit: 100000,
    gmmb: 0.75, gmdb: 1.0, mer: 0.0225, fee: 0.0040,
    curve: { b0: 0.035, b1: -0.010, b2: 0.005, tau: 2.0 },
    a: 0.05, sigmaR: 0.010, lambdaR: 0.0,
    sigmaS: 0.18, rho: -0.20, erp: 0.04,
    mort: { A: 0.00022, B: 2.7e-6, c: 1.124 },
    lapse: { base: 0.05, k: 1.0, lo: 0.2, hi: 2.0 },
  });

  function zeroRate(cv, t) {
    const x = t / cv.tau;
    const g = x < 1e-8 ? 1.0 - x / 2.0 : -Math.expm1(-x) / x;
    return cv.b0 + cv.b1 * g + cv.b2 * (g - Math.exp(-x));
  }
  const logDiscount = (cv, t) => -zeroRate(cv, t) * t;
  const Bfn = (a, tau) => -Math.expm1(-a * tau) / a;
  const varIntX = (a, s, tau) => (s * s / (a * a)) * (tau - 2.0 * Bfn(a, tau) + Bfn(2.0 * a, tau));

  function choleskyPsd(C) {
    const L = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let j = 0; j < 3; j++) {
      let d = C[j][j];
      for (let m = 0; m < j; m++) d -= L[j][m] * L[j][m];
      if (d <= 1e-300) continue;
      L[j][j] = Math.sqrt(d);
      for (let i = j + 1; i < 3; i++) {
        let s = C[i][j];
        for (let m = 0; m < j; m++) s -= L[i][m] * L[j][m];
        L[i][j] = s / L[j][j];
      }
    }
    return L;
  }

  // Per-step table (risk-neutral or real-world), same columns as gpu/table.py
  function buildTable(p, measure) {
    const rw = measure === 'real_world';
    const n = Math.round(p.T / p.dt);
    const times = new Float64Array(n + 1);
    const step = p.T / n;
    for (let k = 0; k <= n; k++) times[k] = k * step;
    times[n] = p.T;
    const a = p.a, sr = p.sigmaR, ss = p.sigmaS, rho = p.rho;
    const erp = rw ? p.erp : 0.0, lam = rw ? p.lambdaR : 0.0;
    const cols = ['decay', 'Bh', 'phi', 'xs', 'ys', 'lnF', 'L00', 'L10', 'L11', 'L20', 'L21', 'L22', 'h', 'q'];
    const tab = {};
    for (const c of cols) tab[c] = new Float64Array(n);
    const cache = new Map();
    const lnc = Math.log(p.mort.c);
    for (let k = 0; k < n; k++) {
      const s = times[k], t = times[k + 1], h = t - s;
      const key = (Math.round(h * 1e14) / 1e14).toString();
      if (!cache.has(key)) {
        const decay = Math.exp(-a * h), b = Bfn(a, h);
        const v11 = sr * sr * Bfn(2 * a, h), v22 = varIntX(a, sr, h);
        const v12 = sr * sr / (2 * a * a) * Math.pow(Math.expm1(-a * h), 2);
        const c13 = rho * sr * ss * b, c23 = rho * sr * ss * (h - b) / a;
        cache.set(key, [decay, b, choleskyPsd([[v11, v12, c13], [v12, v22, c23], [c13, c23, ss * ss * h]])]);
      }
      const [decay, b, L] = cache.get(key);
      tab.decay[k] = decay; tab.Bh[k] = b;
      const conv = (t - s) - 2.0 * (Math.exp(-a * s) - Math.exp(-a * t)) / a
        + (Math.exp(-2 * a * s) - Math.exp(-2 * a * t)) / (2 * a);
      tab.phi[k] = logDiscount(p.curve, s) - logDiscount(p.curve, t) + sr * sr / (2 * a * a) * conv;
      tab.xs[k] = sr * lam * b;
      tab.ys[k] = sr * lam * (h - b) / a;
      tab.lnF[k] = (erp - p.mer - 0.5 * ss * ss) * h;
      tab.L00[k] = L[0][0]; tab.L10[k] = L[1][0]; tab.L11[k] = L[1][1];
      tab.L20[k] = L[2][0]; tab.L21[k] = L[2][1]; tab.L22[k] = L[2][2];
      tab.h[k] = h;
      const age = p.age + s;
      const cum = p.mort.A * h + p.mort.B * Math.pow(p.mort.c, age) * Math.expm1(h * lnc) / lnc;
      tab.q[k] = 1.0 - Math.exp(-cum);
    }
    return { n, times, tab, measure, params: p };
  }

  /* Simulate n paths over steps [kStart, kEnd).
   *   opts.seed, opts.stream (0 main, 1 inner), opts.antiHalf (antithetic block half-size, 0 = off)
   *   opts.init: start state {x, F, GM, GD, l} (scalars, shared by all paths) for nested inner runs
   *   opts.keep: record the fund value at every step for the first `keep` paths (Float32Array)
   *   opts.offset: first global path index
   */
  function simulate(model, n, opts) {
    const o = Object.assign({ seed: 1, stream: 0, antiHalf: 0, kStart: 0, kEnd: model.n, init: null, keep: 0, offset: 0, keepX: false }, opts);
    const p = model.params, tb = model.tab;
    const [k0, k1] = seedKey(o.seed);
    const hz = -Math.log1p(-p.lapse.base);
    const lk = p.lapse.k, llo = p.lapse.lo, lhi = p.lapse.hi;
    const fee = p.fee, final = o.kEnd === model.n;
    const nS = o.kEnd - o.kStart;
    const pvMat = new Float64Array(n), pvDeath = new Float64Array(n), pvFee = new Float64Array(n);
    const FT = new Float64Array(n), DT = new Float64Array(n), lT = new Float64Array(n), UT = new Float64Array(n);
    const xT = new Float64Array(n);
    const keep = Math.min(o.keep, n);
    const path = keep ? new Float32Array(keep * (nS + 1)) : null;
    const xpath = keep && o.keepX ? new Float32Array(keep * (nS + 1)) : null;
    for (let i = 0; i < n; i++) {
      const g = o.offset + i;
      let ctr = g, sgn = 1.0;
      if (o.antiHalf > 0 && (g % (2 * o.antiHalf)) >= o.antiHalf) { ctr = g - o.antiHalf; sgn = -1.0; }
      let x, F, GM, GD, l;
      if (o.init) { x = o.init.x; F = o.init.F; GM = o.init.GM; GD = o.init.GD; l = o.init.l; }
      else { x = 0.0; F = 0.0; GM = 0.0; GD = 0.0; l = 1.0; }
      let U = 1.0, D = 1.0, pd = 0.0, pf = 0.0;
      for (let k = o.kStart; k < o.kEnd; k++) {
        if (k === 0 && !o.init) {
          F += p.deposit; GM += p.gmmb * p.deposit; GD += p.gmdb * p.deposit;
        }
        if (i < keep) {
          path[i * (nS + 1) + (k - o.kStart)] = F;
          if (xpath) xpath[i * (nS + 1) + (k - o.kStart)] = x;
        }
        const h = tb.h[k];
        let mult = lk === 1.0 ? F / GM : Math.pow(F / GM, lk);
        mult = mult < llo ? llo : mult > lhi ? lhi : mult;
        const qL = -Math.expm1(-hz * mult * h);
        const z = normals3(ctr, k, o.stream, k0, k1);
        const z0 = z[0] * sgn, z1 = z[1] * sgn, z2 = z[2] * sgn;
        const e0 = tb.L00[k] * z0;
        const e1 = tb.L10[k] * z0 + tb.L11[k] * z1;
        const e2 = tb.L20[k] * z0 + tb.L21[k] * z1 + tb.L22[k] * z2;
        const intr = x * tb.Bh[k] + tb.ys[k] + e1 + tb.phi[k];
        x = x * tb.decay[k] + tb.xs[k] + e0;
        const growth = Math.exp(intr + tb.lnF[k] + e2);
        F = F * growth; U = U * growth; D = D * Math.exp(-intr);
        const q = tb.q[k];
        const deaths = l * q;
        pd += D * deaths * Math.max(GD - F, 0.0);
        pf += D * l * fee * h * F;
        l = l * (1.0 - q) * (1.0 - qL);
      }
      if (i < keep) {
        path[i * (nS + 1) + nS] = F;
        if (xpath) xpath[i * (nS + 1) + nS] = x;
      }
      pvMat[i] = final ? D * l * Math.max(GM - F, 0.0) : 0.0;
      pvDeath[i] = pd; pvFee[i] = pf; FT[i] = F; DT[i] = D; lT[i] = l; UT[i] = U; xT[i] = x;
    }
    const pvNet = new Float64Array(n);
    for (let i = 0; i < n; i++) pvNet[i] = pvMat[i] + pvDeath[i] - pvFee[i];
    return { n, nSteps: nS, pvNet, pvMat, pvDeath, pvFee, FT, DT, lT, UT, xT, path, xpath, keep };
  }

  // ---------------------------------------------------------------- closed forms and estimators
  // erfc to near machine precision: power series below 2.5, continued fraction above
  function erfc(x) {
    if (x < 0) return 2.0 - erfc(-x);
    if (x < 2.5) {
      // erf(x) = 2/sqrt(pi) e^{-x^2} sum_n 2^n x^{2n+1} / (1*3*...*(2n+1)), all terms positive
      let term = x, sum = x;
      for (let n = 1; n < 200; n++) {
        term *= 2.0 * x * x / (2 * n + 1);
        sum += term;
        if (term < 1e-17 * sum) break;
      }
      return 1.0 - 2.0 / Math.sqrt(Math.PI) * Math.exp(-x * x) * sum;
    }
    // Lentz continued fraction: erfc(x) = e^{-x^2}/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))
    const tiny = 1e-300;
    let f = x, C = x, Dd = 0.0;
    for (let i = 1; i < 300; i++) {
      const an = i / 2.0;
      Dd = x + an * Dd; Dd = Math.abs(Dd) < tiny ? tiny : Dd; Dd = 1.0 / Dd;
      C = x + an / C; C = Math.abs(C) < tiny ? tiny : C;
      const delta = C * Dd;
      f *= delta;
      if (Math.abs(delta - 1.0) < 1e-16) break;
    }
    return Math.exp(-x * x) / Math.sqrt(Math.PI) / f;
  }
  const normCdf = (x) => 0.5 * erfc(-x / Math.SQRT2);

  // European put on the fund under Hull-White + correlated GBM (reference/closed_form.py hw_gbm_put)
  function hwGbmPut(p, F0, K, T) {
    const a = p.a, s = p.sigmaR;
    const P = Math.exp(logDiscount(p.curve, T));
    const fwd = F0 * Math.exp(-p.mer * T) / P;
    const v2 = p.sigmaS * p.sigmaS * T + varIntX(a, s, T) + 2 * p.rho * s * p.sigmaS * (T - Bfn(a, T)) / a;
    if (v2 <= 0) return P * Math.max(K - fwd, 0);
    const v = Math.sqrt(v2);
    const d1 = (Math.log(fwd / K) + 0.5 * v2) / v, d2 = d1 - v;
    return P * (K * normCdf(-d2) - fwd * normCdf(-d1));
  }

  function meanSe(y) {
    let m = 0;
    for (let i = 0; i < y.length; i++) m += y[i];
    m /= y.length;
    let v = 0;
    for (let i = 0; i < y.length; i++) v += (y[i] - m) * (y[i] - m);
    return { mean: m, se: Math.sqrt(v / (y.length - 1) / y.length) };
  }

  // antithetic pairs + Hull-White put control variate (estimators.py), risk-neutral only
  function estimate(res, p, field, control) {
    const y0 = res[field], n = res.n, h = n / 2;
    const y = new Float64Array(h), xc = new Float64Array(h);
    const K = p.gmmb * p.deposit;
    for (let i = 0; i < h; i++) {
      y[i] = 0.5 * (y0[i] + y0[i + h]);
      const xa = res.DT[i] * Math.max(K - p.deposit * res.UT[i], 0.0);
      const xb = res.DT[i + h] * Math.max(K - p.deposit * res.UT[i + h], 0.0);
      xc[i] = 0.5 * (xa + xb);
    }
    if (!control) return meanSe(y);
    const EX = hwGbmPut(p, p.deposit, K, p.T);
    const my = meanSe(y).mean, mx = meanSe(xc).mean;
    let cxy = 0, cxx = 0;
    for (let i = 0; i < h; i++) { cxy += (y[i] - my) * (xc[i] - mx); cxx += (xc[i] - mx) * (xc[i] - mx); }
    const beta = cxx > 0 ? cxy / cxx : 0;
    for (let i = 0; i < h; i++) y[i] = y[i] - beta * (xc[i] - EX);
    return Object.assign(meanSe(y), { beta, EX });
  }

  // CTE with the Manistre-Hancock standard error (risk.py)
  function cte(losses, alpha) {
    const s = Float64Array.from(losses).sort();
    const n = s.length, k = Math.round(n * (1 - alpha));
    let m = 0;
    for (let i = n - k; i < n; i++) m += s[i];
    m /= k;
    const v = s[n - k];
    let vt = 0;
    for (let i = n - k; i < n; i++) vt += (s[i] - m) * (s[i] - m);
    vt /= (k - 1);
    return { alpha, cte: m, var: v, se: Math.sqrt((vt + alpha * (m - v) * (m - v)) / (n * (1 - alpha))), k, n };
  }

  // ---------------------------------------------------------------- one call for the page
  // Risk-neutral price (antithetic + CV) and real-world tail, for a parameter set.
  function value(p, opts) {
    const o = Object.assign({ nRN: 16384, nRW: 16384, seed: 20261113, keep: 0 }, opts);
    const rnModel = buildTable(p, 'risk_neutral');
    const rwModel = buildTable(p, 'real_world');
    const rn = simulate(rnModel, o.nRN, { seed: o.seed, antiHalf: o.nRN / 2 });
    const rw = simulate(rwModel, o.nRW, { seed: o.seed + 1, keep: o.keep });
    const net = estimate(rn, p, 'pvNet', true);
    const claimsArr = new Float64Array(rn.n);
    for (let i = 0; i < rn.n; i++) claimsArr[i] = rn.pvMat[i] + rn.pvDeath[i];
    const claims = estimate(Object.assign({}, rn, { claims: claimsArr }), p, 'claims', true);
    const fees = estimate(rn, p, 'pvFee', false);
    // fees are proportional to the guarantee fee (it is carved out of a fixed MER), so the
    // break-even fee solves fee * PVfees/fee = PVclaims in one step
    const breakeven = p.fee > 0 ? p.fee * claims.mean / fees.mean : NaN;
    let nLoss = 0;
    for (let i = 0; i < rw.n; i++) if (rw.pvNet[i] > 0) nLoss++;
    return {
      net, claims, fees, breakeven, rn, rw, rwModel, rnModel,
      cte95: cte(rw.pvNet, 0.95), cte70: cte(rw.pvNet, 0.70),
      pLoss: nLoss / rw.n, rwMean: meanSe(rw.pvNet),
    };
  }

  return { BASE, buildTable, simulate, value, estimate, cte, hwGbmPut, erfc, normals3, philox, seedKey, meanSe };
}

if (typeof module !== 'undefined' && module.exports) module.exports = segfundEngine();
else self.SF = segfundEngine();
