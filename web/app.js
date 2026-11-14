/* Below the Floor: page logic. Engine in engine.js, measured numbers in data.js. */
(function () {
  'use strict';

  const SF = window.SF, DATA = window.SF_DATA;
  const C = {
    bone: '237,229,211', cinnabar: '255,82,51', brass: '224,173,85', ash: '140,132,118',
    night: '#0c0b09',
  };
  const MONO = '400 10px "Martian Mono", Consolas, monospace';
  const SERIF_IT = 'italic 400 15px "Bodoni Moda", Georgia, serif';
  const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const N_FULL = 16384, N_DRAFT = 4096;
  const DT_YEARS = 1 / 12;

  // ---------------------------------------------------------------- formatting
  const nf0 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
  const money = (v) => (v < 0 ? '−$' : '$') + nf0.format(Math.abs(Math.round(v)));
  const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
  const bp = (v) => Math.round(v * 1e4) + ' bp';
  const shortMoney = (v) => v >= 1e6 ? '$' + (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? '$' + Math.round(v / 1e3) + 'k' : '$' + Math.round(v);
  const $ = (id) => document.getElementById(id);

  function fit(canvas) {
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: r.width, h: r.height, dpr };
  }

  function quantile(sorted, q) {
    const i = (sorted.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
  }

  // a future that ends below the floor is an ordinary trail early on and turns red as it heads there
  function lossGradient(ctx, x0, x1, alpha) {
    const gr = ctx.createLinearGradient(x0, 0, x1, 0);
    gr.addColorStop(0, `rgba(${C.bone},0.05)`);
    gr.addColorStop(0.35, `rgba(${C.cinnabar},${alpha * 0.25})`);
    gr.addColorStop(1, `rgba(${C.cinnabar},${alpha})`);
    return gr;
  }

  // log-scale fund axis shared by the path plates
  const Y_MIN = 30000, Y_MAX = 520000;
  function yScale(top, bottom) {
    const l0 = Math.log(Y_MIN), l1 = Math.log(Y_MAX);
    return (v) => bottom - (Math.log(Math.min(Math.max(v, Y_MIN * 0.7), Y_MAX * 1.4)) - l0) / (l1 - l0) * (bottom - top);
  }

  function drawFundAxes(ctx, x0, x1, top, bottom, ys, floorV, opts = {}) {
    ctx.save();
    ctx.font = MONO;
    ctx.textBaseline = 'middle';
    for (const v of [50000, 100000, 200000, 400000]) {
      const y = ys(v);
      ctx.strokeStyle = `rgba(${C.bone},0.07)`;
      ctx.beginPath(); ctx.moveTo(x0, y + 0.5); ctx.lineTo(x1, y + 0.5); ctx.stroke();
      ctx.fillStyle = `rgba(${C.ash},0.9)`;
      ctx.textAlign = 'right';
      ctx.fillText(shortMoney(v), x0 - 10, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let yr = 0; yr <= 10; yr += opts.yearStep || 2) {
      const x = x0 + (x1 - x0) * yr / 10;
      ctx.fillStyle = `rgba(${C.ash},0.9)`;
      ctx.fillText(yr === 0 ? 'today' : 'yr ' + yr, x, bottom + 10);
      ctx.strokeStyle = `rgba(${C.bone},0.07)`;
      ctx.beginPath(); ctx.moveTo(x + 0.5, bottom); ctx.lineTo(x + 0.5, bottom + 5); ctx.stroke();
    }
    // the floor
    const fy = Math.round(ys(floorV)) + 0.5;
    ctx.strokeStyle = `rgba(${C.cinnabar},0.95)`;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, fy); ctx.lineTo(x1, fy); ctx.stroke();
    if (!opts.noFloorLabel) {
      ctx.font = SERIF_IT;
      const label = 'the floor, ' + money(floorV);
      const tw = ctx.measureText(label).width;
      const lx = opts.floorLabelRight ? x1 - tw - 10 : x0 + 14;
      ctx.fillStyle = 'rgba(12,11,9,0.82)';
      ctx.fillRect(lx - 6, fy - 24, tw + 12, 19);
      ctx.fillStyle = `rgba(${C.cinnabar},1)`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(label, lx, fy - 8);
    }
    ctx.restore();
  }

  // ---------------------------------------------------------------- engine in a background thread
  const Engine = (function () {
    let worker = null, seq = 0, busy = false, pending = null;
    const listeners = [];
    try {
      if (location.hash === '#inline') throw new Error('inline engine requested'); // headless screenshots
      const src = segfundEngine.toString() + `
        const SF = segfundEngine();
        onmessage = (e) => {
          const { id, params, nRN, nRW } = e.data;
          const t0 = performance.now();
          const v = SF.value(params, { nRN, nRW, keep: nRW });
          const ms = performance.now() - t0;
          const rw = v.rw;
          const out = {
            id, ms, nRN, nRW, nSteps: rw.nSteps,
            net: v.net, claims: v.claims, fees: v.fees, breakeven: v.breakeven,
            cte95: v.cte95, cte70: v.cte70, pLoss: v.pLoss, rwMean: v.rwMean,
            path: rw.path, pvNet: rw.pvNet, pvMat: rw.pvMat, pvDeath: rw.pvDeath, pvFee: rw.pvFee, FT: rw.FT, lT: rw.lT,
          };
          postMessage(out, [rw.path.buffer, rw.pvNet.buffer, rw.pvMat.buffer, rw.pvDeath.buffer, rw.pvFee.buffer, rw.FT.buffer, rw.lT.buffer]);
        };`;
      worker = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
      worker.onmessage = (e) => done(e.data);
      worker.onerror = () => { worker = null; };
    } catch (err) { worker = null; }

    function runInline(job) {
      setTimeout(() => {
        const t0 = performance.now();
        const v = SF.value(job.params, { nRN: job.nRN, nRW: job.nRW, keep: job.nRW });
        const rw = v.rw;
        done({
          id: job.id, ms: performance.now() - t0, nRN: job.nRN, nRW: job.nRW, nSteps: rw.nSteps,
          net: v.net, claims: v.claims, fees: v.fees, breakeven: v.breakeven,
          cte95: v.cte95, cte70: v.cte70, pLoss: v.pLoss, rwMean: v.rwMean,
          path: rw.path, pvNet: rw.pvNet, pvMat: rw.pvMat, pvDeath: rw.pvDeath, pvFee: rw.pvFee, FT: rw.FT, lT: rw.lT,
        });
      }, 16);
    }
    function start(job) {
      busy = true;
      listeners.forEach((f) => f({ type: 'start', job }));
      if (worker) worker.postMessage(job); else runInline(job);
    }
    function done(res) {
      busy = false;
      res.threaded = !!worker;
      listeners.forEach((f) => f({ type: 'result', res }));
      if (pending) { const j = pending; pending = null; start(j); }
    }
    return {
      request(params, n) {
        const job = { id: ++seq, params: JSON.parse(JSON.stringify(params)), nRN: n, nRW: n };
        if (busy) pending = job; else start(job);
      },
      on(f) { listeners.push(f); },
      get threaded() { return !!worker; },
    };
  })();

  // ---------------------------------------------------------------- state
  const params = JSON.parse(JSON.stringify(SF.BASE));
  let R = null;            // latest result
  let quantCache = new Map();

  // ---------------------------------------------------------------- I. the exposure
  const Hero = (function () {
    const canvas = $('hero');
    const trails = document.createElement('canvas');
    let g = null, geo = null, order = null, revealed = 0, anim = null, hover = null;
    const N_DRAW = 1400;

    function layout() {
      g = fit(canvas);
      trails.width = canvas.width; trails.height = canvas.height;
      const narrow = g.w < 640;
      const pad = { l: Math.max(narrow ? 52 : 64, g.w * 0.07), r: Math.max(narrow ? 64 : 90, g.w * 0.15), t: narrow ? 58 : 34, b: 40 };
      const x0 = pad.l, x1 = g.w - pad.r, top = pad.t, bottom = g.h - pad.b;
      geo = { x0, x1, top, bottom, ys: yScale(top, bottom), hx0: x1 + 18, hx1: g.w - Math.max(16, g.w * 0.02) };
    }

    function drawPathsInto(ctx, from, to) {
      const n1 = R.nSteps + 1, P = R.path, floor = params.gmmb * params.deposit;
      const { x0, x1, ys } = geo;
      const dx = (x1 - x0) / R.nSteps;
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      ctx.lineWidth = 1;
      for (let pass = 0; pass < 2; pass++) {
        ctx.beginPath();
        for (let j = from; j < to; j++) {
          const i = order[j];
          const below = R.FT[i] < floor;
          if ((pass === 1) !== below) continue;
          const o = i * n1;
          ctx.moveTo(x0, ys(P[o]));
          for (let k = 1; k < n1; k++) ctx.lineTo(x0 + k * dx, ys(P[o + k]));
        }
        ctx.strokeStyle = pass === 0 ? `rgba(${C.bone},0.055)` : lossGradient(ctx, x0, x1, 0.42);
        ctx.stroke();
      }
      ctx.restore();
    }

    function sideHistogram(ctx, upto) {
      const floor = params.gmmb * params.deposit;
      const { top, bottom, hx0, hx1, ys } = geo;
      const NB = 64, l0 = Math.log(Y_MIN), l1 = Math.log(Y_MAX);
      const counts = new Float64Array(NB), countsAll = new Float64Array(NB);
      const n = R.FT.length;
      const lim = Math.round(n * upto / N_DRAW);
      for (let i = 0; i < n; i++) {
        const b = Math.floor((Math.log(Math.min(Math.max(R.FT[i], Y_MIN), Y_MAX * 0.9999)) - l0) / (l1 - l0) * NB);
        countsAll[b]++;
        if (i < lim) counts[b]++;
      }
      const mx = Math.max(...countsAll.slice(1, NB - 1)); // end bins collect everything off-scale
      const bh = (bottom - top) / NB;
      ctx.save();
      ctx.strokeStyle = `rgba(${C.bone},0.18)`;
      ctx.beginPath(); ctx.moveTo(hx0 - 0.5, top); ctx.lineTo(hx0 - 0.5, bottom); ctx.stroke();
      for (let b = 0; b < NB; b++) {
        if (!counts[b]) continue;
        const vMid = Math.exp(l0 + (b + 0.5) / NB * (l1 - l0));
        const y = bottom - (b + 1) * bh;
        const w = Math.min(hx1 - hx0, (hx1 - hx0) * counts[b] / mx);
        ctx.fillStyle = vMid < floor ? `rgba(${C.cinnabar},0.85)` : `rgba(${C.bone},0.55)`;
        ctx.fillRect(hx0, y + 0.5, w, Math.max(1, bh - 1));
      }
      if (g.w > 640) {
        ctx.font = MONO; ctx.fillStyle = `rgba(${C.ash},0.9)`; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText('where they land', hx0, top - 10);
      }
      ctx.restore();
    }

    function frame() {
      const ctx = g.ctx;
      ctx.clearRect(0, 0, g.w, g.h);
      ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.drawImage(trails, 0, 0); ctx.restore();
      drawFundAxes(ctx, geo.x0, geo.x1, geo.top, geo.bottom, geo.ys, params.gmmb * params.deposit);
      // deposit marker
      const y0 = geo.ys(params.deposit);
      ctx.fillStyle = `rgba(${C.bone},1)`;
      ctx.beginPath(); ctx.arc(geo.x0, y0, 3, 0, Math.PI * 2); ctx.fill();
      sideHistogram(ctx, revealed);
      if (hover) drawHover(ctx);
    }

    function stepQuantiles(k) {
      const key = R.id + ':' + k;
      if (quantCache.has(key)) return quantCache.get(key);
      const n = R.FT.length, n1 = R.nSteps + 1, a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = R.path[i * n1 + k];
      a.sort();
      const q = { p05: quantile(a, 0.05), p50: quantile(a, 0.5), p95: quantile(a, 0.95) };
      let below = 0;
      const floor = params.gmmb * params.deposit;
      for (let i = 0; i < n; i++) if (a[i] < floor) below++;
      q.below = below / n;
      quantCache.set(key, q);
      return q;
    }

    function drawHover(ctx) {
      const { x0, x1, top, bottom, ys } = geo;
      const k = Math.round((hover - x0) / (x1 - x0) * R.nSteps);
      if (k < 0 || k > R.nSteps) return;
      const x = x0 + (x1 - x0) * k / R.nSteps;
      const q = stepQuantiles(k);
      ctx.save();
      ctx.strokeStyle = `rgba(${C.brass},0.55)`;
      ctx.beginPath(); ctx.moveTo(x + 0.5, top); ctx.lineTo(x + 0.5, bottom); ctx.stroke();
      ctx.fillStyle = `rgba(${C.brass},1)`;
      for (const v of [q.p05, q.p50, q.p95]) { ctx.beginPath(); ctx.arc(x, ys(v), 2.6, 0, Math.PI * 2); ctx.fill(); }
      const lines = [
        `year ${(k * DT_YEARS).toFixed(1)}`,
        `best 5% above  ${money(q.p95)}`,
        `median         ${money(q.p50)}`,
        `worst 5% below ${money(q.p05)}`,
        `under the floor ${pct(q.below)}`,
      ];
      ctx.font = MONO;
      const tw = Math.max(...lines.map((s) => ctx.measureText(s).width)) + 22;
      const bx = x + 14 + tw > x1 ? x - 14 - tw : x + 14, by = top + 6;
      ctx.fillStyle = 'rgba(12,11,9,0.88)';
      ctx.fillRect(bx, by, tw, lines.length * 17 + 14);
      ctx.strokeStyle = `rgba(${C.brass},0.5)`;
      ctx.strokeRect(bx + 0.5, by + 0.5, tw - 1, lines.length * 17 + 13);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      lines.forEach((s, i) => {
        ctx.fillStyle = i === 0 ? `rgba(${C.brass},1)` : i === 4 ? `rgba(${C.cinnabar},1)` : `rgba(${C.bone},0.9)`;
        ctx.fillText(s, bx + 11, by + 9 + i * 17);
      });
      ctx.restore();
    }

    function tally() {
      const floor = params.gmmb * params.deposit;
      let below = 0;
      for (let i = 0; i < R.FT.length; i++) if (R.FT[i] < floor) below++;
      $('tally').innerHTML = `<b>${nf0.format(Math.min(revealed, N_DRAW))}</b> of ${nf0.format(R.FT.length)} futures drawn &nbsp;·&nbsp; ` +
        `<span class="red">${nf0.format(below)} (${pct(below / R.FT.length)}) end below the floor</span>`;
    }

    function render(animate) {
      if (!R) return;
      layout();
      const n = R.FT.length;
      order = Array.from({ length: Math.min(N_DRAW, n) }, (_, i) => i); // first paths: a fair sample
      const tctx = trails.getContext('2d');
      tctx.setTransform(g.dpr, 0, 0, g.dpr, 0, 0);
      tctx.clearRect(0, 0, g.w, g.h);
      cancelAnimationFrame(anim);
      if (!animate || REDUCED) {
        drawPathsInto(tctx, 0, order.length);
        revealed = order.length;
        frame(); tally();
        return;
      }
      revealed = 0;
      const t0 = performance.now(), dur = 5200;
      const tick = (now) => {
        const target = Math.min(order.length, Math.round(order.length * Math.pow((now - t0) / dur, 1.35)));
        if (target > revealed) { drawPathsInto(tctx, revealed, target); revealed = target; }
        frame(); tally();
        if (revealed < order.length) anim = requestAnimationFrame(tick);
      };
      anim = requestAnimationFrame(tick);
    }

    canvas.addEventListener('pointermove', (e) => {
      if (!R || !geo) return;
      const r = canvas.getBoundingClientRect();
      const x = e.clientX - r.left;
      hover = x >= geo.x0 && x <= geo.x1 ? x : null;
      frame();
    });
    canvas.addEventListener('pointerleave', () => { hover = null; if (R) frame(); });
    return { render, resize: () => render(false) };
  })();

  // ---------------------------------------------------------------- II. dials and ledger
  const DIALS = [
    { key: 'sigmaS', label: 'Market volatility', sub: 'how hard the fund swings, per year', min: 0.06, max: 0.36, step: 0.005, fmt: (v) => pct(v, 1) },
    { key: 'gmmb', label: 'The floor', sub: 'share of the deposit guaranteed at maturity', min: 0.5, max: 1.0, step: 0.05, fmt: (v) => pct(v, 0) },
    { key: 'fee', label: 'Guarantee fee', sub: 'kept by the insurer, out of the 2.25% MER', min: 0, max: 0.015, step: 0.0005, fmt: bp },
    { key: 'age', label: 'Age at purchase', sub: 'older buyers make the death benefit dearer', min: 30, max: 80, step: 1, fmt: (v) => v + ' yrs' },
    { key: 'rate', label: 'Long-term interest rate', sub: 'level of the yield curve', min: 0.01, max: 0.07, step: 0.0025, fmt: (v) => pct(v, 2) },
    { key: 'erp', label: 'Equity risk premium', sub: 'real world only: moves the tail, not the price', min: 0, max: 0.08, step: 0.005, fmt: (v) => pct(v, 1) },
  ];
  const getP = (k) => (k === 'rate' ? params.curve.b0 : params[k]);
  const setP = (k, v) => { if (k === 'rate') params.curve.b0 = v; else params[k] = v; };
  const baseOf = (k) => (k === 'rate' ? SF.BASE.curve.b0 : SF.BASE[k]);

  // shareable scenarios: dials that differ from the base contract live in the query string
  (function readScenario() {
    const q = new URLSearchParams(location.search);
    for (const d of DIALS) {
      if (!q.has(d.key)) continue;
      const v = parseFloat(q.get(d.key));
      if (isFinite(v)) setP(d.key, Math.min(d.max, Math.max(d.min, Math.round((v - d.min) / d.step) * d.step + d.min)));
    }
  })();
  function writeScenario() {
    const q = new URLSearchParams();
    for (const d of DIALS) {
      const v = getP(d.key);
      if (Math.abs(v - baseOf(d.key)) > d.step / 2) q.set(d.key, +v.toFixed(5));
    }
    const s = q.toString();
    try { history.replaceState(null, '', location.pathname + (s ? '?' + s : '') + location.hash); } catch (e) { /* file:// in some browsers */ }
  }

  const Dials = (function () {
    const list = $('dial-list');
    const els = {};
    for (const d of DIALS) {
      const id = 'dial-' + d.key;
      const wrap = document.createElement('div');
      wrap.className = 'dial';
      const homePct = (baseOf(d.key) - d.min) / (d.max - d.min) * 100;
      wrap.innerHTML = `<label for="${id}">${d.label}<small>${d.sub}</small></label><output id="${id}-out"></output>
        <div class="track"><span class="home" style="left:${homePct}%"></span>${d.key === 'fee' ? '<span class="mark" id="be-mark" data-label=""></span>' : ''}
        <input type="range" id="${id}" min="${d.min}" max="${d.max}" step="${d.step}" value="${getP(d.key)}"></div>`;
      list.appendChild(wrap);
      const input = wrap.querySelector('input'), out = wrap.querySelector('output');
      out.textContent = d.fmt(getP(d.key));
      input.addEventListener('input', () => { setP(d.key, +input.value); out.textContent = d.fmt(+input.value); Engine.request(params, N_DRAFT); });
      input.addEventListener('change', () => { writeScenario(); Engine.request(params, N_FULL); });
      els[d.key] = { input, out, d };
    }
    const reset = document.createElement('button');
    reset.className = 'textbtn';
    reset.textContent = 'return to the base contract';
    reset.addEventListener('click', () => {
      for (const d of DIALS) { setP(d.key, baseOf(d.key)); els[d.key].input.value = baseOf(d.key); els[d.key].out.textContent = d.fmt(baseOf(d.key)); }
      writeScenario();
      Engine.request(params, N_FULL);
    });
    list.appendChild(reset);
    return { els };
  })();

  function renderLedger() {
    const net = R.net, short = net.mean > 0;
    $('l-net').innerHTML = `${money(Math.abs(net.mean))}<span class="pm">± ${nf0.format(Math.max(1, Math.round(net.se)))}</span>`;
    $('l-net').className = 'fig ' + (short ? 'red' : 'bone');
    $('l-net').previousElementSibling.firstChild.textContent = short ? 'At hedging cost, the insurer is short' : 'At hedging cost, the fee more than pays';
    $('l-claims').textContent = money(R.claims.mean);
    $('l-fees').textContent = money(R.fees.mean);
    $('l-be').textContent = isFinite(R.breakeven) ? bp(R.breakeven) : 'n/a';
    $('l-ploss').textContent = pct(R.pLoss);
    $('l-cte').innerHTML = `${money(R.cte95.cte)}<span class="pm">± ${nf0.format(Math.round(R.cte95.se))}</span>`;
    const mx = Math.max(R.claims.mean, R.fees.mean, 1);
    $('b-claims').style.width = (R.claims.mean / mx * 100) + '%';
    $('b-fees').style.width = (R.fees.mean / mx * 100) + '%';
    const mark = $('be-mark'), d = Dials.els.fee.d;
    if (isFinite(R.breakeven)) {
      const f = Math.min(1, Math.max(0, (R.breakeven - d.min) / (d.max - d.min)));
      mark.style.left = (f * 100) + '%';
      mark.style.opacity = 1;
      mark.dataset.label = (R.breakeven > d.max ? '> ' : '') + 'break-even ' + bp(Math.min(R.breakeven, 0.99));
    } else mark.style.opacity = 0;
  }

  Engine.on((ev) => {
    const st = $('status');
    if (ev.type === 'start') {
      st.classList.add('working');
      $('status-text').textContent = ev.job.nRN === N_FULL ? `simulating ${nf0.format(2 * N_FULL)} futures…` : `sketching with ${nf0.format(2 * N_DRAFT)} futures while you turn the dial…`;
      return;
    }
    const res = ev.res;
    st.classList.remove('working');
    const first = R === null;
    R = res;
    quantCache = new Map();
    renderLedger();
    $('status-text').textContent = `${nf0.format(res.nRN)} risk-neutral + ${nf0.format(res.nRW)} real-world futures · ` +
      `${(res.ms / 1000).toFixed(2)} s ${res.threaded ? 'in a background thread' : 'on the main thread'} · seed 20261113` +
      (res.nRN < N_FULL ? ' · draft' : '');
    Hero.render(first);
    Tail.data();
  });

  // ---------------------------------------------------------------- III. the tail
  const Tail = (function () {
    const hc = $('hist'), sc = $('slice');
    let gh = null, gs = null, range = null, sel = [-Infinity, Infinity], preset = 'worst5', sorted = null, drag = null;
    const NB = 96;

    function data() {
      if (!R) return;
      sorted = Float64Array.from(R.pvNet).sort();
      const lo = Math.min(quantile(sorted, 0.003), -1), hi = Math.max(quantile(sorted, 0.997), 1);
      range = [lo, hi];
      applyPreset(preset);
    }

    function applyPreset(p) {
      preset = p;
      const q = (x) => quantile(sorted, x);
      if (p === 'worst5') sel = [R.cte95.var, Infinity];
      else if (p === 'loss') sel = [0, Infinity];
      else if (p === 'best') sel = [-Infinity, q(0.25)];
      else if (p === 'middle') sel = [q(0.4), q(0.6)];
      draw();
    }

    const xOf = (v, x0, x1) => x0 + (Math.min(Math.max(v, range[0]), range[1]) - range[0]) / (range[1] - range[0]) * (x1 - x0);
    const vOf = (x, x0, x1) => range[0] + (x - x0) / (x1 - x0) * (range[1] - range[0]);

    function drawHist() {
      gh = fit(hc);
      const ctx = gh.ctx, x0 = 8, x1 = gh.w - 8, top = 26, bottom = gh.h - 26;
      const counts = new Float64Array(NB);
      const n = R.pvNet.length, bw = (range[1] - range[0]) / NB;
      for (let i = 0; i < n; i++) {
        const b = Math.min(NB - 1, Math.max(0, Math.floor((R.pvNet[i] - range[0]) / bw)));
        counts[b]++;
      }
      const mx = Math.max(...counts);
      const W = (x1 - x0) / NB;
      // selection band
      const sx0 = xOf(sel[0], x0, x1), sx1 = xOf(sel[1], x0, x1);
      ctx.fillStyle = `rgba(${C.brass},0.07)`;
      ctx.fillRect(sx0, top - 14, Math.max(1, sx1 - sx0), bottom - top + 14);
      for (let b = 0; b < NB; b++) {
        const v0 = range[0] + b * bw, v1 = v0 + bw, mid = (v0 + v1) / 2;
        const hgt = Math.sqrt(counts[b] / mx) * (bottom - top); // sqrt scale: keeps the thin tail visible
        const inSel = mid >= sel[0] && mid <= sel[1];
        let col;
        if (inSel) col = `rgba(${C.brass},0.95)`;
        else if (mid >= R.cte95.var) col = `rgba(${C.cinnabar},0.55)`;
        else if (mid > 0) col = `rgba(${C.bone},0.42)`;
        else col = `rgba(${C.bone},0.18)`;
        ctx.fillStyle = col;
        ctx.fillRect(x0 + b * W + 0.5, bottom - hgt, Math.max(1, W - 1.5), hgt);
      }
      // zero line and VaR line
      ctx.font = MONO; ctx.textBaseline = 'top';
      const zx = Math.round(xOf(0, x0, x1)) + 0.5;
      ctx.strokeStyle = `rgba(${C.bone},0.5)`;
      ctx.beginPath(); ctx.moveTo(zx, top - 16); ctx.lineTo(zx, bottom + 4); ctx.stroke();
      ctx.fillStyle = `rgba(${C.bone},0.75)`; ctx.textAlign = 'center';
      ctx.fillText('break even', zx, bottom + 8);
      const vx = Math.round(xOf(R.cte95.var, x0, x1)) + 0.5;
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = `rgba(${C.cinnabar},0.9)`;
      ctx.beginPath(); ctx.moveTo(vx, top - 16); ctx.lineTo(vx, bottom + 4); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = `rgba(${C.cinnabar},1)`; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('worst 5% →', vx + 6, top - 16);
      // axis ticks
      ctx.fillStyle = `rgba(${C.ash},0.9)`; ctx.textAlign = 'center';
      const step = niceStep((range[1] - range[0]) / 6);
      for (let v = Math.ceil(range[0] / step) * step; v <= range[1]; v += step) {
        if (Math.abs(v) < step / 2) continue;
        ctx.fillText((v > 0 ? '+' : '') + shortMoney(Math.abs(v)).replace('$', v < 0 ? '−$' : '$'), xOf(v, x0, x1), bottom + 8);
      }
      hc._geo = { x0, x1 };
    }

    function niceStep(raw) {
      const p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
      return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
    }

    function selectedIdx() {
      const out = [];
      for (let i = 0; i < R.pvNet.length; i++) if (R.pvNet[i] >= sel[0] && R.pvNet[i] <= sel[1]) out.push(i);
      return out;
    }

    function drawSlice(idx) {
      gs = fit(sc);
      const ctx = gs.ctx;
      const x0 = 58, x1 = gs.w - 14, top = 18, bottom = gs.h - 34, ys = yScale(top, bottom);
      const n1 = R.nSteps + 1, dx = (x1 - x0) / R.nSteps, P = R.path;
      const inSel = new Uint8Array(R.pvNet.length);
      idx.forEach((i) => { inSel[i] = 1; });
      ctx.save();
      ctx.beginPath();
      let drawn = 0;
      for (let i = 0; i < R.pvNet.length && drawn < 600; i++) {
        if (inSel[i]) continue;
        drawn++;
        const o = i * n1;
        ctx.moveTo(x0, ys(P[o]));
        for (let k = 1; k < n1; k++) ctx.lineTo(x0 + k * dx, ys(P[o + k]));
      }
      ctx.strokeStyle = `rgba(${C.bone},0.035)`;
      ctx.stroke();
      ctx.globalCompositeOperation = 'lighter';
      const show = idx.length > 450 ? idx.filter((_, j) => j % Math.ceil(idx.length / 450) === 0) : idx;
      ctx.beginPath();
      for (const i of show) {
        const o = i * n1;
        ctx.moveTo(x0, ys(P[o]));
        for (let k = 1; k < n1; k++) ctx.lineTo(x0 + k * dx, ys(P[o + k]));
      }
      ctx.strokeStyle = `rgba(${C.brass},${Math.min(0.5, Math.max(0.07, 18 / Math.sqrt(show.length + 1) / 10))})`;
      ctx.stroke();
      ctx.restore();
      drawFundAxes(ctx, x0, x1, top, bottom, ys, params.gmmb * params.deposit, { noFloorLabel: true });
      $('slice-label').innerHTML = `<b>${nf0.format(show.length)}</b> of ${nf0.format(idx.length)} selected futures shown`;
    }

    function facts(idx) {
      const n = idx.length;
      const el = $('facts'), story = $('story');
      if (!n) { el.innerHTML = '<div><dt>No futures in that slice</dt><dd>—</dd></div>'; story.innerHTML = '&nbsp;'; return; }
      const floor = params.gmmb * params.deposit, n1 = R.nSteps + 1;
      let sNet = 0, sMat = 0, sDeath = 0, sFee = 0, below = 0, sL = 0;
      const fT = new Float64Array(n), lows = new Float64Array(n), lowAt = new Float64Array(n);
      idx.forEach((i, j) => {
        sNet += R.pvNet[i]; sMat += R.pvMat[i]; sDeath += R.pvDeath[i]; sFee += R.pvFee[i]; sL += R.lT[i];
        if (R.FT[i] < floor) below++;
        fT[j] = R.FT[i];
        let m = Infinity, at = 0;
        for (let k = 0; k < n1; k++) { const v = R.path[i * n1 + k]; if (v < m) { m = v; at = k; } }
        lows[j] = m; lowAt[j] = at;
      });
      fT.sort(); lows.sort(); lowAt.sort();
      const meanNet = sNet / n, claims = (sMat + sDeath) / n, fees = sFee / n;
      const rows = [
        ['Futures in the slice', `${nf0.format(n)} <span style="color:var(--ash)">(${pct(n / R.pvNet.length)})</span>`, ''],
        ['Average result for the insurer', meanNet > 0 ? `${money(meanNet)} loss` : `${money(-meanNet)} profit`, meanNet > 0 ? 'red' : 'brass'],
        ['Claims paid, average', money(claims), ''],
        ['Fees collected, average', money(fees), ''],
        ['Fund at maturity, median', money(quantile(fT, 0.5)), ''],
        ['Finish below the floor', pct(below / n, 0), below / n > 0.5 ? 'red' : ''],
        ['Lowest point of the fund, median', `${pct(quantile(lows, 0.5) / params.deposit, 0)} of deposit`, ''],
        ['Policies still in force at the end', pct(sL / n, 0), ''],
      ];
      el.innerHTML = rows.map(([a, b, c]) => `<div><dt>${a}</dt><dd class="${c}">${b}</dd></div>`).join('');
      const lowYear = (quantile(lowAt, 0.5) * DT_YEARS).toFixed(1);
      const deathShare = claims > 0 ? (sDeath / n) / claims : 0;
      let s;
      if (meanNet > 0 && below / n > 0.5) {
        s = `Markets break early and never fully recover: the fund bottoms out around year ${lowYear} at ${pct(quantile(lows, 0.5) / params.deposit, 0)} of the deposit, and most of these futures end under the floor.`;
      } else if (meanNet > 0 && deathShare > 0.4) {
        s = `The loss here is mostly death benefits: policyholders die while the fund is down, and the insurer pays the full deposit back. ${pct(deathShare, 0)} of claims are death claims.`;
      } else if (meanNet > 0) {
        s = `Losses without a disaster: the fund drifts sideways, fees stay small because the fund is small, and the claims that do come are enough to tip the balance.`;
      } else if (claims < fees * 0.05) {
        s = `These are the futures that pay for the others. The fund ends at a median of ${money(quantile(fT, 0.5))}; the insurer collects ${money(fees)} in fees and pays almost nothing.`;
      } else {
        s = `Ordinary futures. Some claims, more fees: the insurer keeps a margin of ${money(-meanNet)} on average.`;
      }
      story.textContent = s;
      $('hist-label').innerHTML = `selected <b>${isFinite(sel[0]) ? money(sel[0]) : 'all'} → ${isFinite(sel[1]) ? money(sel[1]) : 'worst'}</b>`;
    }

    function draw() {
      if (!R) return;
      drawHist();
      const idx = selectedIdx();
      drawSlice(idx);
      facts(idx);
    }

    hc.addEventListener('pointerdown', (e) => {
      if (!R) return;
      hc.setPointerCapture(e.pointerId);
      const r = hc.getBoundingClientRect(), g = hc._geo;
      drag = vOf(e.clientX - r.left, g.x0, g.x1);
      sel = [drag, drag];
      preset = null;
      drawHist();
    });
    hc.addEventListener('pointermove', (e) => {
      if (drag === null) return;
      const r = hc.getBoundingClientRect(), g = hc._geo;
      const v = vOf(e.clientX - r.left, g.x0, g.x1);
      sel = [Math.min(drag, v), Math.max(drag, v)];
      drawHist();
    });
    const end = () => {
      if (drag === null) return;
      drag = null;
      const bw = (range[1] - range[0]) / NB;
      if (sel[1] - sel[0] < bw) { sel = [sel[0] - bw / 2, sel[0] + bw / 2]; }
      if (sel[1] >= range[1] - 1e-9) sel[1] = Infinity;
      if (sel[0] <= range[0] + 1e-9) sel[0] = -Infinity;
      draw();
    };
    hc.addEventListener('pointerup', end);
    hc.addEventListener('pointercancel', end);
    document.querySelectorAll('[data-sel]').forEach((b) => b.addEventListener('click', () => applyPreset(b.dataset.sel)));
    return { data, draw };
  })();

  // ---------------------------------------------------------------- IV. nested
  const Nested = (function () {
    const canvas = $('tree');
    const K0 = 12, N_OUT = 40, N_IN = 512;
    let seed = 7, outer = null, inner = null, pick = 0, g = null, anim = null, progress = 1;

    function simulateAll() {
      const p = JSON.parse(JSON.stringify(SF.BASE));
      const rw = SF.buildTable(p, 'real_world'), rn = SF.buildTable(p, 'risk_neutral');
      outer = SF.simulate(rw, N_OUT, { seed: 9000 + seed, kEnd: K0, keep: N_OUT });
      // favour an interesting year: alternate between a bad one and a random one
      const byEnd = Array.from({ length: N_OUT }, (_, i) => i).sort((a, b) => outer.FT[a] - outer.FT[b]);
      pick = seed % 2 === 1 ? byEnd[Math.floor(N_OUT * 0.12)] : byEnd[(seed * 7) % N_OUT];
      const init = { x: outer.xT[pick], F: outer.FT[pick], GM: p.gmmb * p.deposit, GD: p.gmdb * p.deposit, l: outer.lT[pick] };
      inner = SF.simulate(rn, N_IN, { seed: 31000 + seed, stream: 1, antiHalf: N_IN / 2, kStart: K0, init, keep: N_IN });
      const h = N_IN / 2, pairs = new Float64Array(h);
      for (let i = 0; i < h; i++) pairs[i] = 0.5 * (inner.pvNet[i] + inner.pvNet[i + h]);
      inner.value = SF.meanSe(pairs);
      let below = 0;
      for (let i = 0; i < N_IN; i++) if (inner.FT[i] < p.gmmb * p.deposit) below++;
      inner.below = below / N_IN;
      inner.start = init;
    }

    function draw() {
      g = fit(canvas);
      const ctx = g.ctx;
      const x0 = Math.max(58, g.w * 0.06), x1 = g.w - 16, top = 30, bottom = g.h - 36;
      const ys = yScale(top, bottom), X = (yr) => x0 + (x1 - x0) * yr / 10;
      const floor = SF.BASE.gmmb * SF.BASE.deposit;
      // outer ghosts
      const n1o = K0 + 1;
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      ctx.beginPath();
      for (let i = 0; i < N_OUT; i++) {
        if (i === pick) continue;
        const o = i * n1o;
        ctx.moveTo(X(0), ys(outer.path[o]));
        for (let k = 1; k < n1o; k++) ctx.lineTo(X(k * DT_YEARS), ys(outer.path[o + k]));
      }
      ctx.strokeStyle = `rgba(${C.bone},0.16)`; ctx.stroke();
      // inner fan up to `progress`
      const n1i = inner.nSteps + 1, kMax = Math.max(1, Math.round(progress * inner.nSteps));
      for (let pass = 0; pass < 2; pass++) {
        ctx.beginPath();
        for (let i = 0; i < N_IN; i++) {
          const below = inner.FT[i] < floor;
          if ((pass === 1) !== below) continue;
          const o = i * n1i;
          ctx.moveTo(X(1), ys(inner.path[o]));
          for (let k = 1; k <= kMax; k++) ctx.lineTo(X(1 + k * DT_YEARS), ys(inner.path[o + k]));
        }
        ctx.strokeStyle = pass === 0 ? `rgba(${C.bone},0.07)` : lossGradient(ctx, X(1), x1, 0.4);
        ctx.stroke();
      }
      ctx.restore();
      drawFundAxes(ctx, x0, x1, top, bottom, ys, floor, { yearStep: g.w < 640 ? 2 : 1, floorLabelRight: true });
      // the chosen outer year, bright
      ctx.save();
      ctx.shadowColor = `rgba(${C.brass},0.8)`; ctx.shadowBlur = 12;
      ctx.strokeStyle = `rgba(${C.brass},1)`; ctx.lineWidth = 2;
      ctx.beginPath();
      const o = pick * n1o;
      ctx.moveTo(X(0), ys(outer.path[o]));
      for (let k = 1; k < n1o; k++) ctx.lineTo(X(k * DT_YEARS), ys(outer.path[o + k]));
      ctx.stroke();
      ctx.restore();
      const ex = X(1), ey = ys(inner.start.F);
      ctx.fillStyle = `rgba(${C.brass},1)`;
      ctx.beginPath(); ctx.arc(ex, ey, 4, 0, Math.PI * 2); ctx.fill();
      // valuation date
      ctx.save();
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = `rgba(${C.brass},0.6)`;
      ctx.beginPath(); ctx.moveTo(ex + 0.5, top - 10); ctx.lineTo(ex + 0.5, bottom); ctx.stroke();
      ctx.restore();
      ctx.font = SERIF_IT; ctx.fillStyle = `rgba(${C.brass},1)`; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText('valuation date', ex + 8, top + 4);
      ctx.font = MONO; ctx.fillStyle = `rgba(${C.brass},1)`; ctx.textBaseline = 'bottom'; ctx.textAlign = 'right';
      ctx.fillText(money(inner.start.F), ex - 8, ey - 10);
    }

    function label() {
      const v = inner.value;
      $('tree-label').innerHTML = `fund at year 1: <b>${money(inner.start.F)}</b> · remaining promise worth <b>${money(v.mean)} ± ${nf0.format(Math.round(v.se))}</b> · ${N_IN} futures, ${pct(inner.below, 0)} end below the floor`;
    }

    function run(animate) {
      simulateAll();
      label();
      cancelAnimationFrame(anim);
      if (!animate || REDUCED) { progress = 1; draw(); return; }
      const t0 = performance.now();
      const tick = (now) => {
        progress = Math.min(1, (now - t0) / 1400);
        progress = 1 - Math.pow(1 - progress, 3);
        draw();
        if (progress < 1) anim = requestAnimationFrame(tick);
      };
      anim = requestAnimationFrame(tick);
    }

    function cost() {
      const d = DATA.nested;
      const outerN = d.n_outer, innerN = d.inner_paths / d.n_outer;
      const mins = d.cpu_s_est / 60;
      $('cost').innerHTML = `Doing this properly for one capital number:
        <div class="eq">${nf0.format(outerN)} years<span class="x">×</span>${nf0.format(innerN)} futures each<span class="x">=</span>${nf0.format(d.inner_paths)} paths</div>
        <div class="vs"><span>fused GPU kernel</span><span class="t">${d.gpu_s.toFixed(2)} s</span>
        <span>NumPy, one core (estimated)</span><span class="t">${Math.round(mins)} min</span>
        <span>CTE(95) of the year-one loss</span><span class="t">${money(d.cte95)} ± ${Math.round(d.cte95_se)}</span></div>`;
    }

    $('reroll').addEventListener('click', () => { seed++; run(true); });
    return { run, draw, cost };
  })();

  // ---------------------------------------------------------------- V. the race
  const Race = (function () {
    const d = DATA.race, lanesEl = $('lanes');
    const els = [];
    for (const lane of d.lanes) {
      const el = document.createElement('div');
      el.className = 'lane';
      el.dataset.key = lane.key;
      el.innerHTML = `<div class="who">${lane.label}<small>${lane.note}</small></div>
        <div class="road"><div class="fill"></div><div class="tip"></div></div>
        <div class="clock"><span class="t">0.000 s</span><small class="n">0 futures</small></div>`;
      lanesEl.appendChild(el);
      els.push({ lane, el, fill: el.querySelector('.fill'), tip: el.querySelector('.tip'), t: el.querySelector('.t'), n: el.querySelector('.n') });
    }
    const fmtT = (ms) => ms < 1000 ? (ms / 1000).toFixed(3) + ' s' : ms < 60000 ? (ms / 1000).toFixed(2) + ' s'
      : Math.floor(ms / 60000) + ':' + ((ms % 60000) / 1000).toFixed(1).padStart(4, '0');
    let raf = null;

    function go() {
      cancelAnimationFrame(raf);
      els.forEach((e) => e.el.classList.remove('done'));
      $('go').textContent = 'Restart';
      const t0 = performance.now();
      const tick = (now) => {
        const el = now - t0;
        let running = false;
        for (const e of els) {
          const f = Math.min(1, el / e.lane.ms);
          e.fill.style.width = (f * 100) + '%';
          e.tip.style.left = `calc(${f * 100}% - 1px)`;
          e.t.textContent = f >= 1 ? fmtT(e.lane.ms) : fmtT(el);
          e.n.textContent = nf0.format(Math.floor(f * d.n_paths)) + ' futures';
          if (f >= 1) e.el.classList.add('done'); else running = true;
        }
        if (running) raf = requestAnimationFrame(tick); else $('go').textContent = 'Run it again';
      };
      raf = requestAnimationFrame(tick);
    }

    const lane = (k) => d.lanes.find((l) => l.key === k);
    const tri = lane('triton'), th = lane('torch'), npy = lane('numpy'), tc = lane('compile');
    const gpuLanes = d.lanes.filter((l) => l.measured);
    const allGated = gpuLanes.every((l) => l.quotable);
    $('race-meta').textContent = `${d.gpu.replace('NVIDIA GeForce ', '')} · ${d.cpu} · measured ${d.timestamp.slice(0, 10)}`;
    $('ratio').innerHTML = `The fused kernel finishes in <b>${tri.ms.toFixed(1)} ms</b>: <b>${(th.ms / tri.ms).toFixed(1)}×</b> faster than plain PyTorch on the same GPU` +
      (tc ? `, and still <b>${(tc.ms / tri.ms).toFixed(1)}×</b> faster than PyTorch with torch.compile and a CUDA graph, the best the compiler does here` : '') +
      `. Against NumPy's pace it's about <b>${nf0.format(Math.round(npy.ms / tri.ms / 10) * 10)}×</b>. ` +
      (allGated ? 'All GPU times passed' : 'GPU times were checked against') +
      ` the project's clock gate (SM clock at least 70% of max, under 87 °C, run-to-run spread within 5%)` +
      (allGated ? '' : `; ${gpuLanes.filter((l) => !l.quotable).map((l) => l.label).join(', ')} did not pass it and should be read as approximate`) +
      `. The NumPy time is extrapolated from a ${npy.note.match(/[\d,]+ paths/)[0]} run on one core and isn't clock-gated; on this laptop it varied by 2× between runs. ` +
      `A multithreaded CPU engine would close much of that gap, so read it as "prototype vs fused kernel", not "CPU vs GPU".`;
    $('go').addEventListener('click', go);
  })();

  // ---------------------------------------------------------------- page plumbing
  const io = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.12 });
  document.querySelectorAll('.chapter > div:last-child, .colophon').forEach((el) => { el.classList.add('reveal'); io.observe(el); });

  let nestedStarted = false;
  const nio = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting && !nestedStarted) { nestedStarted = true; Nested.run(true); }
  }, { threshold: 0.25 });
  nio.observe($('tree'));
  Nested.cost();

  let rt = null;
  window.addEventListener('resize', () => {
    clearTimeout(rt);
    rt = setTimeout(() => { if (R) { Hero.resize(); Tail.draw(); } if (nestedStarted) Nested.draw(); }, 150);
  });

  const v = DATA.validation;
  $('colo-numbers').textContent = `Prices use 16,384 futures per run (antithetic pairs and a Hull-White put control variate). ` +
    `At the base contract they agree with the GPU engine's ${nf0.format(DATA.reference.n_paths)}-path run (${money(DATA.reference.pv)}) within their error. ` +
    `The race and nested figures are measured results from the repository, whose validation suite passes ${v.cpu[0]}/${v.cpu[1]} CPU and ${v.gpu[0]}/${v.gpu[1]} GPU checks.`;

  // start once fonts are in, so canvas labels use the right faces
  (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(() => Engine.request(params, N_FULL));
})();
