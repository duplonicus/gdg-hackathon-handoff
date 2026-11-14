// The intro: dup sits down, types "/goal win the hackathon", and the camera dives through the hole in the "g" of
// "Winning..." into the capsule picture, where a VLM and a VDM answer the same question. Ends on "Which is better?".
// Everything is computed from t (no timers, no CSS transitions, seeded wobble). Every model output and number on
// screen is in facts.md with its source file and line. Engine patterns from ../film/film.js.
(() => {
const W = 1920, H = 1080;
const INK = '#2b2b2b', GREEN = '#2e8b57', ORANGE = '#e07a1f', RED = '#c0392b', GREY = '#8a8a8a', PAPER = '#fbfaf5';
const HOOD = '#1d1d20', HOOD_FAR = '#45454b', PANTS = '#4b525d', PANTS_FAR = '#6b727d';
const TERM_FG = '#f2eef0', TERM_DIM = '#bfaab8', SPIN = '#f3c97a';
const clamp = (x, a = 0, b = 1) => x < a ? a : x > b ? b : x, lerp = (a, b, u) => a + (b - a) * u;
const E = { io: u => u <= 0 ? 0 : u >= 1 ? 1 : u < .5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2,
            out: u => u <= 0 ? 0 : u >= 1 ? 1 : 1 - Math.pow(1 - u, 3) };
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const hash = s => { let h = 2166136261; for (const c of s) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return h >>> 0; };
const NS = 'http://www.w3.org/2000/svg';
const $ = s => document.querySelector(s);
const gtop = $('#gtop'), gterm = $('#gterm');
const world = $('#world'), paper = $('#paper'), paperIn = $('#paperIn'), term = $('#term');
const DEBUG = new URLSearchParams(location.search).has('debug');

// a hand-drawn polyline: points jittered by a seeded generator, so every render draws the same wobble
function sketcher(rnd) {
  return (pts, close, amp = 2.0) => {
    const q = close ? pts.concat([pts[0]]) : pts; let d = '';
    for (let i = 0; i < q.length - 1; i++) {
      const [x0, y0] = q[i], [x1, y1] = q[i + 1], n = Math.max(2, Math.round(Math.hypot(x1 - x0, y1 - y0) / 60));
      for (let j = 0; j <= n; j++) {
        if (i > 0 && j === 0) continue;
        const u = j / n, e = (j === 0 || j === n) ? .4 : 1;
        d += (d ? ' L' : 'M') + (lerp(x0, x1, u) + (rnd() - .5) * amp * 2 * e).toFixed(1) + ' ' + (lerp(y0, y1, u) + (rnd() - .5) * amp * 2 * e).toFixed(1);
      }
    }
    return d;
  };
}
const roundRectPts = (x, y, w, h, r) => { const p = []; const c = [[x + w - r, y + r, -Math.PI / 2], [x + w - r, y + h - r, 0], [x + r, y + h - r, Math.PI / 2], [x + r, y + r, Math.PI]]; for (const [cx, cy, a0] of c) for (let i = 0; i <= 4; i++) { const a = a0 + i / 4 * Math.PI / 2; p.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); } return p; };
const ellipsePts = (cx, cy, rx, ry, n = 28, a0 = -Math.PI / 2, turns = 1) => { const p = []; for (let i = 0; i < n; i++) { const a = a0 + i / n * Math.PI * 2 * turns; p.push([cx + rx * Math.cos(a), cy + ry * Math.sin(a)]); } return p; };

// ---------------------------------------------------------------- elements (as film.js: drawn in by a pen)
const ELS = {};
function el(id, w, h, build, parent = world, z = 2) {
  const root = document.createElement('div'); root.className = 'el'; root.style.width = w + 'px'; root.style.height = h + 'px'; root.style.zIndex = z;
  const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('width', w); svg.setAttribute('height', h); svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  root.appendChild(svg); parent.appendChild(root);
  const rnd = mulberry32(hash(id)), sk = sketcher(rnd), seq = [];
  const b = {
    path(d, o = {}) {
      const p = document.createElementNS(NS, 'path'); p.setAttribute('d', d);
      p.style.stroke = o.color || INK; p.style.strokeWidth = o.w || 4; p.style.fill = 'none'; svg.appendChild(p);
      const len = p.getTotalLength(); p.style.strokeDasharray = len + ' ' + (len + 10); seq.push({ kind: 'path', el: p, len }); return p;
    },
    line(pts, o = {}) { return b.path(sk(pts, o.close, o.amp), o); },
    rect(x, y, w2, h2, o = {}) { return b.line([[x, y], [x + w2, y], [x + w2, y + h2], [x, y + h2]], { ...o, close: true }); },
    ellipse(cx, cy, rx, ry, o = {}) { return b.line(ellipsePts(cx, cy, rx, ry, Math.max(16, Math.round((rx + ry) / 8)), o.a0 ?? -Math.PI / 2, o.turns ?? 1.06), { ...o, amp: o.amp ?? 2.5 }); },
    arrow(x0, y0, x1, y1, o = {}) {
      b.line([[x0, y0], [x1, y1]], o); const a = Math.atan2(y1 - y0, x1 - x0), L = o.head || 24;
      b.line([[x1 - L * Math.cos(a - .45), y1 - L * Math.sin(a - .45)], [x1, y1], [x1 - L * Math.cos(a + .45), y1 - L * Math.sin(a + .45)]], { ...o, amp: 1 });
    },
    fill(pts, color, opacity = .18) {
      const p = document.createElementNS(NS, 'path'); p.setAttribute('d', sk(pts, true, 1.2) + ' Z');
      p.style.fill = color; p.style.stroke = 'none'; svg.insertBefore(p, svg.firstChild); seq.push({ kind: 'fade', el: p, len: 160, max: opacity }); return p;
    },
    text(x, y, str, o = {}) {
      const d = document.createElement('div'); d.className = 'tx ' + (o.font === 'mono' ? 'mono' : 'hand');
      d.style.left = x + 'px'; d.style.top = y + 'px'; d.style.fontSize = (o.size || 44) + 'px'; d.style.color = o.color || INK;
      if (o.w) { d.style.width = o.w + 'px'; d.style.whiteSpace = 'normal'; }
      if (o.align) d.style.textAlign = o.align;
      if (o.lh) d.style.lineHeight = o.lh;
      d.innerHTML = str.split(' ').map(s => `<span class="w">${s}</span>`).join(' ');
      root.appendChild(d);
      if (!o.manual) for (const w2 of d.querySelectorAll('.w')) seq.push({ kind: 'word', el: w2, len: o.wordLen || 70 });
      return d;
    },
    solid(pts, color, o = {}) {  // a filled shape that fades in once the pen reaches it, outline drawn on top
      const f = document.createElementNS(NS, 'path'); f.setAttribute('d', sk(pts, true, o.amp ?? .5) + ' Z');
      f.style.fill = color; f.style.stroke = 'none'; svg.appendChild(f); seq.push({ kind: 'fade', el: f, len: 40, max: 1 });
      return b.line(pts, { w: o.w || 3.5, close: true, amp: o.amp ?? .5 });
    },
    img(src, x, y, w2, h2) {
      const i = document.createElement('img'); i.src = src; i.className = 'im'; Object.assign(i.style, { left: x + 'px', top: y + 'px', width: w2 + 'px', height: h2 + 'px' });
      root.appendChild(i); seq.push({ kind: 'fade', el: i, len: 300, max: 1 }); return i;
    },
    sk, rnd, svg, root,
  };
  const api = build(b) || {};
  let total = 0; for (const s of seq) { s.start = total; total += s.len; }
  ELS[id] = { id, root, seq, total, update: api.update };
  return ELS[id];
}
function reveal(e, p) {
  const L = p * e.total;
  for (const s of e.seq) {
    const u = clamp((L - s.start) / s.len);
    if (s.kind === 'path') s.el.style.strokeDashoffset = s.len * (1 - u);
    else if (s.kind === 'word') { const v = E.out(u); s.el.style.opacity = v; s.el.style.transform = `translateY(${(1 - v) * 12}px)`; }
    else s.el.style.opacity = E.out(u) * s.max;
  }
}
// place an element: keys [[t, x, y, s, o], ...] eased between, draw = [start, seconds]
function place(e, t, keys, draw) {
  let k = keys[0];
  if (t >= keys[keys.length - 1][0]) k = keys[keys.length - 1];
  else for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i], b2 = keys[i + 1];
    if (t >= a[0] && t < b2[0]) { const u = E.io((t - a[0]) / (b2[0] - a[0])); k = [t, lerp(a[1], b2[1], u), lerp(a[2], b2[2], u), Math.exp(lerp(Math.log(a[3]), Math.log(b2[3]), u)), lerp(a[4], b2[4], u)]; break; }
  }
  const p = draw ? clamp((t - draw[0]) / draw[1]) : 1;
  e.root.style.display = (p <= 0 || k[4] <= 0.001) ? 'none' : 'block';
  reveal(e, p);
  e.root.style.opacity = k[4];
  e.root.style.transform = `translate(${k[1].toFixed(2)}px,${k[2].toFixed(2)}px) scale(${k[3].toFixed(4)})`;
}

// ---------------------------------------------------------------- the timeline (seconds)
const TYPED = '/goal win the hackathon';
const KEYS = []; { const r = mulberry32(hash('keys')); let t = 7.7; for (const c of TYPED) { KEYS.push(t); t += (c === ' ' ? .17 : .1) + r() * .08; } }
const T_ENTER = KEYS[KEYS.length - 1] + .5;
const T_SPIN = T_ENTER + .3, T_WIN = T_SPIN + 1.7, Z0 = T_WIN + .9, Z1 = Z0 + 2.3;
const A = Z1;  // the paper scene settles here; everything after is timed from it
// dup, Nov 6: the dive lands on his question, then the cloud of decision models, then the ladder he tried first; the
// VLM/VDM comparison moves after them (its place in the film is Phase 2's "same base, two answers")
// placement test (Nov 6): the comparison right after the cloud shows what "scores every option" means, before the
// ladder talks about acting when sure
const E0 = A, Q1 = E0 + 9.2, Q2 = Q1 + 16, TT = Q2 + 1.0, T_CMP = TT + 6.4;   // TT: the Phase 1 title card
const SHIFT = T_CMP + 1.2 - A;          // the comparison scene, timed as built, starts this much later
const CMP_LEN = 18.6;                   // the comparison, ending on the truth and the rings
const QL = A + SHIFT + CMP_LEN;          // the ladder starts when the comparison is done
const T_END = QL + 18;
window.T = T_END; window.FPS = 60; window.CUTS = [];
window.SFX = [...KEYS.map((t, i) => ({ t, kind: TYPED[i] === ' ' ? 'space' : 'key' })), { t: T_ENTER, kind: 'enter' }, { t: Z0 + .5, kind: 'whoosh', dur: Z1 - Z0 - .2 }];
window.BEATS = { T_ENTER, T_SPIN, T_WIN, Z0, Z1, A, E0, Q1, Q2, TT, T_CMP, SHIFT, QL, T_END };

// ---------------------------------------------------------------- the room (world coordinates; floor at y 900)
el('floor', 2600, 30, b => { b.line([[0, 10], [2600, 10]], { w: 4, amp: 2.5 }); });
el('chair', 150, 270, b => {
  b.rect(10, 10, 16, 150, { w: 4 }); b.fill([[10, 10], [26, 10], [26, 160], [10, 160]], INK, .5);
  b.rect(10, 158, 118, 14, { w: 4 }); b.line([[70, 172], [70, 228]], { w: 5 });
  b.line([[22, 240], [70, 228], [118, 240]], { w: 4 }); b.ellipse(22, 248, 8, 8, { w: 3, amp: .6 }); b.ellipse(118, 248, 8, 8, { w: 3, amp: .6 });
});
el('desk', 580, 190, b => {
  b.rect(10, 10, 540, 14, { w: 5 }); b.fill([[10, 10], [550, 10], [550, 24], [10, 24]], INK, .12);
  b.line([[190, 24], [190, 170]], { w: 5 }); b.line([[530, 24], [530, 170]], { w: 5 });
  b.line([[200, 70], [520, 70]], { w: 3, color: GREY });
});
el('monitor', 360, 260, b => {
  b.rect(12, 12, 316, 204, { w: 5 });
  b.line([[170, 216], [170, 236]], { w: 6 }); b.line([[126, 238], [214, 238]], { w: 5 });
}, world, 4);
el('keyboard', 160, 30, b => {
  b.rect(5, 8, 135, 7, { w: 3.5 });
  for (let i = 0; i < 7; i++) b.line([[16 + i * 17, 11], [24 + i * 17, 11]], { w: 2, color: GREY, amp: .3 });
});
el('window', 260, 240, b => {
  b.rect(10, 10, 230, 200, { w: 4 }); b.line([[125, 12], [125, 208]], { w: 3 }); b.line([[12, 110], [238, 110]], { w: 3 });
  b.line([[0, 222], [250, 222]], { w: 4 });
  b.line([[40, 60], [70, 40]], { w: 2.5, color: GREY }); b.line([[150, 150], [190, 125]], { w: 2.5, color: GREY });
});
const ROOM = { window: [250, 420], floor: [-300, 890], chair: [690, 640], desk: [820, 730], monitor: [980, 500], keyboard: [855, 725] };
const ROOM_DRAW = { window: [.35, .8], floor: [0, .7], chair: [.15, .7], desk: [.3, .8], monitor: [.45, .7], keyboard: [.7, .4] };

// ---------------------------------------------------------------- the terminal on the monitor (native 1600 x 1000, scaled 0.1875 into the screen)
const tsvg = document.createElementNS(NS, 'svg'); tsvg.setAttribute('width', 1600); tsvg.setAttribute('height', 1000); term.appendChild(tsvg);
const tsk = sketcher(mulberry32(hash('term')));
const tpath = (d, color, w) => { const p = document.createElementNS(NS, 'path'); p.setAttribute('d', d); Object.assign(p.style, { stroke: color, strokeWidth: w, fill: 'none', strokeLinecap: 'round', strokeLinejoin: 'round' }); tsvg.appendChild(p); return p; };
const tline = (top, size, html, color) => { const d = document.createElement('div'); d.className = 'ln'; d.style.top = top + 'px'; d.style.fontSize = size + 'px'; if (color) d.style.color = color; d.innerHTML = html; term.appendChild(d); return d; };
tline(36, 44, '$ claude', TERM_DIM);
tpath(tsk([[60, 112], [1540, 112], [1540, 300], [60, 300]], true, 3), '#d7a3c6', 4);
tline(134, 52, 'Welcome to Claude Code');
tline(212, 40, 'cwd: /mnt/c/Users/dup/dev.win/GDG-Hackathon', TERM_DIM);
const rulesA = [tpath(tsk([[60, 420], [1540, 420]], false, 2.5), TERM_DIM, 3), tpath(tsk([[60, 548], [1540, 548]], false, 2.5), TERM_DIM, 3)];
const rulesB = [tpath(tsk([[60, 700], [1540, 700]], false, 2.5), TERM_DIM, 3), tpath(tsk([[60, 828], [1540, 828]], false, 2.5), TERM_DIM, 3)];
const pl = tline(442, 64, '&gt; <span class="typed"></span><span class="cur"></span>');
const typedSpan = pl.querySelector('.typed'), cur1 = pl.querySelector('.cur');
const pl2 = tline(722, 64, '&gt; <span class="cur"></span>'), cur2 = pl2.querySelector('.cur');
const sp1 = tline(580, 72, '<span class="star">*</span> Confabulating<span class="dots"></span>', SPIN);
const sp2 = tline(580, 72, '<span class="star">*</span> Winnin<span id="gg">g</span><span class="dots"></span>', SPIN);

// a copy of that line above the paper scene, only its g visible, so the glyph's stroke hides the mask's edge
const gcopy = sp2.cloneNode(true); gcopy.removeAttribute('style'); Object.assign(gcopy.style, { top: '580px', fontSize: '72px' }); gcopy.querySelector('#gg').id = ''; gterm.appendChild(gcopy);
const gGlyph = [...gcopy.querySelectorAll('span')].find(x => x.textContent === 'g'); gGlyph.style.color = SPIN;
// the hole in the "g": found by flood-filling the glyph drawn on a canvas; becomes the mask the paper scene shows through
const GM = { F: 400, cw: 300, chh: 420, ox: 50, by: 270, size: 72 };
function measureG() {
  const c = document.createElement('canvas'); c.width = GM.cw; c.height = GM.chh; const x = c.getContext('2d');
  x.font = `${GM.F}px "Ubuntu Mono"`; x.fillStyle = '#000'; x.textBaseline = 'alphabetic'; x.fillText('g', GM.ox, GM.by);
  const a = x.getImageData(0, 0, GM.cw, GM.chh).data, N = GM.cw * GM.chh, out = new Uint8Array(N), stack = [];
  const open = i => a[i * 4 + 3] < 110;
  for (let i = 0; i < GM.cw; i++) { stack.push(i, (GM.chh - 1) * GM.cw + i); }
  for (let j = 0; j < GM.chh; j++) { stack.push(j * GM.cw, j * GM.cw + GM.cw - 1); }
  while (stack.length) { const i = stack.pop(); if (out[i] || !open(i)) continue; out[i] = 1; const px = i % GM.cw, py = (i / GM.cw) | 0;
    if (px > 0) stack.push(i - 1); if (px < GM.cw - 1) stack.push(i + 1); if (py > 0) stack.push(i - GM.cw); if (py < GM.chh - 1) stack.push(i + GM.cw); }
  // enclosed holes: open pixels the border flood never reached; keep the biggest component above the baseline
  const comp = new Int32Array(N).fill(-1), comps = [];
  for (let i = 0; i < N; i++) {
    if (out[i] || !open(i) || comp[i] >= 0) continue;
    const id = comps.length, st = [i]; let n = 0, sx = 0, sy = 0;
    while (st.length) { const k = st.pop(); if (comp[k] >= 0 || out[k] || !open(k)) continue; comp[k] = id; n++; const px = k % GM.cw, py = (k / GM.cw) | 0; sx += px; sy += py;
      if (px > 0) st.push(k - 1); if (px < GM.cw - 1) st.push(k + 1); if (py > 0) st.push(k - GM.cw); if (py < GM.chh - 1) st.push(k + GM.cw); }
    comps.push({ id, n, cx: sx / n, cy: sy / n });
  }
  const best = comps.filter(q => q.cy < GM.by).sort((p, q) => q.n - p.n)[0];
  const m = document.createElement('canvas'); m.width = GM.cw; m.height = GM.chh; const mx = m.getContext('2d'); mx.fillStyle = '#fff';
  for (let i = 0; i < N; i++) if (comp[i] === best.id) { mx.beginPath(); mx.arc(i % GM.cw + .5, ((i / GM.cw) | 0) + .5, 4, 0, Math.PI * 2); mx.fill(); }
  // where the glyph sits in the terminal: its line's layout box, plus the font's ascent for the baseline
  x.font = `${GM.size}px "Ubuntu Mono"`; const asc = x.measureText('g').fontBoundingBoxAscent;
  const span = $('#gg'), gx = sp2.offsetLeft + span.offsetLeft, baseline = sp2.offsetTop + span.offsetTop + asc;
  const f = GM.size / GM.F, k = .1875;
  const toWorld = (cx, cy) => [1000 + (gx + (cx - GM.ox) * f) * k, 520 + (baseline + (cy - GM.by) * f) * k];
  return { url: m.toDataURL(), P: toWorld(best.cx, best.cy), origin: toWorld(0, 0), unit: f * k, r: Math.sqrt(best.n / Math.PI) * f * k };
}
let G = null;

// ---------------------------------------------------------------- dup: black hoodie, aviators, drawn in profile facing right
const TH = 92, SH = 92, UA = 72, FA = 68, STAND = 900 - 190, SEAT = 784;
const SHOULDER = [4, -112];
function makeFig(parent, z, seed) {
  const root = document.createElement('div'); root.className = 'el'; root.style.zIndex = z; parent.appendChild(root);
  const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('width', 1); svg.setAttribute('height', 1); root.appendChild(svg);
  const sk = sketcher(mulberry32(hash(seed)));
  const G2 = (parent) => { const g = document.createElementNS(NS, 'g'); parent.appendChild(g); return g; };
  const P = (g, d, o) => { const p = document.createElementNS(NS, 'path'); p.setAttribute('d', d); Object.assign(p.style, { stroke: o.stroke || 'none', strokeWidth: o.w || 3, fill: o.fill || 'none', strokeLinecap: 'round', strokeLinejoin: 'round', opacity: o.op ?? 1 }); g.appendChild(p); return p; };
  const limb = (g, len, color, w) => { P(g, sk([[0, 0], [0, len]], false, .8), { stroke: INK, w: w + 6 }); P(g, sk([[0, 0], [0, len]], false, .8), { stroke: color, w }); };
  const hand = (g, y) => P(g, sk(ellipsePts(2, y, 9, 9, 14), true, .6) + ' Z', { stroke: INK, w: 3, fill: PAPER });
  const shoe = (g) => P(g, sk([[-10, SH + 2], [4, SH - 4], [26, SH + 2], [24, SH + 8], [-10, SH + 8]], true, .8) + ' Z', { stroke: INK, w: 3, fill: '#2a2a2a' });
  const hip = G2(svg);
  const mkArm = (parent, color) => { const sh = G2(parent), up = G2(sh); limb(up, UA, color, 20); const fo = G2(up); limb(fo, FA, color, 18); hand(fo, FA + 4); return { sh, up, fo }; };
  const mkLeg = (parent, color) => { const th = G2(parent); limb(th, TH, color, 24); const sn = G2(th); limb(sn, SH, color, 20); shoe(sn); return { th, sn }; };
  const back = G2(hip), farArm = mkArm(back, HOOD_FAR);
  const farLeg = mkLeg(hip, PANTS_FAR);
  const torso = G2(hip);
  P(torso, sk([[-30, 10], [-34, -60], [-31, -104], [-18, -122], [12, -124], [27, -110], [31, -60], [30, 10]], true, 1.2) + ' Z', { stroke: INK, w: 3, fill: HOOD });
  P(torso, sk([[8, -14], [12, -50], [30, -52]], false, .6), { stroke: '#5c5c63', w: 3 });
  P(torso, sk([[21, -116], [23, -92]], false, .4), { stroke: '#d9d6cf', w: 3 }); P(torso, sk([[15, -117], [16, -96]], false, .4), { stroke: '#d9d6cf', w: 3 });
  const head = G2(torso); head.setAttribute('transform', 'translate(4,-150)');
  P(head, sk(ellipsePts(-6, 2, 41, 43, 26), true, 1.2) + ' Z', { stroke: INK, w: 3, fill: HOOD });                  // the hood
  P(head, sk(ellipsePts(13, 5, 28, 29, 22), true, .8) + ' Z', { stroke: INK, w: 3, fill: PAPER });                   // the face
  P(head, sk([[40, -2], [48, 9], [41, 13]], false, .4), { stroke: INK, w: 3, fill: PAPER });                          // nose
  P(head, sk([[31, 22], [38, 21]], false, .3), { stroke: INK, w: 2.5 });                                              // mouth
  P(head, sk([[15, -5], [-9, -2]], false, .3), { stroke: INK, w: 2.5 });                                              // temple arm
  P(head, sk([[14, -7], [35, -7], [36, 0], [31, 9], [23, 11], [15, 5]], true, .5) + ' Z', { stroke: INK, w: 2.5, fill: '#2f3a40', op: .92 });  // near aviator lens
  P(head, sk([[44, -6], [51, -6], [51, 4], [47, 8]], true, .3) + ' Z', { stroke: INK, w: 2.5, fill: '#2f3a40', op: .92 });                    // far lens past the nose
  P(head, sk([[14, -10], [51, -9]], false, .3), { stroke: INK, w: 2.5 });                                             // the aviator top bar
  P(head, sk([[19, -3], [25, -5]], false, .2), { stroke: '#e8eef0', w: 2.5 });                                        // glint
  const nearLeg = mkLeg(hip, PANTS);
  const front = G2(hip), nearArm = mkArm(front, HOOD);
  return { root, hip, back, torso, head, front, farArm, nearArm, farLeg, nearLeg };
}
const fig = makeFig(world, 6, 'dup');
const rot = (x, y, a) => { const c = Math.cos(a), s = Math.sin(a); return [x * c - y * s, x * s + y * c]; };
const toRot = (dx, dy) => Math.atan2(-dx, dy) * 180 / Math.PI;  // our rotate() convention: 0 = pointing down
function ik(sx, sy, tx, ty) {  // two-link arm, elbow below the line; returns absolute angles (deg) of upper arm and forearm
  let d = Math.hypot(tx - sx, ty - sy); d = clamp(d, Math.abs(UA - FA) + 1, UA + FA - .5);
  const base = Math.atan2(ty - sy, tx - sx), al = Math.acos(clamp((UA * UA + d * d - FA * FA) / (2 * UA * d), -1, 1));
  const ua = base + al, ex = sx + UA * Math.cos(ua), ey = sy + UA * Math.sin(ua);
  return [toRot(Math.cos(ua), Math.sin(ua)), toRot(tx - ex, ty - ey)];
}
const gauss = (t, c, w) => Math.exp(-(((t - c) / w) ** 2));
function pose(t) {
  const x0 = -170, x1 = 744, a = .62, u = clamp((t - .2) / 3.1);
  const f = (u < a ? u : a + (u - a) - (u - a) ** 2 / (2 * (1 - a))) / (a + (1 - a) / 2), vf = u < a ? 1 : 1 - (u - a) / (1 - a);
  const x = lerp(x0, x1, f), ph = 2 * Math.PI * (x - x0) / 170, amp = clamp(vf * 1.5) * (u > 0 ? 1 : 1);
  const walk = {
    hx: x, hy: STAND + 3 * (1 - Math.cos(2 * ph)) / 2 * amp, lean: 3 * amp,
    nT: -24 * Math.sin(ph) * amp, nK: 38 * Math.max(0, Math.cos(ph)) ** 1.5 * amp, fT: 24 * Math.sin(ph) * amp, fK: 38 * Math.max(0, -Math.cos(ph)) ** 1.5 * amp,
    nU: 18 * Math.sin(ph) * amp, nF: -14, fU: -18 * Math.sin(ph) * amp, fF: -14,
  };
  const s = E.io(clamp((t - 3.35) / 1.0));
  if (s <= 0) return walk;
  const p = { ...walk };
  p.hx = lerp(x1, 760, s); p.hy = lerp(STAND, SEAT, s); p.lean = lerp(walk.lean, 8, s) + 14 * Math.sin(Math.PI * s);
  p.nT = lerp(walk.nT, -80, s); p.nK = lerp(walk.nK, 80, s); p.fT = lerp(walk.fT, -76, s); p.fK = lerp(walk.fK, 76, s);
  p.nU = lerp(walk.nU, 6, s); p.fU = lerp(walk.fU, -4, s);
  const h = E.io(clamp((t - 4.05) / .8));
  if (h > 0) {  // hands to the keys, and a small lift before each keystroke (alternating hands)
    const L = p.lean * Math.PI / 180, [ox, oy] = rot(SHOULDER[0], SHOULDER[1], L), sx = p.hx + ox, sy = p.hy + oy;
    let ln = 0, lf = 0; KEYS.forEach((k, i) => { const g = gauss(t, k - .04, .045); if (i % 2) ln += g; else lf += g; });
    const [nu, nf] = ik(sx, sy, 902 + 4 * ln, 731 - 7 * ln), [fu, ff] = ik(sx - 4, sy, 925, 729 - 7 * lf);
    p.nU = lerp(p.nU, nu - p.lean, h); p.nF = lerp(-14, nf - nu, h); p.fU = lerp(p.fU, fu - p.lean, h); p.fF = lerp(-14, ff - fu, h);
  }
  return p;
}
const STEPS = []; { let last = null; for (let t = .2; t < 3.5; t += .002) { const x = pose(t).hx, k = Math.floor((x + 170) / 85 + .5); if (last !== null && k > last) STEPS.push(+t.toFixed(3)); last = k; } }
function drawFig(t) {
  const p = pose(t);
  fig.root.style.display = p.hx < -260 ? 'none' : 'block';
  applyPose(fig, p);
}
function applyPose(fig, p) {
  const tr = (g, s) => g.setAttribute('transform', s);
  tr(fig.hip, `translate(${p.hx.toFixed(2)},${p.hy.toFixed(2)})`);
  for (const g of [fig.back, fig.torso, fig.front]) tr(g, `rotate(${p.lean.toFixed(3)})`);
  for (const [arm, U, F] of [[fig.nearArm, p.nU, p.nF], [fig.farArm, p.fU, p.fF]]) {
    tr(arm.sh, `translate(${SHOULDER[0] - (arm === fig.farArm ? 4 : 0)},${SHOULDER[1]})`); tr(arm.up, `rotate(${U.toFixed(3)})`); tr(arm.fo, `translate(0,${UA}) rotate(${F.toFixed(3)})`);
  }
  for (const [leg, T2, K] of [[fig.nearLeg, p.nT, p.nK], [fig.farLeg, p.fT, p.fK]]) { tr(leg.th, `rotate(${T2.toFixed(3)})`); tr(leg.sn, `translate(0,${TH}) rotate(${K.toFixed(3)})`); }
  tr(fig.head, `translate(4,-150) rotate(${(p.tilt || 0).toFixed(3)})`);
}

// ---------------------------------------------------------------- the paper scene: one picture, two kinds of model
const POOL = [960, 567.8];  // the leaker's pool in the picture, where the hole in the g lands
el('pic', 500, 320, b => { b.img('assets/finish_ep0456_606.png', 10, 10, 480, 300); b.rect(5, 5, 490, 310, { w: 5 }); }, paperIn, 4);
el('q', 1700, 90, b => { b.text(0, 0, 'Is the finish on this part good or defective?', { size: 62, w: 1700, align: 'center' }); }, paperIn);
el('src1', 1700, 40, b => { b.text(0, 0, 'A real frame from my simulator log: sort log test03, episode 456, decision 606', { size: 26, w: 1700, align: 'center', color: GREY, font: 'mono' }); }, paperIn);
let goodChars = [], confVLM = null;
el('vlm', 660, 680, b => {
  b.rect(4, 4, 652, 672, { w: 5 }); b.fill([[0, 0], [660, 0], [660, 680], [0, 680]], INK, .035);
  b.text(32, 16, 'VLM', { size: 88 });
  b.text(34, 116, 'vision-language model', { size: 30, color: GREY, font: 'mono' });
  b.text(32, 160, 'reads the picture, writes words', { size: 44, w: 600 });
  b.text(34, 274, 'Qwen3.8-27B', { size: 36, font: 'mono' });
  b.line([[30, 330], [630, 330]], { w: 3, color: GREY });
  b.text(32, 340, 'asked for one word, it wrote:', { size: 38, color: GREY, w: 600 });
  const d = document.createElement('div'); d.className = 'tx mono'; Object.assign(d.style, { left: '32px', top: '392px', fontSize: '96px', color: INK });
  d.innerHTML = [...'defective'].map(c => `<span>${c}</span>`).join(''); b.root.appendChild(d); goodChars = [...d.children];
  b.text(32, 520, 'asked how sure, from 0 to 1:', { size: 38, color: GREY, w: 600 });
  confVLM = b.text(32, 566, '0.95', { size: 84, font: 'mono', manual: true });
}, paperIn, 3);
let bars = [], vals = [];
el('vdm', 660, 680, b => {
  b.rect(4, 4, 652, 672, { w: 5, color: ORANGE }); b.fill([[0, 0], [660, 0], [660, 680], [0, 680]], ORANGE, .06);
  b.text(32, 16, 'VDM', { size: 88, color: ORANGE });
  b.text(34, 116, 'vision decision model', { size: 30, color: GREY, font: 'mono' });
  b.text(32, 160, 'reads the picture, gives each option a probability', { size: 44, w: 600, lh: 1.05 });
  b.text(34, 274, 'd3 27B vision', { size: 36, font: 'mono', color: ORANGE });
  b.line([[30, 330], [630, 330]], { w: 3, color: GREY });
  b.text(32, 340, 'the two options, scored:', { size: 38, color: GREY, w: 600 });
  [['good', .9406943731630042, '0.94'], ['defective', .0593056268369958, '0.06']].forEach(([lab, p, s], i) => {   // test03_finish arms.jsonl, decision 169
    const y = 410 + i * 100;
    b.text(32, y + 4, lab, { size: 38, font: 'mono' });
    b.rect(262, y, 220, 52, { w: 3.5 });
    const r = document.createElementNS(NS, 'rect'); Object.assign(r.style, { fill: ORANGE, opacity: .8 }); r.setAttribute('x', 264); r.setAttribute('y', y + 3); r.setAttribute('height', 46); r.setAttribute('width', 0); b.svg.appendChild(r);
    bars.push([r, p]); vals.push(b.text(500, y - 2, s, { size: 50, font: 'mono', manual: true }));
  });
}, paperIn, 3);
el('arrowL', 120, 60, b => { b.arrow(76, 30, 18, 30, { w: 5 }); }, paperIn);
el('arrowR', 120, 60, b => { b.arrow(8, 30, 64, 30, { w: 5, color: ORANGE }); }, paperIn);
el('same', 400, 200, b => { b.text(0, 0, 'Same 27B base: d3 is built on Qwen3.8-27B', { size: 36, w: 400, align: 'center', color: GREY, lh: 1.08 }); }, paperIn);
el('shadow', 400, 40, b => { b.text(0, 0, 'The dark patches are shadows, not paint.', { size: 25, w: 400, align: 'center', color: GREY, font: 'mono' }); }, paperIn);
el('truth', 400, 140, b => { b.text(0, 0, 'the truth, from the simulator:', { size: 32, w: 400, align: 'center', color: GREY }); b.text(0, 42, 'good', { size: 76, w: 400, align: 'center' }); }, paperIn);
el('ringVLM', 660, 680, b => { b.line(roundRectPts(12, 396, 540, 118, 34), { color: RED, w: 7, amp: 2.5, close: true }); }, ELS.vlm.root, 5);
el('ringVDM', 660, 680, b => { b.line(roundRectPts(16, 398, 626, 78, 30), { color: GREEN, w: 7, amp: 2.5, close: true }); }, ELS.vdm.root, 5);
el('foot', 1740, 80, b => { b.text(0, 0, 'One picture, picked to show the difference. Across all 270 finish pictures in this log, both are right about as often (d3 93%, Qwen 94 to 95%), but sure and wrong at 0.9 or more: d3 once, Qwen 6 to 12 times. Exploratory, selection data (VALIDATION.md, "Qwen3.8-27B on the finish question").', { size: 23, w: 1760, color: GREY, font: 'mono', lh: 1.25 }); }, paperIn);
el('better', 1700, 190, b => { b.text(0, 0, 'Which is better?', { size: 150, w: 1700, align: 'center', wordLen: 160 }); }, paperIn, 6);

// positions of the paper scene: [t, x, y, scale, opacity]
const picAt = (cx, cy, s) => [cx - 250 * s, cy - 160 * s, s];
const B2 = A + 3.0, B5 = A + 1000;   // B5: the old shrink to "Which is better?", pushed out of this version
const PAPER_TRACKS = () => ({
  pic: [[0, ...picAt(960, 560, 1.2), 1], [B2, ...picAt(960, 560, 1.2), 1], [B2 + 1.2, ...picAt(960, 520, .72), 1], [B5, ...picAt(960, 520, .72), 1], [B5 + 1.4, ...picAt(960, 715, .45), 1]],
  q: [[0, 110, 130, 1, 1], [B2, 110, 130, 1, 1], [B2 + 1.2, 110, 88, .95, 1], [B5, 110, 88, .95, 1], [B5 + .6, 110, 88, .95, 0]],
  src1: [[0, 110, 830, 1, 1], [B2, 110, 830, 1, 1], [B2 + 1.2, 110, 975, 1, 1], [A + 13.8, 110, 975, 1, 1], [A + 14.3, 110, 975, 1, 0]],
  vlm: [[0, 90, 250, 1, 1], [B5, 90, 250, 1, 1], [B5 + 1.4, 420, 518, .6, 1]],
  vdm: [[0, 1170, 250, 1, 1], [B5, 1170, 250, 1, 1], [B5 + 1.4, 1104, 518, .6, 1]],
  arrowL: [[0, 742, 490, 1, 1], [B5, 742, 490, 1, 1], [B5 + .5, 742, 490, 1, 0]],
  arrowR: [[0, 1100, 490, 1, 1], [B5, 1100, 490, 1, 1], [B5 + .5, 1100, 490, 1, 0]],
  same: [[0, 760, 744, 1, 1], [B5, 760, 744, 1, 1], [B5 + .5, 760, 744, 1, 0]],
  shadow: [[0, 760, 664, 1, 1]],
  truth: [[0, 760, 196, 1, 1], [B5, 760, 196, 1, 1], [B5 + .5, 760, 196, 1, 0]],
  ringVLM: [[0, 0, 0, 1, 1]], ringVDM: [[0, 0, 0, 1, 1]],
  foot: [[0, 80, 984, 1, 1]],
  better: [[0, 110, 290, 1, 1]],
});
const PAPER_DRAW = {
  pic: null, q: [A + .3, 1.2], src1: [A + 1.2, 1.0],
  vlm: [B2 + .4, 1.6], arrowL: [B2 + 1.0, .4], vdm: [A + 8.2, 1.6], arrowR: [A + 8.6, .4], same: [A + 9.4, 1.0],
  truth: [A + 13.6, .9], shadow: [A + 14.6, .9], ringVLM: [A + 14.5, .6], ringVDM: [A + 15.1, .6], foot: [A + 14.6, 1.6], better: [B5 + 1.0, .9],
};
const T_GOOD = A + SHIFT + 5.4, T_CONF = A + SHIFT + 7.0, T_BARS = A + SHIFT + 10.2;
const CMP_DX = 4900 - 960;  // the comparison sits round x 4900 on the canvas: the title card before it, the ladder past it
for (const id in PAPER_DRAW) if (PAPER_DRAW[id]) PAPER_DRAW[id] = [PAPER_DRAW[id][0] + SHIFT, PAPER_DRAW[id][1]];
const CMP_TRACKS = () => { const tr = PAPER_TRACKS(), out = {}; for (const id in tr) out[id] = tr[id].map(([t, x, y, sc, o], i) => [i === 0 && t === 0 ? 0 : t + SHIFT, x + (id.startsWith('ring') ? 0 : CMP_DX), y, sc, o]); return out; };


// ================================================================ after the intro: SCRIPT.md beats 1 to 3
// The paper is one canvas: the intro sits round (960, 540); the question and the cloud round C1; the ladder round C2.
const C2 = [7000, 540], CT = [2700, 540];   // CT: the Phase 1 title card
const FLOOR = 1000;                   // the paper floor under dup in the new scenes
const DUP_S = 1.2, DUP_X = 230, DUP_HIP = FLOOR - 192 * DUP_S;
const HEAD = [DUP_X + 4 * DUP_S, DUP_HIP - 196 * DUP_S];   // the top of his hood, where the models bonk
el('floor2', 3100, 30, b => { b.line([[0, 10], [3100, 10]], { w: 4, amp: 2.5 }); }, paperIn, 2);
const fig2 = makeFig(paperIn, 6, 'dup2');

// beat 1: the thought bubble
const BUB = { x: POOL[0] - 600, y: POOL[1] - 250 };   // the bubble's centre is where the dive through the g lands
const cloudPts = (cx, cy, rx, ry, bumps, depth, n = 120) => { const p = []; for (let i = 0; i < n; i++) { const a = i / n * Math.PI * 2, k = 1 + depth * Math.abs(Math.sin(a * bumps / 2)); p.push([cx + rx * Math.cos(a) * k, cy + ry * Math.sin(a) * k]); } return p; };
el('bubble', 1200, 560, b => {
  const pts = cloudPts(600, 250, 470, 175, 14, .09);
  b.fill(pts, '#ffffff', .9); b.line(pts, { w: 5, close: true, amp: 1.2 });
  b.ellipse(10, 215, 26, 20, { w: 4, amp: .8 }); b.ellipse(-62, 258, 15, 12, { w: 4, amp: .6 });
}, paperIn, 5);
el('bubtext', 1200, 560, b => {
  b.text(190, 118, 'Maybe decision models would be well suited for robotics?', { size: 56, w: 820, align: 'center', lh: 1.05 });
  b.text(190, 268, "After all, aren't robots just making a series of decisions?", { size: 44, w: 820, align: 'center', lh: 1.05, color: '#555' });
}, paperIn, 6);
el('asked', 900, 40, b => { b.text(0, 0, "my question at the hackathon's launch meeting", { size: 26, color: GREY, font: 'mono' }); }, paperIn, 5);

// beat 2: the cloud, raining decision models (real names from the decision-model watch log, state/HISTORY.md)
const CLOUD = [DUP_X + 60, -230];
el('cloud', 900, 400, b => {
  // the hugging face, as a cloud (dup, Nov 6: match the 🤗 emoji so it reads as Hugging Face; our drawing, not their logo)
  const pts = cloudPts(450, 190, 360, 120, 10, .16);
  b.fill(pts, '#e9eef3', 1); b.line(pts, { w: 6, close: true, amp: 1.5 });
  b.line(ellipsePts(392, 168, 22, 18, 12, Math.PI, .5), { w: 8, amp: .3 });   // happy closed eyes: two arches
  b.line(ellipsePts(508, 168, 22, 18, 12, Math.PI, .5), { w: 8, amp: .3 });
  const mouth = [[388, 200], [512, 200], ...ellipsePts(450, 200, 62, 50, 16, 0, .5).slice(1)];   // a wide open grin
  b.solid(mouth, '#6b2d1e', { w: 5 });
  b.solid(ellipsePts(450, 236, 30, 13, 14), '#e8735f', { w: 0.1 });                               // the tongue
  const HAND = '#e9eef3';   // cloud hands, same colour as the cloud (dup, Nov 6)
  const turn = (pts, cx, cy, a) => pts.map(([x, y]) => { const c = Math.cos(a), s2 = Math.sin(a); return [cx + x * c - y * s2, cy + x * s2 + y * c]; });
  for (const [cx, cy, a, side] of [[300, 300, -.42, 1], [600, 300, .42, -1]]) {   // open hands at the bottom corners, fingers pointing out (dup, Nov 6), thumbs in
    for (const [fx, top] of [[-33, -64], [-14, -74], [5, -72], [24, -62]]) b.solid(turn(roundRectPts(fx, top, 17, 62, 8.5), cx, cy, a), HAND, { w: 3.5 });
    b.solid(turn(roundRectPts(-10, -46, 20, 58, 10), cx + side * 28, cy + 4, side * .95), HAND, { w: 3.5 });   // thumb, on the face's side
    b.solid(turn(ellipsePts(-4, 4, 44, 36, 20), cx, cy, a), HAND, { w: 4 });                                            // palm
  }
}, paperIn, 8);
const MODELS = ['Decider 2B', 'd3 27B', 'Intern-Decision', 'StartLux', 'Winnow-12B', 'Nimble 9B', 'Vega-27B', 'Lux-9B', 'Zyot Decider', 'decisio',
  'Jeff-1.1', 'Sieve-31B', 'Wald 4B', 'Elephant-1', 'Mammoth-1', 'Quyet 1.0', 'Lev-3B', 'Vela 3.0', 'GLiDE', 'KnowLine', 'Chon-1.0', 'Drex Max',
  'pplx-decider', 'd1-2B', 'Torchcast', 'bekko', 'Matilda', 'Von-1.0'];
const G_PX = 2600;  // gravity, px/s^2 on the paper
const drops = [];
{
  const r = mulberry32(hash('rain')), all = [['Jev · Sep 15', E0 + 11.0, true]].concat(MODELS.map((m, i) => [m, E0 + 13.2 + 8.6 * Math.pow(i / MODELS.length, .62), false]));
  // the pile: rows of boxes either side of his feet, filled outward from him
  const rows = { L: [], R: [] }, BH = 50;
  all.forEach(([name, t0, big], i) => {
    const w = Math.max(120, name.length * (big ? 25 : 17) + 36), h = big ? 64 : BH;
    const side = i === 0 ? 'R' : (i % 2 ? 'L' : 'R'), row = rows[side];
    let lvl = 0; while ((row[lvl] || 0) + w > 560) lvl++;
    const off = row[lvl] || 0; row[lvl] = off + w + 6;
    const px = side === 'R' ? DUP_X + 70 + off + w / 2 + lvl * 28 : DUP_X - 70 - off - w / 2 - lvl * 28;
    const py = FLOOR - h / 2 - lvl * (BH + 2);
    const bonk = i < 7;      // the first few bounce off his head
    const x0 = bonk ? HEAD[0] + (r() - .5) * 30 : CLOUD[0] + (r() - .5) * 520, y0 = CLOUD[1] + 60;  // every model leaves from inside the cloud
    const yHit = bonk ? HEAD[1] - h / 2 : py, d1 = Math.sqrt(2 * (yHit - y0) / G_PX);
    const d2 = bonk ? .75 + r() * .25 : 0;
    drops.push({ name, big, w, h, t0, x0, y0, d1, d2, bonk, px, py, yHit, spin: (r() < .5 ? -1 : 1) * (300 + r() * 360), rest: (r() - .5) * 10 });
  });
}
for (const d of drops) {
  d.e = el('m_' + d.name, d.w, d.h, b => {
    b.fill([[0, 0], [d.w, 0], [d.w, d.h], [0, d.h]], d.big ? '#fde8d4' : '#ffffff', 1);
    b.rect(2, 2, d.w - 4, d.h - 4, { w: 3.5, color: d.big ? ORANGE : INK, amp: .8 });
    b.text(0, d.big ? 12 : 9, d.name, { size: d.big ? 36 : 27, w: d.w, align: 'center', font: 'mono', color: d.big ? ORANGE : INK, manual: true });
  }, paperIn, 7);
  reveal(d.e, 1);
}
function dropAt(d, t) {  // centre x, y and rotation of a falling model at time t (null before it leaves the cloud)
  const u = t - d.t0; if (u < 0) return null;
  if (u < d.d1) return [d.x0 + (d.bonk ? 0 : (d.px - d.x0) * u / d.d1), d.y0 + .5 * G_PX * u * u, 0];
  if (!d.bonk) return [d.px, d.py, d.rest * E.out(clamp((u - d.d1) / .15))];
  const v = u - d.d1;
  if (v < d.d2) {  // off his head, on a ballistic arc to its spot in the pile
    const vx = (d.px - d.x0) / d.d2, vy = (d.py - d.yHit - .5 * G_PX * d.d2 * d.d2) / d.d2;
    return [d.x0 + vx * v, d.yHit + vy * v + .5 * G_PX * v * v, d.spin * v / d.d2 + d.rest * v / d.d2];
  }
  return [d.px, d.py, d.rest];
}
const HITS = drops.filter(d => d.bonk).map(d => d.t0 + d.d1);
el('rainfoot', 1500, 40, b => { b.text(0, 0, 'Names from my decision-model watch log, Sep 15 to Nov 6, 2026', { size: 26, color: GREY, font: 'mono' }); }, paperIn, 5);

// beat 3: the ladder he tried first, small models up to big ones, then a person
const LAD = { x: C2[0] - 120, top: 250, bot: 930, w: 240 };
const RUNGS = [['2B', 840], ['4B', 710], ['9B', 580], ['27B', 450], ['a person', 320]];
el('ltitle', 1800, 100, b => { b.text(0, 0, "What's the best result I could get?", { size: 72, w: 1800, align: 'center' }); }, paperIn, 4);
el('lrails', 300, 760, b => {
  b.line([[30, 0], [30, 700]], { w: 6 }); b.line([[270, 0], [270, 700]], { w: 6 });
  for (const [, y] of RUNGS.slice(3)) b.line([[30, y - LAD.top], [270, y - LAD.top]], { w: 6 });
}, paperIn, 4);
const smallRung = (lab, y) => el('rung' + lab, 620, 70, b => {
  b.line([[30, 30], [270, 30]], { w: 6 });
  b.text(320, 6, lab + ' model', { size: 44, font: 'mono' });
}, paperIn, 4);
const SMALL = RUNGS.slice(0, 3).map(([lab, y]) => ({ lab, y, e: smallRung(lab, y) }));
const crossFor = (lab) => el('x' + lab, 620, 70, b => { b.line([[10, 6], [290, 58]], { w: 7, color: RED, amp: 1.5 }); b.line([[10, 58], [290, 6]], { w: 7, color: RED, amp: 1.5 }); }, paperIn, 5);
SMALL.forEach(s => { s.x = crossFor(s.lab); });
el('l27', 620, 70, b => { b.text(0, 0, 'd3 27B', { size: 50, font: 'mono', color: ORANGE }); }, paperIn, 4);
el('lperson', 620, 70, b => { b.text(0, 0, 'a person', { size: 50 }); }, paperIn, 4);
el('lfirst', 560, 200, b => { b.text(0, 0, 'Small models first, a bigger one only when unsure', { size: 44, w: 520, lh: 1.08 }); b.arrow(470, 150, 540, 150, { w: 4 }); }, paperIn, 4);
el('lwrong', 560, 120, b => { b.text(0, 0, 'acted wrong too often', { size: 50, color: RED, w: 540 }); }, paperIn, 5);
el('lwin', 560, 220, b => { b.text(0, 0, 'It came down to one: d3\u00a027B, then a person', { size: 56, w: 540, lh: 1.08 }); }, paperIn, 5);
el('lfoot', 1700, 70, b => { b.text(0, 0, 'Selection runs (test03), exploratory: 2B > 4B > 9B > 27B cascades halved the hand-offs but acted wrong more often (VALIDATION.md, "small-model-first cascades")', { size: 24, w: 1700, color: GREY, font: 'mono', lh: 1.25 }); }, paperIn, 5);

// the Phase 1 title card (dup, Nov 6): slide 3's cell from the film, the MH3 rolling in from the left on its wheeled base
const BLUE = '#2f5ec4';
el('p1title', 1700, 120, b => { b.text(0, 0, 'Phase 1: Proof of concept', { size: 96, w: 1700, align: 'center' }); }, paperIn, 4);
el('p1sub', 1700, 70, b => { b.text(0, 0, 'Painted part inspection', { size: 54, w: 1700, align: 'center', color: GREY }); }, paperIn, 4);
el('p1table', 1240, 200, b => { b.line([[0, 20], [1240, 20]], { w: 6 }); b.line([[60, 20], [60, 190]], { w: 5 }); b.line([[1180, 20], [1180, 190]], { w: 5 }); }, paperIn, 3);
el('p1robot', 340, 330, b => {   // film.js robot(), wheels drawn apart so they can turn
  const x = 10, y = 0;
  b.rect(x + 70, y + 10, 90, 72); b.line([[x + 95, y + 40], [x + 103, y + 40]], { w: 5 }); b.line([[x + 127, y + 40], [x + 135, y + 40]], { w: 5 });
  b.line([[x + 115, y + 82], [x + 115, y + 100]]); b.rect(x + 60, y + 100, 110, 140);
  b.line([[x + 170, y + 120], [x + 235, y + 150], [x + 300, y + 136]], { w: 4.5 });
  b.line([[x + 300, y + 136], [x + 316, y + 124]]); b.line([[x + 300, y + 136], [x + 318, y + 146]]);
  b.line([[x + 60, y + 120], [x + 25, y + 190], [x + 40, y + 235]], { w: 4.5 });
  b.line([[x + 115, y + 240], [x + 115, y + 268]]); b.rect(x + 40, y + 268, 150, 50);
}, paperIn, 5);
const wheelEl = (id) => el(id, 44, 44, b => { b.ellipse(22, 22, 18, 18, { w: 4, amp: .5 }); b.line([[10, 22], [34, 22]], { w: 3, amp: .2 }); b.line([[22, 10], [22, 34]], { w: 3, amp: .2 }); }, paperIn, 5);
const P1W = [wheelEl('p1wheelA'), wheelEl('p1wheelB')];
P1W.forEach(w => reveal(w, 1));
el('p1part', 70, 80, b => { b.fill([[8, 8], [54, 8], [54, 64], [8, 64]], BLUE, .35); b.rect(8, 8, 46, 56, { color: BLUE, w: 5 }); b.rect(18, 20, 26, 32, { color: GREY, w: 3 }); }, paperIn, 4);
const p1tray = (id, color, label) => el(id, 280, 100, b => { b.fill([[10, 10], [22, 80], [258, 80], [270, 10]], color, .22); b.line([[10, 10], [22, 80], [258, 80], [270, 10]], { color, w: 6 }); b.text(10, -42, label, { size: 40, color, w: 260, align: 'center' }); }, paperIn, 4);
p1tray('p1trayG', GREEN, 'good'); p1tray('p1trayO', ORANGE, 'rework');
const P1 = { left: CT[0] - 960, rs: 1.25 };            // slide 3's layout, placed on this card
const ROBOT_IN = [TT + .6, TT + 2.6];                    // rolls in from off the left edge
function drawTitleCard(t) {
  const L = P1.left, u = E.out(clamp((t - ROBOT_IN[0]) / (ROBOT_IN[1] - ROBOT_IN[0])));
  const x0 = L - 520, x1 = L + 280, rx = lerp(x0, x1, u), ry = 520 + (u > 0 && u < 1 ? -3 * Math.abs(Math.sin(u * 18)) : 0);
  const shown = t >= ROBOT_IN[0] - .05;
  ELS.p1robot.root.style.display = shown ? 'block' : 'none'; reveal(ELS.p1robot, 1);
  ELS.p1robot.root.style.transform = `translate(${rx.toFixed(2)}px,${ry.toFixed(2)}px) scale(${P1.rs})`;
  const turn = (rx - x0) / (18 * P1.rs) * 180 / Math.PI;   // rolling: angle = distance / radius
  [[82, P1W[0]], [168, P1W[1]]].forEach(([wx, w]) => {
    w.root.style.display = shown ? 'block' : 'none';
    w.root.style.transformOrigin = '22px 22px';
    w.root.style.transform = `translate(${(rx + (wx - 22) * P1.rs).toFixed(2)}px,${(ry + (330 - 22) * P1.rs).toFixed(2)}px) scale(${P1.rs}) rotate(${turn.toFixed(1)}deg)`;
  });
  place(ELS.p1title, t, [[0, CT[0] - 850, 90, 1, 1]], AFTER_DRAW.p1title);
  place(ELS.p1sub, t, [[0, CT[0] - 850, 215, 1, 1]], AFTER_DRAW.p1sub);
  place(ELS.p1table, t, [[0, L + 520, 760, 1, 1]], AFTER_DRAW.p1table);
  place(ELS.p1part, t, [[0, L + 852, 706, 1, 1]], AFTER_DRAW.p1part);
  place(ELS.p1trayG, t, [[0, L + 1090, 694, 1, 1]], AFTER_DRAW.p1trayG);
  place(ELS.p1trayO, t, [[0, L + 1430, 694, 1, 1]], AFTER_DRAW.p1trayO);
}

// the paper camera: [t, zoom, cx, cy]
const PCAM = [[0, 1, 960, 540], [A, 1, 960, 540], [Q1 - .2, .95, 880, 560],
  [Q1 + 1.4, .72, 320, 300], [Q2 - .4, .74, 320, 320], [TT + .2, 1, CT[0], 540], [T_CMP - .2, 1.015, CT[0], 540], [T_CMP + 1.4, 1, 960 + CMP_DX, 540], [QL - .2, 1.012, 960 + CMP_DX, 540], [QL + 1.4, 1, C2[0], 540], [T_END, 1.02, C2[0], 540]];
function paperCam(t) {
  if (t <= PCAM[0][0]) return { z: PCAM[0][1], cx: PCAM[0][2], cy: PCAM[0][3] };
  for (let i = 0; i < PCAM.length - 1; i++) {
    const a = PCAM[i], b2 = PCAM[i + 1];
    if (t < b2[0]) { const u = E.io((t - a[0]) / (b2[0] - a[0])); return { z: Math.exp(lerp(Math.log(a[1]), Math.log(b2[1]), u)), cx: lerp(a[2], b2[2], u), cy: lerp(a[3], b2[3], u) }; }
  }
  const k = PCAM[PCAM.length - 1]; return { z: k[1], cx: k[2], cy: k[3] };
}

// when each new element draws in: [start, seconds]; the scratch cues come from these and the intro's own
const AFTER_DRAW = {
  floor2: [E0 + .3, .8], bubble: [Z0 + .4, 1.5], bubtext: [E0 + .5, 2.4], asked: [E0 + 5.4, .8],
  cloud: [Q1 + .9, 1.4], rainfoot: [Q1 + 10.5, 1.0],
  ltitle: [QL + .7, 1.2], lrails: [QL + 3.0, 1.2], lfirst: [QL + 3.6, 1.2], rung2B: [QL + 3.9, .5], rung4B: [QL + 4.3, .5], rung9B: [QL + 4.7, .5],
  l27: [QL + 5.2, .6], lperson: [QL + 5.6, .6], x2B: [QL + 9.0, .4], x4B: [QL + 9.4, .4], x9B: [QL + 9.8, .4], lwrong: [QL + 10.0, .8],
  lwin: [QL + 12.6, 1.2], lfoot: [QL + 13.6, 1.4],
  p1title: [TT + .3, 1.1], p1sub: [TT + 1.3, .8], p1table: [TT + 1.9, .9], p1part: [TT + 2.7, .5], p1trayG: [TT + 3.0, .6], p1trayO: [TT + 3.4, .6],
};
const SCRATCH = [...Object.entries(ROOM_DRAW).map(([, [a, d]]) => [a, d]), ...Object.values(PAPER_DRAW).filter(Boolean), ...Object.values(AFTER_DRAW)];
window.SFX.push(...STEPS.map(t => ({ t, kind: 'step' })), ...SCRATCH.map(([t, d]) => ({ t, kind: 'scratch', dur: Math.min(d, 1.4) })),
  { t: ROBOT_IN[0], kind: 'roll', dur: ROBOT_IN[1] - ROBOT_IN[0] },
  ...HITS.map(t => ({ t, kind: 'bonk' })), ...drops.map(d => ({ t: d.t0 + d.d1 + d.d2, kind: 'clatter' })));

function drawAfter(t) {
  drawTitleCard(t);
  // dup on the paper: standing; hand to his chin while he asks; a duck of the head at each bonk
  fig2.root.style.display = t >= E0 + .2 ? 'block' : 'none';
  fig2.root.style.opacity = E.out(clamp((t - E0 - .2) / .6));
  fig2.root.style.transform = `translate(${DUP_X}px,${DUP_HIP}px) scale(${DUP_S})`;
  const chin = E.io(clamp((t - E0 - 1.0) / .7)) * (1 - E.io(clamp((t - Q1 + .3) / .6)));
  const bonk = HITS.reduce((s, h) => s + (t >= h ? Math.exp(-(t - h) / .12) * Math.min(1, (t - h) / .03) : 0), 0);
  const look = E.io(clamp((t - Q1 - .6) / .8)) * (1 - E.io(clamp((t - Q2 + .6) / .6)));
  const p = { hx: 0, hy: 0, lean: 2 * Math.sin(t * 1.3) + 9 * Math.min(1, bonk), nT: -4, nK: 6, fT: 5, fK: 4, nU: 6, nF: -12, fU: -5, fF: -10, tilt: -8 * chin - 16 * look + 10 * Math.min(1, bonk) };
  if (chin > 0) { const [ox, oy] = rot(SHOULDER[0], SHOULDER[1], p.lean * Math.PI / 180), [u2, f2] = ik(ox, oy, 38, -120); p.nU = lerp(p.nU, u2 - p.lean, chin); p.nF = lerp(p.nF, f2 - u2, chin); }
  applyPose(fig2, p);
  place(ELS.floor2, t, [[0, -1400, FLOOR, 1, 1]], AFTER_DRAW.floor2);
  const bubOut = 1 - E.out(clamp((t - Q1 - .2) / .5));
  place(ELS.bubble, t, [[0, BUB.x, BUB.y, 1, bubOut]], AFTER_DRAW.bubble);
  place(ELS.bubtext, t, [[0, BUB.x, BUB.y, 1, bubOut]], AFTER_DRAW.bubtext);
  place(ELS.asked, t, [[0, POOL[0] - 330, POOL[1] + 230, 1, bubOut]], AFTER_DRAW.asked);
  place(ELS.cloud, t, [[0, CLOUD[0] - 450, CLOUD[1] - 190, 1, 1]], AFTER_DRAW.cloud);
  place(ELS.rainfoot, t, [[0, DUP_X - 900, FLOOR + 22, 1, 1]], AFTER_DRAW.rainfoot);
  for (const d of drops) {
    const q = dropAt(d, t); d.e.root.style.display = q ? 'block' : 'none'; if (!q) continue;
    d.e.root.style.transform = `translate(${(q[0] - d.w / 2).toFixed(2)}px,${(q[1] - d.h / 2).toFixed(2)}px) rotate(${q[2].toFixed(2)}deg)`;
    d.e.root.style.transformOrigin = `${d.w / 2}px ${d.h / 2}px`;
  }
  // the ladder
  place(ELS.ltitle, t, [[0, C2[0] - 900, 70, 1, 1]], AFTER_DRAW.ltitle);
  place(ELS.lrails, t, [[0, LAD.x - 30, LAD.top, 1, 1]], AFTER_DRAW.lrails);
  place(ELS.lfirst, t, [[0, LAD.x - 640, 560, 1, 1], [QL + 11.6, LAD.x - 640, 560, 1, 1], [QL + 12.2, LAD.x - 640, 560, 1, 0]], AFTER_DRAW.lfirst);
  for (const s2 of SMALL) {
    const fall = E.io(clamp((t - QL + 0 - 11.2 - (s2.lab === '2B' ? 0 : s2.lab === '4B' ? .15 : .3)) / .9));
    const keys = [[0, LAD.x - 30, s2.y - 30 + 240 * fall * fall, 1, 1 - fall]];
    place(s2.e, t, keys, AFTER_DRAW['rung' + s2.lab]);
    place(s2.x, t, [[0, LAD.x - 30, s2.y - 30 + 240 * fall * fall, 1, 1 - fall]], AFTER_DRAW['x' + s2.lab]);
  }
  place(ELS.l27, t, [[0, LAD.x + 290, 420, 1, 1]], AFTER_DRAW.l27);
  place(ELS.lperson, t, [[0, LAD.x + 290, 290, 1, 1]], AFTER_DRAW.lperson);
  place(ELS.lwrong, t, [[0, LAD.x + 640, 690, 1, 1], [QL + 11.2, LAD.x + 640, 690, 1, 1], [QL + 11.9, LAD.x + 640, 690, 1, 0]], AFTER_DRAW.lwrong);
  place(ELS.lwin, t, [[0, LAD.x - 640, 520, 1, 1]], AFTER_DRAW.lwin);
  place(ELS.lfoot, t, [[0, C2[0] - 850, 990, 1, 1]], AFTER_DRAW.lfoot);
}

// ---------------------------------------------------------------- the camera
const camAt = (t) => {  // before the dive: zoom and a centre in world coordinates
  const wide = { Z: 1.5, cx: 760, cy: 662 }, mon = { Z: 5.4, cx: 1150, cy: 613.75 };
  if (t < 4.7) return { Z: wide.Z * (1 + .025 * E.io(t / 4.7)), cx: wide.cx, cy: wide.cy };
  const u = E.io(clamp((t - 4.7) / 2.7)), z0 = wide.Z * 1.025;
  if (u < 1) return { Z: Math.exp(lerp(Math.log(z0), Math.log(mon.Z), u)), cx: lerp(wide.cx, mon.cx, u), cy: lerp(wide.cy, mon.cy, u) };
  return { Z: mon.Z * (1 + .03 * E.io(clamp((t - 7.4) / (Z0 - 7.4)))), cx: mon.cx, cy: mon.cy };
};
const setWorld = (Sx, Sy, Px, Py, Z) => { world.style.transform = gtop.style.transform = `translate(${(Sx - Px * Z).toFixed(3)}px,${(Sy - Py * Z).toFixed(3)}px) scale(${Z.toFixed(5)})`; };

// ---------------------------------------------------------------- one instant
window.seek = async (t) => {
  // the room and dup
  for (const id in ROOM) { const e = ELS[id], [x, y] = ROOM[id], [a, d] = ROOM_DRAW[id]; reveal(e, clamp((t - a) / d)); e.root.style.transform = `translate(${x}px,${y}px)`; e.root.style.display = 'block'; }
  term.style.opacity = E.out(clamp((t - .9) / .5));
  drawFig(t);
  // the terminal
  const n = KEYS.filter(k => t >= k).length, entered = t >= T_ENTER;
  typedSpan.textContent = TYPED.slice(0, n);
  pl.style.color = entered ? TERM_DIM : TERM_FG;
  cur1.style.display = entered ? 'none' : 'inline-block';
  cur1.style.opacity = (n > 0 && t < KEYS[KEYS.length - 1] + .25) || Math.floor(t * 1.8) % 2 === 0 ? 1 : 0;
  for (const r of rulesA) r.style.display = entered ? 'none' : 'block';
  for (const r of rulesB) r.style.display = entered ? 'block' : 'none';
  pl2.style.display = entered ? 'block' : 'none'; cur2.style.opacity = Math.floor(t * 1.8) % 2 === 0 ? 1 : 0;
  sp1.style.display = t >= T_SPIN && t < T_WIN ? 'block' : 'none';
  sp2.style.visibility = t >= T_WIN ? 'visible' : 'hidden';
  for (const sp of [sp1, sp2]) { sp.querySelector('.dots').textContent = '.'.repeat(1 + Math.floor(Math.max(0, t - T_SPIN) * 3) % 3); sp.querySelector('.star').style.transform = `rotate(${((t - T_SPIN) * 200).toFixed(1)}deg)`; }
  // the camera, and the dive through the g
  if (t < Z0) {
    const c = camAt(t); setWorld(960, 540, c.cx, c.cy, c.Z);
    world.style.display = 'block'; paper.style.display = 'none'; gtop.style.display = 'none';
  } else if (t < Z1) {
    const c = camAt(Z0 - 1e-6), k = (t - Z0) / (Z1 - Z0), [Px, Py] = G.P;
    const S0 = [960 + (Px - c.cx) * c.Z, 540 + (Py - c.cy) * c.Z], m = E.io(clamp(k / .8));
    const Z = Math.exp(lerp(Math.log(c.Z), Math.log(1400), Math.pow(k, 2.2)));
    const Sx = lerp(S0[0], POOL[0], m), Sy = lerp(S0[1], POOL[1], m);
    setWorld(Sx, Sy, Px, Py, Z); world.style.display = 'block'; gtop.style.display = 'block';
    gcopy.querySelector('.dots').textContent = sp2.querySelector('.dots').textContent; gcopy.querySelector('.star').style.transform = sp2.querySelector('.star').style.transform;
    // the paper scene shows only through the hole, arriving at the size of the hole and growing to its own
    const ox = Sx + (G.origin[0] - Px) * Z, oy = Sy + (G.origin[1] - Py) * Z, u = G.unit * Z;
    const mask = `url(${G.url})`, pos = `${ox.toFixed(2)}px ${oy.toFixed(2)}px`, size = `${(GM.cw * u).toFixed(2)}px ${(GM.chh * u).toFixed(2)}px`;
    Object.assign(paper.style, { display: 'block', maskImage: mask, webkitMaskImage: mask, maskPosition: pos, webkitMaskPosition: pos, maskSize: size, webkitMaskSize: size });
    const sc = 1 - Math.exp(-(G.r * Z) / 95);
    paperIn.style.transform = `translate(${(Sx - POOL[0] * sc).toFixed(3)}px,${(Sy - POOL[1] * sc).toFixed(3)}px) scale(${sc.toFixed(5)})`;
  } else {
    world.style.display = 'none'; gtop.style.display = 'none';
    Object.assign(paper.style, { display: 'block', maskImage: 'none', webkitMaskImage: 'none' });
    const c = paperCam(t);
    paperIn.style.transform = `translate(${(960 - c.cx * c.z).toFixed(3)}px,${(540 - c.cy * c.z).toFixed(3)}px) scale(${c.z.toFixed(5)})`;
  }
  // the paper scene's elements
  const tracks = CMP_TRACKS();
  for (const id in tracks) place(ELS[id], t, tracks[id], PAPER_DRAW[id]);
  goodChars.forEach((c, i) => { c.style.opacity = t >= T_GOOD + i * .11 ? 1 : 0; });
  const cv = E.out(clamp((t - T_CONF) / .4)); confVLM.querySelectorAll('.w').forEach(w2 => { w2.style.opacity = cv; });
  bars.forEach(([r, p], i) => { r.setAttribute('width', (216 * p * E.out(clamp((t - T_BARS - i * .25) / .9))).toFixed(2)); });
  vals.forEach((v, i) => { const o = E.out(clamp((t - T_BARS - .9 - i * .25) / .35)); v.querySelectorAll('.w').forEach(w2 => { w2.style.opacity = o; }); });
  drawAfter(t);
};

Promise.all([document.fonts.load('44px Patrick'), document.fonts.load('72px "Ubuntu Mono"'), ...[...document.images].map(i => i.decode().catch(() => {}))])
  .then(() => {
    G = measureG(); window.G_HOLE = G;
    if (DEBUG) { const d = document.createElement('div'); Object.assign(d.style, { position: 'absolute', left: (G.P[0] - 1) + 'px', top: (G.P[1] - 1) + 'px', width: '2px', height: '2px', background: '#0f0', zIndex: 9 }); world.appendChild(d); }
    window.ready = true;
  });
})();
