// The Kaggle film: slides are keyframes, every motion shot goes from slide n to slide n+1 (a "magic move").
// One engine, three entry pages: film.html (the whole animatic, footage as placeholder cards), slides.html (one
// second per slide, fully built: the stills), shot_NN_MM.html (one transition, the proof of technique).
// Everything is computed from t: no timers, no CSS transitions, seeded wobble. Numbers on screen: facts.md only.
(() => {
const W = 1920, H = 1080;
const INK = '#2b2b2b', GREEN = '#2e8b57', ORANGE = '#e07a1f', BLUE = '#2a6fd6', RED = '#c0392b', GREY = '#8a8a8a';
const clamp = (x, a = 0, b = 1) => x < a ? a : x > b ? b : x, lerp = (a, b, u) => a + (b - a) * u;
const E = { io: u => u <= 0 ? 0 : u >= 1 ? 1 : u < .5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2,
            out: u => u <= 0 ? 0 : u >= 1 ? 1 : 1 - Math.pow(1 - u, 3),
            o5: u => u <= 0 ? 0 : u >= 1 ? 1 : 1 - Math.pow(1 - u, 5) };
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const hash = s => { let h = 2166136261; for (const c of s) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return h >>> 0; };
const NS = 'http://www.w3.org/2000/svg';
const world = document.getElementById('world');

// ---------------------------------------------------------------- elements
// An element is a div at (x, y, scale) in stage space holding an SVG (local coords) and HTML text. Its parts are
// revealed in the order they were made, like a pen: strokes by length, words and images as fixed-length items.
const ELS = {};
function el(id, w, h, build) {
  const root = document.createElement('div'); root.className = 'el'; root.style.width = w + 'px'; root.style.height = h + 'px';
  const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('width', w); svg.setAttribute('height', h); svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  root.appendChild(svg); world.appendChild(root);
  const rnd = mulberry32(hash(id)), seq = [];
  const sk = (pts, close, amp = 2.0) => {
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
  const b = {
    path(d, o = {}) {
      const p = document.createElementNS(NS, 'path'); p.setAttribute('d', d);
      p.style.stroke = o.color || INK; p.style.strokeWidth = o.w || 4; p.style.fill = 'none'; svg.appendChild(p);
      const len = p.getTotalLength(); p.style.strokeDasharray = len + ' ' + (len + 10); seq.push({ kind: 'path', el: p, len }); return p;
    },
    line(pts, o = {}) { return b.path(sk(pts, o.close, o.amp), o); },
    rect(x, y, w2, h2, o = {}) { return b.line([[x, y], [x + w2, y], [x + w2, y + h2], [x, y + h2]], { ...o, close: true }); },
    circle(cx, cy, r, o = {}) { const pts = [], n = Math.max(14, Math.round(r / 4)); for (let i = 0; i < n; i++) { const a = i / n * Math.PI * 2 - Math.PI / 2; pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); } return b.line(pts, { ...o, close: true, amp: o.amp ?? Math.max(1.2, r * .02) }); },
    arc(cx, cy, r, a0, a1, o = {}) { const pts = [], n = 24; for (let i = 0; i <= n; i++) { const a = lerp(a0, a1, i / n); pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); } return b.line(pts, { ...o, amp: o.amp ?? 1.5 }); },
    arrow(x0, y0, x1, y1, o = {}) {
      b.line([[x0, y0], [x1, y1]], o); const a = Math.atan2(y1 - y0, x1 - x0), L = o.head || 26;
      b.line([[x1 - L * Math.cos(a - .45), y1 - L * Math.sin(a - .45)], [x1, y1], [x1 - L * Math.cos(a + .45), y1 - L * Math.sin(a + .45)]], { ...o, amp: 1 });
    },
    fill(pts, color, opacity = .18) {  // a tint, faded in as one item
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
      for (const w2 of d.querySelectorAll('.w')) seq.push({ kind: 'word', el: w2, len: o.wordLen || 70 });
      return d;
    },
    img(src, x, y, w2, h2) {
      const i = document.createElement('img'); i.src = src; i.className = 'im'; Object.assign(i.style, { left: x + 'px', top: y + 'px', width: w2 + 'px', height: h2 + 'px' });
      root.appendChild(i); seq.push({ kind: 'fade', el: i, len: 500, max: 1 }); return i;
    },
    sk, rnd, svg, root,
  };
  const api = build(b) || {};
  let total = 0; for (const s of seq) { s.start = total; total += s.len; }
  ELS[id] = { id, root, seq, total, w, h, update: api.update, enter: api.enter };
  return ELS[id];
}
function reveal(e, p) {  // p in [0, 1]: how far the pen has got through this element
  const L = p * e.total;
  for (const s of e.seq) {
    const u = clamp((L - s.start) / s.len);
    if (s.kind === 'path') s.el.style.strokeDashoffset = s.len * (1 - u);
    else if (s.kind === 'word') { const v = E.out(u); s.el.style.opacity = v; s.el.style.transform = `translateY(${(1 - v) * 12}px)`; }
    else s.el.style.opacity = E.out(u) * s.max;
  }
}

// ---------------------------------------------------------------- drawings
function robot(b, x, y, o = {}) {  // the MH3: a humanoid torso on a wheeled base (it has no legs); x, y = top left
  const c = o.color || INK;
  b.rect(x + 70, y + 10, 90, 72, { color: c });                                   // head
  b.line([[x + 95, y + 40], [x + 103, y + 40]], { color: c, w: 5 }); b.line([[x + 127, y + 40], [x + 135, y + 40]], { color: c, w: 5 });
  b.line([[x + 115, y + 82], [x + 115, y + 100]], { color: c });                   // neck
  b.rect(x + 60, y + 100, 110, 140, { color: c });                                // torso
  b.line([[x + 170, y + 120], [x + 235, y + 150], [x + 300, y + 136]], { color: c, w: 4.5 });  // reaching arm
  b.line([[x + 300, y + 136], [x + 316, y + 124]], { color: c }); b.line([[x + 300, y + 136], [x + 318, y + 146]], { color: c });
  b.line([[x + 60, y + 120], [x + 25, y + 190], [x + 40, y + 235]], { color: c, w: 4.5 });
  b.line([[x + 115, y + 240], [x + 115, y + 268]], { color: c });                 // waist column
  b.rect(x + 40, y + 268, 150, 50, { color: c });                                 // wheeled base
  b.circle(x + 72, y + 330, 18, { color: c }); b.circle(x + 158, y + 330, 18, { color: c });
}
function tray(b, x, y, w, color, label) {  // a tray seen from the front, with a tint and a label under it
  const h = 70;
  b.fill([[x, y], [x + 12, y + h], [x + w - 12, y + h], [x + w, y]], color, .22);
  b.line([[x, y], [x + 12, y + h], [x + w - 12, y + h], [x + w, y]], { color, w: 6 });
  if (label) b.text(x, y - 52, label, { size: 40, color, w: w, align: 'center' });
}
function part(b, x, y, s = 1) { b.fill([[x, y], [x + 46 * s, y], [x + 46 * s, y + 56 * s], [x, y + 56 * s]], BLUE, .35); b.rect(x, y, 46 * s, 56 * s, { color: BLUE, w: 5 }); b.rect(x + 10 * s, y + 12 * s, 26 * s, 32 * s, { color: GREY, w: 3 }); }
function person(b, x, y, o = {}) {  // x, y = feet centre
  b.circle(x, y - 200, 34); b.line([[x, y - 166], [x, y - 64]]);
  b.line([[x, y - 140], [x - 62, y - 92]]); b.line([[x, y - 140], [x + 58, y - 112]]);
  b.line([[x, y - 64], [x - 34, y]]); b.line([[x, y - 64], [x + 34, y]]);
  b.rect(x + 48, y - 152, 100, 70, { color: BLUE }); b.line([[x + 66, y - 132], [x + 128, y - 132]], { color: BLUE, w: 3 }); b.line([[x + 66, y - 112], [x + 116, y - 112]], { color: BLUE, w: 3 });
}
function card(b, x, y, w, h, str, o = {}) {
  if (o.tint) b.fill([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], o.tint, .12);
  b.rect(x, y, w, h, { color: o.color || INK, w: o.bw || 4 });
  return b.text(x + 24, y + (o.pad ?? 18), str, { size: o.size || 42, w: w - 48, color: o.tcolor || INK, lh: o.lh || 1.12, align: o.align });
}
function gauge(b, cx, cy, r, color) {  // a dial 0 to 1 with the 0.80 threshold marked; update(v) turns the needle
  b.arc(cx, cy, r, Math.PI, 2 * Math.PI, { w: 4 });
  const ta = Math.PI + .8 * Math.PI;
  b.line([[cx + (r - 22) * Math.cos(ta), cy + (r - 22) * Math.sin(ta)], [cx + (r + 22) * Math.cos(ta), cy + (r + 22) * Math.sin(ta)]], { color: GREEN, w: 6, amp: .5 });
  b.text(cx + (r + 26) * Math.cos(ta) - 10, cy + (r + 26) * Math.sin(ta) - 44, '0.80', { size: 36, color: GREEN });
  b.text(cx - r - 20, cy + 10, '0', { size: 32, color: GREY }); b.text(cx + r - 4, cy + 10, '1', { size: 32, color: GREY });
  const needle = document.createElementNS(NS, 'path'); needle.style.stroke = color; needle.style.strokeWidth = 7; needle.style.fill = 'none';
  b.svg.appendChild(needle);
  const hub = b.circle(cx, cy, 9, { color: INK });
  const label = b.text(cx - 80, cy + 18, '0.00', { size: 64, color, w: 160, align: 'center', font: 'mono' });
  const lw = label.querySelector('.w');
  return { needle, cx, cy, r, lw, color };
}
function setGauge(g, v, shown) {
  const a = Math.PI + v * Math.PI, L = g.r * .86 * shown;
  g.needle.setAttribute('d', `M${g.cx} ${g.cy} L${(g.cx + L * Math.cos(a)).toFixed(1)} ${(g.cy + L * Math.sin(a)).toFixed(1)}`);
  g.needle.style.stroke = v >= .8 ? GREEN : RED; g.lw.style.color = v >= .8 ? GREEN : RED;
  g.lw.textContent = v.toFixed(2);
}

// ---------------------------------------------------------------- the elements of every slide
// slide 1: the hook
el('frame', 980, 640, b => {
  b.img('assets/frame_ep1009_104_in_bin.png', 10, 10, 960, 600);
  b.rect(4, 4, 972, 612, { w: 5 });
});
el('wrongmark', 980, 640, b => {  // a red ring round the green tray of the frame above (same box, same transform)
  b.circle(612, 300, 215, { color: RED, w: 7, amp: 4 });
});
el('q1', 760, 300, b => { b.text(0, 0, 'Did the part land in the tray it was meant for?', { size: 66, w: 740, lh: 1.08 }); });
el('ans1', 760, 140, b => {
  b.text(0, 0, 'the model, reading this frame:', { size: 44, color: GREY });
  b.text(0, 54, 'yes · 0.99 sure', { size: 64, color: GREEN, font: 'mono' });
});
el('wrong1', 760, 200, b => {
  b.text(0, 0, 'Wrong.', { size: 96, color: RED });
  b.text(0, 112, 'It was meant for the orange tray.', { size: 44, color: RED, w: 740 });
});
el('src1', 1700, 50, b => { b.text(0, 0, 'Real frame from my test log: test04, episode 1009, decision 104 (VALIDATION.md: "by fault type")', { size: 28, color: GREY, font: 'mono', w: 1700 }); });

// slide 2: the title and the problem
el('title', 1100, 320, b => {
  b.text(0, 0, 'A robot that knows when to ask', { size: 118, w: 1080, lh: 1.0 });
  b.line([[0, 270], [760, 262]], { color: ORANGE, w: 7, amp: 3 });
});
el('sub2', 1000, 260, b => { b.text(0, 0, 'On a real line, sure and wrong ships a bad part. So this project is about trust: when should a robot act alone, and when should it ask?', { size: 46, w: 1000, lh: 1.18 }); });
el('tag2', 1000, 60, b => { b.text(0, 0, 'MH3 humanoid · Genesis simulator · fully offline on one GPU', { size: 34, color: GREY, font: 'mono', w: 1200 }); });

// slide 3: the cell
el('table', 1240, 200, b => {
  b.line([[0, 20], [1240, 20]], { w: 6 });
  b.line([[60, 20], [60, 190]], { w: 5 }); b.line([[1180, 20], [1180, 190]], { w: 5 });
});
el('robot', 340, 360, b => robot(b, 10, 0));
el('part', 70, 80, b => part(b, 8, 8));
el('trayG', 280, 100, b => tray(b, 10, 10, 260, GREEN, 'good'));
el('trayO', 280, 100, b => tray(b, 10, 10, 260, ORANGE, 'rework'));
el('h3', 1700, 100, b => { b.text(0, 0, 'The cell', { size: 92 }); });
el('lab3', 1200, 160, b => {
  b.text(0, 0, 'A Mirsee MH3 humanoid in the Genesis simulator. Parts come in one at a time. Good ones to green, defective ones to orange.', { size: 42, w: 1180, lh: 1.15 });
});

// slide 4: four questions a script cannot answer
const Q = [['qf', '1', 'Is the paint finish defective?'], ['qg', '2', 'Is it safe to grasp?'], ['qb', '3', 'Did it land in the bin?'], ['qd', '4', 'Is the sort done?']];
for (const [id, n, s] of Q) el(id, 410, 150, b => { card(b, 6, 6, 398, 138, s, { size: 46, tint: BLUE, color: INK, pad: 20 }); });
el('h4', 1700, 120, b => { b.text(0, 0, 'Four questions no script can answer', { size: 80, w: 1700 }); });

// slide 5: the split
el('treeBox', 900, 700, b => {
  b.fill([[0, 0], [900, 0], [900, 700], [0, 700]], INK, .04);
  b.rect(4, 4, 892, 692, { w: 5 });
  b.text(36, 26, 'Behaviour tree', { size: 66 });
  b.text(36, 104, 'same input, same action', { size: 34, color: GREY, font: 'mono', w: 600 });
  const rows = ['drive, reach, carry: straight-line arm paths', 'touch grip: the fingertips feel the part', 'hand IMU stops the arm on a bump', 'safety rules: no grasp if fallen or crowded', 'timeouts, and a safe answer for every question'];
  rows.forEach((r, i) => { b.line([[44, 196 + i * 96], [60, 214 + i * 96], [90, 176 + i * 96]], { color: GREEN, w: 5, amp: 1 }); b.text(110, 168 + i * 96, r, { size: 40, w: 760, lh: 1.05 }); });
});
el('aiBox', 560, 700, b => {
  b.fill([[0, 0], [560, 0], [560, 700], [0, 700]], BLUE, .06);
  b.rect(4, 4, 552, 692, { color: BLUE, w: 5 });
  b.text(36, 26, 'Judgement', { size: 66, color: BLUE });
  b.text(36, 104, 'a model, or a person', { size: 34, color: GREY, font: 'mono', w: 500 });
});
el('askArrow', 220, 120, b => { b.arrow(10, 30, 206, 30, { w: 5 }); b.text(20, 46, 'asks', { size: 40 }); });
el('ansArrow', 220, 150, b => { b.arrow(206, 30, 10, 30, { w: 5, color: BLUE }); b.text(20, 46, 'answer and how sure', { size: 38, color: BLUE, w: 190, lh: 1.0 }); });
el('h5', 1800, 120, b => { b.text(0, 0, 'The model answers. It never moves the arm.', { size: 80, w: 1800 }); });

// slide 6: the decision ladder
el('qArrow', 120, 60, b => { b.arrow(6, 30, 110, 30, { w: 5 }); });
el('d3', 470, 200, b => {
  b.fill([[0, 0], [470, 0], [470, 200], [0, 200]], ORANGE, .1);
  b.rect(4, 4, 462, 192, { color: ORANGE, w: 5 });
  b.text(30, 22, 'd3 27B vision', { size: 58, color: ORANGE });
  b.text(30, 104, 'reads the camera frame, on this PC', { size: 32, color: GREY, font: 'mono', w: 420 });
});
let G6 = null;
el('gauge', 460, 300, b => { G6 = gauge(b, 230, 210, 170, GREEN); return { update: (v, p) => setGauge(G6, v, p) }; });
el('act', 520, 160, b => {
  b.arrow(10, 80, 150, 80, { color: GREEN, w: 6 });
  b.text(176, 18, '0.80 or more:', { size: 40, color: GREEN });
  b.text(176, 70, 'act alone', { size: 62, color: GREEN });
});
el('toPerson', 620, 190, b => {
  b.arrow(40, 6, 40, 170, { color: BLUE, w: 6 });
  b.text(90, 30, 'less sure: a person on the operator page', { size: 42, color: BLUE, w: 520, lh: 1.08 });
});
el('person', 340, 250, b => person(b, 110, 230));
el('timeout', 760, 230, b => {
  b.circle(70, 80, 56, { w: 4 }); b.line([[70, 80], [70, 44]], { w: 4 }); b.line([[70, 80], [98, 92]], { w: 4 });
  b.text(150, 24, 'no answer in 120 s:', { size: 42 });
  b.text(150, 76, 'the safe answer', { size: 62, color: RED });
});
el('safeList', 760, 230, b => {
  const rows = ['finish: defective', 'grasp: do not', 'in the bin: no', 'sort done: no'];
  rows.forEach((r, i) => b.text(0, i * 54, r, { size: 38, font: 'mono', w: 760 }));
});
el('h6', 1700, 120, b => { b.text(0, 0, 'The decision ladder', { size: 92 }); });

// slide 7: sure and wrong
el('big29', 700, 240, b => { b.text(0, 0, '29 of 29', { size: 190, color: RED, wordLen: 220 }); });
el('lab7', 760, 260, b => { b.text(0, 0, 'times the model called a wrong-tray part "in the bin": 15 landings, each checked about twice, every answer at 0.9 or more', { size: 44, w: 590, lh: 1.12 }); });
el('h7', 1700, 120, b => { b.text(0, 0, 'Sure and wrong', { size: 92 }); });
el('punch7', 1700, 90, b => { b.text(0, 0, 'No confidence threshold catches a model that is sure and wrong.', { size: 50, w: 1700, color: INK }); });
el('src7', 1700, 50, b => { b.text(0, 0, 'test04: 300 episodes, 3,262 decisions (VALIDATION.md: "by fault type", the wrong-bin table)', { size: 28, color: GREY, font: 'mono', w: 1700 }); });

// slide 8: the scales
el('cellG', 300, 160, b => { b.fill([[40, 6], [260, 6], [260, 40], [40, 40]], INK, .12); b.rect(40, 6, 220, 34, { w: 4 }); b.text(0, 90, 'good +1 part', { size: 42, font: 'mono', color: GREEN, w: 300, align: 'center' }); });
el('cellO', 300, 160, b => { b.fill([[40, 6], [260, 6], [260, 40], [40, 40]], INK, .12); b.rect(40, 6, 220, 34, { w: 4 }); b.text(0, 90, 'rework +0', { size: 42, font: 'mono', color: ORANGE, w: 300, align: 'center' }); });
el('verdict8', 760, 200, b => {
  b.text(0, 0, 'meant for rework, so:', { size: 44 });
  b.text(0, 62, 'in the bin? no', { size: 76, color: RED, font: 'mono' });
});
el('h8', 1800, 120, b => { b.text(0, 0, "Don't ask a camera what a scale can measure", { size: 80, w: 1800 }); });
el('note8', 1700, 60, b => { b.text(0, 0, 'Load cells under each tray. Added after test04: not part of its numbers.', { size: 30, color: GREY, font: 'mono', w: 1700 }); });

// slide 9: when things fail
el('f1', 540, 460, b => {
  b.rect(6, 6, 528, 448, { w: 4 });
  b.text(36, 30, 'The model goes down', { size: 62, w: 470, lh: 1.05 });
  b.arrow(50, 220, 50, 300, { w: 5, color: BLUE });
  b.text(96, 216, 'every question goes to the person', { size: 50, color: BLUE, w: 410, lh: 1.08 });
});
el('f2', 540, 460, b => {
  b.rect(6, 6, 528, 448, { w: 4 });
  b.text(36, 30, 'Nobody answers the page', { size: 62, w: 470, lh: 1.05 });
  b.arrow(50, 220, 50, 300, { w: 5, color: RED });
  b.text(96, 216, 'after 120 s, the safe answer', { size: 50, color: RED, w: 410, lh: 1.08 });
});
el('f3', 540, 460, b => {
  b.fill([[6, 6], [534, 6], [534, 454], [6, 454]], GREEN, .1);
  b.rect(6, 6, 528, 448, { w: 4, color: GREEN });
  b.text(36, 30, 'It never', { size: 62, color: GREEN });
  b.text(36, 140, 'closes its hand without a yes', { size: 50, w: 470, lh: 1.08 });
  b.text(36, 290, 'ships a part nobody vouched for', { size: 50, w: 470, lh: 1.08 });
});
el('h9', 1700, 120, b => { b.text(0, 0, 'When something fails', { size: 92 }); });

// slide 10: on one PC
el('pc', 1060, 640, b => {
  b.rect(6, 6, 1048, 560, { w: 6 });
  b.line([[440, 566], [400, 630]], { w: 5 }); b.line([[620, 566], [660, 630]], { w: 5 }); b.line([[340, 630], [720, 630]], { w: 5 });
  b.text(40, 28, 'One PC · one RTX 3090 (24 GB)', { size: 54 });
  const chips = [['Genesis simulator + the MH3', INK], ['d3 27B vision, 4-bit · about 0.28 s an answer', ORANGE], ['operator page, served on the same PC', BLUE]];
  chips.forEach(([s, c], i) => { b.rect(40, 130 + i * 140, 976, 110, { color: c, w: 4 }); b.text(70, 156 + i * 140, s, { size: 44, color: c, w: 920 }); });
});
el('cloud', 520, 360, b => {
  const pts = []; for (let i = 0; i <= 40; i++) { const a = i / 40 * Math.PI * 2; const r = 120 + 26 * Math.sin(a * 5); pts.push([260 + r * 1.5 * Math.cos(a), 150 + r * .8 * Math.sin(a)]); }
  b.line(pts, { w: 4, color: GREY });
  b.line([[90, 40], [430, 260]], { color: RED, w: 8, amp: 3 }); b.line([[430, 40], [90, 260]], { color: RED, w: 8, amp: 3 });
  b.text(70, 300, 'no cloud calls', { size: 46, color: RED, w: 460, align: 'center' });
});
el('h10', 1800, 120, b => { b.text(0, 0, 'On the shop floor, not in a data centre', { size: 80, w: 1800 }); });
el('src10', 1700, 50, b => { b.text(0, 0, 'Answer time: 277 ms per decision on the 3090 (VALIDATION.md: "27B frame arms as 4-bit GGUFs")', { size: 28, color: GREY, font: 'mono', w: 1700 }); });

// slide 11: the test
el('h11', 1700, 120, b => { b.text(0, 0, 'Pre-registered. Scored once.', { size: 92 }); });
el('sub11', 1700, 120, b => { b.text(0, 0, 'Ladder and threshold frozen before the run. 300 new episodes, 3,262 decisions.', { size: 46, w: 1700 }); });
const NUMS = [['n1', '78.9%', 'decided alone and right', GREEN], ['n2', '18.5%', 'handed to a person', BLUE], ['n3', '2.6%', 'acted wrong (1.7 to 3.5)', RED]];
for (const [id, v, s, c] of NUMS) el(id, 540, 300, b => { b.text(0, 0, v, { size: 150, color: c, wordLen: 220 }); b.text(6, 190, s, { size: 44, w: 530, color: c }); });
el('foot11', 1760, 90, b => { b.text(0, 0, 'Judgement questions (2,262 of the 3,262). 95% intervals by episode. Simulated parts, not a real line. Load cells not included. VALIDATION.md: "Confirmation on test04".', { size: 28, color: GREY, font: 'mono', w: 1760 }); });


// slide 12 (v2): the bridge back to the question
el('h12', 1700, 300, b => { b.text(0, 0, 'When a model says it is sure, which kind can you trust?', { size: 92, w: 1600, lh: 1.05 }); });
el('vlm12', 760, 260, b => { card(b, 6, 6, 740, 240, 'A VLM writes an answer. Ask how sure, and it writes a number.', { size: 44, tint: RED, color: RED, pad: 26 }); });
el('vdm12', 760, 260, b => { card(b, 6, 6, 740, 240, 'A VDM scores every option. The scores are its confidence.', { size: 44, tint: GREEN, color: GREEN, pad: 26 }); });

// slide 13 (v2): Phase 2, the capsule cell
el('h13', 1700, 120, b => { b.text(0, 0, 'Phase 2: soft-gel capsule inspection', { size: 92, w: 1700 }); });
el('sub13', 1600, 120, b => { b.text(0, 0, 'A job I have done by hand. A leaker looks like every other capsule, except for the pool under it.', { size: 44, w: 1560, lh: 1.15 }); });
el('capTray', 1100, 460, b => {
  b.fill([[0, 60], [760, 60], [760, 440], [0, 440]], '#f3e9cf', .9); b.rect(0, 60, 760, 380, { w: 5 });
  for (let side = 0; side < 2; side++) for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    const x = 60 + side * 380 + c * 70, y = 110 + r * 82;
    if (side === 1 && r === 1 && c === 2) { b.fill([[x - 30, y - 18], [x + 52, y - 22], [x + 60, y + 54], [x - 26, y + 60]], '#b07d0a', .75); b.circle(x + 13, y + 18, 44, { color: '#8a6208', w: 3 }); }
    b.fill([[x, y], [x + 26, y], [x + 26, y + 36], [x, y + 36]], '#d4a017', .9); b.rect(x, y, 26, 36, { color: '#a87b0c', w: 3, amp: .6 });
  }
  b.text(520, 0, 'a leaker', { size: 40, color: RED }); b.arrow(600, 50, 620, 170, { color: RED, w: 4, head: 18 });
  for (const bx of [800, 960]) { b.fill([[bx, 220], [bx + 120, 220], [bx + 120, 340], [bx, 340]], RED, .25); b.rect(bx, 220, 120, 120, { color: RED, w: 5 }); }
  b.text(800, 360, 'leaker bins', { size: 38, color: RED });
});

// slide 14 (v2): same picture, two answers (capsule)
el('cap14', 460, 460, b => { b.img('assets/capsule_ep1004_438.png', 10, 10, 440, 440); b.rect(4, 4, 452, 452, { w: 5 }); });
el('q14', 1100, 140, b => { b.text(0, 0, 'Is the capsule in the middle of this picture a leaker?', { size: 58, w: 1100, lh: 1.05 }); });
el('vlm14', 1100, 190, b => {
  b.text(0, 0, 'VLM · Qwen3.8-27B wrote:', { size: 42, color: GREY });
  b.text(0, 56, 'good · 0.95 sure', { size: 72, color: RED, font: 'mono' });
});
el('vdm14', 1100, 190, b => {
  b.text(0, 0, 'VDM · d3 27B, built on Qwen, scored:', { size: 42, color: GREY });
  b.text(0, 56, 'leaker 0.98 · good 0.02', { size: 72, color: GREEN, font: 'mono' });
});
el('truth14', 1100, 80, b => { b.text(0, 0, 'The truth, from the simulator: leaker', { size: 52 }); });
el('src14', 1700, 50, b => { b.text(0, 0, 'One picked picture from my selection log sel01 (episode 1004, picture 438), chosen to show the difference.', { size: 28, color: GREY, font: 'mono', w: 1700 }); });

// slide 15 (v2): the answer, test05
el('h15', 1700, 120, b => { b.text(0, 0, 'Leakers missed while 90% sure or more', { size: 88, w: 1700 }); });
el('sub15', 1700, 80, b => { b.text(0, 0, '467 leaker pictures from 150 new shifts the models had never seen', { size: 44, w: 1700 }); });
const ANS = [['a1', 'd3 27B', 'VDM', '1', GREEN, 1], ['a2', 'Intern-Decision-4B', 'VDM', '1', GREEN, 1], ['a3', 'Qwen3.8-27B, first word', 'VLM', '98', RED, 98], ['a4', 'Qwen3.8-27B, stated', 'VLM', '171', RED, 171]];
for (const [id, name, kind, v, c, n] of ANS) el(id, 1600, 100, b => {
  b.text(0, 18, name, { size: 44, w: 560 }); b.text(560, 22, kind, { size: 36, color: GREY, font: 'mono' });
  const w = Math.max(14, n / 171 * 760); b.fill([[700, 14], [700 + w, 14], [700 + w, 84], [700, 84]], c, .5); b.rect(700, 14, w, 70, { color: c, w: 4, amp: 1 });
  b.text(720 + w, 14, v, { size: 64, color: c, font: 'mono' });
});
el('ladder15', 1700, 80, b => { b.text(0, 0, 'The frozen ladder, Intern-4B then d3 then a person: acted wrong on 0.2% of 3,785 decisions (predicted 0.5%).', { size: 40, w: 1700, color: BLUE }); });
el('foot15', 1760, 60, b => { b.text(0, 0, 'Picked on one log (sel03), frozen, scored once on another (test05). VALIDATION.md: "test05: the capsule confirmation".', { size: 28, color: GREY, font: 'mono', w: 1760 }); });

// slide 12: the close. A grid of every slide's still; it enters zoomed on slide 11's cell, so the cut is invisible.
const GRID = { cols: 4, cw: 452, ch: 254, gap: 16 };
const GRID_SLIDES = [3, 4, 5, 6, 11, 12, 13, 14, 15, 9, 10];  // v2 film order; the last is the slide the close pulls back from
const gridW = GRID.cols * GRID.cw + (GRID.cols - 1) * GRID.gap, gridH = 3 * GRID.ch + 2 * GRID.gap;
el('grid', gridW, gridH, b => {
  const imgs = [];
  for (let i = 0; i < GRID_SLIDES.length; i++) {
    const c = i % GRID.cols, r = Math.floor(i / GRID.cols), x = c * (GRID.cw + GRID.gap), y = r * (GRID.ch + GRID.gap);
    const im = document.createElement('img'); im.src = `stills/slide_${String(GRID_SLIDES[i]).padStart(2, '0')}.png`; im.className = 'im';
    Object.assign(im.style, { left: x + 'px', top: y + 'px', width: GRID.cw + 'px', height: GRID.ch + 'px', outline: '2px solid #cfcac0' });
    b.root.appendChild(im); imgs.push(im);
  }
  // custom entry: every cell but slide 11's fades in while the camera pulls back from that cell
  const L = GRID_SLIDES.length - 1;
  return { enter: (p) => imgs.forEach((im, i) => { im.style.opacity = i === L ? 1 : E.out(clamp((p - .15 - (L - i) * .02) / .5)); }) };
});
el('close', 1500, 420, b => {  // no panel (dup, Nov 11: see the slides behind); a halo in the page colour keeps the lines legible
  b.root.style.filter = 'drop-shadow(0 0 3px #fbfaf5) drop-shadow(0 0 6px #fbfaf5) drop-shadow(0 0 12px #fbfaf5)';
  b.text(60, 40, 'Logic for what must never fail.', { size: 76, w: 1400 });
  b.text(60, 150, 'A model for what can’t be scripted.', { size: 76, w: 1400, color: ORANGE });
  b.text(60, 260, 'A person when it isn’t sure.', { size: 76, w: 1400, color: BLUE });
});

// ---------------------------------------------------------------- slides (keyframes): id -> [x, y, scale, opacity, v]
// An element in slide n and n+1 moves; only in n+1 draws in (at: delay from the start of the move into it); only in
// n fades out. `v` drives an element's own state (the gauge's needle).
const st = (x, y, s = 1, o = 1, extra = {}) => ({ x, y, s, o, ...extra });
// the cell drawn once and reused small: positions of slide 3, and a helper that scales them about a point
const CELL = { table: st(520, 760), robot: st(280, 520, 1.25), part: st(860, 714), trayG: st(1100, 704), trayO: st(1440, 704) };
const cellAt = (k, ox, oy) => Object.fromEntries(Object.entries(CELL).map(([id, s]) => [id, st(ox + (s.x - CELL.robot.x) * k, oy + (s.y - CELL.robot.y) * k, s.s * k)]));
const SLIDES = [
  null,
  { // 1 the hook
    frame: st(90, 210), wrongmark: st(90, 210, 1, 1, { at: 4.2, dur: .6 }),
    q1: st(1110, 220, 1, 1, { at: .3 }), ans1: st(1110, 470, 1, 1, { at: 1.9 }), wrong1: st(1110, 660, 1, 1, { at: 4.6 }),
    src1: st(90, 900, 1, 1, { at: 1.0 }),
  },
  { // 2 the title
    frame: st(1300, 560, .55), wrongmark: st(1300, 560, .55),
    title: st(110, 210, 1, 1, { at: .6 }), sub2: st(110, 560, 1, 1, { at: 1.6 }), tag2: st(110, 900, 1, 1, { at: 3.0 }),
  },
  { // 3 the cell
    ...CELL, h3: st(110, 70, 1, 1, { at: .2 }), lab3: st(110, 190, 1, 1, { at: 1.6 }),
  },
  { // 4 four questions
    ...cellAt(.72, 360, 640), h4: st(110, 70, 1, 1, { at: .4 }),
    qf: st(110, 250, 1, 1, { at: .9 }), qg: st(550, 250, 1, 1, { at: 2.6 }), qb: st(990, 250, 1, 1, { at: 4.0 }), qd: st(1430, 250, 1, 1, { at: 5.4 }),
  },
  { // 5 the split: the cell's robot climbs into the tree box, the four cards move into the judgement box
    h5: st(110, 70, 1, 1, { at: .9 }),
    // the cards fly to the judgement box first; the tree box draws only once they are past it (no text over text)
    aiBox: st(1250, 250, 1, 1, { at: .05 }), treeBox: st(110, 250, 1, 1, { at: 1.45 }),
    qf: st(1355, 400, .85, 1, { at: 0 }), qg: st(1355, 540, .85, 1, { at: .08 }), qb: st(1355, 680, .85, 1, { at: .16 }), qd: st(1355, 820, .85, 1, { at: .24 }),
    robot: st(845, 275, .4, 1, { at: .3 }),
    askArrow: st(1020, 470, 1, 1, { at: 2.6 }), ansArrow: st(1020, 640, 1, 1, { at: 3.4 }),
  },
  { // 6 the ladder: the cards stack as "a question", the judgement box becomes the ladder
    h6: st(110, 60, 1, 1, { at: .9 }),
    qf: st(110, 320, .9), qArrow: st(450, 355, 1, 1, { at: .4 }),
    d3: st(580, 290, 1, 1, { at: .5 }), gauge: st(585, 530, 1, 1, { at: 1.2, v: .58 }),
    act: st(1090, 300, 1, 1, { at: 2.6 }), toPerson: st(1090, 470, 1, 1, { at: 3.6 }), person: st(1060, 650, .9, 1, { at: 4.6 }),
    timeout: st(1320, 660, 1, 1, { at: 6.4 }), safeList: st(1470, 820, .9, 1, { at: 8.0 }),
  },
  { // 7 sure and wrong: the needle swings to 0.99 and the frame comes back
    h7: st(110, 60, 1, 1, { at: .6 }),
    d3: st(110, 220, .8), gauge: st(110, 400, 1.05, 1, { v: .99 }),
    frame: st(1290, 220, .55), wrongmark: st(1290, 220, .55),
    big29: st(640, 220, 1, 1, { at: 1.2 }), lab7: st(640, 450, 1, 1, { at: 2.2 }),
    punch7: st(110, 790, 1, 1, { at: 4.4 }), src7: st(110, 920, 1, 1, { at: 2.6 }),
  },
  { // 8 the scales: the trays come back with a load cell under each, the part in the wrong one
    h8: st(110, 60, 1, 1, { at: .6 }),
    table: st(160, 560, .78), trayG: st(300, 440, 1.15), trayO: st(720, 440, 1.15), part: st(395, 468, 1.0),
    cellG: st(306, 530, 1, 1, { at: 1.4 }), cellO: st(726, 530, 1, 1, { at: 1.8 }),
    verdict8: st(1150, 440, 1, 1, { at: 3.0 }), note8: st(110, 920, 1, 1, { at: 3.6 }),
  },
  { // 9 when things fail
    h9: st(110, 60, 1, 1, { at: .5 }),
    f1: st(110, 320, 1, 1, { at: .9 }), f2: st(690, 320, 1, 1, { at: 2.2 }), f3: st(1270, 320, 1, 1, { at: 3.4 }),
  },
  { // 10 on one PC
    h10: st(110, 60, 1, 1, { at: .5 }), pc: st(110, 220, 1, 1, { at: .9 }), cloud: st(1260, 330, 1, 1, { at: 3.4 }),
    src10: st(110, 940, 1, 1, { at: 2.4 }),
  },
  { // 11 the test
    h11: st(110, 60, 1, 1, { at: .4 }), sub11: st(110, 190, 1, 1, { at: 1.2 }),
    n1: st(110, 380, 1, 1, { at: 3.0 }), n2: st(700, 380, 1, 1, { at: 5.0 }), n3: st(1290, 380, 1, 1, { at: 7.0 }),
    foot11: st(110, 900, 1, 1, { at: 9.0 }),
  },
  { // 12 (v2) the bridge
    h12: st(110, 120, 1, 1, { at: .3 }), vlm12: st(110, 520, 1, 1, { at: 1.8 }), vdm12: st(1000, 520, 1, 1, { at: 3.0 }),
  },
  { // 13 (v2) Phase 2: the capsule cell
    h13: st(110, 60, 1, 1, { at: .3 }), sub13: st(110, 190, 1, 1, { at: 1.2 }), capTray: st(400, 420, 1, 1, { at: 2.0 }),
  },
  { // 14 (v2) same picture, two answers
    cap14: st(110, 250, 1, 1, { at: .3 }), q14: st(660, 120, 1, 1, { at: .6 }),
    vlm14: st(660, 330, 1, 1, { at: 2.4 }), vdm14: st(660, 560, 1, 1, { at: 4.4 }), truth14: st(660, 800, 1, 1, { at: 7.0 }),
    src14: st(110, 960, 1, 1, { at: 1.4 }),
  },
  { // 15 (v2) the answer
    h15: st(110, 60, 1, 1, { at: .3 }), sub15: st(110, 180, 1, 1, { at: 1.0 }),
    a1: st(110, 300, 1, 1, { at: 2.0 }), a2: st(110, 420, 1, 1, { at: 3.0 }), a3: st(110, 540, 1, 1, { at: 4.4 }), a4: st(110, 660, 1, 1, { at: 5.8 }),
    ladder15: st(110, 820, 1, 1, { at: 8.0 }), foot15: st(110, 950, 1, 1, { at: 9.4 }),
  },
  { // 16 the close: the grid, then the three lines over it
    grid: st((W - gridW) / 2, (H - gridH) / 2, 1, .32), close: st(210, 330, 1, 1, { at: 2.4 }),
  },
];
const CLOSE = 16;
const N = SLIDES.length - 1;

// ---------------------------------------------------------------- drawing one instant
// a -> b: the move from slide a to slide b; dt seconds since the move started; D its length. dt >= D is b's hold.
const ENTER_GAP = .18;
function frame(a, b, dt, D) {
  const A = a ? SLIDES[a] : {}, B = SLIDES[b];
  const u = E.io(clamp(dt / D));
  let order = 0;
  for (const id in ELS) {
    const e = ELS[id], sa = A[id], sb = B[id], r = e.root;
    if (!sa && !sb) { r.style.display = 'none'; continue; }
    r.style.display = 'block';
    let x, y, s, o, v;
    if (sa && sb) {  // shared: move
      const k = E.io(clamp((dt - (sb.at ?? .06 * order++)) / (D * .92)));
      x = lerp(sa.x, sb.x, k); y = lerp(sa.y, sb.y, k); s = Math.exp(lerp(Math.log(sa.s), Math.log(sb.s), k)); o = lerp(sa.o, sb.o, k);
      v = sb.v !== undefined ? lerp(sa.v ?? sb.v, sb.v, k) : undefined;
      reveal(e, 1);
    } else if (sb) {  // entering: draw in
      x = sb.x; y = sb.y; s = sb.s; o = sb.o; v = sb.v;
      const at = sb.at ?? D * .35, dur = sb.dur ?? clamp(e.total / 2400, .5, 1.6);
      const p = clamp((dt - at) / dur);
      reveal(e, p);
      if (e.enter) e.enter(p);
    } else if (b === CLOSE) {  // into the close: slide 11's own still takes over at once, so its elements just go
      x = sa.x; y = sa.y; s = sa.s; o = dt > 1e-3 ? 0 : sa.o; v = sa.v; reveal(e, 1);
    } else {  // leaving: fade out early in the move
      x = sa.x; y = sa.y; s = sa.s * (1 - .015 * u); o = sa.o * (1 - E.out(clamp(dt / (D * .45))));
      v = sa.v; reveal(e, 1);
    }
    if (e.update && v !== undefined) e.update(v, sb && !sa ? clamp((dt - (sb.at ?? 0)) / .6) : 1);
    r.style.opacity = o;
    r.style.transform = `translate(${x.toFixed(2)}px,${y.toFixed(2)}px) scale(${s.toFixed(4)})`;
    r.style.zIndex = id === 'close' ? 20 : id === 'grid' ? 1 : (sa && sb) ? 8 : 5;
  }
}
// the close's grid enters as a camera pull-back from slide 11's cell (the cell is slide 11's still, so the cut is invisible)
function gridCamera(dt, D) {
  const g = ELS.grid, sb = SLIDES[CLOSE].grid, i = GRID_SLIDES.length - 1, c = i % GRID.cols, r = Math.floor(i / GRID.cols);
  const cx = c * (GRID.cw + GRID.gap), cy = r * (GRID.ch + GRID.gap), z0 = W / GRID.cw;
  const k = E.io(clamp(dt / D)), z = Math.exp(lerp(Math.log(z0), Math.log(1), k));
  const x = lerp(-cx * z0, sb.x, k), y = lerp(-cy * z0, sb.y, k), o = lerp(1, sb.o, E.io(clamp((dt - D * .7) / (D * .8))));
  g.root.style.transform = `translate(${x.toFixed(2)}px,${y.toFixed(2)}px) scale(${z.toFixed(4)})`; g.root.style.opacity = o;
  g.enter(k * 1.6);
}

// ---------------------------------------------------------------- timelines
const MOVE = 2.2;
// The film: [kind, slide or label, seconds]. 'S' = slide n's hold, 'M' = the move into slide n, 'F' = sim footage.
// v2 (Nov 9): the film after the intro. [kind, ...]: 'S' n secs = slide n's hold; 'M' from to secs = the move from
// slide `from` into slide `to` (0 = draw in from nothing); 'F' label secs = sim footage, spliced in after the render.
const FILM = [  // holds sized to dup's voice-over (Nov 10, media/film/voiceover.py places each line)
  ['M', 0, 3, MOVE], ['S', 3, 6.0], ['F', 'A', 16.0],  // A: one whole sort cycle, reach to bin (dup, Nov 11)
  ['M', 3, 4, MOVE], ['S', 4, 11.5], ['M', 4, 5, MOVE], ['S', 5, 7.0],
  ['M', 5, 6, MOVE], ['S', 6, 16.0], ['F', 'B', 9.0],
  ['M', 6, 11, MOVE], ['S', 11, 21.5],
  ['M', 11, 12, MOVE], ['S', 12, 6.5],
  ['M', 12, 13, MOVE], ['S', 13, 15.5], ['F', 'C', 8.5],
  ['M', 13, 14, MOVE], ['S', 14, 16.5],
  ['M', 14, 15, MOVE], ['S', 15, 18.5],
  ['M', 15, 9, MOVE], ['S', 9, 5.5], ['F', 'D', 13.0],
  ['M', 9, 10, MOVE], ['S', 10, 8.0],
  ['M', 10, 16, 3.0], ['S', 16, 8.0],
];
const mode = window.FILM_MODE || 'film';
const segs = [];
if (mode === 'film') { let t = 0; for (const f of FILM) { const [k] = f; const seg = k === 'M' ? { k, a: f[1], n: f[2], d: f[3] } : { k, n: f[1], d: f[2] }; seg.t0 = t; segs.push(seg); t += seg.d; } window.T = t; }
else if (mode === 'stills') { window.T = N; }
else if (mode === 'shot') {  // FILM_SHOT = [a, b, holdBefore, move, holdAfter]
  const [a, b, h0, m, h1] = window.FILM_SHOT; segs.push({ k: 'S', n: a, d: h0, t0: 0 }, { k: 'M', a, n: b, d: m, t0: h0 }, { k: 'S', n: b, d: h1, t0: h0 + m }); window.T = h0 + m + h1;
}
window.FPS = 60;
window.CUTS = segs.filter((s, i) => s.k === 'F' || (i > 0 && segs[i - 1].k === 'F')).map(s => s.t0);
window.FILM_SEGS = segs;

const foot = document.getElementById('footage');
function camera(z) { world.style.transform = `translate(${W / 2}px,${H / 2}px) scale(${z.toFixed(5)}) translate(${-W / 2}px,${-H / 2}px)`; }
const DRIFT = .012;
window.seek = async (t) => {
  foot.style.display = 'none';
  if (mode === 'stills') { const n = clamp(Math.floor(t) + 1, 1, N); frame(0, n, 999, MOVE); if (n === CLOSE) gridCamera(999, 3); camera(1); return; }
  let i = segs.findIndex(s => t < s.t0 + s.d); if (i < 0) i = segs.length - 1;
  const s = segs[i], lt = t - s.t0;
  if (s.k === 'F') { foot.style.display = 'flex'; foot.querySelector('.lab').textContent = 'FOOTAGE ' + s.n; foot.querySelector('.dur').textContent = `${s.d.toFixed(1)} s · spliced in after the render`; return; }
  if (s.k === 'M') {  // the move into slide n: the camera eases back from the drifted hold
    const D = s.d; frame(s.a, s.n, lt, D); if (s.n === CLOSE) gridCamera(lt, D);
    const prevHeld = i > 0 && segs[i - 1].k === 'S'; camera(1 + (prevHeld ? DRIFT : 0) * (1 - E.io(clamp(lt / D))));
    return;
  }
  // a hold: how long since the move into this slide began (builds keep drawing), and a slow push in
  const mv = segs.slice(0, i).reverse().find(x => x.k === 'M' && x.n === s.n);
  const D = mv ? mv.d : MOVE, since = mv ? t - mv.t0 : lt + (s.n === 1 ? 0 : 999);
  frame(mv ? mv.a : 0, s.n, since, D); if (s.n === CLOSE) gridCamera(since, D);
  camera(1 + DRIFT * E.io(clamp(lt / s.d)));
};
Promise.all([document.fonts.load('44px Patrick'), document.fonts.load('40px "Ubuntu Mono"'), ...[...document.images].map(i => i.decode().catch(() => {}))])
  .then(() => { window.ready = true; });
})();
