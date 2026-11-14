// Preview HUD for a seek(t) film page. Include after the film's own script.
// Space = play/pause, Left/Right = -/+0.5 s (Shift = one frame), R = replay, ?t=12.5 deep link.
// The HUD is the only place a clock is allowed; the renderer sets window.__render and never sees it.
(() => {
  if (window.__render) return;
  const FPS = window.FPS || 60, T = () => window.T || 10;
  let t = parseFloat(new URLSearchParams(location.search).get('t') || '0') || 0, playing = false, last = 0;
  const hud = document.createElement('div');
  hud.style.cssText = 'position:fixed;left:12px;bottom:12px;z-index:99999;font:14px/1.4 ui-monospace,monospace;color:#fff;background:rgba(0,0,0,.72);padding:8px 12px;border-radius:8px;user-select:none;cursor:pointer';
  document.body.appendChild(hud);
  const bar = document.createElement('div');
  bar.style.cssText = 'position:fixed;left:0;right:0;bottom:0;height:5px;z-index:99998;background:rgba(0,0,0,.25);cursor:pointer';
  const fill = document.createElement('div'); fill.style.cssText = 'height:100%;background:#ff6a2a;width:0'; bar.appendChild(fill);
  document.body.appendChild(bar);
  const show = () => { hud.textContent = `${playing ? 'playing' : 'paused'}  t=${t.toFixed(3)}s  f=${Math.round(t * FPS)}  (${T()}s)`; fill.style.width = (100 * t / T()) + '%'; };
  const go = async (nt) => { t = Math.min(Math.max(0, nt), T() - 1e-3); await window.seek(t); show(); };
  const tick = (now) => { if (!playing) return; const dt = (now - last) / 1000; last = now; if (t + dt >= T()) { playing = false; go(T() - 1e-3); return; } go(t + dt); requestAnimationFrame(tick); };
  const play = () => { if (playing) { playing = false; show(); return; } playing = true; last = performance.now(); requestAnimationFrame(tick); };
  addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); play(); }
    else if (e.code === 'ArrowRight') go(t + (e.shiftKey ? 1 / FPS : 0.5));
    else if (e.code === 'ArrowLeft') go(t - (e.shiftKey ? 1 / FPS : 0.5));
    else if (e.key === 'r' || e.key === 'R') { playing = false; go(0); play(); }
  });
  bar.addEventListener('click', (e) => { playing = false; go(T() * e.clientX / innerWidth); });
  hud.addEventListener('click', play);
  const start = () => { go(t); };
  if (window.ready) start(); else { const i = setInterval(() => { if (window.ready) { clearInterval(i); start(); } }, 50); }
})();
