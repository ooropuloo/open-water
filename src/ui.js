// Minimal interface: weather presets, sound, and a tuning sheet. Auto-hides when idle.
import { PRESETS } from './weather.js';
import { AutoDirector } from './auto.js';

const SLIDERS = [
  { key: 'wind', label: 'Wind', min: 0.5, max: 26, step: 0.1, fmt: (v) => `${v.toFixed(1)} m/s · Bf ${beaufort(v)}` },
  { key: 'waveScale', label: 'Waves', min: 0.2, max: 1.8, step: 0.01, fmt: (v) => `${Math.round(v * 100)}%` },
  { key: 'choppy', label: 'Chop', min: 0, max: 1.4, step: 0.01, fmt: (v) => v.toFixed(2) },
  { key: 'rain', label: 'Rain', min: 0, max: 1, step: 0.01, fmt: (v) => `${Math.round(v * 50)} mm/h` },
  { key: 'clouds', label: 'Clouds', min: 0, max: 1, step: 0.01, fmt: (v) => `${Math.round(v * 100)}%` },
  { key: 'lightning', label: 'Lightning', min: 0, max: 1, step: 0.01, fmt: (v) => (v < 0.01 ? 'off' : `${Math.round(v * 100)}%`) },
  { key: 'sunElev', label: 'Sun height', min: -18, max: 75, step: 0.1, fmt: (v) => (v < -7 ? `${v.toFixed(1)}° · moonlight` : `${v.toFixed(1)}°`) },
  { key: 'sunAz', label: 'Sun bearing', min: 0, max: 360, step: 1, fmt: (v) => `${Math.round(v)}°` },
];

// Beaufort force from 10 m wind speed (WMO upper limits, m/s)
const beaufort = (v) => {
  const i = [0.5, 1.5, 3.3, 5.5, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6].findIndex((x) => v < x);
  return i < 0 ? 12 : i;
};

const ICON = {
  sound: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M4 9.5v5h3.5L12 18V6L7.5 9.5H4z"/><path class="wave" d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"/><path class="mute" d="M16 9.5l5 5M21 9.5l-5 5"/></svg>',
  full: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
  tune: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M4 7h10M18 7h2M4 17h2M10 17h10"/><circle cx="16" cy="7" r="2"/><circle cx="8" cy="17" r="2"/></svg>',
  boat: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 14h18l-2.5 4h-13L3 14z"/><path d="M8 14l-4-6M16 14l4-6"/><circle cx="12" cy="9" r="1.6"/><path d="M12 10.6V14"/></svg>',
  ship: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 15h20l-3 4H4.5L2 15z"/><path d="M7 15v-3h8v3M10 12V9h3v3M12 9V4M10.5 6h3M16.5 15v-1.5h2V15"/></svg>',
  // whale
  whale: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 13.5c2.5 3 7 4.2 11 3.2 3.2-.8 5.4-3 6.2-5.6l1.8-2.6-2.9.9c-.6-1.6-2-2.4-3.3-2.1M2.5 13.5c1.5-.6 3-.6 4.5.2M9.5 17l-.8 2.5"/><circle cx="16.3" cy="10.2" r=".6" fill="currentColor"/></svg>',
  breach: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 19c2-1 4-1 6 0s4 1 6 0 4-1 6 0"/><path d="M8 15.5c.5-4.5 3-8.5 7-10.5-.6 3 .2 5.5 2 7.2"/><path d="M13 8.5l1.2 1.8"/></svg>', // whale
  auto: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7h3.5c2 0 3.2 1 4.3 2.6l2.4 4.8c1.1 1.6 2.3 2.6 4.3 2.6H21"/><path d="M3 17h3.5c1.4 0 2.4-.5 3.2-1.3M14.3 8.3c.8-.8 1.8-1.3 3.2-1.3H21"/><path d="M18.5 4.5L21 7l-2.5 2.5M18.5 14.5L21 17l-2.5 2.5"/></svg>',
  // kaiju: a row of dorsal plates over a wave
  kaiju: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17c2 0 2.5-1.2 4.5-1.2S10 17 12 17s2.5-1.2 4.5-1.2S19 17 21 17"/><path d="M5.5 14.5L7 10l1.5 2.2L10.5 6l1.6 4.6L14 8.5l1 3.4 1.6-1.6.9 4.2"/></svg>',
  bolt: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="M13 3L5 13.5h6L10 21l8-10.5h-6L13 3z"/></svg>',
};

// tap -> onTap; press and hold (or Shift+click) -> onHold
function press(btn, onTap, onHold) {
  let timer = 0, held = false;
  const cancel = () => { clearTimeout(timer); timer = 0; };
  btn.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    held = false; cancel();
    timer = setTimeout(() => { held = true; timer = 0; onHold(); }, 550);
  });
  ['pointerup', 'pointerleave', 'pointercancel'].forEach((t) => btn.addEventListener(t, cancel));
  btn.addEventListener('click', (e) => {
    if (held) { held = false; return; }
    e.shiftKey ? onHold() : onTap();
  });
  btn.addEventListener('contextmenu', (e) => e.preventDefault());
}

export class UI {
  constructor(app, audio) {
    this.app = app;
    this.audio = audio;
    this.root = document.createElement('div');
    this.root.className = 'ui';
    this.root.innerHTML = `
      <div class="mark">Open Water</div>
      <div class="cue" aria-live="polite"></div>
      <a class="src" href="https://github.com/Leonxlnx/open-water" target="_blank" rel="noopener noreferrer">Based on Open Water by Leonxlnx ↗</a>
      <div class="credit" role="dialog" aria-label="Credits">
        <div class="credit-k">Based on</div>
        <div class="credit-t">Open Water <span>by Leonxlnx</span></div>
        <a class="credit-l" href="https://github.com/Leonxlnx/open-water" target="_blank" rel="noopener noreferrer">github.com/Leonxlnx/open-water ↗</a>
        <div class="credit-n">MIT License · extended with a rowboat, a warship, a humpback whale, a kaiju and auto mode</div>
      </div>
      <div class="hint">Drag to look · Scroll to rise or sink</div>
      <div class="dock">
        <nav class="presets pill" role="tablist">
          ${Object.keys(PRESETS).map((k) => `<button role="tab" data-preset="${k}">${k[0].toUpperCase() + k.slice(1)}</button>`).join('')}
        </nav>
        <div class="tools pill">
          <button class="icon boat" title="Rowboat (B)" aria-label="Rowboat" aria-pressed="true">${ICON.boat}</button>
          <button class="icon ship" title="Warship (S)" aria-label="Warship" aria-pressed="true">${ICON.ship}</button>
          <button class="icon whale" title="Whale: surface now (K) · hold or Shift+K: on/off" aria-label="Whale" aria-pressed="true">${ICON.whale}</button>
          <button class="icon breach" title="Whale: breach now (J)" aria-label="Whale breach">${ICON.breach}</button>
          <button class="icon kaiju" title="Kaiju: summon now (G) · hold or Shift+G: on/off" aria-label="Kaiju" aria-pressed="true">${ICON.kaiju}</button>
          <button class="icon auto" title="Auto: weather, whale and kaiju on their own (A)" aria-label="Auto" aria-pressed="false">${ICON.auto}</button>
          <button class="icon strike" title="Lightning strike (L)" aria-label="Lightning strike">${ICON.bolt}</button>
          <button class="icon sound muted" title="Sound (M)" aria-label="Sound">${ICON.sound}</button>
          <button class="icon full" title="Full screen (F) · where a page cannot go full screen (iPhone), hides the controls; tap the sea to bring them back" aria-label="Full screen" aria-pressed="false">${ICON.full}</button>
          <button class="icon tune" title="Tune (T)" aria-label="Tune">${ICON.tune}</button>
        </div>
      </div>
      <section class="sheet" aria-label="Tune">
        ${SLIDERS.map((s) => `
          <label class="row">
            <span class="name">${s.label}</span><output data-out="${s.key}"></output>
            <input type="range" data-key="${s.key}" min="${s.min}" max="${s.max}" step="${s.step}">
          </label>`).join('')}
      </section>`;
    document.body.appendChild(this.root);
    this.auto = new AutoDirector(this);
    this.auto.onEvent = (t) => this.cue('Auto · ' + t);
    this.bind();
    this.sync();
    {
      const sp = new URLSearchParams(location.search), q = sp.get('auto');
      let saved = null;
      try { saved = localStorage.getItem('ow-auto'); } catch {}
      const on = q !== null ? q !== '0' : saved === '1';
      if (on && !sp.has('render')) this.toggleAuto(true); // never in the offline film renderer
    }
    this.idle();
    {
      const card = this.root.querySelector('.credit');
      const close = () => card.classList.add('gone');
      card.addEventListener('click', (e) => { if (!e.target.closest('a')) close(); });
      setTimeout(close, 8000);
    }
    // live significant wave height next to the wave slider
    setInterval(() => {
      const hs = this.app.stats && this.app.stats.hs;
      const o = this.root.querySelector('[data-out="waveScale"]');
      if (hs && o) o.textContent = `Hs ${hs.toFixed(1)} m`;
    }, 500);
  }

  bind() {
    const W = this.app.weather;
    this.root.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => {
      W.setPreset(b.dataset.preset);
      this.sync();
    }));
    this.root.querySelectorAll('input[type=range]').forEach((inp) => inp.addEventListener('input', () => {
      W.set(inp.dataset.key, Number(inp.value));
      this.root.querySelectorAll('[data-preset]').forEach((b) => b.classList.remove('on'));
      this.out(inp.dataset.key, Number(inp.value));
    }));
    const tune = this.root.querySelector('.tune');
    tune.addEventListener('click', () => this.root.classList.toggle('open'));
    this.root.querySelector('.sound').addEventListener('click', () => this.toggleSound());
    this.root.querySelector('.boat').addEventListener('click', () => this.toggleBoat());
    this.root.querySelector('.ship').addEventListener('click', () => this.toggleShip());
    press(this.root.querySelector('.whale'), () => this.surfaceWhale(), () => this.toggleWhale()); // whale: tap = surface now, hold = on/off
    this.root.querySelector('.breach').addEventListener('click', () => this.app.whale.breachNow()); // whale
    press(this.root.querySelector('.kaiju'), () => this.summonKaiju(), () => this.toggleKaiju()); // kaiju: tap = summon now, hold = on/off
    this.root.querySelector('.auto').addEventListener('click', () => this.toggleAuto());
    this.root.querySelector('.full').addEventListener('click', () => this.toggleFull());
    ['fullscreenchange', 'webkitfullscreenchange'].forEach((t) => document.addEventListener(t, () => this.syncFull()));
    // controls hidden by the button (no keyboard on a phone): a tap on the sea, not a drag, brings them back
    let down = null;
    window.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; }, { passive: true });
    window.addEventListener('pointerup', (e) => {
      if (!down || !this.root.classList.contains('hidden') || !this.hiddenByButton) return;
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) < 10 && performance.now() - down.t < 400) this.setHidden(false);
    }, { passive: true });
    this.root.querySelector('.strike').addEventListener('click', () => this.app.lightning.trigger(this.app.cam, { base: W.state.cloudBase }));
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      const k = e.key.toLowerCase();
      if (k === 'h') this.root.classList.toggle('hidden');
      if (k === 't') this.root.classList.toggle('open');
      if (k === 'm') this.toggleSound();
      if (k === 'b') this.toggleBoat();
      if (k === 's') this.toggleShip();
      if (k === 'k') e.shiftKey ? this.toggleWhale() : this.surfaceWhale(); // whale
      if (k === 'j') this.app.whale.breachNow(); // whale
      if (k === 'g') e.shiftKey ? this.toggleKaiju() : this.summonKaiju(); // kaiju
      if (k === 'a') this.toggleAuto();
      if (k === 'l') this.app.lightning.trigger(this.app.cam, { base: W.state.cloudBase });
      if (k === 'n') {
        // toggle moonlit night and back to the previous sun height
        const e = W.timeTarget.sunElev;
        if (e > -7) { this.daySun = e; W.set('sunElev', -12); } else W.set('sunElev', this.daySun ?? 18);
        this.sync();
      }
      if (k === 'f') this.toggleFull();
      const idx = ['1', '2', '3', '4'].indexOf(k);
      if (idx >= 0) { W.setPreset(Object.keys(PRESETS)[idx]); this.sync(); }
    });
  }

  async toggleSound() {
    const btn = this.root.querySelector('.sound');
    const on = btn.classList.contains('muted');
    btn.classList.toggle('muted', !on);
    await this.audio.enable(on);
  }

  toggleBoat(on = !this.app.boat.visible) {
    this.app.boat.visible = on;
    const btn = this.root.querySelector('.boat');
    btn.classList.toggle('on', on);
    btn.setAttribute('aria-pressed', String(on));
  }

  toggleShip(on = !this.app.ship.visible) {
    this.app.ship.visible = on;
    const btn = this.root.querySelector('.ship');
    btn.classList.toggle('on', on);
    btn.setAttribute('aria-pressed', String(on));
  }
  // whale
  toggleWhale(on = !this.app.whale.visible) {
    this.app.whale.visible = on;
    const btn = this.root.querySelector('.whale');
    btn.classList.toggle('on', on);
    btn.setAttribute('aria-pressed', String(on));
  }
  surfaceWhale() { this.app.whale.surfaceNow(); this.toggleWhale(true); }
  // kaiju
  toggleKaiju(on = !this.app.kaiju.enabled) {
    if (on !== this.app.kaiju.enabled) this.app.kaiju.setEnabled(on);
    const btn = this.root.querySelector('.kaiju');
    btn.classList.toggle('on', on);
    btn.setAttribute('aria-pressed', String(on));
  }
  summonKaiju() { this.app.kaiju.summon(); this.toggleKaiju(true); }

  // full screen where the browser allows it (desktop, Android); otherwise (iPhone, or an iframe without
  // permission) hide every control so only the sea is left, e.g. for screen mirroring
  fullElement() { return document.fullscreenElement || document.webkitFullscreenElement || null; }
  async toggleFull() {
    if (this.fullElement()) { (document.exitFullscreen || document.webkitExitFullscreen).call(document); return; }
    if (this.root.classList.contains('hidden') && this.hiddenByButton) { this.setHidden(false); return; }
    const el = document.documentElement;
    const req = el.requestFullscreen || el.webkitRequestFullscreen;
    if (req && (document.fullscreenEnabled || document.webkitFullscreenEnabled)) {
      try {
        await req.call(el, { navigationUI: 'hide' });
        try { await screen.orientation?.lock?.('landscape'); } catch {}
        return;
      } catch {}
    }
    this.setHidden(true);
    this.cue('Tap the sea to show the controls');
  }
  setHidden(on) {
    this.root.classList.toggle('hidden', on);
    document.getElementById('build')?.style.setProperty('opacity', on ? '0' : ''); // nothing but the sea while mirroring
    this.hiddenByButton = on;
    this.syncFull();
  }
  syncFull() {
    const on = !!this.fullElement() || this.root.classList.contains('hidden');
    const btn = this.root.querySelector('.full');
    btn.classList.toggle('on', on);
    btn.setAttribute('aria-pressed', String(on));
  }

  // auto mode (remembered per browser; ?auto=1 / ?auto=0 overrides)
  toggleAuto(on = !this.auto.on) {
    this.auto.setOn(on);
    const btn = this.root.querySelector('.auto');
    btn.classList.toggle('on', on);
    btn.setAttribute('aria-pressed', String(on));
    try { localStorage.setItem('ow-auto', on ? '1' : '0'); } catch {}
    this.cue(on ? 'Auto on' : 'Auto off');
  }
  cue(text) {
    const el = this.root.querySelector('.cue');
    el.textContent = text;
    el.classList.add('show');
    clearTimeout(this.cueTimer);
    this.cueTimer = setTimeout(() => el.classList.remove('show'), 3500);
  }

  out(key, v) {
    const s = SLIDERS.find((x) => x.key === key);
    this.root.querySelector(`[data-out="${key}"]`).textContent = s.fmt(v);
  }

  sync() {
    const W = this.app.weather;
    const all = { ...W.target, ...W.timeTarget };
    for (const s of SLIDERS) {
      const inp = this.root.querySelector(`input[data-key="${s.key}"]`);
      inp.value = all[s.key];
      this.out(s.key, all[s.key]);
    }
    this.root.querySelectorAll('[data-preset]').forEach((b) => b.classList.toggle('on', b.dataset.preset === W.preset));
    this.toggleBoat(this.app.boat.visible);
    this.toggleShip(this.app.ship.visible);
    this.toggleWhale(this.app.whale.visible); // whale
    this.toggleKaiju(this.app.kaiju.enabled); // kaiju
  }

  idle() {
    let timer;
    const wake = () => {
      this.root.classList.remove('asleep');
      clearTimeout(timer);
      timer = setTimeout(() => { if (!this.root.matches(':hover') && !this.root.classList.contains('open')) this.root.classList.add('asleep'); }, 3200);
    };
    ['pointermove', 'pointerdown', 'keydown', 'wheel'].forEach((e) => window.addEventListener(e, wake, { passive: true }));
    wake();
    setTimeout(() => this.root.querySelector('.hint').classList.add('gone'), 6000);
  }
}

export const UI_CSS = `
.ui { --glass: rgba(14, 18, 24, 0.34); --line: rgba(255, 255, 255, 0.09); --ink: rgba(255, 255, 255, 0.9); --mute: rgba(255, 255, 255, 0.5);
  position: fixed; inset: 0; pointer-events: none; color: var(--ink);
  font: 500 12px/1 ui-sans-serif, -apple-system, "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif; letter-spacing: 0.02em;
  transition: opacity 0.6s ease; -webkit-font-smoothing: antialiased; }
.ui.asleep .dock, .ui.asleep .mark, .ui.hidden > * { opacity: 0; }
.ui.hidden > * { pointer-events: none !important; }
.ui button { all: unset; cursor: pointer; }
.mark { position: absolute; top: 22px; left: 24px; font-size: 11px; letter-spacing: 0.32em; text-transform: uppercase; color: var(--mute); transition: opacity 0.8s ease; }
.hint { position: absolute; left: 50%; bottom: 84px; transform: translateX(-50%); color: var(--mute); font-size: 11px; letter-spacing: 0.08em; transition: opacity 1.2s ease; white-space: nowrap; }
.hint.gone { opacity: 0; }
.dock { position: absolute; left: 50%; bottom: 22px; transform: translateX(-50%); display: flex; gap: 8px; pointer-events: auto; transition: opacity 0.6s ease; }
.pill { display: flex; align-items: center; gap: 2px; padding: 4px; border-radius: 999px; background: var(--glass); border: 1px solid var(--line);
  -webkit-backdrop-filter: blur(16px) saturate(150%); backdrop-filter: blur(16px) saturate(150%); }
.presets button { padding: 9px 15px; border-radius: 999px; color: var(--mute); transition: color 0.25s, background 0.25s; }
.presets button:hover { color: var(--ink); }
.presets button.on { color: #0b0e12; background: rgba(255, 255, 255, 0.88); }
.icon { width: 32px; height: 32px; display: grid; place-items: center; border-radius: 999px; color: var(--mute); transition: color 0.25s, background 0.25s; }
.icon:hover { color: var(--ink); background: rgba(255, 255, 255, 0.06); }
.icon svg { width: 17px; height: 17px; }
.icon.whale, .icon.kaiju { -webkit-touch-callout: none; -webkit-user-select: none; user-select: none; touch-action: manipulation; }
.icon.boat.on, .icon.ship.on { color: var(--ink); }
.icon.whale.on { color: var(--ink); } /* whale */
.icon.kaiju.on { color: var(--ink); } /* kaiju */
.icon.auto.on, .icon.full.on { color: #0b0e12; background: rgba(255, 255, 255, 0.88); }
.ui.hidden > .cue.show { opacity: 1; }
.cue { position: absolute; top: 42px; left: 24px; font-size: 11px; letter-spacing: 0.12em; color: var(--mute); opacity: 0; transition: opacity 0.6s ease; }
.cue.show { opacity: 1; }
.src { position: absolute; top: 58px; left: 24px; font-size: 10px; letter-spacing: 0.06em; color: var(--mute); text-decoration: none; pointer-events: auto; transition: opacity 0.8s ease; }
.src:hover { color: var(--ink); }
.ui.asleep .src { opacity: 0; }
.credit { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); width: min(420px, calc(100vw - 48px)); box-sizing: border-box;
  padding: 22px 26px; border-radius: 18px; background: var(--glass); border: 1px solid var(--line); text-align: center; pointer-events: auto; cursor: default;
  -webkit-backdrop-filter: blur(18px) saturate(150%); backdrop-filter: blur(18px) saturate(150%); animation: credit-in 1.2s ease both; transition: opacity 1.2s ease, transform 1.2s ease; }
.credit.gone { opacity: 0; transform: translate(-50%, calc(-50% - 8px)); pointer-events: none; }
@keyframes credit-in { from { opacity: 0; transform: translate(-50%, calc(-50% + 8px)); } }
@keyframes credit-fade { from { opacity: 0; } }
.credit-k { font-size: 10px; letter-spacing: 0.32em; text-transform: uppercase; color: var(--mute); }
.credit-t { margin-top: 10px; font-size: 22px; letter-spacing: 0.04em; color: var(--ink); }
.credit-t span { font-size: 13px; color: var(--mute); margin-left: 6px; }
.credit-l { display: inline-block; margin-top: 12px; font-size: 13px; color: var(--ink); text-decoration: none; border-bottom: 1px solid rgba(255, 255, 255, 0.3); padding-bottom: 2px; }
.credit-l:hover { border-bottom-color: var(--ink); }
.credit-n { margin-top: 14px; font-size: 11px; line-height: 1.5; color: var(--mute); }
.sound .mute { display: none; }
.sound.muted .mute { display: inline; }
.sound.muted .wave { display: none; }
.ui.open .tune { color: #0b0e12; background: rgba(255, 255, 255, 0.88); }
.sheet { position: absolute; right: 50%; bottom: 74px; transform: translate(calc(50% + 0px), 8px); width: min(340px, calc(100vw - 32px));
  padding: 14px 18px 8px; border-radius: 18px; background: var(--glass); border: 1px solid var(--line);
  -webkit-backdrop-filter: blur(18px) saturate(150%); backdrop-filter: blur(18px) saturate(150%);
  opacity: 0; pointer-events: none; transition: opacity 0.3s ease, transform 0.3s ease; }
.ui.open .sheet { opacity: 1; transform: translate(50%, 0); pointer-events: auto; }
.row { display: grid; grid-template-columns: 1fr auto; align-items: center; row-gap: 2px; padding: 6px 0; }
.row .name { color: var(--mute); }
.row output { font-variant-numeric: tabular-nums; color: var(--ink); }
.row input { grid-column: 1 / -1; -webkit-appearance: none; appearance: none; width: 100%; height: 18px; background: transparent; margin: 0; cursor: pointer; }
.row input::-webkit-slider-runnable-track { height: 1px; background: rgba(255, 255, 255, 0.22); }
.row input::-moz-range-track { height: 1px; background: rgba(255, 255, 255, 0.22); }
.row input::-webkit-slider-thumb { -webkit-appearance: none; width: 11px; height: 11px; margin-top: -5px; border-radius: 50%; background: #fff; box-shadow: 0 0 0 4px rgba(255,255,255,0.08); }
.row input::-moz-range-thumb { width: 11px; height: 11px; border: 0; border-radius: 50%; background: #fff; }
@media (max-width: 560px) {
  /* phones: weather presets stay at the bottom (clear of the home-indicator area); the tool icons
     become a vertical rail on the right edge so they are reachable with a thumb */
  .dock { left: 0; right: 0; bottom: calc(16px + env(safe-area-inset-bottom, 0px)); transform: none; justify-content: center; }
  .tools { position: fixed; right: calc(10px + env(safe-area-inset-right, 0px)); top: 50%; transform: translateY(-50%); flex-direction: column; }
  .tools .icon { width: 40px; height: 40px; }
  .tools .icon svg { width: 19px; height: 19px; }
  .presets button { padding: 9px 11px; }
  .mark { top: 16px; left: 16px; }
  .cue { top: 34px; left: 16px; }
  .src { top: 50px; left: 16px; }
  .credit { left: 16px; right: calc(68px + env(safe-area-inset-right, 0px)); width: auto; transform: translateY(-50%); animation-name: credit-fade; padding: 18px 16px; }
  .credit.gone { transform: translateY(calc(-50% - 8px)); }
  .credit-t { font-size: 19px; }
  .sheet { bottom: calc(72px + env(safe-area-inset-bottom, 0px)); left: 12px; right: calc(68px + env(safe-area-inset-right, 0px)); width: auto;
    max-height: calc(100% - 150px); overflow-y: auto; transform: translateY(8px); }
  .ui.open .sheet { transform: none; }
  .hint { bottom: calc(76px + env(safe-area-inset-bottom, 0px)); }
}
`;
