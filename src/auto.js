// Auto mode: left alone, the scene directs itself. Weather drifts between presets (storms are rarer and
// shorter), the whale comes up to breathe or breaches, and the kaiju swims past every few minutes.
// Each kind of event runs on its own randomised clock; the first whale and kaiju come soon after switching on.

const rnd = (a, b) => a + Math.random() * (b - a);

// how likely each preset is to come next, and how long it lasts (seconds)
const WEATHER = {
  calm: { w: 2, stay: [70, 140] },
  breeze: { w: 3, stay: [80, 160] },
  rain: { w: 2, stay: [60, 120] },
  storm: { w: 1.4, stay: [45, 90] },
};

export class AutoDirector {
  constructor(ui) {
    this.ui = ui;
    this.app = ui.app;
    this.on = false;
    this.timer = 0;
    this.onEvent = null; // (text) => void, for a short caption
  }

  setOn(on) {
    if (on === this.on) return;
    this.on = on;
    clearInterval(this.timer);
    if (!on) return;
    const W = this.app.weather;
    this.next = {
      weather: rnd(...(WEATHER[W.preset]?.stay ?? [60, 120])) * 0.5,
      whale: rnd(6, 14),
      kaiju: rnd(35, 70),
    };
    this.last = performance.now();
    this.timer = setInterval(() => this.tick(), 1000);
  }

  tick() {
    const now = performance.now();
    const dt = Math.min((now - this.last) / 1000, 5); // a hidden tab must not fire everything at once
    this.last = now;
    for (const k in this.next) this.next[k] -= dt;
    if (this.next.weather <= 0) this.weather();
    if (this.next.whale <= 0) this.whale();
    if (this.next.kaiju <= 0) this.kaiju();
  }

  weather() {
    const W = this.app.weather;
    const choices = Object.keys(WEATHER).filter((k) => k !== W.preset);
    let r = Math.random() * choices.reduce((s, k) => s + WEATHER[k].w, 0);
    const pick = choices.find((k) => (r -= WEATHER[k].w) <= 0) ?? choices[0];
    W.setPreset(pick);
    this.ui.sync();
    this.next.weather = rnd(...WEATHER[pick].stay);
    this.onEvent?.(pick[0].toUpperCase() + pick.slice(1));
  }

  whale() {
    const wh = this.app.whale;
    if (Math.random() < 0.45) {
      if (!wh.visible) this.ui.toggleWhale(true);
      wh.breachNow();
      this.onEvent?.('Whale breach');
    } else {
      this.ui.surfaceWhale();
      this.onEvent?.('Whale');
    }
    this.next.whale = rnd(40, 90);
  }

  kaiju() {
    const kj = this.app.kaiju;
    // a pass already under way: try again once it is over
    if (kj.enabled && kj.state === 'pass') { this.next.kaiju = 30; return; }
    this.ui.summonKaiju();
    this.onEvent?.('Kaiju');
    this.next.kaiju = rnd(180, 320);
  }
}
