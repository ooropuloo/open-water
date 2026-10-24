// kaiju: a synthesized roar (Web Audio, no samples). A distorted, formant-filtered screech with a
// rising-then-falling pitch over a sub-bass growl and a breath of noise, heard from far away: it
// arrives after the sound's travel time (d / 343 m/s), loses its highs to the air and rolls on in a
// long open-sea reverberation.
let impulse = null;
function reverbIR(ctx) {
  if (impulse && impulse.sampleRate === ctx.sampleRate) return impulse;
  const n = Math.floor(ctx.sampleRate * 4.5);
  impulse = ctx.createBuffer(2, n, ctx.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const d = impulse.getChannelData(ch);
    let s = 1234 + ch * 77, lp = 0;
    for (let i = 0; i < n; i++) {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      lp += ((s / 4294967296) * 2 - 1 - lp) * 0.25;
      const t = i / ctx.sampleRate;
      d[i] = lp * Math.exp(-t / 1.25) * (t < 0.02 ? t / 0.02 : 1);
    }
  }
  return impulse;
}
function shaper(ctx, k) {
  const ws = ctx.createWaveShaper(), c = new Float32Array(1024);
  for (let i = 0; i < 1024; i++) { const x = (i / 1023) * 2 - 1; c[i] = Math.tanh(k * x) / Math.tanh(k); }
  ws.curve = c;
  return ws;
}

// ctx: running AudioContext, dest: node to connect to, dist: metres, dur: seconds
export function roar(ctx, dest, dist = 700, dur = 4.6) {
  const t0 = ctx.currentTime + dist / 343;
  const out = ctx.createGain();
  // spherical spreading against a 300 m reference, with a floor (it is very loud)
  out.gain.value = Math.min(1.1, 300 / Math.max(dist, 150)) * 0.9;
  const air = ctx.createBiquadFilter();
  air.type = 'lowpass';
  air.frequency.value = Math.max(1200, 9000 - dist * 4.5);
  const dry = ctx.createGain(); dry.gain.value = 0.55;
  const wet = ctx.createGain(); wet.gain.value = 0.6;
  const conv = ctx.createConvolver(); conv.buffer = reverbIR(ctx);
  out.connect(air);
  air.connect(dry).connect(dest);
  air.connect(conv).connect(wet).connect(dest);

  const env = (g, a, peak, hold, rel) => {
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(peak, t0 + a);
    g.gain.setValueAtTime(peak, t0 + a + hold);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + a + hold + rel);
  };
  // the screech: detuned saws through a vocal tract (formants), distorted, with a fluttering tremolo
  const scream = ctx.createGain();
  const dist1 = shaper(ctx, 3.5);
  const pre = ctx.createGain(); pre.gain.value = 0.5;
  pre.connect(dist1);
  for (const [f, q, g] of [[480, 6, 1.0], [1150, 7, 0.7], [2500, 8, 0.35]]) {
    const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = f; bp.Q.value = q;
    bp.frequency.setValueAtTime(f * 0.85, t0); bp.frequency.linearRampToValueAtTime(f * 1.08, t0 + dur * 0.3); bp.frequency.linearRampToValueAtTime(f * 0.8, t0 + dur);
    const gg = ctx.createGain(); gg.gain.value = g;
    dist1.connect(bp).connect(gg).connect(scream);
  }
  const trem = ctx.createGain(); trem.gain.value = 0.7;
  const lfo = ctx.createOscillator(); lfo.frequency.value = 27;
  const lfoG = ctx.createGain(); lfoG.gain.value = 0.3;
  lfo.connect(lfoG).connect(trem.gain);
  scream.connect(trem).connect(out);
  env(scream, 0.35, 0.9, dur * 0.55, dur * 0.4);
  const oscs = [lfo];
  for (const det of [-14, -4, 6, 17]) {
    const o = ctx.createOscillator(); o.type = 'sawtooth';
    o.detune.value = det;
    o.frequency.setValueAtTime(150, t0);
    o.frequency.exponentialRampToValueAtTime(240, t0 + dur * 0.25);
    o.frequency.exponentialRampToValueAtTime(210, t0 + dur * 0.6);
    o.frequency.exponentialRampToValueAtTime(120, t0 + dur);
    // vocal-fold irregularity
    const j = ctx.createOscillator(); j.frequency.value = 6 + Math.random() * 5;
    const jg = ctx.createGain(); jg.gain.value = 7;
    j.connect(jg).connect(o.frequency);
    o.connect(pre);
    oscs.push(o, j);
  }
  // sub-bass growl
  const growl = ctx.createGain();
  const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 260;
  const d2 = shaper(ctx, 2.2);
  for (const f of [43, 45.5, 61]) {
    const o = ctx.createOscillator(); o.type = 'sawtooth'; o.frequency.setValueAtTime(f, t0); o.frequency.linearRampToValueAtTime(f * 0.82, t0 + dur);
    o.connect(d2); oscs.push(o);
  }
  d2.connect(lp).connect(growl).connect(out);
  env(growl, 0.25, 0.8, dur * 0.6, dur * 0.45);
  // breath
  const nb = ctx.createBuffer(1, Math.floor(ctx.sampleRate * (dur + 1)), ctx.sampleRate);
  const nd = nb.getChannelData(0);
  for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
  const ns = ctx.createBufferSource(); ns.buffer = nb;
  const nbp = ctx.createBiquadFilter(); nbp.type = 'bandpass'; nbp.frequency.value = 900; nbp.Q.value = 0.8;
  const ng = ctx.createGain();
  ns.connect(nbp).connect(ng).connect(out);
  env(ng, 0.4, 0.28, dur * 0.5, dur * 0.4);
  oscs.push(ns);
  for (const o of oscs) { o.start(t0); o.stop(t0 + dur + 0.5); }
  setTimeout(() => { try { out.disconnect(); } catch { /* already gone */ } }, (dist / 343 + dur + 6) * 1000);
}
