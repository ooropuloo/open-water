// Humpback whale (Megaptera novaeangliae), 14 m, built as lofted surfaces (no primitives, no models).
//
// Whale-local metres: x forward (snout tip at +7), y dorsal, z lateral; d = 7 - x is the distance
// from the snout. The body is one loft along the spine: 200 cross-sections, each a superellipse with
// its own dorsal, ventral and lateral radii (flat-topped rostrum, deep pleated throat, laterally
// compressed tail stock with dorsal and ventral keels), with the hump and small dorsal fin and the
// knuckles of the tail stock raised out of the same surface. The long pectoral flippers (a third of
// the body length, scalloped leading edge) and the flukes (swept, serrated trailing edge, central
// notch) are separate airfoil lofts whose roots are buried in the body.
//
// Every vertex carries two bone indices and a weight for GPU skinning: spine bones along the body,
// two per flipper (root and a flex bone at mid-span) and two for the flukes (root pitch, tip flex).
import { makeMesh } from '../gl.js';
import { rng } from '../math.js';

export const SNOUT = 7.0;
export const BODY_END = 13.2;     // d where the tail stock ends (the flukes take over)
const CAP = 0.28;                  // rounded end of the tail stock, inside the fluke root
const TIP = 0.55;                  // length of the rounded snout cap

// ---------------------------------------------------------------- monotone cubic profiles
function mono(pts) {
  const n = pts.length, xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const dk = [], m = new Array(n);
  for (let i = 0; i < n - 1; i++) dk.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));
  m[0] = dk[0]; m[n - 1] = dk[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = dk[i - 1] * dk[i] <= 0 ? 0 : (dk[i - 1] + dk[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (dk[i] === 0) { m[i] = 0; m[i + 1] = 0; continue; }
    const a = m[i] / dk[i], b = m[i + 1] / dk[i], s = a * a + b * b;
    if (s > 9) { const t = 3 / Math.sqrt(s); m[i] = t * a * dk[i]; m[i + 1] = t * b * dk[i]; }
  }
  return (x) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let i = 0;
    while (x > xs[i + 1]) i++;
    const h = xs[i + 1] - xs[i], t = (x - xs[i]) / h, t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * m[i + 1];
  };
}
const sstep = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };

// dorsal line, ventral line, height of the widest point, half-width (all vs d)
const TOP = mono([[0, 0.02], [0.5, 0.17], [1.2, 0.33], [2.2, 0.55], [3.2, 0.78], [3.75, 0.92], [4.05, 0.95], [4.5, 0.97], [5.4, 1.10],
  [6.4, 1.18], [7.4, 1.14], [8.3, 1.02], [9.5, 0.84], [10.5, 0.66], [11.5, 0.49], [12.4, 0.35], [13.0, 0.25], [13.2, 0.21]]);
const BOT = mono([[0, -0.02], [0.5, -0.28], [1.2, -0.58], [2.2, -0.96], [3.2, -1.30], [4.2, -1.56], [5.2, -1.72], [6.2, -1.76],
  [7.2, -1.66], [8.2, -1.44], [9.2, -1.16], [10.2, -0.86], [11.2, -0.58], [12.1, -0.39], [12.8, -0.28], [13.2, -0.22]]);
const MID = mono([[0, -0.02], [1, -0.08], [2, -0.15], [3, -0.22], [4, -0.27], [6, -0.22], [8, -0.15], [10, -0.08], [13.2, -0.01]]);
const WID = mono([[0, 0.36], [0.5, 0.43], [1.2, 0.56], [2.2, 0.72], [3.2, 0.88], [4.2, 1.10], [5.2, 1.32], [6.2, 1.42], [7.2, 1.34],
  [8.2, 1.14], [9.2, 0.95], [10.2, 0.72], [11.2, 0.50], [12.1, 0.32], [12.8, 0.22], [13.2, 0.17]]);
// superellipse exponents of the upper and lower half (2 = ellipse; >2 flat-topped; <2 keeled)
const NTOP = mono([[0, 2.2], [1.0, 2.9], [3.4, 2.8], [4.6, 2.3], [8, 2.15], [10, 1.9], [11.5, 1.5], [13.2, 1.35]]);
const NBOT = mono([[0, 2.1], [2.5, 2.25], [6, 2.2], [9, 2.0], [10.5, 1.75], [12, 1.45], [13.2, 1.4]]);

// dorsal fin on its hump: gradual leading edge, steep (falcate) trailing edge
function dorsalFin(d) {
  if (d < 8.0 || d > 10.4) return 0;
  const rise = sstep(8.2, 9.32, d), fall = 1 - sstep(9.32, 9.72, d);
  return 0.34 * Math.pow(rise, 1.6) * fall;
}
// broad hump under the fin, and the knuckles along the dorsal ridge of the tail stock
function dorsalRidge(d) {
  let r = 0.07 * Math.exp(-(((d - 9.2) / 0.9) ** 2));
  if (d > 10.0 && d < 12.6) {
    const k = (d - 10.0) / 0.42;
    r += 0.028 * Math.pow(Math.max(Math.cos(Math.PI * (k - Math.round(k))), 0), 3) * sstep(10.0, 10.5, d) * (1 - sstep(12.0, 12.6, d));
  }
  return r;
}

// rest-local point of the body surface at distance d from the snout, angle th around the axis (0 = top)
export function bodyPoint(d, th) {
  let yT = TOP(d), yB = BOT(d), yW = MID(d), w = WID(d);
  let k = 1;
  if (d < TIP) { const q = 1 - d / TIP; k = Math.sqrt(Math.max(1 - q * q, 0)); }
  if (d > BODY_END) { const q = (d - BODY_END) / CAP; k = Math.sqrt(Math.max(1 - q * q, 0)); }
  yT = yW + (yT - yW) * k; yB = yW + (yB - yW) * k; w *= k;
  const c = Math.cos(th), s = Math.sin(th);
  const n = c >= 0 ? NTOP(d) : NBOT(d);
  const cy = Math.sign(c) * Math.pow(Math.abs(c), 2 / n);
  const sz = Math.sign(s) * Math.pow(Math.abs(s), 2 / n);
  let y = c >= 0 ? yW + (yT - yW) * cy : yW + (yW - yB) * cy;
  const z = w * sz;
  if (c > 0) {
    const fw = 0.085 + 0.12 * (1 - Math.min(dorsalFin(d) / 0.34, 1));
    y += (dorsalFin(d) * Math.exp(-((z / fw) ** 2)) + dorsalRidge(d) * Math.exp(-((z / 0.28) ** 2))) * k;
    // splash guard: the raised rim in front of the blowholes
    y += 0.05 * Math.exp(-(((d - 3.62) / 0.16) ** 2)) * Math.exp(-((z / 0.32) ** 2));
  }
  return [SNOUT - d, y, z];
}
// cross-section dimensions at d (for proxies: shadows, the waterline wash, buoyancy points)
export function bodySection(d) {
  return { top: TOP(d) - MID(d), bot: MID(d) - BOT(d), w: WID(d), mid: MID(d), yT: TOP(d) };
}

// ---------------------------------------------------------------- bones
export const NSPINE = 28;
export const SPINE_X = Array.from({ length: NSPINE }, (_, i) => -5.9 + (i * (5.5 + 5.9)) / (NSPINE - 1));
export const B_PECT_R = NSPINE, B_PECT_R2 = NSPINE + 1, B_PECT_L = NSPINE + 2, B_PECT_L2 = NSPINE + 3;
export const B_FLUKE = NSPINE + 4, B_FLUKE2 = NSPINE + 5;
export const NBONES = NSPINE + 6;
export const FLUKE_X = -6.08, FLUKE2_X = -6.5;
// spine weights for a rest x: two neighbouring bones
function spineWeights(x) {
  if (x <= SPINE_X[0]) return [0, 0, 0];
  if (x >= SPINE_X[NSPINE - 1]) return [NSPINE - 1, NSPINE - 1, 0];
  const f = (x - SPINE_X[0]) / (SPINE_X[1] - SPINE_X[0]);
  const i = Math.min(Math.floor(f), NSPINE - 2);
  return [i, i + 1, f - i];
}

// ---------------------------------------------------------------- pectoral flipper
export const PECT_LEN = 4.4;
// root on the lower flank behind the gape, d = 4.25
export const PECT_D = 4.25;
function solveFlankZ(d, y) {
  // angle on the z>0 side where the body surface passes height y
  let a = Math.PI * 0.5, b = Math.PI;
  for (let i = 0; i < 40; i++) { const m = (a + b) / 2; if (bodyPoint(d, m)[1] > y) a = m; else b = m; }
  return bodyPoint(d, (a + b) / 2)[2];
}
const PECT_Y = -0.80;
export const PECT_ROOT = [SNOUT - PECT_D, PECT_Y, solveFlankZ(PECT_D, PECT_Y) - 0.16];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
// rest basis of the right flipper: span (down, back, out), chord towards the leading edge, thickness
export const PECT_SPAN = norm([-0.42, -0.60, 0.68]);
const leRaw = [0.85, 0.35, 0.25];
export const PECT_CHORD = norm(leRaw.map((v, i) => v - dot(leRaw, PECT_SPAN) * PECT_SPAN[i]));
export const PECT_THICK = (() => { const t = cross(PECT_SPAN, PECT_CHORD); return t[1] < 0 ? t.map((v) => -v) : t; })();

const tubRng = rng(77);
const TUB = Array.from({ length: 12 }, () => 0.7 + 0.6 * tubRng());
function pectChord(s) {
  // narrow wrist at the root, broadest at a quarter span, long taper, rounded tip
  let c = s < 0.25 ? 0.62 + 0.36 * sstep(0, 0.25, s) : 0.98 - 0.55 * Math.pow((s - 0.25) / 0.75, 1.35);
  if (s > 0.82) c *= Math.sqrt(Math.max(1 - ((s - 0.82) / 0.18) ** 2, 0));
  return Math.max(c, 0);
}
// leading-edge tubercles: about ten knobs from a fifth of the span to the tip
function tubercle(s) {
  const p = (s - 0.2) / 0.074;
  if (p < -0.5 || s > 0.99) return 0;
  const i = Math.round(p), f = p - i;
  const amp = 0.05 * TUB[Math.min(Math.max(i, 0), 11)] * sstep(0.15, 0.32, s) * (1 - 0.35 * sstep(0.7, 1.0, s));
  return amp * Math.pow(Math.cos(Math.PI * f), 2);
}
// fin-local: (span, chordwise towards LE, thickness)
function pectLocal(s, ph) {
  const c = pectChord(s);
  const cp = (1 - Math.cos(ph)) / 2;          // 0 at the leading edge, 1 at the trailing edge
  const upper = ph <= Math.PI ? 1 : -1;
  const tub = tubercle(s);
  const le = 0.40 * c + tub;
  const chordY = le - cp * (c + tub);
  const tmax = (s < 0.1 ? 0.30 - 0.6 * s : 0.24 - 0.15 * sstep(0.1, 1.0, s)) * Math.min(c / 0.9 + 0.25, 1.15);
  const af = 5 * (0.2969 * Math.sqrt(cp) - 0.126 * cp - 0.3516 * cp * cp + 0.2843 * cp ** 3 - 0.1036 * cp ** 4);
  const knob = 1 + 0.9 * (tub / 0.05) * Math.pow(1 - cp, 6);
  const camber = 0.05 * c * Math.sin(Math.PI * cp) * sstep(0, 0.4, s);
  const bow = -0.32 * s * s;   // the flipper curves back towards its tip
  return [s * PECT_LEN, chordY + bow, upper * tmax * 0.5 * af * knob + camber];
}
function pectRest(s, ph, side) {
  const l = pectLocal(s, ph);
  const p = [0, 1, 2].map((i) => PECT_ROOT[i] + PECT_SPAN[i] * l[0] + PECT_CHORD[i] * l[1] + PECT_THICK[i] * l[2]);
  if (side < 0) p[2] = -p[2];
  return p;
}

// ---------------------------------------------------------------- flukes
export const FLUKE_SPAN = 2.32;     // half span (tip to tip 4.6 m, a third of the body)
const FX0 = -6.05;
const serRng = rng(19);
const SER = Array.from({ length: 64 }, () => 0.55 + 0.9 * serRng());
function flukeTE(z) {
  const eta = Math.abs(z) / FLUKE_SPAN;
  // irregular scallops of the trailing edge (the humpback's serrations), larger outboard
  const p = eta * 21 + 0.3;
  const i = Math.floor(p), f = p - i;
  const k = Math.min(i, 63);
  const amp = (z < 0 ? SER[(k * 7 + 3) % 64] : SER[k]) * 0.032 * (0.35 + 0.65 * sstep(0.1, 0.5, eta)) * (1 - sstep(0.88, 1.0, eta));
  return -amp * Math.pow(Math.sin(Math.PI * f), 0.7);
}
function flukeFrame(z) {
  const eta = Math.min(Math.abs(z) / FLUKE_SPAN, 1);
  const xq = FX0 - 0.10 - 1.0 * Math.pow(eta, 1.25);             // swept quarter-chord line
  let c = 1.08 * (1 - 0.3 * eta) * Math.pow(Math.max(1 - eta ** 2.4, 0), 0.6);
  const le = xq + 0.27 * c;
  let te = xq - 0.73 * c;
  // central notch, and the S-shaped trailing edge: concave near the root, convex outboard
  te += 0.20 * Math.exp(-((eta / 0.045) ** 2)) + 0.07 * Math.sin(Math.PI * Math.min(eta / 0.55, 1)) * (1 - eta);
  te += flukeTE(z);
  c = Math.max(le - te, 0);
  const tmax = (0.30 * Math.pow(1 - eta, 2.4) + 0.028) * Math.min(c / 0.5, 1) * (eta > 0.97 ? Math.sqrt(Math.max((1 - eta) / 0.03, 0)) : 1);
  return { le, te, c, tmax, eta };
}
function flukeRest(z, ph) {
  const f = flukeFrame(z);
  const cp = (1 - Math.cos(ph)) / 2;
  const upper = ph <= Math.PI ? 1 : -1;
  const af = 5 * (0.2969 * Math.sqrt(cp) - 0.126 * cp - 0.3516 * cp * cp + 0.2843 * cp ** 3 - 0.1036 * cp ** 4);
  const x = f.le - cp * (f.le - f.te);
  // the root swells into the tail stock (a fillet instead of a crease)
  const fil = 1 + 1.1 * Math.exp(-((f.eta / 0.09) ** 2)) * (1 - 0.6 * cp);
  // slight upward bow of the blades
  const y = -0.01 + upper * f.tmax * 0.5 * af * fil + 0.05 * f.eta * f.eta;
  return [x, y, z];
}

// ---------------------------------------------------------------- mesh assembly
function addGrid(B, rows, cols, fn, wrapCols, normalOut) {
  const base = B.P.length / 3;
  const pts = [];
  let vote = 0;
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) pts.push(fn(i, j));
  const at = (i, j) => {
    i = Math.min(Math.max(i, 0), rows - 1);
    j = wrapCols ? ((j % cols) + cols) % cols : Math.min(Math.max(j, 0), cols - 1);
    return pts[i * cols + j].p;
  };
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const q = pts[i * cols + j];
    let du = [0, 1, 2].map((k) => at(i + 1, j)[k] - at(i - 1, j)[k]);
    let dv = [0, 1, 2].map((k) => at(i, j + 1)[k] - at(i, j - 1)[k]);
    // at a pole (a collapsed ring) use the next ring's spread
    if (Math.hypot(...dv) < 1e-6) {
      const ii = i === 0 ? 1 : rows - 2;
      dv = [0, 1, 2].map((k) => at(ii, j + 1)[k] - at(ii, j - 1)[k]);
      if (Math.hypot(...du) < 1e-6) du = [0, 1, 2].map((k) => at(ii + (i === 0 ? 1 : -1), j)[k] - at(ii, j)[k]);
    }
    const n = norm(cross(du, dv));
    vote += dot(n, normalOut(q)) >= 0 ? 1 : -1;
    B.P.push(...q.p); B.N.push(...(q.pole || n)); B.T.push(...q.t); B.W.push(...q.w);
  }
  // one orientation for the whole surface (per-vertex tests fail at thin airfoil edges)
  if (vote < 0) for (let v = base; v < B.P.length / 3; v++) if (!pts[v - base].pole) for (let k = 0; k < 3; k++) B.N[v * 3 + k] *= -1;
  const cc = wrapCols ? cols : cols - 1;
  for (let i = 0; i < rows - 1; i++) for (let j = 0; j < cc; j++) {
    const j1 = (j + 1) % cols;
    const a = base + i * cols + j, b = base + i * cols + j1, c = a + cols, e = b + cols;
    B.I.push(a, c, b, b, c, e);
  }
}

export function buildWhaleMesh(gl) {
  const B = { P: [], N: [], T: [], W: [], I: [] };
  // body: rings dense at both ends, angles dense along the dorsal and ventral midlines
  const NR = 200, NA = 88, DMAX = BODY_END + CAP;
  addGrid(B, NR + 1, NA, (i, j) => {
    const t = i / NR;
    const d = DMAX * (t - 0.62 * Math.sin(2 * Math.PI * t) / (2 * Math.PI));
    const u = j / NA;
    const th = 2 * Math.PI * u - 0.32 * Math.sin(4 * Math.PI * u);
    const p = bodyPoint(d, th);
    let w;
    if (d > BODY_END - 0.35) {
      const [b0] = spineWeights(p[0]);
      w = [b0, B_FLUKE, sstep(BODY_END - 0.35, BODY_END + 0.1, d), 0];
    } else {
      const [b0, b1, f] = spineWeights(p[0]);
      w = [b0, b1, f, 0];
    }
    const pole = i === 0 ? [1, 0, 0] : i === NR ? [-1, 0, 0] : null;
    return { p, t: [d, Math.cos(th)], w, pole };
  }, true, (q) => [0, q.p[1] - MID(SNOUT - q.p[0]), q.p[2]]);

  // flippers: span from inside the body to the tip, airfoil loop
  const ia = spineWeights(PECT_ROOT[0]);
  const attach = ia[2] < 0.5 ? ia[0] : ia[1];
  for (const side of [1, -1]) {
    const NS = 96, NC = 44;
    addGrid(B, NS + 1, NC, (i, j) => {
      const t = i / NS;
      const s = -0.06 + 1.06 * (1 - Math.pow(1 - t, 1.25));
      const ph = (2 * Math.PI * j) / NC;
      const p = pectRest(Math.min(s, 1), ph, side);
      const b0 = side > 0 ? B_PECT_R : B_PECT_L, b1 = side > 0 ? B_PECT_R2 : B_PECT_L2;
      // the buried root follows the body
      const wb = sstep(0.0, 0.08, s);
      const w = s < 0.08 ? [attach, b0, wb, 1] : [b0, b1, sstep(0.25, 0.75, s), 1];
      return { p, t: [Math.max(s, 0), Math.cos(ph)], w };
    }, true, (q) => {
      // outward: away from the fin's chord plane (a vote over the whole surface)
      const c = [PECT_ROOT[0], PECT_ROOT[1], PECT_ROOT[2] * side];
      const tn = side > 0 ? PECT_THICK : [PECT_THICK[0], PECT_THICK[1], -PECT_THICK[2]];
      const rel = [q.p[0] - c[0], q.p[1] - c[1], q.p[2] - c[2]];
      return tn.map((v) => v * Math.sign(dot(rel, tn)));
    });
  }
  // flukes: span tip to tip, airfoil loop
  {
    const NS = 200, NC = 40;
    addGrid(B, NS + 1, NC, (i, j) => {
      const t = i / NS;
      const z = FLUKE_SPAN * Math.sign(t - 0.5) * Math.pow(Math.abs(2 * t - 1), 1.15) * 0.9999;
      const ph = (2 * Math.PI * j) / NC;
      const p = flukeRest(z, ph);
      const eta = Math.abs(z) / FLUKE_SPAN;
      return { p, t: [z, Math.cos(ph)], w: [B_FLUKE, B_FLUKE2, sstep(0.2, 0.85, eta), 2] };
    }, true, (q) => [0, q.p[1] + 0.01, 0]);   // (upper surface up)
  }
  const mesh = makeMesh(gl, [[new Float32Array(B.P), 3], [new Float32Array(B.N), 3], [new Float32Array(B.T), 2], [new Float32Array(B.W), 4]], new Uint32Array(B.I));
  mesh.vertices = B.P.length / 3;
  mesh.attach = attach;
  return mesh;
}

// rest-local points used on the CPU (effects, proxies)
export const BLOWHOLE = (() => { const p = bodyPoint(3.92, 0); return [p[0], p[1] + 0.02, 0]; })();
export function flukeEdgePoints(n = 24) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const z = FLUKE_SPAN * (-0.97 + (1.94 * i) / (n - 1));
    const f = flukeFrame(z);
    out.push({ p: [f.te, -0.01 + 0.05 * f.eta * f.eta, z], eta: f.eta });
  }
  return out;
}
export function pectPoint(s, side) { return pectRest(s, Math.PI * 0.5, side); }

// tubercles of the head: a median row and paired rows along the rostrum and the upper jaw, and a
// scatter over the chin and lower jaw (rest-local centre, radius)
export function headKnobs() {
  const r = rng(5), out = [];
  const put = (d, th, rad) => { const p = bodyPoint(d, th); out.push([p[0], p[1], p[2], rad]); };
  for (let d = 0.42; d < 3.35; d += 0.24 + 0.08 * r()) put(d, (r() - 0.5) * 0.05, 0.05 + 0.02 * r());
  for (const sg of [1, -1]) {
    for (let d = 0.55; d < 3.4; d += 0.30 + 0.1 * r()) put(d, sg * (0.42 + 0.06 * (r() - 0.5)), 0.045 + 0.02 * r());
    for (let d = 0.35; d < 3.2; d += 0.32 + 0.12 * r()) put(d, sg * (1.18 + 0.1 * (r() - 0.5)), 0.04 + 0.018 * r());
    for (let i = 0; i < 16; i++) put(0.12 + 1.9 * Math.pow(r(), 1.2), sg * (1.85 + 1.25 * r()), 0.05 + 0.03 * r());
  }
  return out.slice(0, 80);
}
export const KNOBS = 80;
