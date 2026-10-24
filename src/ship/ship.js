// A grey warship (modern destroyer / frigate, 130 m overall), procedural: no models, no textures.
// It steams slowly across the view on a large ellipse well beyond the rowboat.
//
// Geometry: one merged static mesh (a lofted hull of ~8k vertices with flared bow, round bilge and
// sheer, a forecastle bulwark; faceted "stealth" superstructure prisms with tumblehome; mast,
// funnels, gun, launcher cells, boats, life-raft canisters, rails, bollards, antennas), each vertex
// tagged with a material id that the shader turns into weathered paint, non-skid, glass, etc.
// The radar rotates (its own small mesh); the bow wave, hull wash and transom churn are foam sheets
// drawn with alpha-to-coverage; the wake on the open sea is drawn by the water shader (wake.js).
//
// Flotation: five FFT probes (bow, stern quarters, both sides, centre) filtered at least as
// coarsely as the water mesh there; slow damped springs for heave, pitch and roll.
// Light: as the sea and the rowboat (sun / moon through the atmosphere, cloud shadow, environment
// light, lightning, aerial perspective, shafts, rain fog), plus the ship's own sun shadow map.
// Navigation lights: HDR sprites with their regulation arcs, at real lamp intensities.
import { Program, bindEmpty, makeMesh, FULLSCREEN_VS } from '../gl.js';
import { mul, viewFromBasis } from '../math.js';
import { SKY_LOOKUP_GLSL } from '../sky/atmosphere.js';
import { SXS, SXB, STEM_WL, shipU, deckH, keelH, beamDeck, halfWidth, SHIP_HULL_GLSL, WAKE_N, WAKE_DS } from './wake.js';

// ---------------------------------------------------------------- small vector helpers
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const sc = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const anyPerp = (d) => norm(cross(d, Math.abs(d[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]));
const fromBasis = (o, X, Y, Z) => new Float32Array([...X, 0, ...Y, 0, ...Z, 0, ...o, 1]);
const G = (x) => deckH(shipU(x));
const edgeZ = (x) => { const u = shipU(x); return halfWidth(u, deckH(u)); };
const deg = Math.PI / 180;

function invertRigid(m) {
  const o = new Float32Array(16);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[c * 4 + r] = m[r * 4 + c];
  for (let r = 0; r < 3; r++) o[12 + r] = -(o[r] * m[12] + o[4 + r] * m[13] + o[8 + r] * m[14]);
  o[15] = 1;
  return o;
}
const rot3 = (B) => new Float32Array([B[0], B[1], B[2], B[4], B[5], B[6], B[8], B[9], B[10]]);

// ---------------------------------------------------------------- mesh building
// material ids (see material() in the shader)
const M = { hull: 0, deck: 1, house: 2, dark: 3, radome: 5, cells: 6, steel: 7, rubber: 8, raft: 9, door: 10, bridge: 12, panel: 13, foam: 14 };

function builder() {
  const P = [], N = [], T = [], A = [], I = [];
  let mat = 0, aux = -1;
  return {
    P, N, T, A, I,
    set(m, a = -1) { mat = m; aux = a; return this; },
    v(p, n, t = [0, 0]) { P.push(p[0], p[1], p[2]); N.push(n[0], n[1], n[2]); T.push(t[0], t[1]); A.push(mat, aux); return P.length / 3 - 1; },
    mesh(gl) {
      return makeMesh(gl, [[new Float32Array(P), 3], [new Float32Array(N), 3], [new Float32Array(T), 2], [new Float32Array(A), 2]], new Uint32Array(I));
    },
  };
}
// smooth grid surface; normals from central differences (orientation fixed in the shader)
function grid(b, rows, cols, fn) {
  const pts = [];
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) pts.push(fn(i, j));
  const at = (i, j) => pts[Math.min(Math.max(i, 0), rows - 1) * cols + Math.min(Math.max(j, 0), cols - 1)].p;
  const base = b.P.length / 3;
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    let n = cross(sub(at(i + 1, j), at(i - 1, j)), sub(at(i, j + 1), at(i, j - 1)));
    if (len(n) < 1e-10) n = [0, 1, 0];
    b.v(pts[i * cols + j].p, norm(n), pts[i * cols + j].t);
  }
  for (let i = 0; i < rows - 1; i++) for (let j = 0; j < cols - 1; j++) {
    const a = base + i * cols + j, c = a + cols;
    b.I.push(a, c, a + 1, a + 1, c, c + 1);
  }
}
// flat convex polygon (Newell normal)
function face(b, pts, uv) {
  const k = pts.length;
  let n = [0, 0, 0];
  for (let i = 0; i < k; i++) {
    const p = pts[i], q = pts[(i + 1) % k];
    n = add(n, [(p[1] - q[1]) * (p[2] + q[2]), (p[2] - q[2]) * (p[0] + q[0]), (p[0] - q[0]) * (p[1] + q[1])]);
  }
  n = norm(n);
  const base = b.P.length / 3;
  pts.forEach((p, i) => b.v(p, n, uv ? uv[i] : [0, 0]));
  for (let i = 1; i < k - 1; i++) b.I.push(base, base + i, base + i + 1);
}
// prism between a bottom and a top polygon given in the xz plane
function prism(b, bot, top, y0, y1, cap = true) {
  const k = bot.length;
  for (let i = 0; i < k; i++) {
    const j = (i + 1) % k;
    face(b, [[bot[i][0], y0, bot[i][1]], [bot[j][0], y0, bot[j][1]], [top[j][0], y1, top[j][1]], [top[i][0], y1, top[i][1]]]);
  }
  if (cap) face(b, top.map((p) => [p[0], y1, p[1]]));
}
// plan of a deckhouse: rectangle x0..x1, half-width hw, with chamfered fore / aft corners
const plan = (x0, x1, hw, chF, chA) => [[x0 + chA, -hw], [x1 - chF, -hw], [x1, -hw + chF], [x1, hw - chF], [x1 - chF, hw], [x0 + chA, hw], [x0, hw - chA], [x0, -hw + chA]];
// faceted deckhouse with tumblehome (sS), sloped front (sF) and back (sA); negative sA rakes it aft
function house(b, { x0, x1, hw, y0, y1, chF = 0, chA = 0, sS = 0, sF = 0, sA = 0 }) {
  prism(b, plan(x0, x1, hw, chF, chA), plan(x0 + sA, x1 - sF, hw - sS, chF * 0.85, chA * 0.85), y0, y1);
}
// box with unit axes X, Y, Z and size s
function obox(b, c, s, X = [1, 0, 0], Y = [0, 1, 0], Z = [0, 0, 1]) {
  const p = (i, j, k) => add(c, add(sc(X, i * s[0] / 2), add(sc(Y, j * s[1] / 2), sc(Z, k * s[2] / 2))));
  face(b, [p(1, -1, -1), p(1, 1, -1), p(1, 1, 1), p(1, -1, 1)]);
  face(b, [p(-1, -1, -1), p(-1, -1, 1), p(-1, 1, 1), p(-1, 1, -1)]);
  face(b, [p(-1, 1, -1), p(-1, 1, 1), p(1, 1, 1), p(1, 1, -1)]);
  face(b, [p(-1, -1, -1), p(1, -1, -1), p(1, -1, 1), p(-1, -1, 1)]);
  face(b, [p(-1, -1, 1), p(1, -1, 1), p(1, 1, 1), p(-1, 1, 1)]);
  face(b, [p(-1, -1, -1), p(-1, 1, -1), p(1, 1, -1), p(1, -1, -1)]);
}
// tapered cylinder from a to e
function cyl(b, a, e, r0, r1 = r0, seg = 12, caps = true) {
  const d = norm(sub(e, a)), X = anyPerp(d), Z = cross(X, d);
  const base = b.P.length / 3;
  for (let k = 0; k <= seg; k++) {
    const t = (k / seg) * 2 * Math.PI, n = add(sc(X, Math.cos(t)), sc(Z, Math.sin(t)));
    b.v(add(a, sc(n, r0)), n); b.v(add(e, sc(n, r1)), n);
  }
  for (let k = 0; k < seg; k++) { const i = base + 2 * k; b.I.push(i, i + 1, i + 2, i + 2, i + 1, i + 3); }
  if (!caps) return;
  for (const [c, r, s] of [[a, r0, -1], [e, r1, 1]]) {
    const ring = [];
    for (let k = 0; k < seg; k++) { const t = (k / seg) * 2 * Math.PI; ring.push(add(c, sc(add(sc(X, Math.cos(t)), sc(Z, Math.sin(t))), r))); }
    face(b, s > 0 ? ring : ring.reverse());
  }
}
function sphere(b, c, r, seg = 14, rings = 9, X = [1, 0, 0], Y = [0, 1, 0], Z = [0, 0, 1]) {
  const base = b.P.length / 3;
  for (let i = 0; i <= rings; i++) for (let k = 0; k <= seg; k++) {
    const t = (i / rings) * Math.PI, a = (k / seg) * 2 * Math.PI;
    const u = [Math.sin(t) * Math.cos(a), Math.cos(t), Math.sin(t) * Math.sin(a)];
    const p = add(sc(X, u[0] * r[0]), add(sc(Y, u[1] * r[1]), sc(Z, u[2] * r[2])));
    const n = norm(add(sc(X, u[0] / r[0]), add(sc(Y, u[1] / r[1]), sc(Z, u[2] / r[2]))));
    b.v(add(c, p), n);
  }
  for (let i = 0; i < rings; i++) for (let k = 0; k < seg; k++) {
    const a = base + i * (seg + 1) + k, c2 = a + seg + 1;
    b.I.push(a, c2, a + 1, a + 1, c2, c2 + 1);
  }
}
// flat octagonal plate (phased-array face) centred at c, facing n
function plate(b, c, n, up, w, h, t, ch = 0.29) {
  n = norm(n);
  const U = norm(cross(up, n)), V = cross(n, U);
  const o = [[-w / 2 + w * ch, -h / 2], [w / 2 - w * ch, -h / 2], [w / 2, -h / 2 + h * ch], [w / 2, h / 2 - h * ch], [w / 2 - w * ch, h / 2], [-w / 2 + w * ch, h / 2], [-w / 2, h / 2 - h * ch], [-w / 2, -h / 2 + h * ch]];
  const P0 = o.map(([a, q]) => add(c, add(sc(U, a), sc(V, q)))), P1 = P0.map((p) => add(p, sc(n, t)));
  for (let i = 0; i < 8; i++) { const j = (i + 1) % 8; face(b, [P0[i], P0[j], P1[j], P1[i]]); }
  face(b, P1); face(b, [...P0].reverse());
}

// ---------------------------------------------------------------- the ship's static geometry
function shipMesh(gl) {
  const b = builder();
  // hull: lofted sections, stations bunched towards the bow where the shape changes fastest
  const NS = 220, NV = 30, cols = 2 * NV + 1;
  const us = (i) => { const t = i / NS; return Math.min(Math.max(t + 0.12 * Math.sin(Math.PI * t), 0), 1); };
  const sect = (u, v, side) => {
    const K = keelH(u), Gd = deckH(u);
    // (sections bunched at the bilge and near the deck edge)
    const vv = v < 0.5 ? 0.5 * Math.pow(2 * v, 1.25) : v;
    const y = K + (Gd - K) * vv;
    return [SXS + u * (SXB - SXS), y, side * halfWidth(u, y)];
  };
  b.set(M.hull);
  grid(b, NS + 1, cols, (i, j) => {
    const u = us(i), v = Math.abs(j - NV) / NV;
    return { p: sect(u, v, j < NV ? -1 : 1), t: [u, v] };
  });
  // transom
  {
    const c = b.v([SXS, (keelH(0) + deckH(0)) / 2, 0], [-1, 0, 0]);
    const first = b.P.length / 3;
    for (let j = 0; j < cols; j++) b.v(sect(0, Math.abs(j - NV) / NV, j < NV ? -1 : 1), [-1, 0, 0]);
    for (let j = 0; j < cols - 1; j++) b.I.push(c, first + j, first + j + 1);
    b.I.push(c, first + cols - 1, first);
  }
  // forecastle bulwark: the hull side carried up 1.1 m over the forward third
  for (const s of [-1, 1]) {
    grid(b, 81, 3, (i, j) => {
      const x = 27 + (SXB - 0.3 - 27) * (i / 80), u = shipU(x);
      const h = 1.1 * Math.pow(Math.min(Math.max((x - 27) / 7, 0), 1), 0.7) * (1 - Math.pow(Math.max((x - 62) / 4, 0), 2) * 0.3);
      const z = halfWidth(u, deckH(u)) * 0.995;
      return { p: [x, deckH(u) - 0.05 + h * (j / 2), s * (z + 0.06 * (j / 2))], t: [u, 1] };
    });
  }
  // deck (non-skid) across the deck edge at every station
  b.set(M.deck);
  grid(b, NS + 1, 11, (i, j) => {
    const u = us(i), Gd = deckH(u), w = halfWidth(u, Gd) * 0.998;
    return { p: [SXS + u * (SXB - SXS), Gd, w * ((j / 10) * 2 - 1)], t: [u, 0] };
  });

  const g = G(0);
  // ------------------------------------------------ superstructure (faceted, with tumblehome)
  b.set(M.house, g + 0.05);
  house(b, { x0: -10, x1: 27, hw: 6.4, y0: g - 0.6, y1: 9.4, chF: 2.2, chA: 0.6, sS: 0.5, sF: 0.9, sA: 0.15 });
  b.set(M.house, 9.4);
  house(b, { x0: 1, x1: 24.6, hw: 5.7, y0: 9.3, y1: 12.4, chF: 2.3, chA: 1.6, sS: 0.45, sF: 0.95, sA: 0.4 });
  b.set(M.bridge, 12.4);
  house(b, { x0: 12, x1: 23.4, hw: 7.75, y0: 12.3, y1: 15.1, chF: 2.6, chA: 0.4, sS: 0.25, sF: 0.6, sA: 0.1 });
  b.set(M.house);
  house(b, { x0: 13.5, x1: 21.6, hw: 5.0, y0: 15.0, y1: 15.55, chF: 1.4, chA: 0.3, sS: 0.15, sF: 0.2, sA: 0.1 });
  // hangar with a vertical aft face for the door
  b.set(M.house, g + 0.05);
  house(b, { x0: -32, x1: -10.5, hw: 6.5, y0: g - 0.6, y1: 12.3, chF: 1.2, chA: 0.5, sS: 0.6, sF: 0.5, sA: 0 });
  b.set(M.door);
  obox(b, [-32.03, 9.0, 0], [0.12, 5.6, 9.0]);
  // phased-array faces on the forward and aft chamfers of the second level
  b.set(M.panel);
  for (const s of [-1, 1]) {
    plate(b, [23.35, 10.95, s * 4.5], [Math.SQRT1_2, 0.28, s * Math.SQRT1_2], [0, 1, 0], 3.0, 3.0, 0.16);
    plate(b, [1.75, 10.95, s * 4.85], [-Math.SQRT1_2, 0.2, s * Math.SQRT1_2], [0, 1, 0], 2.8, 2.8, 0.16);
  }
  // funnels: faceted, raked aft, with exhaust grilles on top
  for (const [x, y0, h, l, hw] of [[-3.5, 9.3, 6.6, 5.8, 2.4], [-16.5, 12.2, 5.2, 5.0, 2.2]]) {
    b.set(M.house);
    house(b, { x0: x - l / 2, x1: x + l / 2, hw, y0, y1: y0 + h, chF: 1.0, chA: 0.6, sS: 0.55, sF: 1.2, sA: -0.9 });
    b.set(M.dark);
    const xt = x - 0.15;
    house(b, { x0: xt - l / 2 + 0.6, x1: xt + l / 2 - 1.1, hw: hw - 0.75, y0: y0 + h - 0.1, y1: y0 + h + 0.45, chF: 0.6, chA: 0.4, sS: 0.05 });
  }
  // ------------------------------------------------ mast (tripod, platforms, yards, domes)
  b.set(M.steel);
  cyl(b, [18.2, 15.4, 0], [16.6, 31.6, 0], 0.62, 0.22, 14);
  for (const s of [-1, 1]) cyl(b, [21.4, 15.4, s * 2.7], [17.65, 24.2, s * 0.25], 0.22, 0.15, 8);
  cyl(b, [14.3, 15.4, 0], [16.95, 23.8, 0], 0.2, 0.14, 8);
  for (const [y, r] of [[21.2, 1.9], [24.0, 1.4], [28.6, 1.0]]) {
    const x = 18.2 - 1.6 * (y - 15.4) / 16.2;
    cyl(b, [x, y - 0.08, 0], [x, y + 0.08, 0], r, r, 16);
  }
  {
    const x = 18.2 - 1.6 * (25.2 - 15.4) / 16.2;
    obox(b, [x, 25.2, 0], [0.32, 0.26, 10.5]);
    for (const s of [-1, 1]) for (const z of [3.2, 5.0]) cyl(b, [x, 25.1, s * z], [x, 23.8, s * z], 0.04, 0.03, 5, false);
    // yards angled up at the ends
    for (const s of [-1, 1]) cyl(b, [x, 25.2, s * 5.2], [x - 0.1, 26.3, s * 6.6], 0.08, 0.05, 6);
  }
  b.set(M.radome);
  sphere(b, [16.55, 32.2, 0], [0.8, 0.8, 0.8]);
  for (let k = 0; k < 4; k++) { const a = k * Math.PI / 2 + Math.PI / 4; sphere(b, [17.2 + Math.cos(a) * 0.75, 29.1, Math.sin(a) * 0.75], [0.26, 0.32, 0.26], 10, 6); }
  for (const s of [-1, 1]) sphere(b, [18.2 - 1.6 * (21.2 - 15.4) / 16.2 + 0.6, 21.7, s * 1.3], [0.42, 0.45, 0.42], 10, 6);
  b.set(M.steel);
  cyl(b, [16.55, 32.9, 0], [16.55, 36.0, 0], 0.05, 0.02, 5, false);
  // aft mast and its satcom domes
  cyl(b, [-20.6, 12.2, 0], [-20.6, 26.2, 0], 0.38, 0.14, 10);
  obox(b, [-20.6, 23.2, 0], [0.25, 0.22, 4.5]);
  b.set(M.radome);
  for (const s of [-1, 1]) { sphere(b, [-24.0, 13.45, s * 4.0], [0.95, 1.05, 0.95]); }
  b.set(M.steel);
  for (const s of [-1, 1]) cyl(b, [-24.0, 12.2, s * 4.0], [-24.0, 12.6, s * 4.0], 0.55, 0.55, 10);
  // ------------------------------------------------ weapons
  // gun: ring mount, faceted turret, mantlet, tapered barrel with muzzle
  const gx = 41, gy = G(41);
  b.set(M.house);
  cyl(b, [gx, gy - 0.4, 0], [gx, gy + 0.35, 0], 2.25, 2.15, 20);
  house(b, { x0: gx - 2.3, x1: gx + 2.6, hw: 1.95, y0: gy + 0.3, y1: gy + 2.15, chF: 1.0, chA: 0.25, sS: 0.42, sF: 1.55, sA: 0.2 });
  const el = 3 * deg, bd = [Math.cos(el), Math.sin(el), 0], b0 = [gx + 1.55, gy + 1.15, 0];
  b.set(M.steel);
  cyl(b, b0, add(b0, sc(bd, 1.6)), 0.34, 0.24, 12);
  cyl(b, add(b0, sc(bd, 1.5)), add(b0, sc(bd, 7.6)), 0.165, 0.12, 12);
  cyl(b, add(b0, sc(bd, 7.4)), add(b0, sc(bd, 8.0)), 0.15, 0.15, 12);
  // vertical launch cells
  b.set(M.cells);
  house(b, { x0: 26.6, x1: 34.0, hw: 3.3, y0: G(30) - 0.5, y1: G(30) + 0.32, sS: 0.05, sF: 0.05, sA: 0.05 });
  // breakwater
  b.set(M.house);
  for (const s of [-1, 1]) {
    const a = [35.0, G(35), s * 0.2], e = [33.3, G(34), s * 4.6];
    const X = norm(sub(e, a));
    obox(b, add(sc(add(a, e), 0.5), [0, 0.5, 0]), [len(sub(e, a)), 1.0, 0.12], X, [0, 1, 0], norm(cross(X, [0, 1, 0])));
  }
  // close-in weapon system on the hangar roof
  b.set(M.steel);
  cyl(b, [-27.6, 12.2, 0], [-27.6, 13.2, 0], 1.15, 1.05, 16);
  cyl(b, [-27.6, 13.2, 0], [-27.6, 14.0, 0], 0.75, 0.75, 14);
  b.set(M.radome);
  sphere(b, [-27.6, 14.75, 0], [0.85, 1.05, 0.85]);
  b.set(M.dark);
  cyl(b, [-27.0, 13.95, 0], [-25.5, 13.95, 0], 0.17, 0.15, 8);
  // ------------------------------------------------ boats, rafts, deck fittings
  for (const s of [-1, 1]) {
    // rigid-hull inflatable in its cradle on the 01 deck
    const c = [-5.5, 10.15, s * 4.45];
    b.set(M.dark);
    sphere(b, add(c, [0, -0.15, 0]), [3.3, 0.42, 1.05], 16, 8);
    b.set(M.rubber);
    for (const z of [-0.85, 0.85]) cyl(b, add(c, [-3.0, 0.15, z]), add(c, [2.4, 0.15, z * 0.85]), 0.3, 0.3, 10);
    sphere(b, add(c, [2.55, 0.15, 0]), [0.55, 0.3, 0.75], 10, 6);
    b.set(M.steel);
    obox(b, add(c, [-0.6, 0.55, 0]), [1.0, 0.7, 0.8]);
    for (const x of [-2.2, 1.8]) cyl(b, [c[0] + x, 9.4, s * 3.3], [c[0] + x, 11.6, s * 3.9], 0.09, 0.09, 6);
    // life-raft canisters racked on the hangar sides
    b.set(M.raft);
    for (let k = 0; k < 5; k++) {
      const x = -29.5 + k * 1.75;
      cyl(b, [x - 0.7, 10.7, s * 6.42], [x + 0.7, 10.7, s * 6.42], 0.34, 0.34, 10);
    }
    b.set(M.steel);
    obox(b, [-26.0, 10.25, s * 6.38], [9.0, 0.12, 0.6]);
  }
  b.set(M.steel);
  // bollards (pairs) and the anchor windlass
  for (const x of [51, 44, -36, -50, -60]) for (const s of [-1, 1]) {
    const z = s * (edgeZ(x) - 0.8);
    for (const dx of [-0.35, 0.35]) cyl(b, [x + dx, G(x) - 0.1, z], [x + dx, G(x) + 0.5, z], 0.17, 0.17, 8);
  }
  for (const s of [-1, 1]) {
    cyl(b, [48.5, G(48.5) - 0.1, s * 1.4], [48.5, G(48.5) + 0.65, s * 1.4], 0.55, 0.5, 14);
    b.set(M.dark);
    obox(b, [50.5, G(50.5) + 0.04, s * 1.75], [3.6, 0.1, 0.35], norm([1, 0, s * 0.18]), [0, 1, 0], norm([-s * 0.18, 0, 1]));
    b.set(M.steel);
  }
  b.solidCount = b.I.length;   // (the height map for ambient occlusion leaves out rails and whips)
  // guard rails along the main deck (stanchions every 1.6 m, two courses)
  for (const s of [-1, 1]) {
    const pts = [];
    for (let x = -31.5; x <= 27.6; x += 1.6) pts.push([x, G(x), s * (edgeZ(x) - 0.12)]);
    for (let k = 0; k < pts.length; k++) {
      const p = pts[k];
      cyl(b, p, add(p, [0, 1.0, 0]), 0.028, 0.028, 5, false);
      if (k) for (const h of [0.5, 1.0]) cyl(b, add(pts[k - 1], [0, h, 0]), add(p, [0, h, 0]), 0.022, 0.022, 5, false);
    }
  }
  // whip antennas
  for (const [x, y, z, lx, lz, l] of [[-8.5, 9.4, 5.5, -0.15, 0.1, 8], [0.5, 9.4, 5.6, 0.1, 0.12, 7], [-31.3, 12.3, 5.4, -0.2, 0.15, 7]]) {
    for (const s of [-1, 1]) cyl(b, [x, y, s * z], [x + lx * l, y + l, s * (z + lz * l)], 0.06, 0.02, 5, false);
  }
  const mesh = b.mesh(gl);
  mesh.solidCount = b.solidCount;
  return mesh;
}

function radarMesh(gl) {
  const b = builder();
  b.set(M.steel);
  cyl(b, [0, -0.7, 0], [0, 0, 0], 0.32, 0.26, 10);
  // a slightly curved slotted-waveguide array, 4.8 m long
  b.set(M.panel);
  grid(b, 17, 2, (i, j) => {
    const z = -2.4 + 4.8 * (i / 16);
    return { p: [0.08 * (z / 2.4) ** 2 + (j ? 0.12 : -0.12), 0.05 + 0.55 * (i % 16 === 0 ? 0 : 1), z], t: [0, 0] };
  });
  obox(b, [0, 0.5, 0], [0.16, 0.9, 4.8]);
  return b.mesh(gl);
}

// foam sheets: bow wave and hull wash along both sides, churned water behind the transom
function foamMesh(gl) {
  const b = builder();
  const BX = 128;
  for (const s of [-1, 1]) {
    b.set(M.foam, 1);
    grid(b, 121, 9, (i, j) => {
      const bx = -2.0 + (BX + 2.0) * Math.pow(i / 120, 1.35), t = j / 8;
      const x = Math.max(STEM_WL - bx, SXS + 0.5), u = shipU(x);
      const hc = 1.7 * Math.exp(-Math.max(bx, 0) / 9) + 0.32 * Math.exp(-Math.max(bx, 0) / 70) + 0.12;
      const W = 1.0 + 5.2 * Math.min(Math.max(bx / 32, 0), 1) * (1 - 0.6 * Math.min(Math.max((bx - 32) / 55, 0), 1));
      const y = -0.45 + (hc + 0.45) * Math.pow(1 - t, 1.7);
      const zh = Math.max(halfWidth(u, Math.min(y, deckH(u) - 0.5)), halfWidth(u, 0), 0);
      return { p: [x, y, s * (zh + 0.07 + t * W)], t: [bx, t] };
    });
  }
  b.set(M.foam, 2);
  grid(b, 33, 15, (i, j) => {
    const d = 36 * Math.pow(i / 32, 1.3), q = (j / 14) * 2 - 1;
    const zm = 6.2 + 0.18 * d;
    return { p: [SXS + 0.3 - d, -0.35 + 0.75 * Math.exp(-d / 9) * (1 - q * q), q * zm], t: [d, Math.abs(q)] };
  });
  return b.mesh(gl);
}

// ---------------------------------------------------------------- shaders
const VS = `
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec2 aUv;
layout(location = 3) in vec2 aMat;
uniform mat4 uViewProj;
uniform mat4 uModel;
uniform mat3 uNrmM;
uniform vec3 uCamPos;
out vec3 vWorld;
out vec3 vN;
out vec2 vUv;
flat out vec2 vMat;
void main() {
  vec3 P = (uModel * vec4(aPos, 1.0)).xyz;
  vec2 dr = P.xz - uCamPos.xz;
  P.y -= dot(dr, dr) / (2.0 * 6360000.0);     // earth curvature, as the sea surface
  vWorld = P;
  vN = uNrmM * aNrm;
  vUv = aUv;
  vMat = aMat;
  gl_Position = uViewProj * vec4(P, 1.0);
}`;

// (in ship-local space: it only changes when the ship turns, pitches or rolls against the light)
const SHADOW_VS = `
layout(location = 0) in vec3 aPos;
uniform mat4 uShadowVP;
void main() { gl_Position = uShadowVP * vec4(aPos, 1.0); }`;
const SHADOW_FS = `
out vec4 o;
void main() { o = vec4(1.0); }`;

const FS = SKY_LOOKUP_GLSL + SHIP_HULL_GLSL + `
in vec3 vWorld;
in vec3 vN;
in vec2 vUv;
flat in vec2 vMat;
out vec4 oColor;
uniform vec3 uCamPos;
uniform sampler2D uEnv;
uniform vec3 uFlash;
uniform float uRain;
uniform float uTime;
uniform mat4 uShipInv;         // world -> ship-local
uniform mat3 uShipRotInv;
uniform float uShipSpeed;
uniform sampler2D uShipShadow; // the ship's own sun shadow (depth)
uniform mat4 uShadowVP;      // ship-local -> shadow clip
uniform vec2 uShadowOn;        // x: on, y: resolution
uniform float uShipDebug;      // 1: ambient diagnostics
uniform sampler2D uShipHeight; // top-down depth of the ship (ship-local), for ambient occlusion
uniform vec4 uHeightRect;      // centre x, z; half-size x, z
uniform vec2 uHeightRange;     // top y, depth range (m)
const float PI = 3.14159265359;
vec2 equirectUv(vec3 d) { return vec2(atan(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI); }
vec3 envL(vec3 d, float lod) { return textureLod(uEnv, equirectUv(normalize(d)), lod).rgb; }
float h1(float x) { return fract(sin(x * 91.3458) * 47453.5453); }
float h2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float sn(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(h2(i), h2(i + vec2(1, 0)), f.x), mix(h2(i + vec2(0, 1)), h2(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p) { return sn(p) * 0.5 + sn(p * 2.03 + 5.1) * 0.3 + sn(p * 4.1 + 1.7) * 0.2; }
// thin feature (a line of width w, in metres) faded to its mean coverage where a pixel is larger
float line(float d, float w, float fp) { float a = 1.0 - smoothstep(w * 0.5, w * 0.5 + fp, abs(d)); return mix(a, w / max(w + fp, 1e-4), smoothstep(w, 4.0 * w, fp)); }

// rust run-off below a source line y0, in columns of width cw along the wall coordinate h
float rustRun(float h, float y, float y0, float cw, float density, float seed) {
  float d = y0 - y;
  if (d < 0.0) return 0.0;
  float c = floor(h / cw);
  if (h1(c + seed) > density) return 0.0;
  float L = 0.5 + 3.5 * h1(c * 1.7 + seed + 3.1);
  float x = fract(h / cw) - 0.5 + 0.08 * sin(d * 1.7 + c * 3.0);
  float w = 0.06 + 0.2 * h1(c + 9.1 + seed);
  float prof = 1.0 - smoothstep(w * 0.25, w, abs(x));
  return prof * exp(-d / L) * smoothstep(0.0, 0.06, d) * (0.55 + 0.45 * sn(vec2(h * 7.0, y * 2.0)));
}
// seven-segment digit in a unit cell (bits: a b c d e f g)
float hbar(vec2 q, float y, float t) { return step(abs(q.y - y), t * 0.5) * step(t * 0.5, q.x) * step(q.x, 1.0 - t * 0.5); }
float vbar(vec2 q, float x, float y0, float y1, float t) { return step(abs(q.x - x), t * 0.5) * step(y0, q.y) * step(q.y, y1); }
float digit(vec2 q, int m) {
  if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) return 0.0;
  float t = 0.19, r = 0.0;
  if ((m & 1) != 0) r += hbar(q, 1.0 - t * 0.5, t);
  if ((m & 2) != 0) r += vbar(q, 1.0 - t * 0.5, 0.5, 1.0, t);
  if ((m & 4) != 0) r += vbar(q, 1.0 - t * 0.5, 0.0, 0.5, t);
  if ((m & 8) != 0) r += hbar(q, t * 0.5, t);
  if ((m & 16) != 0) r += vbar(q, t * 0.5, 0.0, 0.5, t);
  if ((m & 32) != 0) r += vbar(q, t * 0.5, 0.5, 1.0, t);
  if ((m & 64) != 0) r += hbar(q, 0.5, t);
  return min(r, 1.0);
}
float hullNumber(vec2 q) {   // "2 6 1", each digit 2.4 x 3.4 m, 3.3 m pitch
  float i = floor(q.x / 3.3);
  if (i < 0.0 || i > 2.0) return 0.0;
  int m = i < 0.5 ? 0x5B : i < 1.5 ? 0x7D : 0x30;   // ('1' on the left of its cell: even gaps)
  return digit(vec2((q.x - i * 3.3) / 2.4, q.y / 3.4), m);
}
float shipShadow(vec3 P, vec3 Nw, vec3 L) {
  if (uShadowOn.x < 0.5) return 1.0;
  // (normal offset grows at grazing light, where depth texels stretch over the receiver)
  float gr = 1.0 - abs(dot(Nw, L));
  vec4 c = uShadowVP * vec4(P + L * (0.1 + 0.25 * gr) + Nw * (0.06 + 0.6 * gr), 1.0);
  vec3 s = c.xyz * 0.5 + 0.5;
  if (any(lessThan(s.xy, vec2(0.003))) || any(greaterThan(s.xy, vec2(0.997)))) return 1.0;
  // 3x3 bilinear-weighted PCF (a 4x4 texel footprint)
  vec2 t = s.xy * uShadowOn.y - 1.5, f = fract(t);
  ivec2 i0 = ivec2(floor(t));
  float z = s.z - 0.0012;
  float sum = 0.0;
  for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) {
    float wx = x == 0 ? 1.0 - f.x : x == 3 ? f.x : 1.0;
    float wy = y == 0 ? 1.0 - f.y : y == 3 ? f.y : 1.0;
    sum += wx * wy * step(z, texelFetch(uShipShadow, i0 + ivec2(x, y), 0).r);
  }
  return sum / 9.0;
}

float shipH(vec2 xz) {
  vec2 uv = (xz - uHeightRect.xy) / uHeightRect.zw * 0.5 + 0.5;
  if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return -8.0;
  return uHeightRange.x - texture(uShipHeight, uv).r * uHeightRange.y;
}
// ambient occlusion from the ship's height field: horizon angles in eight directions (the outward
// half for walls), cover overhead (under the flare, the bridge wings), and the foot of walls
float shipAO(vec3 lp, vec3 nl) {
  vec3 p = lp + nl * 0.15;
  vec2 n2 = nl.xz;
  float wl = length(n2);
  vec2 nd = n2 / max(wl, 1e-4);
  float wallK = smoothstep(0.3, 0.9, wl);
  float occ = 0.0, ws = 0.0;
  for (int k = 0; k < 8; k++) {
    float a = float(k) * 0.7853982 + 0.3927;
    vec2 d = vec2(cos(a), sin(a));
    float w = mix(1.0, max(dot(d, nd), 0.0), wallK);
    float hz = max(clamp((shipH(p.xz + d * 0.9) - p.y) / 0.9, 0.0, 6.0), clamp((shipH(p.xz + d * 2.6) - p.y) / 2.6, 0.0, 6.0));
    occ += w * hz / sqrt(1.0 + hz * hz);
    ws += w;
  }
  float ao = 1.0 - 0.7 * occ / max(ws, 1e-3);
  if (wallK > 0.0) {
    float h = shipH(p.xz + nd * 0.6);
    ao *= mix(1.0, 0.5, smoothstep(0.2, 1.0, h - p.y) * (1.0 - smoothstep(2.0, 3.5, h - p.y)) * wallK);   // close cover overhead
    ao *= mix(1.0, mix(0.55, 1.0, smoothstep(0.0, 1.3, p.y - h)), wallK * step(-7.0, h)); // foot of the wall
  }
  return clamp(ao, 0.25, 1.0);
}

void main() {
  vec3 P = vWorld;
  vec3 toCam = uCamPos - P;
  float dist = length(toCam);
  vec3 V = toCam / dist;
  vec3 ng = normalize(cross(dFdx(P), dFdy(P)));
  if (dot(ng, V) < 0.0) ng = -ng;
  vec3 N = normalize(vN);
  if (dot(N, ng) < 0.0) N = -N;
  vec3 lp = (uShipInv * vec4(P, 1.0)).xyz;
  vec3 nl = uShipRotInv * N;                  // ship-local normal
  float fp = length(fwidth(lp));              // pixel footprint (m)
  int id = int(vMat.x + 0.5);
  float aux = vMat.y;
  bool wall = abs(nl.y) < 0.55;
  float hc = abs(nl.x) > abs(nl.z) ? lp.z : lp.x;   // horizontal coordinate along a wall

  vec3 alb = vec3(0.21, 0.225, 0.24);
  float rough = 0.55, F0 = 0.04, ao = 1.0, alpha = 1.0;
  float rust = 0.0, dirt = 0.0;
  // paint tone variation: patches touched up at different times, slightly bluer or greyer
  // paint tone: broad soft variation, plus a few rectangular touched-up patches of fresher paint
  float patchN = fbm(vec2(hc, lp.y) * vec2(0.09, 0.22) + lp.z * 0.05);
  vec2 pc = floor(vec2(hc, lp.y) / vec2(4.3, 2.1));
  float touch = 0.5 * step(0.9, h2(pc + 17.0)) * (1.0 - smoothstep(0.5, 1.5, fp));
  vec3 patchT = mix(vec3(0.9, 0.93, 0.98), vec3(1.07, 1.05, 1.0), patchN) * (1.0 + 0.06 * touch);
  if (id == 0) {
    // hull: antifouling red below a black boot-top, waterline staining, plate seams, hull number,
    // hawse pipe and scuppers with rust running down from them
    float G = shDeckH(shipU(lp.x));
    alb = vec3(0.175, 0.185, 0.197) * patchT;
    float wl = lp.y + 0.15 * sn(vec2(lp.x * 0.3, 1.0));
    if (wl < -0.25) { alb = vec3(0.11, 0.03, 0.022) * (0.85 + 0.3 * fbm(lp.xy * vec2(0.4, 1.5))); rough = 0.7; }
    else if (wl < 0.75) { alb = vec3(0.022, 0.023, 0.025); rough = 0.5; }
    else {
      dirt += 0.35 * smoothstep(2.2, 0.8, lp.y) * (0.6 + 0.4 * fbm(vec2(lp.x * 0.5, lp.y * 3.0)));
      // plate seams (welded butts and seams)
      float seam = max(line(fract(lp.y / 2.4 + 0.3) * 2.4 - 1.2, 0.03, fp), line(fract(lp.x / 7.2) * 7.2 - 3.6, 0.03, fp));
      alb *= 1.0 - 0.12 * seam;
      // hull number on both bows, reading forwards, with a thin dark shadow
      float s = lp.z > 0.0 ? lp.x - 38.6 : 49.8 - lp.x;
      vec2 q = vec2(s, lp.y - 3.0);
      float num = hullNumber(q), shd = hullNumber(q + vec2(-0.14, 0.14));
      if (fp < 0.8) { alb = mix(alb, vec3(0.02), max(shd - num, 0.0) * 0.8); alb = mix(alb, vec3(0.72, 0.72, 0.7), num); }
      else alb = mix(alb, vec3(0.5), 0.1 * step(0.0, s) * step(s, 9.0) * step(0.0, q.y) * step(q.y, 3.4));
      // rust: from the deck edge, from scuppers just below it, from the hawse pipe
      float sc = fract((lp.x + 2.0) / 8.5) * 8.5 - 4.25;
      float scup = line(sc, 0.35, fp) * line(lp.y - (G - 0.55), 0.18, fp);
      alb = mix(alb, vec3(0.02), scup);
      rust += rustRun(lp.x, lp.y, G - 0.05, 1.7, 0.1, 1.0) * 0.55 * (0.4 + 0.6 * sn(vec2(lp.x * 0.05, 3.0)));
      rust += rustRun(sc, lp.y, G - 0.62, 0.9, abs(sc) < 0.45 ? 0.55 : 0.0, floor((lp.x + 2.0) / 8.5)) * 0.75;
      vec2 hp = vec2(lp.x - 52.0, lp.y - 6.4);
      float hawse = 1.0 - smoothstep(0.55, 0.6 + fp, length(hp * vec2(0.8, 1.15)));
      alb = mix(alb, vec3(0.015), hawse);
      rust += (1.0 - smoothstep(0.25, 0.7, abs(hp.x + 0.1 * sin(hp.y * 2.0)))) * exp(-max(-hp.y, 0.0) / 3.0) * step(hp.y, 0.0) * 0.9;
    }
    ao = mix(0.8, 1.0, smoothstep(-1.0, 3.0, lp.y));
  } else if (id == 1) {
    // deck: worn non-skid; the flight deck with its circle, edge, centre line and tie-down grid
    alb = vec3(0.075, 0.08, 0.085) * (0.8 + 0.4 * fbm(lp.xz * 0.6));
    alb *= 1.0 + 0.25 * smoothstep(0.55, 0.85, fbm(lp.xz * 0.18 + 3.0));   // scuffed lighter
    rough = 0.88;
    if (lp.x < -32.5) {
      alb = vec3(0.095, 0.1, 0.105) * (0.85 + 0.3 * fbm(lp.xz * 0.5));
      float r = length(lp.xz - vec2(-49.0, 0.0));
      float edge = shBeamDeck(shipU(lp.x)) - abs(lp.z);
      float m = max(max(line(r - 5.6, 0.35, fp), line(edge - 0.8, 0.3, fp)), line(lp.z, 0.25, fp) * step(0.45, fract(lp.x / 3.0)));
      alb = mix(alb, vec3(0.62, 0.62, 0.6) * (0.8 + 0.2 * sn(lp.xz * 2.0)), m);
      vec2 td = fract(lp.xz / 2.2) - 0.5;
      alb *= 1.0 - 0.6 * (1.0 - smoothstep(0.05, 0.08 + fp, length(td * 2.2))) * (1.0 - smoothstep(0.5, 1.5, fp));
      alb = mix(alb, vec3(0.02), smoothstep(0.6, 0.0, edge) * 0.7);   // safety nets at the deck edge
    }
    ao = 0.92;
  } else if (id == 2 || id == 12 || id == 10) {
    // superstructure paint: portholes, rust from them and from the roof edges, grime at the foot
    alb = vec3(0.19, 0.2, 0.213) * patchT;
    if (wall && aux > 0.0) {
      dirt += 0.3 * smoothstep(aux + 0.7, aux, lp.y);
      float yp = aux + 1.55;
      if (id == 2) {
        float cx = floor(hc / 2.6);
        vec2 q = vec2(fract(hc / 2.6) - 0.5, (lp.y - yp) / 2.6) * 2.6;
        bool has = h1(cx * 3.1 + aux) > 0.62 && aux < 9.0;
        float port = has ? 1.0 - smoothstep(0.17, 0.19 + fp, length(q)) : 0.0;
        float rim = has ? line(length(q) - 0.23, 0.06, fp) : 0.0;
        alb *= 1.0 - 0.3 * rim;
        if (port > 0.0) { alb = mix(alb, vec3(0.012, 0.014, 0.018), port); rough = mix(rough, 0.08, port); F0 = mix(F0, 0.06, port); }
        if (has && h1(cx * 7.7 + aux) > 0.45) rust += (1.0 - smoothstep(0.05, 0.22, abs(q.x + 0.03 * sin(lp.y * 3.0)))) * exp(-max(yp - 0.25 - lp.y, 0.0) / 1.2) * step(lp.y, yp - 0.22) * 0.8;
      }
      rust += rustRun(hc, lp.y, aux + 3.0, 1.4, 0.1, aux * 3.0) * 0.45;
      // watertight doors at the foot of the walls: a recessed dark-grey leaf in a raised frame
      if (id == 2) {
        float dc = floor(hc / 9.0);
        vec2 dq = vec2(fract(hc / 9.0) * 9.0 - 4.5, lp.y - aux);
        if (h1(dc * 5.3 + aux * 1.7) > 0.55) {
          float leaf = step(abs(dq.x), 0.42) * step(0.15, dq.y) * step(dq.y, 1.95);
          float frame = max(line(abs(dq.x) - 0.47, 0.08, fp) * step(0.1, dq.y) * step(dq.y, 2.0), line(dq.y - 2.0, 0.08, fp) * step(abs(dq.x), 0.5));
          alb = mix(alb, alb * 0.72, leaf);
          alb *= 1.0 - 0.35 * frame;
          float hand = 1.0 - smoothstep(0.04, 0.06 + fp, length(dq - vec2(0.3, 1.0)));
          alb = mix(alb, vec3(0.03), hand);
          rust += leaf * 0.25 * smoothstep(0.6, 0.15, dq.y) * fbm(dq * 6.0);
        }
      }
      // bridge windows: a band of glass panes with mullions on the front and sides
      if (id == 12 && nl.x > -0.35) {
        float yb = lp.y - (aux + 1.0);
        float band = step(0.0, yb) * step(yb, 1.2);
        float mull = line(fract(hc / 1.15) * 1.15 - 0.575, 0.12, fp);
        float glass = band * (1.0 - mull);
        alb = mix(alb, vec3(0.008, 0.01, 0.012), glass);
        rough = mix(rough, 0.05, glass); F0 = mix(F0, 0.07, glass);
        alb *= 1.0 - 0.4 * line(yb, 0.1, fp) - 0.4 * line(yb - 1.2, 0.1, fp);
      }
    }
    if (id == 10) {
      // hangar door: vertical ribs and panel joints
      alb *= 0.92 + 0.08 * sn(vec2(lp.z * 2.0, lp.y * 0.3));
      alb *= 1.0 - 0.35 * line(fract(lp.z / 0.6) * 0.6 - 0.3, 0.06, fp) - 0.4 * line(fract((lp.y - 6.2) / 1.4) * 1.4 - 0.7, 0.04, fp);
      rust += rustRun(lp.z, lp.y, 11.7, 0.8, 0.25, 7.0) * 0.5;
    }
    if (!wall && nl.y > 0.0) { alb *= 0.85; dirt += 0.25 * fbm(lp.xz * 0.4); }   // roofs: soot and grime
  } else if (id == 3) {
    alb = vec3(0.028, 0.028, 0.03); rough = 0.65;
  } else if (id == 5) {
    alb = vec3(0.58, 0.58, 0.56) * (0.92 + 0.08 * fbm(lp.xy * 1.5)); rough = 0.45;
    dirt += 0.25 * smoothstep(0.3, -0.6, nl.y);
  } else if (id == 6) {
    // launcher cells: a grid of square hatches with dark seams
    alb = vec3(0.15, 0.155, 0.165);
    if (nl.y > 0.7) {
      vec2 c = (lp.xz - vec2(26.6, -3.3)) / 1.23;
      vec2 f = fract(c) - 0.5;
      alb *= 0.9 + 0.2 * h2(floor(c));
      alb *= 1.0 - 0.7 * max(line(f.x * 1.23 - 0.6, 0.07, fp), line(f.y * 1.23 - 0.6, 0.07, fp));
    }
  } else if (id == 7) {
    alb = vec3(0.185, 0.195, 0.208) * patchT; rough = 0.5;
  } else if (id == 8) {
    alb = vec3(0.025, 0.025, 0.027); rough = 0.75;
  } else if (id == 9) {
    alb = vec3(0.6, 0.6, 0.56) * (0.9 + 0.1 * sn(lp.xy * 3.0)); rough = 0.4;
    alb *= 1.0 - 0.5 * line(fract(lp.x / 1.4 + 0.5) * 1.4 - 0.7, 0.08, fp);   // bands
  } else if (id == 13) {
    alb = vec3(0.12, 0.13, 0.14); rough = 0.4;
  } else if (id == 14) {
    // foam sheets: churned white water, streaming aft relative to the ship
    float bx = vUv.x, t = vUv.y;
    float xs = bx + uTime * uShipSpeed;
    // (sampled on the sheet's own surface: streaks along the flow, clumps and holes, a ragged crest)
    vec2 fq = vec2(xs, lp.y * 2.5 + t * 3.0);
    float n = fbm(fq * vec2(0.35, 0.9)) * 0.5 + fbm(fq * vec2(1.3, 2.6) + 4.0) * 0.3 + sn(fq * vec2(4.5, 7.0) + 9.0) * 0.2;
    float dens;
    if (aux < 1.5) dens = pow(1.0 - t, 0.7) * (0.35 + 0.65 * exp(-max(bx, 0.0) / 22.0)) * (1.0 - smoothstep(95.0, 128.0, bx)) * smoothstep(-2.0, -0.5, bx);
    else dens = exp(-bx / 16.0) * (1.0 - smoothstep(0.5, 1.0, t)) * smoothstep(0.0, 0.8, bx + 0.4);
    float spdk = clamp(uShipSpeed / 6.0, 0.0, 1.3);
    dens *= spdk;
    float far = smoothstep(0.15, 0.8, fp);
    alpha = mix(smoothstep(0.42, 0.7, dens * (0.15 + 1.25 * n)), clamp(dens * 0.6, 0.0, 1.0), far);
    if (alpha < 0.01) discard;
    alb = vec3(0.74, 0.77, 0.78) * (0.7 + 0.3 * n); rough = 0.95; F0 = 0.02; ao = mix(0.7, 1.0, n);
    N = normalize(mix(N, vec3(0.0, 1.0, 0.0), 0.6));
  }
  // plating between frames and stiffeners, dished in a few centimetres ("hungry horse"): a normal
  // perturbation that catches grazing light, and a slightly different tone for every plate
  bool plated = wall && (id == 0 || id == 2 || id == 12 || id == 10) && !(id == 0 && lp.y < 0.75);
  if (plated) {
    vec3 t1 = normalize(cross(vec3(0.0, 1.0, 0.0), nl)), t2 = cross(nl, t1);
    float fs = id == 0 ? 2.4 : 1.6, ls = id == 0 ? 1.15 : 0.9;
    vec2 c = vec2(dot(lp, t1) / fs, lp.y / ls);
    vec2 cell = floor(c), f = fract(c);
    // pillow h = -A sin^2(pi fx) sin^2(pi fy): flat and continuous at the welds, dished between them
    float A = (0.004 + 0.012 * h2(cell + 3.7)) * (id == 0 ? 1.0 : 0.45);
    vec2 sx = sin(PI * f), cx2 = sin(2.0 * PI * f);
    float gx = -A * PI / fs * cx2.x * sx.y * sx.y;
    float gy = -A * PI / ls * cx2.y * sx.x * sx.x;
    float res = 1.0 - smoothstep(0.12, 0.45, fp / ls);
    nl = normalize(nl - (gx * t1 + gy * t2) * res);
    N = normalize(transpose(uShipRotInv) * nl);
    alb *= mix(1.0, 0.98 + 0.04 * h2(cell + 11.0), 1.0 - smoothstep(0.3, 1.0, fp / ls));
    // grime and water streaks running down the plating (large, low-frequency, vertical)
    float g = fbm(vec2(hc * 0.55, lp.y * 0.06 + h1(floor(hc * 0.55)) * 3.0));
    dirt += 0.55 * smoothstep(0.5, 0.85, g) * (0.6 + 0.4 * fbm(vec2(hc * 2.5, lp.y * 0.4)));
    dirt += 0.12 * fbm(lp.xy * vec2(0.08, 0.3) + lp.z * 0.08);   // broad stains
  }
  // weathering: rust run-off, grime, rain
  if (fp > 0.6) rust *= 0.6;                  // (unresolved streaks average out)
  alb = mix(alb, vec3(0.15, 0.07, 0.032) * (0.75 + 0.5 * sn(lp.xy * 4.0)), clamp(rust * 1.15, 0.0, 0.8));
  alb *= 1.0 - clamp(dirt, 0.0, 0.8) * 0.5;
  float wet = clamp(uRain * 1.6, 0.0, 1.0);
  alb *= mix(1.0, 0.75, wet * (id == 14 ? 0.0 : 1.0));
  rough = mix(rough, rough * 0.5, wet);
  F0 = mix(F0, 0.05, wet);

  // (not on the hull plating: open to the sky, and its own flared deck edge is no near occluder)
  if (id != 14 && id != 0 && uShipDebug < 1.5) ao *= shipAO(lp, nl);
  vec3 L = uSunDir;
  vec3 sunE = sunIlluminance();
  vec2 suv = (P.xz - L.xz * (max(P.y, 0.0) / max(L.y, 0.02)) - uShaftRect.xy) * uShaftRect.zw;
  float sunVis = mix(uHazeSun, texture(uShaftShadow, suv).r, smoothstep(0.0, 0.04, min(min(suv.x, suv.y), min(1.0 - suv.x, 1.0 - suv.y))));
  sunVis *= shipShadow(lp, nl, uShipRotInv * L);

  vec3 T1 = normalize(cross(N, abs(N.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  vec3 T2 = cross(N, T1);
  // cosine-weighted irradiance: the normal plus a ring of eight directions 55 deg off it (a 5-tap
  // estimate around the normal put ~70% of a wall's weight on the bright horizon band)
  vec3 Eamb = envL(N, 5.0);
  for (int k = 0; k < 8; k++) {
    float a = float(k) * 0.7853982;
    Eamb += envL(N * 0.5736 + (T1 * cos(a) + T2 * sin(a)) * 0.8192, 5.0) * 0.5736;
  }
  Eamb *= PI / (1.0 + 8.0 * 0.5736);
  if (uShipDebug > 0.5 && uShipDebug < 1.5) {
    // ambient irradiance of an up-facing surface from the environment vs the sky irradiance the sea uses
    vec3 U = vec3(0.0, 1.0, 0.0);
    vec3 eu = (envL(U, 5.0) * 0.4 + (envL(U + vec3(1, 0, 0), 5.0) + envL(U - vec3(1, 0, 0), 5.0) + envL(U + vec3(0, 0, 1), 5.0) + envL(U - vec3(0, 0, 1), 5.0)) * 0.15) * PI;
    const vec3 Y = vec3(0.2126, 0.7152, 0.0722);
    oColor = vec4(dot(eu, Y) / max(dot(skyIrradiance(), Y), 1e-12), dot(Eamb, Y) / max(dot(skyIrradiance(), Y), 1e-12), dot(sunE, Y) * max(uSunDir.y, 0.0) / max(dot(skyIrradiance(), Y), 1e-12), 1.0);
    return;
  }

  float nl2 = max(dot(N, L), 0.0), nv = max(dot(N, V), 1e-3);
  vec3 H = normalize(L + V);
  float nh = max(dot(N, H), 0.0);
  float a2 = max(rough * rough * rough * rough, 1e-4);
  float D = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
  float Fh = F0 + (1.0 - F0) * pow(1.0 - max(dot(V, H), 0.0), 5.0);
  float Fv = F0 + (1.0 - F0) * pow(1.0 - nv, 5.0) / (1.0 + 4.0 * rough);
  vec3 Esun = sunE * sunVis * nl2;
  vec3 Ef = uFlash * (0.35 + 0.65 * max(dot(N, uFlashDir), 0.0));
  vec3 col = alb / PI * (Esun + Eamb * ao + Ef) * (1.0 - Fv);
  col += Esun * min(D, 400.0) * Fh / (4.0 * max(nl2, nv));
  vec3 R = reflect(-V, N);
  col += envL(R, mix(1.5, 7.0, rough)) * Fv * ao;

  vec3 aL, aT;
  aerial(-V, dist * 0.001, aL, aT);
  col = col * aT + aL;
  col = withShafts(col, uCamPos, -V, dist);
  col = applyFog(col, -V, dist);
  oColor = vec4(col, alpha);
}`;

// navigation lights: one point sprite each
const LAMP_VS = `
uniform mat4 uViewProj;
uniform vec3 uCamPos;
uniform vec4 uLamp;       // world position, intensity (cd)
uniform mat3 uShipRot;    // ship-local -> world rotation
uniform vec3 uSector;     // visible arc: from, to (rad from ahead), side (-1 port, 1 starboard, 0 both)
uniform float uSize;      // sprite size (px)
out float vVis;
out float vDist;
out vec3 vDir;
void main() {
  vec3 P = uLamp.xyz;
  vec2 dr = P.xz - uCamPos.xz;
  P.y -= dot(dr, dr) / (2.0 * 6360000.0);
  vec3 toCam = uCamPos - P;
  float d = length(toCam);
  vec3 v = transpose(uShipRot) * (toCam / d);
  float th = atan(abs(v.z), v.x);
  float vis = smoothstep(uSector.x - 0.04, uSector.x + 0.04, th) * (1.0 - smoothstep(uSector.y - 0.04, uSector.y + 0.04, th));
  if (uSector.z != 0.0) vis *= smoothstep(-0.02, 0.02, uSector.z * v.z);
  vVis = vis;
  vDist = d;
  vDir = -toCam / d;
  // (pulled towards the camera, out of the fitting it is mounted on)
  gl_Position = uViewProj * vec4(P + toCam / d * 0.8, 1.0);
  gl_PointSize = uSize;
}`;
const LAMP_FS = SKY_LOOKUP_GLSL + `
in float vVis;
in float vDist;
in vec3 vDir;
out vec4 oColor;
uniform vec4 uLamp;
uniform vec3 uLampCol;
uniform float uLampK;      // 1 / illuminance of the light source (lux)
uniform float uPixelAngle;
uniform float uSize;
void main() {
  if (vVis <= 0.0) discard;
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float w = exp(-4.0 * dot(q, q));
  // illuminance at the eye relative to the light source, spread over the sprite's solid angle
  float E = uLamp.w / (vDist * vDist) * uLampK;
  float omega = 0.193 * uSize * uSize * uPixelAngle * uPixelAngle;
  vec3 c = uLampCol * (E / omega) * w * vVis;
  // only the transmittance of the haze and rain between (the light adds, it does not replace)
  vec3 aL, aT;
  aerial(vDir, vDist * 0.001, aL, aT);
  c *= aT;
  c = applyFog(c, vDir, vDist) - applyFog(vec3(0.0), vDir, vDist);
  oColor = vec4(min(max(c, 0.0), vec3(6.0e4)), 1.0);
}`;

// navigation lights: ship-local position, colour, intensity (cd), arc
const LAMPS = [
  { p: [18.25, 22.6, 0], col: [1, 0.93, 0.8], cd: 90, arc: [0, 112.5 * deg, 0] },        // forward masthead
  { p: [-20.15, 26.0, 0], col: [1, 0.93, 0.8], cd: 90, arc: [0, 112.5 * deg, 0] },       // aft masthead (higher)
  { p: [19.5, 13.9, 7.7], col: [0.15, 1, 0.45], cd: 15, arc: [0, 112.5 * deg, 1] },      // starboard: green
  { p: [19.5, 13.9, -7.7], col: [1, 0.07, 0.04], cd: 15, arc: [0, 112.5 * deg, -1] },    // port: red
  { p: [SXS - 0.25, 5.4, 0], col: [1, 0.93, 0.8], cd: 15, arc: [112.5 * deg, 181 * deg, 0] }, // stern
];

const SHADOW_RES = 1536;

export class Ship {
  // anchor: camera base (xz); yaw: camera heading (rad); dist: metres to the course's centre
  constructor(gl, { visible = true, anchor = [0, 0], yaw = 0, dist = 820 } = {}) {
    this.gl = gl;
    this.visible = visible;
    this.prog = new Program(gl, VS, FS, { label: 'ship' });
    this.lampProg = new Program(gl, LAMP_VS, LAMP_FS, { label: 'ship-lamps' });
    this.shadowProg = new Program(gl, SHADOW_VS, SHADOW_FS, { label: 'ship-shadow' });
    this.mesh = shipMesh(gl);
    this.radarMesh = radarMesh(gl);
    this.foamMesh = foamMesh(gl);
    this.probe = new ShipProbe(gl);
    this.initShadow();
    this.initHeight();
    const f = [Math.cos(yaw), Math.sin(yaw)], r = [-Math.sin(yaw), Math.cos(yaw)];
    // a large ellipse in front of the camera, offset to one side: it crosses the view slowly
    const off = 0.15 * dist;
    this.path = { c: [anchor[0] + f[0] * dist + r[0] * off, anchor[1] + f[1] * dist + r[1] * off], f, r, a: 0.45 * dist, b: 0.21 * dist };
    this.phi = -Math.PI / 2 + 0.35;   // (starts off to one side of the view)
    this.speed = 6.2;                  // m/s (12 knots)
    this.radar = 0;
    this.heave = { x: 0, v: 0 }; this.pitch = { x: 0, v: 0 }; this.roll = { x: 0, v: 0 };
    this.settled = false;
    this.pts = new Float32Array(10);
    this.update(0, null, null, 0, false);
  }

  initShadow() {
    const gl = this.gl;
    this.shadowTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT24, SHADOW_RES, SHADOW_RES);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    this.shadowFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.shadowTex, 0);
    gl.drawBuffers([gl.NONE]);
    gl.readBuffer(gl.NONE);
    this.shadowOk = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.shadowOn = 0;
    this.shadowVP = new Float32Array(16);
  }

  // top-down depth of the static ship (ship-local), drawn once: the height field for ambient occlusion
  initHeight() {
    const gl = this.gl, W = 2048, H = 320;
    const cx = (SXS + SXB) / 2, hx = (SXB - SXS) / 2 + 2, cz = 0, hz = 10, top = 40, range = 48;
    this.heightRect = [cx, cz, hx, hz];
    this.heightRange = [top, range];
    this.heightTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.heightTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT24, W, H);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.heightTex, 0);
    gl.drawBuffers([gl.NONE]);
    gl.readBuffer(gl.NONE);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE) {
      // x -> clip x, z -> clip y, depth = (top - y) / range
      const vp = new Float32Array([1 / hx, 0, 0, 0, 0, 0, -2 / range, 0, 0, 1 / hz, 0, 0, -cx / hx, -cz / hz, 2 * top / range - 1, 1]);
      gl.viewport(0, 0, W, H);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LESS);
      gl.depthMask(true);
      gl.clearDepth(1);
      gl.clear(gl.DEPTH_BUFFER_BIT);
      gl.disable(gl.CULL_FACE);
      this.shadowProg.use({ uShadowVP: vp });
      gl.bindVertexArray(this.mesh.vao);
      gl.drawElements(gl.TRIANGLES, this.mesh.solidCount, this.mesh.itype, 0);
      gl.bindVertexArray(null);
      gl.disable(gl.DEPTH_TEST);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fbo);
  }

  pathAt(phi) {
    const { c, f, r, a, b } = this.path;
    const ca = Math.cos(phi), sa = Math.sin(phi);
    return {
      p: [c[0] + r[0] * a * ca + f[0] * b * sa, c[1] + r[1] * a * ca + f[1] * b * sa],
      d: [-r[0] * a * sa + f[0] * b * ca, -r[1] * a * sa + f[1] * b * ca],
    };
  }

  // light: direction to the sun (or moon) for the ship's own shadow
  update(dt, ot, cam, angStep, sync, light) {
    if (!this.visible && ot) return;
    dt = Math.max(dt, 0);
    const cur = this.pathAt(this.phi);
    this.phi += (this.speed * dt) / Math.max(Math.hypot(cur.d[0], cur.d[1]), 1e-3);
    this.radar += dt * (2 * Math.PI / 2.5);
    const { p, d } = this.pathAt(this.phi);
    const fl = Math.hypot(d[0], d[1]) || 1;
    const fwd = [d[0] / fl, 0, d[1] / fl], side = [-fwd[2], 0, fwd[0]];
    this.pos2 = p; this.fwd = fwd; this.side = side;
    if (!ot) return;

    // sea heights at the centre, towards bow and stern, and at both sides
    const LB = 44, BS = 7.0;
    const P = this.pts;
    const put = (i, x, z) => { P[i * 2] = x; P[i * 2 + 1] = z; };
    put(0, p[0], p[1]);
    put(1, p[0] + fwd[0] * LB, p[1] + fwd[2] * LB);
    put(2, p[0] - fwd[0] * LB, p[1] - fwd[2] * LB);
    put(3, p[0] + side[0] * BS, p[1] + side[2] * BS);
    put(4, p[0] - side[0] * BS, p[1] - side[2] * BS);
    // filtered at least as coarsely as the water mesh there, and never finer than ~1/5 of the hull
    const dist = cam ? Math.hypot(p[0] - cam.pos[0], p[1] - cam.pos[2]) : 800;
    const h = this.probe.run(ot, P, Math.max(dist * angStep * 1.5, 26), sync);
    if (h) {
      const heave = 0.2 * (h[0] + h[1] + h[2] + h[3] + h[4]);
      const pitch = Math.atan((h[1] - h[2]) / (2 * LB)) * 0.85;
      const roll = Math.atan((h[3] - h[4]) / (2 * BS)) * 0.5;
      if (!this.settled) {
        this.heave.x = heave; this.pitch.x = pitch; this.roll.x = roll;
        this.settled = true;
      }
      // slow springs: heave ~6 s, pitch ~7 s, a lightly damped ~11 s roll
      const n = Math.max(1, Math.ceil(dt / (1 / 60)));
      const step = dt / n;
      const spring = (s, target, w, z) => {
        for (let i = 0; i < n; i++) { s.v += (w * w * (target - s.x) - 2 * z * w * s.v) * step; s.x += s.v * step; }
      };
      spring(this.heave, heave, 1.05, 0.7);
      spring(this.pitch, Math.max(Math.min(pitch, 5 * deg), -5 * deg), 0.9, 0.55);
      spring(this.roll, Math.max(Math.min(roll, 7 * deg), -7 * deg), 0.57, 0.22);
    }
    if (this.settled && light && cam) this.renderShadow(light, cam);
  }

  frame() {
    const f = this.fwd, s = this.side;
    const X = norm([f[0], Math.tan(this.pitch.x), f[2]]);
    let Z = norm([s[0], Math.tan(this.roll.x), s[2]]);
    const Y = norm(cross(Z, X));
    Z = cross(X, Y);
    return fromBasis([this.pos2[0], this.heave.x, this.pos2[1]], X, Y, Z);
  }

  radarMatrix(B) {
    const a = this.radar, ca = Math.cos(a), sa = Math.sin(a);
    const x = 18.2 - 1.6 * (27.0 - 15.4) / 16.2;
    return mul(B, fromBasis([x, 27.0, 0], [ca, 0, -sa], [0, 1, 0], [sa, 0, ca]));
  }

  // the ship's own shadow: depth from the light, orthographic over the hull, in ship-local space;
  // redrawn only when the light's direction relative to the ship has moved
  renderShadow(L, cam) {
    const gl = this.gl;
    if (!this.shadowOk || L[1] < 0.02) { this.shadowOn = 0; return; }
    const B = this.frame();
    const Ll = norm([B[0] * L[0] + B[1] * L[1] + B[2] * L[2], B[4] * L[0] + B[5] * L[1] + B[6] * L[2], B[8] * L[0] + B[9] * L[1] + B[10] * L[2]]);
    if (this.shadowOn && this.shadowL && dot(Ll, this.shadowL) > Math.cos(0.15 * deg)) return;
    this.shadowL = Ll;
    const c = [0, 10, 0];
    const back = Ll, up0 = Math.abs(back[1]) < 0.99 ? [0, 1, 0] : [1, 0, 0];
    const right = norm(cross(up0, back)), up = cross(back, right);
    const R = 72, D = 110;
    const proj = new Float32Array([1 / R, 0, 0, 0, 0, 1 / R, 0, 0, 0, 0, -1 / D, 0, 0, 0, 0, 1]);
    this.shadowVP = mul(proj, viewFromBasis(c, right, up, back));
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFbo);
    gl.viewport(0, 0, SHADOW_RES, SHADOW_RES);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.disable(gl.CULL_FACE);
    this.shadowProg.use({ uShadowVP: this.shadowVP });
    gl.bindVertexArray(this.mesh.vao);
    gl.drawElements(gl.TRIANGLES, this.mesh.count, this.mesh.itype, 0);
    gl.bindVertexArray(null);
    this.shadowDraws = (this.shadowDraws || 0) + 1;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.disable(gl.DEPTH_TEST);
    this.shadowOn = 1;
  }

  // draw into the bound HDR target (before the water; u: the water's uniforms)
  draw(u) {
    if (!this.visible || !this.settled) return;
    const gl = this.gl;
    const B = this.frame();
    this.B = B;
    const prog = this.prog;
    const Binv = invertRigid(B);
    prog.use({ ...u, uShipInv: Binv, uShipRotInv: rot3(Binv), uShipSpeed: this.speed, uShipShadow: this.shadowTex, uShadowVP: this.shadowVP, uShadowOn: [this.shadowOn, SHADOW_RES], uShipHeight: this.heightTex, uHeightRect: this.heightRect, uHeightRange: this.heightRange });
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
    gl.disable(gl.CULL_FACE);
    const drawMesh = (mesh, Mm) => {
      prog.set('uModel', Mm);
      prog.set('uNrmM', rot3(Mm));
      gl.bindVertexArray(mesh.vao);
      gl.drawElements(gl.TRIANGLES, mesh.count, mesh.itype, 0);
    };
    drawMesh(this.mesh, B);
    drawMesh(this.radarMesh, this.radarMatrix(B));
    // foam sheets: alpha-to-coverage against the multisampled target
    gl.enable(gl.SAMPLE_ALPHA_TO_COVERAGE);
    drawMesh(this.foamMesh, B);
    gl.disable(gl.SAMPLE_ALPHA_TO_COVERAGE);
    gl.bindVertexArray(null);
  }

  // navigation lights, after the water (waves hide them): additive HDR sprites
  drawLights(u, { pixelAngle, night }) {
    if (!this.visible || !this.settled || !this.B) return;
    const gl = this.gl, B = this.B;
    const prog = this.lampProg;
    // the scene is lit at unit illuminance by the sun (~1e5 lx) or the moon (~0.25 lx)
    prog.use({ ...u, uShipRot: rot3(B), uLampK: night ? 1 / 0.25 : 1 / 1e5, uPixelAngle: pixelAngle, uSize: 6 });
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    bindEmpty(gl);
    for (const l of LAMPS) {
      const p = l.p;
      const w = [B[0] * p[0] + B[4] * p[1] + B[8] * p[2] + B[12], B[1] * p[0] + B[5] * p[1] + B[9] * p[2] + B[13], B[2] * p[0] + B[6] * p[1] + B[10] * p[2] + B[14]];
      prog.set('uLamp', [w[0], w[1], w[2], l.cd]);
      prog.set('uLampCol', l.col);
      prog.set('uSector', l.arc);
      gl.drawArrays(gl.POINTS, 0, 1);
    }
    gl.disable(gl.BLEND);
    gl.depthMask(true);
    gl.depthFunc(gl.LESS);
    gl.bindVertexArray(null);
  }

  // uniforms for the sea's wake term (WAKE_GLSL)
  wakeUniforms() {
    if (!this.visible || !this.settled) return { uWakeInfo: [0, 0, 0, 0] };
    const pts = new Float32Array(WAKE_N * 2);
    // the stern's track: walk back along the course from the transom
    let phi = this.phi, s = 0, k = 0, next = -SXS;
    const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    const grow = (x, z, m) => { lo[0] = Math.min(lo[0], x - m); lo[1] = Math.min(lo[1], z - m); hi[0] = Math.max(hi[0], x + m); hi[1] = Math.max(hi[1], z + m); };
    for (let it = 0; it < 4000 && k < WAKE_N; it++) {
      const c = this.pathAt(phi);
      if (s >= next) { pts[k * 2] = c.p[0]; pts[k * 2 + 1] = c.p[1]; grow(c.p[0], c.p[1], 30 + 0.1 * (s + SXS)); k++; next += WAKE_DS; }
      const step = 4;
      phi -= step / Math.max(Math.hypot(c.d[0], c.d[1]), 1e-3);
      s += step;
    }
    const p = this.pos2, f = this.fwd;
    // the bow's Kelvin arms (~0.35 m out per metre aft, fading over a few hundred metres)
    for (const t of [SXB, 0, -200, -450]) grow(p[0] + f[0] * t, p[1] + f[2] * t, Math.min(12 + 0.36 * Math.max(SXB - t, 0), 180));
    return { uWakeShip: [p[0], p[1], f[0], f[2]], uWakeInfo: [1, this.speed, WAKE_DS, 0], uWakeBox: [lo[0], lo[1], hi[0], hi[1]], uWakePts: pts };
  }
}

// sea heights at five points (choppy displacement inverted), read back without stalling (PBO + fence)
const PROBE_FS = `
uniform sampler2D uDisp[3];
uniform float uSize[4];
uniform float uTexel[4];
uniform vec2 uPts[5];
uniform float uFootprint;
out vec4 o;
vec3 disp(vec2 xz) {
  float l = log2(uFootprint);
  vec3 D = textureLod(uDisp[0], xz / uSize[0], max(l - log2(uTexel[0]), 0.0)).xyz;
  D += textureLod(uDisp[1], xz / uSize[1], max(l - log2(uTexel[1]), 0.0)).xyz;
  D += textureLod(uDisp[2], xz / uSize[2], max(l - log2(uTexel[2]), 0.0)).xyz;
  return D;
}
void main() {
  vec2 p = uPts[int(gl_FragCoord.x)];
  vec2 x0 = p;
  for (int k = 0; k < 5; k++) x0 = p - disp(x0).xz;
  o = vec4(disp(x0), 1.0);
}`;
class ShipProbe {
  constructor(gl) {
    this.gl = gl;
    this.prog = new Program(gl, FULLSCREEN_VS, PROBE_FS, { label: 'ship-probe' });
    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, 5, 1);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.pbo = gl.createBuffer();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, 5 * 16, gl.STREAM_READ);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    this.data = new Float32Array(20);
    this.heights = null;
    this.fence = null;
  }
  run(ot, pts, footprint, sync) {
    const gl = this.gl;
    if (this.fence && !sync) {
      if (gl.clientWaitSync(this.fence, 0, 0) === gl.TIMEOUT_EXPIRED) return this.heights;
      gl.deleteSync(this.fence);
      this.fence = null;
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.data);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      this.decode();
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, 5, 1);
    this.prog.use({ uDisp: ot.disp.slice(0, 3), uSize: ot.sizes, uTexel: ot.texels, uPts: pts, uFootprint: footprint });
    bindEmpty(gl);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (sync) {
      gl.readPixels(0, 0, 5, 1, gl.RGBA, gl.FLOAT, this.data);
      this.decode();
    } else {
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
      gl.readPixels(0, 0, 5, 1, gl.RGBA, gl.FLOAT, 0);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      this.fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);
    return this.heights;
  }
  decode() {
    const h = [];
    for (let i = 0; i < 5; i++) h.push(this.data[i * 4 + 1]);
    if (h.every(Number.isFinite)) this.heights = h;
  }
}
