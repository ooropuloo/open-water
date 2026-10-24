// The clinker (lapstrake) rowing boat: hull lines, planking with real lands (each strake lies over the
// one below it, so the hull has a step one plank thick at every land), keel and stem, steam-bent
// ribs, risers, gunwales (inwale, outwale, capping), thwarts with hanging knees, breasthook, quarter
// knees, transom, duckboards, foot stretcher, rowlock chocks; plus the oars and rowlocks.
// Boat-local metres: x towards the bow, y up (0 = design waterline), z to the side.
//
// Material ids (aMat.x) read by the shader: 0 painted planking, 1 varnished wood, 2 weathered bare
// wood, 3 bronze, 4 oar (varnished, painted tip), 5 leather. aMat.y is a per-part seed.
import { Builder, grid, sweep, beam, roundRect, sub3, add3, sc3, dot3, cross3, len3, norm3, perp3, lerp3, clamp, sstep } from './mesh.js';

export const XS = -1.95, XB = 2.15;
export const T = 0.011;                                  // plank thickness
export const hullU = (x) => (x - XS) / (XB - XS);
export function hBeam(u) {
  const t = u - 0.42;
  return t < 0 ? 0.70 - 0.26 * (t / 0.42) ** 2 : 0.70 * Math.pow(Math.max(1 - (t / 0.58) ** 2, 0), 0.65);
}
export const sheer = (u) => 0.36 + 0.22 * Math.pow(u, 2.5) + 0.04 * Math.pow(1 - u, 4);
export const keel = (u) => -0.20 + 0.04 * (1 - u) ** 2 + 0.62 * Math.pow(sstep(0.62, 1, u), 1.6);
// inner (base) moulded surface: v = 0 at the keel, 1 at the gunwale
export function section(u, v, side) {
  const B = hBeam(u), G = sheer(u), K = Math.min(keel(u), G - 0.01);
  const a = v * Math.PI / 2;
  return [XS + u * (XB - XS), K + (G - K) * (1 - Math.cos(a)), side * B * (0.92 + 0.08 * v) * Math.pow(Math.sin(a), 0.75)];
}
// v of the base surface at height y
export function vAt(u, y) {
  const G = sheer(u), K = Math.min(keel(u), G - 0.01);
  return Math.acos(1 - clamp((y - K) / (G - K), 0, 1)) / (Math.PI / 2);
}
export function innerHalfWidth(x, y) {
  const u = hullU(x), v = vAt(u, y);
  return hBeam(u) * (0.92 + 0.08 * v) * Math.pow(Math.sin(v * Math.PI / 2), 0.75);
}
// height of the inside of the bottom at (x, z)
export function bottomY(x, z) {
  const u = hullU(x);
  let lo = 0, hi = 1;
  for (let i = 0; i < 24; i++) { const m = (lo + hi) / 2; if (Math.abs(section(u, m, 1)[2]) < Math.abs(z)) lo = m; else hi = m; }
  return section(u, (lo + hi) / 2, 1)[1];
}
// outward normal of the base surface
export function outNormal(u, v, side) {
  const e = 1e-3;
  const du = sub3(section(Math.min(u + e, 1), v, side), section(Math.max(u - e, 0), v, side));
  const dv = sub3(section(u, Math.min(v + e, 1), side), section(u, Math.max(v - e, 0), side));
  let n = norm3(cross3(du, dv));
  const p = section(u, v, side), K = keel(u), G = sheer(u);
  const toOut = [0, p[1] - (K + (G - K) * 0.9), p[2]];
  if (dot3(n, toOut) < 0) n = sc3(n, -1);
  return n;
}

// GLSL mirror of the hull lines (self-shadowing of the inside, wetness, the wake's waterline)
export const HULL_GLSL = `
const float XS = ${XS.toFixed(3)}, XB = ${XB.toFixed(3)};
float hullU(float x) { return (x - XS) / (XB - XS); }
float hBeam(float u) { float t = u - 0.42; return t < 0.0 ? 0.70 - 0.26 * (t / 0.42) * (t / 0.42) : 0.70 * pow(max(1.0 - (t / 0.58) * (t / 0.58), 0.0), 0.65); }
float hSheer(float u) { return 0.36 + 0.22 * pow(u, 2.5) + 0.04 * pow(1.0 - u, 4.0); }
float hKeel(float u) { float s = smoothstep(0.62, 1.0, u); return -0.20 + 0.04 * (1.0 - u) * (1.0 - u) + 0.62 * pow(s, 1.6); }
float innerHalfWidth(float u, float y) {
  float G = hSheer(u), K = min(hKeel(u), G - 0.01);
  float a = acos(1.0 - clamp((y - K) / (G - K), 0.0, 1.0));
  return hBeam(u) * (0.92 + 0.08 * a / 1.5707963) * pow(sin(a), 0.75);
}
// 1 where a point inside the hull cannot see the light over the gunwale
float hullShadow(vec3 p, vec3 l) {
  float u = hullU(p.x);
  if (u < 0.0 || u > 1.0) return 0.0;
  float G = hSheer(u);
  if (p.y > G - 0.004 || abs(p.z) > innerHalfWidth(u, p.y) + 0.004) return 0.0;  // outside the hull
  if (l.y < 0.02) return 1.0;
  vec3 q = p + l * ((G + 0.02 - p.y) / l.y);
  float u2 = hullU(q.x);
  if (u2 < 0.0 || u2 > 1.0) return 1.0;
  return smoothstep(-0.015, 0.015, abs(q.z) - hBeam(u2));
}`;

// strake boundaries (v): garboard to sheer strake
export const LANDS = [0, 0.17, 0.315, 0.455, 0.59, 0.72, 0.86, 1.0];
const NSTR = LANDS.length - 1;
function strakeOf(v) { let i = 0; while (i < NSTR - 1 && v >= LANDS[i + 1]) i++; return i; }
// outward offset of the outer skin at v (plank i runs 2T at its lower edge to T at its top)
function outerOff(i, v) {
  if (i === 0) return T;
  return T * (1 + (LANDS[i + 1] - v) / (LANDS[i + 1] - LANDS[i]));
}
const innerOff = (i, v) => outerOff(i, v) - T;

const XO = -0.36;                                        // rowlock station
export const ROWLOCK_X = XO;
const STATIC_SIZES = [3, 3, 2, 2];

export function hullMesh(gl) {
  const b = new Builder(STATIC_SIZES);
  const NS = 84;
  const NV = 4;  // segments across each plank
  for (const side of [-1, 1]) {
    for (let i = 0; i < NSTR; i++) {
      const v0 = LANDS[i], v1 = LANDS[i + 1];
      const seed = i + (side > 0 ? 10 : 0);
      // outer face of the plank
      grid(b, NS + 1, NV + 1, (r, c) => {
        const u = r / NS, v = v0 + (v1 - v0) * (c / NV);
        const p = add3(section(u, v, side), sc3(outNormal(u, v, side), outerOff(i, v)));
        return { p, uv: [p[0], c / NV], m: [0, seed] };
      });
      // inner face
      grid(b, NS + 1, NV + 1, (r, c) => {
        const u = r / NS, v = v0 + (v1 - v0) * (c / NV);
        const p = add3(section(u, v, side), sc3(outNormal(u, v, side), innerOff(i, v)));
        return { p, uv: [p[0], c / NV], m: [1, seed] };
      });
      // the plank's lower edge (the land, one plank thick) and its top edge inside
      if (i > 0) {
        grid(b, NS + 1, 2, (r, c) => {
          const u = r / NS, n = outNormal(u, v0, side);
          const p = add3(section(u, v0, side), sc3(n, c ? 2 * T : T));
          return { p, uv: [p[0], -0.05], m: [0, seed] };
        });
        grid(b, NS + 1, 2, (r, c) => {
          const u = r / NS, n = outNormal(u, v0, side);
          const p = add3(section(u, v0, side), sc3(n, c ? T : 0));
          return { p, uv: [p[0], 1.02], m: [1, seed] };
        });
      }
    }
  }
  // transom: a thick board over the stern section (outer face painted, inner varnished)
  {
    const NVt = 40;
    const outline = [];
    for (let k = 0; k <= NVt; k++) {
      const t = k / NVt, side = t < 0.5 ? -1 : 1, v = Math.abs(1 - 2 * t);
      const i = strakeOf(Math.min(v, 0.999));
      const p = add3(section(0, v, side), sc3(outNormal(0.001, v, side), outerOff(i, v) + 0.004));
      outline.push(p);
    }
    for (const [x, nx, id] of [[XS - 0.014, -1, 0], [XS + 0.022, 1, 1]]) {
      const cy = (keel(0) + sheer(0)) / 2;
      const c = b.v([x, cy, 0], [nx, 0, 0], [cy, 0.5], [id, 98]);
      const f = b.n;
      for (const p of outline) b.v([x, p[1], p[2]], [nx, 0, 0], [p[1], p[2]], [id, 98]);
      for (let k = 0; k < outline.length - 1; k++) b.tri(c, f + k, f + k + 1);
      b.tri(c, f + outline.length - 1, f);
    }
    // edge band
    grid(b, outline.length, 2, (k, c) => ({ p: [c ? XS + 0.022 : XS - 0.014, outline[k][1], outline[k][2]], uv: [outline[k][1], 0.5], m: [1, 98] }));
  }
  return b.upload(gl);
}

// everything wooden or metal that does not move
export function fittingsMesh(gl) {
  const b = new Builder(STATIC_SIZES);
  const wood = (seed) => () => [2, seed];
  const varn = (seed) => () => [1, seed];
  // keel and stem: one timber swept along the keel line and up the stem
  {
    const pts = [];
    for (let k = 0; k <= 60; k++) { const u = k / 60; pts.push([XS + u * (XB - XS), keel(u) - 0.004, 0]); }
    const topStem = sheer(1) + 0.035;
    for (let k = 1; k <= 8; k++) pts.push([XB, keel(1) + (topStem - keel(1)) * (k / 8), 0]);
    const path = (t) => {
      const f = t * (pts.length - 1), i = Math.min(Math.floor(f), pts.length - 2), a = f - i;
      const o = lerp3(pts[i], pts[i + 1], a);
      const tg = norm3(sub3(pts[Math.min(i + 2, pts.length - 1)], pts[Math.max(i - 1, 0)]));
      const out = norm3(cross3(tg, [0, 0, 1]));     // down along the keel, forwards up the stem
      return { o: add3(o, sc3(out, 0.018)), A: [0, 0, 1], B: out };
    };
    sweep(b, path, pts.length * 2, roundRect(0.026, 0.03, 0.008), () => [0, 50]);
  }
  // ribs: steam-bent frames inside, gunwale to gunwale over the keel
  for (let x = XS + 0.32; x < XB - 0.45; x += 0.3) {
    const u = hullU(x);
    const path = (t) => {
      const side = t < 0.5 ? -1 : 1, v = Math.min(Math.abs(1 - 2 * t), 0.955);
      const p = section(u, v, side), n = outNormal(u, Math.max(v, 0.02), side);
      const e = 0.01, p2 = section(u, Math.min(v + e, 1), side);
      const tg = norm3(sub3(p2, p));
      const A = norm3(cross3(n, tg));
      return { o: add3(p, sc3(n, -0.009)), A, B: n };
    };
    sweep(b, path, 40, roundRect(0.017, 0.009, 0.004), varn(30 + Math.round(x * 10)));
  }
  // risers: longitudinal stringers the thwarts rest on
  for (const side of [-1, 1]) {
    const yR = 0.085;
    const x0 = -1.62, x1 = 1.62;
    const path = (t) => {
      const x = x0 + (x1 - x0) * t, u = hullU(x), v = vAt(u, yR);
      const p = section(u, v, side), n = outNormal(u, v, side);
      const hn = norm3([0, 0, n[2]]);
      return { o: add3(p, sc3(hn, -0.012)), A: hn, B: [0, 1, 0] };
    };
    sweep(b, path, 50, roundRect(0.012, 0.024, 0.004), varn(60 + side));
  }
  // gunwales: inwale, outwale and capping rail along the sheer
  for (const side of [-1, 1]) {
    const path = (off, upOff) => (t) => {
      const u = 0.004 + t * 0.99, p = section(u, 1, side);
      const n = outNormal(u, 0.985, side), hn = norm3([0, 0, n[2]]);
      const e = 0.003, p2 = section(Math.min(u + e, 1), 1, side), p1 = section(Math.max(u - e, 0), 1, side);
      const tg = norm3(sub3(p2, p1));
      const up = perp3([0, 1, 0], tg), A = norm3(cross3(up, tg));
      const s = Math.sign(dot3(A, hn)) || 1;
      return { o: add3(p, add3(sc3(A, s * off), sc3(up, upOff))), A: sc3(A, s), B: up };
    };
    sweep(b, path(0.027, -0.022), 90, roundRect(0.014, 0.024, 0.006), varn(70 + side));     // outwale (rubbing strake)
    sweep(b, path(-0.016, -0.02), 90, roundRect(0.013, 0.022, 0.004), varn(72 + side));     // inwale
    sweep(b, path(0.006, 0.008), 90, roundRect(0.038, 0.008, 0.006), wood(74 + side));      // capping
  }
  // thwarts (rowing thwart, forward thwart, stern sheets) with hanging knees
  const thwart = (x, w, seed, knees) => {
    const y = 0.12, half = innerHalfWidth(x, y) - 0.004;
    beam(b, [x, y, -half], [x, y, half], w, 0.026, [1, 0, 0], 0.008, wood(seed), 2);
    if (!knees) return;
    for (const side of [-1, 1]) {
      // knee: horizontal arm under the thwart, vertical arm down the hull side, curved throat
      const yT = y - 0.013, hw = innerHalfWidth(x, yT) - 0.004;
      const prof = [];
      const N = 10;
      for (let k = 0; k <= N; k++) {
        const a = (k / N) * Math.PI / 2;
        const zz = side * (hw - 0.13 + 0.13 * (1 - Math.cos(a)) * 0.0 + 0.11 * (1 - Math.sin(a)));
        prof.push([zz, yT - 0.11 + 0.11 * (1 - Math.cos(a))]);
      }
      const pts = [[side * (hw - 0.14), yT], ...prof.map(([zz, yy]) => [zz, yy]).reverse(), [side * innerHalfWidth(x, yT - 0.15), yT - 0.15], [side * hw, yT]];
      // extrude along x
      const th = 0.022;
      for (const dx of [-th / 2, th / 2]) {
        const c = b.v([x + dx, yT - 0.04, side * (hw - 0.03)], [Math.sign(dx), 0, 0], [0, 0], [2, seed + 5]);
        const f = b.n;
        for (const [zz, yy] of pts) b.v([x + dx, yy, zz], [Math.sign(dx), 0, 0], [zz, yy], [2, seed + 5]);
        for (let k = 0; k < pts.length - 1; k++) b.tri(c, f + k, f + k + 1);
      }
      grid(b, pts.length, 2, (k, c) => ({ p: [x + (c ? th / 2 : -th / 2), pts[k][1], pts[k][0]], uv: [pts[k][0], pts[k][1]], m: [2, seed + 5] }));
    }
  };
  thwart(0.04, 0.24, 80, true);
  thwart(1.22, 0.2, 82, false);
  thwart(-1.42, 0.32, 84, false);
  // stern sheets side benches
  for (const side of [-1, 1]) {
    const y = 0.12;
    const zIn = (x) => side * (innerHalfWidth(x, y) - 0.17);
    const zOut = (x) => side * (innerHalfWidth(x, y) - 0.004);
    const path = (t) => { const x = -1.27 + t * 0.62; return { o: [x, y, (zIn(x) + zOut(x)) / 2], A: [0, 0, 1], B: [0, 1, 0] }; };
    sweep(b, path, 6, roundRect(0.085, 0.013, 0.006), wood(86 + side));
  }
  // breasthook and quarter knees (horizontal knees at gunwale level)
  const plate = (pts, y, th, m) => {
    for (const [dy, ny] of [[th / 2, 1], [-th / 2, -1]]) {
      const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length, cz = pts.reduce((s, p) => s + p[1], 0) / pts.length;
      const c = b.v([cx, y + dy, cz], [0, ny, 0], [cx, cz], m);
      const f = b.n;
      for (const [px, pz] of pts) b.v([px, y + dy, pz], [0, ny, 0], [px, pz], m);
      for (let k = 0; k < pts.length; k++) b.tri(c, f + k, f + ((k + 1) % pts.length));
    }
    grid(b, pts.length + 1, 2, (k, c) => { const p = pts[k % pts.length]; return { p: [p[0], y + (c ? th / 2 : -th / 2), p[1]], uv: [p[0], p[1]], m }; });
  };
  {
    const pts = [];
    const xs = XB - 0.42, yb = sheer(1) - 0.035;
    for (let k = 0; k <= 10; k++) { const x = xs + (XB - 0.04 - xs) * (k / 10); pts.push([x, innerHalfWidth(x, yb) - 0.003]); }
    for (let k = 10; k >= 0; k--) { const x = xs + (XB - 0.04 - xs) * (k / 10); if (k < 10) pts.push([x - 0.0 + (k === 0 ? 0 : 0), -(innerHalfWidth(x, yb) - 0.003)]); }
    // concave aft edge
    const aft = [];
    for (let k = 1; k < 8; k++) { const t = k / 8, z = (1 - 2 * t) * -(innerHalfWidth(xs, yb) - 0.003); aft.push([xs + 0.09 * Math.sin(t * Math.PI), z]); }
    plate([...pts, ...aft], yb, 0.026, [2, 90]);
  }
  for (const side of [-1, 1]) {
    const yq = sheer(0) - 0.035, hw = innerHalfWidth(XS + 0.03, yq) - 0.003;
    const pts = [[XS + 0.024, side * hw], [XS + 0.024, side * (hw - 0.2)]];
    for (let k = 1; k <= 8; k++) { const t = k / 8; pts.push([XS + 0.024 + 0.22 * (1 - Math.cos(t * Math.PI / 2)) * 0.9 + 0.02 * t, side * (hw - 0.2 + 0.2 * Math.sin(t * Math.PI / 2))]); }
    pts.push([XS + 0.26, side * (innerHalfWidth(XS + 0.26, yq) - 0.003)]);
    plate(side > 0 ? pts : pts.slice().reverse(), yq, 0.026, [2, 92 + side]);
  }
  // duckboards: slats on two battens over the bilge
  {
    const slats = [-0.22, -0.11, 0, 0.11, 0.22];
    for (const z of slats) {
      const x0 = -1.18, x1 = 0.86;
      const N = 24;
      const pts = [];
      for (let k = 0; k <= N; k++) {
        const x = x0 + (x1 - x0) * (k / N);
        pts.push([x, bottomY(x, Math.abs(z) + 0.04) + 0.045, z]);
      }
      const path = (t) => {
        const f = t * N, i = Math.min(Math.floor(f), N - 1), a = f - i;
        const tg = norm3(sub3(pts[i + 1], pts[i]));
        return { o: lerp3(pts[i], pts[i + 1], a), A: [0, 0, 1], B: norm3(cross3([0, 0, 1], tg)) };
      };
      sweep(b, path, N, roundRect(0.038, 0.008, 0.003), wood(100 + Math.round(z * 50)));
    }
    for (const x of [-0.85, 0.5]) {
      const y = bottomY(x, 0.26) + 0.028;
      beam(b, [x, y, -0.27], [x, y, 0.27], 0.05, 0.022, [1, 0, 0], 0.004, wood(110), 1);
    }
  }
  // foot stretcher: an inclined board on cleats, the rower's feet press against its after face
  {
    const xs = -0.5, y0 = bottomY(xs, 0.2) + 0.06;      // (rower.js STRETCH_X, STRETCH_A)
    const a = 42 * Math.PI / 180;                        // leans towards the stern at the top
    const c = add3([xs, y0, 0], sc3([-Math.cos(a), Math.sin(a), 0], 0.06));
    beam(b, sub3(c, [0, 0, 0.26]), add3(c, [0, 0, 0.26]), 0.13, 0.022, [-Math.cos(a), Math.sin(a), 0], 0.006, varn(120), 1);
    for (const z of [-0.27, 0.27]) beam(b, [xs + 0.06, y0 - 0.03, z], [xs - 0.14, y0 - 0.03, z], 0.05, 0.03, [0, 1, 0], 0.006, wood(122), 1);
  }
  // rowlock chocks (pads on the gunwale) with bronze socket plates
  for (const side of [-1, 1]) {
    const u = hullU(XO), p = section(u, 1, side);
    const y = p[1] + 0.025;
    beam(b, [XO - 0.11, y, p[2] + side * 0.006], [XO + 0.11, y, p[2] + side * 0.006], 0.07, 0.022, [0, 0, 1], 0.01, wood(130 + side), 1);
    beam(b, [XO - 0.035, y + 0.0115, p[2] + side * 0.006], [XO + 0.035, y + 0.0115, p[2] + side * 0.006], 0.05, 0.003, [0, 0, 1], 0.002, () => [3, 0], 1);
  }
  return b.upload(gl);
}

// rowlock pivot (centre of the horn) for a side
export function rowlockPivot(side) {
  const u = hullU(XO), p = section(u, 1, side);
  return [XO, p[1] + 0.037 + 0.045, p[2] + side * 0.006];
}

// rowlock: a bronze U (horns) on a shank, in its own frame: origin at the centre of the U,
// y up, z along the oar (horizontal), the horns open upwards
export function rowlockMesh(gl) {
  const b = new Builder(STATIC_SIZES);
  const R = 0.033, r = 0.0075;
  const path = (t) => {
    const a = Math.PI * (1.08 + 0.84 * t);   // from one horn tip round the bottom to the other
    const o = [R * Math.cos(a), R * Math.sin(a), 0];
    const tg = [-Math.sin(a), Math.cos(a), 0];
    return { o, A: [0, 0, 1], B: norm3(cross3(tg, [0, 0, 1])) };
  };
  const circ = [];
  for (let k = 0; k < 10; k++) { const a = (k / 10) * 2 * Math.PI; circ.push([r * Math.cos(a) * 1.25, r * Math.sin(a)]); }
  sweep(b, (t) => { const f = path(t * 0.5 + 0.0); return f; }, 12, circ, () => [3, 1]);
  sweep(b, (t) => path(0.5 + t * 0.5), 12, circ, () => [3, 1]);
  // horn tips folded outwards
  for (const sgn of [-1, 1]) {
    const a = sgn < 0 ? Math.PI * 1.08 : Math.PI * 1.92;
    const p0 = [R * Math.cos(a), R * Math.sin(a), 0];
    sweep(b, (t) => ({ o: add3(p0, [sgn * 0.004 * t, 0.012 * t, 0]), A: [0, 0, 1], B: [1, 0, 0] }), 2, circ, () => [3, 1]);
  }
  // neck and shank
  sweep(b, (t) => ({ o: [0, -R - 0.004 - 0.1 * t, 0], A: [1, 0, 0], B: [0, 0, 1] }), 4, circ.map(([x, y]) => [x * 0.9, y * 1.2]), () => [3, 1]);
  return b.upload(gl);
}

// oar along +y (oar frame): origin at the rowlock pivot, handle end at -LIN, blade tip at +LOUT.
// x: the blade's face normal, z: the blade's width.
export const LIN = 0.66, LOUT = 1.62, BLADE = 0.52;
export function oarMesh(gl) {
  const b = new Builder(STATIC_SIZES);
  const N = 120, M = 20;
  const s0 = -LIN, s1 = LOUT;
  const bladeStart = LOUT - BLADE;
  const radius = (s) => {
    if (s < -LIN + 0.14) return 0.0165 + 0.0012 * Math.sin(Math.PI * (s + LIN) / 0.14);    // grip
    if (s < -LIN + 0.2) return 0.0165 + (0.0235 - 0.0165) * sstep(-LIN + 0.14, -LIN + 0.2, s);
    if (s < 0.12) return 0.0235 + 0.0025 * sstep(-0.5, 0.0, s);                               // loom
    return 0.026 - 0.0075 * sstep(0.12, bladeStart, s);                                        // shaft taper
  };
  const sections = [];
  for (let i = 0; i <= N; i++) {
    // denser at the ends and through the blade
    const t = i / N;
    const s = s0 + (s1 - s0) * t;
    sections.push(s);
  }
  const ring = [];
  grid(b, N + 1, M + 1, (i, j) => {
    const s = sections[i], a = (j / M) * 2 * Math.PI;
    const ca = Math.cos(a), sa = Math.sin(a);
    let x, z, y = s;
    const r = radius(s);
    // end caps: round off the handle end and the blade tip
    if (s >= bladeStart) {
      const q = (s - bladeStart) / BLADE;                       // 0 at the throat, 1 at the tip
      let w = 0.019 + (0.074 - 0.019) * sstep(0, 0.42, q);      // half width
      const tip = sstep(0.86, 1.0, q);
      w *= Math.sqrt(Math.max(1 - tip * tip * 0.85, 0.05));
      const th = 0.017 - 0.009 * sstep(0, 0.5, q) - 0.004 * sstep(0.85, 1, q);   // half thickness of the spine
      const edge = 0.0022;
      const zz = Math.sign(ca) * Math.pow(Math.abs(ca), 0.55) * w;
      const prof = 1 - (zz / w) * (zz / w);
      x = sa * (edge + (th - edge) * Math.pow(Math.max(prof, 0), 1.2)) * (q < 0.02 ? 1 : 1);
      // spoon: the power face is cupped along and across
      x += 0.008 * prof * sstep(0, 0.4, q) - 0.012 * (q * q);
      z = zz;
      // blend the throat from the round shaft
      const blend = sstep(0, 0.12, q);
      x = (r * sa) * (1 - blend) + x * blend;
      z = (r * ca) * (1 - blend) + z * blend;
    } else {
      x = r * sa; z = r * ca;
      if (s < -LIN + 0.012) { const e = (s + LIN) / 0.012; const k = Math.sqrt(Math.max(1 - (1 - e) * (1 - e), 0)); x *= k; z *= k; y = s; }
    }
    if (i === N) { x *= 0.15; z *= 0.15; }
    // region id: 0 grip, 1 loom, 2 leather, 3 shaft, 4 blade
    const region = s < -LIN + 0.15 ? 0 : (s > -0.14 && s < 0.1) ? 2 : s < bladeStart ? 3 : 4;
    ring.push(region);
    return { p: [x, y, z], uv: [s, j / M], m: [4, region] };
  });
  // leather button (the collar that stops the oar slipping through the rowlock)
  sweep(b, (t) => ({ o: [0, -0.17 + 0.022 * t, 0], A: [1, 0, 0], B: [0, 0, 1] }), 2,
    Array.from({ length: 16 }, (_, k) => { const a = (k / 16) * 2 * Math.PI; return [0.034 * Math.cos(a), 0.034 * Math.sin(a)]; }), () => [5, 0]);
  return b.upload(gl);
}

// depth-only lid over the hull opening, just below the gunwale (keeps the sea surface out of the hull)
export function lidMesh(gl) {
  const b = new Builder(STATIC_SIZES);
  const NS = 60;
  grid(b, NS + 1, 2, (i, j) => {
    const u = i / NS, p = section(u, 1, j ? 1 : -1);
    return { p: [p[0], p[1] - 0.045, p[2]], uv: [0, 0], m: [0, 0] };
  });
  return b.upload(gl);
}

// bilge water: a grid over the bottom; the shader puts it on a level plane and clips it to the hull
export function bilgeMesh(gl) {
  const b = new Builder(STATIC_SIZES);
  grid(b, 40, 12, (i, j) => ({ p: [-1.6 + 3.3 * (i / 39), 0, -0.4 + 0.8 * (j / 11)], uv: [0, 0], m: [7, 0] }));
  return b.upload(gl);
}
