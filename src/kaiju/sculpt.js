// kaiju: procedural sculpt of a ~100 m (standing height) saurian. No models, no textures.
//
// Rest frame (metres): x along the body (snout at +x, tail tip at -x, hips at 0), y up (dorsal),
// z to the side. The skeleton is the straight spine y = z = 0; every vertex is skinned to the spine by
// its rest x (see kaiju.js).
//
// Body: one signed distance field (a variable elliptical section swept along the spine, smooth-unioned
// with shoulders, arms, thighs, legs, a skull with brow ridges, cheeks, muzzle and a hinged lower jaw,
// eye sockets, nostrils and a carved mouth), with skin folds and lumps displaced into it, polygonised
// with surface nets and projected onto the surface. Dorsal plates are separate thick bony leaves with
// an irregular maple-leaf outline (three rows); teeth are curved cones along the lips.
// Usable from a worker (sculpt-worker.js) or synchronously.

export const REST = { X0: -133, X1: 76, NB: 64 };   // spine parameter range and bone count
export const JAW = { hx: 57.6, hy: -1.2 };           // jaw hinge (rest)

const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const mix = (a, b, t) => a + (b - a) * t;

// ------------------------------------------------------------------ profiles of the swept trunk
// half-width W, height of the back T and depth of the belly B above / below the spine
const PX = [-133, -120, -105, -90, -75, -60, -45, -30, -15, 0, 10, 20, 30, 38, 46, 52, 58];
const PW = [0.30, 1.15, 2.15, 3.05, 3.95, 4.85, 5.85, 7.05, 8.75, 10.6, 12.0, 12.6, 11.9, 10.4, 7.7, 6.3, 5.3];
const PT = [0.35, 1.35, 2.55, 3.55, 4.55, 5.55, 6.55, 7.75, 9.15, 10.3, 11.3, 11.9, 11.7, 10.8, 8.5, 6.7, 5.2];
const PB = [0.30, 1.15, 2.15, 3.05, 3.95, 4.85, 5.85, 6.95, 8.55, 10.6, 12.8, 14.0, 13.0, 11.3, 8.3, 6.7, 5.6];
const LUT_DX = 0.25, LUT_X0 = PX[0], LUT_N = Math.ceil((PX[PX.length - 1] - PX[0]) / LUT_DX) + 2;
function buildLut(ys) {
  const n = PX.length, out = new Float32Array(LUT_N);
  const tan = ys.map((_, i) => {
    const a = Math.max(i - 1, 0), b = Math.min(i + 1, n - 1);
    return (ys[b] - ys[a]) / (PX[b] - PX[a]);
  });
  for (let k = 0; k < LUT_N; k++) {
    const x = Math.min(LUT_X0 + k * LUT_DX, PX[n - 1]);
    let i = 0;
    while (i < n - 2 && x > PX[i + 1]) i++;
    const h = PX[i + 1] - PX[i], t = (x - PX[i]) / h;
    const t2 = t * t, t3 = t2 * t;
    out[k] = (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * tan[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * tan[i + 1];
  }
  return out;
}
const LW = buildLut(PW), LT = buildLut(PT), LB = buildLut(PB);
function lut(L, x) {
  const f = clamp((x - LUT_X0) / LUT_DX, 0, LUT_N - 1.001);
  const i = f | 0, t = f - i;
  return L[i] + (L[i + 1] - L[i]) * t;
}
export const profW = (x) => lut(LW, x);
export const profT = (x) => lut(LT, x);
export const profB = (x) => lut(LB, x);

// ------------------------------------------------------------------ SDF primitives
function smin(a, b, k) { const h = Math.max(k - Math.abs(a - b), 0) / k; return Math.min(a, b) - h * h * k * 0.25; }
function smax(a, b, k) { return -smin(-a, -b, k); }
function sdEll(px, py, pz, cx, cy, cz, rx, ry, rz) {
  const x = (px - cx) / rx, y = (py - cy) / ry, z = (pz - cz) / rz;
  const k0 = Math.sqrt(x * x + y * y + z * z);
  const k1 = Math.sqrt((x / rx) * (x / rx) + (y / ry) * (y / ry) + (z / rz) * (z / rz));
  return k1 > 1e-9 ? (k0 * (k0 - 1)) / k1 : -Math.min(rx, ry, rz);
}
// round cone (Quilez), a -> b, radii r1 -> r2
function sdRC(px, py, pz, ax, ay, az, bx, by, bz, r1, r2) {
  const bax = bx - ax, bay = by - ay, baz = bz - az;
  const l2 = bax * bax + bay * bay + baz * baz, rr = r1 - r2, a2 = l2 - rr * rr, il2 = 1 / l2;
  const pax = px - ax, pay = py - ay, paz = pz - az;
  const y = pax * bax + pay * bay + paz * baz, z = y - l2;
  const qx = pax * l2 - bax * y, qy = pay * l2 - bay * y, qz = paz * l2 - baz * y;
  const x2 = qx * qx + qy * qy + qz * qz, y2 = y * y * l2, z2 = z * z * l2;
  const k = Math.sign(rr) * rr * rr * x2;
  if (Math.sign(z) * a2 * z2 > k) return Math.sqrt(x2 + z2) * il2 - r2;
  if (Math.sign(y) * a2 * y2 < k) return Math.sqrt(x2 + y2) * il2 - r1;
  return (Math.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

// ------------------------------------------------------------------ noise (value noise, 3D)
function hash3(i, j, k) {
  let h = (Math.imul(i, 374761393) + Math.imul(j, 668265263) + Math.imul(k, 1274126177)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1103515245);
  h ^= h >>> 16;
  return (h & 0xffffff) / 0x1000000;
}
function vnoise(x, y, z) {
  const i = Math.floor(x), j = Math.floor(y), k = Math.floor(z);
  let fx = x - i, fy = y - j, fz = z - k;
  fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
  const a = hash3(i, j, k), b = hash3(i + 1, j, k), c = hash3(i, j + 1, k), d = hash3(i + 1, j + 1, k);
  const e = hash3(i, j, k + 1), f = hash3(i + 1, j, k + 1), g = hash3(i, j + 1, k + 1), h = hash3(i + 1, j + 1, k + 1);
  const x1 = a + (b - a) * fx, x2 = c + (d - c) * fx, x3 = e + (f - e) * fx, x4 = g + (h - g) * fx;
  const y1 = x1 + (x2 - x1) * fy, y2 = x3 + (x4 - x3) * fy;
  return y1 + (y2 - y1) * fz;
}
function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
}

// ------------------------------------------------------------------ head geometry helpers
// lip line (centre of the mouth slot) and the half gap of the slot
export const lipY = (x) => -1.05 + 0.012 * (x - 62) - 0.55 * sstep(61.8, 58.4, x);
const GAP = 0.2, SLOT_X = 58.4;

// the head (x > ~52), with or without the carved mouth
function sdHead(x, y, z, az, slot) {
  // skull and neck-to-skull blend
  let d = sdEll(x, y, z, 60.8, 1.7, 0, 6.8, 4.7, 5.3);
  // muzzle (upper jaw): short, broad and blunt
  d = smin(d, sdRC(x, y, z / 1.18, 61.4, 1.1, 0, 68.9, 0.25, 0, 4.25, 2.7), 2.0);
  // deep lower jaw, hinged under the cheeks
  d = smin(d, sdRC(x, y, z / 1.22, 56.8, -2.7, 0, 68.0, -2.55, 0, 3.75, 2.3), 1.6);
  // jaw muscles bulging behind the mouth corners
  d = smin(d, sdEll(x, y, az, 58.2, -0.4, 4.2, 3.6, 3.2, 2.0), 1.6);
  // heavy, overhanging brow ridges, and a medial crest running back onto the neck
  d = smin(d, sdRC(x, y, az, 60.2, 5.0, 3.05, 66.3, 3.75, 3.05, 1.65, 1.05), 0.8);
  d = smin(d, sdRC(x, y, z, 63.0, 5.6, 0, 54.0, 6.7, 0, 0.8, 1.15), 1.6);
  // a chin boss and the throat
  d = smin(d, sdEll(x, y, z, 64.4, -4.3, 0, 4.2, 1.7, 2.5), 1.4);
  // eye sockets (deep, under the brows) and nostrils
  d = smax(d, -sdEll(x, y, az, 65.0, 2.55, 4.15, 0.95, 0.8, 0.9), 0.3);
  d = smax(d, -sdEll(x, y, az, 70.4, 2.05, 1.05, 0.5, 0.36, 0.36), 0.2);
  if (slot) {
    // mouth: a slot from the snout back to the corners under the cheeks
    const sl = Math.abs(y - lipY(x)) - GAP;
    const back = SLOT_X - x;
    const slotD = Math.max(sl, back) < 0 ? Math.max(sl, back) : Math.hypot(Math.max(sl, 0), Math.max(back, 0));
    d = smax(d, -slotD, 0.18);
  }
  // eyeballs (recessed)
  d = Math.min(d, sdEll(x, y, az, 64.85, 2.5, 3.45, 0.76, 0.7, 0.76));
  return d;
}

// ------------------------------------------------------------------ the body SDF
function sdTrunk(x, y, z) {
  const XT0 = -133, XT1 = 58;
  const xc = clamp(x, XT0, XT1);
  const W = lut(LW, xc), T = lut(LT, xc), B = lut(LB, xc);
  const ry = y > 0 ? T : B;
  const a = y / ry, b = z / W;
  const q = Math.sqrt(a * a + b * b);
  const g = Math.sqrt((a / ry) * (a / ry) + (b / W) * (b / W));
  let d = q > 1e-6 ? ((q - 1) * q) / Math.max(g, 1e-9) : -Math.min(ry, W);
  const dx = x < XT0 ? XT0 - x : x > XT1 ? x - XT1 : 0;
  if (dx > 0) d = Math.hypot(Math.max(d, 0), dx) + Math.min(d, 0);
  // dorsal crest where the plates grow
  if (x > -124 && x < 56) {
    const fade = sstep(-124, -100, x) * sstep(56, 46, x);
    const cy = T - 0.95, rz = (0.45 + 0.11 * W) * (0.4 + 0.6 * fade), rr = 1.75 * (0.5 + 0.5 * fade);
    const u = (y - cy) / rr, v = z / rz;
    const qq = Math.sqrt(u * u + v * v), gg = Math.sqrt((u / rr) * (u / rr) + (v / rz) * (v / rz));
    d = smin(d, ((qq - 1) * qq) / Math.max(gg, 1e-9), 1.3);
  }
  return d;
}

// detail: lumps, folds and rings (applied only near the surface)
function detail(x, y, z, d) {
  let n = (vnoise(x * 0.42, y * 0.42, z * 0.42) - 0.5) * 0.55 + (vnoise(x * 1.05 + 7.1, y * 1.05, z * 1.05) - 0.5) * 0.26;
  // lumpy, plated back; smoother on the snout and around the eyes
  const head = sstep(58, 64, x);
  n *= 1 - 0.55 * head;
  let f = 0;
  // neck and throat folds
  if (x > 38 && x < 58) {
    const m = sstep(38, 43, x) * sstep(58, 54, x);
    f += 0.24 * m * Math.pow(0.5 + 0.5 * Math.cos(((x - 38) * 2 * Math.PI) / 2.3 + 0.35 * Math.sin(z * 0.5 + y * 0.3)), 3);
  }
  // segmented tail
  if (x < -12) f += 0.17 * sstep(-12, -22, x) * Math.pow(0.5 + 0.5 * Math.cos((x * 2 * Math.PI) / 3.4 + 0.25 * Math.sin(y * 0.6)), 2.5);
  // transverse belly plates
  if (y < 0 && x > -20 && x < 56) {
    const m = sstep(0, -6, y) * sstep(-20, -10, x) * sstep(56, 50, x);
    f += 0.13 * m * Math.pow(0.5 + 0.5 * Math.cos((x * 2 * Math.PI) / 1.9), 6);
  }
  return d + n + f;
}

export function sdf(x, y, z, full = true) {
  const az = Math.abs(z);
  let d = sdTrunk(x, y, z);
  if (x > 20 && x < 46) {
    // shoulders and pectorals
    d = smin(d, sdEll(x, y, az, 34.5, -2.6, 8.4, 8.2, 7.6, 4.8), 3.0);
    // arms, folded against the chest while swimming
    d = smin(d, sdRC(x, y, az, 35.2, -6.0, 10.0, 28.2, -12.6, 12.0, 3.1, 2.3), 2.0);
    d = smin(d, sdRC(x, y, az, 28.2, -12.6, 12.0, 35.6, -16.2, 8.6, 2.25, 1.45), 1.0);
    d = smin(d, sdEll(x, y, az, 37.6, -16.6, 7.7, 2.3, 1.05, 1.9), 0.8);
    for (let k = 0; k < 4; k++) {
      d = smin(d, sdRC(x, y, az, 38.6, -16.8, 6.6 + k * 0.78, 41.6, -17.7, 6.1 + k * 0.95, 0.42, 0.07), 0.3);
    }
  }
  if (x > -50 && x < 14) {
    // thighs, shins trailing back, feet and toes
    d = smin(d, sdEll(x, y, az, -3.0, -6.6, 10.6, 12.5, 9.0, 6.1), 4.0);
    d = smin(d, sdRC(x, y, az, -13.5, -14.0, 11.6, -28.6, -15.3, 10.4, 4.4, 2.6), 2.5);
    d = smin(d, sdEll(x, y, az, -34.0, -15.9, 10.2, 6.0, 1.75, 3.1), 1.5);
    for (let k = -1; k <= 1; k++) {
      d = smin(d, sdRC(x, y, az, -37.5, -16.1, 10.2 + k * 1.35, -42.4, -16.7, 10.2 + k * 1.9, 0.9, 0.2), 0.5);
    }
  }
  if (x > 50) d = smin(d, sdHead(x, y, z, az, true), 3.2);
  if (full && Math.abs(d) < 2.2) d = detail(x, y, z, d);
  return d;
}

// material and jaw binding of a body vertex: [mouth, jaw weight, material id]
//   ids: 0 skin, 2 eye, 3 claw / tooth
function attrs(x, y, z) {
  const az = Math.abs(z);
  let jaw = 0, mouth = 0, id = 0;
  if (x > 52) {
    const lip = lipY(x);
    // below the mouth: lower jaw; the corner and the throat stretch
    const below = sstep(lip + 0.12, lip - 0.12, y);
    jaw = below * sstep(55.4, 59.0, x);
    jaw = Math.max(jaw, sstep(53.5, 58.5, x) * sstep(-2.2, -4.5, y) * 0.55);
    // inside the slot (the uncarved head contains the point)
    const inside = sdHead(x, y, z, az, false);
    mouth = sstep(-0.12, -0.55, inside) * sstep(GAP + 0.5, GAP + 0.05, Math.abs(y - lip)) * sstep(SLOT_X - 0.3, SLOT_X + 0.5, x);
    // eyeball
    if (sdEll(x, y, az, 64.85, 2.5, 3.45, 0.76, 0.7, 0.76) < 0.08) id = 2;
  }
  // claws
  if (x > 37.6 && y < -16 && az > 5.5 && az < 10.5) id = 3;
  if (x < -36.5 && y < -15.4 && az > 7.5 && az < 13.5) id = 3;
  return [mouth, jaw, id];
}

// ------------------------------------------------------------------ surface nets
function surfaceNets(fn, x0, y0, z0, h, nx, ny, nz) {
  const N = nx * ny * nz;
  const F = new Float32Array(N);
  const idx = (i, j, k) => (k * ny + j) * nx + i;
  // coarse culling: blocks of B^3 points far from the surface take the block-centre value
  const B = 4, br = h * B * 0.866 * 1.35 + 0.8;
  for (let kb = 0; kb < nz; kb += B) for (let jb = 0; jb < ny; jb += B) for (let ib = 0; ib < nx; ib += B) {
    const dc = fn(x0 + (ib + B / 2) * h, y0 + (jb + B / 2) * h, z0 + (kb + B / 2) * h, false);
    const far = Math.abs(dc) > br + 2.5;
    for (let k = kb; k < Math.min(kb + B, nz); k++) for (let j = jb; j < Math.min(jb + B, ny); j++) for (let i = ib; i < Math.min(ib + B, nx); i++) {
      F[idx(i, j, k)] = far ? dc : fn(x0 + i * h, y0 + j * h, z0 + k * h, true);
    }
  }
  // one vertex per sign-changing cell, at the mean of its edge crossings
  const cellV = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const cidx = (i, j, k) => (k * (ny - 1) + j) * (nx - 1) + i;
  const P = [];
  const E = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const cv = new Float32Array(8);
  for (let k = 0; k < nz - 1; k++) for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    let mask = 0;
    for (let c = 0; c < 8; c++) {
      const v = F[idx(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1))];
      cv[c] = v;
      if (v < 0) mask |= 1 << c;
    }
    if (mask === 0 || mask === 255) continue;
    let sx = 0, sy = 0, sz = 0, n = 0;
    for (const [a, b] of E) {
      const va = cv[a], vb = cv[b];
      if ((va < 0) === (vb < 0)) continue;
      const t = va / (va - vb);
      sx += (a & 1) + (((b & 1) - (a & 1)) * t);
      sy += ((a >> 1) & 1) + ((((b >> 1) & 1) - ((a >> 1) & 1)) * t);
      sz += ((a >> 2) & 1) + ((((b >> 2) & 1) - ((a >> 2) & 1)) * t);
      n++;
    }
    cellV[cidx(i, j, k)] = P.length / 3;
    P.push(x0 + (i + sx / n) * h, y0 + (j + sy / n) * h, z0 + (k + sz / n) * h);
  }
  // quads across sign-changing grid edges
  const I = [];
  const quad = (a, b, c, d, flip) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (flip) I.push(a, d, c, a, c, b); else I.push(a, b, c, a, c, d);
  };
  for (let k = 1; k < nz - 1; k++) for (let j = 1; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const s0 = F[idx(i, j, k)] < 0, s1 = F[idx(i + 1, j, k)] < 0;
    if (s0 === s1) continue;
    quad(cellV[cidx(i, j - 1, k - 1)], cellV[cidx(i, j, k - 1)], cellV[cidx(i, j, k)], cellV[cidx(i, j - 1, k)], s1);
  }
  for (let k = 1; k < nz - 1; k++) for (let j = 0; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
    const s0 = F[idx(i, j, k)] < 0, s1 = F[idx(i, j + 1, k)] < 0;
    if (s0 === s1) continue;
    quad(cellV[cidx(i - 1, j, k - 1)], cellV[cidx(i - 1, j, k)], cellV[cidx(i, j, k)], cellV[cidx(i, j, k - 1)], s1);
  }
  for (let k = 0; k < nz - 1; k++) for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
    const s0 = F[idx(i, j, k)] < 0, s1 = F[idx(i, j, k + 1)] < 0;
    if (s0 === s1) continue;
    quad(cellV[cidx(i - 1, j - 1, k)], cellV[cidx(i, j - 1, k)], cellV[cidx(i, j, k)], cellV[cidx(i - 1, j, k)], s1);
  }
  return { P, I };
}

// ------------------------------------------------------------------ mesh accumulator
function acc() {
  const P = [], N = [], A = [], I = [];
  return {
    P, N, A, I,
    v(p, n, a) { P.push(p[0], p[1], p[2]); N.push(n[0], n[1], n[2]); A.push(a[0], a[1], a[2], a[3]); return P.length / 3 - 1; },
  };
}
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sc3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// ------------------------------------------------------------------ the body
function bodyMesh(out, h) {
  const x0 = -134.5, x1 = 75.5, y0 = -19.5, y1 = 15.5, z0 = -18.8, z1 = 18.8;
  const nx = Math.ceil((x1 - x0) / h) + 1, ny = Math.ceil((y1 - y0) / h) + 1, nz = Math.ceil((z1 - z0) / h) + 1;
  const { P, I } = surfaceNets(sdf, x0, y0, z0, h, nx, ny, nz);
  const nv = P.length / 3;
  const e = 0.12;
  const grad = (x, y, z) => {
    const gx = sdf(x + e, y, z) - sdf(x - e, y, z), gy = sdf(x, y + e, z) - sdf(x, y - e, z), gz = sdf(x, y, z + e) - sdf(x, y, z - e);
    const l = Math.hypot(gx, gy, gz) || 1;
    return [gx / l, gy / l, gz / l];
  };
  const base = out.P.length / 3;
  for (let v = 0; v < nv; v++) {
    let x = P[v * 3], y = P[v * 3 + 1], z = P[v * 3 + 2];
    // project onto the surface (two Newton steps)
    for (let it = 0; it < 2; it++) {
      const d = sdf(x, y, z), g = grad(x, y, z);
      const s = clamp(d, -h, h);
      x -= g[0] * s; y -= g[1] * s; z -= g[2] * s;
    }
    const n = grad(x, y, z);
    // ambient occlusion from the field (Quilez)
    let occ = 0, w = 1;
    for (let i = 1; i <= 5; i++) {
      const hd = 0.35 + 1.1 * i;
      occ += (hd - Math.min(sdf(x + n[0] * hd, y + n[1] * hd, z + n[2] * hd, false), hd)) * w;
      w *= 0.62;
    }
    const ao = clamp(1 - 0.16 * occ, 0, 1);
    const [mouth, jaw, id] = attrs(x, y, z);
    out.v([x, y, z], n, [mouth, jaw, ao, id]);
  }
  for (let i = 0; i < I.length; i++) out.I.push(I[i] + base);
}

// ------------------------------------------------------------------ dorsal plates
// height of the back surface at (x, z) (top of the field along y)
function backTop(x, z) {
  let lo = -2, hi = 18;
  for (let i = 0; i < 26; i++) { const m = (lo + hi) / 2; if (sdf(x, m, z, false) < 0) lo = m; else hi = m; }
  return lo;
}
// a bony leaf: base centre c (embedded), plane basis ax (along the body), up (outward), nrm; size S
function plate(out, r, c, ax, up, nrm, S, seed, lateral, low) {
  const K = 3 + Math.floor(r() * 2.6);
  const lobes = [];
  const span = 0.74 * Math.PI;
  for (let k = 0; k < K; k++) {
    const f = (k + 0.5) / K;
    const th = Math.PI * 0.13 + span * f + (r() - 0.5) * 0.1;
    const centre = 1 - Math.pow(Math.abs(f - 0.55) * 2, 1.6);
    lobes.push({ th, l: (0.68 + 0.32 * centre) * (0.86 + 0.24 * r()), w: (span / K) * (0.62 + 0.12 * r()) });
  }
  const lm = Math.max(...lobes.map((o) => o.l));
  lobes.forEach((o) => (o.l /= lm));
  const ph = r() * 10, serN = 9 + Math.floor(r() * 7);
  const sweep = 0.32 + 0.16 * r();             // tips raked back
  const cup = (r() - 0.5) * 0.09, twist = (r() - 0.5) * 0.06;
  const R = (th) => {
    let m = 0.36 + 0.2 * Math.sin(clamp(th, 0, Math.PI));
    let ridge = 0;
    for (const o of lobes) {
      const t = Math.abs(th - o.th) / o.w;
      // a pointed lobe with full shoulders: the leaf's tips stand out of a broad blade
      if (t < 1) m = Math.max(m, o.l * (1 - 0.42 * Math.pow(t, 0.75)) * Math.pow(1 - t * t, 0.35));
      ridge += Math.exp(-Math.pow((th - o.th) / (0.16 * o.w), 2));
    }
    // serrated, irregular margin
    const sv = Math.abs(((th * serN / Math.PI + ph) % 1) - 0.5) * 2;
    m *= 1 - 0.06 * sv * sv - 0.05 * (vnoise(th * 5 + ph, seed, 0.5) - 0.5);
    return { m, ridge: Math.min(ridge, 1) };
  };
  const NA = low ? (lateral ? 16 : 22) : lateral ? 40 : 56, NR = low ? 4 : lateral ? 8 : 11;
  const TH0 = -0.32, TH1 = Math.PI + 0.32;
  const pts = [[], []];
  for (let side = 0; side < 2; side++) {
    for (let i = 0; i <= NA; i++) {
      const th = TH0 + ((TH1 - TH0) * i) / NA;
      const o = R(th);
      for (let j = 0; j <= NR; j++) {
        const rr = j / NR;
        const rad = rr * o.m * S;
        const a = rad * Math.cos(th), b = rad * Math.sin(th);
        const a2 = a - Math.max(b, 0) * Math.tan(sweep);
        const thick = S * ((0.07 * Math.pow(1 - rr, 1.3) + 0.007) + 0.03 * (1 - rr) * o.ridge);
        const bend = S * (cup * Math.sin(Math.PI * rr) + twist * (a / S) * rr) ;
        const off = (side ? -0.5 : 0.5) * thick + bend;
        const p = add3(add3(add3(c, sc3(ax, a2)), sc3(up, b)), sc3(nrm, off));
        pts[side].push({ p, rr, th });
      }
    }
  }
  const at = (side, i, j) => pts[side][clamp(i, 0, NA) * (NR + 1) + clamp(j, 0, NR)];
  const seedF = (seed * 0.6180339) % 1;
  const base = [[], []];
  for (let side = 0; side < 2; side++) {
    for (let i = 0; i <= NA; i++) for (let j = 0; j <= NR; j++) {
      const q = at(side, i, j);
      const du = sub3(at(side, i + 1, j).p, at(side, i - 1, j).p);
      let dv = sub3(at(side, i, j + 1).p, at(side, i, j - 1).p);
      if (j === 0) dv = sub3(at(side, i, 1).p, q.p);
      let n = norm3(cross3(dv, du));
      // orient to the face's side
      const dn = n[0] * nrm[0] + n[1] * nrm[1] + n[2] * nrm[2];
      if ((side === 0 && dn < 0) || (side === 1 && dn > 0)) n = sc3(n, -1);
      const thn = (q.th - TH0) / (TH1 - TH0);
      base[side].push(out.v(q.p, n, [q.rr, thn, mix(0.45, 1, sstep(0, 0.35, q.rr)), 4 + seedF * 0.9]));
    }
    for (let i = 0; i < NA; i++) for (let j = 0; j < NR; j++) {
      const a = base[side][i * (NR + 1) + j], b = base[side][(i + 1) * (NR + 1) + j];
      const c2 = base[side][(i + 1) * (NR + 1) + j + 1], d = base[side][i * (NR + 1) + j + 1];
      if (side) out.I.push(a, c2, b, a, d, c2); else out.I.push(a, b, c2, a, c2, d);
    }
  }
  // rim joining the two faces along the margin (normal: outward in the plate's plane)
  const rim = [];
  for (let i = 0; i <= NA; i++) {
    const p0 = at(0, i, NR).p, p1 = at(1, i, NR).p;
    const t = sub3(at(0, i + 1, NR).p, at(0, i - 1, NR).p);
    let n = norm3(cross3(t, nrm));
    const outw = sub3(p0, c);
    if (n[0] * outw[0] + n[1] * outw[1] + n[2] * outw[2] < 0) n = sc3(n, -1);
    const thn = (at(0, i, NR).th - TH0) / (TH1 - TH0);
    rim.push(out.v(p0, n, [1, thn, 1, 4 + seedF * 0.9]), out.v(p1, n, [1, thn, 1, 4 + seedF * 0.9]));
  }
  for (let i = 0; i < NA; i++) {
    const a = rim[i * 2], b = rim[i * 2 + 1], c2 = rim[i * 2 + 2], d = rim[i * 2 + 3];
    out.I.push(a, b, d, a, d, c2);
  }
}

export const plateSize = (x) => {
  if (x > 14) return 2.6 + 13.2 * Math.pow(clamp((52 - x) / 38, 0, 1), 1.25);
  return Math.max(15.8 * Math.pow(clamp((x + 126) / 140, 0, 1), 1.35), 1.1);
};

function platesMesh(out, low) {
  const r = rng(1954);
  const fins = [];
  // central row: from the nape to near the tail tip, spacing scaled to the plate
  let x = 51.5, seed = 1;
  while (x > -120) {
    const S = plateSize(x) * (0.9 + 0.2 * r());
    const top = backTop(x, 0);
    const yaw = (r() - 0.5) * 0.07;
    const c = [x, top - (0.12 * S + 0.45), (r() - 0.5) * 0.15];
    const ax = [Math.cos(yaw), 0, Math.sin(yaw)], up = [0, 1, 0], nrm = [-Math.sin(yaw), 0, Math.cos(yaw)];
    plate(out, r, c, ax, up, nrm, S, seed++, false, low);
    fins.push([x, top, 0, S]);
    x -= 0.52 * S + 1.1;
  }
  // two lateral rows, staggered, leaning outwards
  for (const side of [-1, 1]) {
    let xl = 46.5 + (side > 0 ? 0.6 : 0);
    while (xl > -104) {
      const S = plateSize(xl) * 0.6 * (0.88 + 0.24 * r());
      const zb = side * (0.42 + 0.06 * r()) * (0.45 + 0.11 * profW(xl));
      const top = backTop(xl, zb);
      const lean = 0.42 + 0.12 * r();
      const up = [0, Math.cos(lean), side * Math.sin(lean)];
      const yaw = (r() - 0.5) * 0.1;
      const ax = norm3([Math.cos(yaw), 0, Math.sin(yaw) * side]);
      const nrm = norm3(cross3(ax, up));
      const c = [xl, top - (0.12 * S + 0.4), zb];
      plate(out, r, c, ax, up, nrm, S, seed++, true, low);
      fins.push([xl, top, zb, S]);
      xl -= 0.55 * S / 0.6 * 0.62 + 1.2;
    }
  }
  return fins;
}

// ------------------------------------------------------------------ teeth
function lipEdgeZ(x, y) {
  // outer surface of the jaws at height y (bisection in z)
  let lo = 0, hi = 7;
  for (let i = 0; i < 22; i++) { const m = (lo + hi) / 2; if (sdHead(x, y, m, m, false) < 0) lo = m; else hi = m; }
  return lo;
}
function tooth(out, base, dir, side, len, rad, jaw) {
  const SEG = 7, RING = 5;
  const d = norm3(dir);
  const t1 = norm3(cross3(d, Math.abs(d[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]));
  const t2 = cross3(d, t1);
  const rows = [];
  for (let j = 0; j <= RING; j++) {
    const t = j / RING;
    // a curved, laterally flattened cone, recurved towards the throat
    const cen = add3(add3(base, sc3(d, len * t)), [-0.22 * len * t * t, 0, 0]);
    const rr = rad * Math.pow(1 - t, 0.85) + 0.015;
    const row = [];
    for (let k = 0; k < SEG; k++) {
      const a = (k / SEG) * Math.PI * 2;
      const off = add3(sc3(t1, Math.cos(a) * rr), sc3(t2, Math.sin(a) * rr * 0.62));
      row.push(out.v(add3(cen, off), norm3(add3(off, sc3(d, rad * 0.35))), [0, jaw, 1, 3]));
    }
    rows.push(row);
  }
  const tipP = add3(add3(base, sc3(d, len * 1.04)), [-0.22 * len, 0, 0]);
  const tip = out.v(tipP, d, [0, jaw, 1, 3]);
  for (let j = 0; j < RING; j++) for (let k = 0; k < SEG; k++) {
    const a = rows[j][k], b = rows[j][(k + 1) % SEG], c = rows[j + 1][(k + 1) % SEG], e = rows[j + 1][k];
    out.I.push(a, b, c, a, c, e);
  }
  for (let k = 0; k < SEG; k++) out.I.push(rows[RING][k], rows[RING][(k + 1) % SEG], tip);
}
function teethMesh(out) {
  const r = rng(77);
  for (const side of [-1, 1]) {
    for (let x = 59.4; x < 70.6; x += 0.8 + r() * 0.12) {
      const lip = lipY(x);
      const z = lipEdgeZ(x, lip + 0.35) - 0.42;
      if (z < 0.6) continue;
      const len = (0.62 + 0.5 * r()) * (1 + 0.45 * Math.exp(-Math.pow((x - 68.0) / 1.3, 2))) * (x < 61 ? 0.75 : 1);
      tooth(out, [x, lip + 0.38, side * z], [-0.12, -1, side * 0.05], side, len, 0.2 + 0.1 * r(), 0);
    }
    for (let x = 59.8; x < 69.6; x += 0.82 + r() * 0.12) {
      const lip = lipY(x);
      const z = lipEdgeZ(x, lip - 0.35) - 0.55;
      if (z < 0.6) continue;
      const len = (0.55 + 0.45 * r()) * (1 + 0.4 * Math.exp(-Math.pow((x - 67.4) / 1.3, 2))) * (x < 61 ? 0.75 : 1);
      tooth(out, [x, lip - 0.38, side * z], [-0.08, 1, -side * 0.04], side, len, 0.19 + 0.09 * r(), 1);
    }
  }
}

// ------------------------------------------------------------------ everything
// low: a coarse version (shadow map, far away): 1 m cells, coarse plates, no teeth
export function sculpt({ cell = 0.5, low = false } = {}) {
  const t0 = Date.now();
  const out = acc();
  bodyMesh(out, cell);
  const bodyVerts = out.P.length / 3, bodyIdx = out.I.length;
  const fins = platesMesh(out, low);
  if (!low) teethMesh(out);
  const nv = out.P.length / 3;
  return {
    pos: new Float32Array(out.P), nrm: new Float32Array(out.N), aux: new Float32Array(out.A),
    idx: nv > 65535 ? new Uint32Array(out.I) : new Uint16Array(out.I),
    bodyVerts, bodyIdx, fins: new Float32Array(fins.flat()), ms: Date.now() - t0,
  };
}
