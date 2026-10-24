// Small mesh-building toolkit for the rowboat: vector helpers, an attribute builder, swept profiles
// and grids with smooth normals, rigid frames, and a surface-nets polygoniser for signed distance
// fields (the rower's body, head and hands are sculpted as SDFs and polygonised once at load time).
import { makeMesh } from '../gl.js';

export const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sc3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const len3 = (a) => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
export const norm3 = (a) => { const l = len3(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
export const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
export const madd3 = (a, b, s) => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
export const clamp = (x, a, b) => Math.min(Math.max(x, a), b);
export const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
// component of v perpendicular to unit d, normalised
export const perp3 = (v, d) => norm3(sub3(v, sc3(d, dot3(v, d))));
// rotate v about unit axis k by angle a (Rodrigues)
export function rot3(v, k, a) {
  const c = Math.cos(a), s = Math.sin(a), kv = cross3(k, v), kd = dot3(k, v) * (1 - c);
  return [v[0] * c + kv[0] * s + k[0] * kd, v[1] * c + kv[1] * s + k[1] * kd, v[2] * c + kv[2] * s + k[2] * kd];
}

// ---------------------------------------------------------------- frames (column-major mat4)
// a rigid frame: origin o, orthonormal axes X, Y, Z
export const frameMat = (o, X, Y, Z) => new Float32Array([X[0], X[1], X[2], 0, Y[0], Y[1], Y[2], 0, Z[0], Z[1], Z[2], 0, o[0], o[1], o[2], 1]);
export function invRigid(m) {
  const o = new Float32Array(16);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[c * 4 + r] = m[r * 4 + c];
  for (let r = 0; r < 3; r++) o[12 + r] = -(o[r] * m[12] + o[4 + r] * m[13] + o[8 + r] * m[14]);
  o[15] = 1;
  return o;
}
export function xform(m, p) {
  return [m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]];
}
export function xdir(m, p) {
  return [m[0] * p[0] + m[4] * p[1] + m[8] * p[2], m[1] * p[0] + m[5] * p[1] + m[9] * p[2], m[2] * p[0] + m[6] * p[1] + m[10] * p[2]];
}
// normal matrix (mat3) of an affine transform with orthogonal (possibly scaled) columns
export function normalMat(m) {
  const o = new Float32Array(9);
  for (let c = 0; c < 3; c++) {
    const x = m[c * 4], y = m[c * 4 + 1], z = m[c * 4 + 2], l2 = x * x + y * y + z * z || 1;
    o[c * 3] = x / l2; o[c * 3 + 1] = y / l2; o[c * 3 + 2] = z / l2;
  }
  return o;
}
// two-bone IK: middle joint between root a and end b (segment lengths l1, l2), bending towards pole
export function ik(a, b, l1, l2, pole) {
  const d = sub3(b, a), dist = clamp(len3(d), Math.abs(l1 - l2) + 1e-3, l1 + l2 - 1e-4), dir = norm3(d);
  const x = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist), h = Math.sqrt(Math.max(l1 * l1 - x * x, 0));
  const p = perp3(pole, dir);
  return add3(add3(a, sc3(dir, x)), sc3(p, h));
}

// ---------------------------------------------------------------- attribute builder
export class Builder {
  constructor(sizes) { this.sizes = sizes; this.data = sizes.map(() => []); this.I = []; this.n = 0; }
  v(...attrs) {
    for (let k = 0; k < this.sizes.length; k++) {
      const d = this.data[k], a = attrs[k] || [];
      for (let i = 0; i < this.sizes[k]; i++) d.push(a[i] ?? 0);
    }
    return this.n++;
  }
  tri(a, b, c) { this.I.push(a, b, c); }
  quad(a, b, c, d) { this.I.push(a, b, c, a, c, d); }
  upload(gl) { return uploadArrays(gl, this.arrays()); }
  // plain typed arrays (transferable from a worker)
  arrays() { return { data: this.data.map((d) => new Float32Array(d)), sizes: this.sizes.slice(), idx: new Uint32Array(this.I) }; }
}
export function uploadArrays(gl, a) {
  return makeMesh(gl, a.data.map((d, k) => [d, a.sizes[k]]), a.idx);
}

// grid surface with smooth normals from central differences (wrapJ: columns form a closed loop).
// fn(i, j) -> { p, uv, m } (uv: 2 floats, m: material id + seed). Normal sign is fixed in the shader.
export function grid(b, rows, cols, fn, { wrapJ = false } = {}) {
  const pts = [];
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) pts.push(fn(i, j));
  const at = (i, j) => {
    i = clamp(i, 0, rows - 1);
    j = wrapJ ? ((j % (cols - 1)) + (cols - 1)) % (cols - 1) : clamp(j, 0, cols - 1);
    return pts[i * cols + j].p;
  };
  const base = b.n;
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const du = sub3(at(i + 1, j), at(i - 1, j)), dv = sub3(at(i, j + 1), at(i, j - 1));
    let n = cross3(du, dv);
    if (len3(n) < 1e-12) n = pts[i * cols + j].n || [0, 1, 0];
    const q = pts[i * cols + j];
    b.v(q.p, norm3(n), q.uv || [0, 0], q.m || [0, 0]);
  }
  for (let i = 0; i < rows - 1; i++) for (let j = 0; j < cols - 1; j++) {
    const a = base + i * cols + j, c = a + cols;
    b.I.push(a, c, a + 1, a + 1, c, c + 1);
  }
  return base;
}

// rounded-rectangle profile (closed, counter-clockwise), half sizes hx, hy, corner radius r
export function roundRect(hx, hy, r, seg = 3) {
  const out = [];
  r = Math.min(r, hx * 0.999, hy * 0.999);
  const cs = [[hx - r, hy - r, 0], [-hx + r, hy - r, Math.PI / 2], [-hx + r, -hy + r, Math.PI], [hx - r, -hy + r, 1.5 * Math.PI]];
  for (const [cx, cy, a0] of cs) for (let k = 0; k <= seg; k++) {
    const a = a0 + (k / seg) * (Math.PI / 2);
    out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return out;
}

// sweep a closed 2D profile along a path. path(t) -> { o, A, B } (profile x along A, y along B).
// mat(t, k) -> [id, seed]; uv: [arc length along the path (m), k / n]. Caps at both ends.
export function sweep(b, path, n, profile, mat, { caps = true, uvScale = 1 } = {}) {
  const P = profile.concat([profile[0]]);
  const frames = [];
  for (let i = 0; i <= n; i++) frames.push(path(i / n));
  let s = 0;
  const arc = [0];
  for (let i = 1; i <= n; i++) { s += len3(sub3(frames[i].o, frames[i - 1].o)); arc.push(s); }
  grid(b, n + 1, P.length, (i, k) => {
    const f = frames[i], q = P[k];
    return { p: add3(f.o, add3(sc3(f.A, q[0]), sc3(f.B, q[1]))), uv: [arc[i] * uvScale, k / profile.length], m: mat(i / n, k) };
  }, { wrapJ: true });
  if (!caps) return;
  for (const i of [0, n]) {
    const f = frames[i], nrm = norm3(cross3(f.A, f.B));
    const c = b.v(f.o, nrm, [arc[i], 0.5], mat(i / n, -1));
    const first = b.n;
    for (const q of profile) b.v(add3(f.o, add3(sc3(f.A, q[0]), sc3(f.B, q[1]))), nrm, [arc[i] + q[0], 0.5 + q[1]], mat(i / n, -1));
    for (let k = 0; k < profile.length; k++) b.tri(c, first + k, first + ((k + 1) % profile.length));
  }
}

// straight rounded box from a to b (length axis), width along `wDir` (made perpendicular), thickness
export function beam(b, a, c, width, thick, wDir, r, mat, n = 1) {
  const d = norm3(sub3(c, a)), W = perp3(wDir, d), T = cross3(d, W);
  sweep(b, (t) => ({ o: lerp3(a, c, t), A: W, B: T }), n, roundRect(width / 2, thick / 2, r), mat);
}

// ---------------------------------------------------------------- SDF helpers
export const smin = (a, b, k) => { if (k <= 0) return Math.min(a, b); const h = Math.max(k - Math.abs(a - b), 0) / k; return Math.min(a, b) - h * h * k * 0.25; };
export const smax = (a, b, k) => -smin(-a, -b, k);
// round cone (capsule with radii r1 at a and r2 at b) - Quilez
export function sdRoundCone(p, a, b, r1, r2) {
  const bx = b[0] - a[0], by = b[1] - a[1], bz = b[2] - a[2];
  const l2 = bx * bx + by * by + bz * bz, rr = r1 - r2, a2 = l2 - rr * rr, il2 = 1 / l2;
  const px = p[0] - a[0], py = p[1] - a[1], pz = p[2] - a[2];
  const y = px * bx + py * by + pz * bz, z = y - l2;
  const xx = px * l2 - bx * y, xy = py * l2 - by * y, xz = pz * l2 - bz * y;
  const x2 = xx * xx + xy * xy + xz * xz, y2 = y * y * l2, z2 = z * z * l2;
  const k = Math.sign(rr) * rr * rr * x2;
  if (Math.sign(z) * a2 * z2 > k) return Math.sqrt(x2 + z2) * il2 - r2;
  if (Math.sign(y) * a2 * y2 < k) return Math.sqrt(x2 + y2) * il2 - r1;
  return (Math.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}
export function sdCapsule(p, a, b, r) {
  const px = p[0] - a[0], py = p[1] - a[1], pz = p[2] - a[2], bx = b[0] - a[0], by = b[1] - a[1], bz = b[2] - a[2];
  const h = clamp((px * bx + py * by + pz * bz) / (bx * bx + by * by + bz * bz), 0, 1);
  const dx = px - bx * h, dy = py - by * h, dz = pz - bz * h;
  return Math.sqrt(dx * dx + dy * dy + dz * dz) - r;
}
// ellipsoid (approximate distance, Quilez) centred at c with radii r
export function sdEllipsoid(p, c, r) {
  const qx = (p[0] - c[0]) / r[0], qy = (p[1] - c[1]) / r[1], qz = (p[2] - c[2]) / r[2];
  const k0 = Math.sqrt(qx * qx + qy * qy + qz * qz);
  const ax = qx / r[0], ay = qy / r[1], az = qz / r[2], k1 = Math.sqrt(ax * ax + ay * ay + az * az);
  return k1 > 1e-9 ? (k0 * (k0 - 1)) / k1 : -Math.min(r[0], r[1], r[2]);
}
export function sdBox(p, c, h, r = 0) {
  const qx = Math.abs(p[0] - c[0]) - h[0] + r, qy = Math.abs(p[1] - c[1]) - h[1] + r, qz = Math.abs(p[2] - c[2]) - h[2] + r;
  const mx = Math.max(qx, 0), my = Math.max(qy, 0), mz = Math.max(qz, 0);
  return Math.sqrt(mx * mx + my * my + mz * mz) + Math.min(Math.max(qx, qy, qz), 0) - r;
}

// Surface nets over [lo, hi] with cell size h. f(p) -> signed distance (negative inside).
// Blocks far from the surface (|d| at the block centre > its radius) are skipped. Vertices are
// projected onto the surface along the gradient; normals are the normalised gradient.
export function surfaceNets(f, lo, hi, h) {
  const nx = Math.ceil((hi[0] - lo[0]) / h) + 1, ny = Math.ceil((hi[1] - lo[1]) / h) + 1, nz = Math.ceil((hi[2] - lo[2]) / h) + 1;
  const val = new Float32Array(nx * ny * nz).fill(1e3);
  const idx = (i, j, k) => (k * ny + j) * nx + i;
  const B = 6, rad = Math.sqrt(3) * (B * h) / 2;
  for (let bk = 0; bk < nz; bk += B) for (let bj = 0; bj < ny; bj += B) for (let bi = 0; bi < nx; bi += B) {
    const c = [lo[0] + (bi + B / 2) * h, lo[1] + (bj + B / 2) * h, lo[2] + (bk + B / 2) * h];
    const dc = f(c);
    if (Math.abs(dc) > rad * 1.4 + h) {
      for (let k = bk; k < Math.min(bk + B + 1, nz); k++) for (let j = bj; j < Math.min(bj + B + 1, ny); j++) for (let i = bi; i < Math.min(bi + B + 1, nx); i++) {
        const o = idx(i, j, k);
        if (val[o] === 1e3) val[o] = dc > 0 ? 1e2 : -1e2;
      }
      continue;
    }
    for (let k = bk; k < Math.min(bk + B + 1, nz); k++) for (let j = bj; j < Math.min(bj + B + 1, ny); j++) for (let i = bi; i < Math.min(bi + B + 1, nx); i++) {
      const o = idx(i, j, k);
      if (val[o] === 1e3 || Math.abs(val[o]) === 1e2) val[o] = f([lo[0] + i * h, lo[1] + j * h, lo[2] + k * h]);
    }
  }
  // one vertex per sign-changing cell
  const vid = new Int32Array(nx * ny * nz).fill(-1);
  const pos = [];
  const corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const cv = new Float32Array(8);
  for (let k = 0; k < nz - 1; k++) for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    let inside = 0;
    for (let c = 0; c < 8; c++) { cv[c] = val[idx(i + corners[c][0], j + corners[c][1], k + corners[c][2])]; if (cv[c] < 0) inside++; }
    if (inside === 0 || inside === 8) continue;
    let sx = 0, sy = 0, sz = 0, n = 0;
    for (const [a, b] of edges) {
      if ((cv[a] < 0) === (cv[b] < 0)) continue;
      const t = cv[a] / (cv[a] - cv[b]);
      sx += corners[a][0] + (corners[b][0] - corners[a][0]) * t;
      sy += corners[a][1] + (corners[b][1] - corners[a][1]) * t;
      sz += corners[a][2] + (corners[b][2] - corners[a][2]) * t;
      n++;
    }
    vid[idx(i, j, k)] = pos.length;
    pos.push([lo[0] + (i + sx / n) * h, lo[1] + (j + sy / n) * h, lo[2] + (k + sz / n) * h]);
  }
  // quads across sign-changing grid edges
  const tris = [];
  for (let k = 1; k < nz - 1; k++) for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
    const v0 = val[idx(i, j, k)] < 0;
    for (let ax = 0; ax < 3; ax++) {
      const i1 = i + (ax === 0), j1 = j + (ax === 1), k1 = k + (ax === 2);
      if (i1 >= nx || j1 >= ny || k1 >= nz) continue;
      const v1 = val[idx(i1, j1, k1)] < 0;
      if (v0 === v1) continue;
      // the four cells around this edge
      let q;
      if (ax === 0) q = [idx(i, j - 1, k - 1), idx(i, j, k - 1), idx(i, j, k), idx(i, j - 1, k)];
      else if (ax === 1) q = [idx(i - 1, j, k - 1), idx(i - 1, j, k), idx(i, j, k), idx(i, j, k - 1)];
      else q = [idx(i - 1, j - 1, k), idx(i, j - 1, k), idx(i, j, k), idx(i - 1, j, k)];
      const a = vid[q[0]], b = vid[q[1]], c = vid[q[2]], d = vid[q[3]];
      if (a < 0 || b < 0 || c < 0 || d < 0) continue;
      if (v0) tris.push(a, b, c, a, c, d); else tris.push(a, c, b, a, d, c);
    }
  }
  // project onto the surface, normals from the gradient
  // (tetrahedral gradient: four evaluations)
  const e = h * 0.3;
  const grad = (p) => {
    const a = f([p[0] + e, p[1] - e, p[2] - e]), b = f([p[0] - e, p[1] - e, p[2] + e]), c = f([p[0] - e, p[1] + e, p[2] - e]), d = f([p[0] + e, p[1] + e, p[2] + e]);
    return { g: norm3([a - b - c + d, -a - b + c + d, -a + b - c + d]), d: (a + b + c + d) / 4 };
  };
  const nrm = [];
  for (let v = 0; v < pos.length; v++) {
    const { g, d } = grad(pos[v]);
    pos[v] = sub3(pos[v], sc3(g, clamp(d, -h, h)));
    nrm.push(g);
  }
  return { pos, nrm, tris };
}

// Laplacian-free tidy-up: drop degenerate triangles (surface nets can emit slivers at thin parts)
export function cleanTris(pos, tris) {
  const out = [];
  for (let t = 0; t < tris.length; t += 3) {
    const a = pos[tris[t]], b = pos[tris[t + 1]], c = pos[tris[t + 2]];
    if (len3(cross3(sub3(b, a), sub3(c, a))) > 1e-12) out.push(tris[t], tris[t + 1], tris[t + 2]);
  }
  return out;
}

// character meshes: position, normal, fold (along, around, strength), fold tangent, material
// weights (+ bone indices and weights when skinned)
export function makeSkinBuilder(skinned = true) {
  return new Builder(skinned ? [3, 3, 3, 3, 4, 4, 4] : [3, 3, 3, 3, 4]);
}
