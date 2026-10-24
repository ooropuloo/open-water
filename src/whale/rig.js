// Whale skeleton and behaviour.
//
// The spine is solved by forward kinematics from a root point: the body's curvature (in the dorsal
// plane), lateral bend and twist are integrated along the rest spine, so the pose is a few smooth
// fields rather than per-bone keys. While swimming the curvature is taken from the swimming path
// itself: the body point x metres ahead of the centre of mass sits on the path where the centre of
// mass will be (x / v) seconds later (follow the leader), so the head, back, dorsal fin and tail stock
// roll over the surface one after the other as a real surfacing whale does. A travelling wave of
// curvature, growing towards the tail, is the fluke beat. The fluke-up dive pivots the body about the
// tail stock, and the breach is ballistic flight with twist and roll.
//
// The cycle (about 84 s) is deterministic in its time, so any moment can be shown directly (?whale=).
import { SPINE_X, NSPINE, B_PECT_R, B_PECT_R2, B_PECT_L, B_PECT_L2, B_FLUKE, B_FLUKE2, NBONES, FLUKE_X, FLUKE2_X,
  PECT_ROOT, PECT_SPAN, PECT_CHORD, PECT_LEN } from './shape.js';

export const XC = 0.6;              // rest x of the centre of mass
const sstep = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
const lerp = (a, b, t) => a + (b - a) * t;

// ---------------------------------------------------------------- 3x3 rotations (columns X, Y, Z)
const rotZ = (R, a) => { const c = Math.cos(a), s = Math.sin(a); const [X, Y, Z] = R; return [X.map((v, i) => v * c + Y[i] * s), Y.map((v, i) => -X[i] * s + v * c), Z]; };
const rotY = (R, a) => { const c = Math.cos(a), s = Math.sin(a); const [X, Y, Z] = R; return [X.map((v, i) => v * c - Z[i] * s), Y, Z.map((v, i) => X[i] * s + v * c)]; };
const rotX = (R, a) => { const c = Math.cos(a), s = Math.sin(a); const [X, Y, Z] = R; return [X, Y.map((v, i) => v * c + Z[i] * s), Z.map((v, i) => -Y[i] * s + v * c)]; };
const nrm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const crs = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dt3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function ortho(R) {
  const X = nrm(R[0]);
  const Y = nrm(R[1].map((v, i) => v - dt3(R[1], X) * X[i]));
  return [X, Y, crs(X, Y)];
}
// world rotation from heading (xz unit vector), pitch and roll
export function attitude(h, pitch, roll) {
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const X = [cp * h[0], sp, cp * h[1]];
  const Y0 = [-sp * h[0], cp, -sp * h[1]];
  const Z0 = crs(X, Y0);
  const c = Math.cos(roll), s = Math.sin(roll);
  const Y = Y0.map((v, i) => v * c + Z0[i] * s);
  return [X, Y, crs(X, Y)];
}

// ---------------------------------------------------------------- forward kinematics
// pose: { rootX, pos (world), R (world rotation at the root), kappa(x), lat(x), twist(x) }
// returns spine frames [{ p, R }] at SPINE_X
export function solveSpine(pose) {
  const out = new Array(NSPINE);
  const k = pose.kappa, la = pose.lat || (() => 0), tw = pose.twist || (() => 0);
  for (const dir of [1, -1]) {
    let R = pose.R, p = pose.pos.slice(), x = pose.rootX;
    const ids = [];
    for (let i = 0; i < NSPINE; i++) if (dir > 0 ? SPINE_X[i] >= pose.rootX : SPINE_X[i] < pose.rootX) ids.push(i);
    if (dir < 0) ids.reverse();
    for (const i of ids) {
      const target = SPINE_X[i];
      while (Math.abs(target - x) > 1e-6) {
        const h = Math.sign(target - x) * Math.min(Math.abs(target - x), 0.12);
        const xm = x + h / 2;
        let Rh = rotX(rotY(rotZ(R, k(xm) * h / 2), la(xm) * h / 2), tw(xm) * h / 2);
        p = p.map((v, j) => v + Rh[0][j] * h);
        R = rotX(rotY(rotZ(Rh, k(xm) * h / 2), la(xm) * h / 2), tw(xm) * h / 2);
        x += h;
      }
      R = ortho(R);
      out[i] = { p: p.slice(), R };
    }
  }
  return out;
}
export function blendFrames(a, b, w) {
  if (w <= 0) return a;
  if (w >= 1) return b;
  return a.map((fa, i) => {
    const fb = b[i];
    return { p: fa.p.map((v, j) => lerp(v, fb.p[j], w)), R: ortho([0, 1, 2].map((c) => fa.R[c].map((v, j) => lerp(v, fb.R[c][j], w)))) };
  });
}

// ---------------------------------------------------------------- bone matrices
// column-major mat4 of a frame (R, p) composed with translate(-x0) of the rest pose
function boneMat(out, o, R, p, x0) {
  const [X, Y, Z] = R;
  out.set([X[0], X[1], X[2], 0, Y[0], Y[1], Y[2], 0, Z[0], Z[1], Z[2], 0, p[0] - X[0] * x0, p[1] - X[1] * x0, p[2] - X[2] * x0, 1], o);
}
// rest-local rigid transform as {R (3 cols), t}: apply q -> R q + t
const tApply = (T, q) => [0, 1, 2].map((i) => T.R[0][i] * q[0] + T.R[1][i] * q[1] + T.R[2][i] * q[2] + T.t[i]);
const tRot = (T, q) => [0, 1, 2].map((i) => T.R[0][i] * q[0] + T.R[1][i] * q[1] + T.R[2][i] * q[2]);
const tMul = (A, B) => ({ R: B.R.map((c) => tRot(A, c)), t: tApply(A, B.t) });
// rotation by angle a about unit axis u through point c (rest-local)
function about(u, a, c) {
  const co = Math.cos(a), si = Math.sin(a), t = 1 - co;
  const [x, y, z] = u;
  const R = [[t * x * x + co, t * x * y + si * z, t * x * z - si * y], [t * x * y - si * z, t * y * y + co, t * y * z + si * x], [t * x * z + si * y, t * y * z - si * x, t * z * z + co]];
  const T = { R, t: [0, 0, 0] };
  const rc = tRot(T, c);
  T.t = [c[0] - rc[0], c[1] - rc[1], c[2] - rc[2]];
  return T;
}
const mirrorT = (T) => ({ R: T.R.map((c, i) => [c[0] * (i === 2 ? -1 : 1), c[1] * (i === 2 ? -1 : 1), -c[2] * (i === 2 ? -1 : 1)]), t: [T.t[0], T.t[1], -T.t[2]] });
// a frame (world) times a rest-local transform -> matrix
function frameTimes(out, o, F, x0, T) {
  // world(q) = F.R (q - x0 e_x) + F.p, then q = T(q_rest)
  const R = T.R.map((c) => [0, 1, 2].map((i) => F.R[0][i] * c[0] + F.R[1][i] * c[1] + F.R[2][i] * c[2]));
  const tt = [T.t[0] - x0, T.t[1], T.t[2]];
  const p = [0, 1, 2].map((i) => F.R[0][i] * tt[0] + F.R[1][i] * tt[1] + F.R[2][i] * tt[2] + F.p[i]);
  out.set([R[0][0], R[0][1], R[0][2], 0, R[1][0], R[1][1], R[1][2], 0, R[2][0], R[2][1], R[2][2], 0, p[0], p[1], p[2], 1], o);
}

// fills mats (Float32Array NBONES*16) from spine frames and appendage angles
export function boneMatrices(mats, frames, ap, attach) {
  for (let i = 0; i < NSPINE; i++) boneMat(mats, i * 16, frames[i].R, frames[i].p, SPINE_X[i]);
  // flippers: abduction (raise) about the body axis, sweep about the vertical, twist about the span
  for (const side of [1, -1]) {
    const a = side > 0 ? ap.pR : ap.pL;
    let K = about([1, 0, 0], -a.ab, PECT_ROOT);
    K = tMul(K, about([0, 1, 0], a.sw, PECT_ROOT));
    K = tMul(K, about(PECT_SPAN, a.tw, PECT_ROOT));
    const mid = PECT_ROOT.map((v, i) => v + PECT_SPAN[i] * PECT_LEN * 0.45);
    const K2 = tMul(K, about(PECT_CHORD, a.flex, mid));
    const KK = side > 0 ? K : mirrorT(K), KK2 = side > 0 ? K2 : mirrorT(K2);
    const F = frames[attach];
    frameTimes(mats, (side > 0 ? B_PECT_R : B_PECT_L) * 16, F, SPINE_X[attach], KK);
    frameTimes(mats, (side > 0 ? B_PECT_R2 : B_PECT_L2) * 16, F, SPINE_X[attach], KK2);
  }
  // flukes: continue the spine to the fluke root, then pitch; the outer blades flex further
  const F0 = frames[0];
  const dx = FLUKE_X - SPINE_X[0];
  const pF = F0.p.map((v, i) => v + F0.R[0][i] * dx);
  const RF = rotZ(F0.R, ap.fluke);
  boneMat(mats, B_FLUKE * 16, RF, pF, FLUKE_X);
  const pF2 = pF.map((v, i) => v + RF[0][i] * (FLUKE2_X - FLUKE_X));
  boneMat(mats, B_FLUKE2 * 16, rotZ(RF, ap.flex), pF2, FLUKE2_X);
}
// transform a rest-local point by bone b
export function applyBone(mats, b, q) {
  const o = b * 16;
  return [0, 1, 2].map((i) => mats[o + i] * q[0] + mats[o + 4 + i] * q[1] + mats[o + 8 + i] * q[2] + mats[o + 12 + i]);
}
export function applyBoneDir(mats, b, q) {
  const o = b * 16;
  return [0, 1, 2].map((i) => mats[o + i] * q[0] + mats[o + 4 + i] * q[1] + mats[o + 8 + i] * q[2]);
}
export { NBONES };

// ---------------------------------------------------------------- behaviour
export const CYCLE = 84;
export const BLOWS = [9, 22, 35];
const APEX_LAG = 2.0;                // centre of mass passes the top of its arc after the blow
const T_DIVE = 37.6;                 // terminal dive (flukes up) starts
export const T_EXIT = 64;            // breach: centre of mass leaves the water
const V = 2.0;                       // surface swimming speed (m/s)
const R_ARC = 22;                    // radius of the rolling arc at the surface
const Y_APEX = -0.42;                // spine height at the top of the arc (the back clears ~0.7 m)
const softmax = (a, b, k) => Math.max(a, b) + Math.log1p(Math.exp(-k * Math.abs(a - b))) / k;
// surface sequence: three rolling arcs over a shallow cruise. Between the second and third breath the
// whale turns away from the camera (so the terminal dive shows the flukes from behind).
const PSI = 1.1, TURN0 = 27.0, TURN1 = 31.5;
const psiAt = (t) => PSI * sstep(TURN0, TURN1, t);
// horizontal track position (X along the initial heading, Z towards the camera) at time t
function surfXZ(t) {
  const X0 = V * (TURN0 - BLOWS[1] - APEX_LAG);
  if (t <= TURN0) return [V * (t - BLOWS[1] - APEX_LAG), 0];
  const t1 = Math.min(t, TURN1), n = 24, h = (t1 - TURN0) / n;
  let X = X0, Z = 0;
  for (let i = 0; i < n; i++) { const a = psiAt(TURN0 + (i + 0.5) * h); X += V * Math.cos(a) * h; Z -= V * Math.sin(a) * h; }
  if (t > TURN1) { X += V * Math.cos(PSI) * (t - TURN1); Z -= V * Math.sin(PSI) * (t - TURN1); }
  return [X, Z];
}
function surfY(t) {
  let Y = -3.6 - 8 * (1 - sstep(0, 6, t));
  for (const b of BLOWS) {
    const dx = V * (t - b - APEX_LAG);
    // the last arc rounds over more steeply into the dive
    const r = b === BLOWS[2] && dx > 0 ? R_ARC * 0.55 : R_ARC;
    Y = softmax(Y, Y_APEX - (dx * dx) / (2 * r), 1.6);
  }
  return Y;
}
// pitch of the path (the horizontal speed is V throughout)
function pathTangent(t) { const e = 0.05; return Math.atan2(surfY(t + e) - surfY(t - e), 2 * e * V); }
function pathCurv(t) { const e = 0.08; return (pathTangent(t + e) - pathTangent(t - e)) / (2 * e * V); }
const turnRate = (t) => { const e = 0.05; return (psiAt(t + e) - psiAt(t - e)) / (2 * e); };
const headingAt = (track, t) => { const a = psiAt(t); return [track.h[0] * Math.cos(a) - track.n[0] * Math.sin(a), track.h[1] * Math.cos(a) - track.n[1] * Math.sin(a)]; };

// fluke beat: curvature wave travelling to the tail, plus the fluke's pitch lagging the heave
function swim(amp, period, t) {
  const w = (2 * Math.PI) / period, kk = (2 * Math.PI) / 13;
  return {
    kappa: (x) => amp * sstep(3.0, -5.9, x) * Math.sin(w * t + kk * x),
    fluke: (amp / 0.03) * 0.32 * Math.sin(w * t + kk * FLUKE_X - 1.4),
  };
}
const restPect = (t, ab = 0.18) => ({ ab: ab + 0.05 * Math.sin(t * 0.7), sw: 0.05, tw: 0.08, flex: 0.04 * Math.sin(t * 0.9) });

// returns { pose, ap (appendages), phase, surf (0..1 near the surface), ride (follow the swell) }
// track: { c: [x, z] centre, h: [x, z] heading, n: [x, z] lateral (towards the camera), zs: side of the
// whale's local z that faces the camera (+1 / -1) }
export function behaviour(t, track) {
  const W = (X, Y, Z = 0) => [track.c[0] + track.h[0] * X + track.n[0] * Z, Y, track.c[1] + track.h[1] * X + track.n[1] * Z];
  // surface sequence (blows at 9, 22, 35 s)
  const surface = () => {
    const [X, Z] = surfXZ(t), Y = surfY(t);
    const th = pathTangent(t);
    const sw = swim(0.022, 3.4, t);
    const pose = {
      rootX: XC, pos: W(X, Y, Z), R: attitude(headingAt(track, t), th, 0.04 * Math.sin(t * 0.31)),
      kappa: (x) => pathCurv(t + (x - XC) / V) + sw.kappa(x),
      // (turning away from the camera: the body follows the bend of its path)
      lat: (x) => track.zs * turnRate(t + (x - XC) / V) / V + 0.004 * Math.sin(t * 0.2 + x * 0.1),
    };
    return { pose, ap: { pR: restPect(t), pL: restPect(t + 1.3), fluke: sw.fluke, flex: sw.fluke * 0.5 }, ride: 1 };
  };
  // terminal dive: arch, pivot about the tail stock, flukes up, slide down along the body axis
  const dive = () => {
    const u = t - T_DIVE;
    const [X0, Z0] = surfXZ(T_DIVE), Y0 = surfY(T_DIVE);
    const hd = headingAt(track, T_DIVE);
    const th0 = pathTangent(T_DIVE);
    const XP = -3.4;                                  // pivot: the tail stock behind the dorsal fin
    // 2D in the vertical plane of the heading: (along, up), relative to the centre of mass at T_DIVE
    let P = [Math.cos(th0) * (XP - XC), Y0 + Math.sin(th0) * (XP - XC) - 0.12];
    const th = th0 + (-1.36 - th0) * sstep(0.2, 4.2, u);
    P[0] += V * 1.6 * (1 - Math.exp(-u / 1.6));
    P[1] -= 0.10 * u;
    const sl = u > 2.9 ? 0.5 * 0.55 * (u - 2.9) ** 2 + 0.45 * (u - 2.9) : 0;
    P = [P[0] + Math.cos(th) * sl, P[1] + Math.sin(th) * sl];
    const base = W(X0, 0, Z0);
    const arch = -0.045 * (1 - sstep(0.6, 3.4, u)) - 0.012;
    const tail = -0.05 * sstep(2.4, 4.6, u) * (1 - sstep(6.0, 8.0, u));
    const sw = swim(0.02 * (1 - sstep(0, 1.2, u)), 3.4, t);
    const pose = {
      rootX: XP, pos: [base[0] + hd[0] * P[0], P[1], base[2] + hd[1] * P[0]], R: attitude(hd, th, 0.03),
      kappa: (x) => arch + (x < -2.5 ? tail * sstep(-2.5, -4.5, x) : 0) + sw.kappa(x),
    };
    const fl = 0.10 * sstep(2.6, 4.0, u) * (1 - sstep(5.0, 7.0, u));
    return { pose, ap: { pR: restPect(t, -0.12), pL: restPect(t + 1.3, -0.12), fluke: sw.fluke + fl, flex: 0.05 + sw.fluke * 0.4 }, ride: 1 - sstep(1.0, 3.0, u) };
  };
  // breach: steep run from depth, out of the water to two thirds of its length, twist, fall on its side
  const breach = () => {
    const u = t - T_EXIT;
    const thB = 1.26, vB = 8.2, g = 8.6;
    const vx = vB * Math.cos(thB), vy = vB * Math.sin(thB);
    const Ti = (vy + Math.sqrt(vy * vy - 2 * g * 0.9)) / g;
    const XB = 2, ZB = 8;
    let X, Y, th, roll, kap = 0.008, sw;
    if (u < 0) {
      X = XB + vx * u; Y = vy * u; th = thB; roll = 0.35 * sstep(-1.6, 0, u);
      sw = swim(0.05 * (1 - sstep(-1.0, 0.3, u)), 1.5, t);
    } else if (u < Ti) {
      X = XB + vx * u; Y = vy * u - 0.5 * g * u * u;
      th = thB - 1.0 * sstep(0.12, 1.85, u);
      roll = 0.35 + 1.72 * sstep(0.0, 1.75, u);
      kap = 0.008 + 0.012 * Math.sin(Math.PI * Math.min(u / Ti, 1));
      sw = swim(0.01, 2.0, t);
    } else {
      const u2 = u - Ti;
      const vyi = vy - g * Ti;
      X = XB + vx * Ti + vx * 0.9 * (1 - Math.exp(-u2 / 0.9));
      Y = 0.9 + vyi * 0.55 * (1 - Math.exp(-u2 / 0.55)) - 0.55 * u2;
      th = thB - 1.0 + (-0.22 - (thB - 1.0)) * sstep(0, 3.5, u2);
      roll = 2.07 * (1 - sstep(1.0, 6.5, u2));
      kap = 0.02 * Math.exp(-u2 / 1.5);
      sw = swim(0.02 * sstep(1.5, 4, u2), 3.0, t);
    }
    // (rolls its belly and flippers towards the camera)
    const pose = { rootX: XC, pos: W(X, Y, ZB), R: attitude(track.h, th, -track.zs * roll), kappa: (x) => kap + sw.kappa(x) };
    // flippers flung out as it leaves the water, waving through the twist
    const fly = sstep(-0.9, 0.2, u) * (1 - sstep(Ti + 0.5, Ti + 3.5, u));
    const pR = { ab: lerp(0.18, 1.05, fly) + 0.25 * fly * Math.sin(u * 3.1), sw: lerp(0.05, 0.45, fly), tw: lerp(0.08, -0.3, fly), flex: 0.18 * fly * Math.sin(u * 3.7 + 1) };
    const pL = { ab: lerp(0.18, 0.85, fly) + 0.3 * fly * Math.sin(u * 2.7 + 2), sw: lerp(0.05, 0.3, fly), tw: lerp(0.08, -0.2, fly), flex: 0.2 * fly * Math.sin(u * 3.3) };
    return { pose, ap: { pR, pL, fluke: sw.fluke - 0.1 * fly, flex: sw.fluke * 0.5 }, ride: u < -1.2 || u > Ti + 1 ? 1 : 0, Ti };
  };
  const deep = () => ({ pose: { rootX: XC, pos: W(0, -40), R: attitude(track.h, 0, 0), kappa: () => 0 }, ap: { pR: restPect(t), pL: restPect(t), fluke: 0, flex: 0 }, ride: 0 });

  let phase, A, Bp = null, w = 0;
  if (t < T_DIVE) { phase = 'surface'; A = surface(); }
  else if (t < T_DIVE + 12) {
    phase = 'dive'; A = dive();
    w = 1 - sstep(0, 1.5, t - T_DIVE);
    if (w > 0) Bp = surface();
  } else if (t < T_EXIT - 4.5) { phase = 'deep'; A = deep(); }
  else if (t < T_EXIT + 15) { phase = 'breach'; A = breach(); }
  else { phase = 'deep'; A = deep(); }
  return { main: A, other: Bp, w, phase };
}
export const PHASE_T = { surface: 0, blow: BLOWS[0] - 1.5, blow2: BLOWS[1] - 1.5, blow3: BLOWS[2] - 1.5, dive: T_DIVE + 1.5, flukes: T_DIVE + 3.8, deep: T_DIVE + 14, breach: T_EXIT - 3.5, air: T_EXIT + 0.7, splash: T_EXIT + 1.85 };
export { T_DIVE };
