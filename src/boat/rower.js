// The rower: a seated man in an oilskin jacket and sou'wester, sculpted as signed distance fields
// and polygonised once at load (surface nets), so the body is one continuous smooth surface.
// The body is skinned (linear blend, 14 bones, weights from the sculpt's own primitives); the head
// and hands are separate finer meshes carried rigidly by the head bone and the oar grips (the
// collar and the oilskin cuffs hide the joins).
//
// Rowing kinematics (fixed-seat): the body drives the oar, not the reverse. For each stroke phase
// the legs (seat slide + heels), the back (lean) and the arms (elbow bend) follow the classic
// sequence - drive: legs, then back, then arms; recovery: arms away, body over, then legs - and the
// oar angle is solved so that the grip lies where the hands can reach. Grips, rowlocks and blades
// follow from it, so the blade path in the water is the body's motion seen through the oar lever.
import {
  sub3, add3, sc3, dot3, cross3, len3, norm3, perp3, rot3, lerp3, madd3, clamp, sstep,
  frameMat, invRigid, xform, xdir, ik, surfaceNets, cleanTris, makeSkinBuilder, uploadArrays,
  smin, smax, sdRoundCone, sdCapsule, sdEllipsoid, sdBox,
} from './mesh.js';
import { innerHalfWidth, bottomY, rowlockPivot, LIN } from './hull.js';

const deg = Math.PI / 180;
const ease = (x) => { const t = clamp(x, 0, 1); return t * t * (3 - 2 * t); };
const mix = (a, b, t) => a + (b - a) * t;
const rotZ = (v, a) => [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a), v[2]];

// anthropometry (m)
const THIGH = 0.44, SHIN = 0.43, UPPER = 0.295, FORE = 0.265, SPINE = 0.21, CHEST = 0.26;
const HIP_W = 0.088, SHOULDER_W = 0.183;
export const BONES = ['pelvis', 'spine', 'chest', 'head', 'upperR', 'foreR', 'upperL', 'foreL', 'thighR', 'shinR', 'footR', 'thighL', 'shinL', 'footL'];
const B = Object.fromEntries(BONES.map((n, i) => [n, i]));
const sideKey = (s) => (s < 0 ? 'R' : 'L');   // the rower faces -x: his right hand is on the -z side

// oar grip in the hand frame (x: back of the hand, y: wrist -> knuckles, z: across the palm)
const GRIP_C = [-0.031, 0.083, 0], GRIP_R = 0.0168;
const GRIP_AT = LIN - 0.078;                 // grip centre: distance from the rowlock pivot

// foot stretcher (must match fittingsMesh in hull.js)
export const STRETCH_A = 42 * deg, STRETCH_X = -0.5;
export function stretcher() {
  const xs = STRETCH_X, y0 = bottomY(xs, 0.2) + 0.06;
  const w = [-Math.cos(STRETCH_A), Math.sin(STRETCH_A), 0];     // up the board (its top leans aft)
  const n = [Math.sin(STRETCH_A), Math.cos(STRETCH_A), 0];      // its after face (towards the rower)
  const c = add3([xs, y0, 0], sc3(w, 0.06));
  return { c, w, n };
}

// ---------------------------------------------------------------- stroke sequence
// phase p in [0, 1): 0 = the catch (blade just buried), drive to 0.36, release to 0.43, recovery
export function strokeSeq(p) {
  const LEAN_C = 25 * deg, LEAN_F = -15 * deg;
  const legs = p < 0.36 ? ease(p / 0.2) : p < 0.66 ? 1 : p < 0.9 ? 1 - ease((p - 0.66) / 0.24) : 0;
  const lean = p < 0.36 ? mix(LEAN_C, LEAN_F, ease((p - 0.05) / 0.25)) : p < 0.45 ? LEAN_F : p < 0.76 ? mix(LEAN_F, LEAN_C, ease((p - 0.45) / 0.31)) : LEAN_C;
  const arms = p < 0.36 ? 1 - ease((p - 0.17) / 0.19) : p < 0.4 ? 0 : p < 0.58 ? ease((p - 0.4) / 0.18) : 1;
  const DIN = 25 * deg, DOUT = 14 * deg;
  let dip;
  if (p < 0.35) dip = DIN;
  else if (p < 0.43) dip = mix(DIN, DOUT, ease((p - 0.35) / 0.08));
  else if (p < 0.9) dip = DOUT + 0.012 * Math.sin((p - 0.43) / 0.47 * Math.PI);
  else dip = mix(DOUT, DIN, ease((p - 0.9) / 0.085));
  const feather = p < 0.37 ? 0 : p < 0.45 ? ease((p - 0.37) / 0.08) : p < 0.8 ? 1 : p < 0.9 ? 1 - ease((p - 0.8) / 0.1) : 0;
  return { legs, lean, arms, dip, feather };
}

// oar direction (pivot -> blade) for sweep angle th (+ = blade towards the bow) and dip
export const oarDir = (th, dip, s) => [Math.sin(th) * Math.cos(dip), -Math.sin(dip), s * Math.cos(th) * Math.cos(dip)];

// hand frame on a grip: x (back of the hand) up and rolled with the feather, y towards the stern
function handFrame(grip, dir, roll) {
  let X = perp3([0, 1, 0], dir);
  X = rot3(X, dir, roll);
  let Z = dir;
  let Y = cross3(Z, X);
  if (Y[0] > 0) { Z = sc3(dir, -1); Y = cross3(Z, X); }
  // wrist: the grip sits under the palm at GRIP_C
  const o = sub3(grip, add3(sc3(X, GRIP_C[0]), sc3(Y, GRIP_C[1])));
  return { o, X, Y, Z };
}

// Full pose at phase p. Returns joint frames (boat-local), hand frames, oars, shadow capsules.
export function solvePose(p) {
  const q = strokeSeq(p);
  const lean = q.lean;
  // pelvis slides a little on the thwart with the leg drive; it rolls with the back
  const H = [0.03 + 0.07 * q.legs, 0.232, 0];
  const pel = { o: H, a: 0.45 * lean + 6 * deg };
  const fr = {};
  const torso = (o, a) => { const X = rotZ([-1, 0, 0], a), Y = rotZ([0, 1, 0], a); return { o, X, Y, Z: cross3(X, Y) }; };
  fr.pelvis = torso(H, pel.a);
  const L5 = add3(H, add3(sc3(fr.pelvis.Y, 0.09), sc3(fr.pelvis.X, -0.035)));
  fr.spine = torso(L5, 0.7 * lean + 3 * deg);
  const T12 = madd3(L5, fr.spine.Y, SPINE);
  fr.chest = torso(T12, lean);
  const C7 = madd3(T12, fr.chest.Y, CHEST);
  const neckDir = rotZ([0, 1, 0], 0.5 * lean + 14 * deg);
  fr.head = torso(add3(add3(C7, sc3(fr.chest.X, 0.025)), sc3(neckDir, 0.095)), 0.22 * lean + 2 * deg);
  // shoulders reach forward (protraction) at the catch and draw back at the finish
  const protract = 0.04 * q.arms * sstep(-5 * deg, 25 * deg, lean) - 0.022 * (1 - q.arms);
  const bend = (12 + (1 - q.arms) * 112) * deg;
  const reach = Math.sqrt(UPPER * UPPER + FORE * FORE - 2 * UPPER * FORE * Math.cos(Math.PI - bend));
  const hands = {}, oars = {};
  for (const s of [-1, 1]) {
    const S = add3(add3(madd3(T12, fr.chest.Y, 0.205), sc3(fr.chest.X, protract)), [0, 0, s * SHOULDER_W]);
    const pk = rowlockPivot(s);
    const roll = -q.feather * 42 * deg * s;
    const wristAt = (th) => {
      const dir = oarDir(th, q.dip, s);
      return handFrame(madd3(pk, dir, -GRIP_AT), dir, roll);
    };
    // the sweep angle at which the wrist is `reach` from the shoulder: the distance has a minimum
    // (the grips pass closest to the body), the stroke uses the branch towards the stern from it
    const g = (th) => len3(sub3(wristAt(th).o, S)) - reach;
    let a = -60 * deg, c = 62 * deg;
    for (let i = 0; i < 40; i++) {
      const m1 = a + (c - a) * 0.382, m2 = a + (c - a) * 0.618;
      if (g(m1) < g(m2)) c = m2; else a = m1;
    }
    let lo = (a + c) / 2, hi = 62 * deg, th;
    if (g(lo) > 0) th = lo; else if (g(hi) < 0) th = hi;
    else { for (let i = 0; i < 26; i++) { const m = (lo + hi) / 2; if (g(m) > 0) hi = m; else lo = m; } th = (lo + hi) / 2; }
    const hf = wristAt(th);
    const W = hf.o;
    const E = ik(S, W, UPPER, FORE, [0.6, -0.75, s * 0.55]);
    const k = sideKey(s);
    const Yu = norm3(sub3(E, S)), Yf = norm3(sub3(W, E));
    let Zb = cross3(Yu, Yf);
    Zb = len3(Zb) > 1e-4 ? norm3(Zb) : [0, 0, -1];
    fr['upper' + k] = { o: S, Y: Yu, Z: Zb, X: cross3(Yu, Zb) };
    fr['fore' + k] = { o: E, Y: Yf, Z: Zb, X: cross3(Yf, Zb) };
    hands[k] = hf;
    // oar frame: y along the oar (pivot -> blade), z the blade's width (vertical when square)
    const dir = oarDir(th, q.dip, s);
    const hz = norm3(cross3([0, 1, 0], dir)), vq = cross3(dir, hz);
    const fe = q.feather * 88 * deg;
    const Wd = add3(sc3(vq, Math.cos(fe)), sc3(hz, Math.sin(fe) * s));
    oars[k] = { pivot: pk, dir, Wd, X: cross3(dir, Wd), th, dip: q.dip, side: s };
    // legs: the ball of the foot on the stretcher, heels lift towards the catch
    const st = stretcher();
    const lift = (1 - q.legs) * 20 * deg;
    const ball = add3(add3(st.c, sc3(st.w, 0.025)), [0, 0, s * 0.112]);
    const fy = rot3(st.w, [0, 0, 1], lift), fn = rot3(st.n, [0, 0, 1], lift);
    const A = add3(add3(ball, sc3(fn, 0.072)), sc3(fy, -0.15));
    const Hj = add3(H, [0, -0.005, s * HIP_W]);
    const K = ik(Hj, A, THIGH, SHIN, [-0.45, 1, s * 0.22]);
    const Yt = norm3(sub3(K, Hj)), Ys = norm3(sub3(A, K));
    let Zl = cross3(Yt, Ys);
    Zl = len3(Zl) > 1e-4 ? norm3(Zl) : [0, 0, 1];
    fr['thigh' + k] = { o: Hj, Y: Yt, Z: Zl, X: cross3(Yt, Zl) };
    fr['shin' + k] = { o: K, Y: Ys, Z: Zl, X: cross3(Ys, Zl) };
    const Zf = perp3(Zl, fy);
    fr['foot' + k] = { o: A, Y: fy, Z: Zf, X: cross3(fy, Zf) };
  }
  // shadow / occlusion capsules (a, b, radius): kept inside the skin
  const caps = [];
  const cap = (a, b, r) => caps.push([a, b, r]);
  cap(madd3(H, fr.pelvis.Y, 0.04), madd3(T12, fr.chest.Y, 0.16), 0.1);
  cap(fr.head.o, madd3(fr.head.o, fr.head.Y, 0.14), 0.085);
  for (const k of ['R', 'L']) {
    cap(fr['upper' + k].o, fr['fore' + k].o, 0.045);
    cap(fr['fore' + k].o, hands[k].o, 0.038);
    cap(fr['thigh' + k].o, fr['shin' + k].o, 0.06);
    cap(fr['shin' + k].o, fr['foot' + k].o, 0.045);
  }
  return { fr, hands, oars, caps, seq: q, p, dbg: { reach } };
}

// ---------------------------------------------------------------- sculpt
// primitives: { bone, mat, g (group), f(pLocal) -> distance }
const P = (bone, mat, g, f) => ({ bone, mat, g, f });
// every primitive carries a bounding sphere (bs: centre, radius) so evaluation can skip far ones
const withBS = (f, c, r) => { f.bs = [c[0], c[1], c[2], r]; return f; };
const rc = (a, b, r1, r2) => withBS((p) => sdRoundCone(p, a, b, r1, r2), lerp3(a, b, 0.5), len3(sub3(b, a)) / 2 + Math.max(r1, r2));
const el = (c, r) => withBS((p) => sdEllipsoid(p, c, r), c, Math.max(...r));
const cp = (a, b, r) => withBS((p) => sdCapsule(p, a, b, r), lerp3(a, b, 0.5), len3(sub3(b, a)) / 2 + r);
const bx = (c, h, r) => withBS((p) => sdBox(p, c, h, r), c, len3(h));
const far = (pr, p, lim) => {
  const dx = p[0] - pr.wc[0], dy = p[1] - pr.wc[1], dz = p[2] - pr.wc[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz) - pr.wr > lim;
};
// smooth union of a primitive list (skipping primitives that cannot change the result)
// Per 3 cm cell, only the primitives that can reach the union there are kept (from exact distances at
// the cell centre and the distance fields' 1-Lipschitz bound, with a margin for the ellipsoids).
const CELL = 0.03, CELL_R = CELL * 0.866;
function unionPruned(prims, p, k, locAt) {
  const cache = prims.cache || (prims.cache = new Map());
  const ix = Math.floor(p[0] / CELL), iy = Math.floor(p[1] / CELL), iz = Math.floor(p[2] / CELL);
  const key = ((ix + 512) * 1024 + (iy + 512)) * 1024 + (iz + 512);
  let list = cache.get(key);
  if (!list) {
    const c = [(ix + 0.5) * CELL, (iy + 0.5) * CELL, (iz + 0.5) * CELL];
    const loc = locAt ? locAt(c) : null;
    const ds = prims.map((pr) => pr.f(loc ? loc(pr.bone) : c));
    const dm = Math.min(...ds);
    list = prims.filter((pr, i) => ds[i] < dm + 3 * CELL_R + k);
    cache.set(key, list);
  }
  const loc = locAt ? locAt(p) : null;
  let d = 1e3;
  for (const pr of list) {
    if (far(pr, p, d + k)) continue;
    d = smin(d, pr.f(loc ? loc(pr.bone) : p), k);
  }
  return d;
}
const bsWorld = (pr, m) => { pr.wc = m ? xform(m, pr.f.bs) : pr.f.bs.slice(0, 3); pr.wr = pr.f.bs[3]; return pr; };

function bodyPrims() {
  const L = [];
  // jacket (group 0): torso, collar, sleeves with wide cuffs
  // a loose smock: boxy, hanging from the shoulders, flaring over the hips
  L.push(P(B.pelvis, 1, 0, bx([0.0, 0.075, 0], [0.12, 0.05, 0.165], 0.06)));
  L.push(P(B.spine, 1, 0, bx([0.008, 0.1, 0], [0.118, 0.13, 0.168], 0.08)));
  L.push(P(B.chest, 1, 0, bx([0.004, 0.1, 0], [0.12, 0.13, 0.172], 0.085)));
  L.push(P(B.chest, 1, 0, cp([-0.005, 0.2, -0.135], [-0.005, 0.2, 0.135], 0.062)));   // across the shoulders
  L.push(P(B.chest, 1, 0, rc([0.008, 0.235, 0], [0.016, 0.325, 0], 0.084, 0.076)));    // collar
  for (const k of ['R', 'L']) {
    L.push(P(B['upper' + k], 1, 4, rc([0, -0.01, 0], [0, UPPER, 0], 0.064, 0.055)));
    L.push(P(B['fore' + k], 1, 4, rc([0, 0, 0], [0, FORE - 0.04, 0], 0.055, 0.047)));
    L.push(P(B['fore' + k], 1, 4, rc([0, FORE - 0.06, 0], [0, FORE + 0.022, 0], 0.046, 0.046)));   // cuff
  }
  // trousers (group 1): seat, thighs, knees, calves (tucked into the boots)
  L.push(P(B.pelvis, 2, 1, el([0.0, 0.0, 0], [0.125, 0.095, 0.165])));
  for (const s of [-1, 1]) L.push(P(B.pelvis, 2, 1, el([-0.055, -0.02, -s * 0.075], [0.095, 0.075, 0.085])));
  for (const k of ['R', 'L']) {
    L.push(P(B['thigh' + k], 2, 1, rc([0, 0.0, 0], [0, THIGH, 0], 0.088, 0.058)));
    L.push(P(B['thigh' + k], 2, 1, el([0.018, 0.17, 0], [0.072, 0.17, 0.078])));
    L.push(P(B['shin' + k], 2, 1, el([0.022, 0.0, 0], [0.055, 0.058, 0.06])));
    L.push(P(B['shin' + k], 2, 1, rc([0, 0, 0], [0, SHIN * 0.6, 0], 0.057, 0.05)));
    L.push(P(B['shin' + k], 2, 1, el([-0.028, 0.13, 0], [0.055, 0.1, 0.053])));
  }
  // boots (group 2): shafts and feet with heel and sole
  for (const k of ['R', 'L']) {
    L.push(P(B['shin' + k], 3, 2, rc([0, SHIN * 0.5, 0], [0, SHIN + 0.01, 0], 0.06, 0.05)));
    L.push(P(B['foot' + k], 3, 2, bx([-0.022, 0.045, 0], [0.042, 0.125, 0.046], 0.034)));
    L.push(P(B['foot' + k], 3, 2, el([-0.02, 0.15, 0], [0.04, 0.07, 0.05])));
    L.push(P(B['foot' + k], 3, 2, bx([-0.066, 0.05, 0], [0.008, 0.135, 0.052], 0.008)));   // sole
  }
  // skin (group 3): the neck inside the collar
  L.push(P(B.chest, 0, 3, rc([0.012, 0.22, 0], [0.02, 0.3, 0], 0.058, 0.055)));
  L.push(P(B.head, 0, 3, rc([0.0, -0.06, 0], [0.005, 0.03, 0], 0.056, 0.05)));
  return L;
}
// holes: the collar opening and the cuff openings
function bodyHoles() {
  const H = [P(B.chest, -1, -1, rc([0.016, 0.3, 0], [0.024, 0.4, 0], 0.06, 0.061))];
  for (const k of ['R', 'L']) H.push(P(B['fore' + k], -1, -1, cp([0, FORE - 0.0, 0], [0, FORE + 0.09, 0], 0.034)));
  return H;
}

function headPrims() {
  const L = [];
  // skull, face, jaw (group 0, skin)
  L.push(P(0, 0, 0, el([-0.006, 0.112, 0], [0.097, 0.1, 0.077])));
  L.push(P(0, 0, 0, el([0.038, 0.122, 0], [0.062, 0.07, 0.068])));
  L.push(P(0, 0, 0, el([0.045, 0.057, 0], [0.054, 0.066, 0.058])));
  for (const s of [-1, 1]) {
    L.push(P(0, 0, 0, cp([-0.006, 0.05, s * 0.05], [0.066, 0.004, s * 0.017], 0.016)));
    L.push(P(0, 0, 0, el([0.062, 0.073, s * 0.046], [0.022, 0.016, 0.024])));
    L.push(P(0, 0, 0, el([0.005, 0.06, s * 0.058], [0.04, 0.045, 0.02])));
  }
  L.push(P(0, 0, 0, el([0.075, 0.002, 0], [0.021, 0.022, 0.027])));
  L.push(P(0, 0, 0, cp([0.084, 0.104, -0.043], [0.084, 0.104, 0.043], 0.011)));
  L.push(P(0, 0, 0, rc([0.09, 0.094, 0], [0.117, 0.052, 0], 0.0085, 0.0125)));
  for (const s of [-1, 1]) L.push(P(0, 0, 0, el([0.102, 0.049, s * 0.013], [0.012, 0.009, 0.0095])));
  L.push(P(0, 0, 0, el([0.095, 0.03, 0], [0.012, 0.0095, 0.027])));
  L.push(P(0, 0, 0, el([0.092, 0.013, 0], [0.011, 0.008, 0.023])));
  for (const s of [-1, 1]) {
    L.push(P(0, 0, 0, el([0.084, 0.093, s * 0.033], [0.012, 0.007, 0.017])));    // upper lid
    L.push(P(0, 0, 0, el([-0.008, 0.08, s * 0.075], [0.029, 0.034, 0.0095])));   // ear
  }
  L.push(P(0, 0, 0, rc([-0.012, -0.08, 0], [0.0, 0.045, 0], 0.057, 0.05)));       // neck
  return L;
}
function headCuts() {
  const C = [];
  for (const s of [-1, 1]) {
    C.push({ k: 0.012, f: el([0.096, 0.086, s * 0.033], [0.02, 0.0145, 0.019]) });      // eye sockets
    C.push({ k: 0.004, f: el([-0.004, 0.079, s * 0.083], [0.018, 0.022, 0.006]) });     // concha
  }
  C.push({ k: 0.003, f: cp([0.106, 0.0215, -0.021], [0.106, 0.0215, 0.021], 0.0022) });  // mouth line
  return C;
}
// sou'wester: crown and a brim that is narrow in front and long at the back, drooping
function hatSDF(p) {
  // tilt: back of the hat down a little
  const a = -6 * deg;
  const x = p[0] * Math.cos(a) + p[1] * Math.sin(a), y = -p[0] * Math.sin(a) + p[1] * Math.cos(a), z = p[2];
  // crown, a little peaked
  const cr = sdEllipsoid([x, y, z], [-0.01, 0.12, 0], [0.105, 0.1, 0.092]);
  const crown = smax(cr, 0.098 - y, 0.012);
  const rho = Math.sqrt((x + 0.01) * (x + 0.01) + z * z), th = Math.atan2(z, x + 0.01);
  const back = sstep(0.1, -1, Math.cos(th));
  // brim: ~5 cm in front (turned up a little), long and drooping at the back to shed the rain
  const R = 0.152 + 0.075 * back;
  const yb = 0.1 - (rho - 0.09) * (0.22 + 0.7 * back) + 0.12 * Math.max(rho - 0.1, 0) * (1 - back) * sstep(0.3, 1, Math.cos(th)) - 0.9 * Math.max(rho - 0.13, 0) ** 2 * back;
  const brim = smax(Math.abs(y - yb) - 0.0032, rho - R, 0.005);
  const band = sdRoundCone([x, y, z], [-0.01, 0.094, 0], [-0.01, 0.108, 0], 0.097, 0.095);
  return smin(smin(crown, brim, 0.012), band, 0.003);
}

function handPrims(t) {
  // t: +1 thumb towards +z (hand-local)
  const L = [];
  L.push({ f: bx([0.0, 0.048, -t * 0.002], [0.012, 0.042, 0.038], 0.012) });
  L.push({ f: el([0.0, 0.0, 0], [0.021, 0.04, 0.03]) });
  L.push({ f: el([-0.011, 0.034, t * 0.024], [0.015, 0.03, 0.018]) });                    // thenar
  L.push({ f: el([-0.009, 0.042, -t * 0.028], [0.012, 0.03, 0.013]) });                  // hypothenar
  const C = GRIP_C, R = GRIP_R;
  const fingers = [[t * 0.026, 0.092, 0.95, 0.94], [t * 0.008, 0.095, 1, 1], [-t * 0.011, 0.091, 0.95, 0.94], [-t * 0.028, 0.084, 0.78, 0.84]];
  for (const [z, y, ls, rs] of fingers) {
    const lens = [0.046 * ls, 0.028 * ls, 0.023 * ls], rads = [0.0098 * rs, 0.0088 * rs, 0.008 * rs];
    let prev = [0.002, y, z];
    let ang = Math.atan2(prev[1] - C[1], prev[0] - C[0]);
    const Rf = R + 0.0088 * rs;
    L.push({ f: el([0.006, y - 0.003, z], [0.008, 0.009, 0.009]) });                      // knuckle
    for (let k = 0; k < 3; k++) {
      // next joint on the circle of radius Rf round the grip at chord length lens[k]
      let a0 = ang, a1 = ang - Math.PI;
      for (let i = 0; i < 30; i++) {
        const m = (a0 + a1) / 2, q = [C[0] + Rf * Math.cos(m), C[1] + Rf * Math.sin(m), z];
        if (len3(sub3(q, prev)) < lens[k]) a0 = m; else a1 = m;
      }
      ang = (a0 + a1) / 2;
      const q = [C[0] + Rf * Math.cos(ang), C[1] + Rf * Math.sin(ang), z];
      const r2 = k < 2 ? rads[k + 1] : rads[k] * 0.85;
      L.push({ f: rc(prev, q, rads[k], r2) });
      prev = q;
    }
  }
  // thumb: from the base of the palm round under the grip
  const th = [[-0.004, 0.012, t * 0.03], [-0.026, 0.05, t * 0.043], [-0.044, 0.07, t * 0.04], [-0.055, 0.088, t * 0.03]];
  const tr = [0.014, 0.0115, 0.0102, 0.0088];
  for (let k = 0; k < 3; k++) L.push({ f: rc(th[k], th[k + 1], tr[k], tr[k + 1]) });
  return L;
}

// ---------------------------------------------------------------- mesh generation
function frameOf(f) { return frameMat(f.o, f.X, f.Y, f.Z); }

function buildBody(bind) {
  const prims = bodyPrims(), holes = bodyHoles();
  const inv = BONES.map((n) => invRigid(frameOf(bind.fr[n])));
  const KG = [0.07, 0.04, 0.02, 0.03, 0.035];
  const bindM = BONES.map((n) => frameOf(bind.fr[n]));
  for (const pr of prims.concat(holes)) bsWorld(pr, bindM[pr.bone]);
  const byGroup = [0, 1, 2, 3, 4].map((g) => prims.filter((pr) => pr.g === g));
  const local = (p) => inv.map((m) => xform(m, p));
  // lazily transformed local points (one per bone), memoised for the last point asked
  let lastP = null, lastLoc = null;
  const locAt = (q) => {
    if (q === lastP) return lastLoc;
    const cache = new Array(BONES.length);
    lastP = q; lastLoc = (i) => cache[i] || (cache[i] = xform(inv[i], q));
    return lastLoc;
  };
  const groups = (lp) => {
    const g = [1e3, 1e3, 1e3, 1e3, 1e3];
    for (const pr of prims) g[pr.g] = smin(g[pr.g], pr.f(lp[pr.bone]), KG[pr.g]);
    g[0] = smin(g[0], g[4], 0.022);
    return g;
  };
  const sdf = (p) => {
    const loc = locAt(p);
    const g = byGroup.map((L, i) => unionPruned(L, p, KG[i], locAt));
    g[0] = smin(g[0], g[4], 0.022);      // sleeves set into the smock with a small seam fillet
    let d = smin(g[0], g[1], 0.008);      // the jacket hem over the trousers: a crease, not a blend
    d = smin(d, g[2], 0.006);
    for (const h of holes) if (!far(h, p, -d + 0.012)) d = smax(d, -h.f(loc(h.bone)), 0.012);
    return smin(d, g[3], 0.02);
  };
  // bounds from the bind joints
  const pts = Object.values(bind.fr).map((f) => f.o);
  const lo = [0, 1, 2].map((a) => Math.min(...pts.map((q) => q[a])) - 0.2);
  const hi = [0, 1, 2].map((a) => Math.max(...pts.map((q) => q[a])) + 0.2);
  hi[1] += 0.05;
  const m = surfaceNets(sdf, lo, hi, 0.0085);
  const tris = cleanTris(m.pos, m.tris);
  // per vertex: material weights, bone weights, folds
  const b = makeSkinBuilder();
  for (let v = 0; v < m.pos.length; v++) {
    const p = m.pos[v], lp = local(p);
    const g = groups(lp);
    const gm = Math.min(...g);
    // material weights: skin, jacket, trousers, boots
    const mw = [g[3], g[0], g[1], g[2]].map((d) => Math.exp(-(d - gm) / 0.004));
    const ms = mw.reduce((a, c) => a + c, 0);
    // bone weights: soft minimum over each bone's primitives
    const bd = new Array(BONES.length).fill(1e3);
    for (const pr of prims) bd[pr.bone] = Math.min(bd[pr.bone], pr.f(lp[pr.bone]));
    const dm = Math.min(...bd);
    const ranked = bd.map((d, i) => [Math.exp(-(d - dm) / 0.028), i]).sort((a, c) => c[0] - a[0]).slice(0, 4);
    const ws = ranked.reduce((a, c) => a + c[0], 0);
    const bi = ranked.map((r) => r[1]), bw = ranked.map((r) => r[0] / ws);
    // folds: along / around the dominant bone, strength by material
    const f = bind.fr[BONES[bi[0]]], d = sub3(p, f.o);
    const along = dot3(d, f.Y), around = Math.atan2(dot3(d, f.Z), dot3(d, f.X));
    const strength = (mw[1] * 1.0 + mw[2] * 0.75 + mw[3] * 0.25) / ms + (bi[0] <= B.chest ? 2 : 0);   // +2: torso
    b.v(p, m.nrm[v], [along, around, strength], f.Y, mw.map((w) => w / ms), bi, bw);
  }
  for (const t of tris) b.I.push(t);
  return b;
}

function buildRigid(sdf, classify, lo, hi, h, axis) {
  const m = surfaceNets(sdf, lo, hi, h);
  const tris = cleanTris(m.pos, m.tris);
  const b = makeSkinBuilder(false);
  for (let v = 0; v < m.pos.length; v++) {
    const p = m.pos[v];
    const mw = classify(p);
    const d = sub3(p, axis.o);
    b.v(p, m.nrm[v], [dot3(d, axis.Y), Math.atan2(d[2], d[0]), 0], axis.Y, mw);
  }
  for (const t of tris) b.I.push(t);
  return b;
}

function buildHead() {
  const prims = headPrims().map((pr) => bsWorld(pr, null)), cuts = headCuts().map((c) => bsWorld(c, null));
  const skin = (p) => {
    let d = unionPruned(prims, p, 0.022, null);
    for (const c of cuts) if (!far(c, p, -d + c.k)) d = smax(d, -c.f(p), c.k);
    return d;
  };
  const eyes = (p) => Math.min(sdEllipsoid(p, [0.079, 0.086, -0.033], [0.0125, 0.0125, 0.0125]), sdEllipsoid(p, [0.079, 0.086, 0.033], [0.0125, 0.0125, 0.0125]));
  const sdf = (p) => Math.min(smin(skin(p), hatSDF(p), 0.003), eyes(p));
  const classify = (p) => {
    const ds = skin(p), dh = hatSDF(p), de = eyes(p), m = Math.min(ds, dh, de);
    const w = [ds, dh, de, 1e3].map((d) => Math.exp(-(d - m) / 0.0015));
    const s = w.reduce((a, c) => a + c, 0);
    return w.map((x) => x / s);
  };
  return buildRigid(sdf, classify, [-0.22, -0.13, -0.21], [0.21, 0.26, 0.21], 0.003, { o: [0, 0, 0], Y: [0, 1, 0] });
}

function buildHand(t) {
  const prims = handPrims(t).map((pr) => bsWorld(pr, null));
  const sdf = (p) => {
    let d = unionPruned(prims, p, 0.006, null);
    // the grip: nothing goes through it (the fingers and palm wrap it)
    const gc = Math.sqrt((p[0] - GRIP_C[0]) ** 2 + (p[1] - GRIP_C[1]) ** 2) - (GRIP_R + 0.0004);
    d = smax(d, -gc, 0.003);
    // cut at the wrist (inside the cuff)
    return smax(d, -0.035 - p[1], 0.004);
  };
  return buildRigid(sdf, () => [1, 0, 0, 0], [-0.085, -0.045, -0.075], [0.04, 0.135, 0.075], 0.0022, { o: [0, 0, 0], Y: [0, 1, 0] });
}

// ---------------------------------------------------------------- the rower
const BIND_P = 0.22;
// all four meshes as typed arrays (run in a worker: the sculpt takes a few seconds)
export function buildRowerArrays() {
  const t0 = performance.now();
  const bind = solvePose(BIND_P);
  const out = { body: buildBody(bind).arrays(), head: buildHead().arrays(), handR: buildHand(1).arrays(), handL: buildHand(-1).arrays() };
  out.ms = performance.now() - t0;
  return out;
}

export class Rower {
  constructor(gl, { sync = false } = {}) {
    this.gl = gl;
    this.bind = solvePose(BIND_P);
    this.bindInv = BONES.map((n) => invRigid(frameOf(this.bind.fr[n])));
    this.palette = new Float32Array(BONES.length * 8);   // dual quaternions (real, dual)
    this.ready = false;
    const done = (a) => {
      this.body = uploadArrays(gl, a.body);
      this.head = uploadArrays(gl, a.head);
      this.handR = uploadArrays(gl, a.handR);
      this.handL = uploadArrays(gl, a.handL);
      this.buildMs = a.ms;
      this.ready = true;
    };
    let worker = null;
    if (!sync && typeof Worker !== 'undefined') {
      try { worker = new Worker(new URL('./rowerWorker.js', import.meta.url), { type: 'module' }); } catch { worker = null; }
    }
    if (worker) {
      worker.onmessage = (e) => { done(e.data); worker.terminate(); };
      worker.onerror = (e) => { e.preventDefault?.(); worker.terminate(); done(buildRowerArrays()); };
      worker.postMessage('build');
    } else done(buildRowerArrays());
  }

  // skinning palette and rigid frames for a pose (boat-local)
  pose(P) {
    for (let i = 0; i < BONES.length; i++) {
      const M = mulRigid(frameOf(P.fr[BONES[i]]), this.bindInv[i]);
      this.palette.set(dualQuat(M), i * 8);
    }
    return {
      palette: this.palette,
      head: frameOf(P.fr.head),
      handR: frameOf(P.hands.R),
      handL: frameOf(P.hands.L),
    };
  }
}

// rigid transform -> dual quaternion [qx, qy, qz, qw, dx, dy, dz, dw] (skinning without the
// volume loss of blended matrices at elbows and knees)
function dualQuat(m) {
  const m00 = m[0], m01 = m[4], m02 = m[8], m10 = m[1], m11 = m[5], m12 = m[9], m20 = m[2], m21 = m[6], m22 = m[10];
  let x, y, z, w;
  const tr = m00 + m11 + m22;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s; }
  else if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1 + m00 - m11 - m22) * 2; w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s; }
  else if (m11 > m22) { const s = Math.sqrt(1 + m11 - m00 - m22) * 2; w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s; }
  else { const s = Math.sqrt(1 + m22 - m00 - m11) * 2; w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s; }
  const t = [m[12], m[13], m[14]];
  // dual = 0.5 (t, 0) q
  const dx = 0.5 * (w * t[0] + t[1] * z - t[2] * y), dy = 0.5 * (w * t[1] + t[2] * x - t[0] * z), dz = 0.5 * (w * t[2] + t[0] * y - t[1] * x);
  const dw = -0.5 * (t[0] * x + t[1] * y + t[2] * z);
  return [x, y, z, w, dx, dy, dz, dw];
}

function mulRigid(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}

// the hand frame's thumb side must be inboard on the grip: the mesh for each hand is chosen in boat.js
export { GRIP_C, GRIP_R, xdir, lerp3, innerHalfWidth };
