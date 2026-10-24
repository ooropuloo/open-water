// A clinker-built wooden rowing boat and its rower, floating on the FFT sea.
//
// Files: hull.js (hull lines, planking, fittings, oars, rowlocks), rower.js (sculpted, skinned rower
// and the rowing kinematics), shade.js (materials and lighting), wake.js (the boat's mark on the
// sea: wake, puddles, contact foam, spray).
//
// Flotation: five points (centre, bow, stern, both sides) are sampled from the same FFT displacement
// cascades the water mesh uses, at the mip level the water mesh itself uses at the boat's distance,
// with the choppy (horizontal) displacement inverted, and read back asynchronously (PBO + fence).
// Heave, pitch and roll follow those heights through damped springs (the hull's inertia).
//
// Drawing: the boat is drawn before the water, then bilge water (blended), then a depth-only lid
// over the hull opening keeps the sea out of the hull; the water drawn afterwards hides the
// submerged hull and the blades in the water. The sea shader reads the wake field (waterFx());
// spray is drawn after the sea (drawFx()).
import { Program, Texture, Target, fullscreen, FULLSCREEN_VS } from '../gl.js';
import { mul } from '../math.js';
import { hullMesh, fittingsMesh, oarMesh, rowlockMesh, lidMesh, bilgeMesh, LOUT, BLADE, XB, XS } from './hull.js';
import { Rower, solvePose } from './rower.js';
import { VS_PROP, FS_PROP, VS_CHAR, FS_CHAR, VS_BILGE, FS_BILGE, NCAPS } from './shade.js';
import { BoatWake } from './wake.js';
import { frameMat, invRigid, normalMat, xform, xdir, norm3, cross3, add3, sc3, sub3, madd3 } from './mesh.js';

// probe: heights of the displaced sea at a few points (choppy displacement inverted)
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

const NP = 5;
class BoatProbe {
  constructor(gl) {
    this.gl = gl;
    this.prog = new Program(gl, FULLSCREEN_VS, PROBE_FS, { label: 'boat-probe' });
    this.tex = new Texture(gl, { w: NP, h: 1, format: 'rgba32f', filter: 'nearest' });
    this.target = new Target(gl, [this.tex]);
    this.pbo = gl.createBuffer();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, NP * 16, gl.STREAM_READ);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    this.data = new Float32Array(NP * 4);
    this.heights = null;
    this.fence = null;
  }
  // t: simulation time; the heights returned are those of time this.sampleT (the readback lags)
  run(ot, pts, footprint, sync, t = 0) {
    const gl = this.gl;
    if (this.fence && !sync) {
      if (gl.clientWaitSync(this.fence, 0, 0) === gl.TIMEOUT_EXPIRED) return this.heights;
      gl.deleteSync(this.fence);
      this.fence = null;
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.data);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      this.decode(this.issueT);
    }
    this.issueT = t;
    fullscreen(gl, this.prog, { uDisp: ot.disp.slice(0, 3), uSize: ot.sizes, uTexel: ot.texels, uPts: pts, uFootprint: footprint }, this.target);
    if (sync) {
      gl.readPixels(0, 0, NP, 1, gl.RGBA, gl.FLOAT, this.data);
      this.decode(t);
      return this.heights;
    }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
    gl.readPixels(0, 0, NP, 1, gl.RGBA, gl.FLOAT, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    this.fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    return this.heights;
  }
  decode(t) {
    const h = [];
    for (let i = 0; i < NP; i++) h.push(this.data[i * 4 + 1]);
    if (!h.every(Number.isFinite)) return;
    // rate of change between readbacks: lets the boat extrapolate across the readback latency
    if (this.heights && t > this.sampleT + 1e-4) this.rate = h.map((x, i) => Math.max(Math.min((x - this.heights[i]) / (t - this.sampleT), 1.2), -1.2));
    this.heights = h;
    this.sampleT = t;
  }
}

const RENDER = typeof location !== 'undefined' && new URLSearchParams(location.search).has('render');
const BLADE_MID = LOUT - BLADE * 0.5;

export class Boat {
  // anchor: camera base (xz); yaw: camera heading (rad) the boat is placed in front of; dist: metres
  constructor(gl, { visible = true, anchor = [0, 0], yaw = 0, dist = 45 } = {}) {
    this.gl = gl;
    this.visible = visible;
    this.prop = new Program(gl, VS_PROP, FS_PROP, { label: 'boat-prop' });
    this.skin = new Program(gl, VS_CHAR, FS_CHAR, { label: 'boat-rower', defines: { SKIN: 1 } });
    this.rigid = new Program(gl, VS_CHAR, FS_CHAR, { label: 'boat-rower-rigid' });
    this.bilgeProg = new Program(gl, VS_BILGE, FS_BILGE, { label: 'boat-bilge' });
    this.mesh = { hull: hullMesh(gl), fit: fittingsMesh(gl), oar: oarMesh(gl), lock: rowlockMesh(gl), lid: lidMesh(gl), bilge: bilgeMesh(gl) };
    this.rower = new Rower(gl, { sync: RENDER });
    this.wake = new BoatWake(gl);
    this.probe = new BoatProbe(gl);
    const f = [Math.cos(yaw), Math.sin(yaw)], r = [-Math.sin(yaw), Math.cos(yaw)];
    // an elliptical course in front of the camera: 2a across the view, 2b in depth
    this.path = { c: [anchor[0] + f[0] * dist, anchor[1] + f[1] * dist], f, r, a: 16, b: 9 };
    this.phi = -Math.PI / 2;           // start at the nearest point, crossing the view
    this.stroke = 0;                   // rowing phase (cycles)
    this.period = 3.3;                 // s per stroke (~18 strokes a minute)
    this.speed = 1.1;                  // m/s mean
    this.freeze = null;                // (debug) hold the stroke at this phase
    this.timeScale = 1;                // (debug) slow motion for the boat, rower and spray
    this.sink = 0.04;                  // the rower's weight: the hull floats below its design waterline
    this.time = 0;
    this.heave = { x: 0, v: 0 }; this.pitch = { x: 0, v: 0 }; this.roll = { x: 0, v: 0 };
    this.settled = false;
    this.pts = new Float32Array(NP * 2);
    this.caps = new Float32Array(NCAPS * 8);
    this.sea = { h: 0, gx: 0, gz: 0, x: 0, z: 0 };
    this.waterPlane = [0, 0, 0];
    this.blades = { R: { inWater: false, tOut: -10, pud: null, nextDrip: 0 }, L: { inWater: false, tOut: -10, pud: null, nextDrip: 0 } };
    this.update(0, null, null, 0, false);
  }

  pathAt(phi) {
    const { c, f, r, a, b } = this.path;
    const ca = Math.cos(phi), sa = Math.sin(phi);
    return {
      p: [c[0] + r[0] * a * ca + f[0] * b * sa, c[1] + r[1] * a * ca + f[1] * b * sa],
      d: [-r[0] * a * sa + f[0] * b * ca, -r[1] * a * sa + f[1] * b * ca],
    };
  }

  // sea height (world) near the boat, from the last probe (a plane through the samples)
  seaY(x, z) { const s = this.sea; return s.h + s.gx * (x - s.x) + s.gz * (z - s.z); }

  update(dt, ot, cam, angStep, sync) {
    if (!this.visible && ot) return;
    this.simT = (this.simT || 0) + Math.max(dt, 0);      // (sea time: not slowed down)
    dt = Math.max(dt, 0) * this.timeScale;   // (the live loop's first frame can carry a negative dt)
    this.time += dt;
    // advance along the course; the boat surges on each drive and runs down on the recovery
    const p0 = this.phase();
    const v = this.speed * (1 + 0.28 * Math.cos(2 * Math.PI * (p0 - 0.36)));
    this.vNow = v;
    const cur = this.pathAt(this.phi);
    this.phi += (v * dt) / Math.max(Math.hypot(cur.d[0], cur.d[1]), 1e-3);
    if (this.freeze === null) this.stroke += dt / this.period;
    const { p, d } = this.pathAt(this.phi);
    const fl = Math.hypot(d[0], d[1]) || 1;
    const fwd = [d[0] / fl, 0, d[1] / fl], side = [-fwd[2], 0, fwd[0]];
    this.pos2 = p; this.fwd = fwd; this.side = side;
    if (!ot) return;

    // float: sea heights at the centre, bow, stern and both sides
    const LB = 1.6, BS = 0.62;
    const P = this.pts;
    const put = (i, x, z) => { P[i * 2] = x; P[i * 2 + 1] = z; };
    put(0, p[0], p[1]);
    put(1, p[0] + fwd[0] * LB, p[1] + fwd[2] * LB);
    put(2, p[0] - fwd[0] * LB, p[1] - fwd[2] * LB);
    put(3, p[0] + side[0] * BS, p[1] + side[2] * BS);
    put(4, p[0] - side[0] * BS, p[1] - side[2] * BS);
    // filter the sea as the water mesh does at this distance (the boat floats on the drawn surface)
    const dist = cam ? Math.hypot(p[0] - cam.pos[0], p[1] - cam.pos[2]) : 40;
    const h0 = this.probe.run(ot, P, Math.max(dist * angStep * 1.5, 0.6), sync, this.simT);
    if (!h0) return;
    // (the sea moved on while the heights were read back: extrapolate them to now)
    const lag = Math.min(Math.max(this.simT - this.probe.sampleT, 0), 0.6);
    const h = this.probe.rate ? h0.map((x, i) => x + this.probe.rate[i] * lag) : h0;
    const heave = 0.4 * h[0] + 0.15 * (h[1] + h[2] + h[3] + h[4]);
    const pitch = Math.atan((h[1] - h[2]) / (2 * LB)) + 0.012 * Math.sin(2 * Math.PI * p0);
    const roll = Math.atan((h[3] - h[4]) / (2 * BS)) * 0.8;
    // the sea as a plane through the samples (spray, the waterline on the hull)
    const gF = (h[1] - h[2]) / (2 * LB), gS = (h[3] - h[4]) / (2 * BS);
    this.sea = { h: h[0], x: p[0], z: p[1], gx: gF * fwd[0] + gS * side[0], gz: gF * fwd[2] + gS * side[2] };
    if (!this.settled) {
      this.heave.x = heave; this.pitch.x = pitch; this.roll.x = roll;
      this.settled = true;
    }
    // damped springs: a light hull follows the long waves closely
    const n = Math.max(1, Math.ceil(dt / (1 / 120)));
    const step = dt / n;
    const hv0 = this.heave.x, pv0 = this.pitch.x;
    // (springs track the target's own velocity too: no lag behind a long swell)
    const spring = (s, target, w, z, vt = 0) => {
      for (let i = 0; i < n; i++) { s.v += (w * w * (target - s.x) + 2 * z * w * (vt - s.v)) * step; s.x += s.v * step; }
    };
    const rt = this.probe.rate;
    const heaveV = rt ? 0.4 * rt[0] + 0.15 * (rt[1] + rt[2] + rt[3] + rt[4]) : 0;
    spring(this.heave, heave, 7, 0.75, heaveV * 0.7);
    spring(this.pitch, Math.max(Math.min(pitch, 0.5), -0.5), 6, 0.6);
    spring(this.roll, Math.max(Math.min(roll, 0.45), -0.45), 5, 0.45);
    const B = this.frame();
    this.B = B;
    this.Binv = invRigid(B);
    // the sea in boat-local coordinates: y = a + b x + c z
    const loc = (x, z) => { const w = xform(B, [x, 0, z]); return (this.seaY(w[0], w[2]) - w[1]) / Math.max(B[5], 0.5); };
    const a0 = loc(0, 0);
    this.waterPlane = [a0, loc(1, 0) - a0, loc(0, 1) - a0];
    // bilge: a level plane (in the world) through the bilge, in boat-local coordinates
    const up = xdir(this.Binv, [0, 1, 0]);
    this.bilge = [-0.172, -up[0] / up[1], -up[2] / up[1]];

    // the rower and the oars
    this.pose = solvePose(this.phase());
    this.events(dt);
    // the wake field (before the main pass binds its target)
    this.wake.stepDrops(dt, (x, z) => this.seaY(x, z));
    const motion = [v, dt > 0 ? (this.heave.x - hv0) / dt : 0, dt > 0 ? (this.pitch.x - pv0) / dt : 0, this.time];
    const c = add3([p[0], 0, p[1]], sc3(fwd, -12));
    this.wake.track([p[0] + fwd[0] * XB, p[1] + fwd[2] * XB], this.time);
    this.wake.render([c[0], c[2]], B, this.Binv, this.waterPlane, motion, this.time);
  }

  phase() { return this.freeze !== null ? this.freeze : (((this.stroke % 1) + 1) % 1); }

  // blades entering and leaving the water: splashes, puddles, drips
  events(dt) {
    const B = this.B, W = this.wake, t = this.time;
    for (const k of ['R', 'L']) {
      const o = this.pose.oars[k], st = this.blades[k];
      const midL = madd3(o.pivot, o.dir, BLADE_MID), tipL = madd3(o.pivot, o.dir, LOUT);
      const mid = xform(B, midL), tip = xform(B, tipL);
      const sea = this.seaY(mid[0], mid[2]);
      const depth = sea - mid[1];
      const ph = this.phase();
      const inW = ph >= 0.955 || ph < 0.37;
      void depth;
      // blade velocity (world) for the spray
      const vel = st.last && dt > 0 ? sc3(sub3(mid, st.last), 1 / dt) : [0, 0, 0];
      st.last = mid;
      const dirW = norm3(xdir(B, o.dir));
      const hdir = norm3([dirW[0], 0, dirW[2]]);
      const fwdW = this.fwd;
      if (inW && !st.inWater) {
        // the catch: a crown of spray thrown forwards and up, a ring, a puddle that follows the blade
        st.pud = W.catchAt([mid[0], mid[2]], [hdir[0], hdir[2]], 1);
        for (let i = 0; i < 130; i++) {
          const along = (W.rand() - 0.5) * BLADE * 0.9;
          const q = madd3([mid[0], sea + 0.05 + W.rand() * 0.04, mid[2]], hdir, along);
          const sp = 0.7 + W.rand() * 2.0;
          const v = add3(sc3(fwdW, sp * (0.3 + W.rand() * 0.6)), [0, sp * (0.6 + W.rand() * 0.9), 0]);
          W.spawn(q, add3(v, sc3(hdir, (W.rand() - 0.5) * 0.8)), 0.0012 + W.rand() * W.rand() * 0.004, 0.9);
        }
      }
      if (inW && st.pud) { st.pud.x = mid[0]; st.pud.z = mid[2]; }
      if (!inW && st.inWater) {
        // the release: the blade lifts out feathering; water sheets and runs off it
        if (st.pud) { st.pud.t0 = t; st.pud = null; }
        st.tOut = t;
        st.draining = true;
      }
      // the moment the blade clears the surface: the water on it sheets off and breaks into drops
      if (st.draining && !inW && depth < -0.01) {
        st.draining = false;
        st.tOut = t;
        for (let i = 0; i < 90; i++) {
          const along = (W.rand() - 0.25) * BLADE;
          const q = madd3(mid, dirW, along);
          W.spawn(add3(q, [0, -0.01, 0]), add3(sc3(vel, 0.4 + W.rand() * 0.6), [(W.rand() - 0.5) * 0.5, W.rand() * 0.4, (W.rand() - 0.5) * 0.5]), 0.0015 + W.rand() * 0.004, 0.95);
        }
      }
      st.inWater = inW;
      st.wet = inW ? 1 : Math.exp(-(t - st.tOut) / 10);
      // drips from the tip and the edge during the recovery, slowing as the blade drains
      if (!inW && !st.draining) {
        const since = t - st.tOut;
        const rate = 28 * Math.exp(-since / 0.7) + 2.5 * Math.exp(-since / 3);
        st.nextDrip -= dt * rate;
        while (st.nextDrip < 0) {
          st.nextDrip += 1;
          const q = W.rand() < 0.6 ? tip : madd3(mid, dirW, (W.rand() - 0.5) * BLADE);
          W.spawn(add3(q, [0, -0.01, 0]), add3(sc3(vel, 0.9), [0, -0.1, 0]), 0.0018 + W.rand() * 0.0022, 0.95);
        }
      }
    }
  }

  // boat frame (world): origin at the design waterline, X to the bow, Y up, Z to the side
  frame() {
    const f = this.fwd, s = this.side;
    const X = norm3([f[0], Math.tan(this.pitch.x), f[2]]);
    let Z = norm3([s[0], Math.tan(this.roll.x), s[2]]);
    const Y = norm3(cross3(Z, X));
    Z = cross3(X, Y);
    return frameMat([this.pos2[0], this.heave.x - this.sink, this.pos2[1]], X, Y, Z);
  }

  // uniforms the sea shader needs (the wake field)
  waterFx() {
    const on = this.visible && this.settled && this.B;
    return { ...this.wake.uniforms(on), uBoatW2L: this.Binv || new Float32Array(16), uBoatL2W: this.B || new Float32Array(16), uBoatRower: [0.0, 1.22, 0.24, 0.21] };
  }

  // draw into the bound HDR target, before the water (u: the water's uniforms: sky, fog, camera)
  draw(u) {
    if (!this.visible || !this.settled || !this.pose) return;
    const gl = this.gl;
    const B = this.B, Binv = this.Binv;
    const P = this.pose;
    const caps = this.caps;
    caps.fill(0);
    P.caps.slice(0, NCAPS).forEach(([a, b, r], i) => caps.set([a[0], a[1], a[2], r, b[0], b[1], b[2], 0], i * 8));
    const common = { ...u, uBoatInv: Binv, uCaps: caps, uWaterPlane: this.waterPlane, uSpray: 0, uTime: this.time };
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
    gl.disable(gl.CULL_FACE);
    const prog = this.prop;
    prog.use({ ...common, uOar: [0, 0] });
    const drawProp = (mesh, M) => {
      prog.set('uModel', M);
      prog.set('uNrmM', normalMat(M));
      gl.bindVertexArray(mesh.vao);
      gl.drawElements(gl.TRIANGLES, mesh.count, mesh.itype, 0);
    };
    drawProp(this.mesh.hull, B);
    drawProp(this.mesh.fit, B);
    for (const k of ['R', 'L']) {
      const o = P.oars[k];
      const hz = norm3([o.dir[0], 0, o.dir[2]]);
      drawProp(this.mesh.lock, mul(B, frameMat(o.pivot, cross3([0, 1, 0], hz), [0, 1, 0], hz)));
    }
    for (const k of ['R', 'L']) {
      const o = P.oars[k];
      prog.set('uOar', [1, this.blades[k].wet ?? 0]);
      drawProp(this.mesh.oar, mul(B, frameMat(o.pivot, o.X, o.dir, o.Wd)));
    }
    // the rower
    const R = this.rower;
    if (R.ready) {
      const F = R.pose(P);
      this.skin.use({ ...common, uModel: B, uDQ: F.palette, uPart: 0 });
      gl.bindVertexArray(R.body.vao);
      gl.drawElements(gl.TRIANGLES, R.body.count, R.body.itype, 0);
      const rg = this.rigid;
      rg.use({ ...common });
      const part = (mesh, M, id) => {
        rg.set('uModel', M);
        rg.set('uPart', id);
        gl.bindVertexArray(mesh.vao);
        gl.drawElements(gl.TRIANGLES, mesh.count, mesh.itype, 0);
      };
      part(R.head, mul(B, F.head), 1);
      part(R.handR, mul(B, F.handR), 2);
      part(R.handL, mul(B, F.handL), 2);
    }
    // bilge water: blended over the bottom boards, no depth writes
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    this.bilgeProg.use({ ...common, uModel: B, uBilge: this.bilge });
    gl.bindVertexArray(this.mesh.bilge.vao);
    gl.drawElements(gl.TRIANGLES, this.mesh.bilge.count, this.mesh.bilge.itype, 0);
    gl.disable(gl.BLEND);
    gl.depthMask(true);
    // the lid: depth only, so the sea drawn next stays outside the hull
    gl.colorMask(false, false, false, false);
    prog.use({});
    drawProp(this.mesh.lid, B);
    gl.colorMask(true, true, true, true);
    gl.bindVertexArray(null);
  }

  // after the water: spray drops
  drawFx(u, viewport, dt, pixelAngle) {
    if (!this.visible || !this.settled) return;
    // (a 180-degree shutter; slowed down with the boat in slow motion)
    this.wake.drawDrops(u, viewport, Math.min(Math.max(dt, 1 / 120), 1 / 30) * 0.5 * this.timeScale, pixelAngle);
  }
}
void XS;
