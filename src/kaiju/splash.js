// kaiju: thrown water. CPU-integrated particles (ballistic, with air drag toward the wind) drawn as
// instanced billboards: dense spray clots, falling sheets / streams (stretched along their velocity)
// and drifting mist. Lit like the sea's own spindrift (sky + sun with a forward-scattering lobe,
// cloud shadow, lightning), then the sea's aerial perspective and rain / mist fog.
import { Program } from '../gl.js';
import { SKY_LOOKUP_GLSL } from '../sky/atmosphere.js';

export const SPRAY = 0, STREAM = 1, MIST = 2;
const STRIDE = 12;

// (the sky helpers in the vertex stage: the crepuscular-ray jitter has no fragment coordinate here)
const VS = SKY_LOOKUP_GLSL.replace(/gl_FragCoord\.xy/g, 'vec2(float(gl_VertexID), float(gl_InstanceID))') + `
layout(location = 0) in vec4 aPos;    // xyz, size
layout(location = 1) in vec4 aVel;    // velocity, alpha
layout(location = 2) in vec4 aInfo;   // kind, seed, age (0..1), 0
uniform mat4 uViewProj;
uniform vec3 uCamPos;
uniform vec3 uFlash;
uniform float uPixelAngle;
centroid out vec2 vUv;
out vec4 vCol;      // premultiplied-ready colour (fogged), alpha
flat out vec3 vInfo;
float hgp(float c, float g) { return (1.0 - g * g) / (12.566 * pow(1.0 + g * g - 2.0 * g * c, 1.5)); }
void main() {
  vec3 P = aPos.xyz;
  vec3 toCam = uCamPos - P;
  float dist = length(toCam);
  vec3 V = toCam / max(dist, 1e-3);
  vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), toCam));
  vec3 up = cross(V, right);
  vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) - 0.5;
  float kind = aInfo.x;
  float size = aPos.w;
  // at least ~1.5 px: a clot further than that is unresolved and keeps its energy as a dimmer, larger disk
  float minS = dist * uPixelAngle * 1.5;
  float a = aVel.w * min(1.0, (size * size) / (minS * minS + 1e-6));
  size = max(size, minS);
  vec3 W;
  if (kind > 0.5 && kind < 1.5) {
    // a fall: a thin ribbon of water whose head is the particle and whose body trails back along the
    // path it fell (about 0.6 s of its motion): continuous, vertical when falling, never a round clot
    // (water falling off a moving body falls straight down past it: the ribbon leans only a little)
    vec3 v = aVel.xyz * vec3(0.35, 1.0, 0.35) + vec3(0.0, -2.0, 0.0);
    float sp = length(v);
    vec3 dir = sp > 1e-3 ? v / sp : vec3(0.0, -1.0, 0.0);
    vec3 side = cross(V, dir);
    float sl = length(side);
    side = sl > 1e-3 ? side / sl : right;
    float len = 2.5 + sp * 0.6;
    W = P + dir * (c.y - 0.5) * len + side * c.x * size * 0.5;
  } else {
    float r = aInfo.y * 6.2831;
    vec2 cr = vec2(cos(r) * c.x - sin(r) * c.y, sin(r) * c.x + cos(r) * c.y);
    W = P + (right * cr.x + up * cr.y) * size * (kind > 1.5 ? 1.0 : 1.15);
  }
  vUv = c + 0.5;
  vInfo = aInfo.xyz;
  // light: the grey of the sky and sun light around the drops (they scatter all wavelengths alike),
  // a forward lobe toward the sun, cloud shadow at the drop
  vec3 L = uSunDir;
  vec2 suv = (P.xz - L.xz * (max(P.y, 0.0) / max(L.y, 0.02)) - uShaftRect.xy) * uShaftRect.zw;
  float vis = mix(uHazeSun, textureLod(uShaftShadow, suv, 0.0).r, smoothstep(0.0, 0.04, min(min(suv.x, suv.y), min(1.0 - suv.x, 1.0 - suv.y))));
  vec3 Es = sunIlluminance() * vis;
  // a forward lobe of single scattering, and the multiply-scattered light of dense spray (like a cloud's)
  float ph = hgp(dot(-V, L), 0.65) * 0.45 + 0.55 * 0.16;
  vec3 Lc = skyIrradiance() * (0.8 / 3.14159) + Es * ph * (kind > 1.5 ? 1.1 : 1.3) * max(L.y + 0.05, 0.0) / max(L.y + 0.05, 0.05) + uFlash * 0.25;
  // dense clots are bright white on their lit side, a little greyer in their own shade
  vec3 col = vec3(dot(Lc, vec3(0.2126, 0.7152, 0.0722))) * vec3(0.97, 0.99, 1.02) * (kind > 1.5 ? 0.9 : 0.82);
  col = max(col, vec3(dot(fogRadiance(-V, uHazeSun), vec3(0.2126, 0.7152, 0.0722))) * (kind > 1.5 ? 1.0 : 0.9));
  vec3 aL, aT;
  aerial(-V, dist * 0.001, aL, aT);
  col = col * aT + aL;
  col = applyFog(col, -V, dist);
  vCol = vec4(col, a);
  gl_Position = uViewProj * vec4(W, 1.0);
}`;

const FS = `
centroid in vec2 vUv;
in vec4 vCol;
flat in vec3 vInfo;
out vec4 oColor;
float hs(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
float vn(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hs(i), hs(i + vec2(1, 0)), f.x), mix(hs(i + vec2(0, 1)), hs(i + vec2(1, 1)), f.x), f.y); }
void main() {
  vec2 q = vUv * 2.0 - 1.0;
  float kind = vInfo.x, seed = vInfo.y * 37.0, age = vInfo.z;
  float a;
  if (kind > 1.5) {
    // mist: a very soft, slowly eroding puff
    float n = vn(vUv * 3.0 + seed) * 0.6 + vn(vUv * 7.0 + seed * 1.7) * 0.4;
    a = exp(-dot(q, q) * 2.2) * smoothstep(0.15, 0.75, n + 0.25 - 0.3 * dot(q, q));
  } else if (kind > 0.5) {
    // a fall: a translucent sheet, denser (brighter) at its edges, streaked along its length, breaking
    // into ropes and then drops as it falls (q.y: -1 the top of the ribbon, +1 its leading end)
    float ax = abs(q.x);
    float across = mix(0.45, 1.0, smoothstep(0.25, 0.8, ax)) * smoothstep(1.0, 0.8, ax);
    float n = vn(vec2(q.x * 2.5 + seed, q.y * 3.0)) * 0.55 + vn(vec2(q.x * 6.0 + seed, q.y * 9.0 - seed)) * 0.45;
    float breakup = mix(0.15, 0.62, age);
    a = across * smoothstep(breakup, breakup + 0.2, n) * smoothstep(-1.0, -0.3, q.y) * smoothstep(1.0, 0.8, q.y);
  } else {
    // a clot of spray: lumpy, its edge eroding into droplets as it ages
    float r = dot(q, q);
    float n = vn(vUv * 4.0 + seed) * 0.55 + vn(vUv * 9.0 + seed * 2.3) * 0.3 + vn(vUv * 19.0 + seed) * 0.15;
    // (a dense core, a ragged translucent fringe)
    a = smoothstep(1.0, 0.15, r + (n - 0.5) * (1.2 + 0.8 * age)) * mix(0.55 + 0.45 * smoothstep(0.3, 0.75, n), smoothstep(0.4, 0.75, n), age);
  }
  a *= vCol.a;
  if (a < 0.003) discard;
  oColor = vec4(vCol.rgb * a, a);
}`;

export class Splash {
  constructor(gl, cap = 9000) {
    this.gl = gl;
    this.cap = cap;
    this.n = 0;
    this.prog = new Program(gl, VS, FS, { label: 'kaiju-splash' });
    this.d = new Float32Array(cap * STRIDE);   // pos+size, vel+alpha, kind, seed, age, life
    this.base = new Float32Array(cap * 2);      // base size, base alpha
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    this.buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferData(gl.ARRAY_BUFFER, this.d.byteLength, gl.DYNAMIC_DRAW);
    for (let i = 0; i < 3; i++) {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, 4, gl.FLOAT, false, STRIDE * 4, i * 16);
      gl.vertexAttribDivisor(i, 1);
    }
    gl.bindVertexArray(null);
    this.gpu = new Float32Array(cap * STRIDE);
  }

  // kind, position, velocity, size (m), life (s), alpha
  add(kind, x, y, z, vx, vy, vz, size, life, alpha) {
    if (this.n >= this.cap) return;
    const i = this.n++, o = i * STRIDE, d = this.d;
    d[o] = x; d[o + 1] = y; d[o + 2] = z; d[o + 3] = size;
    d[o + 4] = vx; d[o + 5] = vy; d[o + 6] = vz; d[o + 7] = alpha;
    d[o + 8] = kind; d[o + 9] = Math.random(); d[o + 10] = 0; d[o + 11] = life;
    this.base[i * 2] = size; this.base[i * 2 + 1] = alpha;
  }

  // sea(x, z): sea height; wind: [wx, wz] m/s
  update(dt, sea, wind, onImpact) {
    const d = this.d, b = this.base;
    let w = 0;
    for (let i = 0; i < this.n; i++) {
      const o = i * STRIDE;
      const kind = d[o + 8];
      let age = d[o + 10] + dt;
      const life = d[o + 11];
      let x = d[o], y = d[o + 1], z = d[o + 2], vx = d[o + 4], vy = d[o + 5], vz = d[o + 6];
      // drag toward the wind (fine mist follows it, heavy clots barely)
      const k = kind === MIST ? 0.9 : kind === STREAM ? 0.12 : 0.35;
      const f = 1 - Math.exp(-k * dt);
      vx += (wind[0] - vx) * f; vz += (wind[1] - vz) * f;
      vy += (kind === MIST ? (0.15 - vy) * f : -vy * f * 0.5) - (kind === MIST ? 0 : 9.81 * dt);
      x += vx * dt; y += vy * dt; z += vz * dt;
      const h = sea(x, z);
      let dead = age > life;
      if (!dead && kind !== MIST && y < h - 0.3 && vy < 0) {
        dead = true;
        if (onImpact) onImpact(kind, x, h, z, vy, b[i * 2]);
      }
      if (dead) continue;
      const t = age / life;
      const o2 = w * STRIDE;
      d[o2] = x; d[o2 + 1] = Math.max(y, kind === MIST ? h + 0.5 : y); d[o2 + 2] = z;
      d[o2 + 3] = b[i * 2] * (kind === MIST ? 1 + 1.6 * t : kind === SPRAY ? 1 + 1.3 * t : 1 + 0.4 * t);
      d[o2 + 4] = vx; d[o2 + 5] = vy; d[o2 + 6] = vz;
      d[o2 + 7] = b[i * 2 + 1] * (kind === MIST ? Math.sin(Math.PI * Math.min(t * 1.6, 1)) * (1 - t) * 1.6 : Math.pow(1 - t, 1.3) * Math.min(age * 12, 1));
      d[o2 + 8] = kind; d[o2 + 9] = d[o + 9]; d[o2 + 10] = age; d[o2 + 11] = life;
      b[w * 2] = b[i * 2]; b[w * 2 + 1] = b[i * 2 + 1];
      w++;
    }
    this.n = w;
  }

  draw(u) {
    if (!this.n) return;
    const gl = this.gl, d = this.d, g = this.gpu;
    // the GPU copy carries the normalised age
    for (let i = 0; i < this.n; i++) {
      const o = i * STRIDE;
      for (let k = 0; k < 10; k++) g[o + k] = d[o + k];
      g[o + 10] = d[o + 10] / d[o + 11];
      g[o + 11] = 0;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, g, 0, this.n * STRIDE);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    gl.enable(gl.DEPTH_TEST);
    this.prog.use(u);
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.n);
    gl.bindVertexArray(null);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }
}
