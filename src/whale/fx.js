// Whale and water: the blow, spray, splashes and foam.
//
// Particles (simulated on the CPU, a few thousand at most, drawn as one instanced pass):
//  0 blow mist: the exhaled cloud of droplets leaves the blowholes at ~15 m/s, is braked by the air
//    within a few metres (a bushy column 3-4 m tall), billows out, drifts downwind and evaporates.
//    Lit by single scattering with a strong forward lobe (it glows against a low sun) plus the diffuse
//    sky, and shadowed by its own optical depth towards the sun.
//  1 spray: clumps of drops, ballistic, motion-blurred along their velocity; water sheets streaming off
//    the body and the curtain falling from the trailing edge of raised flukes.
//  2 splash plumes: heavy aerated water thrown up by a breach and its landing.
//  3 splash mist: the fine spray cloud that hangs over a landing and drifts away.
//
// Sea surface (drawn after the water, on the water mesh's own polar grid so the depths match): fresh
// froth and residual foam from the landing and the exit (using the sea's own whitecap pattern), the
// white wash where the body cuts the surface (from the spine sections), smooth "fluke prints" where the
// tail stirred the water, and the aquamarine glow of white flippers just under the surface.
import { Program, makeMesh } from '../gl.js';
import { SKY_LOOKUP_GLSL } from '../sky/atmosphere.js';
import { FOAM_GLSL } from '../ocean/foam.js';
import { rng } from '../math.js';
import { NSPINE, SPINE_X, SNOUT } from './shape.js';

const MAXP = 7000;
const NSPOT = 16;

// ---------------------------------------------------------------- particle shaders
const PVS = `
layout(location = 0) in vec4 aA;   // position, size
layout(location = 1) in vec4 aB;   // velocity, type
layout(location = 2) in vec4 aC;   // age, life, seed, alpha
layout(location = 3) in vec4 aD;   // sun transmittance, sea height, -, -
uniform mat4 uViewProj;
uniform vec3 uCamPos;
uniform float uPixelAngle;
out vec2 vQ;
out float vType;
out float vAlpha;
out float vSeed;
out float vAge;
out float vShade;
out vec3 vPos;
out float vStretch;
void main() {
  vec3 pos = aA.xyz;
  float size = aA.w;
  int type = int(aB.w + 0.5);
  vec3 toCam = uCamPos - pos;
  float dist = length(toCam);
  vec3 vd = toCam / max(dist, 1e-3);
  vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), vd));
  vec3 up = cross(vd, right);
  vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
  float alpha = aC.w;
  // a particle never shrinks below ~1.5 px: smaller ones keep their energy as lower opacity
  float minS = dist * uPixelAngle * (type == 1 || type == 4 ? 2.2 : 1.5);
  if (size < minS) { alpha *= (size * size) / (minS * minS); size = minS; }
  vec3 P;
  float stretch = 1.0;
  if (type == 1 || type == 4) {
    // drops (1) and water ropes / sheets (4): oriented along their motion, a little motion blur,
    // never thinner than ~2 px (a falling drop is a soft bead, not a hairline)
    vec3 v = aB.xyz;
    vec3 vp = v - vd * dot(v, vd);
    float l = length(vp);
    vec3 ax = l > 1e-3 ? vp / l : up;
    vec3 ay = normalize(cross(vd, ax));
    float el = type == 4 ? 2.6 + 1.4 * fract(aC.z * 13.0) : 1.0;
    float len = size * el + min(l / 60.0, size * 1.5);
    stretch = len / size;
    alpha /= mix(1.0, sqrt(stretch), 0.5);
    P = pos + ax * c.x * len * 0.5 + ay * c.y * size * 0.5;
  } else if (type == 2) {
    // aerated water: stretched a little along its motion, lumpy
    vec3 v = aB.xyz;
    vec3 vp = v - vd * dot(v, vd);
    float l = length(vp);
    vec3 ax = l > 1e-3 ? vp / l : up;
    vec3 ay = normalize(cross(vd, ax));
    float st = 1.0 + min(l * 0.08, 0.8);
    stretch = st;
    P = pos + (ax * c.x * st + ay * c.y / sqrt(st)) * size * 0.5;
  } else {
    // billowing puffs turn slowly
    float a = aC.z * 6.2831 + aC.x * 0.3 * (fract(aC.z * 7.0) - 0.5);
    vec2 cr = vec2(c.x * cos(a) - c.y * sin(a), c.x * sin(a) + c.y * cos(a));
    P = pos + (right * cr.x + up * cr.y) * size * 0.5;
  }
  // mist and plumes fade into the sea instead of cutting it with a hard line
  if (type != 1 && type != 4) alpha *= smoothstep(-0.2 * size, 0.15 * size, pos.y - aD.y);
  vQ = c;
  vType = float(type);
  vAlpha = alpha;
  vSeed = aC.z;
  vAge = aC.x / max(aC.y, 1e-3);
  vShade = aD.x;
  vPos = pos;
  vStretch = stretch;
  gl_Position = uViewProj * vec4(P, 1.0);
}`;

const PFS = SKY_LOOKUP_GLSL + `
in vec2 vQ;
in float vType;
in float vAlpha;
in float vSeed;
in float vAge;
in float vShade;
in vec3 vPos;
in float vStretch;
out vec4 oColor;
uniform vec3 uCamPos;
uniform vec3 uFlash;
float h2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float n2(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(h2(i), h2(i + vec2(1, 0)), f.x), mix(h2(i + vec2(0, 1)), h2(i + vec2(1, 1)), f.x), f.y);
}
float hg(float c, float g) { return (1.0 - g * g) / (4.0 * 3.14159 * pow(1.0 + g * g - 2.0 * g * c, 1.5)); }
void main() {
  int type = int(vType + 0.5);
  float r2 = dot(vQ, vQ);
  if (r2 > 1.0) discard;
  float a;
  vec2 o = vec2(vSeed * 37.0, vSeed * 91.0);
  float glint = 0.0;
  if (type == 1) {
    // a drop (or a few): a soft bead with a bright refraction highlight off-centre
    a = smoothstep(1.0, 0.2, r2) * vAlpha * mix(0.8, 1.0, smoothstep(0.2, 0.8, r2));
    glint = 0.6 * exp(-dot(vQ - vec2(0.2, 0.25), vQ - vec2(0.2, 0.25)) * 14.0);
  } else if (type == 4) {
    // a rope / ribbon of water: translucent body, brighter rims and lumps, beading up as it falls
    float along = vQ.x * vStretch;
    float wob = (n2(vec2(along * 1.3, 0.0) + o) - 0.5) * 0.5;
    float yy = vQ.y - wob;
    float wid = 0.55 + 0.45 * n2(vec2(along * 0.9 + 5.0, 1.0) + o);
    float body = smoothstep(wid, wid * 0.4, abs(yy));
    float bead = smoothstep(0.25 + 0.5 * vAge, 0.75 + 0.2 * vAge, n2(vec2(along * 2.2, yy) + o * 1.3) * 0.7 + 0.3 * (1.0 - vAge));
    float ends = smoothstep(1.0, 0.7, abs(vQ.x));
    a = body * mix(1.0, bead, smoothstep(0.1, 0.6, vAge)) * ends * vAlpha;
    float rim = smoothstep(wid * 0.35, wid * 0.85, abs(yy)) * body;
    glint = max(rim * 0.35, pow(n2(vec2(along * 3.0, yy * 2.0) + o * 2.1), 10.0) * 2.0);
    a *= mix(0.55, 1.0, rim);
  } else {
    // three octaves of noise eat into a soft disk
    vec2 w = vQ + 0.25 * vec2(n2(vQ * 1.3 + o + 4.0), n2(vQ * 1.3 + o + 9.0)) - 0.125;
    float n = n2(w * 1.9 + o) * 0.5 + n2(w * 4.1 + o * 1.3) * 0.3 + n2(w * 8.3 + o * 0.7) * 0.2;
    if (type == 2) {
      // aerated water: a cauliflower mass with a crisp, lumpy edge
      float n2b = n2(w * 15.0 + o * 2.3);
      a = smoothstep(0.0, 0.18, (1.0 - r2) * 0.85 - 1.25 * (n - 0.3) - 0.25 * n2b) * vAlpha;
    } else {
      // droplet mist: soft, mottled, ragged as it ages
      // (billows: the fine octave sharpened so a cloud of puffs keeps a turbulent, grainy texture)
      float fine = n2(w * 16.0 + o * 3.1);
      a = clamp(exp(-r2 * 2.4) * (0.1 + 1.9 * n * (0.7 + 0.6 * fine)) - 0.25 - 0.2 * vAge * n, 0.0, 1.0) * vAlpha * 1.25;
    }
  }
  if (a < 0.002) discard;
  vec3 toCam = uCamPos - vPos;
  float dist = length(toCam);
  vec3 V = toCam / dist;
  // light: sun (or moon) through the cloud-shadow map and the cloud's own depth, plus the diffuse sky
  vec2 suv = (vPos.xz - uSunDir.xz * (max(vPos.y, 0.0) / max(uSunDir.y, 0.02)) - uShaftRect.xy) * uShaftRect.zw;
  float vis = mix(uHazeSun, texture(uShaftShadow, suv).r, smoothstep(0.0, 0.04, min(min(suv.x, suv.y), min(1.0 - suv.x, 1.0 - suv.y))));
  vis *= step(0.0, uSunDir.y + 0.02);
  float cs = dot(-V, uSunDir);
  vec3 Es = sunIlluminance() * vis * vShade;
  vec3 Ea = skyIrradiance();
  vec3 col;
  if (type == 1 || type == 4) {
    // clear water: transmits the scene behind (low alpha), scatters a little skylight, and its curved
    // surfaces throw small bright images of the sun (refraction and reflection highlights)
    float spec = 0.6 + 2.5 * hg(cs, 0.65) * 3.14159 + 0.6 * hg(cs, -0.5) * 3.14159;
    col = (Ea * 0.5 + Es * 0.2) / 3.14159 + Es * glint * spec * 0.18;
  } else if (type == 2) {
    // aerated water: a white diffuser (multiple scattering, albedo ~0.8) lit on the side the sun sees,
    // glowing a little when backlit (light through thin sheets)
    col = 0.8 / 3.14159 * (Es * (0.35 + 0.45 * (0.5 - 0.5 * cs) + 0.4 * 3.14159 * hg(cs, 0.5)) + Ea * 0.8);
  } else {
    // droplet mist: Mie-like single scattering (strong forward lobe, weak back lobe) plus the multiple
    // scattering that makes a dense droplet cloud white from every side
    float ph = mix(hg(cs, 0.78), hg(cs, -0.25), 0.18);
    col = Es * (ph * 0.9 + 0.22 * (0.6 + 0.4 * (0.5 - 0.5 * cs))) + Ea * 0.25;
  }
  col += uFlash * 0.25;
  vec3 aL, aT;
  aerial(-V, dist * 0.001, aL, aT);
  col = col * aT + aL;
  col = applyFog(col, -V, dist);
  oColor = vec4(col * a, a);
}`;

// ---------------------------------------------------------------- sea-surface pass (the water's VS, own FS)
const SVS = `
layout(location = 0) in vec2 aPolar;
uniform mat4 uViewProj;
uniform vec3 uCamPos;
uniform sampler2D uDisp[3];
uniform float uSize[4];
uniform float uTexel[4];
uniform float uAngStep;
uniform float uDispScale;
out vec3 vWorld;
out vec2 vXZ;
void main() {
  float r = aPolar.x;
  vec2 xz = uCamPos.xz + vec2(cos(aPolar.y), sin(aPolar.y)) * r;
  float spacing = max(r * uAngStep, 0.02);
  vec3 D = vec3(0.0);
  float lsp = log2(spacing * 1.5);
  D += textureLod(uDisp[0], xz / uSize[0], max(lsp - log2(uTexel[0]), 0.0)).xyz;
  D += textureLod(uDisp[1], xz / uSize[1], max(lsp - log2(uTexel[1]), 0.0)).xyz;
  D += textureLod(uDisp[2], xz / uSize[2], max(lsp - log2(uTexel[2]), 0.0)).xyz;
  D *= uDispScale;
  vec3 P = vec3(xz.x + D.x, D.y, xz.y + D.z);
  float dr = length(P.xz - uCamPos.xz);
  P.y -= dr * dr / (2.0 * 6360000.0);
  vWorld = P;
  vXZ = xz;
  gl_Position = uViewProj * vec4(P, 1.0);
}`;

const SFS = SKY_LOOKUP_GLSL + FOAM_GLSL + `
in vec3 vWorld;
in vec2 vXZ;
out vec4 oColor;
uniform vec3 uCamPos;
uniform sampler2D uEnv;
uniform sampler2D uSlope[4];
uniform float uSize[4];
uniform vec3 uRrs;
uniform vec3 uFlash;
uniform float uTime;
uniform vec4 uSpotA[${NSPOT}];   // xz, radius, age (s)
uniform vec4 uSpotB[${NSPOT}];   // strength, axis cos, axis sin, aspect
uniform float uSpotT[${NSPOT}];  // 0 splash foam, 1 wash, 2 fluke print (slick)
uniform vec4 uSecC[${NSPINE}];   // spine section centre (world), half-height above
uniform vec4 uSecX[${NSPINE}];   // body axis, half-height below
uniform vec4 uSecY[${NSPINE}];   // dorsal axis, half-width
uniform float uSecW[${NSPINE}];  // wash strength of the section
uniform vec4 uPect[6];           // flipper points (world), radius
uniform float uSpacing;
const float M_PI = 3.14159265359;
vec2 equirectUv(vec3 d) { return vec2(atan(d.z, d.x) / (2.0 * M_PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / M_PI); }
float h2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float n2(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(h2(i), h2(i + vec2(1, 0)), f.x), mix(h2(i + vec2(0, 1)), h2(i + vec2(1, 1)), f.x), f.y);
}
// the affine part of aerial perspective + fog along this ray: col -> k col + b
void chain(vec3 dir, float dist, out vec3 k, out vec3 b) {
  vec3 aL, aT;
  aerial(dir, dist * 0.001, aL, aT);
  vec3 c0 = applyFog(aL, dir, dist);
  vec3 c1 = applyFog(aT + aL, dir, dist);
  k = c1 - c0;
  b = c0;
}
void main() {
  vec3 P = vWorld;
  vec3 toCam = uCamPos - P;
  float dist = length(toCam);
  vec3 V = toCam / dist;
  // ---- foam spots: fresh froth that ages into residual foam, spreading and opening into lace
  float act = 0.0, res = 0.0, slick = 0.0, plume = 0.0;
  float rag = n2(vXZ * 0.35) - 0.5;
  // residual foam gathers in lumps and streaks with open water between them
  float lump = smoothstep(0.35, 0.75, n2(vXZ * 0.21 + 3.0) * 0.6 + n2(vXZ * 0.53 + 7.0) * 0.4) * 1.5;
  // fractal breakup: five octaves, so the patches fray into islands and filaments
  float fb = n2(vXZ * 0.31) * 0.4 + n2(vXZ * 0.73 + 2.0) * 0.25 + n2(vXZ * 1.7 + 5.0) * 0.17 + n2(vXZ * 3.9 + 1.0) * 0.1 + n2(vXZ * 8.3 + 9.0) * 0.08;
  float frac = smoothstep(0.38, 0.7, fb);
  for (int i = 0; i < ${NSPOT}; i++) {
    vec4 A = uSpotA[i], B = uSpotB[i];
    if (B.x <= 0.0) continue;
    vec2 rel = vXZ - A.xy;
    vec2 e1 = B.yz;
    vec2 q = vec2(dot(rel, e1) / B.w, dot(rel, vec2(-e1.y, e1.x)));
    float age = A.w;
    float R = A.z * (1.0 + 0.55 * sqrt(age / 12.0));
    float r = length(q) / R * (1.0 + 0.35 * rag);
    if (r > 1.6) continue;
    int ty = int(uSpotT[i] + 0.5);
    if (ty == 2) { slick += B.x * exp(-age / 45.0) * smoothstep(1.0, 0.55, r) * smoothstep(0.0, 2.0, age); continue; }
    float tA = ty == 1 ? 0.7 : 2.2, tR = ty == 1 ? 6.0 : 30.0;
    // froth only where the fractal breakup allows; it thins out towards the rim
    float edgeT = smoothstep(1.05, 0.3, r);
    act += B.x * exp(-age / tA) * edgeT * (ty == 1 ? 0.6 : 2.0) * mix(frac, lump, smoothstep(1.0, 3.0, age));
    plume += B.x * exp(-age / (ty == 1 ? 6.0 : 14.0)) * smoothstep(1.1, 0.4, r);
    // the residual foam is pushed into a ring around a smoother centre
    float ring = mix(1.0, 0.45 + 1.1 * smoothstep(0.35, 0.9, r), smoothstep(2.0, 12.0, age));
    res += B.x * exp(-age / tR) * smoothstep(1.15, 0.5, r) * ring * (ty == 1 ? 0.35 : 1.3) * lump * frac;
  }
  // ---- the waterline: white wash where the body cuts the surface; flippers glowing below it
  float wash = 0.0;
  for (int i = 0; i < ${NSPINE}; i++) {
    float sw = uSecW[i];
    if (sw <= 0.0) continue;
    vec3 rel = P - uSecC[i].xyz;
    vec3 X = uSecX[i].xyz, Y = uSecY[i].xyz;
    float ax = dot(rel, X);
    if (abs(ax) > uSpacing * 0.75) continue;
    vec3 Z = cross(X, Y);
    float y = dot(rel, Y), z = dot(rel, Z);
    float ry = y > 0.0 ? uSecC[i].w : uSecX[i].w, rz = uSecY[i].w;
    float e = sqrt((y * y) / (ry * ry) + (z * z) / (rz * rz));
    float out_ = (e - 1.0) * min(ry, rz);
    float band = 0.35 + 0.5 * sw;
    wash = max(wash, sw * smoothstep(band, 0.0, out_) * smoothstep(-0.3, 0.0, out_) * smoothstep(0.2, 0.8, n2(vXZ * 1.3 + uTime * 0.6) * 0.6 + n2(vXZ * 0.45 - uTime * 0.3) * 0.6));
  }
  act += wash * 1.6 * frac;
  res += wash * 0.2 * frac;
  vec3 glow = vec3(0.0);
  for (int i = 0; i < 6; i++) {
    vec4 p = uPect[i];
    float dep = P.y - p.y;
    if (p.w <= 0.0 || dep < 0.0) continue;
    float hd = length(P.xz - p.xz);
    float g = exp(-dep / 1.3) * smoothstep(p.w * (1.0 + dep * 0.6), 0.0, hd);
    glow += g * exp(-vec3(0.45, 0.075, 0.05) * 2.0 * dep);
  }
  vec2 fc = foamCoverage(vXZ, res, 0.0, act, dist);
  float foam = fc.x;
  float aGlow = dot(glow, vec3(0.33));
  if (foam < 0.002 && slick < 0.002 && aGlow < 0.002 && plume < 0.002) discard;
  vec3 k, b;
  chain(-V, dist, k, b);
  vec3 Lsun = uSunDir;
  vec3 sunE = sunIlluminance();
  vec2 suv = (P.xz - uShaftRect.xy) * uShaftRect.zw;
  float sunVis = mix(uHazeSun, texture(uShaftShadow, suv).r, smoothstep(0.0, 0.04, min(min(suv.x, suv.y), min(1.0 - suv.x, 1.0 - suv.y))));
  vec3 skyE = skyIrradiance();
  vec3 Ed = sunE * sunVis * max(Lsun.y, 0.0) + skyE + uFlash * 0.3;
  float nv = max(V.y, 0.02);
  float F = 0.02 + 0.98 * pow(1.0 - nv, 5.0);
  vec3 outC = vec3(0.0);
  float alpha = 0.0;
  // calm water of a fluke print: a smoother mirror over the long waves
  if (slick > 0.002) {
    vec2 s = texture(uSlope[0], vXZ / uSize[0]).xy + texture(uSlope[1], vXZ / uSize[1]).xy;
    vec3 N = normalize(vec3(-s.x, 1.0, -s.y));
    vec3 R = reflect(-V, N);
    R.y = abs(R.y);
    float Fs = 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    vec3 calm = Fs * textureLod(uEnv, equirectUv(normalize(R)), 2.5).rgb + (1.0 - Fs) * uRrs * Ed;
    float aS = clamp(slick, 0.0, 1.0) * 0.55;
    outC = calm * aS;
    alpha = aS;
  }
  if (foam > 0.0) {
    float thick = fc.y;
    vec3 Nf = normalize(vec3(foamBump.x * 1.8, 1.0, foamBump.y * 1.8));
    float wrapL = max((dot(Nf, Lsun) + 0.2) / 1.2, 0.0);
    vec3 Ein = sunE * sunVis * wrapL + skyE * (0.45 + 0.55 * Nf.y) + uFlash;
    vec3 foamCol = mix(0.5, 0.9, thick) * mix(vec3(0.93, 0.97, 0.98), vec3(1.0), thick) * Ein / M_PI;
    float aF = foam * mix(0.55, 1.0, thick);
    outC = outC * (1.0 - aF) + foamCol * foam;
    alpha = alpha + aF * (1.0 - alpha);
  }
  // light scattered back by a white flipper under the surface (added to the water-leaving light)
  vec3 add = (1.0 - F) * glow * 0.55 * Ed / M_PI * (1.0 - alpha);
  // the bubble cloud under a fresh footprint backscatters light: pale aquamarine water
  float pl = 1.0 - exp(-plume * 1.6);
  add += (1.0 - F) * pl * vec3(0.05, 0.10, 0.085) * Ed / M_PI * (1.0 - alpha * 0.5);
  oColor = vec4(k * (outC + add) + alpha * b, alpha);
}`;

// ---------------------------------------------------------------- CPU particles
class Pool {
  constructor(n) {
    this.n = 0;
    this.max = n;
    this.p = new Float32Array(n * 3); this.v = new Float32Array(n * 3);
    this.size = new Float32Array(n); this.grow = new Float32Array(n); this.age = new Float32Array(n); this.life = new Float32Array(n);
    this.type = new Uint8Array(n); this.seed = new Float32Array(n); this.a0 = new Float32Array(n); this.drag = new Float32Array(n);
    this.grav = new Float32Array(n); this.group = new Int16Array(n); this.shade = new Float32Array(n); this.s0 = new Float32Array(n);
  }
  add(p, v, o) {
    if (this.n >= this.max) return;
    const i = this.n++;
    this.p.set(p, i * 3); this.v.set(v, i * 3);
    this.size[i] = o.size; this.grow[i] = o.grow || 0; this.age[i] = 0; this.life[i] = o.life; this.type[i] = o.type;
    this.seed[i] = o.seed; this.a0[i] = o.alpha; this.drag[i] = o.drag; this.grav[i] = o.grav ?? 9.81; this.group[i] = o.group ?? -1; this.shade[i] = 1; this.s0[i] = o.size;
  }
  // emitted at a random moment within the last frame: advance it so continuous emitters stay continuous
  pre(dt, r) {
    const i = this.n - 1;
    if (i < 0) return;
    const t = dt * r, o = i * 3;
    for (let c = 0; c < 3; c++) this.p[o + c] += this.v[o + c] * t - (c === 1 ? 0.5 * this.grav[i] * t * t : 0);
    this.v[o + 1] -= this.grav[i] * t;
    this.age[i] = t;
  }
  kill(i) {
    const j = --this.n;
    if (i === j) return;
    for (const [a, k] of [[this.p, 3], [this.v, 3]]) for (let c = 0; c < k; c++) a[i * k + c] = a[j * k + c];
    for (const a of [this.size, this.grow, this.age, this.life, this.type, this.seed, this.a0, this.drag, this.grav, this.group, this.shade, this.s0]) a[i] = a[j];
  }
}

const sstep = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
const nrm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

export class WhaleFx {
  constructor(gl) {
    this.gl = gl;
    this.pprog = new Program(gl, PVS, PFS, { label: 'whale-particles' });
    this.sprog = new Program(gl, SVS, SFS, { label: 'whale-surface' });
    this.pool = new Pool(MAXP);
    this.gpu = new Float32Array(MAXP * 16);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    this.buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferData(gl.ARRAY_BUFFER, this.gpu.byteLength, gl.DYNAMIC_DRAW);
    for (let k = 0; k < 4; k++) {
      gl.enableVertexAttribArray(k);
      gl.vertexAttribPointer(k, 4, gl.FLOAT, false, 64, k * 16);
      gl.vertexAttribDivisor(k, 1);
    }
    gl.bindVertexArray(null);
    this.r = rng(1234);
    this.spots = [];
    this.groups = [];            // blow clouds: centroid tracking for self-shadowing
    this.sec = { C: new Float32Array(NSPINE * 4), X: new Float32Array(NSPINE * 4), Y: new Float32Array(NSPINE * 4), W: new Float32Array(NSPINE) };
    this.pect = new Float32Array(24);
    this.spotA = new Float32Array(NSPOT * 4); this.spotB = new Float32Array(NSPOT * 4); this.spotT = new Float32Array(NSPOT);
    this.lastWashSpot = -1e9;
    this.surface = null;
    this.prevEdge = null;
    this.fired = new Set();
  }

  clearEvents() { this.fired.clear(); }

  spot(x, z, r, strength, type, dir = [1, 0], aspect = 1, now = 0) {
    this.spots.push({ x, z, r, s: strength, type, dir, aspect, born: now });
    if (this.spots.length > NSPOT) {
      // drop the faintest
      let worst = 0, wv = Infinity;
      this.spots.forEach((s, i) => { const v = s.s * Math.exp(-(now - s.born) / (s.type === 2 ? 45 : 30)); if (v < wv) { wv = v; worst = i; } });
      this.spots.splice(worst, 1);
    }
  }

  // which named moments did the clock just pass (once each per cycle)
  crossed(ev, T) { return ev.prevT < T && ev.t >= T; }

  update(ev, env) {
    const R = this.r, P = this.pool, dt = ev.dt;
    const wind = env.windVec || [0, 0];
    const air = [wind[0] * 0.75, 0, wind[1] * 0.75];
    const sea0 = ev.sea(28);
    let washLevel = 0;
    // ---------------- blows: an explosive burst over ~0.55 s from the two blowholes, following the head.
    // Many small puffs: a dense column that is braked by the air within 3-5 m, billows into a bushy crown,
    // thins as it spreads and drifts downwind.
    for (const b of ev.blows) {
      const tb = ev.t - b;
      if (tb >= 0 && tb < 0.55 && ev.phase === 'surface') {
        const last = this.groups[this.groups.length - 1];
        if (!last || ev.now - last.born > 1.0) this.groups.push({ born: ev.now, c: ev.blowhole.slice(), n: 0 });
        const gid = this.groups.length - 1;
        const strength = Math.exp(-tb / 0.28);
        const n = Math.round(2400 * dt * (0.25 + strength));
        const up = nrm([ev.up[0] * 0.2, 1, ev.up[2] * 0.2]);
        for (let i = 0; i < n; i++) {
          const sd = R() < 0.5 ? -1 : 1;
          const o = ev.blowhole.map((v, j) => v + ev.side[j] * sd * 0.08 + ev.fwd[j] * (R() - 0.5) * 0.08);
          // the two jets diverge only slightly (a humpback's blow is one bushy column)
          const cone = 0.20 * Math.sqrt(R()), ca = R() * 2 * Math.PI;
          const ex = nrm(ev.side), ez = nrm(ev.fwd);
          const dir = nrm(up.map((v, j) => v + ex[j] * (sd * 0.07 + cone * Math.cos(ca)) + ez[j] * (0.04 + cone * Math.sin(ca))));
          const sp = (13 + 10 * R()) * (0.6 + 0.4 * strength);
          if (R() < 0.04) {
            P.add(o, dir.map((v) => v * sp * 0.8), { type: 1, size: 0.02 + 0.03 * R(), life: 2.5, seed: R(), alpha: 0.25, drag: 0.9, grav: 9.81 });
          } else {
            const s0 = 0.10 + 0.10 * R();
            P.add(o, dir.map((v) => v * sp), { type: 0, size: s0, grow: 0.45 + 0.45 * R(), life: 4.5 + 3 * R(), seed: R(), alpha: 0.45 + 0.25 * R(), drag: 2.9 + 1.2 * R(), grav: 0.15, group: gid });
          }
          P.pre(dt, R());
        }
      }
    }
    // ---------------- breach: the exit (a skirt of water dragged up round the body), the landing
    const te = ev.exit;
    if (ev.phase === 'breach') {
      const u = ev.t - te;
      if (u > -0.9 && u < 1.2) {
        for (let i = 0; i < NSPINE; i++) {
          const f = ev.frames[i], s = ev.sect[i];
          const c = f.p.map((v, j) => v + f.R[1][j] * s.mid);
          const sea = ev.sea(i);
          const A = f.R[1][1] * (s.top + s.bot) * 0.5, Bz = f.R[2][1] * s.w, C = sea - c[1];
          const m = Math.hypot(A, Bz);
          if (m < 1e-3 || Math.abs(C) > m + 0.3) continue;
          const vy = ev.vy[i];
          if (vy < 0.5) continue;
          const nEm = Math.round(dt * 420 * Math.min(vy / 8, 1.2));
          for (let k = 0; k < nEm; k++) {
            // a point on the section's waterline, or round the section in the horizontal plane
            const ang = R() * 2 * Math.PI;
            const yy = Math.cos(ang) * (Math.cos(ang) > 0 ? s.top : s.bot) * 1.05, zz = Math.sin(ang) * s.w * 1.05;
            const p = [0, 1, 2].map((j) => c[j] + f.R[1][j] * yy + f.R[2][j] * zz);
            const out = nrm([f.R[1][0] * yy + f.R[2][0] * zz, 0, f.R[1][2] * yy + f.R[2][2] * zz]);
            const vu = vy * (0.35 + 0.65 * R() ** 0.7);
            const vo = 0.6 + 2.6 * R();
            const r = R();
            const pp = [p[0], sea + 0.05, p[2]];
            if (r < 0.45) P.add(pp, [out[0] * vo, vu, out[2] * vo], { type: 2, size: 0.25 + 0.45 * R(), grow: 0.9, life: 1.4 + R(), seed: R(), alpha: 0.6, drag: 0.5, grav: 9.81 });
            else if (r < 0.7) P.add(pp, [out[0] * vo * 1.2, vu * 1.05, out[2] * vo * 1.2], { type: 4, size: 0.15 + 0.35 * R(), life: 3, seed: R(), alpha: 0.4, drag: 0.25, grav: 9.81 });
            else P.add(pp, [out[0] * vo * 1.2, vu * 1.05, out[2] * vo * 1.2], { type: 1, size: 0.04 + 0.1 * R(), life: 3, seed: R(), alpha: 0.6, drag: 0.25, grav: 9.81 });
            P.pre(dt, R());
          }
        }
      }
      // the churning mound of white water round the base while the body stands out of the sea
      if (u > -0.7 && u < 1.6) {
        const n = Math.round(dt * 260);
        for (let k = 0; k < n; k++) {
          const i = Math.floor(R() * NSPINE);
          const f = ev.frames[i], s = ev.sect[i], sea = ev.sea(i);
          const c = f.p.map((v, j) => v + f.R[1][j] * s.mid);
          const reach = Math.abs(f.R[1][1]) * Math.max(s.top, s.bot) + Math.abs(f.R[2][1]) * s.w + Math.abs(f.R[0][1]) * 0.25;
          if (Math.abs(c[1] - sea) > reach) continue;
          const a = R() * 2 * Math.PI, rr = Math.max(s.w, s.top) * (0.9 + 0.8 * R());
          P.add([c[0] + Math.cos(a) * rr, sea + 0.05, c[2] + Math.sin(a) * rr], [Math.cos(a) * (0.5 + 1.5 * R()), 1.0 + 3.0 * R(), Math.sin(a) * (0.5 + 1.5 * R())],
            { type: 2, size: 0.25 + 0.4 * R(), grow: 0.6, life: 1.6, seed: R(), alpha: 0.8, drag: 0.8, grav: 9.81 });
          P.pre(dt, R());
        }
      }
      if (this.crossed(ev, te - 0.6)) { const c = ev.frames[20].p; this.spot(c[0], c[2], 4.5, 1.0, 0, [ev.track.h[0], ev.track.h[1]], 1.0, ev.now); }
      const Ti = 1.69;
      if (this.crossed(ev, te + Ti - 0.12)) this.landing(ev);
      if (u > -0.6 && u < Ti + 0.4) washLevel = 1;
    }
    // ---------------- runoff: water pouring off the freshly emerged body in streams, from the skin
    // that faces down (fixed sources, so each pours as a strand, not as scattered drops)
    {
      const rate = ev.phase === 'breach' ? 9 : 8;      // drops / sheets per second per stream
      for (let i = 0; i < NSPINE; i++) {
        if (ev.dry[i] > 2.2 || ev.rep[i][1] < ev.sea(i) + 0.1) continue;
        const d = SNOUT - SPINE_X[i];
        const o = i * 16, M = ev.mats;
        for (let k = 0; k < 9; k++) {
          const th = ((k + 0.5) / 9 + 0.37 * Math.sin(i * 1.7 + k * 2.3)) * 2 * Math.PI;
          const nW = nrm([0, 1, 2].map((j) => M[o + 4 + j] * Math.cos(th) + M[o + 8 + j] * Math.sin(th)));
          if (nW[1] > -0.2) continue;                   // water leaves where the skin faces down
          const q = ev.bodyPoint(d, th);
          const p = [0, 1, 2].map((j) => M[o + j] * q[0] + M[o + 4 + j] * q[1] + M[o + 8 + j] * q[2] + M[o + 12 + j]);
          if (p[1] < ev.sea(i) + 0.05) continue;
          const n = Math.floor(rate * dt * (0.6 + 0.8 * R()) + R());
          const bv = ev.vel[i];
          for (let m = 0; m < n; m++) {
            const sheet = ev.phase === 'breach' && R() < 0.5;
            P.add(p, [bv[0] * 0.9 + nW[0] * 0.3, bv[1] * 0.9 - 0.2, bv[2] * 0.9 + nW[2] * 0.3], sheet
              ? { type: 4, size: 0.1 + 0.25 * R(), life: 2.5, seed: R(), alpha: 0.28, drag: 0.3, grav: 9.81 }
              : { type: 1, size: 0.03 + 0.05 * R(), life: 2.5, seed: R(), alpha: 0.55, drag: 0.3, grav: 9.81 });
            P.pre(dt, R());
          }
        }
      }
      // flippers in the air stream water off their trailing edges
      if (ev.phase === 'breach') for (let s = 0; s < 2; s++) for (let k = 0; k < 3; k++) {
        const p = ev.pect[s][k];
        if (p[1] < sea0 + 0.3) continue;
        const nn = Math.round(dt * 90);
        for (let m = 0; m < nn; m++) P.add(p.map((x) => x + (R() - 0.5) * 0.6), [(R() - 0.5) * 0.5, -0.5 * R(), (R() - 0.5) * 0.5], R() < 0.5 ? { type: 4, size: 0.06 + 0.12 * R(), life: 2.5, seed: R(), alpha: 0.5, drag: 0.3, grav: 9.81 } : { type: 1, size: 0.03 + 0.05 * R(), life: 2.5, seed: R(), alpha: 0.5, drag: 0.3, grav: 9.81 });
      }
    }
    // ---------------- raised flukes: water leaves the trailing edge as a few thick ropes and sheets
    // (irregular spacing and flow), which break into drops, plus a wide thin veil of fine drops
    {
      const edge = ev.flukeEdge, ne = edge.length;
      if (!this.ropes || this.ropeCycle !== ev.cycle) {
        this.ropeCycle = ev.cycle;
        const rr = rng(97 + ev.cycle * 13);
        this.ropes = Array.from({ length: 6 }, () => ({ f: 0.04 + 0.92 * rr(), q: 0.3 + 1.2 * rr() * rr(), w: 0.025 + 0.09 * rr() * rr() }));
      }
      if (this.prevEdge && dt > 0) {
        const at = (f) => { const x = f * (ne - 1), i = Math.min(Math.floor(x), ne - 2), u = x - i; return [0, 1, 2].map((c) => edge[i].p[c] + (edge[i + 1].p[c] - edge[i].p[c]) * u); };
        const velAt = (f) => { const i = Math.min(Math.round(f * (ne - 1)), ne - 1); return edge[i].p.map((v, c) => (v - this.prevEdge[i][c]) / dt); };
        const sea = ev.sea(30);
        for (const r of this.ropes) {
          const p = at(r.f);
          if (p[1] < sea + 0.15) continue;
          const v = velAt(r.f);
          const n = Math.floor(dt * 80 * r.q + R());
          for (let k = 0; k < n; k++) {
            P.add(p.map((x, c) => x + (c === 1 ? 0 : (R() - 0.5) * r.w * 0.6)), [v[0] * 0.8 + (R() - 0.5) * 0.15, Math.min(v[1] * 0.6, 0) - 0.4, v[2] * 0.8 + (R() - 0.5) * 0.15],
              { type: 4, size: r.w * (0.6 + 0.6 * R()), life: 3, seed: R(), alpha: 0.6, drag: 0.1, grav: 9.81 });
            P.pre(dt, R());
          }
        }
        // the veil: sparse fine drops along the whole edge, with some larger ones
        const nv = Math.floor(dt * 260 + R());
        for (let k = 0; k < nv; k++) {
          const f = R(), p = at(f);
          if (p[1] < sea + 0.15) continue;
          const v = velAt(f);
          const big = R() < 0.15;
          P.add(p, [v[0] * 0.8 + (R() - 0.5) * 0.4, Math.min(v[1] * 0.6, 0) - 0.3 * R(), v[2] * 0.8 + (R() - 0.5) * 0.4],
            { type: 1, size: big ? 0.03 + 0.03 * R() : 0.012 + 0.02 * R(), life: 3, seed: R(), alpha: big ? 0.5 : 0.3, drag: 0.15, grav: 9.81 });
          P.pre(dt, R());
        }
      }
      this.prevEdge = edge.map((e) => e.p.slice());
    }
    // ---------------- foam memory: wash patches while the body works at the surface, fluke prints
    let crossing = 0, cx = 0, cz = 0;
    for (let i = 0; i < NSPINE; i++) {
      const f = ev.frames[i], s = ev.sect[i];
      const cY = f.p[1] + f.R[1][1] * s.mid;
      const reach = Math.abs(f.R[1][1]) * Math.max(s.top, s.bot) + Math.abs(f.R[2][1]) * s.w;
      const sea = ev.sea(i);
      const cut = Math.abs(cY - sea) < reach;
      // white water where the body cuts the surface: stronger when it moves through it fast
      const w = cut ? Math.min(0.06 + Math.abs(ev.vy[i]) * 0.3, 1.6) * sstep(reach, reach * 0.3, Math.abs(cY - sea)) : 0;
      this.sec.C.set([f.p[0] + f.R[1][0] * s.mid, cY, f.p[2] + f.R[1][2] * s.mid, s.top], i * 4);
      this.sec.X.set([f.R[0][0], f.R[0][1], f.R[0][2], s.bot], i * 4);
      this.sec.Y.set([f.R[1][0], f.R[1][1], f.R[1][2], s.w], i * 4);
      this.sec.W[i] = w;
      if (w > 0.35) { crossing += w; cx += f.p[0] * w; cz += f.p[2] * w; }
    }
    // (a calm surfacing leaves only a smooth swirl; white water only where the body moves fast through the surface)
    const fast = ev.vy.slice(0, NSPINE).reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    if (crossing > 1.5 && ev.now - this.lastWashSpot > 0.6) {
      this.lastWashSpot = ev.now;
      const st = Math.min(crossing / 8, 0.7) * sstep(2.5, 5.0, fast);
      if (st > 0.05) this.spot(cx / crossing, cz / crossing, 3.0 + 0.4 * R(), st, 1, [ev.track.h[0], ev.track.h[1]], 2.2, ev.now);
      else this.spot(cx / crossing, cz / crossing, 3.5, 0.35, 2, [ev.track.h[0], ev.track.h[1]], 2.0, ev.now);
    }
    if (ev.phase === 'surface') for (const b of ev.blows.slice(0, 2)) if (this.crossed(ev, b + 6.8)) { const p = ev.frames[1].p; this.spot(p[0], p[2], 4.2, 0.8, 2, [ev.track.h[0], ev.track.h[1]], 1.4, ev.now); }
    if (ev.phase === 'dive' && this.crossed(ev, ev.tDive + 6.5)) { const p = ev.flukeEdge[12].p; this.spot(p[0], p[2], 5.5, 1.0, 2, [1, 0], 1.0, ev.now); this.spot(p[0], p[2], 3.0, 0.5, 1, [1, 0], 1.0, ev.now); }
    if (ev.phase === 'dive') {
      // flukes slipping under leave a swirl of white water
      const fk = ev.flukeEdge[12].p;
      if (Math.abs(fk[1] - ev.sea(30)) < 0.6 && ev.now - this.lastWashSpot > 0.4) { this.lastWashSpot = ev.now; this.spot(fk[0], fk[2], 2.0, 0.35, 1, [1, 0], 1.3, ev.now); }
    }
    for (let s = 0; s < 2; s++) for (let k = 0; k < 3; k++) {
      const p = ev.pect[s][k];
      this.pect.set([p[0], p[1], p[2], 0.55 - 0.12 * k], (s * 3 + k) * 4);
    }
    // ---------------- integrate
    const g = this.groups;
    for (const grp of g) { grp.sum = [0, 0, 0]; grp.cnt = 0; }
    const t = ev.now;
    for (let i = P.n - 1; i >= 0; i--) {
      P.age[i] += dt;
      if (P.age[i] < 0) continue;                       // (delayed: not born yet)
      const o = i * 3;
      const ty = P.type[i];
      if (P.age[i] > P.life[i] || ((ty === 1 || ty === 2 || ty === 4) && P.p[o + 1] < sea0 - 0.1 && P.v[o + 1] < 0)) { P.kill(i); continue; }
      const kd = P.drag[i];
      const sd = P.seed[i] * 50;
      // turbulence for the clouds: slowly varying random accelerations
      const turb = ty === 0 || ty === 3 ? 1.1 : ty === 2 ? 0.5 : 0;
      const tx = turb * Math.sin(sd + t * 1.7), ty2 = turb * 0.6 * Math.sin(sd * 1.3 + t * 2.1), tz = turb * Math.cos(sd * 0.7 + t * 1.5);
      for (let c = 0; c < 3; c++) {
        const acc = -(P.v[o + c] - air[c]) * kd + (c === 0 ? tx : c === 1 ? ty2 - P.grav[i] : tz);
        P.v[o + c] += acc * dt;
        P.p[o + c] += P.v[o + c] * dt;
      }
      P.size[i] += P.grow[i] * dt * (ty === 0 ? Math.exp(-P.age[i] / 2.2) : ty === 2 ? Math.exp(-P.age[i] / 1.2) : 1);
      const gi = P.group[i];
      if (gi >= 0 && g[gi]) { const G = g[gi]; G.sum[0] += P.p[o]; G.sum[1] += P.p[o + 1]; G.sum[2] += P.p[o + 2]; G.cnt++; }
    }
    // the blow's own shadow: optical depth towards the sun through an ellipsoid round the cloud
    const sun = env.sunDir || [0, 1, 0];
    for (const grp of g) if (grp.cnt) { grp.c = grp.sum.map((v) => v / grp.cnt); }
    for (let i = 0; i < P.n; i++) {
      const gi = P.group[i];
      if (gi < 0 || !g[gi] || !g[gi].cnt) continue;
      const G = g[gi], o = i * 3;
      const age = ev.now - G.born;
      const Rc = 0.5 + 0.45 * age, Hc = 1.3 + 0.3 * age;
      const rel = [(P.p[o] - G.c[0]) / Rc, (P.p[o + 1] - G.c[1]) / Hc, (P.p[o + 2] - G.c[2]) / Rc];
      const sl = nrm([sun[0] / Rc, sun[1] / Hc, sun[2] / Rc]);
      const along = rel[0] * sl[0] + rel[1] * sl[1] + rel[2] * sl[2];
      P.shade[i] = Math.exp(-Math.max(1 - along, 0) * 1.4 * Math.exp(-age / 2.0));
    }
    // ---------------- pack (opacity: a puff thins as it spreads, its droplets evaporate)
    const Gp = this.gpu;
    for (let i = 0; i < P.n; i++) {
      const o = i * 16, k = i * 3;
      const life = P.life[i], age = P.age[i];
      let a = P.a0[i];
      const spread = Math.pow(P.s0[i] / Math.max(P.size[i], 1e-3), 1.3);
      if (age < 0) a = 0;
      else if (P.type[i] === 0) a *= Math.pow(P.s0[i] / Math.max(P.size[i], 1e-3), 0.9) * Math.exp(-age / 3.0) * sstep(0, 0.04, age) * (1 - sstep(life * 0.45, life, age));
      else if (P.type[i] === 3) a *= Math.pow(P.s0[i] / P.size[i], 1.0) * sstep(0, 0.6, age) * (1 - sstep(life * 0.5, life, age));
      else if (P.type[i] === 2) a *= Math.pow(P.s0[i] / P.size[i], 0.8) * Math.exp(-age / 1.4) * (1 - sstep(life * 0.6, life, age));
      else a *= 1 - sstep(life * 0.8, life, age);
      Gp[o] = P.p[k]; Gp[o + 1] = P.p[k + 1]; Gp[o + 2] = P.p[k + 2]; Gp[o + 3] = P.size[i];
      Gp[o + 4] = P.v[k]; Gp[o + 5] = P.v[k + 1]; Gp[o + 6] = P.v[k + 2]; Gp[o + 7] = P.type[i];
      Gp[o + 8] = Math.max(age, 0); Gp[o + 9] = life; Gp[o + 10] = P.seed[i]; Gp[o + 11] = a;
      Gp[o + 12] = P.shade[i]; Gp[o + 13] = sea0; Gp[o + 14] = 0; Gp[o + 15] = 0;
    }
    // ---------------- spots -> uniforms
    this.spots = this.spots.filter((s) => ev.now - s.born < (s.type === 2 ? 120 : 110));
    this.spotA.fill(0); this.spotB.fill(0); this.spotT.fill(0);
    this.spots.forEach((s, i) => {
      this.spotA.set([s.x, s.z, s.r, ev.now - s.born], i * 4);
      const l = Math.hypot(s.dir[0], s.dir[1]) || 1;
      this.spotB.set([s.s, s.dir[0] / l, s.dir[1] / l, s.aspect], i * 4);
      this.spotT[i] = s.type;
    });
    this.ev = ev;
    this.active = this.spots.length > 0 || this.sec.W.some((w) => w > 0) || ev.pect.some((f) => f.some((p) => p[1] > ev.sea(28) - 6));
    return washLevel;
  }

  // the landing: an explosion of white water along the body's footprint, heaviest under the chest,
  // thrown out sideways and up (the crown over the chest reaches ~10 m), then a thin mist that hangs
  landing(ev) {
    const R = this.r, P = this.pool;
    const fr = ev.frames;
    const ax = nrm([fr[NSPINE - 1].p[0] - fr[0].p[0], 0, fr[NSPINE - 1].p[2] - fr[0].p[2]]);
    const at = (i, sd, wf) => {
      const f = fr[i], s = ev.sect[i];
      const lat = [-ax[2] * sd, 0, ax[0] * sd];
      return { p: [f.p[0] + lat[0] * s.w * wf, ev.sea(i) + 0.05, f.p[2] + lat[2] * s.w * wf], lat };
    };
    // where along the body: mostly the chest and the flippers, less at the tail
    const pick = () => Math.min(Math.max(Math.round(16 + 6.5 * (R() + R() + R() - 1.5) * 1.6), 0), NSPINE - 1);
    for (let k = 0; k < 3400; k++) {
      const i = pick(), sd = R() < 0.5 ? -1 : 1;
      const { p, lat } = at(i, sd, 0.5 + 0.7 * R());
      const mid = Math.exp(-(((i - 16) / 8) ** 2));
      // a fan: steep in the middle of the footprint, flatter at the ends; heavy-tailed heights
      const up = (4 + 11 * Math.pow(R(), 1.5)) * (0.35 + 0.75 * mid);
      const outv = 1.5 + 6 * R();
      const v = [lat[0] * outv + ax[0] * (R() - 0.5) * 4, up, lat[2] * outv + ax[2] * (R() - 0.5) * 4];
      const delay = -0.08 * R();
      if (R() < 0.45) P.add(p, v, R() < 0.5 ? { type: 4, size: 0.12 + 0.3 * R(), life: 4, seed: R(), alpha: 0.6, drag: 0.22, grav: 9.81 } : { type: 1, size: 0.04 + 0.14 * R() * R(), life: 4, seed: R(), alpha: 0.7, drag: 0.22, grav: 9.81 });
      else P.add(p, v.map((x) => x * 0.8), { type: 2, size: 0.18 + 0.32 * R(), grow: 0.8, life: 2.4 + 1.2 * R(), seed: R(), alpha: 0.85, drag: 0.55 + 0.4 * R(), grav: 9.81 });
      P.age[P.n - 1] = delay;
    }
    // the crown over the chest
    const c = fr[17].p;
    for (let k = 0; k < 500; k++) {
      const a = R() * 2 * Math.PI, rr = 1.0 + 2.2 * R();
      const p = [c[0] + Math.cos(a) * rr, ev.sea(17) + 0.2, c[2] + Math.sin(a) * rr];
      const v = [Math.cos(a) * (1 + 2.5 * R()), 9 + 9 * Math.pow(R(), 0.8), Math.sin(a) * (1 + 2.5 * R())];
      if (R() < 0.55) P.add(p, v, { type: 2, size: 0.2 + 0.35 * R(), grow: 0.7, life: 2.6, seed: R(), alpha: 0.8, drag: 0.45, grav: 9.81 });
      else P.add(p, v, { type: 1, size: 0.1 + 0.25 * R(), life: 4, seed: R(), alpha: 0.8, drag: 0.18, grav: 9.81 });
      P.age[P.n - 1] = 0.05 - 0.15 * R();
    }
    // the mist that hangs over the splash once the water has fallen (born late, thin, drifting)
    for (let k = 0; k < 70; k++) {
      const i = pick();
      const { p } = at(i, R() < 0.5 ? -1 : 1, 1.5 * R());
      p[1] += 0.5 + 6 * Math.pow(R(), 1.5);
      P.add(p, [(R() - 0.5) * 1.5, 0.6 * R(), (R() - 0.5) * 1.5], { type: 3, size: 1.0 + 1.4 * R(), grow: 0.6, life: 4 + 3 * R(), seed: R(), alpha: 0.07 + 0.05 * R(), drag: 1.2, grav: 0.15 });
      P.age[P.n - 1] = -0.3 - 0.9 * R();
    }
    const m = fr[14].p;
    this.spot(m[0], m[2], 8.5, 1.6, 0, [ax[0], ax[2]], 1.7, ev.now);
  }

  // the sea-surface pass on the water's own polar grid, over the region the whale uses
  surfaceMesh(u) {
    const gl = this.gl;
    const cam = u.uCamPos, ang = u.uAngStep;
    const c = this.ev.track.c;
    const rx = c[0] - cam[0], rz = c[1] - cam[2];
    const R0 = Math.hypot(rx, rz), th0 = Math.atan2(rz, rx);
    const half = 62;
    const key = [Math.round(R0 / 4), Math.round(th0 / ang), ang].join(',');
    if (this.surface && this.surface.key === key) return this.surface;
    // the water's rings: r_i = a (q^i - 1), q = 1 + ang, a = 0.03 / ang
    const q = 1 + ang, a = 0.03 / ang;
    const ringR = (i) => a * (Math.pow(q, i) - 1);
    const ringI = (r) => Math.log(r / a + 1) / Math.log(q);
    const i0 = Math.max(Math.floor(ringI(Math.max(R0 - half, 1))) - 1, 0), i1 = Math.ceil(ringI(R0 + half)) + 1;
    const dth = Math.asin(Math.min(half / Math.max(R0, half + 1), 1)) + 0.02;
    const j0 = Math.floor((th0 - dth) / ang), j1 = Math.ceil((th0 + dth) / ang);
    const nj = j1 - j0 + 1, ni = i1 - i0 + 1;
    const verts = new Float32Array(ni * nj * 2);
    let o = 0;
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
      // (the water's own angle values: j * ang over [0, 2 pi])
      const jj = ((j % Math.round(2 * Math.PI / ang)) + Math.round(2 * Math.PI / ang)) % Math.round(2 * Math.PI / ang);
      verts[o++] = ringR(i); verts[o++] = jj * ang;
    }
    const idx = new Uint32Array((ni - 1) * (nj - 1) * 6);
    o = 0;
    for (let i = 0; i < ni - 1; i++) for (let j = 0; j < nj - 1; j++) {
      const a0 = i * nj + j, a1 = a0 + 1, b0 = a0 + nj, b1 = b0 + 1;
      idx[o++] = a0; idx[o++] = b0; idx[o++] = a1;
      idx[o++] = a1; idx[o++] = b0; idx[o++] = b1;
    }
    if (this.surface) { gl.deleteVertexArray(this.surface.vao); }
    this.surface = { ...makeMesh(gl, [[verts, 2]], idx), key };
    return this.surface;
  }

  draw(u, fxU) {
    const gl = this.gl;
    // ---- sea surface
    if (this.active && this.ev) {
      const m = this.surfaceMesh(u);
      this.sprog.use({
        ...u, uSpotA: this.spotA, uSpotB: this.spotB, uSpotT: this.spotT,
        uSecC: this.sec.C, uSecX: this.sec.X, uSecY: this.sec.Y, uSecW: this.sec.W, uPect: this.pect, uSpacing: SPINE_X[1] - SPINE_X[0],
      });
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.depthMask(false);
      gl.enable(gl.POLYGON_OFFSET_FILL);
      gl.polygonOffset(-1, -2);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(m.vao);
      gl.drawElements(gl.TRIANGLES, m.count, m.itype, 0);
      gl.bindVertexArray(null);
      gl.disable(gl.POLYGON_OFFSET_FILL);
      gl.depthFunc(gl.LESS);
    }
    // ---- particles
    const P = this.pool;
    if (P.n > 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.gpu, 0, P.n * 16);
      this.pprog.use({ ...fxU.skyU, uViewProj: fxU.viewProj, uCamPos: fxU.camPos, uPixelAngle: fxU.pixelAngle, uFlash: fxU.flash });
      gl.enable(gl.DEPTH_TEST);
      gl.depthMask(false);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(this.vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, P.n);
      gl.bindVertexArray(null);
    }
    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }
}
