// Where the boat meets the sea.
//
// A world-aligned field around the boat (1024^2 texels over 44 m, re-rendered every frame from
// analytic sources, no feedback) holds what the sea shader adds: r,g slope (surface normal
// perturbation), b foam (+) or shading (-), a calm (the slick that smooths short waves). Sources:
//   - the track: Kelvin wake (divergent + transverse waves at the boat's speed) along the curved
//     path, the wake slick, and stern-wash bubbles;
//   - the hull: contact foam round the waterline, the bow wave and the water darkened next to the
//     hull (it hides the sky the water would reflect);
//   - oar puddles: each stroke leaves two counter-rotating tip vortices one blade-length apart -
//     dimples with spiral foam that drift apart, smooth out and fade over ~20 s;
//   - ripple rings from drips and from the blade entering.
// Spray particles (catch splash, sheet off the blade at the release, drips on the recovery) are
// drawn after the sea as lit, motion-blurred water drops.
import { Program, Texture, Target } from '../gl.js';
import { SKY_LOOKUP_GLSL } from '../sky/atmosphere.js';
import { HULL_GLSL, XS, XB } from './hull.js';

export const FX_RES = 1024, FX_SIZE = 44;
const NTRAIL = 64, NPUD = 16, NRING = 48;

// GLSL for the sea shader (src/ocean/water.js): the boat's field at a world position
export const BOAT_WATER_GLSL = HULL_GLSL + `
uniform sampler2D uBoatFx;
uniform vec4 uBoatFxRect;      // origin xz, 1 / size (0: off), gain (debug, 1)
vec4 boatFx(vec2 xz) {
  vec2 uv = (xz - uBoatFxRect.xy) * uBoatFxRect.z;
  if (uBoatFxRect.z <= 0.0 || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return vec4(0.0);
  float edge = smoothstep(0.0, 0.03, min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y)));
  return texture(uBoatFx, uv) * edge * uBoatFxRect.w;
}
uniform mat4 uBoatW2L;         // world -> boat-local
uniform mat4 uBoatL2W;         // boat-local -> world
uniform vec4 uBoatRower;       // rower proxy (boat-local): centre x, top y, half length, half width
// The boat mirrored in the sea: the reflected ray is marched through a proxy of the hull (its real
// lines, solid) and the rower (a box); a hit returns the lit paint instead of the sky.
vec3 boatReflect1(vec3 P, vec3 R, vec3 sky) {
  if (uBoatFxRect.z <= 0.0 || R.y <= 0.0) return sky;
  vec3 o = (uBoatW2L * vec4(P, 1.0)).xyz;
  vec3 r = mat3(uBoatW2L) * R;
  vec3 bmin = vec3(XS - 0.05, -0.1, -0.8), bmax = vec3(XB + 0.05, max(uBoatRower.y, 0.65), 0.8);
  vec3 inv = 1.0 / (r + vec3(1e-6));
  vec3 t0 = (bmin - o) * inv, t1 = (bmax - o) * inv;
  vec3 tmn = min(t0, t1), tmx = max(t0, t1);
  float ta = max(max(tmn.x, tmn.y), max(tmn.z, 0.0)), tb = min(min(tmx.x, tmx.y), tmx.z);
  if (tb <= ta || ta > 60.0) return sky;
  float dt = (tb - ta) / 28.0;
  for (int i = 0; i < 28; i++) {
    float t = ta + dt * (float(i) + 0.5);
    vec3 q = o + r * t;
    float u = hullU(q.x);
    bool hull = u > 0.0 && u < 1.0 && q.y < hSheer(u) + 0.02 && q.y > hKeel(u) - 0.05 && abs(q.z) < innerHalfWidth(u, q.y) + 0.024;
    bool man = uBoatRower.z > 0.0 && abs(q.x - uBoatRower.x) < uBoatRower.z && abs(q.z) < uBoatRower.w && q.y > 0.15 && q.y < uBoatRower.y;
    if (!hull && !man) continue;
    // refine the hit along the ray (the paint bands need it)
    float a = t - dt, b = t;
    for (int k = 0; k < 5; k++) {
      float m = 0.5 * (a + b);
      vec3 qm = o + r * m;
      float um = hullU(qm.x);
      bool h2 = um > 0.0 && um < 1.0 && qm.y < hSheer(um) + 0.02 && qm.y > hKeel(um) - 0.05 && abs(qm.z) < innerHalfWidth(um, qm.y) + 0.024;
      bool m2 = uBoatRower.z > 0.0 && abs(qm.x - uBoatRower.x) < uBoatRower.z && abs(qm.z) < uBoatRower.w && qm.y > 0.15 && qm.y < uBoatRower.y;
      if (h2 || m2) b = m; else a = m;
    }
    q = o + r * b;
    u = clamp(hullU(q.x), 0.0, 1.0);
    vec3 alb;
    vec3 nL;
    if (!hull) { alb = vec3(0.45, 0.3, 0.045); nL = normalize(vec3(-r.x, 0.0, -r.z)); }
    else {
      float G = hSheer(u);
      alb = q.y < 0.03 ? vec3(0.05, 0.018, 0.013) : q.y > G - 0.075 ? vec3(0.028, 0.085, 0.06) : vec3(0.58, 0.56, 0.48);
      if (q.y > G - 0.012) alb = vec3(0.3, 0.22, 0.13);
      nL = normalize(vec3(0.0, -0.3, sign(q.z)));
    }
    vec3 n = normalize(mat3(uBoatL2W) * nL);
    vec3 E = sunIlluminance() * max(dot(n, uSunDir), 0.0) * uHazeSun + skyIrradiance() * (0.45 + 0.25 * n.y);
    return alb * E / 3.14159265;
  }
  return sky;
}
// two rays a little apart (the facets in a pixel do not all agree): softer, less aliased edges
vec3 boatReflect(vec3 P, vec3 R, vec3 sky) {
  if (uBoatFxRect.z <= 0.0 || R.y <= 0.0) return sky;
  vec3 t = normalize(cross(R, vec3(0.0, 1.0, 0.0)) + vec3(1e-5));
  vec3 b = cross(t, R);
  float s = 0.012;
  return 0.5 * (boatReflect1(P, normalize(R + (t + b) * s), sky) + boatReflect1(P, normalize(R - (t + b) * s), sky));
}`;

const NOISE = `
float wh(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float wn(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(wh(i), wh(i + vec2(1, 0)), f.x), mix(wh(i + vec2(0, 1)), wh(i + vec2(1, 1)), f.x), f.y);
}
float wfbm(vec2 p) { return 0.5 * wn(p) + 0.25 * wn(p * 2.1 + 3.1) + 0.125 * wn(p * 4.3 + 7.7) + 0.0625 * wn(p * 8.7 + 1.3); }
// foam: bubbly, cellular lace thresholded by an amount
float lace(vec2 p, float amount) {
  float n = wfbm(p) * 0.65 + wn(p * 3.7) * 0.35;
  return smoothstep(1.0 - amount, 1.0 - amount + 0.25, n);
}`;

// ---------------------------------------------------------------- pass 1: the track (fullscreen)
const FULL_VS = `
uniform vec4 uRect;
out vec2 vXZ;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vXZ = uRect.xy + p * uRect.z;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const TRAIL_FS = NOISE + `
in vec2 vXZ;
out vec4 o;
uniform vec4 uTrail[${NTRAIL}];  // xz, distance along the track from the bow (m), age (s)
uniform int uTrailN;
uniform vec2 uSpeed;              // mean speed (m/s), boat length (m)
uniform float uTime;
const float G = 9.81;
void main() {
  vec2 q = vXZ;
  // nearest point of the track polyline
  float best = 1e9, sAt = 0.0, lat = 0.0, age = 0.0;
  vec2 tan = vec2(1.0, 0.0);
  for (int i = 0; i < ${NTRAIL - 1}; i++) {
    if (i >= uTrailN - 1) break;
    vec4 a = uTrail[i], b = uTrail[i + 1];
    vec2 ab = b.xy - a.xy;
    float l2 = max(dot(ab, ab), 1e-6);
    float h = clamp(dot(q - a.xy, ab) / l2, 0.0, 1.0);
    vec2 d = q - (a.xy + ab * h);
    float dd = dot(d, d);
    if (dd < best) {
      best = dd; sAt = mix(a.z, b.z, h); age = mix(a.w, b.w, h);
      tan = ab / sqrt(l2);
      lat = d.x * tan.y - d.y * tan.x;
    }
  }
  float dist = sqrt(best);
  if (dist > 14.0) { o = vec4(0.0); return; }
  // tan points from newer to older (backwards along the track): "s" grows behind the bow
  float s = sAt;
  vec2 back = tan, side = vec2(tan.y, -tan.x);
  vec2 slope = vec2(0.0);
  // Kelvin wake: for x behind the bow and y abeam, waves of heading t satisfy
  // y / x = tan t / (1 + 2 tan^2 t); phase k0 (x cos t + y sin t) / cos^2 t, k0 = g / U^2
  float U = max(uSpeed.x, 0.3), k0 = G / (U * U);
  float x = s, y = abs(lat);
  if (x > 0.15 && x < 16.0) {
    float r = y / x;
    float disc = 1.0 - 8.0 * r * r;
    float wedge = smoothstep(0.0, 0.12, disc);
    float sq = sqrt(max(disc, 0.0));
    float decay = exp(-x / 7.0) / sqrt(1.0 + 0.5 * x);
    // (slope amplitude: a few hundredths for a rowing boat)
    float S = 0.055 * min(U * U / 1.2, 1.6) * decay;
    for (int k = 0; k < 2; k++) {
      float T = r < 1e-3 ? (k == 0 ? 0.0 : 1e3) : (k == 0 ? (1.0 - sq) / (4.0 * r) : (1.0 + sq) / (4.0 * r));
      T = min(T, 3.0);
      float c = 1.0 / sqrt(1.0 + T * T), sn = T * c;
      float kk = k0 / max(c * c, 0.2);
      float ph = kk * (x * c + y * sn) + 0.6;
      // transverse waves are weak behind a small, slow hull; divergent ones carry the V
      float amp = S * (k == 0 ? 0.4 : 1.0) * wedge * (1.0 + 1.2 * exp(-max(disc, 0.0) * 8.0));
      vec2 dirL = vec2(c, sn * sign(lat + 1e-6));        // (back, side) components
      float g = -amp * sin(ph);
      slope += (back * dirL.x + side * dirL.y) * g;
    }
  }
  // wake slick (smoothed short waves) and stern wash bubbles
  float sb = s - uSpeed.y;
  float w = 0.55 + 0.05 * max(sb, 0.0);
  float calm = exp(-pow(abs(lat) / w, 2.0)) * smoothstep(-0.5, 0.5, sb) * exp(-age / 70.0) * 0.8;
  float wash = exp(-pow(abs(lat) / (0.25 + 0.04 * max(sb, 0.0)), 2.0)) * smoothstep(-0.3, 0.2, sb) * exp(-max(sb, 0.0) / (U * 2.2));
  vec2 fq = q * 9.0 + back * 0.0;
  float foam = lace(fq, 0.55 * wash) * wash;
  o = vec4(slope, foam * 0.8, calm);
}`;

// ---------------------------------------------------------------- pass 2: splats (additive)
const SPLAT_VS = `
uniform vec4 uRect;
uniform int uKind;               // 0 hull, 1 puddles, 2 rings
uniform mat4 uBoat;              // boat-local -> world
uniform vec4 uPud[${NPUD * 2}];
uniform vec4 uRing[${NRING}];
out vec2 vXZ;
flat out int vI;
void main() {
  int v = gl_VertexID % 6;
  vec2 c = v == 0 ? vec2(-1, -1) : v == 1 ? vec2(1, -1) : v == 2 ? vec2(1, 1) : v == 3 ? vec2(-1, -1) : v == 4 ? vec2(1, 1) : vec2(-1, 1);
  vec2 xz;
  vI = gl_InstanceID;
  if (uKind == 0) {
    vec2 l = vec2(mix(-3.2, 3.6, c.x * 0.5 + 0.5), c.y * 2.4);
    xz = (uBoat * vec4(l.x, 0.0, l.y, 1.0)).xz;
  } else if (uKind == 1) {
    vec4 a = uPud[gl_InstanceID * 2];
    xz = a.xy + c * 1.1;
  } else {
    vec4 r = uRing[gl_InstanceID];
    xz = r.xy + c * (0.12 + 0.3 * r.z + 0.05);
  }
  vXZ = xz;
  vec2 p = (xz - uRect.xy) / uRect.z;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const SPLAT_FS = HULL_GLSL + NOISE + `
in vec2 vXZ;
flat in int vI;
out vec4 o;
uniform int uKind;
uniform mat4 uBoatInv;
uniform mat4 uBoat;
uniform vec4 uPud[${NPUD * 2}];   // (x, z, dir x, dir z), (age, strength, act, seed)
uniform vec4 uRing[${NRING}];      // (x, z, age, amplitude)
uniform vec3 uWaterPlane;
uniform vec4 uMotion;              // speed (m/s), heave velocity, pitch rate, time
// outer half-breadth of the hull at local height y
float outerHalf(float x, float y) {
  float u = hullU(x);
  if (u <= 0.0 || u >= 1.0) return -1.0;
  if (y < hKeel(u) - 0.02) return -1.0;
  return innerHalfWidth(u, y) + 0.022;
}
void main() {
  if (uKind == 0) {
    vec3 lp = (uBoatInv * vec4(vXZ.x, 0.0, vXZ.y, 1.0)).xyz;
    float wl = uWaterPlane.x + uWaterPlane.y * lp.x + uWaterPlane.z * lp.z;
    float hw = outerHalf(clamp(lp.x, XS + 0.001, XB - 0.001), wl);
    // distance to the waterline outline (approximate: across, then past the ends)
    float ax = lp.x < XS ? XS - lp.x : (lp.x > XB ? lp.x - XB : 0.0);
    float d = hw > 0.0 ? abs(lp.z) - hw : length(vec2(ax, lp.z));
    d = max(d, ax);
    float U = uMotion.x;
    float bow = smoothstep(0.55, 0.92, hullU(lp.x));
    // contact foam: a ragged band hugging the hull, thicker at the bow and when the hull pumps
    float pump = clamp(abs(uMotion.y) * 3.0 + abs(uMotion.z) * 1.5, 0.0, 1.0);
    float band = exp(-max(d, 0.0) / (0.05 + 0.07 * bow * U + 0.04 * pump));
    float n = wfbm(vXZ * 14.0 + uMotion.w * vec2(0.4, -0.3));
    float foam = band * smoothstep(0.3, 0.7, n + 0.3 * band + 0.15 * bow * U + 0.2 * pump) * (0.7 + 0.7 * bow * U);
    // the hull hides the sky from the water beside it (and reflects in it, darker)
    float shadeH = -0.45 * exp(-max(d, 0.0) / 0.22) * step(0.0, d + 0.02);
    // bow wave: a crest pushed ahead and out from the stem, sliding aft along the sides
    vec2 slope = vec2(0.0);
    {
      float sx = lp.x - XB;
      float r = d;
      float A = 0.018 * U * U * (0.4 + bow);
      float crest = exp(-pow((r - 0.07 - 0.05 * U) / 0.06, 2.0)) - 0.6 * exp(-pow((r - 0.2 - 0.06 * U) / 0.07, 2.0));
      float along = smoothstep(-2.2, -0.3, sx) * (1.0 - smoothstep(0.0, 0.5, sx));
      float h = A * crest * along;
      // gradient across the hull (outwards in z, forwards at the bow)
      float dr = A * along * (-2.0 * (r - 0.07 - 0.05 * U) / 0.0036 * exp(-pow((r - 0.07 - 0.05 * U) / 0.06, 2.0)) + 1.2 * (r - 0.2 - 0.06 * U) / 0.0049 * exp(-pow((r - 0.2 - 0.06 * U) / 0.07, 2.0)));
      vec2 nrm = hw > 0.0 ? vec2(0.0, sign(lp.z)) : normalize(vec2(lp.x - XB, lp.z) + 1e-5);
      vec2 gl = nrm * dr;
      slope = (mat3(uBoat) * vec3(gl.x, 0.0, gl.y)).xz;
      foam += 0.5 * smoothstep(0.004, 0.012, h) * smoothstep(0.4, 0.7, n) * U;
    }
    float inside = step(d, -0.01);
    o = vec4(slope, foam + shadeH, 0.25 * band) * (1.0 - inside);
    return;
  }
  if (uKind == 1) {
    vec4 a = uPud[vI * 2], b = uPud[vI * 2 + 1];
    float age = b.x, str = b.y, act = b.z, seed = b.w;
    vec2 dir = normalize(a.zw), perp = vec2(-dir.y, dir.x);
    vec2 slope = vec2(0.0);
    float foam = 0.0, calm = 0.0;
    float grow = 1.0 + age * 0.12;
    float sep = 0.21 + age * 0.012;
    for (int k = 0; k < 2; k++) {
      float sgn = k == 0 ? -1.0 : 1.0;
      vec2 c = a.xy + dir * sep * sgn;
      vec2 d = vXZ - c;
      float r = length(d);
      float R = 0.1 * grow;
      // dimple: the vortex core pulls the surface down
      float A = 0.075 * str * exp(-age / 8.0) * (1.0 - 0.6 * act);   // (a puddle dimples a few cm)
      float e = exp(-r * r / (R * R));
      slope += d * (2.0 * A / (R * R)) * e;
      // spiral foam: tracer pattern advected by a Lamb-Oseen vortex
      float rc = 0.06 * grow;
      float om = 1.6 * str / (r * r + rc * rc) * (1.0 - exp(-r * r / (rc * rc))) * 0.08;
      float th = atan(d.y, d.x) + sgn * om * min(age, 12.0) * 6.0;
      vec2 pp = vec2(cos(th), sin(th)) * r;
      float ring = exp(-pow((r - 0.13 * grow) / (0.1 * grow), 2.0));
      float amt = (0.55 * exp(-age / 1.8) + 0.2 * exp(-age / 8.0)) * str;
      foam += lace(pp * 22.0 + seed * 7.0 + vec2(sgn * 3.1, 0.0), 0.55 * amt) * ring * amt;
      calm += exp(-r * r / (0.34 * 0.34 * grow * grow)) * exp(-age / 22.0) * 0.9;
    }
    // churned, aerated water while the blade is pulling through (and right after)
    vec2 dl = vXZ - a.xy;
    float along = dot(dl, dir), acr = dot(dl, perp);
    float churn = exp(-pow(along / 0.32, 2.0) - pow(acr / 0.16, 2.0)) * (act + exp(-age / 0.8) * (1.0 - act));
    float cn = wfbm(vXZ * 13.0 + vec2(seed, uMotion.w * 3.0));
    foam += churn * smoothstep(0.45, 0.75, cn) * 0.8 * str;
    slope += (vec2(wn(vXZ * 25.0 + uMotion.w * 6.0), wn(vXZ * 25.0 - uMotion.w * 6.0 + 4.0)) - 0.5) * churn * 0.6;
    o = vec4(slope, foam, calm);
    return;
  }
  // ripple ring
  vec4 r = uRing[vI];
  vec2 d = vXZ - r.xy;
  float rr = length(d);
  float r0 = 0.02 + 0.24 * r.z;
  float k = 70.0;
  float env = exp(-pow((rr - r0) / 0.035, 2.0)) * exp(-r.z / 0.9) * r.w;
  float g = env * k * cos(k * (rr - r0)) * 0.0025;
  o = vec4(d / max(rr, 1e-4) * g, 0.0, 0.0);
}`;

// ---------------------------------------------------------------- spray particles
const DROP_VS = `
layout(location = 0) in vec4 aPS;        // position, radius
layout(location = 1) in vec4 aVA;        // velocity, alpha
uniform mat4 uViewProj;
uniform vec3 uCamPos;
uniform vec2 uViewport;
uniform float uExposure;
uniform float uPixelAngle;
out vec2 vQ;
out vec3 vWorld;
out float vA;
void main() {
  int v = gl_VertexID;
  vec2 c = v == 0 ? vec2(-1, -1) : v == 1 ? vec2(1, -1) : v == 2 ? vec2(1, 1) : v == 3 ? vec2(-1, -1) : v == 4 ? vec2(1, 1) : vec2(-1, 1);
  vec3 p0 = aPS.xyz, p1 = aPS.xyz - aVA.xyz * uExposure;
  vec4 c0 = uViewProj * vec4(p0, 1.0), c1 = uViewProj * vec4(p1, 1.0);
  vec2 s0 = c0.xy / c0.w, s1 = c1.xy / c1.w;
  vec2 px = 2.0 / uViewport;
  float dist = length(p0 - uCamPos);
  // radius in pixels (at least one pixel: the drop's light is spread, its alpha reduced)
  float rpx = aPS.w / (dist * uPixelAngle + 1e-6);
  vec2 ax = s1 - s0;
  float al = length(ax / px);
  vec2 dir = al > 0.5 ? normalize(ax / px) : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  float rad = max(rpx, 0.75);
  vec2 off = (dir * c.x * (rad + al * 0.5) + nrm * c.y * rad) * px;
  vec2 mid = (s0 + s1) * 0.5;
  gl_Position = vec4((mid + off) * c0.w, c0.z, c0.w);
  vQ = c;
  vWorld = p0;
  // coverage: small drops cover a fraction of the pixel, streaks spread over their length
  vA = aVA.w * min(rpx * rpx / (rad * rad), 1.0) * rad / (rad + al * 0.5);
}`;
const DROP_FS = SKY_LOOKUP_GLSL + `
in vec2 vQ;
in vec3 vWorld;
in float vA;
out vec4 o;
uniform vec3 uCamPos;
uniform sampler2D uEnv;
uniform vec3 uFlash;
const float PI = 3.14159265359;
vec2 equirectUv(vec3 d) { return vec2(atan(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI); }
void main() {
  float r2 = vQ.y * vQ.y + max(abs(vQ.x) - 0.0, 0.0) * 0.0;
  float m = 1.0 - smoothstep(0.6, 1.0, length(vec2(vQ.x * 0.25, vQ.y)) * 1.0 + abs(vQ.y) * 0.3);
  vec3 V = normalize(uCamPos - vWorld);
  float dist = length(uCamPos - vWorld);
  // a drop: reflects the sky over its rim, glints the sun, and shows the (brighter) world behind it
  vec3 N = normalize(V + vec3(0.0, vQ.y * 0.8, 0.0));
  vec3 R = reflect(-V, N);
  vec3 sky = textureLod(uEnv, equirectUv(R), 2.0).rgb;
  vec3 back = textureLod(uEnv, equirectUv(normalize(-V + vec3(0.0, 0.25, 0.0))), 3.0).rgb;
  vec3 H = normalize(uSunDir + V);
  float glint = pow(max(dot(normalize(N + vec3(0.0, 0.3, 0.0)), H), 0.0), 60.0) * 6.0;
  // sunlight scattered by a drop (reflection off its whole surface plus the strong forward lobe of
  // the light refracted through it), the sky it mirrors, the scene it refracts
  float fwd = pow(max(dot(-V, uSunDir), 0.0), 6.0);
  vec3 col = sky * 0.5 + back * 0.45 + sunIlluminance() * (0.025 + 0.25 * fwd + glint * 0.05) + skyIrradiance() * 0.06 + uFlash * 0.05;
  vec3 aL, aT;
  aerial(-V, dist * 0.001, aL, aT);
  col = col * aT + aL;
  col = applyFog(col, -V, dist);
  float a = clamp(vA * m, 0.0, 1.0) * 0.85;
  o = vec4(col * a, a);
}`;

const MAXP = 1400;
export class BoatWake {
  constructor(gl) {
    this.gl = gl;
    this.tex = new Texture(gl, { w: FX_RES, h: FX_RES, format: 'rgba16f', filter: 'linear' });
    this.target = new Target(gl, [this.tex]);
    this.trailProg = new Program(gl, FULL_VS, TRAIL_FS, { label: 'boat-trail' });
    this.splatProg = new Program(gl, SPLAT_VS, SPLAT_FS, { label: 'boat-splat' });
    this.dropProg = new Program(gl, DROP_VS, DROP_FS, { label: 'boat-drops' });
    this.vao = gl.createVertexArray();
    this.rect = [0, 0, FX_SIZE, 1 / FX_SIZE];
    this.trail = [];            // newest first: { x, z, t }
    this.trailU = new Float32Array(NTRAIL * 4);
    this.puddles = [];          // { x, z, dx, dz, t0 (release time, or null while pulling), str, seed }
    this.pudU = new Float32Array(NPUD * 8);
    this.rings = [];            // { x, z, t0, amp }
    this.ringU = new Float32Array(NRING * 4);
    // particles
    this.P = new Float32Array(MAXP * 8);
    this.np = 0;
    this.dropBuf = gl.createBuffer();
    this.dropVao = gl.createVertexArray();
    gl.bindVertexArray(this.dropVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dropBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.P.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 32, 0); gl.vertexAttribDivisor(0, 1);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 32, 16); gl.vertexAttribDivisor(1, 1);
    gl.bindVertexArray(null);
    this.time = 0;
    this.seed = 1;
    this.clear();
  }
  rand() { this.seed = (this.seed * 16807) % 2147483647; return (this.seed - 1) / 2147483646; }
  clear() {
    const gl = this.gl;
    this.target.bind();
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  // record the bow position (newest first), one point per ~0.6 m of travel
  // (trail[0] is the live bow, trail[1..] fixed points behind it)
  track(bow, t) {
    const p = { x: bow[0], z: bow[1], t };
    if (!this.trail.length) this.trail.push(p, { ...p });
    this.trail[0] = p;
    const p1 = this.trail[1];
    if (Math.hypot(p.x - p1.x, p.z - p1.z) >= 0.6) this.trail.splice(1, 0, { ...p });
    if (this.trail.length > NTRAIL) this.trail.length = NTRAIL;
  }

  // a blade has entered the water: a puddle that follows the blade until released
  catchAt(xz, dir, str) {
    const p = { x: xz[0], z: xz[1], dx: dir[0], dz: dir[1], t0: null, str, seed: this.rand() * 10 };
    this.puddles.unshift(p);
    if (this.puddles.length > NPUD) this.puddles.length = NPUD;
    this.ring(xz, 1.2);
    return p;
  }
  ring(xz, amp) {
    this.rings.unshift({ x: xz[0], z: xz[1], t0: this.time, amp });
    if (this.rings.length > NRING) this.rings.length = NRING;
  }

  // spray particle: position, velocity, radius (m), alpha
  spawn(p, v, r, a = 1) {
    if (this.np >= MAXP) return;
    const o = this.np++ * 8;
    this.P.set([p[0], p[1], p[2], r, v[0], v[1], v[2], a], o);
  }
  // advance drops; seaY(x, z) gives the sea height (world). Drops hitting the sea leave rings.
  stepDrops(dt, seaY) {
    const P = this.P;
    let j = 0;
    for (let i = 0; i < this.np; i++) {
      const o = i * 8;
      let x = P[o], y = P[o + 1], z = P[o + 2];
      const r = P[o + 3];
      let vx = P[o + 4], vy = P[o + 5], vz = P[o + 6];
      const drag = Math.exp(-dt * (0.08 / Math.max(r, 5e-4)) * 0.01);
      vx *= drag; vz *= drag; vy = vy * drag - 9.81 * dt;
      x += vx * dt; y += vy * dt; z += vz * dt;
      if (y < seaY(x, z)) {
        if (r > 0.0022 && this.rand() < 0.35) this.ring([x, z], Math.min(r * 160, 1));
        continue;
      }
      const q = j++ * 8;
      P[q] = x; P[q + 1] = y; P[q + 2] = z; P[q + 3] = r; P[q + 4] = vx; P[q + 5] = vy; P[q + 6] = vz; P[q + 7] = P[o + 7];
    }
    this.np = j;
  }

  // render the field around `center` (world xz)
  render(center, boatM, boatInv, waterPlane, motion, time) {
    const gl = this.gl;
    this.time = time;
    const tx = FX_SIZE / FX_RES;
    const x0 = Math.round((center[0] - FX_SIZE / 2) / tx) * tx, z0 = Math.round((center[1] - FX_SIZE / 2) / tx) * tx;
    this.rect = [x0, z0, FX_SIZE, 1 / FX_SIZE];
    const rectU = [x0, z0, FX_SIZE, 0];
    // track: arc length from the bow
    const T = this.trailU;
    T.fill(0);
    let s = 0;
    const n = Math.min(this.trail.length, NTRAIL);
    for (let i = 0; i < n; i++) {
      const p = this.trail[i];
      if (i) s += Math.hypot(p.x - this.trail[i - 1].x, p.z - this.trail[i - 1].z);
      T.set([p.x, p.z, s, time - p.t], i * 4);
    }
    this.target.bind();
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(this.vao);
    this.trailProg.use({ uRect: rectU, uTrail: T, uTrailN: n, uSpeed: [motion[0], XB - XS], uTime: time });
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    // splats
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    const pu = this.pudU;
    pu.fill(0);
    this.puddles = this.puddles.filter((p) => p.t0 === null || time - p.t0 < 30);
    this.puddles.forEach((p, i) => {
      const age = p.t0 === null ? 0 : time - p.t0;
      pu.set([p.x, p.z, p.dx, p.dz, age, p.str * Math.min(1, (p.t0 === null ? 0.3 : 1)), p.t0 === null ? 1 : 0, p.seed], i * 8);
    });
    this.rings = this.rings.filter((r) => time - r.t0 < 4);
    const ru = this.ringU;
    ru.fill(0);
    this.rings.forEach((r, i) => ru.set([r.x, r.z, time - r.t0, r.amp], i * 4));
    const prog = this.splatProg;
    prog.use({ uRect: rectU, uBoat: boatM, uBoatInv: boatInv, uPud: pu, uRing: ru, uWaterPlane: waterPlane, uMotion: motion });
    prog.set('uKind', 0);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, 1);
    if (this.puddles.length) { prog.set('uKind', 1); gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.puddles.length); }
    if (this.rings.length) { prog.set('uKind', 2); gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.rings.length); }
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);
  }

  uniforms(on) {
    return { uBoatFx: this.tex, uBoatFxRect: on ? [this.rect[0], this.rect[1], 1 / FX_SIZE, this.gain ?? 1] : [0, 0, 0, 0] };
  }

  // spray, after the sea (depth test against it, no depth writes)
  drawDrops(u, viewport, exposure, pixelAngle) {
    if (!this.np) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dropBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.P, 0, this.np * 8);
    this.dropProg.use({ ...u, uViewport: viewport, uExposure: exposure, uPixelAngle: pixelAngle });
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(this.dropVao);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.np);
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
    gl.depthMask(true);
  }
}
