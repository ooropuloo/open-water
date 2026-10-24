// kaiju: what the creature does to the sea surface. Included by the water shader (vertex and fragment
// stage); everything is off (one uniform compare) unless the kaiju is near the surface.
//
//  - Kelvin ship wake: the stationary-phase solution behind a moving pressure source. For a point x
//    behind the source and y to the side (in coordinates that follow the creature's curved track),
//    waves of propagation angle theta (t = tan theta) with 2 y t^2 + x t + y = 0 are stationary; the two
//    roots are the transverse and the divergent system, which merge on the cusp line at 19.47 deg.
//    Phase psi = k0 (x + y t) sqrt(1 + t^2), k0 = g / U^2. Height and its analytic gradient.
//  - The hull: water heaped against the flanks and ahead of the head (bow wave), with fresh whitewater
//    in a contact band around the waterline footprint of the body (a chain of capsules).
//  - The turbulent wake: whitewater just behind, then residual foam that spreads and frays, a band of
//    bubbles (pale aquamarine body colour) and damped short waves (the glassy "slick" behind a ship).
export const KJT = 48, KJB = 16;
export const KAIJU_WATER_GLSL = `
#define KJT ${KJT}
#define KJB ${KJB}
uniform float uKjOn;
uniform float uKjN;              // valid trail points
uniform vec4 uKjBox;             // xz bounds of the disturbed sea
uniform vec4 uKjTrail[KJT];      // xz, arc length from the front (m), age (s)
uniform vec4 uKjTrailB[KJT];     // surface strength, half-width (m), speed (m/s), 0
uniform vec4 uKjBody[KJB];       // xz, waterline half-width (m), strength (head first)
uniform vec4 uKjSrc;             // front contact xz, speed (m/s), strength
uniform vec4 uKjRing;            // a breach / re-entry: xz, age (s), strength
uniform float uKjT;              // clock (s)
uniform float uKjSurge;          // 0..1: the body heaving up / down through the surface
uniform vec4 uKjDbg;             // debug: 1 switches off (Kelvin, hull + bow, wake, ring)
struct KjSea { float h; vec2 g; float act; float res; float plume; float calm; };
float kjH(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
// value noise with its gradient
vec3 kjN(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f), du = 6.0 * f * (1.0 - f);
  float a = kjH(i), b = kjH(i + vec2(1, 0)), c = kjH(i + vec2(0, 1)), d = kjH(i + vec2(1, 1));
  return vec3(a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y,
              du * vec2(b - a + (a - b - c + d) * u.y, c - a + (a - b - c + d) * u.x));
}
// one Kelvin system: root t, amplitude a; adds height and gradient (track axes: x back, y side)
void kjWave(inout KjSea o, float x, float y, float t, float k0, float a, vec2 ex, vec2 ey, float foot) {
  float q = sqrt(1.0 + t * t);
  float k = k0 * q * q;
  // a long, blunt body barely excites waves much shorter than the transverse wave (its pressure
  // field's spectrum), so the divergent waves near the track are low; waves shorter than a few
  // footprints are not carried (mesh spacing / pixel); slopes stay those of real ship waves
  a *= exp(-0.4 * q * q) * smoothstep(2.5, 5.0, 6.2832 / (k * foot));
  a = min(a, 0.09 / k);
  if (a <= 0.0) return;
  float psi = k0 * (x + y * t) * q;
  vec2 kv = k0 * q * (ex + t * ey);
  o.h += a * cos(psi);
  o.g -= a * sin(psi) * kv;
}
KjSea kaijuSea(vec2 xz, float foot) {
  KjSea o = KjSea(0.0, vec2(0.0), 0.0, 0.0, 0.0, 0.0);
  if (uKjOn < 0.5 || xz.x < uKjBox.x || xz.y < uKjBox.y || xz.x > uKjBox.z || xz.y > uKjBox.w) return o;
  // nearest point of the track (front first)
  float best = 1e12, sx = 0.0, sy = 0.0, age = 0.0, str = 0.0, hw = 0.0;
  vec2 ex = vec2(1.0, 0.0);
  int n = int(uKjN);
  for (int i = 0; i < KJT - 1; i++) {
    if (i >= n - 1) break;
    vec4 a = uKjTrail[i], b = uKjTrail[i + 1];
    vec2 ab = b.xy - a.xy;
    float l2 = max(dot(ab, ab), 1e-3), l = sqrt(l2);
    float t = clamp(dot(xz - a.xy, ab) / l2, 0.0, 1.0);
    vec2 d = xz - a.xy - ab * t;
    float dd = dot(d, d);
    if (dd < best) {
      best = dd;
      vec2 dir = ab / l;
      ex = dir;
      sx = a.z + (i == 0 ? dot(xz - a.xy, dir) : t * (b.z - a.z));
      sy = dir.x * d.y - dir.y * d.x;
      age = mix(a.w, b.w, t);
      vec4 A = uKjTrailB[i], B = uKjTrailB[i + 1];
      str = mix(A.x, B.x, t);
      hw = mix(A.y, B.y, t);
    }
  }
  vec2 ey = vec2(-ex.y, ex.x);
  float U = max(uKjSrc.z, 0.5);
  // ---- Kelvin wake
  if (sx > 2.0 && str > 0.0 && uKjDbg.x < 0.5) {
    float k0 = 9.81 / (U * U);
    float x = sx, y = sy;
    float disc = x * x - 8.0 * y * y;
    float out_ = exp(-max(-disc, 0.0) / (x * x) * 30.0);
    float sd = sqrt(max(disc, 0.0));
    float A = str * 0.55 * (U / 10.0) * (U / 10.0) * sqrt(80.0 / (x + 80.0)) * exp(-x / 1400.0) * out_;
    // amplitude grows toward the cusp line where the two systems merge
    // (|disc|: outside the wedge the waves die off, they do not pile up at the cusp value)
    float cusp = 1.0 + 1.4 * exp(-abs(disc) / (x * x) * 10.0);
    float ys = abs(y) < 0.05 ? (y < 0.0 ? -0.05 : 0.05) : y;
    float tT = -2.0 * ys / (x + sd);
    float tD = -(x + sd) / (4.0 * ys);
    kjWave(o, x, y, tT, k0, A * cusp, ex, ey, foot);
    kjWave(o, x, y, tD, k0, A * cusp * 0.75, ex, ey, foot);
    // the cusp lines break into short whitecaps near the source
    o.act += str * 0.5 * smoothstep(0.35, 1.0, cusp - 1.0) * out_ * exp(-x / 220.0) * smoothstep(4.0, 9.0, U);
  }
  // ---- the hull: heaped water along the flanks, whitewater in a contact band
  float dB = 1e9, sB = 0.0;
  vec2 nB = vec2(0.0);
  for (int i = 0; i < KJB - 1; i++) {
    vec4 a = uKjBody[i], b = uKjBody[i + 1];
    if (a.z <= 0.0 && b.z <= 0.0) continue;
    vec2 ab = b.xy - a.xy;
    float t = clamp(dot(xz - a.xy, ab) / max(dot(ab, ab), 1e-3), 0.0, 1.0);
    vec2 rv = xz - a.xy - ab * t;
    float rl = length(rv);
    float dd = rl - mix(a.z, b.z, t);
    if (dd < dB) { dB = dd; sB = mix(a.w, b.w, t); nB = rv / max(rl, 1e-3); }
  }
  float spd = smoothstep(1.0, 9.0, U);
  if (dB < 80.0 && uKjDbg.y < 0.5) {
    float d = max(dB, 0.0);
    float K = sB * (0.25 + 0.5 * spd);
    float e1 = exp(-d * d / 60.0), e2 = exp(-(d - 16.0) * (d - 16.0) / 90.0);
    o.h += K * (e1 - 0.35 * e2);
    o.g += K * (-2.0 * d / 60.0 * e1 + 0.35 * 2.0 * (d - 16.0) / 90.0 * e2) * (dB > 0.0 ? nB : vec2(0.0));
    vec3 nz = kjN(xz / 4.5);
    o.act += sB * (0.3 + 0.7 * spd) * exp(-d / mix(2.0, 4.0, spd)) * smoothstep(0.3, 0.85, nz.x) * 1.4;
    // heaving through the surface the body shoves the sea aside: a mound several metres high against
    // it, broken and boiling, whitewater over it and churned foam spreading out from its base
    if (uKjSurge > 0.0) {
      float sg = uKjSurge * sB;
      vec3 b1 = kjN(xz / 7.0 + vec2(uKjT * 0.35, 0.0)), b2 = kjN(xz / 2.6 - vec2(0.0, uKjT * 0.6));
      float boil = 0.6 * b1.x + 0.4 * b2.x;
      // (a heaped, lumpy surface: domes of upwelling a few metres across)
      float mH = 3.0 * sg * exp(-d * d / 220.0) * (0.6 + 0.8 * boil) + 1.6 * sg * exp(-d / 16.0) * (b2.x - 0.5);
      o.h += mH;
      o.g += 3.8 * sg * (-2.0 * d / 220.0 * exp(-d * d / 220.0)) * (dB > 0.0 ? nB : vec2(0.0)) * 0.8;
      o.g += sg * exp(-d / 18.0) * (b1.yz / 7.0 * 2.4 + b2.yz / 2.6 * 1.6);
      // whitewater in patches and streaks over the boils, not a solid sheet
      o.act += sg * 1.6 * exp(-d / 10.0) * smoothstep(0.5, 0.85, boil);
      o.res += sg * 1.2 * exp(-d / 40.0) * smoothstep(0.35, 0.8, 0.5 * kjN(xz / 11.0 + 9.0).x + 0.5 * b2.x);
      o.plume += sg * 2.5 * exp(-d / 30.0);
    }
  }
  // ---- bow wave ahead of the front
  if (uKjDbg.y < 0.5) {
    vec2 fw = -ex;      // (the first segment points back along the track)
    vec2 r = xz - uKjSrc.xy;
    float f = dot(r, fw), l = dot(r, vec2(-fw.y, fw.x));
    float w0 = uKjTrailB[0].y + 3.0;
    float c = 9.0 + 0.6 * U, fe = max(f, -8.0);
    float m = exp(-fe * fe / (c * c)) * exp(-l * l / (w0 * w0)) * smoothstep(-12.0, 0.0, f);
    float Hb = uKjSrc.w * (0.3 + 0.9 * spd);
    o.h += Hb * m;
    o.g += Hb * m * ((f > -8.0 ? -2.0 * fe / (c * c) : 0.0) * fw - 2.0 * l / (w0 * w0) * vec2(-fw.y, fw.x));
    o.act += uKjSrc.w * m * 1.4 * spd * smoothstep(0.15, 0.7, kjN(xz / 6.0 + 3.0).x);
  }
  // ---- turbulent wake behind the front
  if (sx > -6.0 && str > 0.0 && uKjDbg.z < 0.5) {
    float w = hw + 1.3 * sqrt(age * 10.0 + 1.0);
    float v = sy / w;
    float prof = exp(-v * v * 1.6);
    vec3 nz = kjN(vec2(sx / 22.0, sy / 9.0) + vec2(0.0, age * 0.03));
    vec3 nz2 = kjN(vec2(sx / 7.0 + 3.1, sy / 4.0));
    float boil = smoothstep(0.25, 0.85, nz.x * 0.75 + nz2.x * 0.35) * 1.4;
    float on = smoothstep(-6.0, 12.0, sx);
    o.act += on * str * prof * 1.2 * exp(-age / 4.0) * boil * spd;
    o.res += on * str * prof * 1.9 * exp(-age / 60.0) * (0.35 + boil) * smoothstep(0.5, 4.0, U);
    float prof2 = exp(-v * v * 1.2);
    o.plume += on * str * prof2 * 2.0 * exp(-age / 80.0);
    o.calm += on * str * prof2 * 0.85 * exp(-age / 100.0);
    // boils: churned short slopes in the fresh wake
    o.g += on * str * prof * exp(-age / 25.0) * 0.06 * (nz.yz / 22.0 * 9.0 + nz2.yz / 7.0 * 2.0) * smoothstep(1.0, 0.2, foot);
  }
  // ---- a breach / re-entry: a spreading ring of whitewater
  if (uKjRing.w > 0.0 && uKjDbg.w < 0.5) {
    // (the whitewater spreads like a diffusing patch, ~140 m across at most, and thins out)
    float rr = length(xz - uKjRing.xy), R = min(18.0 + 22.0 * sqrt(uKjRing.z), 70.0);
    vec3 rn = kjN(xz / 9.0 + 5.0);
    o.act += uKjRing.w * exp(-pow((rr - R) / 12.0, 2.0)) * 1.0 * exp(-uKjRing.z / 6.0) * smoothstep(0.3, 0.8, rn.x) + uKjRing.w * exp(-rr * rr / (R * R)) * 0.5 * exp(-uKjRing.z / 4.0) * smoothstep(0.3, 0.8, rn.x);
    o.res += uKjRing.w * exp(-rr * rr / ((R + 15.0) * (R + 15.0))) * 1.2 * exp(-uKjRing.z / 25.0) * smoothstep(0.25, 0.75, rn.x);
    float front = 25.0 + 9.0 * uKjRing.z, kr = 0.125;
    float pk = exp(-pow((rr - front * 0.8) / (0.45 * front), 2.0)) * smoothstep(8.0, 30.0, rr);
    float amp = uKjRing.w * 1.8 * exp(-uKjRing.z / 16.0) * sqrt(40.0 / (rr + 40.0)) * pk;
    vec2 rd = (xz - uKjRing.xy) / max(rr, 1e-3);
    o.h += amp * cos(kr * (rr - front));
    o.g -= amp * kr * sin(kr * (rr - front)) * rd;
  }
  float edge = min(min(xz.x - uKjBox.x, uKjBox.z - xz.x), min(xz.y - uKjBox.y, uKjBox.w - xz.y));
  float fe = smoothstep(0.0, 40.0, edge);
  o.h *= fe; o.g *= fe; o.act *= fe; o.res *= fe; o.plume *= fe;
  o.calm = clamp(o.calm, 0.0, 1.0) * fe;
  return o;
}
`;
