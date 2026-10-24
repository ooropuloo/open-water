// Warship hull shape (shared by the mesh builder, the ship shader and the sea's wake term) and the
// wake the sea surface draws around it.
//
// Ship-local metres: x towards the bow, y up (0 = design waterline), z to starboard.
// A modern destroyer / frigate hull: 130 m overall, 16 m beam, ~5 m draught, a transom stern, a raked
// clipper stem whose sections flare out above the waterline towards the bow, and a sheer that
// rises over the forward third.

export const SXS = -64, SXB = 66;   // transom, stem head
const ss = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
export const shipU = (x) => (x - SXS) / (SXB - SXS);
export const deckH = (u) => 6.2 + 3.4 * Math.pow(ss(0.55, 1, u), 1.6);
export const keelH = (u) => -5.2 + 3.0 * Math.pow(1 - ss(0, 0.25, u), 1.5) + (deckH(1) + 5.2) * Math.pow(ss(0.8, 1, u), 2.2);
export const beamDeck = (u) => (u < 0.45 ? 8.0 - 1.0 * ((0.45 - u) / 0.45) ** 2 : 8.0 * Math.pow(Math.max(1 - ((u - 0.45) / 0.55) ** 2, 0), 0.62));
export const beamWL = (u) => (u < 0.42 ? 7.7 - 1.4 * ((0.42 - u) / 0.42) ** 2 : 7.7 * Math.pow(Math.max(1 - ((u - 0.42) / 0.58) ** 2, 0), 1.05));
// half-width of the hull at section u and height y (0 outside the hull's height range)
export function halfWidth(u, y) {
  const G = deckH(u), K = keelH(u);
  if (y < K || y > G + 1e-6) return 0;
  const f = Math.min(Math.max((y + 1) / (G + 1), 0), 1);
  const w = beamWL(u) + (beamDeck(u) - beamWL(u)) * Math.pow(f, 1.5);
  const bil = Math.pow(Math.sin(Math.min((y - K) / Math.max(Math.min(2.5, G - K), 0.01), 1) * Math.PI / 2), 0.5);
  return w * bil;
}
// where the stem cuts the design waterline
export const STEM_WL = (() => {
  let x = SXB;
  while (x > 0 && halfWidth(shipU(x), 0) <= 0.05) x -= 0.25;
  return x;
})();

export const WAKE_N = 16;      // points along the stern's track
export const WAKE_DS = 40;     // metres between them

const f3 = (x) => x.toFixed(3);
export const SHIP_HULL_GLSL = `
const float SXS = ${f3(SXS)}, SXB = ${f3(SXB)}, STEM_WL = ${f3(STEM_WL)};
float shipU(float x) { return (x - SXS) / (SXB - SXS); }
float shDeckH(float u) { return 6.2 + 3.4 * pow(smoothstep(0.55, 1.0, u), 1.6); }
float shKeelH(float u) { return -5.2 + 3.0 * pow(1.0 - smoothstep(0.0, 0.25, u), 1.5) + (shDeckH(1.0) + 5.2) * pow(smoothstep(0.8, 1.0, u), 2.2); }
float shBeamDeck(float u) { float t = (u - 0.45) / 0.55; return u < 0.45 ? 8.0 - 1.0 * pow((0.45 - u) / 0.45, 2.0) : 8.0 * pow(max(1.0 - t * t, 0.0), 0.62); }
float shBeamWL(float u) { float t = (u - 0.42) / 0.58; return u < 0.42 ? 7.7 - 1.4 * pow((0.42 - u) / 0.42, 2.0) : 7.7 * pow(max(1.0 - t * t, 0.0), 1.05); }
float shHalfWidth(float u, float y) {
  float G = shDeckH(u), K = shKeelH(u);
  if (u < 0.0 || u > 1.0 || y < K || y > G) return 0.0;
  float f = clamp((y + 1.0) / (G + 1.0), 0.0, 1.0);
  float w = shBeamWL(u) + (shBeamDeck(u) - shBeamWL(u)) * pow(f, 1.5);
  return w * pow(sin(min((y - K) / max(min(2.5, G - K), 0.01), 1.0) * 1.5707963), 0.5);
}`;

// Wake on the sea surface: foam and a smoothed (slick) strip, evaluated in the water shader only
// inside the wake's bounding box (one uniform branch elsewhere).
//  - bow wave: a foam cushion against the stem and flaring bow, thinning aft along the hull
//  - hull wash along the waterline (the white line where hull and sea meet), stronger at the quarter
//  - Kelvin wake: divergent wave crests at 19.5 deg from the bow and stern shoulders, breaking white
//  - the turbulent propeller / transom wake along the stern's actual track, spreading and decaying
//    with age, inside a long-lived smooth strip where the turbulence damps the capillary waves
export const WAKE_GLSL = SHIP_HULL_GLSL + `
#define WAKE_N ${WAKE_N}
uniform vec4 uWakeShip;   // xy: ship centre (xz), zw: heading (unit, xz)
uniform vec4 uWakeInfo;   // x: on, y: speed (m/s), z: spacing of the track points (m)
uniform vec4 uWakeBox;    // bounding box of the wake: xz min, xz max
uniform vec2 uWakePts[WAKE_N];   // the stern's track, from the transom backwards
float wkHash(vec2 p) { return fract(sin(dot(p, vec2(41.31, 289.13))) * 43758.5453); }
float wkNoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(wkHash(i), wkHash(i + vec2(1, 0)), f.x), mix(wkHash(i + vec2(0, 1)), wkHash(i + vec2(1, 1)), f.x), f.y);
}
// p: sea surface point (xz); pix: pixel footprint (m). Returns x: foam coverage, y: slick 0..1.
vec2 shipWake(vec2 p, float pix) {
  if (uWakeInfo.x < 0.5) return vec2(0.0);
  if (any(lessThan(p, uWakeBox.xy)) || any(greaterThan(p, uWakeBox.zw))) return vec2(0.0);
  vec2 fw = uWakeShip.zw, sd = vec2(-fw.y, fw.x);
  vec2 d = p - uWakeShip.xy;
  float lx = dot(d, fw), lz = abs(dot(d, sd));
  float spd = clamp(uWakeInfo.y / 7.0, 0.0, 1.4);
  float foam = 0.0;
  // bow wave and hull wash
  float bx = STEM_WL - lx;                 // metres aft of the stem at the waterline
  if (lx > SXS - 2.0 && lx < STEM_WL + 4.0) {
    float hw = shHalfWidth(shipU(min(lx, STEM_WL)), 0.0);
    float e = lz - hw;                     // metres out from the hull side
    float bow = (1.0 - smoothstep(12.0, 48.0, bx)) * smoothstep(-3.0, 1.0, bx);
    float wB = 1.4 + 0.10 * max(bx, 0.0);
    float wS = 1.1 + 0.9 * smoothstep(SXS + 45.0, SXS + 5.0, lx);
    foam += bow * spd * 1.3 * (1.0 - smoothstep(0.25 * wB, wB, e)) * step(-1.5, e);
    foam += (0.85 + 0.6 * smoothstep(SXS + 40.0, SXS + 4.0, lx)) * spd * (1.0 - smoothstep(0.35 * wS, wS + 0.8, e)) * step(-1.5, e);
  }
  // Kelvin wake: divergent crests from the bow and the stern shoulders (tan 19.47 deg = 0.354)
  if (bx > 0.0) {
    float w1 = 1.6 + 0.035 * bx;
    float g1 = (lz - (1.5 + 0.354 * bx)) / w1;
    foam += 0.8 * spd * exp(-bx / 170.0) * exp(-g1 * g1) * smoothstep(0.0, 8.0, bx);
    float bx2 = lx > SXS + 30.0 ? 0.0 : (SXS + 30.0) - lx;
    float w2 = 1.8 + 0.035 * bx2;
    float g2 = (lz - (6.0 + 0.354 * bx2)) / w2;
    foam += 0.5 * spd * exp(-bx2 / 150.0) * exp(-g2 * g2) * step(0.001, bx2);
  }
  // turbulent wake along the stern's track
  float best = 1e9, along = 0.0;
  for (int k = 0; k < WAKE_N - 1; k++) {
    vec2 a = uWakePts[k], ab = uWakePts[k + 1] - a;
    float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    vec2 q = p - a - ab * t;
    float dd = dot(q, q);
    if (dd < best) { best = dd; along = (float(k) + t) * uWakeInfo.z; }
  }
  best = sqrt(best);
  float aft = 1.0 - smoothstep(SXS + 1.0, SXS + 6.0, lx);   // (not alongside the hull)
  float hwk = 6.0 + 0.09 * along;   // (the turbulent fan spreads ~5 deg each side)
  float turb = spd * 1.25 * exp(-along / 300.0) * (1.0 - smoothstep(0.3 * hwk, hwk, best)) * aft;
  float slick = spd * exp(-along / 900.0) * (1.0 - smoothstep(1.0 * hwk, 1.9 * hwk + 4.0, best)) * aft;
  // breakup: streaky froth near the camera, its mean where a pixel spans metres
  float n = wkNoise(p / 3.2) * 0.55 + wkNoise(vec2(dot(p, fw), dot(p, sd)) * vec2(0.18, 1.1) + 7.7) * 0.45;
  float far = smoothstep(0.4, 3.0, pix);
  // the old wake breaks into patches and streaks as it ages
  float age = smoothstep(40.0, 400.0, along) * aft;
  turb *= mix(1.0, smoothstep(0.3, 0.75, wkNoise(p / 22.0) * 0.6 + wkNoise(p / 7.0 + 3.3) * 0.4), age);
  foam += turb;
  float cov = mix(smoothstep(0.2, 0.75, foam * (0.3 + 1.3 * n)), clamp(foam * 0.8, 0.0, 1.0), far);
  return vec2(clamp(cov, 0.0, 1.0), clamp(slick, 0.0, 1.0));
}`;
