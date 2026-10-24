// A humpback whale that surfaces to breathe (three blows), dives with its flukes raised, and breaches.
//
// Geometry: lofted body, flippers and flukes (shape.js), skinned on the GPU by a 28-bone spine plus
// flipper and fluke bones (rig.js). Skin is procedural: black back, mottled white throat and belly
// crossed by the ventral pleats, tubercles and barnacles on the head, the gape and eye, blowholes,
// white flippers and an individual fluke pattern. A water film that runs off the freshly emerged skin
// (time since emergence is estimated per vertex from the sea height and the bone's rise speed) makes
// it glossy and streaked. Light is the sea's own: sun or moon through the atmosphere, the cloud-shadow
// map, the reflection environment, lightning, aerial perspective, shafts and fog; sun shadows are cast
// by sphere proxies along the spine (the body on its flippers and on itself).
//
// Floating: the sea height is probed (choppy displacement inverted, PBO + fence, as the boat does) at
// every spine bone; the whale rides the swell near the surface and the readings drive the wet film,
// the waterline wash, the spray and the blow.
import { Program, Texture, Target, fullscreen, FULLSCREEN_VS } from '../gl.js';
import { SKY_LOOKUP_GLSL } from '../sky/atmosphere.js';
import { buildWhaleMesh, headKnobs, KNOBS, NSPINE, SPINE_X, bodySection, BLOWHOLE, PECT_ROOT, PECT_THICK, PECT_SPAN,
  PECT_LEN, B_PECT_R, B_PECT_L, B_FLUKE, B_FLUKE2, flukeEdgePoints, bodyPoint, SNOUT } from './shape.js';
import { behaviour, solveSpine, blendFrames, boneMatrices, applyBone, applyBoneDir, NBONES, CYCLE, BLOWS, T_EXIT, T_DIVE, PHASE_T, XC } from './rig.js';
import { WhaleFx } from './fx.js';

const VS = `
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec2 aUv;
layout(location = 3) in vec4 aBone;
uniform mat4 uViewProj;
uniform vec3 uCamPos;
uniform mat4 uBones[${NBONES}];
uniform float uBoneVy[${NBONES}];
uniform float uBoneDry[${NBONES}];
uniform float uBend[${NBONES}];   // dorsal-plane curvature of the spine at each bone (1/m)
uniform sampler2D uDisp[3];
uniform float uSize[4];
uniform float uTexel[4];
uniform float uAngStep;
out vec3 vWorld;
out vec3 vN;
out vec3 vRest;
out vec2 vUv;
out float vPart;
out float vH;
out float vEm;
out float vBend;
vec3 seaDisp(vec2 xz, float l) {
  vec3 D = textureLod(uDisp[0], xz / uSize[0], max(l - log2(uTexel[0]), 0.0)).xyz;
  D += textureLod(uDisp[1], xz / uSize[1], max(l - log2(uTexel[1]), 0.0)).xyz;
  D += textureLod(uDisp[2], xz / uSize[2], max(l - log2(uTexel[2]), 0.0)).xyz;
  return D;
}
void main() {
  int b0 = int(aBone.x + 0.5), b1 = int(aBone.y + 0.5);
  float w = aBone.z;
  mat4 M = uBones[b0] * (1.0 - w) + uBones[b1] * w;
  vec3 P = (M * vec4(aPos, 1.0)).xyz;
  vN = mat3(M) * aNrm;
  // the sea height here, as the water mesh draws it at this distance (choppy displacement inverted)
  float r = length(P.xz - uCamPos.xz);
  float l = log2(max(r * uAngStep, 0.02) * 1.5);
  vec2 x0 = P.xz;
  for (int k = 0; k < 2; k++) x0 = P.xz - seaDisp(x0, l).xz;
  float h = P.y - seaDisp(x0, l).y;
  float vy = mix(uBoneVy[b0], uBoneVy[b1], w);
  float dry = mix(uBoneDry[b0], uBoneDry[b1], w);
  // time since this skin left the water: a rising surface emerged h / v ago
  vEm = h <= 0.0 ? 0.0 : (vy > 0.15 ? min(h / vy, dry + 0.3) : dry);
  vH = h;
  vBend = mix(uBend[b0], uBend[b1], w);
  vec2 dr = P.xz - uCamPos.xz;
  P.y -= dot(dr, dr) / (2.0 * 6360000.0);
  vWorld = P;
  vRest = aPos;
  vUv = aUv;
  vPart = aBone.w;
  gl_Position = uViewProj * vec4(P, 1.0);
}`;

const f3 = (v) => `vec3(${v.map((x) => x.toFixed(5)).join(', ')})`;
const FS = SKY_LOOKUP_GLSL + `
in vec3 vWorld;
in vec3 vN;
in vec3 vRest;
in vec2 vUv;
in float vPart;
in float vH;
in float vEm;
in float vBend;
out vec4 oColor;
uniform vec3 uCamPos;
uniform sampler2D uEnv;
uniform vec3 uFlash;
uniform float uRain;
uniform float uTime;
uniform vec4 uKnob[${KNOBS}];
uniform vec4 uOcc[${NSPINE}];     // sun-shadow proxies along the spine (world centre, radius)
uniform float uWash;             // white water streaming off the body (breach, surfacing)
const float PI = 3.14159265359;
const vec3 PECT_ROOT = ${f3(PECT_ROOT)};
const vec3 PECT_THICK = ${f3(PECT_THICK)};
const vec3 PECT_SPAN = ${f3(PECT_SPAN)};
vec2 equirectUv(vec3 d) { return vec2(atan(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI); }
vec3 envL(vec3 d, float lod) { return textureLod(uEnv, equirectUv(normalize(d)), lod).rgb; }
float hs(vec3 p) { p = fract(p * vec3(0.1031, 0.1030, 0.0973)); p += dot(p, p.yxz + 33.33); return fract((p.x + p.y) * p.z); }
float hs2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vn2(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hs2(i), hs2(i + vec2(1, 0)), f.x), mix(hs2(i + vec2(0, 1)), hs2(i + vec2(1, 1)), f.x), f.y);
}
float fbm2(vec2 p) { return vn2(p) * 0.5 + vn2(p * 2.03 + 3.1) * 0.25 + vn2(p * 4.1 + 7.7) * 0.15 + vn2(p * 8.3 + 1.3) * 0.1; }
float vn3(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hs(i), hs(i + vec3(1, 0, 0)), f.x), mix(hs(i + vec3(0, 1, 0)), hs(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(hs(i + vec3(0, 0, 1)), hs(i + vec3(1, 0, 1)), f.x), mix(hs(i + vec3(0, 1, 1)), hs(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
// barnacle crusts: clusters of small cones (white-grey rims, dark centres) in 3D cells of the rest pose
float barnacles(vec3 p, float mask, float fp, out float rim) {
  rim = 0.0;
  if (mask < 0.01) return 0.0;
  const float C = 0.055;
  vec3 c = floor(p / C);
  float h = 0.0;
  for (int i = 0; i < 8; i++) {
    vec3 cc = c + vec3(float(i & 1), float((i >> 1) & 1), float((i >> 2) & 1)) - 0.5;
    vec3 cid = floor(cc + 0.5);
    float pr = hs(cid + 17.0);
    if (pr > mask) continue;
    vec3 ctr = (cid + 0.2 + 0.6 * vec3(hs(cid), hs(cid + 3.1), hs(cid + 5.7))) * C;
    float rad = C * (0.32 + 0.3 * hs(cid + 9.2));
    float r = length(p - ctr) / rad;
    if (r < 1.0) {
      h = max(h, rad * 0.8 * (1.0 - r * r) * smoothstep(0.0, 0.45, r));
      rim = max(rim, smoothstep(0.25, 0.45, r) * (1.0 - smoothstep(0.8, 1.0, r)));
    }
  }
  return h;
}
// sun visibility through the body proxies (soft: the sun disk and the coarse proxies)
float proxyShadow(vec3 P, vec3 L) {
  float vis = 1.0;
  for (int i = 0; i < ${NSPINE}; i++) {
    vec4 s = uOcc[i];
    vec3 oc = s.xyz - P;
    float t = dot(oc, L);
    if (t < 0.05) continue;
    float d2 = dot(oc, oc) - t * t;
    float r = s.w;
    vis *= smoothstep(r * r * 0.55, r * r * 1.15, d2);
  }
  return vis;
}

void main() {
  vec3 P = vWorld;
  vec3 toCam = uCamPos - P;
  float dist = length(toCam);
  vec3 V = toCam / dist;
  vec3 N = normalize(vN);
  int part = int(vPart + 0.5);
  vec3 q = vRest;
  // metres covered by one pixel (features below it fade to their mean)
  float fp = max(length(fwidth(q)), 1e-5);
  float H = 0.0;           // bump height (m)
  float cav = 1.0;         // cavity occlusion
  float rough = 0.30;
  vec3 alb;
  float barnMask = 0.0;
  float mottle = fbm2(q.xz * 3.1 + q.y * 1.7);

  if (part == 0) {
    // ------------------------------------------------ body
    float d = vUv.x, c = vUv.y;
    float vv = 0.5 - 0.5 * c;                        // 0 dorsal ... 1 ventral
    float a = sign(q.z) * (PI - acos(clamp(c, -1.0, 1.0)));   // angle from the ventral midline
    // dark grey-black back and flanks, slightly lighter low on the flanks
    vec3 black = vec3(0.020, 0.022, 0.026) * (0.8 + 0.35 * mottle) * mix(1.0, 1.35, smoothstep(0.45, 0.75, vv));
    // white throat and belly, the boundary ragged and mottled; black patches inside it
    float edge = fbm2(vec2(d * 1.05, a * 2.4)) - 0.5;
    float thr = mix(0.80, 0.68, smoothstep(0.6, 2.6, d)) + 0.07 * smoothstep(4.5, 8.5, d) + 0.25 * smoothstep(9.0, 10.8, d);
    float white = smoothstep(thr - 0.035, thr + 0.035, vv + edge * 0.24) * smoothstep(0.25, 0.8, d);
    // a few large dark areas with frayed, speckled edges, and fine dark freckles
    float blot = fbm2(vec2(d * 0.9 + 5.0, a * 2.2)) + 0.22 * (fbm2(vec2(d * 7.0, a * 14.0)) - 0.5);
    white *= 1.0 - 0.8 * smoothstep(0.60, 0.76, blot);
    white *= 1.0 - 0.75 * smoothstep(0.78, 0.84, fbm2(vec2(d * 9.0, a * 17.0)));
    // the white is never uniform: grey clouding at several scales, grey freckles, pale scars
    float cloud = fbm2(vec2(d * 1.7 + 11.0, a * 3.6)) * 0.6 + fbm2(vec2(d * 5.0, a * 9.0)) * 0.4;
    vec3 whiteC = vec3(0.55, 0.56, 0.55) * mix(0.45, 1.05, smoothstep(0.25, 0.75, cloud));
    whiteC = mix(whiteC, vec3(0.16, 0.165, 0.17), smoothstep(0.66, 0.74, fbm2(vec2(d * 14.0, a * 26.0))) * 0.8);
    alb = mix(black, whiteC, white);
    // ventral pleats: about thirty grooves from the chin to the navel, staggered ends
    float pw = 1.30 * smoothstep(0.3, 2.6, d) * (1.0 - 0.15 * smoothstep(4.0, 7.0, d));
    // (about 25 grooves in all, 15-20 cm apart on the throat)
    // (the ventral side is compressed where the spine bends ventrally: the pleats bunch and deepen there)
    float comp = clamp(-vBend * 14.0, -1.0, 1.0);
    float qq = a / (0.19 * (1.0 - 0.3 * max(comp, 0.0))) + 0.3 * sin(d * 0.85 + 0.3) + 0.1 * sin(d * 2.7) + 0.35 * (vn2(vec2(d * 0.7, a * 3.0)) - 0.5);
    float gi = floor(qq);
    float dEnd = 6.5 + 1.1 * hs2(vec2(gi, 3.0));
    float inP = smoothstep(pw, pw - 0.1, abs(a)) * smoothstep(0.3, 0.9, d) * (1.0 - smoothstep(dEnd - 0.5, dEnd, d));
    float gf = fract(qq) - 0.5;
    // ridges between the grooves are uneven: each pleat has its own width and depth
    float gw = 0.012 + 0.016 * hs2(vec2(gi, 7.0));
    float groove = exp(-gf * gf / (gw * (1.0 - 0.45 * max(comp, 0.0)) * (1.0 + 0.8 * max(-comp, 0.0))));
    groove *= 0.65 + 0.35 * vn2(vec2(d * 3.0, gi));
    float gRes = 1.0 - smoothstep(0.015, 0.05, fp);           // grooves resolved
    H -= 0.026 * (1.0 + 0.9 * comp) * groove * inP * gRes;
    // cross creases where the pleated skin bunches
    H -= 0.006 * max(comp, 0.0) * inP * gRes * smoothstep(0.55, 0.85, vn2(vec2(d * 9.0, qq * 1.3)));
    float gMean = mix(0.35, groove, gRes) * inP;
    alb = mix(alb, mix(alb * 0.55, vec3(0.16, 0.155, 0.15), 1.0 - white), gMean * 0.7);
    cav *= 1.0 - 0.6 * groove * inP * gRes;
    // the gape: from the snout tip back along the side, curving down to the corner of the mouth
    float ym = -0.02 - 0.035 * d - 0.017 * d * d;
    float mouth = abs(q.z) > 0.1 ? exp(-pow((q.y - ym) / 0.02, 2.0)) * (1.0 - smoothstep(3.8, 4.05, d)) : 0.0;
    H -= 0.03 * mouth;
    H += 0.012 * smoothstep(ym + 0.01, ym - 0.06, q.y) * smoothstep(ym - 0.4, ym - 0.1, q.y) * (1.0 - smoothstep(3.6, 4.0, d)) * step(0.1, abs(q.z));
    alb = mix(alb, vec3(0.008), mouth * mix(1.0, 0.5, smoothstep(0.02, 0.06, fp)));
    // the eye, above the corner of the mouth, in a wrinkled socket
    vec2 e = vec2((d - 4.30) / 0.075, (q.y + 0.37) / 0.048);
    float er = length(e);
    float onSide = smoothstep(0.3, 0.6, abs(q.z));
    float eye = exp(-er * er * 1.5) * onSide;
    H += onSide * (0.014 * exp(-pow(er - 1.7, 2.0) / 0.25) - 0.012 * eye);
    alb = mix(alb, vec3(0.004, 0.004, 0.005), smoothstep(0.3, 0.7, eye));
    rough = mix(rough, 0.08, eye);
    // blowholes: two slits converging forward behind the splash guard
    float bs = (d - 3.93) / 0.14;
    float bz = abs(q.z) - (0.03 + 0.05 * clamp(bs * 0.5 + 0.5, 0.0, 1.0));
    float blow = q.y > 0.6 ? exp(-bz * bz / 0.0003) * (1.0 - smoothstep(0.7, 1.0, abs(bs))) : 0.0;
    H -= 0.03 * blow;
    alb = mix(alb, vec3(0.006), blow);
    // tubercles of the head (each a rounded knob, often with a barnacle)
    float knobRing = 0.0;
    if (d < 4.0) {
      for (int i = 0; i < ${KNOBS}; i++) {
        vec4 k = uKnob[i];
        if (k.w <= 0.0) continue;
        float r = length(q - k.xyz) / k.w;
        if (r < 1.0) {
          H += k.w * 0.8 * pow(1.0 - r * r, 1.5);
          knobRing = max(knobRing, smoothstep(0.55, 0.85, r) * (1.0 - smoothstep(0.85, 1.0, r)));
          barnMask = max(barnMask, (1.0 - r) * step(0.45, hs(k.xyz * 31.0)));
        }
      }
    }
    alb *= 1.0 - 0.45 * knobRing;
    // barnacle crusts on the chin and the throat's front, and sparse on the flanks
    barnMask = max(barnMask, smoothstep(1.5, 0.5, d) * smoothstep(0.55, 0.8, vv) * 0.8);
    barnMask = max(barnMask, smoothstep(0.72, 0.8, fbm2(q.xz * 1.7 + 4.0)) * 0.5 * smoothstep(0.4, 0.7, vv));
    // undulations and fine wrinkles of the skin (they break up the highlights of the wet back)
    H += 0.006 * (vn3(q * 4.5) - 0.5) * (1.0 - smoothstep(0.03, 0.1, fp));
    H += 0.003 * (vn3(q * 12.0) - 0.5) * (1.0 - smoothstep(0.01, 0.04, fp));
    H += 0.0018 * (vn3(q * 30.0) - 0.5) * (1.0 - smoothstep(0.004, 0.02, fp));
    // white scars: a few rake marks on the tail stock
    float rk = abs(fract(q.y * 4.0 - q.x * 1.3) - 0.5);
    alb = mix(alb, vec3(0.32), smoothstep(0.035, 0.0, rk) * step(9.6, d) * step(d, 11.4) * step(0.62, fbm2(q.xz * 1.3)) * (1.0 - smoothstep(0.02, 0.06, fp)));
  } else if (part == 1) {
    // ------------------------------------------------ flippers: white, dark over the root on top, black mottles
    float s = vUv.x;
    float cp = 0.5 - 0.5 * vUv.y;
    vec3 rr = vec3(q.x, q.y, abs(q.z)) - PECT_ROOT;
    float upper = step(0.0, dot(rr, PECT_THICK) - 0.0);
    float dark = upper * (1.0 - smoothstep(0.18, 0.45, s + 0.12 * (fbm2(vec2(s * 6.0, cp * 3.0)) - 0.5))) + (1.0 - upper) * (1.0 - smoothstep(0.03, 0.1, s));
    dark = max(dark, smoothstep(0.70, 0.78, fbm2(vec2(s * 5.0, cp * 2.5 + upper * 7.0))) * 0.8);
    float cl = fbm2(vec2(s * 7.0 + 3.0, cp * 3.0 + upper * 5.0));
    vec3 wf = vec3(0.58, 0.59, 0.58) * mix(0.55, 1.05, smoothstep(0.25, 0.75, cl));
    alb = mix(wf, vec3(0.025, 0.027, 0.03), dark);
    // the cambered section: the trailing half falls away from the light, the leading edge is rounded
    cav *= mix(1.0, 0.62, smoothstep(0.35, 1.0, cp)) * mix(0.8, 1.0, smoothstep(0.0, 0.12, cp));
    // barnacles on the knobs of the leading edge and the tip
    barnMask = smoothstep(0.25, 0.04, cp) * smoothstep(0.3, 0.6, s) * 0.85 + smoothstep(0.9, 1.0, s) * 0.6;
    H += 0.0015 * (vn3(q * 20.0) - 0.5) * (1.0 - smoothstep(0.004, 0.02, fp));
    cav *= mix(0.55, 1.0, smoothstep(0.0, 0.12, s));
  } else {
    // ------------------------------------------------ flukes: black above; below, an individual white pattern
    float z = vUv.x;
    float cp = 0.5 - 0.5 * vUv.y;
    float eta = abs(z) / 2.32;
    float upper = step(-0.01 + 0.05 * eta * eta, q.y);
    float pat = fbm2(vec2(z * 1.35 + 2.0, cp * 2.2 + 0.3)) + 0.18 * (1.0 - smoothstep(0.0, 0.5, eta));
    float wht = (1.0 - upper) * smoothstep(0.47, 0.51, pat) * (1.0 - smoothstep(0.80, 0.9, cp)) * smoothstep(0.06, 0.16, cp) * smoothstep(0.08, 0.2, eta);
    wht *= 1.0 - smoothstep(0.66, 0.7, fbm2(vec2(z * 6.0, cp * 5.0)));
    alb = mix(vec3(0.021, 0.023, 0.026) * (0.85 + 0.3 * mottle), vec3(0.62, 0.62, 0.6), wht);
    // rake marks across the tips (orca teeth), white on black
    float rk = abs(fract(cp * 5.0 + z * 0.8) - 0.5);
    alb = mix(alb, vec3(0.35), smoothstep(0.04, 0.0, rk) * smoothstep(0.62, 0.75, eta) * step(0.55, hs2(vec2(floor(cp * 5.0 + z * 0.8), 2.0))) * (1.0 - smoothstep(0.02, 0.06, fp)));
    barnMask = smoothstep(0.85, 0.97, eta) * 0.7 + smoothstep(0.08, 0.0, eta) * smoothstep(0.7, 0.95, cp) * 0.6;
    H += 0.0015 * (vn3(q * 20.0) - 0.5) * (1.0 - smoothstep(0.004, 0.02, fp));
  }

  // barnacles (crust) everywhere their mask allows
  float rim;
  float bh = barnacles(q, barnMask, fp, rim);
  float bRes = 1.0 - smoothstep(0.01, 0.03, fp);
  H += bh * bRes;
  float bCov = mix(barnMask * 0.3, smoothstep(0.0, 0.004, bh), bRes);
  alb = mix(alb, mix(vec3(0.42, 0.41, 0.38), vec3(0.62, 0.6, 0.55), rim) * (0.8 + 0.3 * mottle), bCov);
  rough = mix(rough, 0.75, bCov);

  // bump mapping from the height field (Mikkelsen 2010, screen-space derivatives)
  vec3 dpx = dFdx(P), dpy = dFdy(P);
  float dhx = dFdx(H), dhy = dFdy(H);
  vec3 r1 = cross(dpy, N), r2 = cross(N, dpx);
  float det = dot(dpx, r1);
  vec3 grad = sign(det) * (dhx * r1 + dhy * r2);
  vec3 Nb = normalize(abs(det) * N - grad);
  if (abs(det) > 1e-14) N = Nb;
  if (dot(N, V) < -0.05) N = normalize(N + V * (0.05 - dot(N, V)));

  // the water film: fresh skin from the sea is glossy, sheets run down it; it never dries out fully
  float fresh = exp(-vEm / 1.6) * step(0.0, vH);
  vec2 fl = vec2(q.x * 3.0 + q.z * 2.2, (vH + uTime * 1.9) * 1.7);
  float sheet = smoothstep(0.45, 0.75, vn2(fl) * 0.65 + vn2(fl * vec2(2.3, 1.4) + 5.0) * 0.35);
  rough = mix(rough, mix(0.20, 0.07, sheet), fresh * (1.0 - bCov));
  rough = mix(rough, 0.22, 0.5 * (1.0 - bCov));          // always wet skin
  alb *= mix(1.0, 0.82, fresh);
  // aerated water pouring off a body that has just burst out of the sea
  float pour = uWash * fresh * exp(-vEm / 0.5) * smoothstep(1.8, 0.0, vH) * smoothstep(0.35, 0.65, vn2(fl * vec2(1.0, 0.6) + 2.0));

  vec3 L = uSunDir;
  vec3 sunE = sunIlluminance();
  vec2 suv = (P.xz - L.xz * (max(P.y, 0.0) / max(L.y, 0.02)) - uShaftRect.xy) * uShaftRect.zw;
  float sunVis = mix(uHazeSun, texture(uShaftShadow, suv).r, smoothstep(0.0, 0.04, min(min(suv.x, suv.y), min(1.0 - suv.x, 1.0 - suv.y))));
  if (L.y > 0.0) sunVis *= proxyShadow(P + N * 0.06, L);

  vec3 T1 = normalize(cross(N, abs(N.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  vec3 T2 = cross(N, T1);
  vec3 Eamb = (envL(N, 5.0) * 0.4 + (envL(N + T1, 5.0) + envL(N - T1, 5.0) + envL(N + T2, 5.0) + envL(N - T2, 5.0)) * 0.15) * PI;
  // occlusion: the throat and belly face the dark sea, the flipper roots and the pleats are cavities
  float ao = cav * mix(0.75, 1.0, smoothstep(-0.6, 0.4, N.y));
  float nl = max(dot(N, L), 0.0), nv = max(dot(N, V), 1e-3);
  vec3 Hh = normalize(L + V);
  float nh = max(dot(N, Hh), 0.0);
  float a2 = max(rough * rough * rough * rough, 2e-4);
  float D = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
  float F0 = 0.028;
  float Fh = F0 + (1.0 - F0) * pow(1.0 - max(dot(V, Hh), 0.0), 5.0);
  float Fv = F0 + (1.0 - F0) * pow(1.0 - nv, 5.0) / (1.0 + 4.0 * rough);
  vec3 Esun = sunE * sunVis * nl;
  vec3 Ef = uFlash * (0.35 + 0.65 * max(dot(N, uFlashDir), 0.0));
  vec3 col = alb / PI * (Esun + Eamb * ao + Ef) * (1.0 - Fv);
  // (Smith-like visibility, energy-limited)
  float k = rough * rough * 0.5;
  float G = 1.0 / ((nl * (1.0 - k) + k) * (nv * (1.0 - k) + k));
  col += Esun * min(D * G * 0.25, 600.0) * Fh;
  vec3 R = reflect(-V, N);
  col += envL(R, mix(0.5, 7.0, rough)) * Fv * ao;
  // white water pouring off
  col = mix(col, 0.75 * (sunE * sunVis * max(dot(N, L) * 0.6 + 0.4, 0.0) + Eamb) / PI, pour * 0.85);

  vec3 aL, aT;
  aerial(-V, dist * 0.001, aL, aT);
  col = col * aT + aL;
  col = withShafts(col, uCamPos, -V, dist);
  col = applyFog(col, -V, dist);
  oColor = vec4(col, 1.0);
}`;

// ---------------------------------------------------------------- sea probe (as the boat's)
const PROBE_FS = `
uniform sampler2D uDisp[3];
uniform float uSize[4];
uniform float uTexel[4];
uniform vec2 uPts[32];
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
const NP = 32;
class SeaProbe {
  constructor(gl) {
    this.gl = gl;
    this.prog = new Program(gl, FULLSCREEN_VS, PROBE_FS, { label: 'whale-probe' });
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
  run(ot, pts, footprint, sync) {
    const gl = this.gl;
    if (this.fence && !sync) {
      if (gl.clientWaitSync(this.fence, 0, 0) === gl.TIMEOUT_EXPIRED) return this.heights;
      gl.deleteSync(this.fence);
      this.fence = null;
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.data);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      this.decode();
    }
    if (this.fence) return this.heights;
    fullscreen(gl, this.prog, { uDisp: ot.disp.slice(0, 3), uSize: ot.sizes, uTexel: ot.texels, uPts: pts, uFootprint: footprint }, this.target);
    if (sync) {
      gl.readPixels(0, 0, NP, 1, gl.RGBA, gl.FLOAT, this.data);
      this.decode();
      return this.heights;
    }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
    gl.readPixels(0, 0, NP, 1, gl.RGBA, gl.FLOAT, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    this.fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    return this.heights;
  }
  decode() {
    const h = [];
    for (let i = 0; i < NP; i++) h.push(this.data[i * 4 + 1]);
    if (h.every(Number.isFinite)) this.heights = h;
  }
}

// rest-local representative point of each bone (velocity, emergence)
const SECT = SPINE_X.map((x) => bodySection(SNOUT - x));
const REP = [
  ...SPINE_X.map((x, i) => [x, SECT[i].mid + SECT[i].top * 0.9, 0]),
  ...[1, -1].flatMap((s) => [0.25, 0.75].map((f) => PECT_ROOT.map((v, i) => (v + PECT_SPAN[i] * PECT_LEN * f) * (i === 2 ? s : 1)))),
  [-6.3, 0.0, 0], [-6.9, 0.0, 1.6],
];

const PRESETS = { surface: 0, ...PHASE_T };
const nearestBone = (x) => SPINE_X.reduce((b, v, i) => (Math.abs(v - x) < Math.abs(SPINE_X[b] - x) ? i : b), 0);
const I_COM = nearestBone(XC), I_BLOW = nearestBone(BLOWHOLE[0]);

export class Whale {
  // anchor: camera base (xz); yaw: camera heading; dist: metres; az: bearing offset (deg)
  constructor(gl, { visible = true, anchor = [0, 0], yaw = 0, dist = 175, az = 15, start = '' } = {}) {
    this.gl = gl;
    this.visible = visible;
    this.prog = new Program(gl, VS, FS, { label: 'whale' });
    this.mesh = buildWhaleMesh(gl);
    this.probe = new SeaProbe(gl);
    this.fx = new WhaleFx(gl);
    this.knobs = new Float32Array(KNOBS * 4);
    headKnobs().forEach((k, i) => this.knobs.set(k, i * 4));
    this.mats = new Float32Array(NBONES * 16);
    this.occ = new Float32Array(NSPINE * 4);
    this.vy = new Float32Array(NBONES);
    this.dry = new Float32Array(NBONES);
    this.bend = new Float32Array(NBONES);
    this.lastWet = new Float64Array(NBONES);
    this.prevRep = null;
    this.pts = new Float32Array(NP * 2);
    this.heave = 0;
    const b = yaw + (az * Math.PI) / 180;
    const dir = [Math.cos(b), Math.sin(b)];
    // the track: broadside to the camera, at dist metres, a little to one side of the view
    this.track = { c: [anchor[0] + dir[0] * dist, anchor[1] + dir[1] * dist], h0: [-dir[1], dir[0]], n: [-dir[0], -dir[1]] };
    this.track.h = this.track.h0;
    // ?whale=<seconds into the cycle> or a named moment (blow, dive, flukes, breach, air, splash)
    const st = String(start);
    this.clock = st in PRESETS ? PRESETS[st] : Number.isFinite(Number(st)) && st !== '' ? Number(st) : 2;
    this.cycle = 0;
    this.now = 0;
    this.prevT = this.clock;
    this.pending = false;
    this.flukeEdge = flukeEdgePoints(24);
    this.ready = false;
  }

  // jump to a breach: at once while the whale is deep enough to be unseen, otherwise as soon as it is
  breachNow() { this.pending = true; }
  // come up for a breath now: shown at once if hidden; jumps to the surfacing if the whale is down deep (else it is in sight already)
  surfaceNow() {
    if (!this.visible) { this.visible = true; this.clock = BLOWS[0] - 4; this.fx.clearEvents?.(); return; }
    this.pendingSurface = true;
  }

  update(dt, ot, cam, angStep, sync, env) {
    if (!this.visible) return;
    dt = Math.min(Math.max(dt, 0), 0.1);
    this.now += dt;
    const prevT = this.clock;
    this.clock += dt;
    if (this.pending && this.deepEnough) { this.clock = T_EXIT - 3.2; this.pending = false; this.fx.clearEvents?.(); }
    if (this.pendingSurface) { if (this.deepEnough && !this.pending) { this.clock = BLOWS[0] - 4; this.fx.clearEvents?.(); } this.pendingSurface = false; }
    if (this.clock >= CYCLE) { this.clock -= CYCLE; this.cycle++; }
    const t = this.clock;
    // alternate the swimming direction from cycle to cycle
    this.track.h = this.cycle % 2 ? [-this.track.h0[0], -this.track.h0[1]] : this.track.h0;
    // the whale's local z (heading x up) against the direction to the camera
    this.track.zs = Math.sign(-this.track.h[1] * this.track.n[0] + this.track.h[0] * this.track.n[1]) || 1;
    const B = behaviour(t, this.track);
    let frames = solveSpine(B.main.pose);
    if (B.other && B.w > 0) frames = blendFrames(frames, solveSpine(B.other.pose), B.w);
    const ap = B.main.ap;
    this.phase = B.phase;
    // ride the swell near the surface
    const H = this.probe.heights;
    if (H) {
      const k = 1 - Math.exp(-dt * 3);
      this.heave += (H[28] - this.heave) * k;
    }
    const ride = B.main.ride;
    this.rideW = (this.rideW ?? ride) + (ride - (this.rideW ?? ride)) * (1 - Math.exp(-dt * 2));
    for (const f of frames) f.p[1] += this.heave * this.rideW;
    boneMatrices(this.mats, frames, ap, this.mesh.attach);
    this.frames = frames;
    // probe the sea at the spine bones, the centre of mass and the blowholes
    const P = this.pts;
    for (let i = 0; i < NSPINE; i++) { P[i * 2] = frames[i].p[0]; P[i * 2 + 1] = frames[i].p[2]; }
    const com = applyBone(this.mats, I_COM, [XC, 0, 0]);
    const bh = applyBone(this.mats, I_BLOW, BLOWHOLE);
    const fk = applyBone(this.mats, B_FLUKE, [-6.6, 0, 0]);
    P.set([com[0], com[2], bh[0], bh[2], fk[0], fk[2], com[0] + 4, com[2]], 28 * 2);
    const dist = cam ? Math.hypot(com[0] - cam.pos[0], com[2] - cam.pos[2]) : 170;
    if (ot) this.probe.run(ot, P, Math.max(dist * angStep * 1.5, 0.8), sync);
    const sea = (i) => (this.probe.heights ? this.probe.heights[Math.min(i, 31)] : 0);
    // bone velocities and the time since each bone's top emerged
    const rep = REP.map((q, i) => applyBone(this.mats, i, q));
    if (!this.vel) this.vel = rep.map(() => [0, 0, 0]);
    if (this.prevRep && dt > 0) rep.forEach((r, i) => { for (let c = 0; c < 3; c++) this.vel[i][c] += ((r[c] - this.prevRep[i][c]) / dt - this.vel[i][c]) * (1 - Math.exp(-dt * 8)); });
    for (let i = 0; i < NBONES; i++) {
      const vyNow = this.prevRep && dt > 0 ? (rep[i][1] - this.prevRep[i][1]) / dt : 0;
      this.vy[i] += (vyNow - this.vy[i]) * (1 - Math.exp(-dt * 8));
      const si = i < NSPINE ? sea(i) : i < NSPINE + 4 ? sea(Math.round(SPINE_X.findIndex((x) => x > 2.7))) : sea(30);
      if (rep[i][1] < si + 0.02) this.lastWet[i] = this.now;
      this.dry[i] = Math.min(this.now - this.lastWet[i], 60);
    }
    // the spine's dorsal-plane curvature at each bone (positive: concave on the back)
    for (let i = 0; i < NSPINE; i++) {
      const a = frames[Math.max(i - 1, 0)], b = frames[Math.min(i + 1, NSPINE - 1)];
      const cx = a.R[0][1] * b.R[0][2] - a.R[0][2] * b.R[0][1], cy = a.R[0][2] * b.R[0][0] - a.R[0][0] * b.R[0][2], cz = a.R[0][0] * b.R[0][1] - a.R[0][1] * b.R[0][0];
      const Z = frames[i].R[2];
      const span = (Math.min(i + 1, NSPINE - 1) - Math.max(i - 1, 0)) * (SPINE_X[1] - SPINE_X[0]);
      this.bend[i] = (cx * Z[0] + cy * Z[1] + cz * Z[2]) / span;
    }
    // sun-shadow proxies along the spine
    for (let i = 0; i < NSPINE; i++) {
      const f = frames[i], s = SECT[i];
      const c = f.p.map((v, j) => v + f.R[1][j] * s.mid);
      this.occ.set([c[0], c[1], c[2], Math.min(s.top, s.bot, s.w) * 0.95], i * 4);
    }
    this.deepEnough = rep.slice(0, NSPINE).every((r, i) => r[1] < sea(i) - 2.2) || B.phase === 'deep';
    // effects: blows, spray, the waterline wash, foam
    const ev = { t, prevT, dt, now: this.now, phase: B.phase, frames, mats: this.mats, rep, vy: this.vy, dry: this.dry, sea, heave: this.heave,
      blowhole: bh, up: applyBoneDir(this.mats, I_BLOW, [0, 1, 0]), side: applyBoneDir(this.mats, I_BLOW, [0, 0, 1]), fwd: applyBoneDir(this.mats, I_BLOW, [1, 0, 0]),
      flukeEdge: this.flukeEdge.map((e) => ({ p: applyBone(this.mats, e.eta > 0.4 ? B_FLUKE2 : B_FLUKE, e.p), eta: e.eta })),
      pect: [B_PECT_R, B_PECT_L].map((b, si) => [0.2, 0.55, 0.9].map((f) => applyBone(this.mats, b, PECT_ROOT.map((v, i) => (v + PECT_SPAN[i] * PECT_LEN * f) * (i === 2 ? (si ? -1 : 1) : 1))))),
      cycle: this.cycle, sect: SECT, blows: BLOWS, exit: T_EXIT, vel: this.vel, tDive: T_DIVE, track: this.track, bodyPoint, mats2: this.mats, B_FLUKE, B_FLUKE2 };
    this.washLevel = this.fx.update(ev, env);
    this.prevRep = rep;
    this.ready = !!this.probe.heights;
  }

  draw(u) {
    if (!this.visible || !this.frames) return;
    const gl = this.gl;
    this.prog.use({
      ...u, uBones: this.mats, uBoneVy: this.vy, uBoneDry: this.dry, uBend: this.bend, uKnob: this.knobs, uOcc: this.occ, uAngStep: u.uAngStep,
      uWash: this.washLevel || 0,
    });
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
    gl.disable(gl.CULL_FACE);
    gl.bindVertexArray(this.mesh.vao);
    gl.drawElements(gl.TRIANGLES, this.mesh.count, this.mesh.itype, 0);
    gl.bindVertexArray(null);
  }

  // after the water: the foam and wash on the sea surface, then blow and spray
  drawAfterWater(u, fxU) {
    if (!this.visible || !this.frames) return;
    this.fx.draw(u, fxU);
  }
}
