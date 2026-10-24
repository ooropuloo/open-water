// kaiju: a ~100 m saurian that swims past now and then.
//
// Behaviour: every pass it comes in from ~2.5 km with only its dorsal plates out of the water, turns to
// cross the view 500-800 m out (well beyond the rowboat), slows, rears its head and chest out of the sea,
// roars, sinks back, swims on and dives into the distance. Passes repeat every few minutes.
//
// Skeleton: a chain of 64 spine bones. The spine follows the head's own path through the water (as a
// snake's body follows its head), with a travelling lateral wave that grows toward the tail (carangiform
// swimming), bends distributed over the hips, the neck and the head for rearing up, and heave from the
// FFT sea measured along the body (PBO + fence readback). Every vertex is skinned on the GPU by its rest
// position along the spine; the lower jaw is a hinged sub-bone.
//
// Light: as everything else in the scene: sun or moon through the atmosphere, the cloud-shadow map, the
// reflection environment for sky and sea light, lightning, self-shadowing from a small shadow map, then
// the sea's aerial perspective, crepuscular rays and rain / mist fog, into the HDR target before post.
// Drawn before the water, so the sea hides whatever is submerged.
import { Program, Texture, Target, fullscreen, FULLSCREEN_VS, makeMesh } from '../gl.js';
import { SKY_LOOKUP_GLSL } from '../sky/atmosphere.js';
import { REST, JAW, profW, profT, profB, plateSize, sculpt } from './sculpt.js';
import { KJT, KJB } from './wake.js';
import { Splash, SPRAY, STREAM, MIST } from './splash.js';
import { roar } from './roar.js';

const NB = REST.NB, X0 = REST.X0, X1 = REST.X1, DX = (X1 - X0) / (NB - 1);
const XREF = 70;                       // the point on the snout that leads along the path
const NODE_X = Array.from({ length: NB }, (_, i) => X0 + i * DX);
const NP = 16;                          // sea probes along the body (head first)
const PROBE_NODE = Array.from({ length: NP }, (_, k) => NB - 1 - 4 * k);
const SH = 1024;                        // shadow map size

const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const mix = (a, b, t) => a + (b - a) * t;
const win = (t, a, b, c, d) => sstep(a, b, t) * (1 - sstep(c, d, t));  // ramp up a..b, down c..d

// ---------------------------------------------------------------- shaders
const SKIN_GLSL = `
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec4 aAux;
uniform vec4 uKjP[${NB}];
uniform vec4 uKjQ[${NB}];
uniform float uKjJaw;
vec3 qrot(vec4 q, vec3 v) { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
void kjSkin(out vec3 P, out vec3 N, out vec4 Qo) {
  vec3 p = aPos, n = aNrm;
  float id = floor(aAux.w + 0.01);
  float w = id < 3.5 ? aAux.y : 0.0;
  if (w > 0.0) {
    float a = -uKjJaw * w, c = cos(a), s = sin(a);
    vec2 d = p.xy - vec2(${JAW.hx.toFixed(2)}, ${JAW.hy.toFixed(2)});
    p.xy = vec2(${JAW.hx.toFixed(2)}, ${JAW.hy.toFixed(2)}) + vec2(c * d.x - s * d.y, s * d.x + c * d.y);
    n.xy = vec2(c * n.x - s * n.y, s * n.x + c * n.y);
  }
  float f = clamp((aPos.x - ${X0.toFixed(3)}) * ${(1 / DX).toFixed(6)}, 0.0, ${(NB - 1.001).toFixed(3)});
  int i = int(f);
  float t = f - float(i);
  vec4 q = normalize(mix(uKjQ[i], uKjQ[i + 1], t));
  vec3 o = vec3(p.x - aPos.x, p.y, p.z);
  P = mix(uKjP[i].xyz, uKjP[i + 1].xyz, t) + qrot(q, o);
  N = qrot(q, n);
  Qo = q;
}`;

const VS = SKIN_GLSL + `
uniform mat4 uViewProj;
uniform vec3 uCamPos;
uniform float uKjSea[${NP}];
out vec3 vW;
out vec3 vWs;
out vec3 vN;
out vec3 vR;
out vec4 vA;
out float vH;
out vec4 vQ;
void main() {
  vec3 P, N;
  kjSkin(P, N, vQ);
  vWs = P;
  // height above the local sea (probes along the body, head first)
  float k = clamp((${X1.toFixed(3)} - aPos.x) / ${(4 * DX).toFixed(4)}, 0.0, ${(NP - 1.001).toFixed(3)});
  int j = int(k);
  vH = P.y - mix(uKjSea[j], uKjSea[j + 1], k - float(j));
  vec2 dr = P.xz - uCamPos.xz;
  P.y -= dot(dr, dr) / (2.0 * 6360000.0);
  vW = P;
  vN = N;
  vR = aPos;
  vA = aAux;
  gl_Position = uViewProj * vec4(P, 1.0);
}`;

const SHADOW_VS = SKIN_GLSL + `
uniform vec4 uKjShC;
uniform vec3 uKjShX, uKjShY, uKjShZ;
out float vD;
void main() {
  vec3 P, N;
  vec4 Q;
  kjSkin(P, N, Q);
  vec3 q = P - uKjShC.xyz;
  vD = dot(q, uKjShZ);
  gl_Position = vec4(dot(q, uKjShX) * uKjShC.w, dot(q, uKjShY) * uKjShC.w, -vD * uKjShC.w, 1.0);
}`;
const SHADOW_FS = `
in float vD;
out vec4 o;
void main() { o = vec4(vD, 0.0, 0.0, 1.0); }`;

const FS = SKY_LOOKUP_GLSL + `
in vec3 vW;
in vec3 vWs;
in vec3 vN;
in vec3 vR;
in vec4 vA;
in float vH;
in vec4 vQ;
out vec4 oColor;
uniform vec3 uCamPos;
uniform sampler2D uEnv;
uniform vec3 uFlash;
uniform float uRain;
uniform float uTime;
uniform float uKjPixel;          // radians per pixel
uniform sampler2D uKjShadow;
uniform vec4 uKjShC;
uniform vec3 uKjShX, uKjShY, uKjShZ;
uniform float uKjShOn;
uniform vec4 uKjGlow;            // steady glow, surge glow (render units), pulse position (rest x), pulse width
uniform vec3 uKjWater;           // pouring off (0..1), waterline foam (0..1), time since surfacing
const float PI = 3.14159265359;
vec2 equirectUv(vec3 d) { return vec2(atan(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI); }
vec3 envL(vec3 d, float lod) { return textureLod(uEnv, equirectUv(normalize(d)), lod).rgb; }
vec3 kh3(vec3 p) {
  p = vec3(dot(p, vec3(127.1, 311.7, 74.7)), dot(p, vec3(269.5, 183.3, 246.1)), dot(p, vec3(113.5, 271.9, 124.6)));
  return fract(sin(p) * 43758.5453);
}
float kh1(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
float kn3(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(kh1(i), kh1(i + vec3(1, 0, 0)), f.x), mix(kh1(i + vec3(0, 1, 0)), kh1(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(kh1(i + vec3(0, 0, 1)), kh1(i + vec3(1, 0, 1)), f.x), mix(kh1(i + vec3(0, 1, 1)), kh1(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
vec3 qrotF(vec4 q, vec3 v) { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
// value noise and its gradient
vec4 kn3g(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f), du = 6.0 * f * (1.0 - f);
  float a = kh1(i), b = kh1(i + vec3(1, 0, 0)), c = kh1(i + vec3(0, 1, 0)), d = kh1(i + vec3(1, 1, 0));
  float e = kh1(i + vec3(0, 0, 1)), f1 = kh1(i + vec3(1, 0, 1)), g = kh1(i + vec3(0, 1, 1)), h = kh1(i + vec3(1, 1, 1));
  float k1 = b - a, k2 = c - a, k3 = e - a, k4 = a - b - c + d, k5 = a - c - e + g, k6 = a - b - e + f1, k7 = -a + b + c - d + e - f1 - g + h;
  float v = a + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z;
  vec3 gr = du * vec3(k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z, k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x, k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y);
  return vec4(v, gr);
}
// Worley: F1, F2, a random value of the nearest cell; g: gradient of F2 - F1 (kjGF1: of F1)
vec3 kjGF1 = vec3(0.0);
vec3 worleyG(vec3 p, out vec3 g) {
  vec3 i = floor(p), f = fract(p);
  float d1 = 9.0, d2 = 9.0, id = 0.0;
  vec3 r1 = vec3(0.0), r2 = vec3(0.0);
  for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    vec3 gg = vec3(float(x), float(y), float(z));
    vec3 h = kh3(i + gg);
    vec3 r = gg + 0.1 + 0.8 * h - f;
    float d = dot(r, r);
    if (d < d1) { d2 = d1; r2 = r1; d1 = d; r1 = r; id = h.z; } else if (d < d2) { d2 = d; r2 = r; }
  }
  float F1 = sqrt(d1), F2 = sqrt(d2);
  g = -r2 / max(F2, 1e-4) + r1 / max(F1, 1e-4);
  kjGF1 = -r1 / max(F1, 1e-4);
  return vec3(F1, F2, id);
}
float dss(float a, float b, float x) { float t = clamp((x - a) / (b - a), 0.0, 1.0); return 6.0 * t * (1.0 - t) / (b - a); }
// F1, F2 and a random value of the nearest cell
vec3 worley(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  float d1 = 9.0, d2 = 9.0, id = 0.0;
  for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    vec3 g = vec3(float(x), float(y), float(z));
    vec3 h = kh3(i + g);
    vec3 r = g + 0.1 + 0.8 * h - f;
    float d = dot(r, r);
    if (d < d1) { d2 = d1; d1 = d; id = h.z; } else if (d < d2) d2 = d;
  }
  return vec3(sqrt(d1), sqrt(d2), id);
}
float kjShadow(vec3 Pw, vec3 n) {
  if (uKjShOn < 0.5) return 1.0;
  vec3 q = Pw - uKjShC.xyz + n * 0.35;
  vec2 uv = vec2(dot(q, uKjShX), dot(q, uKjShY)) * uKjShC.w * 0.5 + 0.5;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return 1.0;
  float z = dot(q, uKjShZ);
  float s = 0.0;
  for (int i = 0; i < 5; i++) {
    vec2 o = i == 0 ? vec2(0.0) : vec2(i == 1 ? 1.0 : i == 2 ? -1.0 : 0.0, i == 3 ? 1.0 : i == 4 ? -1.0 : 0.0);
    float d = texture(uKjShadow, uv + o * (1.4 / ${SH}.0)).r;
    s += z >= d - 0.45 ? 1.0 : 0.0;
  }
  return s / 5.0;
}
void main() {
  vec3 P = vW;
  vec3 toCam = uCamPos - P;
  float dist = length(toCam);
  vec3 V = toCam / dist;
  vec3 ng = normalize(cross(dFdx(P), dFdy(P)));
  if (dot(ng, V) < 0.0) ng = -ng;
  vec3 N = normalize(vN);
  if (dot(N, ng) < 0.0) N = -N;
  float id = floor(vA.w + 0.01);
  vec3 p = vR;
  float mpp = dist * uKjPixel;        // metres per pixel
  vec3 alb;
  float rough = 0.72, F0 = 0.035, h = 0.0, cav = 1.0, glowMask = 0.0, plateTop = 0.0;
  vec3 gR = vec3(0.0);   // gradient of the relief height in rest space (m / m)
  if (id < 0.5) {
    // skin: cracked volcanic plates (larger on the back, finer on the head), nodules inside each plate
    float sc = mix(1.3, 0.55, smoothstep(55.0, 64.0, p.x));
    vec4 wp = kn3g(p * 0.45);
    vec3 q = p / sc + vec3(wp.x, kn3(p * 0.45 + 7.3), kn3(p * 0.45 + 3.1)) * 0.9;
    vec3 g1, g2;
    vec3 w1 = worleyG(q, g1);
    vec3 gF1a = kjGF1;
    float f1 = 1.0 - smoothstep(0.3, 1.2, mpp / sc);
    // a rounded nodule per cell (heights vary), a crevice along the cell borders
    float s1 = smoothstep(0.0, 0.9, w1.x);
    float dome = pow(1.0 - s1, 1.4);
    float e1 = w1.y - w1.x;
    float crev = 1.0 - smoothstep(0.0, 0.1, e1);
    float hgt = (0.45 + 0.55 * w1.z) * 0.2 * f1;
    float f2 = 1.0 - smoothstep(0.3, 1.2, mpp / (sc * 0.3));
    vec3 w2 = vec3(1.0), gF1b = vec3(0.0);
    g2 = vec3(0.0);
    if (f2 > 0.0) { w2 = worleyG(p / (sc * 0.3) + 17.0, g2); gF1b = kjGF1; }
    float wart = pow(1.0 - smoothstep(0.0, 0.8, w2.x), 2.0);
    vec4 n1 = kn3g(p * 3.1);
    // (the skin's relief is all in the analytic gradient: smooth at any distance)
    gR = (hgt * (-1.4 * pow(1.0 - s1, 0.4) * dss(0.0, 0.9, w1.x) * gF1a + 0.35 * dss(0.0, 0.1, e1) * g1)) / sc
       + 0.05 * f2 * (-2.0 * (1.0 - smoothstep(0.0, 0.8, w2.x)) * dss(0.0, 0.8, w2.x) * gF1b) / (sc * 0.3)
       + 0.025 * f2 * n1.yzw * 3.1;
    cav = mix(1.0, mix(1.0, 0.3, crev) * mix(0.7, 1.0, dome), f1) * mix(1.0, mix(0.75, 1.0, wart), f2);
    plateTop = dome * f1;
    float tone = mix(1.0, 0.92 + 0.18 * w1.z, f1) * (0.85 + 0.3 * kn3(p * 0.11));
    alb = vec3(0.03, 0.03, 0.029) * tone;
    // weathered, paler crests of the plates; a paler, banded belly
    // weathered, paler tops of the biggest nodules
    alb = mix(alb, vec3(0.048, 0.047, 0.045), dome * smoothstep(0.6, 0.95, w1.z) * f1 * 0.6);
    float belly = smoothstep(-3.0, -8.0, p.y) * smoothstep(-25.0, -12.0, p.x) * (1.0 - smoothstep(56.0, 60.0, p.x));
    alb = mix(alb, vec3(0.058, 0.055, 0.05), belly * 0.7);
    // inside the mouth: wet flesh
    alb = mix(alb, vec3(0.15, 0.03, 0.026), vA.x);
    rough = mix(0.72, 0.4, vA.x);
  } else if (id < 2.5) {
    // eye: dark, with a dull amber iris, glassy
    vec3 d = p - vec3(64.85, 2.5, sign(p.z) * 3.45);
    float rr = length(d.xy);
    alb = mix(vec3(0.035, 0.018, 0.004), vec3(0.004), smoothstep(0.16, 0.1, rr) + smoothstep(0.42, 0.6, rr));
    rough = 0.06; F0 = 0.05;
  } else if (id < 3.5) {
    if (p.x > 50.0) { alb = vec3(0.40, 0.36, 0.27) * (0.75 + 0.4 * kn3(p * 9.0)); rough = 0.32; }
    else { alb = vec3(0.07, 0.066, 0.06); rough = 0.4; }
  } else {
    // dorsal plate: bone, dark and skin-like at its root, pale and weathered toward the margin,
    // ribbed from root to tips, pitted
    float r = vA.x, seed = fract(vA.w);
    float th = vA.y * 26.0 + seed * 40.0;
    float rib = kn3(vec3(th, r * 1.4, seed * 13.0)) * 0.7 + kn3(vec3(th * 2.1, r * 2.6, seed * 7.0)) * 0.3;
    vec3 gp = vec3(0.0);
    float fr = 1.0 - smoothstep(0.3, 1.2, mpp / 0.7);
    vec3 wf = vec3(0.0, 1.0, 0.0);
    if (fr > 0.0) wf = worleyG(p / 0.7 + seed * 31.0, gp);
    float pit = smoothstep(0.0, 0.16, wf.y - wf.x);
    vec4 nb = kn3g(p * 1.7 + seed * 5.0);
    h = rib * 0.035 * (1.0 - smoothstep(0.3, 1.0, mpp));
    gR = 0.012 * fr * dss(0.0, 0.16, wf.y - wf.x) * gp / 0.7 + 0.06 * nb.yzw * 1.7;
    cav = mix(1.0, mix(0.8, 1.0, pit), fr) * (0.75 + 0.25 * rib);
    vec3 bone = vec3(0.075, 0.075, 0.072) * (0.75 + 0.45 * rib) * (0.8 + 0.4 * nb.x);
    alb = mix(vec3(0.03, 0.03, 0.029), bone, smoothstep(0.15, 0.65, r));
    alb = mix(alb, vec3(0.11, 0.108, 0.104), smoothstep(0.85, 1.0, r) * 0.4 * (0.5 + nb.x));
    rough = 0.6;
    glowMask = smoothstep(0.6, 1.0, r) * smoothstep(0.35, 0.8, rib);
  }
  // relief: surface gradient from screen derivatives (Mikkelsen 2010)
  {
    vec3 dpdx = dFdx(P), dpdy = dFdy(P);
    float dhx = dFdx(h), dhy = dFdy(h);
    vec3 r1 = cross(dpdy, N), r2 = cross(N, dpdx);
    float det = dot(dpdx, r1);
    vec3 sg = sign(det) * (dhx * r1 + dhy * r2);
    N = normalize(abs(det) * N - sg);
    vec3 gW = qrotF(normalize(vQ), gR);
    N = normalize(N - (gW - N * dot(gW, N)));
  }
  // wet: a film of sea water everywhere, running down in rivulets (in sheets when it has just risen)
  float above = vH;
  float hz = dot(p.xz, vec2(0.83, 0.56)) + p.y * 0.21;
  float run = kn3(vec3(hz * 1.3, (P.y + uTime * (1.5 + 5.0 * uKjWater.x)) * 0.2, 0.0)) * 0.65 + kn3(vec3(hz * 3.7, (P.y + uTime * 3.0) * 0.6, 3.0)) * 0.35;
  float film = smoothstep(0.42, 0.75, run);
  float wet = clamp(0.7 + 0.3 * smoothstep(12.0, 0.0, above) + uRain, 0.0, 1.0);
  float coat = wet * mix(0.45, 0.9, max(film, uKjWater.x * 0.7));
  alb *= mix(1.0, 0.7, coat * (id > 3.5 ? 0.8 : 0.6));
  float coatRough = mix(0.32, 0.12, max(film * 0.6, uKjWater.x * film));
  // whitewater clinging at the waterline
  float band = smoothstep(-0.8, 0.1, above) * (1.0 - smoothstep(0.3, 1.0 + 3.0 * uKjWater.y, above));
  float fnz = kn3(vec3(hz * 0.7, P.y * 0.45 - uTime * 0.6, 7.0)) * 0.6 + kn3(vec3(hz * 2.3, P.y * 1.3 - uTime, 1.0)) * 0.4;
  float foam = clamp(band * smoothstep(0.4, 0.62, fnz) * uKjWater.y, 0.0, 1.0);
  alb = mix(alb, vec3(0.72), foam);
  coat *= 1.0 - foam;
  rough = mix(rough, 0.95, foam);

  vec3 L = uSunDir;
  vec3 sunE = sunIlluminance();
  vec2 suv = (P.xz - L.xz * (max(P.y, 0.0) / max(L.y, 0.02)) - uShaftRect.xy) * uShaftRect.zw;
  float sunVis = mix(uHazeSun, texture(uShaftShadow, suv).r, smoothstep(0.0, 0.04, min(min(suv.x, suv.y), min(1.0 - suv.x, 1.0 - suv.y))));
  sunVis *= kjShadow(vWs, ng);
  vec3 T1 = normalize(cross(N, abs(N.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  vec3 T2 = cross(N, T1);
  vec3 Eamb = (envL(N, 5.0) * 0.4 + (envL(N + T1, 5.0) + envL(N - T1, 5.0) + envL(N + T2, 5.0) + envL(N - T2, 5.0)) * 0.15) * PI;
  float ao = vA.z * cav;
  float nl = max(dot(N, L), 0.0), nv = max(dot(N, V), 1e-3);
  vec3 H = normalize(L + V);
  float nh = max(dot(N, H), 0.0), vh = max(dot(V, H), 0.0);
  float a2 = max(pow(rough, 4.0), 1e-4);
  float D = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
  float Fh = F0 + (1.0 - F0) * pow(1.0 - vh, 5.0);
  float Fv = F0 + (1.0 - F0) * pow(1.0 - nv, 5.0) / (1.0 + 4.0 * rough);
  vec3 Esun = sunE * sunVis * nl;
  vec3 Ef = uFlash * (0.35 + 0.65 * max(dot(N, uFlashDir), 0.0));
  vec3 R = reflect(-V, N);
  // the coat: a water film (F0 0.02) over the rough skin
  float c2 = max(pow(coatRough, 4.0), 1e-4);
  float Dc = c2 / (PI * pow(nh * nh * (c2 - 1.0) + 1.0, 2.0));
  float Fc = (0.02 + 0.98 * pow(1.0 - vh, 5.0)) * coat;
  float Fcv = (0.02 + 0.98 * pow(1.0 - nv, 5.0) / (1.0 + 3.0 * coatRough)) * coat;
  vec3 col = alb / PI * (Esun + Eamb * ao + Ef) * (1.0 - Fv) * (1.0 - Fcv);
  col += (Esun * min(D, 300.0) * Fh / (4.0 * max(nl, nv)) + envL(R, mix(2.0, 7.0, rough)) * Fv * ao) * (1.0 - Fcv);
  // (reflection occlusion: cavities and the underside of the body see little of the sky)
  float specOcc = clamp(ao * 1.2, 0.0, 1.0) * mix(0.5, 1.0, smoothstep(-0.6, 0.3, R.y));
  col += Esun * min(Dc, 2000.0) * Fc / (4.0 * max(nl, nv)) + envL(R, mix(0.8, 4.0, coatRough * 2.5)) * Fcv * specOcc;
  // bioluminescence along the plates' margins and ribs: a pulse running up the back now and then
  if (id > 3.5) {
    float pulse = exp(-pow((p.x - uKjGlow.z) / uKjGlow.w, 2.0));
    col += vec3(0.16, 0.48, 1.0) * glowMask * (uKjGlow.x * (0.35 + 0.65 * pulse) + uKjGlow.y * pulse);
  }
  vec3 aL, aT;
  aerial(-V, dist * 0.001, aL, aT);
  col = col * aT + aL;
  col = withShafts(col, uCamPos, -V, dist);
  col = applyFog(col, -V, dist);
  oColor = vec4(col, 1.0);
}`;

// sea probe at NP points along the body (displaced surface, choppy displacement inverted)
const PROBE_FS = `
uniform sampler2D uDisp[3];
uniform float uSize[4];
uniform float uTexel[4];
uniform vec2 uPts[${NP}];
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
  for (int k = 0; k < 4; k++) x0 = p - disp(x0).xz;
  o = vec4(disp(x0), 1.0);
}`;

class SeaProbe {
  constructor(gl) {
    this.gl = gl;
    this.prog = new Program(gl, FULLSCREEN_VS, PROBE_FS, { label: 'kaiju-probe' });
    this.tex = new Texture(gl, { w: NP, h: 1, format: 'rgba32f', filter: 'nearest' });
    this.target = new Target(gl, [this.tex]);
    this.pbo = gl.createBuffer();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, NP * 16, gl.STREAM_READ);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    this.data = new Float32Array(NP * 4);
    this.h = null;
    this.fence = null;
  }
  run(ot, pts, footprint, sync) {
    const gl = this.gl;
    if (this.fence && !sync) {
      if (gl.clientWaitSync(this.fence, 0, 0) === gl.TIMEOUT_EXPIRED) return this.h;
      gl.deleteSync(this.fence);
      this.fence = null;
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.data);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      this.decode();
    }
    fullscreen(gl, this.prog, { uDisp: ot.disp.slice(0, 3), uSize: ot.sizes, uTexel: ot.texels, uPts: pts, uFootprint: footprint }, this.target);
    if (sync) {
      gl.readPixels(0, 0, NP, 1, gl.RGBA, gl.FLOAT, this.data);
      this.decode();
      return this.h;
    }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
    gl.readPixels(0, 0, NP, 1, gl.RGBA, gl.FLOAT, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    this.fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    return this.h;
  }
  decode() {
    const h = [];
    for (let i = 0; i < NP; i++) h.push(this.data[i * 4 + 1]);
    if (h.every(Number.isFinite)) this.h = h;
  }
}

// ---------------------------------------------------------------- the swim path
// control points in the camera's frame at spawn: [right, forward] metres
const COURSE = [[-1050, 1900], [-690, 1100], [-370, 640], [0, 540], [370, 590], [720, 880], [1100, 1450], [1450, 2100]];
function buildPath(base, yaw) {
  const f = [Math.cos(yaw), Math.sin(yaw)], r = [-Math.sin(yaw), Math.cos(yaw)];
  const C = COURSE.map(([a, b]) => [base[0] + r[0] * a + f[0] * b, base[1] + r[1] * a + f[1] * b]);
  const ext = (p, q, d) => { const dx = p[0] - q[0], dz = p[1] - q[1], l = Math.hypot(dx, dz); return [p[0] + dx / l * d, p[1] + dz / l * d]; };
  const Q = [ext(C[0], C[1], 700), ext(C[0], C[1], 350), ...C, ext(C[C.length - 1], C[C.length - 2], 400)];
  const pts = [];
  for (let i = 0; i < Q.length - 3; i++) {
    const [p0, p1, p2, p3] = [Q[i], Q[i + 1], Q[i + 2], Q[i + 3]];
    for (let k = 0; k < 200; k++) {
      const t = k / 200, t2 = t * t, t3 = t2 * t;
      const cr = (a, b, c, d) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
      pts.push([cr(p0[0], p1[0], p2[0], p3[0]), cr(p0[1], p1[1], p2[1], p3[1])]);
    }
  }
  pts.push(Q[Q.length - 2]);
  // resample to 1 m of arc
  const arc = [0];
  for (let i = 1; i < pts.length; i++) arc.push(arc[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const L = arc[arc.length - 1], n = Math.floor(L) + 1;
  const X = new Float32Array(n), Z = new Float32Array(n);
  let j = 0;
  for (let s = 0; s < n; s++) {
    while (j < arc.length - 2 && arc[j + 1] < s) j++;
    const t = (s - arc[j]) / Math.max(arc[j + 1] - arc[j], 1e-6);
    X[s] = pts[j][0] + (pts[j + 1][0] - pts[j][0]) * t;
    Z[s] = pts[j][1] + (pts[j + 1][1] - pts[j][1]) * t;
  }
  const arcOf = (c) => { let best = 0, bd = 1e18; for (let s = 0; s < n; s++) { const d = (X[s] - c[0]) ** 2 + (Z[s] - c[1]) ** 2; if (d < bd) { bd = d; best = s; } } return best; };
  return { X, Z, n, marks: C.map(arcOf) };
}
function pathAt(P, a, out) {
  const n = P.n;
  const i = clamp(Math.floor(a), 0, n - 2);
  const t = a - i;
  let tx = P.X[i + 1] - P.X[i], tz = P.Z[i + 1] - P.Z[i];
  const l = Math.hypot(tx, tz) || 1;
  tx /= l; tz /= l;
  // (t runs outside 0..1 beyond the ends: straight extrapolation)
  out[0] = P.X[i] + (P.X[i + 1] - P.X[i]) * t;
  out[1] = P.Z[i] + (P.Z[i + 1] - P.Z[i]) * t;
  out[2] = tx; out[3] = tz;
  return out;
}

// body section at rest x (head approximated)
const secW = (x) => (x > 58 ? 5.2 : profW(x));
const secT = (x) => (x > 58 ? 5.6 : profT(x) + 0.7);
const secB = (x) => (x > 58 ? 5.5 : profB(x));
const finTop = (x) => (x > -122 && x < 52 ? plateSize(x) * 0.9 : 0);

export class Kaiju {
  // mode: 'off' | 'auto' | 'now' | seconds into a pass; gap: seconds between passes
  constructor(gl, { mode = 'auto', gap = null, firstDelay = 6 } = {}) {
    this.gl = gl;
    // ?kaiju= : 0 off, 1 / absent automatic, now, seconds into a pass, or a named moment
    const named = { approach: 30, cross: 80, rise: 102, roar: 107, wake: 128, leave: 165 };
    mode = String(mode).toLowerCase();
    mode = mode === '0' || mode === 'off' ? 'off' : mode === 'now' ? 'now' : mode in named ? named[mode] : Number(mode) > 1 ? Number(mode) : 'auto';
    this.enabled = mode !== 'off';
    this.gapRange = gap ? [gap, gap] : [120, 180];
    this.ready = false;
    this.failed = false;
    this.state = 'wait';
    this.countdown = firstDelay;
    this.pendingSeek = mode === 'now' ? 0 : typeof mode === 'number' ? mode : null;
    this.clock = 0;
    this.events = [];
    this.P = new Float32Array(NB * 4);
    this.Q = new Float32Array(NB * 4);
    this.trail = [];
    this.sea = new Float32Array(NP);
    this.seaSmooth = new Float32Array(NP);
    this.water = { w: new Float32Array(NP), s: new Float32Array(NP) };
    this.wu = { uKjOn: 0 };
    this.ring = [0, 0, 0, 0];
    this.glow = [0, 0, 0, 20];
    this.pulseT = 25;
    this.visible = false;
    if (this.enabled) this.load();
  }

  load() {
    if (this.loading) return;
    this.loading = true;
    const done = (m, again = true) => {
      try { this.build(m); } catch (e) {
        if (again) { console.warn('kaiju build retry: ' + (e && e.message ? e.message : e)); setTimeout(() => done(m, false), 1500); return; }
        this.failed = true;
        console.error('kaiju build: ' + (e && e.message ? e.message : e));
      }
    };
    try {
      const w = new Worker(new URL('./sculpt-worker.js', import.meta.url), { type: 'module' });
      w.onmessage = (e) => { done(e.data); w.terminate(); };
      w.onerror = (e) => { e.preventDefault?.(); w.terminate(); const m = sculpt(); m.low = sculpt({ cell: 1.0, low: true }); done(m); };
      w.postMessage({ cell: 0.5 });
    } catch {
      const m = sculpt();
      m.low = sculpt({ cell: 1.0, low: true });
      done(m);
    }
  }

  build(m) {
    const gl = this.gl;
    this.mesh = makeMesh(gl, [[m.pos, 3], [m.nrm, 3], [m.aux, 4]], m.idx);
    // a coarse version for the shadow map and for far away
    this.meshLow = m.low ? makeMesh(gl, [[m.low.pos, 3], [m.low.nrm, 3], [m.low.aux, 4]], m.low.idx) : this.mesh;
    this.prog = new Program(gl, VS, FS, { label: 'kaiju' });
    this.shProg = new Program(gl, SHADOW_VS, SHADOW_FS, { label: 'kaiju-shadow' });
    this.shTex = new Texture(gl, { w: SH, h: SH, format: 'r32f', filter: 'nearest' });
    this.shTarget = new Target(gl, [this.shTex], { depth: true });
    this.probe = new SeaProbe(gl);
    this.splash = new Splash(gl, 9000);
    this.fins = [];
    for (let i = 0; i < m.fins.length; i += 4) this.fins.push([m.fins[i], m.fins[i + 1], m.fins[i + 2], m.fins[i + 3]]);
    // surface samples for water pouring off the body (body vertices only)
    const ns = 2600;
    this.samples = new Float32Array(ns * 4);   // rest xyz, wet clock
    this.sampleN = new Float32Array(ns * 3);    // rest normal
    this.sampleW = new Float32Array(ns * 3);
    this.samplePrev = new Float32Array(ns * 3);
    for (let k = 0; k < ns; k++) {
      let v;
      do { v = Math.floor(Math.random() * m.bodyVerts); } while (m.pos[v * 3 + 1] < -6 && Math.random() < 0.8);
      this.samples[k * 4] = m.pos[v * 3]; this.samples[k * 4 + 1] = m.pos[v * 3 + 1]; this.samples[k * 4 + 2] = m.pos[v * 3 + 2]; this.samples[k * 4 + 3] = -1e3;
      this.sampleN[k * 3] = m.nrm[v * 3]; this.sampleN[k * 3 + 1] = m.nrm[v * 3 + 1]; this.sampleN[k * 3 + 2] = m.nrm[v * 3 + 2];
    }
    this.buildMs = m.ms;
    this.ready = true;
    console.log(`kaiju sculpt: ${m.pos.length / 3} vertices, ${m.idx.length / 3} triangles, ${m.ms} ms`);
  }

  setEnabled(on) {
    this.enabled = on;
    if (on) { this.load(); if (this.state === 'wait') this.countdown = Math.min(this.countdown, 6); }
    else { this.state = 'wait'; this.trail = []; this.visible = false; this.wu = { uKjOn: 0 }; if (this.splash) this.splash.n = 0; }
  }

  // appear now: a pass already under way. Default: already crossing in front, ~12 s before it rears up and roars
  summon(at = 90) {
    if (!this.enabled) this.setEnabled(true);
    this.pendingSeek = at;
  }

  // ---------------------------------------------------------------- pass script
  startPass(cam, seek = 0) {
    this.path = buildPath(cam.base, cam.yaw);
    this.camBase = [cam.base[0], cam.base[1]];
    const mk = this.path.marks;
    this.mk = { a0: mk[0], x0: mk[2], mid: mk[3], x1: mk[4], dive: mk[5], end: mk[7] };
    this.S = this.mk.a0;
    this.phase = 0;
    this.rise = -1;
    this.riseDone = false;
    this.tau = 0;
    this.state = 'pass';
    this.trail = [];
    this.lastRec = null;
    this.seaSmooth.fill(0);
    this.sea.fill(0);
    if (this.splash) this.splash.n = 0;
    this.probe && (this.probe.h = null);
    for (let k = 0; k < this.samples.length / 4; k++) this.samples[k * 4 + 3] = -1e3;
    if (seek > 0) {
      this.seeking = true;
      const st = 0.25;
      for (let t = 0; t < seek; t += st) { this.clock += st; this.step(st, null); if (this.state !== 'pass') break; }
      this.seeking = false;
    }
  }

  // script values at arc S (and the rearing-up timeline rho)
  script() {
    const S = this.S, m = this.mk, rho = this.rise;
    let speed = mix(22, 12.5, sstep(m.x0 - 380, m.x0, S));
    speed = mix(speed, 18, sstep(m.x1, m.dive, S));
    let sub = mix(-2.6, 0.6, sstep(m.x0 - 650, m.x0 - 60, S));
    sub = mix(sub, -6.0, sstep(m.dive - 80, m.dive + 260, S));
    sub = mix(sub, -42, sstep(m.dive + 300, m.end - 120, S));
    let kn = mix(0.2, 0.37, sstep(m.x0 - 650, m.x0 - 60, S));
    kn = mix(kn, 0.05, sstep(m.dive - 80, m.dive + 260, S));
    let aBody = -0.11 * sstep(m.dive + 200, m.end - 300, S), aNeck = 0, aHead = 0, yaw = 0, jaw = 0, pour = 0, lift = 0;
    if (rho >= 0) {
      speed *= mix(1, 0.32, sstep(0, 4, rho) * (1 - sstep(19, 28, rho)));
      lift = sstep(1.0, 8.0, rho) * (1 - sstep(17.5, 25, rho));
      sub += 6.5 * lift;
      aBody += 0.43 * lift;
      aNeck += 0.55 * lift;
      aHead += -0.55 * lift;
      yaw = 0.42 * win(rho, 6, 10, 16.5, 21);
      const roarW = win(rho, 10.2, 11.2, 14.0, 15.4);
      aHead += 0.26 * roarW;
      aNeck += 0.08 * roarW;
      jaw = 0.62 * win(rho, 10.4, 11.3, 14.0, 15.2);
      pour = win(rho, 2.5, 6, 11, 20);
    }
    kn *= 1 - lift;
    return { speed, sub, kn, aBody, aNeck, aHead, yaw, jaw, pour, lift };
  }

  // advance the creature (dt seconds); ot: sea textures (null while seeking)
  step(dt, cam) {
    this.tau += dt;
    // rear up once, as it crosses the middle of the view
    if (!this.riseDone && this.rise < 0 && this.S >= this.mk.mid - 90) { this.rise = 0; this.ringStart = true; }
    if (this.rise >= 0) {
      this.rise += dt;
      if (this.rise > 30) { this.rise = -1; this.riseDone = true; }
      if (this.rise >= 10.6 && !this.roared) { this.roared = true; if (!this.seeking) { const p = this.skin(64, 1, 0, [0, 0, 0]), c = this.camPos || [0, 0, 0]; this.events.push({ type: 'roar', dist: Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]) }); } }
    } else this.roared = this.riseDone;
    const sc = this.script();
    this.sc = sc;
    this.S += sc.speed * dt;
    this.phase += (dt * sc.speed) / (150 * 0.78);
    if (this.S >= this.mk.end) { this.endPass(); return; }
    this.pose(sc, cam);
    this.waterline(sc);
    this.record(dt);
  }

  endPass() {
    this.state = 'wait';
    this.countdown = this.gapRange[0] + Math.random() * (this.gapRange[1] - this.gapRange[0]);
    this.visible = false;
  }

  pose(sc, cam) {
    const loc = new Float32Array(NB * 3);   // arc, lateral, height (path frame)
    const S = this.S;
    for (let i = 0; i < NB; i++) {
      const x = NODE_X[i], s = XREF - x;
      const A = 0.35 + 15.5 * Math.pow(Math.max(0, (s - 70) / 140), 1.7);
      const lat = A * Math.sin(2 * Math.PI * (s / 150 - this.phase));
      let y = sc.sub - 12.6 + (x < 0 ? 0.055 * x : 0) + sc.kn * Math.max(0, x - 36) * sstep(30, 42, x);
      y += 0.5 * Math.sin(2 * Math.PI * (s / 150 - this.phase) + 1.3) * sstep(60, 160, s);
      loc[i * 3] = S - s; loc[i * 3 + 1] = lat; loc[i * 3 + 2] = y;
    }
    // bends, distributed over a few bones each: (from, to) rest x, angle, plane (0: pitch, 1: yaw)
    const bend = (xa, xb, ang, plane) => {
      if (Math.abs(ang) < 1e-5) return;
      const ia = Math.round((xa - X0) / DX), ib = Math.round((xb - X0) / DX), cnt = ib - ia + 1;
      const c = Math.cos(ang / cnt), s = Math.sin(ang / cnt);
      for (let k = ia; k <= ib; k++) {
        const pa = loc[k * 3], pl = loc[k * 3 + 1], py = loc[k * 3 + 2];
        for (let j = k + 1; j < NB; j++) {
          const da = loc[j * 3] - pa;
          if (plane === 0) {
            const dy = loc[j * 3 + 2] - py;
            loc[j * 3] = pa + da * c - dy * s; loc[j * 3 + 2] = py + da * s + dy * c;
          } else {
            const dl = loc[j * 3 + 1] - pl;
            loc[j * 3] = pa + da * c - dl * s; loc[j * 3 + 1] = pl + da * s + dl * c;
          }
        }
      }
    };
    bend(-12, 18, sc.aBody, 0);
    bend(34, 54, sc.aNeck, 0);
    bend(55, 60, sc.aHead, 0);
    // turn the head toward the camera when rearing up
    if (sc.yaw > 0 && cam) {
      const tmp = [0, 0, 0, 0];
      pathAt(this.path, loc[(NB - 1) * 3], tmp);
      const side = Math.sign((cam.pos[0] - tmp[0]) * -tmp[3] + (cam.pos[2] - tmp[1]) * tmp[2]) || 1;
      this.yawSide = side;
    }
    bend(36, 56, sc.yaw * (this.yawSide || 1), 1);
    // to the world
    const tmp = [0, 0, 0, 0];
    const W = new Float32Array(NB * 3);
    for (let i = 0; i < NB; i++) {
      pathAt(this.path, loc[i * 3], tmp);
      const nx = -tmp[3], nz = tmp[2];
      const k = clamp((X1 - NODE_X[i]) / (4 * DX), 0, NP - 1.001), j = Math.floor(k), t = k - j;
      const heave = 0.65 * (this.seaSmooth[j] + (this.seaSmooth[j + 1] - this.seaSmooth[j]) * t);
      W[i * 3] = tmp[0] + nx * loc[i * 3 + 1];
      W[i * 3 + 1] = loc[i * 3 + 2] + heave;
      W[i * 3 + 2] = tmp[1] + nz * loc[i * 3 + 1];
    }
    // bone frames: forward along the chain, up as vertical as the chain allows
    let pq = null;
    for (let i = 0; i < NB; i++) {
      const a = Math.max(i - 1, 0), b = Math.min(i + 1, NB - 1);
      let tx = W[b * 3] - W[a * 3], ty = W[b * 3 + 1] - W[a * 3 + 1], tz = W[b * 3 + 2] - W[a * 3 + 2];
      const tl = Math.hypot(tx, ty, tz) || 1; tx /= tl; ty /= tl; tz /= tl;
      let ux = -ty * tx, uy = 1 - ty * ty, uz = -ty * tz;
      const ul = Math.hypot(ux, uy, uz) || 1; ux /= ul; uy /= ul; uz /= ul;
      const zx = ty * uz - tz * uy, zy = tz * ux - tx * uz, zz = tx * uy - ty * ux;
      // rotation matrix columns (T, U, Z) -> quaternion
      const m00 = tx, m10 = ty, m20 = tz, m01 = ux, m11 = uy, m21 = uz, m02 = zx, m12 = zy, m22 = zz;
      let qw, qx, qy, qz;
      const tr = m00 + m11 + m22;
      if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; qw = 0.25 * s; qx = (m21 - m12) / s; qy = (m02 - m20) / s; qz = (m10 - m01) / s; }
      else if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1 + m00 - m11 - m22) * 2; qw = (m21 - m12) / s; qx = 0.25 * s; qy = (m01 + m10) / s; qz = (m02 + m20) / s; }
      else if (m11 > m22) { const s = Math.sqrt(1 + m11 - m00 - m22) * 2; qw = (m02 - m20) / s; qx = (m01 + m10) / s; qy = 0.25 * s; qz = (m12 + m21) / s; }
      else { const s = Math.sqrt(1 + m22 - m00 - m11) * 2; qw = (m10 - m01) / s; qx = (m02 + m20) / s; qy = (m12 + m21) / s; qz = 0.25 * s; }
      if (pq && pq[0] * qx + pq[1] * qy + pq[2] * qz + pq[3] * qw < 0) { qx = -qx; qy = -qy; qz = -qz; qw = -qw; }
      pq = [qx, qy, qz, qw];
      this.P[i * 4] = W[i * 3]; this.P[i * 4 + 1] = W[i * 3 + 1]; this.P[i * 4 + 2] = W[i * 3 + 2]; this.P[i * 4 + 3] = 1;
      this.Q[i * 4] = qx; this.Q[i * 4 + 1] = qy; this.Q[i * 4 + 2] = qz; this.Q[i * 4 + 3] = qw;
    }
    this.jaw = -0.035 + sc.jaw;
  }

  // CPU skinning of a rest point (same as the vertex shader, without the jaw)
  skin(x, y, z, out) {
    const f = clamp((x - X0) / DX, 0, NB - 1.001), i = Math.floor(f), t = f - i;
    const P = this.P, Q = this.Q, a = i * 4, b = a + 4;
    let qx = Q[a] + (Q[b] - Q[a]) * t, qy = Q[a + 1] + (Q[b + 1] - Q[a + 1]) * t, qz = Q[a + 2] + (Q[b + 2] - Q[a + 2]) * t, qw = Q[a + 3] + (Q[b + 3] - Q[a + 3]) * t;
    const ql = Math.hypot(qx, qy, qz, qw) || 1; qx /= ql; qy /= ql; qz /= ql; qw /= ql;
    // v + 2 q x (q x v + w v), v = (0, y, z)
    const cx = qy * z - qz * y + qw * 0, cy = qz * 0 - qx * z + qw * y, cz = qx * y - qy * 0 + qw * z;
    const rx = 2 * (qy * cz - qz * cy), ry = 2 * (qz * cx - qx * cz), rz = 2 * (qx * cy - qy * cx);
    out[0] = P[a] + (P[b] - P[a]) * t + rx;
    out[1] = P[a + 1] + (P[b + 1] - P[a + 1]) * t + y + ry;
    out[2] = P[a + 2] + (P[b + 2] - P[a + 2]) * t + z + rz;
    return out;
  }
  skinDir(x, y, z, out) {
    // (the bone rotation at the shoulders-to-neck region is close enough for a direction)
    const i = 44 * 4, Q = this.Q;
    const qx = Q[i], qy = Q[i + 1], qz = Q[i + 2], qw = Q[i + 3];
    const cx = qy * z - qz * y + qw * x, cy = qz * x - qx * z + qw * y, cz = qx * y - qy * x + qw * z;
    out[0] = x + 2 * (qy * cz - qz * cy); out[1] = y + 2 * (qz * cx - qx * cz); out[2] = z + 2 * (qx * cy - qy * cx);
    return out;
  }
  // sea height under rest x of the body
  seaAtX(x) {
    const k = clamp((X1 - x) / (4 * DX), 0, NP - 1.001), j = Math.floor(k), t = k - j;
    return this.sea[j] + (this.sea[j + 1] - this.sea[j]) * t;
  }

  // waterline footprint of the body at the probe nodes; the front contact; surface strength
  waterline(sc) {
    const w = this.water.w, s = this.water.s;
    this.front = -1;
    let surf = 0, cnt = 0, maxTop = -1e9;
    for (let k = 0; k < NP; k++) {
      const i = PROBE_NODE[k], x = NODE_X[i];
      const yc = this.P[i * 4 + 1], h = this.sea[k];
      const rel = h - yc, T = secT(x), B = secB(x), Wd = secW(x), F = finTop(x);
      maxTop = Math.max(maxTop, yc + T + F - h, yc + (x > 50 ? 6 : T) - h);
      let ww = 0, ss = 0;
      if (rel > -B && rel < T) { ww = Wd * Math.sqrt(1 - (rel / (rel > 0 ? T : B)) ** 2); ss = clamp(ww / 5, 0.35, 1); }
      else if (rel >= T && rel < T + F) { ww = 1.2; ss = 0.3; }
      w[k] = ww; s[k] = ss;
      if (ww > 0 && this.front < 0) this.front = k;
      if (ww > 0) { surf += ss; cnt++; }
    }
    this.maxTop = maxTop;
    this.surf = cnt ? surf / cnt : 0;
    this.visible = maxTop > -1.5;
    // the Kelvin source: the front of the contact, or the shoulders of a body just under the surface
    let k = this.front, str;
    if (k < 0) {
      k = 6;
      const i = PROBE_NODE[k];
      const depth = this.sea[k] - (this.P[i * 4 + 1] + secT(NODE_X[i]));
      str = 0.45 * Math.exp(-Math.max(depth, 0) / 5);
    } else str = clamp(0.35 + w[k] / 4, 0.35, 1) * (k <= 3 ? 1 : 0.7);
    const i = PROBE_NODE[k];
    this.src = [this.P[i * 4], this.P[i * 4 + 2], sc.speed, str, w[k]];
    // wake strength: how much of the creature cuts the surface
    this.wakeStr = Math.max(this.surf * Math.min(cnt / 6, 1), str * 0.6);
  }

  record() {
    const [x, z, U, , w0] = this.src;
    const last = this.lastRec;
    if (!last || Math.hypot(x - last.x, z - last.z) > 15 || this.clock - last.t > 3) {
      this.lastRec = { x, z, t: this.clock, str: this.wakeStr, w: Math.max(w0, 6 + 8 * this.surf), U };
      this.trail.unshift(this.lastRec);
      if (this.trail.length > KJT - 1) this.trail.pop();
    }
  }

  // ---------------------------------------------------------------- per frame
  update(dt, ot, cam, angStep, sync, sun) {
    const t0 = performance.now();
    this.update_(dt, ot, cam, angStep, sync, sun);
    // (debug: hidden = simulated but not drawn, for A/B timing)
    if (this.hidden) { this.wu = { uKjOn: 0 }; this.shadowOn = 0; }
    this.cpuMs = (this.cpuMs ?? 0) * 0.95 + (performance.now() - t0) * 0.05;
  }

  update_(dt, ot, cam, angStep, sync, sun) {
    dt = Math.max(dt, 0);
    if (!this.enabled) { this.wu = { uKjOn: 0 }; return; }
    if (!this.ready) return;
    this.camPos = cam.pos;
    this.clock += dt;
    if (this.pendingSeek !== null) {
      const s = this.pendingSeek;
      this.pendingSeek = null;
      this.startPass(cam, s);
    }
    if (this.state === 'wait') {
      this.countdown -= dt;
      if (this.countdown <= 0) this.startPass(cam, 0);
    }
    if (this.state === 'pass') {
      // the sea along the body (as drawn at that distance)
      if (ot && this.P[3] === 1) {
        const pts = new Float32Array(NP * 2);
        for (let k = 0; k < NP; k++) { const i = PROBE_NODE[k]; pts[k * 2] = this.P[i * 4]; pts[k * 2 + 1] = this.P[i * 4 + 2]; }
        const d = Math.hypot(this.P[40 * 4] - cam.pos[0], this.P[40 * 4 + 2] - cam.pos[2]);
        const h = this.probe.run(ot, pts, Math.max(d * angStep * 1.5, 2), sync);
        if (h) for (let k = 0; k < NP; k++) this.sea[k] = h[k];
      }
      const kk = 1 - Math.exp(-dt / 1.4);
      for (let k = 0; k < NP; k++) this.seaSmooth[k] += (this.sea[k] - this.seaSmooth[k]) * kk;
      this.step(dt, cam);
    }
    // the wake outlives the pass
    while (this.trail.length && this.clock - this.trail[this.trail.length - 1].t > 160) this.trail.pop();
    if (this.state === 'pass') this.emit(dt, cam);
    if (this.splash) {
      const wind = this.windVec || [0, 0];
      this.splash.update(dt, (x, z) => this.seaNear(x, z), wind, (kind, x, h, z, vy, size) => this.impact(kind, x, h, z, vy, size));
    }
    if (this.ring[3] > 0) { this.ring[2] += dt; if (this.ring[2] > 60) this.ring[3] = 0; }
    if (this.state === 'pass' && this.sc) {
      const yv = this.P[44 * 4 + 1], v = dt > 0 && this.prevY44 !== undefined ? (yv - this.prevY44) / dt : 0;
      this.prevY44 = yv;
      const target = clamp(Math.abs(v) / 3.5 + this.sc.lift * 0.45, 0, 1);
      this.surge = (this.surge || 0) + (target - (this.surge || 0)) * (1 - Math.exp(-dt / (target > (this.surge || 0) ? 0.6 : 5)));
    } else this.surge = 0;
    this.glowUpdate(dt);
    this.packWater();
    this.camPos = cam.pos;
    if (this.state === 'pass' && this.visible && sun[1] > -0.02 && !this.hidden) this.renderShadow(sun);
    else this.shadowOn = 0;
  }

  // the sea near the body (for spray landing), else 0
  seaNear(x, z) {
    if (this.state !== 'pass') return 0;
    // nearest probe node
    let best = 0, bd = 1e18;
    for (let k = 0; k < NP; k += 3) { const i = PROBE_NODE[k]; const d = (this.P[i * 4] - x) ** 2 + (this.P[i * 4 + 2] - z) ** 2; if (d < bd) { bd = d; best = k; } }
    return this.sea[best];
  }

  glowUpdate(dt) {
    // an occasional pulse up the back (tail to head over ~3 s); a strong one before the roar
    this.pulseT -= dt;
    if (this.pulseT <= 0 && this.state === 'pass') { this.pulseX = -140; this.pulseT = 25 + Math.random() * 35; this.pulseAmp = 1; }
    if (this.rise >= 8.2 && !this.risePulse) { this.risePulse = true; this.pulseX = -140; this.pulseAmp = 2.5; }
    if (this.rise < 0) this.risePulse = false;
    if (this.pulseX !== undefined) { this.pulseX += dt * 70; if (this.pulseX > 90) this.pulseX = undefined; }
    // the light source: the glow is a fixed radiance, so relative to moonlight it is ~4e5 times brighter
    const mag = this.lightMag ?? -26.74;
    const srcRel = Math.pow(10, -0.4 * (mag + 26.74));
    const steady = Math.min(1.0e-8 / srcRel, 0.006);
    const surge = this.pulseX !== undefined ? (this.pulseAmp || 1) : 0;
    this.glow = [steady, surge * (steady * 1.6 + 0.00035), this.pulseX ?? -1e4, 16];
  }

  // ---------------------------------------------------------------- water and spray
  packWater() {
    if (!this.trail.length && this.state !== 'pass') { this.wu = { uKjOn: 0 }; return; }
    const n = Math.min(this.trail.length + (this.state === 'pass' ? 1 : 0), KJT);
    const T = new Float32Array(KJT * 4), TB = new Float32Array(KJT * 4), Bd = new Float32Array(KJB * 4);
    const pts = this.state === 'pass' ? [{ x: this.src[0], z: this.src[1], t: this.clock, str: this.wakeStr, w: Math.max(this.src[4], 6 + 8 * this.surf), U: this.src[2] }, ...this.trail] : this.trail;
    let arc = 0, mnx = 1e9, mnz = 1e9, mxx = -1e9, mxz = -1e9;
    for (let i = 0; i < n; i++) {
      const p = pts[i];
      if (i > 0) arc += Math.hypot(p.x - pts[i - 1].x, p.z - pts[i - 1].z);
      T[i * 4] = p.x; T[i * 4 + 1] = p.z; T[i * 4 + 2] = arc; T[i * 4 + 3] = this.clock - p.t;
      TB[i * 4] = p.str; TB[i * 4 + 1] = p.w; TB[i * 4 + 2] = p.U;
      mnx = Math.min(mnx, p.x); mnz = Math.min(mnz, p.z); mxx = Math.max(mxx, p.x); mxz = Math.max(mxz, p.z);
    }
    if (this.state === 'pass') {
      for (let k = 0; k < KJB; k++) {
        const i = PROBE_NODE[k];
        Bd[k * 4] = this.P[i * 4]; Bd[k * 4 + 1] = this.P[i * 4 + 2]; Bd[k * 4 + 2] = this.water.w[k]; Bd[k * 4 + 3] = this.water.s[k];
        mnx = Math.min(mnx, this.P[i * 4]); mnz = Math.min(mnz, this.P[i * 4 + 2]); mxx = Math.max(mxx, this.P[i * 4]); mxz = Math.max(mxz, this.P[i * 4 + 2]);
      }
    }
    if (this.ring[3] > 0) { mnx = Math.min(mnx, this.ring[0] - 110); mnz = Math.min(mnz, this.ring[1] - 110); mxx = Math.max(mxx, this.ring[0] + 110); mxz = Math.max(mxz, this.ring[1] + 110); }
    // (the Kelvin wedge spreads at tan 19.5 deg = 0.35, but its far waves are faint)
    let wmax = 0;
    for (let i = 0; i < n; i++) wmax = Math.max(wmax, TB[i * 4 + 1] + 1.3 * Math.sqrt(T[i * 4 + 3] * 10 + 1));
    const m = Math.max(Math.min(0.3 * arc, 160), 2.6 * wmax) + 60;
    const src = this.state === 'pass' ? this.src : [pts[0]?.x ?? 0, pts[0]?.z ?? 0, pts[0]?.U ?? 1, 0];
    this.wu = {
      uKjOn: n > 1 || this.state === 'pass' ? 1 : 0, uKjN: n, uKjBox: [mnx - m, mnz - m, mxx + m, mxz + m],
      uKjTrail: T, uKjTrailB: TB, uKjBody: Bd,
      uKjSrc: [src[0], src[1], src[2], this.state === 'pass' ? src[3] * (this.front >= 0 && this.front <= 4 ? 1 : 0.35) : 0],
      uKjRing: this.ring, uKjDbg: this.dbg || [0, 0, 0, 0], uKjSurge: this.surge || 0, uKjT: this.clock % 1000,
    };
  }

  impact(kind, x, h, z, vy, size) {
    const sp = this.splash;
    if (kind === STREAM && Math.random() < 0.55) {
      sp.add(SPRAY, x, h + 0.4, z, (Math.random() - 0.5) * 3, Math.min(-vy * 0.25, 6) * Math.random(), (Math.random() - 0.5) * 3, 0.9 + size * 1.2, 0.7 + Math.random() * 0.8, 0.55);
      if (Math.random() < 0.12) sp.add(MIST, x, h + 2, z, (Math.random() - 0.5) * 2, 0.2, (Math.random() - 0.5) * 2, 18 + Math.random() * 18, 9 + Math.random() * 7, 0.05);
    }
  }

  emit(dt, cam) {
    if (this.seeking || !this.splash) return;
    const sp = this.splash, sc = this.sc, U = sc.speed;
    const R = Math.random;
    const tmp = [0, 0, 0], tmp2 = [0, 0, 0];
    // heading at the shoulders
    const i0 = 44 - 2, i1 = 44 + 2;
    let hx = this.P[i1 * 4] - this.P[i0 * 4], hz = this.P[i1 * 4 + 2] - this.P[i0 * 4 + 2];
    const hl = Math.hypot(hx, hz) || 1; hx /= hl; hz /= hl;
    const sx = -hz, sz = hx;
    const spd = sstep(2, 12, U);
    // 1) plates cutting the surface: a sheet of spray off each side of every plate in the water line
    for (const [fx, top, fz, S] of this.fins) {
      this.skin(fx, top, fz, tmp);
      this.skin(fx - 0.3 * S, top + 0.85 * S, fz, tmp2);
      const h = this.seaAtX(fx);
      if (!(tmp[1] < h + 0.5 && tmp2[1] > h)) continue;
      const t = clamp((h - tmp[1]) / Math.max(tmp2[1] - tmp[1], 0.1), 0, 1);
      const cx = tmp[0] + (tmp2[0] - tmp[0]) * t, cz = tmp[2] + (tmp2[2] - tmp[2]) * t;
      const rate = 24 * spd * spd * (S / 9) * (1 - 0.6 * t) * (0.3 + 0.7 * (1 - this.surf * 0.5));
      let nE = rate * dt; nE = Math.floor(nE) + (R() < nE % 1 ? 1 : 0);
      for (let e = 0; e < nE; e++) {
        const side = R() < 0.5 ? -1 : 1, up = (1.5 + 5 * R()) * Math.min(U / 11, 1.3);
        const lat = side * (1.5 + 5 * R()) * (U / 11), fwd = U * (0.25 + 0.35 * R());
        sp.add(R() < 0.4 ? STREAM : SPRAY, cx + side * sx * 0.8, h + 0.3, cz + side * sz * 0.8, hx * fwd + sx * lat, up, hz * fwd + sz * lat, (0.5 + 0.9 * R()) * (0.5 + S / 18), 1.0 + 1.2 * R(), 0.5);
      }
    }
    // 2) the head / chest pushing the sea: a bow wave breaking into a wall of spray
    if (this.front >= 0 && this.front <= 6 && this.water.w[this.front] > 1.8) {
      const k = this.front, i = PROBE_NODE[k], h = this.sea[k];
      const bx = this.P[i * 4] + hx * this.water.w[k] * 0.6, bz = this.P[i * 4 + 2] + hz * this.water.w[k] * 0.6;
      const wv = this.water.w[k];
      let nE = 220 * spd * spd * (0.5 + wv / 8) * dt * (sc.lift > 0.1 ? 0.3 : 1);
      nE = Math.floor(nE) + (R() < nE % 1 ? 1 : 0);
      for (let e = 0; e < nE; e++) {
        const side = R() < 0.5 ? -1 : 1, a = R();
        const ox = bx + sx * side * wv * a * 1.1 - hx * a * wv * 0.8, oz = bz + sz * side * wv * a * 1.1 - hz * a * wv * 0.8;
        const up = (5 + 11 * R()) * (U / 11), lat = side * (3 + 8 * R()) * (U / 11), fwd = U * (0.35 + 0.4 * R());
        sp.add(R() < 0.35 ? STREAM : SPRAY, ox, h + 0.5, oz, hx * fwd + sx * lat, up, hz * fwd + sz * lat, 0.9 + 1.6 * R(), 1.6 + 1.6 * R(), 0.5);
      }
      if (R() < 6 * spd * dt) sp.add(MIST, bx, h + 2, bz, hx * U * 0.3, 0.3, hz * U * 0.3, 14 + 10 * R(), 7 + 5 * R(), 0.08);
    }
    // 3) the flanks: spray and whitewater thrown off along the waterline
    for (let k = 0; k < NP; k++) {
      const wv = this.water.w[k];
      if (wv < 2) continue;
      const i = PROBE_NODE[k], h = this.sea[k];
      let nE = 9 * spd * dt * (wv / 6);
      nE = Math.floor(nE) + (R() < nE % 1 ? 1 : 0);
      for (let e = 0; e < nE; e++) {
        const side = R() < 0.5 ? -1 : 1;
        sp.add(SPRAY, this.P[i * 4] + sx * side * wv, h + 0.3, this.P[i * 4 + 2] + sz * side * wv, sx * side * (1 + 3 * R()) + hx * U * 0.2, 1.5 + 4 * R() * spd, sz * side * (1 + 3 * R()) + hz * U * 0.2, 1.0 + 1.6 * R(), 1 + R(), 0.6);
      }
    }
    // 4) rising out of the sea: water pours off everything that has just come up; what bursts up
    //    through the surface throws up a wall of water; what slaps back down splashes
    const ns = this.samples.length / 4, S4 = this.samples, Wp = this.sampleW, Pp = this.samplePrev;
    const active = sc.lift > 0.02 || this.rise >= 0;
    for (let k = 0; k < ns; k++) {
      this.skin(S4[k * 4], S4[k * 4 + 1], S4[k * 4 + 2], tmp);
      const h = this.seaAtX(S4[k * 4]);
      Pp[k * 3] = Wp[k * 3]; Pp[k * 3 + 1] = Wp[k * 3 + 1]; Pp[k * 3 + 2] = Wp[k * 3 + 2];
      Wp[k * 3] = tmp[0]; Wp[k * 3 + 1] = tmp[1]; Wp[k * 3 + 2] = tmp[2];
      const vy = (Wp[k * 3 + 1] - Pp[k * 3 + 1]) / Math.max(dt, 1e-3);
      if (tmp[1] < h) { S4[k * 4 + 3] = this.clock; }
      if (!active || dt <= 0) continue;
      const vx = (Wp[k * 3] - Pp[k * 3]) / dt, vz = (Wp[k * 3 + 2] - Pp[k * 3 + 2]) / dt;
      const wetAge = this.clock - S4[k * 4 + 3];
      const above = tmp[1] - h;
      // water sheeting off: thin falls from the flanks (normals sideways / down), and a few heavy
      // cascades from the ridge between the plates; they start just off the skin and fall clear of it
      if (above > 1.0 && wetAge < 16) {
        const ny = this.sampleN[k * 3 + 1];
        const ridge = ny > 0.8 && Math.abs(S4[k * 4 + 2]) < 3.5;
        if (ny < 0.5 || ridge) {
          const rate = (ridge ? 0.7 : 3.2) * Math.exp(-wetAge / 5) * (0.3 + sc.pour);
          if (R() < rate * dt) {
            const wn = this.skinDir(this.sampleN[k * 3], ny, this.sampleN[k * 3 + 2], tmp2);
            const o = ridge ? 0.3 : 0.7;
            sp.add(STREAM, tmp[0] + wn[0] * o, tmp[1] + wn[1] * o, tmp[2] + wn[2] * o,
              vx * 0.9 + wn[0] * 0.6, Math.min(vy, 4) * 0.8 - 1.0, vz * 0.9 + wn[2] * 0.6,
              ridge ? 1.6 + 1.4 * R() : 0.5 + 0.8 * R(), 5, ridge ? 0.6 : 0.3 + 0.25 * R());
          }
        }
      }
      if (Math.abs(above) < 3 && vy > 2.5 && R() < 0.45 * dt * vy) {
        // the sea lifted with the body: sheets that ride up a few metres and fall back (no clots in the
        // air), a little low spray at the base
        sp.add(STREAM, tmp[0] + (R() - 0.5) * 3, h + 0.3, tmp[2] + (R() - 0.5) * 3, vx + (R() - 0.5) * 3, Math.min(vy * (0.8 + 0.5 * R()), 7), vz + (R() - 0.5) * 3, 0.7 + 0.9 * R(), 2.5, 0.45);
        if (R() < 0.5) sp.add(SPRAY, tmp[0] + (R() - 0.5) * 5, h + 0.3, tmp[2] + (R() - 0.5) * 5, (R() - 0.5) * 4, 1 + 1.5 * R(), (R() - 0.5) * 4, 0.8 + 0.7 * R(), 0.9 + 0.6 * R(), 0.5);
        if (R() < 0.12) sp.add(MIST, tmp[0], h + 1, tmp[2], vx * 0.3, 0.2, vz * 0.3, 14 + 12 * R(), 9 + 6 * R(), 0.07);
      }
      if (Math.abs(above) < 3 && vy < -2.5 && R() < 0.6 * dt * -vy) {
        sp.add(STREAM, tmp[0] + (R() - 0.5) * 3, h + 0.3, tmp[2] + (R() - 0.5) * 3, (R() - 0.5) * 5, Math.min(-vy * (0.5 + 0.6 * R()), 8), (R() - 0.5) * 5, 0.8 + 1.0 * R(), 2.5, 0.5);
        if (R() < 0.6) sp.add(SPRAY, tmp[0] + (R() - 0.5) * 5, h + 0.3, tmp[2] + (R() - 0.5) * 5, (R() - 0.5) * 5, 1.5 + 2 * R(), (R() - 0.5) * 5, 0.9 + 0.8 * R(), 1 + 0.6 * R(), 0.5);
      }
    }
    const sg = this.surge || 0;
    if (sg > 0.08) {
      for (let k = 0; k < NP; k++) {
        const wv = this.water.w[k];
        if (wv < 2) continue;
        const i = PROBE_NODE[k], h = this.sea[k];
        let nE = 110 * sg * dt * Math.min(wv / 6, 1.5);
        nE = Math.floor(nE) + (R() < nE % 1 ? 1 : 0);
        for (let e = 0; e < nE; e++) {
          const side = R() < 0.5 ? -1 : 1, d = wv + (R() - 0.3) * 6;
          // tumbling whitewater hugging the flank: low, short-lived, lumpy, some sheets slumping back
          sp.add(R() < 0.3 ? STREAM : SPRAY, this.P[i * 4] + sx * side * d + hx * (R() - 0.5) * 14, h + 0.4 + sg * 2.5 * R(), this.P[i * 4 + 2] + sz * side * d + hz * (R() - 0.5) * 14,
            sx * side * (0.5 + 2.5 * R()), 1.0 + 4 * R() * sg, sz * side * (0.5 + 2.5 * R()), 0.7 + 1.3 * R(), 0.6 + 0.6 * R(), 0.6);
        }
      }
    }
    if (this.ringStart && this.rise > 2.5) {
      this.ringStart = false;
      this.skin(48, 0, 0, tmp);
      this.ring = [tmp[0], tmp[2], 0, 1];
    }
    if (this.rise > 19.5 && !this.ringBack) {
      this.ringBack = true;
      this.skin(48, 0, 0, tmp);
      this.ring = [tmp[0], tmp[2], 0, 0.8];
    }
    if (this.rise < 0) this.ringBack = false;
  }

  // ---------------------------------------------------------------- GPU
  renderShadow(sun) {
    const gl = this.gl;
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < NB; i += 3) { cx += this.P[i * 4]; cy += this.P[i * 4 + 1]; cz += this.P[i * 4 + 2]; }
    const c = NB / 3 > 0 ? Math.ceil(NB / 3) : 1;
    cx /= c; cy /= c; cz /= c;
    const L = sun, up = Math.abs(L[1]) < 0.99 ? [0, 1, 0] : [1, 0, 0];
    let X = [up[1] * L[2] - up[2] * L[1], up[2] * L[0] - up[0] * L[2], up[0] * L[1] - up[1] * L[0]];
    const xl = Math.hypot(...X); X = X.map((v) => v / xl);
    const Y = [L[1] * X[2] - L[2] * X[1], L[2] * X[0] - L[0] * X[2], L[0] * X[1] - L[1] * X[0]];
    const R = 150;
    this.sh = { uKjShC: [cx, cy + 8, cz, 1 / R], uKjShX: X, uKjShY: Y, uKjShZ: [L[0], L[1], L[2]] };
    this.shTarget.bind();
    gl.clearColor(-1e5, 0, 0, 1);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.disable(gl.CULL_FACE);
    this.shProg.use({ ...this.sh, uKjP: this.P, uKjQ: this.Q, uKjJaw: this.jaw });
    gl.bindVertexArray(this.meshLow.vao);
    gl.drawElements(gl.TRIANGLES, this.meshLow.count, this.meshLow.itype, 0);
    gl.bindVertexArray(null);
    gl.clearColor(0, 0, 0, 1);
    this.shadowOn = 1;
  }

  // the sea shader's uniforms (merged into the water's)
  waterUniforms() { return this.wu; }

  // draw into the bound HDR target, before the water (u: the water's uniforms)
  draw(u, pixelAngle) {
    if (!this.enabled || !this.ready || this.state !== 'pass' || !this.visible || this.hidden) return;
    const gl = this.gl;
    this.lightMag = u.uLightMag;
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
    gl.disable(gl.CULL_FACE);
    const sinceUp = this.rise >= 0 ? this.rise : 99;
    this.prog.use({
      ...u, ...(this.sh || {}), uKjShadow: this.shTex, uKjShOn: this.shadowOn || 0,
      uKjP: this.P, uKjQ: this.Q, uKjJaw: this.jaw, uKjSea: this.sea, uKjPixel: pixelAngle,
      uKjGlow: this.glow, uKjWater: [this.sc ? this.sc.pour : 0, clamp(Math.max(0.35 + 0.65 * sstep(3, 11, this.sc ? this.sc.speed : 0), (this.surge || 0) * 1.4), 0, 1.4), sinceUp],
    });
    // (beyond ~1.1 km a vertex of the fine mesh is well under a pixel)
    const cp = u.uCamPos, mid = 30 * 4;
    const far = Math.hypot(this.P[mid] - cp[0], this.P[mid + 2] - cp[2]) * pixelAngle > 0.9;
    const mesh = far ? this.meshLow : this.mesh;
    gl.bindVertexArray(mesh.vao);
    gl.drawElements(gl.TRIANGLES, mesh.count, mesh.itype, 0);
    gl.bindVertexArray(null);
  }

  // spray, after the water (u: the water's uniforms)
  drawSpray(u, pixelAngle, wind) {
    this.windVec = wind;
    if (!this.enabled || !this.splash || this.hidden) return;
    this.splash.draw({ ...u, uPixelAngle: pixelAngle });
  }

  // ---------------------------------------------------------------- tools
  // point of interest in the world: 'head', 'body', 'wake'
  target(what = 'head') {
    const t = [0, 0, 0];
    if (!this.ready || this.state !== 'pass') return null;
    if (what === 'head') return this.skin(64, 1, 0, t);
    if (what === 'wake' && this.trail.length > 4) { const p = this.trail[Math.min(6, this.trail.length - 1)]; return [p.x, 0, p.z]; }
    return this.skin(5, 8, 0, t);
  }
  // point a camera at it
  aim(cam, what = 'head', lift = 0) {
    const p = this.target(what);
    if (!p) return false;
    const dx = p[0] - cam.pos[0], dz = p[2] - cam.pos[2];
    cam.yaw = Math.atan2(dz, dx);
    cam.pitch = Math.atan2(p[1] + lift - cam.pos[1], Math.hypot(dx, dz)) - (cam.tilt || 0);
    return true;
  }
  debugInfo() {
    if (!this.ready) return { ready: false };
    const head = this.state === 'pass' ? this.skin(64, 1, 0, [0, 0, 0]) : null;
    const cp = this.camPos || [0, 0, 0];
    return {
      state: this.state, tau: +(this.tau || 0).toFixed(1), countdown: +(this.countdown || 0).toFixed(1),
      rise: +(this.rise ?? -1).toFixed(1), speed: this.sc ? +this.sc.speed.toFixed(1) : 0,
      head: head ? head.map((v) => +v.toFixed(1)) : null,
      dist: head ? Math.round(Math.hypot(head[0] - cp[0], head[2] - cp[2])) : null,
      front: this.front, surf: +(this.surf || 0).toFixed(2), trail: this.trail.length,
      particles: this.splash ? this.splash.n : 0, visible: this.visible, buildMs: this.buildMs, cpuMs: +(this.cpuMs || 0).toFixed(2),
    };
  }
}
