// Shaders for the boat, oars and rower. One lighting model for everything: the sun (or moon)
// through the atmosphere and the cloud-shadow map, shadowed by the hull walls and by capsules that
// stand in for the rower; sky, cloud and sea light from the reflection environment (cosine-weighted
// lookups), occluded by the same capsules; GGX highlights; then the sea's aerial perspective,
// crepuscular rays and rain/mist fog. Materials are procedural (no textures).
import { SKY_LOOKUP_GLSL } from '../sky/atmosphere.js';
import { HULL_GLSL, LIN, LOUT, ROWLOCK_X } from './hull.js';

export const NCAPS = 12;

const VS_COMMON = `
uniform mat4 uViewProj;
uniform vec3 uCamPos;
vec4 project(inout vec3 P) {
  // earth curvature, as the sea surface
  vec2 dr = P.xz - uCamPos.xz;
  P.y -= dot(dr, dr) / (2.0 * 6360000.0);
  return uViewProj * vec4(P, 1.0);
}`;

export const VS_PROP = VS_COMMON + `
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec2 aUv;
layout(location = 3) in vec2 aMat;
uniform mat4 uModel;
uniform mat3 uNrmM;
out vec3 vWorld;
out vec3 vN;
out vec2 vUv;
out vec2 vMat;
out vec3 vObj;
void main() {
  vec3 P = (uModel * vec4(aPos, 1.0)).xyz;
  vN = uNrmM * aNrm;
  vUv = aUv; vMat = aMat; vObj = aPos;
  gl_Position = project(P);
  vWorld = P;
}`;

// character: skinned body (SKIN defined) or rigid head / hands
export const VS_CHAR = VS_COMMON + `
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec3 aFold;
layout(location = 3) in vec3 aTan;
layout(location = 4) in vec4 aMatW;
#ifdef SKIN
layout(location = 5) in vec4 aBone;
layout(location = 6) in vec4 aWeight;
uniform vec4 uDQ[28];          // per bone: rotation quaternion, dual part
#endif
uniform mat4 uModel;
out vec3 vWorld;
out vec3 vN;
out vec3 vT;
out vec3 vFold;
out vec4 vMatW;
out vec3 vRest;
void main() {
  vec3 p = aPos, n = aNrm, t = aTan;
#ifdef SKIN
  // dual-quaternion blend (keeps the volume at bent joints)
  ivec4 bi = ivec4(aBone + 0.5);
  vec4 r0 = uDQ[2 * bi.x];
  vec4 r = vec4(0.0), d = vec4(0.0);
  for (int k = 0; k < 4; k++) {
    int b = bi[k];
    vec4 rk = uDQ[2 * b], dk = uDQ[2 * b + 1];
    float wk = aWeight[k] * (dot(rk, r0) < 0.0 ? -1.0 : 1.0);
    r += rk * wk; d += dk * wk;
  }
  float ln = length(r);
  r /= ln; d /= ln;
  vec3 tr = 2.0 * (r.w * d.xyz - d.w * r.xyz + cross(r.xyz, d.xyz));
  p = p + 2.0 * cross(r.xyz, cross(r.xyz, p) + r.w * p) + tr;
  n = n + 2.0 * cross(r.xyz, cross(r.xyz, n) + r.w * n);
  t = t + 2.0 * cross(r.xyz, cross(r.xyz, t) + r.w * t);
#endif
  vec3 P = (uModel * vec4(p, 1.0)).xyz;
  vN = mat3(uModel) * n;
  vT = mat3(uModel) * t;
  vFold = aFold; vMatW = aMatW; vRest = aPos;
  gl_Position = project(P);
  vWorld = P;
}`;

// ---------------------------------------------------------------- shared lighting
const LIGHT_GLSL = SKY_LOOKUP_GLSL + HULL_GLSL + `
uniform vec3 uCamPos;
uniform sampler2D uEnv;        // reflection environment (sky, clouds and sea; mips = blur)
uniform vec3 uFlash;           // lightning illuminance (relative to the light source)
uniform float uRain;
uniform float uTime;
uniform mat4 uBoatInv;         // world -> boat-local
uniform vec4 uCaps[${NCAPS * 2}];      // rower capsules (boat-local): (a, r), (b, 0)
uniform vec3 uWaterPlane;      // sea surface in boat-local: y = x + y x + z z
uniform float uSpray;          // spray / rain wetting of everything
const float PI = 3.14159265359;
vec2 equirectUv(vec3 d) { return vec2(atan(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI); }
vec3 envL(vec3 d, float lod) { return textureLod(uEnv, equirectUv(normalize(d)), lod).rgb; }
float bh(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float bn(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(bh(i), bh(i + vec2(1, 0)), f.x), mix(bh(i + vec2(0, 1)), bh(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p) { return 0.5 * bn(p) + 0.25 * bn(p * 2.03 + 1.7) + 0.125 * bn(p * 4.01 + 3.1) + 0.0625 * bn(p * 8.1 + 5.3); }
float bn3(vec3 p) { return bn(p.xy + p.z * vec2(17.3, 31.7)); }
float waterY(vec3 lp) { return uWaterPlane.x + uWaterPlane.y * lp.x + uWaterPlane.z * lp.z; }

// soft shadow of the rower's capsules along the light (boat-local)
float capShadow(vec3 p, vec3 l) {
  float s = 1.0;
  for (int i = 0; i < ${NCAPS}; i++) {
    vec4 A = uCaps[2 * i], Bc = uCaps[2 * i + 1];
    if (A.w <= 0.0) continue;
    // closest approach between the ray p + t l (t > 0) and the segment A..B
    vec3 ba = Bc.xyz - A.xyz, oa = p - A.xyz;
    float a = dot(ba, ba), b = dot(ba, l), c = dot(ba, oa), e = dot(l, oa);
    float den = a - b * b;
    float h = den > 1e-6 ? clamp((c - b * e) / den, 0.0, 1.0) : 0.0;
    float t = max(dot(A.xyz + ba * h - p, l), 0.0);
    h = clamp(dot(p + l * t - A.xyz, ba) / a, 0.0, 1.0);
    vec3 q = A.xyz + ba * h;
    t = dot(q - p, l);
    if (t <= 0.0) continue;
    float d = length(p + l * t - q);
    float pen = 0.012 + t * 0.03;
    s *= smoothstep(A.w - pen, A.w + pen, d);
  }
  return s;
}
// ambient occlusion by the rower's capsules
float capAO(vec3 p, vec3 n) {
  float o = 1.0;
  for (int i = 0; i < ${NCAPS}; i++) {
    vec4 A = uCaps[2 * i], Bc = uCaps[2 * i + 1];
    if (A.w <= 0.0) continue;
    vec3 ba = Bc.xyz - A.xyz;
    float h = clamp(dot(p - A.xyz, ba) / dot(ba, ba), 0.0, 1.0);
    vec3 d = A.xyz + ba * h - p;
    float l = length(d);
    float r = A.w * 1.35;
    o *= 1.0 - clamp(dot(n, d / l) * 0.5 + 0.5, 0.0, 1.0) * clamp(r * r / (l * l), 0.0, 1.0) * 0.85;
  }
  return o;
}

// cosine-weighted irradiance from the environment: the normal plus a ring of eight directions 55 deg
// off it (a 5-tap estimate round the normal put ~70% of a wall's weight on the bright horizon band:
// walls came out 1.6x too bright at night and 4x at golden hour; method after ow-boat/src/ship)
vec3 ambientE(vec3 N) {
  vec3 T1 = normalize(cross(N, abs(N.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  vec3 T2 = cross(N, T1);
  vec3 E = envL(N, 5.0);
  for (int k = 0; k < 8; k++) {
    float a = float(k) * 0.7853982;
    E += envL(N * 0.5736 + (T1 * cos(a) + T2 * sin(a)) * 0.8192, 5.0) * 0.5736;
  }
  return E * PI / (1.0 + 8.0 * 0.5736);
}

struct Surf {
  vec3 alb;     // diffuse albedo
  vec3 F0;      // specular reflectance at normal incidence
  float rough;
  float sss;    // skin: light wrapped round the terminator, reddened
  float sheen;  // cloth: grazing back-scatter
  float ao;
};

vec3 shade(Surf s, vec3 P, vec3 N, vec3 V, float dist, vec3 lp, vec3 lN) {
  vec3 L = uSunDir;
  vec3 sunE = sunIlluminance();
  vec3 Lloc = mat3(uBoatInv) * L;
  // sun visibility: cloud-shadow map (as the sea), the hull's own walls, the rower
  vec2 suv = (P.xz - L.xz * (max(P.y, 0.0) / max(L.y, 0.02)) - uShaftRect.xy) * uShaftRect.zw;
  float sunVis = mix(uHazeSun, texture(uShaftShadow, suv).r, smoothstep(0.0, 0.04, min(min(suv.x, suv.y), min(1.0 - suv.x, 1.0 - suv.y))));
  sunVis *= 1.0 - hullShadow(lp, Lloc);
  sunVis *= capShadow(lp + lN * 0.025, Lloc);
  // diffuse light from sky, clouds and sea: cosine-weighted lookups of the environment
  vec3 Eamb = ambientE(N);
  // occlusion: inside the hull the walls hide most of the sky; the rower's body
  float u = hullU(lp.x);
  float ihw = innerHalfWidth(clamp(u, 0.0, 1.0), lp.y);
  bool facingIn = lN.z * sign(lp.z) < 0.0 || lN.y > 0.6;
  float inside = (u > 0.0 && u < 1.0 && lp.y < hSheer(u) && (abs(lp.z) < ihw - 0.004 || (abs(lp.z) < ihw + 0.02 && facingIn))) ? 1.0 : 0.0;
  float ao = s.ao * capAO(lp, lN) * mix(1.0, mix(0.35, 1.0, smoothstep(-0.2, hSheer(clamp(u, 0.0, 1.0)), lp.y)), inside);
  float nl = dot(N, L), nv = max(dot(N, V), 1e-3);
  vec3 H = normalize(L + V);
  float nh = max(dot(N, H), 0.0);
  float a2 = max(s.rough * s.rough * s.rough * s.rough, 1e-4);
  float D = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
  vec3 Fh = s.F0 + (1.0 - s.F0) * pow(1.0 - max(dot(V, H), 0.0), 5.0);
  vec3 Fv = s.F0 + (1.0 - s.F0) * pow(1.0 - nv, 5.0) / (1.0 + 4.0 * s.rough);
  float vis = 1.0 / (4.0 * max(max(nl, 0.0), nv) );
  vec3 Esun = sunE * sunVis;
  // skin: wrapped diffuse (light scattered under the skin), redder round the terminator
  float wrapL = max((nl + 0.45) / 1.45, 0.0);
  vec3 diffSun = mix(vec3(max(nl, 0.0)), wrapL * vec3(1.0, 0.62, 0.5) * mix(1.0, 1.0, 0.0), s.sss);
  vec3 Ef = uFlash * (0.35 + 0.65 * max(dot(N, uFlashDir), 0.0));
  vec3 col = s.alb / PI * (Esun * diffSun + Eamb * ao + Ef * ao) * (1.0 - Fv);
  // cloth sheen
  col += s.sheen * s.alb * (Esun * max(nl, 0.0) + Eamb * ao * 0.3) * pow(1.0 - nv, 3.0) * 0.6 / PI;
  col += Esun * max(nl, 0.0) * min(D, 400.0) * Fh * vis;
  vec3 R = reflect(-V, N);
  // reflections from below the horizon on the inside of the hull see wood, not sea
  col += envL(R, mix(1.0, 7.0, s.rough)) * Fv * ao;
  // aerial perspective, light shafts and rain / mist, exactly as the sea surface
  vec3 aL, aT;
  aerial(-V, dist * 0.001, aL, aT);
  col = col * aT + aL;
  col = withShafts(col, uCamPos, -V, dist);
  return applyFog(col, -V, dist);
}

// orient the interpolated normal to the side we see (thin parts are seen from both sides)
vec3 faceNormal(vec3 P, vec3 n, vec3 V) {
  vec3 ng = normalize(cross(dFdx(P), dFdy(P)));
  if (dot(ng, V) < 0.0) ng = -ng;
  vec3 N = normalize(n);
  if (dot(N, ng) < 0.0) N = -N;
  return N;
}

// wood: straight grain along g.x, growth rings across (g.y), fibres
vec3 woodAlb(vec2 g, float seed, vec3 base, float contrast) {
  float n = fbm(vec2(g.x * 0.7 + seed * 3.1, g.y * 9.0 + seed * 7.3));
  float rings = fract(g.y * 42.0 + n * 3.2 + seed * 0.37);
  float late = smoothstep(0.5, 0.92, rings) * (1.0 - smoothstep(0.92, 1.0, rings));
  float fib = bn(vec2(g.x * 2.5 + seed, g.y * 700.0));
  float streak = fbm(vec2(g.x * 0.25 + seed * 1.7, g.y * 30.0));
  return base * mix(1.12, 0.7, late * contrast) * (0.9 + 0.2 * fib) * (0.85 + 0.3 * streak);
}
// how wet a point is: below the sea plus a splash band, inside below the bilge, rain and spray
float wetness(vec3 lp, float outside, float n) {
  float wl = waterY(lp);
  float w = outside * (1.0 - smoothstep(wl + 0.025, wl + 0.11 + 0.06 * n, lp.y));
  return clamp(max(w, uRain * 1.4 + uSpray), 0.0, 1.0);
}
void applyWet(inout Surf s, float w, float porous) {
  s.alb *= mix(1.0, mix(0.82, 0.55, porous), w);
  s.rough = mix(s.rough, 0.12, w * 0.85);
  s.F0 = mix(s.F0, vec3(0.03), w * (1.0 - step(0.2, s.F0.r)));
}
`;

// ---------------------------------------------------------------- props: hull, fittings, oars, rowlocks
export const FS_PROP = LIGHT_GLSL + `
in vec3 vWorld;
in vec3 vN;
in vec2 vUv;
in vec2 vMat;
in vec3 vObj;
out vec4 oColor;
uniform vec2 uOar;             // x: 1 for the oars, y: blade wetness
const vec3 TOPSIDE = vec3(0.60, 0.58, 0.50), SHEER = vec3(0.028, 0.085, 0.06), ANTIFOUL = vec3(0.11, 0.035, 0.025);
const vec3 HONEY = vec3(0.42, 0.24, 0.10), OAK = vec3(0.30, 0.25, 0.19), BARE = vec3(0.24, 0.215, 0.18);
void main() {
  vec3 P = vWorld;
  vec3 toCam = uCamPos - P;
  float dist = length(toCam);
  vec3 V = toCam / dist;
  vec3 N = faceNormal(P, vN, V);
  vec3 lp = (uBoatInv * vec4(P, 1.0)).xyz;
  vec3 lN = mat3(uBoatInv) * N;
  int id = int(vMat.x + 0.5);
  float seed = vMat.y;
  Surf s = Surf(vec3(0.5), vec3(0.04), 0.5, 0.0, 0.0, 1.0);
  float n1 = fbm(lp.xy * vec2(3.0, 9.0) + seed);
  float porous = 1.0;
  float outside = 0.0;
  if (id == 0 && seed > 97.5) id = 1;          // transom: varnished
  if (id == 0) {
    // painted planking (outside): antifouling below the boot-top, cream topsides, green sheer strake
    outside = 1.0;
    float strake = mod(seed, 10.0);
    vec3 paint = strake > 5.5 && seed < 49.5 ? SHEER : TOPSIDE;
    paint *= 0.93 + 0.1 * fract(sin(seed * 12.9898) * 43758.5453);   // each plank weathered a little differently
    float wlLine = 0.035 + 0.004 * sin(lp.x * 1.3);
    if (lp.y < wlLine) paint = ANTIFOUL;
    // grime: streaks running down from the gunwale, scum above the waterline
    float streak = fbm(vec2(lp.x * 9.0 + seed, lp.y * 1.2));
    paint *= mix(1.0, 0.82, smoothstep(0.45, 0.8, streak) * (paint == TOPSIDE ? 1.0 : 0.3));
    float scum = smoothstep(wlLine, wlLine + 0.01, lp.y) * (1.0 - smoothstep(wlLine + 0.04, wlLine + 0.11, lp.y));
    paint = mix(paint, vec3(0.16, 0.15, 0.085), scum * (0.45 + 0.35 * fbm(lp.xz * 6.0)));
    // rust bleeding down from the rowlock sockets
    float rust = exp(-pow((lp.x - ${ROWLOCK_X.toFixed(3)}) / 0.025, 2.0)) * smoothstep(0.1, 0.38, lp.y) * (0.6 + 0.4 * bn(vec2(lp.x * 80.0, lp.y * 4.0)));
    paint = mix(paint, vec3(0.20, 0.075, 0.03), rust * 0.7 * (paint == TOPSIDE ? 1.0 : 0.4));
    // chipped paint: worst on the plank edges (lands) and the sheer, showing grey weathered wood
    float edge = vUv.y < 0.0 ? 1.0 : 1.0 - smoothstep(0.0, 0.16, vUv.y);
    edge = max(edge, smoothstep(0.85, 1.0, vUv.y) * step(5.5, strake));
    float chipN = fbm(vec2(lp.x * 30.0 + seed * 3.0, lp.y * 70.0 + lp.z * 25.0));
    float th = 0.8 - 0.22 * edge;
    float chip = smoothstep(th, th + 0.02, chipN) * step(wlLine + 0.02, lp.y);
    vec3 wood = woodAlb(vec2(lp.x, vUv.y * 0.12 + seed * 0.21), seed, BARE, 0.6);
    s.alb = mix(paint, wood, chip);
    // a dark rim round each chip
    s.alb *= 1.0 - 0.3 * smoothstep(th - 0.05, th, chipN) * (1.0 - chip);
    s.rough = mix(mix(0.42, 0.62, smoothstep(0.3, 0.8, streak)), 0.75, chip);
    s.rough = lp.y < wlLine ? 0.8 : s.rough;
    // where the sea meets the hull: a ragged band of froth and a film of water climbing the paint
    float wlv = waterY(lp);
    float frothN = fbm(vec2(lp.x * 22.0 + uTime * 0.7 * sign(lp.z), lp.y * 50.0));
    float froth = exp(-pow((lp.y - wlv - 0.004) / 0.013, 2.0)) * smoothstep(0.35, 0.65, frothN);
    s.alb = mix(s.alb, vec3(0.72, 0.74, 0.74), froth * 0.85);
    s.F0 = vec3(0.04);
    porous = chip;
  } else if (id == 1) {
    // varnished wood inside: honey planking, worn and dirty in the bottom
    vec2 g = vUv.y > 1.5 || vUv.y < -0.5 ? vec2(lp.x, lp.y * 0.6 + seed) : vec2(vUv.x, vUv.y * 0.12 + seed * 0.13);
    if (seed > 97.5) g = vec2(lp.z, lp.y * 0.9);
    s.alb = woodAlb(g, seed, HONEY, 1.0);
    float scuff = (1.0 - smoothstep(-0.2, 0.05, lp.y)) * smoothstep(0.4, 0.75, fbm(lp.xz * 7.0));
    s.alb = mix(s.alb, vec3(0.2, 0.17, 0.13), scuff * 0.6);
    s.alb *= mix(0.6, 1.0, smoothstep(-0.2, 0.0, lp.y));
    s.rough = mix(0.28, 0.6, scuff + 0.4 * fbm(lp.xy * 11.0));
    s.F0 = vec3(0.045);
    porous = 0.4 + 0.6 * scuff;
  } else if (id == 2) {
    // weathered oak: thwarts, capping, duckboards; seats polished pale where sat on
    vec2 g = vec2(vUv.x, vUv.y * 0.05 + seed * 0.11);
    s.alb = woodAlb(g, seed, OAK, 0.8);
    float sat = (1.0 - smoothstep(0.0, 0.14, length(lp.xz - vec2(0.08, 0.0)))) * step(0.12, lp.y) * step(lp.y, 0.16);
    s.alb = mix(s.alb, s.alb * vec3(1.15, 1.08, 1.0), sat);
    s.alb = mix(s.alb, vec3(0.23, 0.21, 0.19), 0.35 * smoothstep(0.5, 0.8, fbm(lp.xz * 4.0 + seed)));
    s.rough = 0.62 - 0.2 * sat;
  } else if (id == 3) {
    // bronze, with verdigris where it is not handled
    float pat = smoothstep(0.45, 0.75, fbm(vObj.xy * 90.0 + seed));
    s.alb = mix(vec3(0.0), vec3(0.11, 0.2, 0.16), pat);
    s.F0 = mix(vec3(0.62, 0.42, 0.24), vec3(0.04), pat);
    s.rough = mix(0.32, 0.7, pat);
    porous = 0.0;
  } else if (id == 4) {
    // oar: spruce loom, hand-polished dark grip, varnished blade with a painted tip
    float sAl = vUv.x;
    int region = int(seed + 0.5);
    vec2 g = vec2(sAl, vUv.y * 0.05 + 0.3);
    s.alb = woodAlb(g, 4.0, vec3(0.50, 0.34, 0.17), 0.7);
    s.rough = 0.32;
    s.F0 = vec3(0.045);
    if (region == 0) { s.alb *= vec3(0.55, 0.5, 0.45); s.rough = 0.38 - 0.1 * bn(vec2(sAl * 40.0, vUv.y * 9.0)); }
    else if (region == 2) { s.alb = vec3(0.13, 0.075, 0.042) * (0.8 + 0.4 * fbm(vec2(sAl * 30.0, vUv.y * 20.0))); s.rough = 0.62; }
    if (sAl > ${(LOUT - 0.11).toFixed(3)}) { s.alb = SHEER * 1.1; s.rough = 0.4; }
    // the outboard end is wet after each stroke
    float wb = uOar.y * smoothstep(0.75, 1.15, sAl);
    applyWet(s, wb, 0.5);
    porous = 0.3;
  } else if (id == 5) {
    s.alb = vec3(0.12, 0.07, 0.04) * (0.8 + 0.4 * fbm(vObj.xy * 200.0));
    s.rough = 0.6;
  }
  if (id != 4) applyWet(s, wetness(lp, outside, n1), porous);
  oColor = vec4(shade(s, P, N, V, dist, lp, lN), 1.0);
}`;

// ---------------------------------------------------------------- the rower
export const FS_CHAR = LIGHT_GLSL + `
in vec3 vWorld;
in vec3 vN;
in vec3 vT;
in vec3 vFold;
in vec4 vMatW;
in vec3 vRest;
out vec4 oColor;
uniform int uPart;             // 0 body (skin, jacket, trousers, boots), 1 head (skin, hat, eye), 2 hand
const vec3 OILSKIN = vec3(0.50, 0.335, 0.045), SERGE = vec3(0.03, 0.037, 0.058), SKINC = vec3(0.40, 0.22, 0.145);
// cloth folds: compression rings round a limb, broken and wavering
float foldH(vec2 f) {
  // soft ridges ~7 cm apart that wander round the limb and die out, plus a little crumple
  float w = f.x * 70.0 + 7.0 * bn(vec2(f.x * 3.0, f.y * 0.8)) + 1.2 * f.y + 1.6 * sin(f.y * 1.0 + f.x * 13.0);
  float r = 0.5 + 0.5 * sin(w);
  r *= r;
  float big = smoothstep(0.25, 0.75, bn(vec2(f.x * 10.0 + 3.0, f.y * 0.9)));
  return r * big + 0.35 * bn(vec2(f.x * 30.0, f.y * 3.0));
}
// torso: a few broad vertical drapes and soft creases round the waist
float drapeH(vec2 f) {
  float v = 0.5 + 0.5 * sin(f.y * 2.5 + 2.0 * bn(vec2(f.x * 5.0, f.y * 0.7)));
  float w = 0.5 + 0.5 * sin(f.x * 50.0 + 2.5 * bn(vec2(f.y * 1.5, f.x * 4.0)));
  return v * v * 0.7 + w * w * 0.6 * (1.0 - smoothstep(0.06, 0.22, f.x)) + 0.25 * bn(vec2(f.x * 20.0, f.y * 3.0));
}
float clothH(vec2 f, bool torso) { return torso ? drapeH(f) : foldH(f); }
void main() {
  vec3 P = vWorld;
  vec3 toCam = uCamPos - P;
  float dist = length(toCam);
  vec3 V = toCam / dist;
  vec3 N = faceNormal(P, vN, V);
  vec3 lp = (uBoatInv * vec4(P, 1.0)).xyz;
  // material: the largest weight
  vec4 w = vMatW;
  int m = 0;
  float best = w.x;
  if (w.y > best) { best = w.y; m = 1; }
  if (w.z > best) { best = w.z; m = 2; }
  if (w.w > best) { m = 3; }
  int kind = uPart == 1 ? (m == 1 ? 1 : m == 2 ? 4 : 0) : uPart == 2 ? 0 : m;   // 0 skin 1 oilskin 2 serge 3 rubber 4 eye
  Surf s = Surf(SKINC, vec3(0.028), 0.5, 0.0, 0.0, 1.0);
  float wetBias = 0.0;
  if (kind == 1) {
    // oilskin: glossy, creased, dirty in the folds
    bool torso = vFold.z > 1.5;
    float fs = torso ? vFold.z - 2.0 : vFold.z;
    vec2 f = vFold.xy;
    float e = 0.003, ea = 0.06;
    float h0 = clothH(f, torso), hx = clothH(f + vec2(e, 0.0), torso), hy = clothH(f + vec2(0.0, ea), torso);
    vec3 T = normalize(vT - N * dot(vT, N));
    vec3 Bt = cross(N, T);
    float amp = (torso ? 0.0024 : 0.0038) * fs;
    float rad = torso ? 0.15 : 0.055;
    N = normalize(N - amp * ((hx - h0) / e * T + (hy - h0) / (ea * rad) * Bt));
    // dirt and wear collect in the creases; the tops of folds are scuffed paler
    s.alb = OILSKIN * (0.85 + 0.2 * fbm(vRest.xy * 30.0 + vRest.z * 7.0));
    s.alb *= mix(1.04, 0.8, smoothstep(0.3, 1.0, h0) * fs);
    s.alb = mix(s.alb, vec3(0.22, 0.18, 0.09), 0.3 * smoothstep(0.55, 0.85, fbm(vRest.xz * 14.0 + vRest.y * 5.0)));
    s.rough = 0.36 + 0.2 * h0 * fs + 0.15 * fbm(vRest.xy * 50.0);
    if (uPart == 1) {
      // sou'wester: stitched crown panels, a stitched brim edge, matt and scuffed
      vec3 r = vRest;
      float ang = atan(r.z, r.x + 0.01);
      float seam = (1.0 - smoothstep(0.0, 0.035, abs(fract(ang / 1.0472 + 0.5) - 0.5))) * step(0.1, r.y);
      float rho = length(vec2(r.x + 0.01, r.z));
      float stitch = smoothstep(0.004, 0.0, abs(rho - 0.125 - 0.06 * smoothstep(0.1, -1.0, cos(ang)) * 0.0) - 0.002) * step(r.y, 0.1);
      s.alb *= 1.0 - 0.35 * max(seam, stitch);
      s.rough = 0.48 + 0.15 * fbm(r.xz * 60.0);
    }
    s.F0 = vec3(0.045);
    s.sheen = 0.0;
    wetBias = 0.25;
  } else if (kind == 2) {
    float fs = vFold.z;
    vec2 f = vFold.xy * vec2(0.8, 1.0);
    float e = 0.004;
    float h0 = foldH(f), hx = foldH(f + vec2(e, 0.0));
    vec3 T = normalize(vT - N * dot(vT, N));
    N = normalize(N - 0.003 * fs * (hx - h0) / e * T);
    s.alb = SERGE * (0.85 + 0.3 * bn(vRest.xy * 400.0)) * (0.9 + 0.2 * fbm(vRest.xz * 20.0));
    s.rough = 0.88;
    s.sheen = 1.0;
  } else if (kind == 3) {
    s.alb = vec3(0.022, 0.024, 0.026) * (0.9 + 0.3 * fbm(vRest.xy * 60.0));
    s.rough = 0.42;
    s.F0 = vec3(0.04);
    wetBias = 0.2;
  } else if (kind == 4) {
    // eye: wet cornea over a brown iris
    vec3 c = vec3(0.079, 0.086, sign(vRest.z) * 0.033);
    vec3 d = normalize(vRest - c);
    float iris = smoothstep(0.86, 0.9, d.x), pupil = smoothstep(0.955, 0.965, d.x);
    s.alb = mix(vec3(0.55, 0.5, 0.45), mix(vec3(0.12, 0.07, 0.03), vec3(0.01), pupil), iris);
    s.rough = 0.06; s.F0 = vec3(0.03);
  } else {
    // weathered skin: wind-burnt cheeks, nose and ears, stubble on the jaw, darker brows
    s.alb = SKINC * (0.92 + 0.16 * fbm(vRest.xy * 60.0 + vRest.z * 20.0));
    s.sss = 1.0;
    s.rough = 0.45;
    if (uPart == 1) {
      vec3 r = vRest;
      float cheek = exp(-pow(length((r - vec3(0.07, 0.065, sign(r.z) * 0.045)) / vec3(0.025, 0.02, 0.025)), 2.0));
      float nose = exp(-pow(length((r - vec3(0.115, 0.055, 0.0)) / vec3(0.015, 0.02, 0.015)), 2.0));
      float ear = smoothstep(0.065, 0.075, abs(r.z)) * step(r.x, 0.03) * step(0.04, r.y);
      s.alb = mix(s.alb, s.alb * vec3(1.25, 0.82, 0.78), clamp(cheek * 0.8 + nose * 0.7 + ear * 0.6, 0.0, 1.0));
      // stubble: jaw, chin and upper lip (below the cheekbones, in front of the ears)
      float jaw = smoothstep(0.06, 0.03, r.y) * smoothstep(0.0, 0.03, r.x + 0.01) * (1.0 - smoothstep(0.055, 0.075, abs(r.z) + max(r.y - 0.03, 0.0)));
      float lip = smoothstep(0.004, 0.0, abs(r.y - 0.021) - 0.006) * step(0.085, r.x) * step(abs(r.z), 0.022);
      float stub = clamp(jaw, 0.0, 1.0) * (1.0 - lip) * (0.6 + 0.4 * bh(floor(r.xy * 2500.0)));
      s.alb = mix(s.alb, s.alb * vec3(0.55, 0.5, 0.48), stub * 0.6);
      s.alb = mix(s.alb, vec3(0.3, 0.1, 0.08), lip * 0.5);
      s.rough = mix(s.rough, 0.7, stub);
      float brow = smoothstep(0.006, 0.0, abs(r.y - 0.104) - 0.004) * smoothstep(0.012, 0.018, abs(r.z)) * (1.0 - smoothstep(0.045, 0.055, abs(r.z))) * step(0.075, r.x);
      s.alb = mix(s.alb, vec3(0.05, 0.035, 0.025), brow * 0.85);
      // hair at the back of the neck under the hat
      float hair = step(r.x, -0.02) * smoothstep(0.03, 0.06, r.y) * step(r.y, 0.11);
      s.alb = mix(s.alb, vec3(0.07, 0.055, 0.045), hair * 0.9);
      s.rough = mix(s.rough, 0.75, hair);
    } else if (uPart == 2) {
      float kn = smoothstep(0.004, 0.012, vRest.x) * smoothstep(0.07, 0.09, vRest.y);
      s.alb = mix(s.alb, s.alb * vec3(1.2, 0.85, 0.8), kn * 0.6);
    }
  }
  float wv = clamp(uRain * 1.4 + uSpray + wetBias * uRain, 0.0, 1.0);
  if (kind != 4) applyWet(s, wv, kind == 2 ? 1.0 : 0.15);
  vec3 lN = mat3(uBoatInv) * N;
  oColor = vec4(shade(s, P, N, V, dist, lp, lN), 1.0);
}`;

// ---------------------------------------------------------------- bilge water
export const VS_BILGE = VS_COMMON + `
layout(location = 0) in vec3 aPos;
uniform mat4 uModel;
uniform vec3 uBilge;           // level plane in boat-local: y = x + y x + z z
out vec3 vWorld;
out vec3 vLocal;
void main() {
  vec3 p = vec3(aPos.x, uBilge.x + uBilge.y * aPos.x + uBilge.z * aPos.z, aPos.z);
  vLocal = p;
  vec3 P = (uModel * vec4(p, 1.0)).xyz;
  gl_Position = project(P);
  vWorld = P;
}`;
export const FS_BILGE = LIGHT_GLSL + `
in vec3 vWorld;
in vec3 vLocal;
out vec4 oColor;
void main() {
  vec3 lp = vLocal;
  float u = hullU(lp.x);
  if (u < 0.03 || u > 0.97 || abs(lp.z) > innerHalfWidth(u, lp.y) - 0.004) discard;
  vec3 P = vWorld;
  vec3 toCam = uCamPos - P;
  float dist = length(toCam);
  vec3 V = toCam / dist;
  // slop: small waves sloshing in the bilge, rain rings
  vec2 q = lp.xz;
  vec2 sl = vec2(0.0);
  for (int k = 0; k < 4; k++) {
    float fk = float(k);
    vec2 dir = vec2(cos(fk * 2.1 + 0.3), sin(fk * 2.1 + 0.3));
    float kk = 18.0 + fk * 9.0;
    sl += dir * kk * 0.0006 * cos(dot(q, dir) * kk - uTime * (3.0 + fk));
  }
  sl += (vec2(bn(q * 30.0 + uTime * 3.0), bn(q * 30.0 - uTime * 3.0 + 5.0)) - 0.5) * 0.12 * uRain;
  vec3 N = normalize(vec3(-sl.x, 1.0, -sl.y));
  N = normalize(mat3(inverse(uBoatInv)) * N);
  N = normalize(N + vec3(0.0, 1.0, 0.0) * 2.0);  // (the bilge stays level in the world)
  float nv = max(dot(N, V), 0.0);
  float F = 0.02 + 0.98 * pow(1.0 - nv, 5.0);
  vec3 R = reflect(-V, N);
  float sh = 1.0 - hullShadow(lp, mat3(uBoatInv) * uSunDir);
  sh *= capShadow(lp + vec3(0.0, 0.02, 0.0), mat3(uBoatInv) * uSunDir);
  // reflected sky is mostly the hull's own side: dim it towards the walls
  float wall = smoothstep(0.0, 0.25, innerHalfWidth(u, lp.y) - abs(lp.z));
  vec3 refl = envL(R, 1.0) * mix(0.25, 1.0, wall * smoothstep(0.0, 0.5, R.y));
  vec3 H = normalize(uSunDir + V);
  float spec = pow(max(dot(N, H), 0.0), 900.0) * 30.0;
  vec3 col = F * refl + sunIlluminance() * sh * spec * F + (1.0 - F) * vec3(0.03, 0.022, 0.012) * (skyIrradiance() + sunIlluminance() * sh * max(uSunDir.y, 0.0)) / PI;
  float alpha = F + (1.0 - F) * 0.42;
  vec3 aL, aT;
  aerial(-V, dist * 0.001, aL, aT);
  col = col * aT + aL * alpha;
  col = withShafts(col, uCamPos, -V, dist);
  col = applyFog(col, -V, dist);
  oColor = vec4(col, alpha);
}`;
