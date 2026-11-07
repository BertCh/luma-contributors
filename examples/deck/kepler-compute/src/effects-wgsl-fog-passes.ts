// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL ports of Kepler's distance fog and surface fog post-processing passes
 * (`@kepler.gl/effects` shader-passes/distance-fog.js and surface-fog.js).
 *
 * The math is the Kepler math: the depth buffer is linearized with the camera planes, basemap
 * pixels (depth 1) intersect a ray with the z = 0 ground plane, and fog is composited over a
 * premultiplied scene color. Deck's projection matrices keep GL clip conventions, so a stored
 * depth `d` in [0, 1] still reconstructs NDC z as `d * 2 - 1` through `pixelUnprojectionMatrix`.
 * Texture coordinates are top-left origin on WebGPU, so the Kepler `1 - coordinate.y` flip is
 * dropped.
 */

import type {Texture} from '@luma.gl/core';
import type {ShaderPass} from '@luma.gl/shaders';
import type {NumberArray16} from '@math.gl/core';

/** Camera data both fog passes need, read from a Deck viewport. */
export type FogCameraUniforms = {
  /** Column-major `viewport.pixelUnprojectionMatrix` (pixel + NDC depth to common space). */
  inverseMatrix: Readonly<NumberArray16>;
  viewportSize: [number, number];
  nearPlane: number;
  farPlane: number;
  /** Common-space units per metre along z (surface fog heights are metres). */
  unitsPerMeter: number;
};

export type FogViewportLike = {
  width: number;
  height: number;
  projectionMatrix?: ArrayLike<number>;
  pixelUnprojectionMatrix?: ArrayLike<number>;
  distanceScales?: {unitsPerMeter?: ArrayLike<number> | number};
};

const IDENTITY_MATRIX: NumberArray16 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Mirrors the viewport reads of Kepler's `DeckDistanceFogEffect` and `DeckSurfaceFogEffect`. */
export function getFogCameraUniforms(viewport: FogViewportLike | undefined): FogCameraUniforms {
  const result: FogCameraUniforms = {
    inverseMatrix: IDENTITY_MATRIX,
    viewportSize: [1, 1],
    nearPlane: 0.1,
    farPlane: 1000,
    unitsPerMeter: 1
  };
  if (!viewport) return result;
  result.viewportSize = [viewport.width, viewport.height];
  const projection = viewport.projectionMatrix;
  if (projection) {
    const nearPlane = projection[14] / (projection[10] - 1);
    const farPlane = projection[14] / (projection[10] + 1);
    if (
      Number.isFinite(nearPlane) &&
      Number.isFinite(farPlane) &&
      nearPlane > 0 &&
      farPlane > nearPlane
    ) {
      result.nearPlane = nearPlane;
      result.farPlane = farPlane;
    }
  }
  if (viewport.pixelUnprojectionMatrix)
    result.inverseMatrix = Array.from(viewport.pixelUnprojectionMatrix) as NumberArray16;
  const unitsPerMeter = viewport.distanceScales?.unitsPerMeter;
  const unitsPerMeterZ = Array.isArray(unitsPerMeter) ? unitsPerMeter[2] : unitsPerMeter;
  if (typeof unitsPerMeterZ === 'number' && Number.isFinite(unitsPerMeterZ) && unitsPerMeterZ > 0) {
    result.unitsPerMeter = unitsPerMeterZ;
  }
  return result;
}

export type KeplerDistanceFogUniforms = {
  inverseMatrix: Readonly<NumberArray16>;
  fogColor: [number, number, number];
  density: number;
  fogStart: number;
  fogRange: number;
  nearPlane: number;
  farPlane: number;
  viewportSize: [number, number];
};
type DepthBindings = {depthTexture?: Texture};

export const RAY_GROUND_WGSL = /* wgsl */ `
fn keplerFog_getGroundPoint(inverseMatrix: mat4x4f, screenXY: vec2f) -> vec3f {
  let nearH = inverseMatrix * vec4f(screenXY, -1.0, 1.0);
  let farH = inverseMatrix * vec4f(screenXY, 1.0, 1.0);
  let nearP = nearH.xyz / nearH.w;
  let farP = farH.xyz / farH.w;
  let denominator = farP.z - nearP.z;
  var t = 0.0;
  if (abs(denominator) > 1e-6) { t = -nearP.z / denominator; }
  return vec3f(nearP.xy + t * (farP.xy - nearP.xy), clamp(t, 0.0, 1.0));
}
fn keplerFog_loadDepth(depthTexture: texture_depth_2d, texCoord: vec2f) -> f32 {
  let dimensions = textureDimensions(depthTexture);
  let coordinate = clamp(vec2i(texCoord * vec2f(dimensions)), vec2i(0), vec2i(dimensions) - 1);
  return textureLoad(depthTexture, coordinate, 0);
}
fn keplerFog_composite(color: vec4f, fogColor: vec3f, fogFactor: f32) -> vec4f {
  // Transparent pixels (basemap gaps) are treated as if they already contain the fog color.
  var baseRgb = fogColor;
  if (color.a > 0.001) { baseRgb = color.rgb / color.a; }
  let straightRgb = mix(baseRgb, fogColor, fogFactor);
  let outAlpha = fogFactor + color.a * (1.0 - fogFactor);
  return vec4f(straightRgb * outAlpha, outAlpha);
}
`;

const DISTANCE_FOG_WGSL = /* wgsl */ `
struct keplerDistanceFogUniforms {
  inverseMatrix: mat4x4f,
  fogColor: vec3f,
  density: f32,
  fogStart: f32,
  fogRange: f32,
  nearPlane: f32,
  farPlane: f32,
  viewportSize: vec2f,
};
@group(0) @binding(auto) var<uniform> keplerDistanceFog: keplerDistanceFogUniforms;
@group(0) @binding(auto) var depthTexture: texture_depth_2d;
${RAY_GROUND_WGSL}
fn keplerDistanceFog_sampleColor(sourceTexture: texture_2d<f32>, sourceTextureSampler: sampler, texSize: vec2f, texCoord: vec2f) -> vec4f {
  let color = textureSampleLevel(sourceTexture, sourceTextureSampler, texCoord, 0);
  let depth = keplerFog_loadDepth(depthTexture, texCoord);
  let n = keplerDistanceFog.nearPlane;
  let f = keplerDistanceFog.farPlane;
  var linearDepth = 0.0;
  if (depth >= 0.999) {
    let screenXY = texCoord * keplerDistanceFog.viewportSize;
    linearDepth = keplerFog_getGroundPoint(keplerDistanceFog.inverseMatrix, screenXY).z;
  } else {
    let eyeZ = n * f / (f - depth * (f - n));
    linearDepth = (eyeZ - n) / (f - n);
  }
  let fogColor = keplerDistanceFog.fogColor / 255.0;
  let fogEnd = keplerDistanceFog.fogStart + keplerDistanceFog.fogRange;
  let fogFactor = smoothstep(keplerDistanceFog.fogStart, fogEnd, linearDepth) * keplerDistanceFog.density;
  if (fogFactor <= 0.001) { return color; }
  return keplerFog_composite(color, fogColor, fogFactor);
}
`;

/** Kepler distance fog: depth-linearized fog with ground-plane ray distance for basemap pixels. */
export const keplerDistanceFogPass = {
  name: 'keplerDistanceFog',
  source: DISTANCE_FOG_WGSL,
  bindingLayout: [
    {name: 'keplerDistanceFog', group: 0},
    {name: 'depthTexture', group: 0}
  ],
  props: {} as Partial<KeplerDistanceFogUniforms> & DepthBindings,
  uniforms: {} as KeplerDistanceFogUniforms,
  bindings: {} as DepthBindings,
  uniformTypes: {
    inverseMatrix: 'mat4x4<f32>',
    fogColor: 'vec3<f32>',
    density: 'f32',
    fogStart: 'f32',
    fogRange: 'f32',
    nearPlane: 'f32',
    farPlane: 'f32',
    viewportSize: 'vec2<f32>'
  },
  defaultUniforms: {
    inverseMatrix: IDENTITY_MATRIX,
    fogColor: [217, 222, 230],
    density: 0.5,
    fogStart: 0.3,
    fogRange: 0.5,
    nearPlane: 0.1,
    farPlane: 1000,
    viewportSize: [1, 1]
  },
  passes: [{sampler: true}]
} as const satisfies ShaderPass<
  Partial<KeplerDistanceFogUniforms> & DepthBindings,
  KeplerDistanceFogUniforms,
  DepthBindings
>;

export type KeplerSurfaceFogUniforms = {
  inverseMatrix: Readonly<NumberArray16>;
  fogColor: [number, number, number];
  density: number;
  fogHeight: number;
  fogThickness: number;
  unitsPerMeter: number;
  pattern: number;
  viewportSize: [number, number];
};

const SURFACE_FOG_WGSL = /* wgsl */ `
struct keplerSurfaceFogUniforms {
  inverseMatrix: mat4x4f,
  fogColor: vec3f,
  density: f32,
  fogHeight: f32,
  fogThickness: f32,
  unitsPerMeter: f32,
  pattern: f32,
  viewportSize: vec2f,
};
@group(0) @binding(auto) var<uniform> keplerSurfaceFog: keplerSurfaceFogUniforms;
@group(0) @binding(auto) var depthTexture: texture_depth_2d;
${RAY_GROUND_WGSL}
fn keplerSurfaceFog_hashGradient(p: vec2f) -> vec2f {
  let h = dot(p, vec2f(127.1, 311.7));
  let a = fract(sin(h) * 43758.5453) * 6.2831853;
  return vec2f(cos(a), sin(a));
}
fn keplerSurfaceFog_noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let n00 = dot(keplerSurfaceFog_hashGradient(i), f);
  let n10 = dot(keplerSurfaceFog_hashGradient(i + vec2f(1.0, 0.0)), f - vec2f(1.0, 0.0));
  let n01 = dot(keplerSurfaceFog_hashGradient(i + vec2f(0.0, 1.0)), f - vec2f(0.0, 1.0));
  let n11 = dot(keplerSurfaceFog_hashGradient(i + vec2f(1.0, 1.0)), f - vec2f(1.0, 1.0));
  return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y);
}
fn keplerSurfaceFog_fbm(start: vec2f) -> f32 {
  var p = start;
  var v = 0.0;
  var a = 0.5;
  let rotation = mat2x2f(0.6731, 0.7395, -0.7395, 0.6731);
  for (var i = 0; i < 3; i++) {
    v += a * keplerSurfaceFog_noise(p);
    p = rotation * p * 2.03;
    a *= 0.5;
  }
  return v;
}
fn keplerSurfaceFog_wave(m: vec2f, frequency: vec2f, amplitude: f32) -> vec2f {
  return cos(dot(m, frequency)) * amplitude * frequency;
}
fn keplerSurfaceFog_waveNormal(m: vec2f) -> vec3f {
  let normalScale = 6.0;
  var dh = vec2f(0.0);
  dh += keplerSurfaceFog_wave(m, vec2f(0.014, 0.008), 0.50);
  dh += keplerSurfaceFog_wave(m, vec2f(-0.009, 0.019), 0.40);
  dh += keplerSurfaceFog_wave(m, vec2f(0.022, -0.011), 0.30);
  dh += keplerSurfaceFog_wave(m, vec2f(-0.005, -0.025), 0.25);
  dh += keplerSurfaceFog_wave(m, vec2f(0.041, 0.029), 0.18);
  dh += keplerSurfaceFog_wave(m, vec2f(-0.033, 0.037), 0.15);
  dh += keplerSurfaceFog_wave(m, vec2f(0.057, -0.019), 0.10);
  dh += keplerSurfaceFog_wave(m, vec2f(-0.018, -0.051), 0.08);
  dh += keplerSurfaceFog_wave(m, vec2f(0.083, 0.061), 0.06);
  dh += keplerSurfaceFog_wave(m, vec2f(-0.071, 0.089), 0.05);
  dh += keplerSurfaceFog_wave(m, vec2f(0.097, -0.067), 0.04);
  dh += keplerSurfaceFog_wave(m, vec2f(-0.059, -0.103), 0.04);
  dh += keplerSurfaceFog_wave(m, vec2f(0.131, 0.047), 0.03);
  dh += keplerSurfaceFog_wave(m, vec2f(-0.113, 0.127), 0.03);
  dh += keplerSurfaceFog_wave(m, vec2f(0.173, -0.139), 0.025);
  dh += keplerSurfaceFog_wave(m, vec2f(-0.149, 0.181), 0.020);
  dh += keplerSurfaceFog_wave(m, vec2f(0.211, 0.157), 0.018);
  dh += keplerSurfaceFog_wave(m, vec2f(-0.193, -0.167), 0.015);
  let eps = 0.5;
  let nc = keplerSurfaceFog_fbm(m * 0.04);
  let nx = keplerSurfaceFog_fbm((m + vec2f(eps, 0.0)) * 0.04);
  let ny = keplerSurfaceFog_fbm((m + vec2f(0.0, eps)) * 0.04);
  dh += vec2f(nc - nx, nc - ny) * (0.4 / eps);
  return normalize(vec3f(-dh.x * normalScale, 1.0, -dh.y * normalScale));
}
fn keplerSurfaceFog_sampleColor(sourceTexture: texture_2d<f32>, sourceTextureSampler: sampler, texSize: vec2f, texCoord: vec2f) -> vec4f {
  let color = textureSampleLevel(sourceTexture, sourceTextureSampler, texCoord, 0);
  let depth = keplerFog_loadDepth(depthTexture, texCoord);
  let screenXY = texCoord * keplerSurfaceFog.viewportSize;
  let fogHeight = keplerSurfaceFog.fogHeight;
  let fogTop = fogHeight + keplerSurfaceFog.fogThickness;
  let usePattern = keplerSurfaceFog.pattern > 0.5;
  var submersion = 0.0;
  var worldMeters = vec2f(0.0);
  if (depth >= 0.999) {
    // The basemap sits at z = 0 in common space.
    submersion = 1.0 - smoothstep(fogHeight, fogTop, 0.0);
    if (submersion * keplerSurfaceFog.density <= 0.001) { return color; }
    worldMeters = keplerFog_getGroundPoint(keplerSurfaceFog.inverseMatrix, screenXY).xy / keplerSurfaceFog.unitsPerMeter;
  } else {
    var commonPosition = keplerSurfaceFog.inverseMatrix * vec4f(screenXY, depth * 2.0 - 1.0, 1.0);
    commonPosition = commonPosition / commonPosition.w;
    submersion = 1.0 - smoothstep(fogHeight, fogTop, commonPosition.z);
    if (usePattern) { worldMeters = commonPosition.xy / keplerSurfaceFog.unitsPerMeter; }
  }
  let fogFactor = submersion * keplerSurfaceFog.density;
  if (fogFactor <= 0.001) { return color; }
  let fogColor = keplerSurfaceFog.fogColor / 255.0;
  let clampedFactor = clamp(fogFactor, 0.0, 1.0);
  if (!usePattern) { return keplerFog_composite(color, fogColor, clampedFactor); }
  let sunDirection = normalize(vec3f(0.3, 0.8, 0.5));
  let sunColor = vec3f(1.0, 0.95, 0.8);
  let normal = keplerSurfaceFog_waveNormal(worldMeters);
  let eyeDirection = vec3f(0.0, 1.0, 0.0);
  let reflected = normalize(reflect(-sunDirection, normal));
  let specular = pow(max(0.0, dot(eyeDirection, reflected)), 100.0) * sunColor * 2.0;
  let diffuse = max(dot(sunDirection, normal), 0.0) * sunColor * 0.5;
  var baseRgb = fogColor;
  if (color.a > 0.001) { baseRgb = color.rgb / color.a; }
  let tinted = mix(baseRgb, fogColor, submersion);
  let waterSurface = clamp(tinted * (diffuse + vec3f(0.1)) + specular, vec3f(0.0), vec3f(1.0));
  let straightRgb = mix(baseRgb, waterSurface, clampedFactor);
  let outAlpha = clampedFactor + color.a * (1.0 - clampedFactor);
  return vec4f(straightRgb * outAlpha, outAlpha);
}
`;

/** Kepler surface fog: fog below a metre height, optional animated-looking water pattern. */
export const keplerSurfaceFogPass = {
  name: 'keplerSurfaceFog',
  source: SURFACE_FOG_WGSL,
  bindingLayout: [
    {name: 'keplerSurfaceFog', group: 0},
    {name: 'depthTexture', group: 0}
  ],
  props: {} as Partial<KeplerSurfaceFogUniforms> & DepthBindings,
  uniforms: {} as KeplerSurfaceFogUniforms,
  bindings: {} as DepthBindings,
  uniformTypes: {
    inverseMatrix: 'mat4x4<f32>',
    fogColor: 'vec3<f32>',
    density: 'f32',
    fogHeight: 'f32',
    fogThickness: 'f32',
    unitsPerMeter: 'f32',
    pattern: 'f32',
    viewportSize: 'vec2<f32>'
  },
  defaultUniforms: {
    inverseMatrix: IDENTITY_MATRIX,
    fogColor: [230, 235, 242],
    density: 0.6,
    fogHeight: 50,
    fogThickness: 50,
    unitsPerMeter: 1,
    pattern: 0,
    viewportSize: [1, 1]
  },
  passes: [{sampler: true}]
} as const satisfies ShaderPass<
  Partial<KeplerSurfaceFogUniforms> & DepthBindings,
  KeplerSurfaceFogUniforms,
  DepthBindings
>;
