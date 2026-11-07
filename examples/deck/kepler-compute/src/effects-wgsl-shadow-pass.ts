// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL screen-space directional shadows for Kepler's Light and Shadow effect on WebGPU.
 *
 * FALLBACK ONLY: real shadow maps live in `WGSLShadowLightingEffect` (deck-gpu-layers). This pass
 * runs only while that effect drew no shadow map (globe or non-map views, no participating layer).
 *
 * Deck's shadow module injects GLSL into every layer shader and renders a second depth pass from
 * the light; layer WGSL shaders have no such injection hooks. This pass reaches the same visual
 * result for directional lights without touching layer shaders: for every pixel it reconstructs
 * the common-space position from the scene depth (basemap pixels use the z = 0 ground plane),
 * marches toward the light in common space, projects each sample back to the screen and tests it
 * against the scene depth buffer. A sample hidden behind nearer scene geometry within a thickness
 * tolerance marks the pixel as shadowed, and the shadow color is blended like Deck's own
 * `shadow_filterShadowColor`.
 *
 * Limits: shadows come only from geometry visible in the current frame (no off-screen casters),
 * the march covers `SHADOW_MARCH_STEPS * SHADOW_STEP_PIXELS` pixels, and at most two lights.
 */

import type {Texture} from '@luma.gl/core';
import type {ShaderPass} from '@luma.gl/shaders';
import type {NumberArray16} from '@math.gl/core';
import {RAY_GROUND_WGSL} from './effects-wgsl-fog-passes';

export const SHADOW_MARCH_STEPS = 32;
export const SHADOW_STEP_PIXELS = 4;
export const DEFAULT_SHADOW_COLOR: [number, number, number, number] = [0, 0, 0, 200 / 255];

export type KeplerShadowUniforms = {
  inverseMatrix: Readonly<NumberArray16>;
  projectionMatrix: Readonly<NumberArray16>;
  /** xyz = unit vector pointing toward the light in common space, w = 1 when the light casts. */
  lightDirection0: [number, number, number, number];
  lightDirection1: [number, number, number, number];
  /** Straight RGB in [0, 1] plus maximum shadow opacity. */
  shadowColor: [number, number, number, number];
  nearPlane: number;
  farPlane: number;
  viewportSize: [number, number];
};
type DepthBindings = {depthTexture?: Texture};

const IDENTITY_MATRIX: NumberArray16 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

const SHADOW_WGSL = /* wgsl */ `
struct keplerShadowUniforms {
  inverseMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  lightDirection0: vec4f,
  lightDirection1: vec4f,
  shadowColor: vec4f,
  nearPlane: f32,
  farPlane: f32,
  viewportSize: vec2f,
};
@group(0) @binding(auto) var<uniform> keplerShadow: keplerShadowUniforms;
@group(0) @binding(auto) var depthTexture: texture_depth_2d;
${RAY_GROUND_WGSL}
fn keplerShadow_eyeDepth(ndcZ: f32) -> f32 {
  let n = keplerShadow.nearPlane;
  let f = keplerShadow.farPlane;
  let depth01 = clamp(ndcZ * 0.5 + 0.5, 0.0, 1.0);
  return n * f / (f - depth01 * (f - n));
}
fn keplerShadow_getOcclusion(origin: vec3f, toLight: vec3f, pixelSize: f32) -> f32 {
  let stepLength = pixelSize * ${SHADOW_STEP_PIXELS}.0;
  let thickness = stepLength * 4.0;
  let bias = pixelSize * 2.0;
  var occlusion = 0.0;
  for (var stepIndex = 1; stepIndex <= ${SHADOW_MARCH_STEPS}; stepIndex++) {
    let samplePoint = origin + toLight * (stepLength * f32(stepIndex));
    let clip = keplerShadow.projectionMatrix * vec4f(samplePoint, 1.0);
    if (clip.w <= 0.0) { break; }
    let screen = clip.xyz / clip.w;
    if (screen.x < 0.0 || screen.y < 0.0 || screen.x >= keplerShadow.viewportSize.x || screen.y >= keplerShadow.viewportSize.y || screen.z > 1.0) { break; }
    let sceneDepth = keplerFog_loadDepth(depthTexture, screen.xy / keplerShadow.viewportSize);
    if (sceneDepth >= 0.999) { continue; }
    let sceneEye = keplerShadow_eyeDepth(sceneDepth * 2.0 - 1.0);
    let sampleEye = keplerShadow_eyeDepth(screen.z);
    let behind = sampleEye - sceneEye;
    if (behind > bias && behind < thickness) { occlusion = 1.0; break; }
  }
  return occlusion;
}
fn keplerShadow_sampleColor(sourceTexture: texture_2d<f32>, sourceTextureSampler: sampler, texSize: vec2f, texCoord: vec2f) -> vec4f {
  let color = textureSampleLevel(sourceTexture, sourceTextureSampler, texCoord, 0);
  let depth = keplerFog_loadDepth(depthTexture, texCoord);
  let screenXY = texCoord * keplerShadow.viewportSize;
  var origin = vec3f(0.0);
  if (depth >= 0.999) {
    let ground = keplerFog_getGroundPoint(keplerShadow.inverseMatrix, screenXY);
    if (ground.z >= 1.0) { return color; }
    origin = vec3f(ground.xy, 0.0);
  } else {
    let unprojected = keplerShadow.inverseMatrix * vec4f(screenXY, depth * 2.0 - 1.0, 1.0);
    origin = unprojected.xyz / unprojected.w;
  }
  let neighborDepth = select(depth, 0.5, depth >= 0.999);
  let neighbor = keplerShadow.inverseMatrix * vec4f(screenXY + vec2f(1.0, 0.0), neighborDepth * 2.0 - 1.0, 1.0);
  let reference = keplerShadow.inverseMatrix * vec4f(screenXY, neighborDepth * 2.0 - 1.0, 1.0);
  let pixelSize = max(length(neighbor.xyz / neighbor.w - reference.xyz / reference.w), 1e-9);
  var occlusion = 0.0;
  if (keplerShadow.lightDirection0.w > 0.5) {
    occlusion = max(occlusion, keplerShadow_getOcclusion(origin, keplerShadow.lightDirection0.xyz, pixelSize));
  }
  if (occlusion < 0.5 && keplerShadow.lightDirection1.w > 0.5) {
    occlusion = max(occlusion, keplerShadow_getOcclusion(origin, keplerShadow.lightDirection1.xyz, pixelSize));
  }
  let shadowAlpha = occlusion * keplerShadow.shadowColor.a;
  if (shadowAlpha <= 0.001) { return color; }
  let blendedAlpha = shadowAlpha + color.a * (1.0 - shadowAlpha);
  var baseRgb = keplerShadow.shadowColor.rgb;
  if (color.a > 0.001) { baseRgb = color.rgb / color.a; }
  let straightRgb = mix(baseRgb, keplerShadow.shadowColor.rgb, shadowAlpha / blendedAlpha);
  return vec4f(straightRgb * blendedAlpha, blendedAlpha);
}
`;

/** Screen-space directional shadows over the scene color using the scene depth attachment. */
export const keplerShadowPass = {
  name: 'keplerShadow',
  source: SHADOW_WGSL,
  bindingLayout: [
    {name: 'keplerShadow', group: 0},
    {name: 'depthTexture', group: 0}
  ],
  props: {} as Partial<KeplerShadowUniforms> & DepthBindings,
  uniforms: {} as KeplerShadowUniforms,
  bindings: {} as DepthBindings,
  uniformTypes: {
    inverseMatrix: 'mat4x4<f32>',
    projectionMatrix: 'mat4x4<f32>',
    lightDirection0: 'vec4<f32>',
    lightDirection1: 'vec4<f32>',
    shadowColor: 'vec4<f32>',
    nearPlane: 'f32',
    farPlane: 'f32',
    viewportSize: 'vec2<f32>'
  },
  defaultUniforms: {
    inverseMatrix: IDENTITY_MATRIX,
    projectionMatrix: IDENTITY_MATRIX,
    lightDirection0: [0, 0, 1, 0],
    lightDirection1: [0, 0, 1, 0],
    shadowColor: DEFAULT_SHADOW_COLOR,
    nearPlane: 0.1,
    farPlane: 1000,
    viewportSize: [1, 1]
  },
  passes: [{sampler: true}]
} as const satisfies ShaderPass<
  Partial<KeplerShadowUniforms> & DepthBindings,
  KeplerShadowUniforms,
  DepthBindings
>;

type LightWithShadow = {shadow?: boolean; direction?: ArrayLike<number>};

/** Unit vector toward a directional light (Deck lights store the direction the light travels). */
export function getDirectionToLight(
  light: LightWithShadow | undefined
): [number, number, number, number] {
  const direction = light?.direction;
  if (!light?.shadow || !direction || direction.length < 3) return [0, 0, 1, 0];
  const length = Math.hypot(direction[0], direction[1], direction[2]);
  if (!(length > 0)) return [0, 0, 1, 0];
  return [-direction[0] / length, -direction[1] / length, -direction[2] / length, 1];
}

type ViewportLike = {
  width: number;
  height: number;
  pixelProjectionMatrix?: ArrayLike<number>;
  pixelUnprojectionMatrix?: ArrayLike<number>;
};

/** Builds one frame of shadow pass uniforms from the Kepler lights and the Deck viewport. */
export function getShadowPassUniforms(
  directionalLights: readonly LightWithShadow[] | undefined,
  shadowColor: ArrayLike<number> | undefined,
  viewport: ViewportLike | undefined,
  cameraPlanes: {nearPlane: number; farPlane: number}
): KeplerShadowUniforms {
  const casting = (directionalLights ?? []).filter(light => light?.shadow);
  const color = shadowColor ?? DEFAULT_SHADOW_COLOR;
  // Deck stores shadow colors with 0-1 RGB and alpha; some Kepler builds pass 0-255 RGB.
  const scale = Math.max(color[0] ?? 0, color[1] ?? 0, color[2] ?? 0) > 1 ? 255 : 1;
  return {
    inverseMatrix: viewport?.pixelUnprojectionMatrix
      ? (Array.from(viewport.pixelUnprojectionMatrix) as NumberArray16)
      : IDENTITY_MATRIX,
    projectionMatrix: viewport?.pixelProjectionMatrix
      ? (Array.from(viewport.pixelProjectionMatrix) as NumberArray16)
      : IDENTITY_MATRIX,
    lightDirection0: getDirectionToLight(casting[0]),
    lightDirection1: getDirectionToLight(casting[1]),
    shadowColor: [
      (color[0] ?? 0) / scale,
      (color[1] ?? 0) / scale,
      (color[2] ?? 0) / scale,
      color[3] ?? DEFAULT_SHADOW_COLOR[3]
    ],
    nearPlane: cameraPlanes.nearPlane,
    farPlane: cameraPlanes.farPlane,
    viewportSize: [viewport?.width ?? 1, viewport?.height ?? 1]
  };
}
