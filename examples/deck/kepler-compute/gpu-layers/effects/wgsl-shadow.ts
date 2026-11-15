// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {project, shadow as deckShadowModule} from '@deck.gl/core';
import type {Device, Texture} from '@luma.gl/core';
import type {ShaderModule} from '@luma.gl/shaders';

/**
 * WGSL port of Deck's `shadow` shader module for WebGPU.
 *
 * Deck's module is GLSL only and relies on injection hooks that WGSL layer shaders do not have.
 * This module keeps Deck's uniform contract (`shadowMatrices`, `shadowMaps`, `shadowColor`,
 * `drawToShadowMap`, `shadowLightId`) so Deck's own matrix computation is reused, and exposes
 * explicit WGSL entry points that participating layer shaders call:
 *
 * - `shadow_setVertexPosition(positionCommon, clipPosition)` replaces the clip position with the
 *   light's orthographic projection while the shadow map is drawn.
 * - `shadow_getVertexPosition(lightIndex, positionCommon)` returns the shadow map coordinate
 *   (xy = texture uv, z = light depth) interpolated to the fragment stage.
 * - `shadow_filterPremultiplied(color, fragmentDepth, shadowPosition0, shadowPosition1)` writes the
 *   packed light depth while the shadow map is drawn and otherwise blends the shadow color like
 *   Deck's `shadow_filterShadowColor`, returning a premultiplied color.
 *
 * Shadow maps use Deck's RGBA8 depth packing; lookups use a four tap bilinear weight over exact
 * texel loads so packed depths are never blended.
 */
export const WGSL_SHADOW_MODULE_NAME = 'shadow';

/** WGSL fields added to a layer's vertex output struct for the two light-space positions. */
export const WGSL_SHADOW_VARYING_FIELDS = (firstLocation: number): string =>
  `@location(${firstLocation}) shadowPosition0: vec3<f32>,\n  @location(${firstLocation + 1}) shadowPosition1: vec3<f32>,`;

const SHADOW_WGSL = /* wgsl */ `
struct shadowUniforms {
  drawShadowMap: f32,
  useShadowMap: f32,
  lightId: f32,
  lightCount: f32,
  outputUniformShadow: f32,
  useSimplePhong: f32,
  color: vec4<f32>,
  viewProjectionMatrix0: mat4x4<f32>,
  viewProjectionMatrix1: mat4x4<f32>,
  projectCenter0: vec4<f32>,
  projectCenter1: vec4<f32>,
  // Per light: x = common units per depth unit (0 keeps the legacy normalized bias), y = bias start in common units.
  depthParams0: vec4<f32>,
  depthParams1: vec4<f32>,
};
@group(0) @binding(auto) var<uniform> shadow: shadowUniforms;
@group(0) @binding(auto) var shadowMap0: texture_2d<f32>;
@group(0) @binding(auto) var shadowMap1: texture_2d<f32>;

const SHADOW_BIT_PACK_SHIFT = vec4<f32>(1.0, 255.0, 65025.0, 16581375.0);
const SHADOW_BIT_UNPACK_SHIFT = vec4<f32>(1.0, 1.0 / 255.0, 1.0 / 65025.0, 1.0 / 16581375.0);
const SHADOW_BIT_MASK = vec4<f32>(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 0.0);

fn shadow_projectToLight(lightIndex: u32, positionCommon: vec4<f32>) -> vec4<f32> {
  if (lightIndex == 0u) {
    return project_common_position_to_clipspace_with_projection(positionCommon, shadow.viewProjectionMatrix0, shadow.projectCenter0);
  }
  return project_common_position_to_clipspace_with_projection(positionCommon, shadow.viewProjectionMatrix1, shadow.projectCenter1);
}

fn shadow_setVertexPosition(positionCommon: vec4<f32>, clipPosition: vec4<f32>) -> vec4<f32> {
  if (shadow.drawShadowMap > 0.5) {
    return shadow_projectToLight(u32(shadow.lightId), positionCommon);
  }
  return clipPosition;
}

fn shadow_getVertexPosition(lightIndex: u32, positionCommon: vec4<f32>) -> vec3<f32> {
  if (shadow.useShadowMap < 0.5 || f32(lightIndex) >= shadow.lightCount) { return vec3<f32>(0.0); }
  let clip = shadow_projectToLight(lightIndex, positionCommon);
  let ndc = clip.xyz / clip.w;
  // Clip z is already in the WebGPU [0, 1] range; texture v runs top to bottom.
  return vec3<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

fn shadow_loadDepth(shadowMap: texture_2d<f32>, texel: vec2<i32>) -> f32 {
  let size = vec2<i32>(textureDimensions(shadowMap));
  let rgbaDepth = textureLoad(shadowMap, clamp(texel, vec2<i32>(0), size - vec2<i32>(1)), 0);
  return dot(rgbaDepth, SHADOW_BIT_UNPACK_SHIFT);
}

// Bias is expressed in common (world) units so it does not depend on how deep the light frustum is:
// a globe frustum spans the whole shell (about 12,000 km), where a fixed fraction of the depth range
// would swallow building-scale casters.
fn shadow_getOccluded(depthDifference: f32, depthParams: vec4<f32>) -> f32 {
  if (depthParams.x <= 0.0) { return smoothstep(0.001, 0.01, depthDifference); }
  return smoothstep(depthParams.y, depthParams.y * 10.0, depthDifference * depthParams.x);
}

fn shadow_getShadowWeight(position: vec3<f32>, shadowMap: texture_2d<f32>, depthParams: vec4<f32>) -> f32 {
  // Night time in Kepler: the whole scene is in uniform shadow instead of sampling the map.
  if (shadow.outputUniformShadow > 0.5) { return 1.0; }
  if (position.x < 0.0 || position.y < 0.0 || position.x > 1.0 || position.y > 1.0 || position.z > 1.0) { return 0.0; }
  let size = vec2<f32>(textureDimensions(shadowMap));
  let texel = position.xy * size - vec2<f32>(0.5);
  let base = vec2<i32>(floor(texel));
  let blend = fract(texel);
  let w00 = shadow_getOccluded(position.z - shadow_loadDepth(shadowMap, base), depthParams);
  let w10 = shadow_getOccluded(position.z - shadow_loadDepth(shadowMap, base + vec2<i32>(1, 0)), depthParams);
  let w01 = shadow_getOccluded(position.z - shadow_loadDepth(shadowMap, base + vec2<i32>(0, 1)), depthParams);
  let w11 = shadow_getOccluded(position.z - shadow_loadDepth(shadowMap, base + vec2<i32>(1, 1)), depthParams);
  return mix(mix(w00, w10, blend.x), mix(w01, w11, blend.x), blend.y);
}

fn shadow_filterPremultiplied(color: vec4<f32>, fragmentDepth: f32, shadowPosition0: vec3<f32>, shadowPosition1: vec3<f32>) -> vec4<f32> {
  if (shadow.drawShadowMap > 0.5) {
    var rgbaDepth = fract(fragmentDepth * SHADOW_BIT_PACK_SHIFT);
    rgbaDepth -= rgbaDepth.gbaa * SHADOW_BIT_MASK;
    return rgbaDepth;
  }
  if (shadow.useShadowMap > 0.5) {
    var shadowAlpha = shadow_getShadowWeight(shadowPosition0, shadowMap0, shadow.depthParams0);
    if (shadow.lightCount > 1.0) {
      shadowAlpha += shadow_getShadowWeight(shadowPosition1, shadowMap1, shadow.depthParams1);
    }
    shadowAlpha *= shadow.color.a / shadow.lightCount;
    let blendedAlpha = shadowAlpha + color.a * (1.0 - shadowAlpha);
    if (blendedAlpha > 0.0) {
      let shadowed = vec4<f32>(mix(color.rgb, shadow.color.rgb, shadowAlpha / blendedAlpha), blendedAlpha);
      return vec4<f32>(shadowed.rgb * shadowed.a, shadowed.a);
    }
  }
  return vec4<f32>(color.rgb * color.a, color.a);
}
`;

export type WGSLShadowProps = {
  /** Deck project module props (carries the viewport). */
  project?: unknown;
  shadowMatrices?: unknown[];
  /**
   * Pre-fitted light projections (one per light). They replace Deck's frustum fitting, which
   * assumes a map view; used for orbit, orthographic and globe viewports.
   */
  shadowProjections?: {
    viewProjectionMatrix: number[];
    projectCenter: number[];
    /** [common units per depth unit, bias start in common units]; omitted keeps the legacy normalized bias. */
    depthParams?: number[];
  }[];
  shadowMaps?: Texture[];
  dummyShadowMap?: Texture;
  shadowColor?: ArrayLike<number>;
  shadowLightId?: number;
  drawToShadowMap?: boolean;
  /** False clears every shadow flag, for example on a globe or after the effect is removed. */
  shadowEnabled?: boolean;
  /** Kepler night time: every shadow-receiving pixel gets the full shadow instead of a map lookup. */
  outputUniformShadow?: boolean;
  /** Kepler 3D tiles: PBR layers are shaded with ambient plus Lambertian diffuse instead of PBR. */
  useSimplePhong?: boolean;
};

const fallbackTextures = new Map<Device, Texture>();
const fallbackDeviceListeners = new Set<Device>();

/** CPU mirror of `shadow_getOccluded`: occlusion weight for a light-space NDC depth difference. */
export function getWGSLShadowOcclusion(
  depthDifference: number,
  depthParams?: ArrayLike<number>
): number {
  const range = depthParams?.[0] ?? 0;
  const [low, high] = range > 0 ? [depthParams![1], depthParams![1] * 10] : [0.001, 0.01];
  const value = range > 0 ? depthDifference * range : depthDifference;
  const t = Math.min(Math.max((value - low) / (high - low), 0), 1);
  return t * t * (3 - 2 * t);
}

/** One cached 1x1 texture per device that stands in for a shadow map. Never destroyed with layers. */
export function getWGSLShadowFallbackTexture(device: Device): Texture {
  let texture = fallbackTextures.get(device);
  if (!texture || texture.destroyed || texture.device !== device) {
    texture = device.createTexture({
      id: 'wgsl-shadow-fallback',
      format: 'rgba8unorm',
      width: 1,
      height: 1
    });
    fallbackTextures.set(device, texture);
    if (!fallbackDeviceListeners.has(device)) {
      fallbackDeviceListeners.add(device);
      // Release the entry when the device is destroyed or lost so the map never grows.
      const release = () => {
        fallbackTextures.delete(device);
        fallbackDeviceListeners.delete(device);
      };
      device.lost.then(release, release);
    }
  }
  return texture;
}

function padVector4(value: ArrayLike<number> | undefined): number[] {
  const result = [0, 0, 0, 0];
  if (value)
    for (let index = 0; index < Math.min(4, value.length); index++) result[index] = value[index];
  return result;
}

type DeckShadowUniforms = Record<string, unknown>;

/** Adapts Deck's shadow uniforms (GLSL naming, boolean flags) to the WGSL module layout. */
export function getWGSLShadowUniforms(props: WGSLShadowProps): Record<string, unknown> {
  // Callers supply `dummyShadowMap` from getWGSLShadowFallbackTexture(device); there is no
  // module-global fallback because it could belong to another or a destroyed device.
  const dummyShadowMap = props.dummyShadowMap;
  const projections = props.shadowProjections;
  if (projections?.length && props.shadowEnabled !== false) {
    const color = props.shadowColor ?? [0, 0, 0, 1];
    const uniforms: Record<string, unknown> = {
      drawShadowMap: props.drawToShadowMap ? 1 : 0,
      useShadowMap: props.shadowMaps?.length ? 1 : 0,
      outputUniformShadow: props.outputUniformShadow ? 1 : 0,
      useSimplePhong: props.useSimplePhong ? 1 : 0,
      color: padVector4(color),
      lightId: props.shadowLightId || 0,
      lightCount: projections.length,
      shadowMap0: props.shadowMaps?.[0] ?? dummyShadowMap,
      shadowMap1: props.shadowMaps?.[1] ?? dummyShadowMap
    };
    projections.slice(0, 2).forEach((projection, index) => {
      uniforms[`viewProjectionMatrix${index}`] = projection.viewProjectionMatrix;
      uniforms[`projectCenter${index}`] = padVector4(projection.projectCenter);
      uniforms[`depthParams${index}`] = padVector4(projection.depthParams);
    });
    return uniforms;
  }
  const deckUniforms = (
    deckShadowModule.getUniforms as (options: WGSLShadowProps) => DeckShadowUniforms
  )({...props, dummyShadowMap}) as DeckShadowUniforms;
  const uniforms: Record<string, unknown> = {
    drawShadowMap: deckUniforms['drawShadowMap'] ? 1 : 0,
    useShadowMap: deckUniforms['useShadowMap'] ? 1 : 0,
    outputUniformShadow: props.outputUniformShadow ? 1 : 0,
    useSimplePhong: props.useSimplePhong ? 1 : 0,
    shadowMap0: deckUniforms['shadow_uShadowMap0'] ?? dummyShadowMap,
    shadowMap1: deckUniforms['shadow_uShadowMap1'] ?? dummyShadowMap
  };
  if (deckUniforms['color']) uniforms['color'] = padVector4(deckUniforms['color'] as number[]);
  if (deckUniforms['lightId'] !== undefined) uniforms['lightId'] = Number(deckUniforms['lightId']);
  if (deckUniforms['lightCount'] !== undefined) {
    uniforms['lightCount'] = Number(deckUniforms['lightCount']);
  }
  for (const index of [0, 1]) {
    const matrix = deckUniforms[`viewProjectionMatrix${index}`];
    if (matrix) uniforms[`viewProjectionMatrix${index}`] = Array.from(matrix as ArrayLike<number>);
    const center = deckUniforms[`projectCenter${index}`];
    if (center) uniforms[`projectCenter${index}`] = padVector4(center as ArrayLike<number>);
  }
  return uniforms;
}

const IDENTITY_MATRIX = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Shader module that participating layers add to their module list. */
export const wgslShadow = {
  name: WGSL_SHADOW_MODULE_NAME,
  source: SHADOW_WGSL,
  dependencies: [project],
  bindingLayout: [
    {name: 'shadow', group: 0},
    {name: 'shadowMap0', group: 0},
    {name: 'shadowMap1', group: 0}
  ],
  getUniforms: getWGSLShadowUniforms as unknown as ShaderModule['getUniforms'],
  uniformTypes: {
    drawShadowMap: 'f32',
    useShadowMap: 'f32',
    lightId: 'f32',
    lightCount: 'f32',
    outputUniformShadow: 'f32',
    useSimplePhong: 'f32',
    color: 'vec4<f32>',
    viewProjectionMatrix0: 'mat4x4<f32>',
    viewProjectionMatrix1: 'mat4x4<f32>',
    projectCenter0: 'vec4<f32>',
    projectCenter1: 'vec4<f32>',
    depthParams0: 'vec4<f32>',
    depthParams1: 'vec4<f32>'
  },
  defaultUniforms: {
    drawShadowMap: 0,
    useShadowMap: 0,
    lightId: 0,
    lightCount: 0,
    outputUniformShadow: 0,
    useSimplePhong: 0,
    color: [0, 0, 0, 1],
    viewProjectionMatrix0: IDENTITY_MATRIX,
    viewProjectionMatrix1: IDENTITY_MATRIX,
    projectCenter0: [0, 0, 0, 0],
    projectCenter1: [0, 0, 0, 0],
    depthParams0: [0, 0, 0, 0],
    depthParams1: [0, 0, 0, 0]
  }
} as unknown as ShaderModule;

/**
 * Binds the fallback shadow map to models that include the shadow module and have not received
 * shadow props yet. Cheap enough to call before every draw.
 */
export function ensureWGSLShadowInputs(
  models: readonly {shaderInputs: unknown}[],
  device: Device
): void {
  for (const model of models) {
    const inputs = model.shaderInputs as
      | {
          modules?: Record<string, unknown>;
          moduleBindings?: Record<string, Record<string, unknown>>;
          setProps(props: Record<string, unknown>): void;
        }
      | undefined;
    if (
      inputs?.modules?.[WGSL_SHADOW_MODULE_NAME] &&
      !inputs.moduleBindings?.[WGSL_SHADOW_MODULE_NAME]?.['shadowMap0']
    ) {
      inputs.setProps({
        [WGSL_SHADOW_MODULE_NAME]: {
          shadowEnabled: false,
          dummyShadowMap: getWGSLShadowFallbackTexture(device)
        }
      });
    }
  }
}

/** Inserts the shadow varying fields after the first line of `struct <name> {` containing marker text. */
export function insertWGSLShadowVaryings(
  source: string,
  anchor: string,
  firstLocation: number
): string {
  if (!source.includes(anchor)) return source;
  return source.replace(anchor, `${anchor}\n  ${WGSL_SHADOW_VARYING_FIELDS(firstLocation)}`);
}

/** Replaces `target` once; returns the source unchanged when the anchor is missing. */
export function replaceWGSLShadowAnchor(
  source: string,
  target: string,
  replacement: string
): string {
  return source.includes(target) ? source.replace(target, replacement) : source;
}

/**
 * Adds shadow participation to a layer shader: two light-space varyings, the clip position swap
 * while the shadow map is drawn, and the shadow-aware final color.
 */
export function applyWGSLShadow(
  source: string,
  options: {
    varyingAnchor: string;
    firstLocation: number;
    positionAssignment: string;
    commonPosition: string;
    clipPosition: string;
    finalReturn: string;
    finalColorExpression: string;
    outputName?: string;
    inputName?: string;
  }
): string {
  const output = options.outputName ?? 'output';
  const input = options.inputName ?? 'input';
  let result = insertWGSLShadowVaryings(source, options.varyingAnchor, options.firstLocation);
  result = result
    .split(options.positionAssignment)
    .join(
      `${output}.position = shadow_setVertexPosition(${options.commonPosition}, ${options.clipPosition});\n  ` +
        `${output}.shadowPosition0 = shadow_getVertexPosition(0u, ${options.commonPosition});\n  ` +
        `${output}.shadowPosition1 = shadow_getVertexPosition(1u, ${options.commonPosition});`
    );
  return replaceWGSLShadowAnchor(
    result,
    options.finalReturn,
    `return shadow_filterPremultiplied(${options.finalColorExpression}, ${input}.position.z, ${input}.shadowPosition0, ${input}.shadowPosition1);`
  );
}
