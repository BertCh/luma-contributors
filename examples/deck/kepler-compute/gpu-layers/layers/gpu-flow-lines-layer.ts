// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright (c) Flowmap.gl contributors
// SPDX-FileCopyrightText: Copyright (c) 2018-2020 Teralytics

// Shader logic in this file is adapted from the FlowLinesLayer, CurvedFlowLinesLayer and
// AnimatedFlowLinesLayer GLSL of flowmap.gl (Apache-2.0, Copyright (c) Flowmap.gl contributors,
// Copyright (c) 2018-2020 Teralytics), ported to WGSL for the WebGPU device.

import {
  Layer,
  picking,
  project32,
  type LayerContext,
  type LayerProps,
  type UpdateParameters
} from '@deck.gl/core';
import {type Buffer, type BufferLayout, type Device, type RenderPass} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import type {ShaderModule} from '@luma.gl/shaders';

/** One binary flowmap.gl attribute: a typed array with a per-row component count. */
export type GPUFlowAttribute<ArrayType extends ArrayBufferView = Float32Array> = {
  value: ArrayType;
  size: number;
};

/** Structural copy of flowmap.gl `FlowLinesLayerAttributes`; no flowmap.gl import is required. */
export type GPUFlowLinesData = {
  length: number;
  attributes: {
    /** Float64 lng/lat(/z) per row. */
    getSourcePosition: GPUFlowAttribute<Float64Array | Float32Array>;
    getTargetPosition: GPUFlowAttribute<Float64Array | Float32Array>;
    /** Thickness in 0..0.5, one value per row. */
    getThickness: GPUFlowAttribute<Float32Array>;
    /** RGBA bytes, four per row. */
    getColor: GPUFlowAttribute<Uint8Array | Uint8ClampedArray>;
    /** Start and end trim in pixels, two per row. */
    getEndpointOffsets: GPUFlowAttribute<Float32Array>;
    /** Animation phase per row; derived from the row index when omitted. */
    getStaggering?: GPUFlowAttribute<Float32Array>;
    /** Curve bulge in pixels per row; zero when omitted. */
    getCurveOffset?: GPUFlowAttribute<Float32Array>;
  };
};

export type GPUFlowLinesMode = 'straight' | 'curved' | 'animated-straight';

export type GPUFlowLinesLayerProps = Omit<LayerProps, 'data'> & {
  data: GPUFlowLinesData;
  /** Defaults to 'straight'. */
  mode?: GPUFlowLinesMode;
  /** Defaults to true. */
  drawOutline?: boolean;
  /** RGBA 0-255, defaults to white. */
  outlineColor?: readonly [number, number, number, number];
  /** Inset outline width in pixels, defaults to 1. */
  outlineThickness?: number;
  /** Pixels per thickness unit, defaults to 12. */
  thicknessUnit?: number;
  /** Scales the curve bulge of the 'curved' mode, defaults to 1. */
  curviness?: number;
  /** Fraction of each animation period that fades in, defaults to 0.7. */
  animationTailLength?: number;
};

type GPUFlowLinesLayerState = {
  models: Model[];
  buffers: Buffer[];
  boundData: GPUFlowLinesData | null;
  boundMode: GPUFlowLinesMode | null;
};

/** Number of straight segments in the curved shaft; the arrow head adds one more. */
export const GPU_FLOW_CURVED_SHAFT_SEGMENTS = 24;
const STRAIGHT_VERTEX_COUNT = 9;
const CURVED_VERTEX_COUNT = (2 * (GPU_FLOW_CURVED_SHAFT_SEGMENTS - 1) + 2) * 3;
const ANIMATED_VERTEX_COUNT = 6;
const ANIMATION_LOOP_LENGTH = 1800;
const ANIMATION_SPEED = 20;
const ANIMATION_LOOP_TIME = ANIMATION_LOOP_LENGTH / ANIMATION_SPEED;

/** Uniforms shared by the straight, curved and animated flow line shaders. */
export const gpuFlowLines = {
  name: 'flowLines',
  uniformTypes: {
    outlineColor: 'vec4<f32>',
    style: 'vec4<f32>',
    animation: 'vec4<f32>'
  },
  defaultUniforms: {
    outlineColor: [1, 1, 1, 1],
    style: [24, 1, 1, 0.5],
    animation: [1, 0, 0.7, 0]
  },
  source: /* wgsl */ `
struct flowLinesUniforms {
  // Linear RGBA outline color.
  outlineColor: vec4<f32>,
  // x thicknessUnit (already doubled like flowmap.gl), y outlineThickness, z drawOutline, w gap.
  style: vec4<f32>,
  // x curviness, y animation time, z animation tail length, w unused.
  animation: vec4<f32>,
};
@group(0) @binding(auto) var<uniform> flowLines: flowLinesUniforms;
`
} as const satisfies ShaderModule;

/** Shared WGSL helpers and the vertex output of every flow line shader. */
export const GPU_FLOW_LINES_COMMON_WGSL = /* wgsl */ `
struct FlowLineVertex {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) barycentrics: vec3<f32>,
  @location(2) @interpolate(flat) edgeMask: vec3<f32>,
  @location(3) @interpolate(flat) pickingColor: vec3<f32>,
  @location(4) phase: f32,
};
fn encodePickingColor(rowIndex: u32) -> vec3<f32> {
  let value = rowIndex + 1u;
  return vec3<f32>(f32(value % 256u), f32((value / 256u) % 256u), f32((value / 65536u) % 256u)) / 255.0;
}
fn getBarycentric(vertexIndex: u32) -> vec3<f32> {
  let corner = vertexIndex % 3u;
  return vec3<f32>(select(0.0, 1.0, corner == 0u), select(0.0, 1.0, corner == 1u), select(0.0, 1.0, corner == 2u));
}
fn getSafeDirection(delta: vec2<f32>) -> vec2<f32> {
  let deltaLength = length(delta);
  return select(vec2<f32>(1.0, 0.0), delta / max(deltaLength, 1e-30), deltaLength > 0.0);
}
`;

/** Vertex stage of the straight arrowed flow line, derived from flowmap.gl FlowLinesLayer. */
export const GPU_FLOW_STRAIGHT_VERTEX_WGSL = /* wgsl */ `
// source_target_mix, perpendicular_offset_in_thickness_units, direction_of_travel_offset_in_thickness_units
fn getStraightShape(vertexIndex: u32) -> vec3<f32> {
  let shapes = array<vec3<f32>, 9>(
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 2.0, -3.0), vec3<f32>(1.0, 1.0, -3.0),
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, -3.0), vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)
  );
  return shapes[vertexIndex];
}
fn getStraightOutlineCoefficients(vertexIndex: u32) -> vec2<f32> {
  let coefficients = array<vec2<f32>, 9>(
    vec2<f32>(0.0, 2.0), vec2<f32>(2.0, -1.0), vec2<f32>(1.0, -1.0),
    vec2<f32>(0.0, 2.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, -1.0),
    vec2<f32>(0.0, 2.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, -1.0)
  );
  return coefficients[vertexIndex];
}
fn getStraightOutlineConstants(vertexIndex: u32) -> vec2<f32> {
  let constants = array<vec2<f32>, 9>(
    vec2<f32>(-0.5, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0),
    vec2<f32>(-0.5, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0),
    vec2<f32>(-0.5, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(-0.5, 0.0)
  );
  return constants[vertexIndex];
}
fn getStraightEdgeMask(vertexIndex: u32) -> vec3<f32> {
  let triangleIndex = vertexIndex / 3u;
  if (triangleIndex == 0u) { return vec3<f32>(1.0, 0.0, 1.0); }
  if (triangleIndex == 1u) { return vec3<f32>(1.0, 0.0, 0.0); }
  return vec3<f32>(1.0, 1.0, 0.0);
}
@vertex fn vertexMain(
  @location(0) sourcePositions: vec3<f32>,
  @location(1) targetPositions: vec3<f32>,
  @location(2) sourcePositionsLow: vec3<f32>,
  @location(3) targetPositionsLow: vec3<f32>,
  @location(4) colors: vec4<f32>,
  @location(5) thicknesses: f32,
  @location(6) endpointOffsets: vec2<f32>,
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> FlowLineVertex {
  let shape = getStraightShape(vertexIndex);
  let sourceProjected = project_position_to_clipspace_and_commonspace(sourcePositions, sourcePositionsLow, vec3<f32>(0.0));
  let targetProjected = project_position_to_clipspace_and_commonspace(targetPositions, targetPositionsLow, vec3<f32>(0.0));
  let sourceCommon = sourceProjected.commonPosition;
  let targetCommon = targetProjected.commonPosition;
  let sourceOrTarget = shape.x;
  let lengthCommon = length(targetCommon.xyz - sourceCommon.xyz);
  let limitedOffsetDistances = clamp(
    project_pixel_size_vec2(shape.yz) * flowLines.style.x,
    vec2<f32>(-lengthCommon * 0.8),
    vec2<f32>(lengthCommon * 0.8)
  );
  let startOffsetCommon = project_pixel_size_float(endpointOffsets.x);
  let endOffsetCommon = project_pixel_size_float(endpointOffsets.y);
  let endpointOffset = mix(
    clamp(startOffsetCommon, 0.0, lengthCommon * 0.2),
    -clamp(endOffsetCommon, 0.0, lengthCommon * 0.2),
    sourceOrTarget
  );
  let flowlineDirection = getSafeDirection(targetCommon.xy - sourceCommon.xy);
  let perpendicularDirection = vec2<f32>(-flowlineDirection.y, flowlineDirection.x);
  let outlinePixelOffset = (
    getStraightOutlineCoefficients(vertexIndex) * flowLines.style.y + getStraightOutlineConstants(vertexIndex)
  ) * flowLines.style.z;
  let pixelOffsetCommon = project_pixel_size_vec2(outlinePixelOffset);
  let gapCommon = project_pixel_size_float(flowLines.style.w);
  let offsetCommon = vec4<f32>(
    flowlineDirection * (thicknesses * limitedOffsetDistances.y + pixelOffsetCommon.y + endpointOffset * 1.05) -
      perpendicularDirection * (thicknesses * limitedOffsetDistances.x + gapCommon + pixelOffsetCommon.x),
    0.0,
    0.0
  );
  var output: FlowLineVertex;
  output.position = project_common_position_to_clipspace(mix(sourceCommon, targetCommon, sourceOrTarget) + offsetCommon);
  output.color = colors;
  output.barycentrics = getBarycentric(vertexIndex);
  output.edgeMask = getStraightEdgeMask(vertexIndex);
  output.pickingColor = encodePickingColor(instanceIndex);
  output.phase = 0.0;
  return output;
}
`;

/** Vertex stage of the quadratic-curve arrowed flow line, derived from flowmap.gl CurvedFlowLinesLayer. */
export const GPU_FLOW_CURVED_VERTEX_WGSL = /* wgsl */ `
const HEAD_START_T: f32 = 0.95833333;
const SHAFT_SEGMENTS: f32 = 24.0;
fn getCurvedShape(vertexIndex: u32) -> vec3<f32> {
  let triangleIndex = vertexIndex / 3u;
  let corner = vertexIndex % 3u;
  if (triangleIndex == 46u) {
    let head = array<vec3<f32>, 3>(vec3<f32>(HEAD_START_T, 1.0, 0.0), vec3<f32>(1.0, 1.7, -4.4), vec3<f32>(HEAD_START_T, 0.0, 0.0));
    return head[corner];
  }
  if (triangleIndex == 47u) {
    let head = array<vec3<f32>, 3>(vec3<f32>(HEAD_START_T, 0.0, 0.0), vec3<f32>(1.0, 1.7, -4.4), vec3<f32>(1.0, 0.0, 0.0));
    return head[corner];
  }
  let segment = f32(triangleIndex / 2u);
  let t0 = segment / SHAFT_SEGMENTS;
  let t1 = (segment + 1.0) / SHAFT_SEGMENTS;
  if (triangleIndex % 2u == 0u) {
    let first = array<vec3<f32>, 3>(vec3<f32>(t0, 1.0, 0.0), vec3<f32>(t1, 1.0, 0.0), vec3<f32>(t0, 0.0, 0.0));
    return first[corner];
  }
  let second = array<vec3<f32>, 3>(vec3<f32>(t0, 0.0, 0.0), vec3<f32>(t1, 1.0, 0.0), vec3<f32>(t1, 0.0, 0.0));
  return second[corner];
}
fn getCurvedEdgeMask(vertexIndex: u32) -> vec3<f32> {
  let triangleIndex = vertexIndex / 3u;
  if (triangleIndex == 46u) { return vec3<f32>(0.0, 0.0, 1.0); }
  if (triangleIndex == 47u) { return vec3<f32>(1.0, 1.0, 0.0); }
  if (triangleIndex % 2u == 0u) { return vec3<f32>(0.0, select(0.0, 1.0, triangleIndex == 0u), 1.0); }
  return vec3<f32>(0.0, 1.0, 0.0);
}
fn quadraticBezier(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, t: f32) -> vec3<f32> {
  let oneMinusT = 1.0 - t;
  return oneMinusT * oneMinusT * p0 + 2.0 * oneMinusT * t * p1 + t * t * p2;
}
fn quadraticBezierTangent(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, t: f32) -> vec3<f32> {
  return 2.0 * (1.0 - t) * (p1 - p0) + 2.0 * t * (p2 - p1);
}
@vertex fn vertexMain(
  @location(0) sourcePositions: vec3<f32>,
  @location(1) targetPositions: vec3<f32>,
  @location(2) sourcePositionsLow: vec3<f32>,
  @location(3) targetPositionsLow: vec3<f32>,
  @location(4) colors: vec4<f32>,
  @location(5) thicknesses: f32,
  @location(6) endpointOffsets: vec2<f32>,
  @location(7) curveOffsets: f32,
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> FlowLineVertex {
  let shape = getCurvedShape(vertexIndex);
  let sourceProjected = project_position_to_clipspace_and_commonspace(sourcePositions, sourcePositionsLow, vec3<f32>(0.0));
  let targetProjected = project_position_to_clipspace_and_commonspace(targetPositions, targetPositionsLow, vec3<f32>(0.0));
  let sourceCommon = sourceProjected.commonPosition;
  let targetCommon = targetProjected.commonPosition;
  let chord = targetCommon.xy - sourceCommon.xy;
  let chordLengthCommon = max(length(chord), 1e-6);
  let startTrim = clamp(project_pixel_size_float(endpointOffsets.x) / chordLengthCommon, 0.0, 0.35);
  var endTrim = 1.0 - clamp(project_pixel_size_float(endpointOffsets.y) / chordLengthCommon, 0.0, 0.35);
  endTrim = max(startTrim + 0.05, endTrim);
  let baseHeadBacktrackT = project_pixel_size_float(thicknesses * 3.0 * flowLines.style.x) / chordLengthCommon;
  let availableSpanT = max(endTrim - startTrim, 0.0);
  let headBacktrackT = min(baseHeadBacktrackT, availableSpanT * 0.45);
  let headScale = select(1.0, clamp(headBacktrackT / max(baseHeadBacktrackT, 1e-6), 0.0, 1.0), baseHeadBacktrackT > 1e-6);
  // A soft nonlinear fade of the head deformation avoids tiny heads folding into themselves.
  let headDeformation = smoothstep(0.0, 1.0, headScale);
  let shaftEndTrim = max(startTrim + 0.02, endTrim - headBacktrackT);

  let curveT = select(endTrim, mix(startTrim, shaftEndTrim, shape.x / HEAD_START_T), shape.x < 1.0);
  let headWeight = smoothstep(HEAD_START_T, 1.0, shape.x);
  let tangentT = mix(curveT, endTrim, headWeight);
  let chordDirection = chord / chordLengthCommon;
  let curveNormal = select(vec2<f32>(0.0, 1.0), vec2<f32>(chordDirection.y, -chordDirection.x), length(chord) > 1e-6);
  var controlCommon = mix(sourceCommon.xyz, targetCommon.xyz, 0.5);
  controlCommon = vec3<f32>(
    controlCommon.xy + curveNormal * project_pixel_size_float(abs(curveOffsets)) * flowLines.animation.x,
    controlCommon.z
  );
  let curvePoint = quadraticBezier(sourceCommon.xyz, controlCommon, targetCommon.xyz, curveT);
  var tangent = quadraticBezierTangent(sourceCommon.xyz, controlCommon, targetCommon.xyz, tangentT);
  if (length(tangent.xy) < 1e-6) {
    tangent = targetCommon.xyz - sourceCommon.xyz;
  }
  let flowlineDirection = getSafeDirection(tangent.xy);
  let perpendicularDirection = vec2<f32>(-flowlineDirection.y, flowlineDirection.x);
  let widthScale = mix(1.0, headScale, headWeight);
  let lengthScale = mix(1.0, headScale, headWeight);
  let shapeY = mix(min(shape.y, 1.0), shape.y, headDeformation);
  let shapeZ = shape.z * headDeformation;
  let normalDistanceCommon = clamp(
    project_pixel_size_float(thicknesses * shapeY * widthScale * flowLines.style.x),
    -chordLengthCommon * 0.8,
    chordLengthCommon * 0.8
  );
  let tangentDistanceCommon = clamp(
    project_pixel_size_float(thicknesses * shapeZ * lengthScale * flowLines.style.x),
    -chordLengthCommon * 0.8,
    chordLengthCommon * 0.8
  );
  let gapCommon = project_pixel_size_float(flowLines.style.w);
  let offsetCommon = vec3<f32>(
    flowlineDirection * tangentDistanceCommon - perpendicularDirection * (normalDistanceCommon + gapCommon),
    0.0
  );
  var output: FlowLineVertex;
  output.position = project_common_position_to_clipspace(vec4<f32>(curvePoint + offsetCommon, 1.0));
  output.color = colors;
  output.barycentrics = getBarycentric(vertexIndex);
  output.edgeMask = getCurvedEdgeMask(vertexIndex);
  output.pickingColor = encodePickingColor(instanceIndex);
  output.phase = 0.0;
  return output;
}
`;

/** Fragment stage with the inset barycentric outline, shared by the straight and curved modes. */
export const GPU_FLOW_LINES_FRAGMENT_WGSL = /* wgsl */ `
@fragment fn fragmentMain(input: FlowLineVertex) -> @location(0) vec4<f32> {
  // Derivatives are computed first so they stay in uniform control flow.
  let edgeDistancePixels = input.barycentrics / max(fwidth(input.barycentrics), vec3<f32>(1e-4));
  let maskedDistancePixels = mix(vec3<f32>(1e6), edgeDistancePixels, step(vec3<f32>(0.5), input.edgeMask));
  let minimumBoundaryDistance = min(maskedDistancePixels.x, min(maskedDistancePixels.y, maskedDistancePixels.z));
  let outlineThickness = max(flowLines.style.y, 0.001);
  let outlineMix = (1.0 - smoothstep(max(outlineThickness - 1.0, 0.0), outlineThickness, minimumBoundaryDistance)) *
    select(0.0, 1.0, flowLines.style.z > 0.5);
  let fillColor = vec4<f32>(input.color.rgb, input.color.a * layer.opacity);
  if (fillColor.a <= 0.0) { discard; }
  if (picking.isActive > 0.5) { return vec4<f32>(input.pickingColor, 1.0); }
  let color = mix(
    fillColor,
    vec4<f32>(flowLines.outlineColor.rgb, flowLines.outlineColor.a * fillColor.a),
    outlineMix
  );
  return vec4<f32>(color.rgb * color.a, color.a);
}
`;

/** Vertex and fragment stages of the animated straight flow line, derived from flowmap.gl AnimatedFlowLinesLayer. */
export const GPU_FLOW_ANIMATED_WGSL = /* wgsl */ `
const NUM_PARTS: f32 = 5.0;
const ANIMATION_SPEED_FACTOR: f32 = 0.015;
fn getAnimatedCorner(vertexIndex: u32) -> vec2<f32> {
  // x: 0 at the source and 1 at the target; y: -1 left and 1 right of the line.
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0)
  );
  return corners[vertexIndex];
}
@vertex fn vertexMain(
  @location(0) sourcePositions: vec3<f32>,
  @location(1) targetPositions: vec3<f32>,
  @location(2) sourcePositionsLow: vec3<f32>,
  @location(3) targetPositionsLow: vec3<f32>,
  @location(4) colors: vec4<f32>,
  @location(5) thicknesses: f32,
  @location(6) staggerings: f32,
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> FlowLineVertex {
  let corner = getAnimatedCorner(vertexIndex);
  let sourceClip = project_position_to_clipspace(sourcePositions, sourcePositionsLow, vec3<f32>(0.0));
  let targetClip = project_position_to_clipspace(targetPositions, targetPositionsLow, vec3<f32>(0.0));
  let widthPixels = thicknesses * flowLines.style.x;
  let screenDirection = getSafeDirection((targetClip.xy / targetClip.w - sourceClip.xy / sourceClip.w) * project.viewportSize);
  let perpendicularDirection = vec2<f32>(-screenDirection.y, screenDirection.x);
  let mixedClip = mix(sourceClip, targetClip, corner.x);
  var output: FlowLineVertex;
  output.position = vec4<f32>(
    mixedClip.xy + project_pixel_size_to_clipspace(perpendicularDirection * corner.y * widthPixels * 0.5) * mixedClip.w,
    mixedClip.z,
    mixedClip.w
  );
  output.color = colors;
  output.barycentrics = vec3<f32>(0.0);
  output.edgeMask = vec3<f32>(0.0);
  output.pickingColor = encodePickingColor(instanceIndex);
  output.phase = corner.x * length(sourceClip - targetClip) * NUM_PARTS - flowLines.animation.y * ANIMATION_SPEED_FACTOR + staggerings;
  return output;
}
@fragment fn fragmentMain(input: FlowLineVertex) -> @location(0) vec4<f32> {
  let tailLength = flowLines.animation.z;
  let alpha = input.color.a * layer.opacity * smoothstep(1.0 - tailLength, 1.0, fract(input.phase));
  if (alpha <= 0.0) { discard; }
  if (picking.isActive > 0.5) { return vec4<f32>(input.pickingColor, 1.0); }
  return vec4<f32>(input.color.rgb * alpha, alpha);
}
`;

/** Returns the complete WGSL source for one flow line mode. */
export function getFlowLinesShaderSource(mode: GPUFlowLinesMode): string {
  if (mode === 'animated-straight') {
    return `${GPU_FLOW_LINES_COMMON_WGSL}${GPU_FLOW_ANIMATED_WGSL}`;
  }
  const vertex = mode === 'curved' ? GPU_FLOW_CURVED_VERTEX_WGSL : GPU_FLOW_STRAIGHT_VERTEX_WGSL;
  return `${GPU_FLOW_LINES_COMMON_WGSL}${vertex}${GPU_FLOW_LINES_FRAGMENT_WGSL}`;
}

/** Splits positions into float32 high parts and float32 residual low parts (three components per row). */
export function packFlowPositions(
  value: ArrayLike<number>,
  size: number,
  length: number
): {high: Float32Array; low: Float32Array} {
  const high = new Float32Array(length * 3);
  const low = new Float32Array(length * 3);
  for (let rowIndex = 0; rowIndex < length; rowIndex++) {
    for (let componentIndex = 0; componentIndex < Math.min(size, 3); componentIndex++) {
      const source = value[rowIndex * size + componentIndex];
      const rounded = Math.fround(source);
      high[rowIndex * 3 + componentIndex] = rounded;
      low[rowIndex * 3 + componentIndex] = source - rounded;
    }
  }
  return {high, low};
}

/** Deterministic pseudo-random animation phase in [0, 1) for rows without getStaggering. */
export function packFlowStaggering(length: number): Float32Array {
  const staggering = new Float32Array(length);
  for (let rowIndex = 0; rowIndex < length; rowIndex++) {
    const value = Math.sin((rowIndex + 1) * 12.9898) * 43758.5453;
    staggering[rowIndex] = value - Math.floor(value);
  }
  return staggering;
}

/** Returns animation time in flowmap.gl units for a time in seconds. */
export function getFlowAnimationTime(seconds: number): number {
  return ((seconds % ANIMATION_LOOP_TIME) / ANIMATION_LOOP_TIME) * ANIMATION_LOOP_LENGTH;
}

/** Copies a typed-array view when it does not start at its buffer origin so GPU uploads see only the view. */
export function getOwnedFlowArray<ArrayType extends Float32Array | Uint8Array | Uint8ClampedArray>(
  value: ArrayType
): ArrayType {
  return (value.byteOffset === 0 ? value : value.slice()) as ArrayType;
}

/** Creates one instance-stepped attribute buffer and its layout. */
export function createFlowInstanceAttribute(
  device: Device,
  id: string,
  name: string,
  data: Float32Array | Uint8Array | Uint8ClampedArray,
  format: 'float32' | 'float32x2' | 'float32x3' | 'unorm8x4'
): {buffer: Buffer; layout: BufferLayout} {
  const bytes = data instanceof Uint8ClampedArray ? new Uint8Array(data) : data;
  return {
    buffer: device.createBuffer({id, data: getOwnedFlowArray(bytes as Float32Array)}),
    layout: {name, format, stepMode: 'instance'}
  };
}

/** Deck layer drawing flowmap.gl style arrowed, curved or animated flow lines from binary attributes on WebGPU. */
export class GPUFlowLinesLayer extends Layer<GPUFlowLinesLayerProps> {
  static override layerName = 'GPUFlowLinesLayer';
  static override defaultProps = {
    mode: 'straight',
    drawOutline: true,
    outlineColor: [255, 255, 255, 255],
    outlineThickness: 1,
    thicknessUnit: 12,
    curviness: 1,
    animationTailLength: 0.7,
    parameters: {depthCompare: 'always', depthWriteEnabled: false, cullMode: 'none'}
  };
  override getAttributeManager() {
    return null;
  }

  override initializeState(context: LayerContext): void {
    this.setState({
      models: [],
      buffers: [],
      boundData: null,
      boundMode: null
    } satisfies GPUFlowLinesLayerState);
    this.rebuildResources(context.device);
  }

  override updateState({props}: UpdateParameters<this>): void {
    const state = this.state as GPUFlowLinesLayerState;
    const mode = props.mode ?? 'straight';
    if (props.data !== state.boundData || mode !== state.boundMode) {
      this.rebuildResources(this.context.device);
    }
  }

  override getModels(): Model[] {
    return (this.state as GPUFlowLinesLayerState).models;
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {models} = this.state as GPUFlowLinesLayerState;
    const mode = this.props.mode ?? 'straight';
    const [red, green, blue, alpha] = this.props.outlineColor ?? [255, 255, 255, 255];
    const animationTime = getFlowAnimationTime(performance.now() / 1000);
    for (const model of models) {
      model.shaderInputs.setProps({
        flowLines: {
          outlineColor: [red / 255, green / 255, blue / 255, alpha / 255],
          style: [
            (this.props.thicknessUnit ?? 12) * 2,
            this.props.outlineThickness ?? 1,
            (this.props.drawOutline ?? true) ? 1 : 0,
            0.5
          ],
          animation: [
            this.props.curviness ?? 1,
            animationTime,
            this.props.animationTailLength ?? 0.7,
            0
          ]
        }
      });
      model.draw(renderPass);
    }
    // The animation advances with wall-clock time, so keep requesting frames.
    if (mode === 'animated-straight' && models.length > 0) this.setNeedsRedraw();
  }

  override finalizeState(context: LayerContext): void {
    this.destroyResources();
    super.finalizeState(context);
  }

  private rebuildResources(device: Device): void {
    this.destroyResources();
    const {data} = this.props;
    const mode = this.props.mode ?? 'straight';
    const rowCount = data?.length ?? 0;
    this.setState({boundData: data ?? null, boundMode: mode});
    if (rowCount === 0) return;

    const attributes = data.attributes;
    const buffers: Buffer[] = [];
    const layouts: BufferLayout[] = [];
    const bindings: Record<string, Buffer> = {};
    const addAttribute = (
      name: string,
      array: Float32Array | Uint8Array | Uint8ClampedArray,
      format: 'float32' | 'float32x2' | 'float32x3' | 'unorm8x4'
    ) => {
      const attribute = createFlowInstanceAttribute(
        device,
        `${this.id}-${name}`,
        name,
        array,
        format
      );
      buffers.push(attribute.buffer);
      layouts.push(attribute.layout);
      bindings[name] = attribute.buffer;
    };

    const sourcePositions = packFlowPositions(
      attributes.getSourcePosition.value,
      attributes.getSourcePosition.size,
      rowCount
    );
    const targetPositions = packFlowPositions(
      attributes.getTargetPosition.value,
      attributes.getTargetPosition.size,
      rowCount
    );
    addAttribute('sourcePositions', sourcePositions.high, 'float32x3');
    addAttribute('targetPositions', targetPositions.high, 'float32x3');
    addAttribute('sourcePositionsLow', sourcePositions.low, 'float32x3');
    addAttribute('targetPositionsLow', targetPositions.low, 'float32x3');
    addAttribute('colors', attributes.getColor.value, 'unorm8x4');
    addAttribute('thicknesses', attributes.getThickness.value, 'float32');
    if (mode === 'animated-straight') {
      addAttribute(
        'staggerings',
        attributes.getStaggering?.value ?? packFlowStaggering(rowCount),
        'float32'
      );
    } else {
      addAttribute('endpointOffsets', attributes.getEndpointOffsets.value, 'float32x2');
      if (mode === 'curved') {
        addAttribute(
          'curveOffsets',
          attributes.getCurveOffset?.value ?? new Float32Array(rowCount),
          'float32'
        );
      }
    }

    const vertexCount =
      mode === 'curved'
        ? CURVED_VERTEX_COUNT
        : mode === 'animated-straight'
          ? ANIMATED_VERTEX_COUNT
          : STRAIGHT_VERTEX_COUNT;
    const model = new Model(device, {
      ...this.getShaders({
        source: getFlowLinesShaderSource(mode),
        modules: [project32, picking, gpuFlowLines]
      }),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount,
      instanceCount: rowCount,
      attributes: bindings,
      bufferLayout: layouts
    });
    this.setState({models: [model], buffers});
  }

  private destroyResources(): void {
    const state = this.state as GPUFlowLinesLayerState;
    state.models?.forEach(model => model.destroy());
    state.buffers?.forEach(buffer => buffer.destroy());
    this.setState({models: [], buffers: []});
  }
}
