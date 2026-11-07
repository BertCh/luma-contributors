// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_TERRAIN_VISIBILITY} from '@luma-contributors/graph/gpu-terrain';

/** An RGBA color, 0-255 per channel; alpha defaults to 255. */
export type SpatialAnalysisColor = readonly [number, number, number, number?];
export const BLEND_PARAMETERS = {
  depthWriteEnabled: false,
  depthCompare: 'always',
  blend: true,
  blendColorOperation: 'add',
  blendAlphaOperation: 'add',
  blendColorSrcFactor: 'src-alpha',
  blendColorDstFactor: 'one-minus-src-alpha',
  blendAlphaSrcFactor: 'one',
  blendAlphaDstFactor: 'one-minus-src-alpha'
} as const;

export const STORAGE_MODEL_BLEND = {
  depthWriteEnabled: false,
  depthCompare: 'always',
  blend: true,
  blendColorOperation: 'add',
  blendAlphaOperation: 'add',
  blendColorSrcFactor: 'src-alpha',
  blendColorDstFactor: 'one-minus-src-alpha',
  blendAlphaSrcFactor: 'one',
  blendAlphaDstFactor: 'one-minus-src-alpha'
} as const;

export const SEGMENT_BODY = /* wgsl */ `
  let total = offsets[offsetsOffset + ROW_COUNT];
  if (index >= total) {
    segments[segmentsOffset + index * 4u] = 0.0;
    segments[segmentsOffset + index * 4u + 1u] = 0.0;
    segments[segmentsOffset + index * 4u + 2u] = 0.0;
    segments[segmentsOffset + index * 4u + 3u] = 0.0;
    fade[fadeOffset + index] = 0.0;
    return;
  }
  var low = 0u;
  var high = ROW_COUNT;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let neighbor = neighbors[neighborsOffset + index];
  segments[segmentsOffset + index * 4u] = positions[positionsOffset + low * 2u];
  segments[segmentsOffset + index * 4u + 1u] = positions[positionsOffset + low * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = positions[positionsOffset + neighbor * 2u];
  segments[segmentsOffset + index * 4u + 3u] = positions[positionsOffset + neighbor * 2u + 1u];
  fade[fadeOffset + index] = 1.0;`;

/** Marker of the shared style color function that {@link PackedColorRasterLayer} replaces. */
export const STYLE_COLOR_FUNCTION = 'fn getSpatialAnalysisColor(valueRow: u32) -> vec4<f32> {';

export const REGION_PALETTE = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
] as const;

export const ALPHA_BLEND = {
  depthWriteEnabled: false,
  depthCompare: 'always',
  blend: true,
  blendColorOperation: 'add',
  blendAlphaOperation: 'add',
  blendColorSrcFactor: 'src-alpha',
  blendColorDstFactor: 'one-minus-src-alpha',
  blendAlphaSrcFactor: 'one',
  blendAlphaDstFactor: 'one-minus-src-alpha'
} as const;

/** Cluster `i` is drawn with `CLUSTER_PALETTE[i % 8]` by both the points and the centroids. */
export const CLUSTER_PALETTE = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
] as const;

/** Fill epsilon choices in meters; index 0 leaves filled depressions as exact flats. */
export const FILL_EPSILONS = [0, 0.0001, 0.001, 0.01, 0.1] as const;

export const BASIN_PALETTE = [
  [230, 90, 90, 215],
  [240, 170, 60, 215],
  [225, 215, 80, 215],
  [110, 205, 100, 215],
  [70, 200, 190, 215],
  [80, 140, 240, 215],
  [150, 110, 235, 215],
  [225, 100, 200, 215]
] as const;

export const ORDER_PALETTE = [
  [0, 0, 0, 0],
  [120, 190, 255, 255],
  [60, 225, 225, 255],
  [110, 235, 110, 255],
  [250, 230, 70, 255],
  [255, 150, 50, 255],
  [255, 80, 80, 255],
  [240, 90, 235, 255]
] as const;

/** Line widths in CSS pixels for Strahler orders 1, 2, 3 and 4 or more. */
export const ORDER_WIDTHS = [1.2, 2, 3.2, 4.6] as const;

/**
 * Unknown, not a stream: GPU_TERRAIN_FLOW_STREAM_UNRESOLVED (0xffffffff) in `streams` and NaN
 * height above drainage (GPU_TERRAIN_DRAINAGE_UNRESOLVED).
 */
export const UNRESOLVED_COLOR = [255, 176, 0, 170] as const;

/** Gaussian noise per point from a hash of the point index and a seed (Box-Muller). */
export const JITTER_DECLARATIONS = /* wgsl */ `
fn hashValue(value: u32) -> u32 {
  let state = value * 747796405u + 2891336653u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn uniformValue(value: u32) -> f32 {
  return (f32(hashValue(value) >> 8u) + 0.5) / 16777216.0;
}`;

export const SEGMENT_FRAGMENT_WGSL = /* wgsl */ `
@fragment fn fragmentMain(input: GeometryVertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * geometryStyle.opacity * coverage);
}
`;

export const LINE_WGSL = /* wgsl */ `
struct LineOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) side: f32,
};

fn getHiddenLine() -> LineOutput {
  var output: LineOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  output.side = 0.0;
  return output;
}

// One screen-space quad (six vertices) along the segment from start to end.
fn getLineVertex(start: vec2<f32>, end: vec2<f32>, vertexIndex: u32, color: vec4<f32>) -> LineOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  if (!isFiniteVector(start) || !isFiniteVector(end)) { return getHiddenLine(); }
  let startClip = projectShapePosition(start);
  let endClip = projectShapePosition(end);
  let corner = corners[vertexIndex];
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) { direction = screenDirection / directionLength; }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * shapeStyle.numbers.x * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  var output: LineOutput;
  output.position = clipPosition;
  output.color = vec4<f32>(color.rgb, color.a * shapeStyle.numbers.y);
  output.side = corner.y;
  return output;
}

@fragment fn fragmentMain(input: LineOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}
`;

export const MARKER_SHAPES = {disc: 0, ring: 1, diamond: 2} as const;

export const RASTER_BINDINGS_MARKER = '@group(0) @binding(auto) var<storage, read> rasterBounds';

export const SIZE_LINE =
  'project_pixel_size_to_clipspace(corner * spatialAnalysisStyle.sizePixels)';

export const FOCUS_BODY = /* wgsl */ `
  let focus = min(focusRow[focusRowOffset], ROW_COUNT - 1u);
  let start = offsets[offsetsOffset + focus];
  let end = offsets[offsetsOffset + focus + 1u];
  if (index == 0u) {
    var weightSum = 0.0;
    for (var slot = start; slot < end; slot++) {
      weightSum += weights[weightsOffset + slot];
    }
    info[infoOffset] = f32(end - start);
    info[infoOffset + 1u] = weightSum;
  }
  let slot = start + index;
  var other = focus;
  if (slot < end) {
    other = neighbors[neighborsOffset + slot];
  }
  // Unused slots become NaN segments, which the segment layer skips.
  var nanBits = 0x7fc00000u;
  let hidden = bitcast<f32>(nanBits);
  segments[segmentsOffset + index * 4u] = select(hidden, positions[positionsOffset + focus * 2u], slot < end);
  segments[segmentsOffset + index * 4u + 1u] = positions[positionsOffset + focus * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = positions[positionsOffset + other * 2u];
  segments[segmentsOffset + index * 4u + 3u] = positions[positionsOffset + other * 2u + 1u];
  neighborIds[neighborIdsOffset + index] = other;`;

/**
 * Gi* bins are `sint32` in `-3..3`. Reinterpreted as uint32 and reduced modulo the palette size 8
 * (which divides 2^32) they select entries `0` (not significant), `1..3` (hot 90/95/99%) and
 * `7, 6, 5` (cold 90/95/99%).
 */
export const GI_COLORS: Record<number, SpatialAnalysisColor> = {
  1: [253, 174, 97, 235],
  2: [244, 109, 67, 245],
  3: [165, 0, 38, 255],
  7: [171, 217, 233, 235],
  6: [116, 173, 209, 245],
  5: [49, 54, 149, 255]
};

/** Local Moran quadrant codes 1 HH, 2 LH, 3 LL, 4 HL. */
export const MORAN_COLORS: Record<number, SpatialAnalysisColor> = {
  1: [165, 0, 38, 255],
  2: [171, 217, 233, 245],
  3: [49, 54, 149, 255],
  4: [253, 174, 97, 245]
};

export const VISIBILITY_NAMES: Record<number, string> = {
  [GPU_TERRAIN_VISIBILITY.hidden]: 'hidden',
  [GPU_TERRAIN_VISIBILITY.visible]: 'visible',
  [GPU_TERRAIN_VISIBILITY.outOfRange]: 'out of range',
  [GPU_TERRAIN_VISIBILITY.noData]: 'no data',
  [GPU_TERRAIN_VISIBILITY.marginal]: 'marginal'
};

/** Standard source-over blending used by every layer here. */
export const STORAGE_LAYER_BLEND_PARAMETERS = {
  depthWriteEnabled: false,
  depthCompare: 'always',
  blend: true,
  blendColorOperation: 'add',
  blendAlphaOperation: 'add',
  blendColorSrcFactor: 'src-alpha',
  blendColorDstFactor: 'one-minus-src-alpha',
  blendAlphaSrcFactor: 'one',
  blendAlphaDstFactor: 'one-minus-src-alpha'
} as const;

/** WGSL shared by the layers: projection of planar meters to a WebGPU clip position. */
export const STORAGE_LAYER_PROJECTION_WGSL = /* wgsl */ `
fn projectStoragePosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

// Non-finite floats are tested through the exponent bits: compilers may fold x != x.
fn isNonFiniteStorageFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) >= 0x7f800000u;
}
`;

export const SEGMENT_CORNERS_WGSL = /* wgsl */ `
var<private> SEGMENT_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
  vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
);
`;

/** Count tiers (group labels) of the group-border outline: `min(floor(log2(count)), 4)`. */
export const TIER_COLORS = [
  [90, 110, 190, 255],
  [70, 190, 210, 255],
  [110, 220, 130, 255],
  [250, 210, 80, 255],
  [255, 110, 90, 255]
] as const;

/** Cover fill colors by `core` flag: border cells, then core cells. */
export const CORE_COLORS = [
  [255, 150, 60, 215],
  [60, 215, 140, 215]
] as const;

/** Colors of the assembled outline rings, by ring index. */
export const RING_COLORS = [
  [255, 255, 255, 255],
  [255, 214, 64, 255],
  [120, 230, 255, 255],
  [255, 150, 200, 255]
] as const;
