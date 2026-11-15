// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Layer,
  UNIT,
  color,
  picking,
  project32,
  type Color,
  type LayerContext,
  type LayerProps,
  type PickingInfo,
  type UpdateParameters
} from '@deck.gl/core';
import {Buffer, type RenderPass, type ShaderLayout} from '@luma.gl/core';
import type {Model} from '@luma.gl/engine';
import type {GPUVector, VertexList} from '@luma.gl/graph/gpu-data';
import {PathStorageModel} from '@luma.gl/engine/models';
import {
  applyGPUBrushingVisibility,
  gpuBrushing,
  setGPUBrushingProps,
  type GPUBrushingProps
} from './gpu-brushing';
import {GPU_GLOBE_WGSL} from './gpu-globe';
import {GPU_PATH_JOINS_SHADER} from './gpu-path-joins';
import {
  getGPUDataBuffer,
  getGPUVectorPickingProvenance,
  validateGPUVectorPickingCapacity,
  type GPUVectorLayerPickingInfo
} from './gpu-vector-layer-utils';
import {
  GPU_HIGHLIGHT_RANGE_WGSL,
  gpuHighlightRange,
  setGPUHighlightRangeProps,
  type GPUHighlightRangeProps
} from './gpu-highlight';

import {applyWGSLShadow, ensureWGSLShadowInputs, wgslShadow} from '../effects/wgsl-shadow';
import {
  borrowGPUGeometryChunk,
  GPUGeometrySelectionBinding,
  GPU_GEOMETRY_SELECTION_SHADER,
  validateGPUGeometrySelection,
  validateGPUGeometryChunks
} from './gpu-geometry-selection';

/** GPUVector-native variable-length path props. Input vectors are borrowed. */
export type GPUPathLayerProps = Omit<LayerProps, 'data'> &
  GPUHighlightRangeProps &
  GPUBrushingProps & {
    /** One nonzero flag per source path; borrowed and read on every draw. */
    selectionMask?: GPUVector<'uint32'>;
    getPath: GPUVector<VertexList<'float32x2' | 'float32x3' | 'float32x4'>>;
    /**
     * Optional fp64 low part of XY, vertex-aligned with `getPath`. The vertex shader subtracts the
     * viewport origin from high and low parts separately (Deck `position64Low`), which only has
     * effect in Deck's LNGLAT offset mode at high zoom. Omit for the unchanged Float32 path.
     */
    getPathLow?: GPUVector<VertexList<'float32x2'>>;
    getColor?: Color | GPUVector<'unorm8x4' | VertexList<'unorm8x4'>>;
    getWidth?: number | GPUVector<'float32'>;
    /** Defaults to pixels. Meter/common widths use Deck's projection scale. */
    widthUnits?: 'pixels' | 'meters' | 'common';
    widthScale?: number;
    widthMinPixels?: number;
    widthMaxPixels?: number;
    /** Defaults to true; false extrudes each segment in the projected ground plane. */
    billboard?: boolean;
    /** Native miter joins by default; true rounds their outer silhouette. */
    jointRounded?: boolean;
    /** Native flat caps by default; true extends endpoints by a round half-width. */
    capRounded?: boolean;
    /** Native miter limit, in stroke-width ratios; defaults to 4. */
    miterLimit?: number;
    /** Native analytic one-pixel edge coverage; defaults to false. */
    antialiasing?: boolean;
    getTimestamps?: GPUVector<VertexList<'float32'>>;
    /** Defaults to window clipping. Trips keeps history when fadeTrail=false and wraps closed cycles. */
    timestampMode?: 'window' | 'trips';
    viewOrigins?: GPUVector<'float32x4'>;
    currentTime?: number;
    trailLength?: number;
    fadeTrail?: boolean;
  };

type GPUPathLayerBatch = {
  model: PathStorageModel;
  lowBinding?: {buffer: Buffer; offset: number; size?: number};
  selection: GPUGeometrySelectionBinding;
  chunkIndex: number;
  rowIndexBase: number;
};

type GPUPathLayerState = {
  batches: GPUPathLayerBatch[];
  boundInputs: unknown[];
  temporalBuffer: Buffer | null;
  defaultTimestampBuffer: Buffer | null;
};

const GPU_PATH_SHADER_LAYOUT: ShaderLayout = {
  attributes: [
    {name: 'segmentStartPointIndices', location: 0, type: 'u32', stepMode: 'instance'},
    {name: 'segmentFlags', location: 1, type: 'u32', stepMode: 'instance'},
    {name: 'rowIndices', location: 2, type: 'u32', stepMode: 'instance'}
  ],
  bindings: []
};

const GPU_PATH_SHADER = /* wgsl */ `
${GPU_HIGHLIGHT_RANGE_WGSL}
${GPU_GLOBE_WGSL}
${GPU_GEOMETRY_SELECTION_SHADER}
@group(0) @binding(auto) var<storage, read> pathValues: array<f32>;
@group(0) @binding(auto) var<storage, read> pathRanges: array<vec4<u32>>;
@group(0) @binding(auto) var<storage, read> pathViewOrigins: array<vec4<f32>>;
@group(0) @binding(auto) var<storage, read> pathRowColors: array<u32>;
@group(0) @binding(auto) var<storage, read> pathVertexColors: array<u32>;
@group(0) @binding(auto) var<storage, read> pathRowWidths: array<f32>;
// Timestamps and fp64 low parts share this storage binding: default WebGPU devices allow only
// eight storage buffers per vertex stage, so the two streams are mutually exclusive.
@group(0) @binding(auto) var<storage, read> pathTimestamps: array<f32>;

struct PathStorageStyleConfig {
  constantColor: vec4<f32>,
  constantWidth: f32,
  useRowColors: u32,
  useRowWidths: u32,
  batchRowIndexBase: u32,
  pathComponentCount: u32,
  useViewOrigins: u32,
  useVertexColors: u32,
  _padding1: u32,
};
@group(0) @binding(auto) var<uniform> pathStorageStyleConfig: PathStorageStyleConfig;

struct PathLayerStyle {
  currentTime: f32,
  trailLength: f32,
  temporalMode: u32,
  fadeTrail: u32,
  widthScale: f32,
  widthMinPixels: f32,
  widthMaxPixels: f32,
  widthUnits: i32,
  billboard: u32,
  jointRounded: u32,
  capRounded: u32,
  miterLimit: f32,
  antialiasing: u32,
  timestampMode: u32,
  useLowValues: u32,
  _padding2: u32,
};
@group(0) @binding(auto) var<uniform> pathLayerStyle: PathLayerStyle;

struct VertexInputs {
  @location(0) segmentStartPointIndex: u32,
  @location(1) segmentFlags: u32,
  @location(2) rowIndex: u32,
};

struct VertexOutputs {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) cornerOffset: vec2<f32>,
  @location(2) miterLength: f32,
  @location(3) pathPosition: vec2<f32>,
  @location(4) pathLength: f32,
  @location(5) jointType: f32,
  @location(6) measure: f32,
  @location(7) @interpolate(flat) pickingColor: vec3<f32>,
  @location(8) @interpolate(flat) visible: f32,
  @location(9) brushed: f32,
  @location(10) globePosition: vec3<f32>,
};

fn encodePickingColor(rowIndex: u32) -> vec3<f32> {
  let value = rowIndex + 1u;
  return vec3<f32>(f32(value % 256u), f32((value / 256u) % 256u), f32((value / 65536u) % 256u)) / 255.0;
}

fn unpackPathColor(colorWord: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(colorWord & 255u),
    f32((colorWord >> 8u) & 255u),
    f32((colorWord >> 16u) & 255u),
    f32((colorWord >> 24u) & 255u)
  ) / 255.0;
}

fn readPathComponent(pointIndex: u32, componentIndex: u32) -> f32 {
  if (componentIndex >= pathStorageStyleConfig.pathComponentCount) { return 0.0; }
  return pathValues[pointIndex * pathStorageStyleConfig.pathComponentCount + componentIndex];
}

// fp64 low part of XY (Deck position64Low); zero unless a low stream is bound.
fn readPathLow(pointIndex: u32) -> vec3<f32> {
  if (pathLayerStyle.useLowValues == 0u) { return vec3<f32>(0.0); }
  return vec3<f32>(pathTimestamps[pointIndex * 2u], pathTimestamps[pointIndex * 2u + 1u], 0.0);
}

fn readPathPoint(pointIndex: u32) -> vec4<f32> {
  return vec4<f32>(
    readPathComponent(pointIndex, 0u),
    readPathComponent(pointIndex, 1u),
    readPathComponent(pointIndex, 2u),
    readPathComponent(pointIndex, 3u)
  );
}

${GPU_PATH_JOINS_SHADER}
/*BRUSHING_VISIBILITY*/

@vertex fn vertexMain(
  inputs: VertexInputs,
  @builtin(vertex_index) vertexIndex: u32
) -> VertexOutputs {
  let localRowIndex = inputs.rowIndex - pathStorageStyleConfig.batchRowIndexBase;
  let pathRange = pathRanges[localRowIndex];
  let endPointIndex = min(inputs.segmentStartPointIndex + 1u, pathRange.y - 1u);
  let viewOrigin = select(vec4<f32>(0.0), pathViewOrigins[localRowIndex], pathStorageStyleConfig.useViewOrigins != 0u);
  let startWorld = readPathPoint(inputs.segmentStartPointIndex) + viewOrigin;
  let endWorld = readPathPoint(endPointIndex) + viewOrigin;
  let corner = getPathCorner(vertexIndex);
  // XYZM uses M as a measure, not a spatial coordinate for ring closure.
  let closed = (inputs.segmentFlags & 4u) != 0u ||
    (pathStorageStyleConfig.pathComponentCount == 4u && pathRange.y - pathRange.x >= 3u &&
      all(readPathPoint(pathRange.x).xyz == readPathPoint(pathRange.y - 1u).xyz));
  let first = (inputs.segmentFlags & 1u) != 0u;
  let last = (inputs.segmentFlags & 2u) != 0u;
  var previousIndex = inputs.segmentStartPointIndex;
  if (!first) { previousIndex -= 1u; }
  else if (closed) { previousIndex = pathRange.y - 2u; }
  var nextIndex = endPointIndex;
  if (!last) { nextIndex += 1u; }
  else if (closed) { nextIndex = pathRange.x + 1u; }
  let previousWorld = mix(readPathPoint(previousIndex).xyz + viewOrigin.xyz, startWorld.xyz, corner.x);
  let currentWorld = mix(startWorld.xyz, endWorld.xyz, corner.x);
  let nextWorld = mix(endWorld.xyz, readPathPoint(nextIndex).xyz + viewOrigin.xyz, corner.x);
  let startLow = readPathLow(inputs.segmentStartPointIndex);
  let previousLow = mix(readPathLow(previousIndex), startLow, corner.x);
  let currentLow = mix(startLow, readPathLow(endPointIndex), corner.x);
  let nextLow = mix(readPathLow(endPointIndex), readPathLow(nextIndex), corner.x);
  geometry.worldPosition = currentWorld;
  let widthPixels = clamp(project_unit_size_to_pixel(select(pathStorageStyleConfig.constantWidth, pathRowWidths[localRowIndex], pathStorageStyleConfig.useRowWidths != 0u) * pathLayerStyle.widthScale, pathLayerStyle.widthUnits), pathLayerStyle.widthMinPixels, pathLayerStyle.widthMaxPixels) * 0.5;
  let coverageScale = select(1.0, (widthPixels + 0.5 / project.devicePixelRatio) / max(widthPixels, 0.000001), pathLayerStyle.antialiasing != 0u && widthPixels > 0.0);
  var join: PathJoin;
  var clipPosition: vec4<f32>;
  // The shadow map needs the ribbon's real extent: Deck's billboard path collapses to its center line there.
  if (pathLayerStyle.billboard != 0u && shadow.drawShadowMap < 0.5) {
    var previousClip = project_position_to_clipspace(previousWorld, previousLow, vec3<f32>(0.0));
    let currentProjection = project_position_to_clipspace_and_commonspace(currentWorld, currentLow, vec3<f32>(0.0));
    var currentClip = currentProjection.clipPosition;
    var nextClip = project_position_to_clipspace(nextWorld, nextLow, vec3<f32>(0.0));
    geometry.position = currentProjection.commonPosition;
    previousClip = clipPathLine(previousClip, currentClip);
    nextClip = clipPathLine(nextClip, currentClip);
    currentClip = clipPathLine(currentClip, mix(nextClip, previousClip, corner.x));
    join = getPathJoin(previousClip.xyz / previousClip.w, currentClip.xyz / currentClip.w, nextClip.xyz / nextClip.w,
      project_pixel_size_to_clipspace(vec2<f32>(max(widthPixels, 0.000001))), coverageScale, corner, first && !closed, last && !closed);
    clipPosition = vec4<f32>(currentClip.xyz + join.offset * currentClip.w, currentClip.w);
  } else {
    let previousCommon = project_position_vec3_f64(previousWorld, previousLow);
    let currentCommon = project_position_vec3_f64(currentWorld, currentLow);
    let nextCommon = project_position_vec3_f64(nextWorld, nextLow);
    join = getPathJoin(previousCommon, currentCommon, nextCommon,
      vec2<f32>(project_pixel_size_float(max(widthPixels, 0.000001))), coverageScale, corner, first && !closed, last && !closed);
    geometry.position = vec4<f32>(currentCommon + join.offset, 1.0);
    clipPosition = project_common_position_to_clipspace(geometry.position);
  }
  geometry.uv = join.pathPosition;
  var color = pathStorageStyleConfig.constantColor;
  if (pathStorageStyleConfig.useVertexColors != 0u) {
    color = mix(unpackPathColor(pathVertexColors[inputs.segmentStartPointIndex]), unpackPathColor(pathVertexColors[endPointIndex]), corner.x);
  } else if (pathStorageStyleConfig.useRowColors != 0u) {
    color = unpackPathColor(pathRowColors[localRowIndex]);
  }
  let useTimestampColumn = pathLayerStyle.temporalMode == 2u;
  let startMeasure = select(startWorld.w, pathTimestamps[inputs.segmentStartPointIndex], useTimestampColumn);
  let nativeTrips = useTimestampColumn && pathLayerStyle.timestampMode != 0u;
  let timestampEndIndex = select(endPointIndex, pathRange.x, nativeTrips && closed && last);
  let endMeasure = select(endWorld.w, pathTimestamps[timestampEndIndex], useTimestampColumn);
  let temporalEnabled = pathLayerStyle.temporalMode != 0u;
  let visible = widthPixels > 0.0 && isGeometryRowSelected(inputs.rowIndex) && (nativeTrips || !temporalEnabled || (endMeasure >= pathLayerStyle.currentTime - pathLayerStyle.trailLength && startMeasure <= pathLayerStyle.currentTime));
  // Native Trips measures the complete extruded segment, including cap/join extensions.
  let vertexMeasure = mix(startMeasure, endMeasure, select(corner.x, join.pathPosition.y / max(join.pathLength, 0.000001), pathLayerStyle.temporalMode == 2u));
  let pickingColor = encodePickingColor(inputs.rowIndex);
  geometry.worldPosition = mix(startWorld.xyz, endWorld.xyz, corner.x);
  geometry.pickingColor = pickingColor;
  var output: VertexOutputs;
  output.position = clipPosition;
  output.color = vec4<f32>(color.rgb, color.a * layer.opacity);
  output.pickingColor = pickingColor;
  output.visible = select(0.0, 1.0, visible);
  // Native path sublayers brush the interpolated segment-vertex position, per vertex.
  output.brushed = brushingVisibility(currentWorld.xy);
  output.measure = vertexMeasure;
  output.cornerOffset = join.cornerOffset;
  output.miterLength = join.miterLength;
  output.pathPosition = join.pathPosition;
  output.pathLength = join.pathLength;
  output.jointType = join.jointType;
  output.globePosition = geometry.position.xyz;
  return output;
}

@fragment fn fragmentMain(input: VertexOutputs) -> @location(0) vec4<f32> {
  // Derivatives precede all selection/time discards so analytic coverage stays defined.
  let bodyCoordinate = abs(input.pathPosition.x);
  let cornerCoordinate = length(input.cornerOffset);
  let bodyPixels = (1.0 - bodyCoordinate) / max(fwidth(bodyCoordinate), 0.000001);
  let cornerPixels = (1.0 - cornerCoordinate) / max(fwidth(cornerCoordinate), 0.000001);
  let isCorner = input.pathPosition.y < 0.0 || input.pathPosition.y > input.pathLength;
  let isRound = input.jointType > 0.5;
  let edgePixels = select(bodyPixels, cornerPixels, isCorner && isRound);
  if (input.visible < 0.5 || input.brushed < 0.5 || gpuGlobeIsCommonHidden(input.globePosition)) { discard; }
  if (pathLayerStyle.antialiasing != 0u) {
    if (edgePixels <= -0.5) { discard; }
  } else if (isCorner && isRound && cornerCoordinate > 1.0) { discard; }
  if (isCorner && !isRound && input.miterLength > pathLayerStyle.miterLimit + 1.0) { discard; }
  let trailStart = pathLayerStyle.currentTime - pathLayerStyle.trailLength;
  let clipTrailStart = pathLayerStyle.temporalMode != 2u || pathLayerStyle.timestampMode == 0u || pathLayerStyle.fadeTrail != 0u;
  if (pathLayerStyle.temporalMode != 0u && ((clipTrailStart && input.measure < trailStart) || input.measure > pathLayerStyle.currentTime)) { discard; }
  let trailOpacity = select(1.0, clamp((input.measure - trailStart) / max(pathLayerStyle.trailLength, 0.000001), 0.0, 1.0), pathLayerStyle.temporalMode == 2u && pathLayerStyle.fadeTrail != 0u);
  if (picking.isActive > 0.5) { return vec4<f32>(input.pickingColor, 1.0); }
  let coverage = select(1.0, smoothstep(-0.5, 0.5, edgePixels), pathLayerStyle.antialiasing != 0u);
  let highlighted = gpuHighlightRows(input.color, input.pickingColor);
  return deckgl_premultiplied_alpha(vec4<f32>(highlighted.rgb, highlighted.a * trailOpacity * coverage));
}`;

/** Deck host for the GPU-only variable-length path model. */
export class GPUPathLayer extends Layer<GPUPathLayerProps> {
  static override layerName = 'GPUPathLayer';

  private styleScratch?: ArrayBuffer;

  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error('GPUPathLayer requires WebGPU');
    this.setState({
      batches: [],
      boundInputs: [],
      temporalBuffer: null,
      defaultTimestampBuffer: null
    });
    this.createModel();
  }

  override updateState({props}: UpdateParameters<this>): void {
    validateGPUVectorPickingCapacity(props.getPath.length, Boolean(props.pickable));
    validateGPUGeometrySelection(this.context.device, props.getPath, props.selectionMask);
    const {boundInputs} = this.state as GPUPathLayerState;
    const nextInputs = getGPUPathBoundInputs(props);
    // Constant color and width compare by value and appended chunks change the signature, so
    // inline literals do not rebuild models while in-place chunk appends do.
    if (
      nextInputs.length !== boundInputs.length ||
      nextInputs.some((input, index) => input !== boundInputs[index])
    ) {
      this.destroyModel();
      this.createModel();
    }
  }

  override getModels(): Model[] {
    return (this.state as GPUPathLayerState | undefined)?.batches.map(batch => batch.model) ?? [];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {batches, temporalBuffer, defaultTimestampBuffer} = this.state as GPUPathLayerState;
    if (!temporalBuffer) return;
    ensureWGSLShadowInputs(this.getModels(), this.context.device);
    setGPUHighlightRangeProps(this.getModels(), this.props);
    validateGPUVectorPickingCapacity(this.props.getPath.length, Boolean(this.props.pickable));
    validateGPUGeometrySelection(this.context.device, this.props.getPath, this.props.selectionMask);
    // One scratch buffer per layer; every field is rewritten below, so no stale data remains.
    this.styleScratch ??= new ArrayBuffer(64);
    const values = this.styleScratch;
    const floats = new Float32Array(values);
    const uints = new Uint32Array(values);
    floats.set([this.props.currentTime ?? 0, this.props.trailLength ?? 0]);
    uints.set(
      [
        this.props.getTimestamps ? 2 : this.props.currentTime !== undefined ? 1 : 0,
        this.props.fadeTrail ? 1 : 0
      ],
      2
    );
    floats.set(
      [
        this.props.widthScale ?? 1,
        this.props.widthMinPixels ?? 0,
        this.props.widthMaxPixels ?? 1e9
      ],
      4
    );
    uints[7] = UNIT[this.props.widthUnits ?? 'pixels'];
    uints[8] = this.props.billboard === false ? 0 : 1;
    uints[9] = this.props.jointRounded ? 1 : 0;
    uints[10] = this.props.capRounded ? 1 : 0;
    floats[11] = this.props.miterLimit ?? 4;
    uints[12] = this.props.antialiasing ? 1 : 0;
    uints[13] = this.props.timestampMode === 'trips' ? 1 : 0;
    uints[14] = this.props.getPathLow ? 1 : 0;
    temporalBuffer.write(new Uint8Array(values));
    for (const batch of batches) {
      setGPUBrushingProps(batch.model, this.context, this.props);
      batch.model.setProps({
        bindings: {
          pathLayerStyle: temporalBuffer,
          pathTimestamps: batch.lowBinding ?? defaultTimestampBuffer!,
          ...batch.selection.update(this.props.selectionMask, batch.chunkIndex, batch.rowIndexBase)
        }
      });
      batch.model.draw(renderPass);
    }
  }

  override getPickingInfo({info}: {info: PickingInfo}): PickingInfo {
    const result = info as GPUVectorLayerPickingInfo;
    result.gpuVector = getGPUVectorPickingProvenance(this.props.getPath, result.index);
    return result;
  }

  override finalizeState(context: LayerContext): void {
    this.destroyModel();
    super.finalizeState(context);
  }

  private createModel(): void {
    const {device} = this.context;
    const temporalBuffer = device.createBuffer({
      id: `${this.id}-temporal-style`,
      byteLength: 64,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const defaultTimestampBuffer = device.createBuffer({
      id: `${this.id}-default-timestamp`,
      data: new Float32Array([0]),
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    const constantColor = isGPUVector(this.props.getColor)
      ? undefined
      : normalizeColor(this.props.getColor ?? [0, 0, 0, 255]);
    const pathModules = [project32, color, picking, gpuHighlightRange, wgslShadow];
    if (this.props.brushingEnabled) pathModules.push(gpuBrushing);
    // getPathLow and getTimestamps share one storage binding (vertex storage buffer limit).
    if (this.props.getPathLow && this.props.getTimestamps)
      throw new Error('GPUPathLayer getPathLow excludes getTimestamps');
    const batches: GPUPathLayerBatch[] = [];
    let rowIndexBase = 0;
    try {
      validateGPUVectorPickingCapacity(this.props.getPath.length, Boolean(this.props.pickable));
      validateGPUGeometrySelection(device, this.props.getPath, this.props.selectionMask);
      validateGPUGeometryChunks(this.props.getPath, [
        isGPUVector(this.props.getColor) ? this.props.getColor : undefined,
        isGPUVector(this.props.getWidth) ? this.props.getWidth : undefined,
        this.props.getTimestamps,
        this.props.viewOrigins,
        this.props.getPathLow
      ]);
      for (const [chunkIndex, chunk] of this.props.getPath.data.entries()) {
        if (chunk.length > 0) {
          const selection = new GPUGeometrySelectionBinding(device, `${this.id}-${chunkIndex}`);
          try {
            const lowChunk = this.props.getPathLow?.data[chunkIndex];
            const lowBinding = lowChunk
              ? {
                  buffer: getGPUDataBuffer(lowChunk),
                  offset: lowChunk.byteOffset,
                  size: Math.max(4, lowChunk.valueLength * this.props.getPathLow!.byteStride)
                }
              : undefined;
            const model = new PathStorageModel(device, {
              shaderAssembler: this.context.shaderAssembler,
              ...this.getShaders({
                modules: pathModules,
                source: applyGPUBrushingVisibility(
                  applyWGSLShadow(GPU_PATH_SHADER, {
                    varyingAnchor: '@location(10) globePosition: vec3<f32>,',
                    firstLocation: 11,
                    positionAssignment: 'output.position = clipPosition;',
                    commonPosition: 'geometry.position',
                    clipPosition: 'clipPosition',
                    finalReturn:
                      'return deckgl_premultiplied_alpha(vec4<f32>(highlighted.rgb, highlighted.a * trailOpacity * coverage));',
                    finalColorExpression:
                      'vec4<f32>(highlighted.rgb, highlighted.a * trailOpacity * coverage)'
                  }),
                  Boolean(this.props.brushingEnabled)
                )
              }),
              id: `${this.id}-model-${chunkIndex}`,
              shaderLayout: GPU_PATH_SHADER_LAYOUT,
              paths: borrowGPUGeometryChunk(this.props.getPath, chunkIndex),
              rowIndexBase,
              ...(isGPUVector(this.props.getColor)
                ? {colors: borrowGPUGeometryChunk(this.props.getColor, chunkIndex)}
                : {color: constantColor}),
              ...(isGPUVector(this.props.getWidth)
                ? {widths: borrowGPUGeometryChunk(this.props.getWidth, chunkIndex)}
                : {width: typeof this.props.getWidth === 'number' ? this.props.getWidth : 1}),
              ...(this.props.getTimestamps
                ? {timestamps: borrowGPUGeometryChunk(this.props.getTimestamps, chunkIndex)}
                : {}),
              ...(this.props.viewOrigins
                ? {viewOrigins: borrowGPUGeometryChunk(this.props.viewOrigins, chunkIndex)}
                : {}),
              bindings: {
                pathLayerStyle: temporalBuffer,
                pathTimestamps: lowBinding ?? defaultTimestampBuffer,
                ...selection.update(this.props.selectionMask, chunkIndex, rowIndexBase)
              },
              topology: 'triangle-list',
              vertexCount: 12
            });
            batches.push({model, lowBinding, selection, chunkIndex, rowIndexBase});
          } catch (error) {
            selection.destroy();
            throw error;
          }
        }
        rowIndexBase += chunk.length;
      }
    } catch (error) {
      for (const batch of batches) {
        batch.model.destroy();
        batch.selection.destroy();
      }
      temporalBuffer.destroy();
      defaultTimestampBuffer.destroy();
      throw error;
    }
    this.setState({
      batches,
      temporalBuffer,
      defaultTimestampBuffer,
      boundInputs: getGPUPathBoundInputs(this.props)
    } satisfies GPUPathLayerState);
  }

  private destroyModel(): void {
    const state = this.state as GPUPathLayerState;
    for (const batch of state.batches) {
      batch.model.destroy();
      batch.selection.destroy();
    }
    state.temporalBuffer?.destroy();
    state.defaultTimestampBuffer?.destroy();
    this.setState({
      batches: [],
      boundInputs: [],
      temporalBuffer: null,
      defaultTimestampBuffer: null
    });
  }
}

/**
 * Identity signature of everything baked into path models. GPUVectors compare by identity,
 * constants by value, and the trailing entries are the position chunks so appends rebuild.
 */
export function getGPUPathBoundInputs(props: GPUPathLayerProps): unknown[] {
  return [
    props.getPath,
    isGPUVector(props.getColor)
      ? props.getColor
      : normalizeColor(props.getColor ?? [0, 0, 0, 255]).join(','),
    isGPUVector(props.getWidth)
      ? props.getWidth
      : typeof props.getWidth === 'number'
        ? props.getWidth
        : 1,
    props.getTimestamps,
    props.viewOrigins,
    Boolean(props.brushingEnabled),
    props.getPathLow,
    ...props.getPath.data
  ];
}

function normalizeColor(color: Color): [number, number, number, number] {
  return [color[0], color[1], color[2], color[3] ?? 255];
}

function isGPUVector(value: unknown): value is GPUVector {
  return Boolean(value && typeof value === 'object' && 'data' in value && 'format' in value);
}
