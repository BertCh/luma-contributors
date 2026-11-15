// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Layer,
  project32,
  picking,
  color,
  type Color,
  type LayerContext,
  type LayerProps,
  type PickingInfo,
  type UpdateParameters
} from '@deck.gl/core';
import {assert, Buffer, type Device, type RenderPass} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import type {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import type {DynamicTexture} from '@luma.gl/core/dynamic';
import type {FontAtlas} from '@luma.gl/engine/text';
import {makeTextGlyphAlphaWgsl} from '@luma.gl/engine/text-experimental';
import {
  WGSL_SHADOW_VARYING_FIELDS,
  ensureWGSLShadowInputs,
  wgslShadow
} from '../effects/wgsl-shadow';
import {GPU_GLOBE_WGSL} from './gpu-globe';
import {
  gpuBrushing,
  replaceGPUBrushingMarker,
  setGPUBrushingProps,
  type GPUBrushingProps
} from './gpu-brushing';
import {
  GPUGeometrySelectionBinding,
  GPU_GEOMETRY_SELECTION_SHADER,
  validateGPUGeometryChunks,
  validateGPUGeometrySelection
} from './gpu-geometry-selection';
import {
  getGPUDataBuffer,
  getGPUVectorPickingProvenance,
  validateGPUVectorPickingCapacity,
  type GPUVectorLayerPickingInfo
} from './gpu-vector-layer-utils';

/** Two Float32 vec4 records per glyph: offsetXY/rowAdvance/lineHeight, then atlasXYWH. */
export type GPUPointLabelBatch = {
  glyphLayouts: GPUData<'float32x4'>;
  /** Source-chunk-local row index and atlas page for each glyph. */
  glyphRows: GPUData<'uint32x2'>;
  /** Optional per-row [widestLineAdvance, paragraphHeight, 0, 0] used by label backgrounds. */
  rowBounds?: GPUData<'float32x4'>;
  rowCount: number;
  rowIndexBase: number;
};

/**
 * The shared atlas a label layer reads. `GPUTextResources` satisfies it; so does a variant whose
 * atlas texture carries a mip chain, which bitmap labels need to match native minification.
 */
export type GPUPointLabelResources = {
  readonly device: Device;
  readonly fontAtlas: FontAtlas;
  readonly atlasTexture: DynamicTexture;
};

/** Prepared labels borrow all glyph, atlas, point and per-row styling resources. */
export type GPUPointLabelLayerProps = Omit<LayerProps, 'data'> &
  GPUBrushingProps & {
    batches: readonly GPUPointLabelBatch[];
    resources: GPUPointLabelResources;
    getPosition: GPUVector<'float32x2' | 'float32x3'>;
    /**
     * Optional fp64 low part of `getPosition`: same format and chunk boundaries. Defaults to zero.
     */
    getPositionLow?: GPUVector<'float32x2' | 'float32x3'>;
    selectionMask?: GPUVector<'uint32'>;
    getRadius?: number | GPUVector<'float32'>;
    getColor?: Color;
    /** Effective CSS text size, after the adapter applies native size multipliers. */
    size?: number;
    /** Font-atlas rasterization size; glyph metrics retain floating-point advances. */
    fontSize?: number;
    textAnchor?: 'start' | 'middle' | 'end';
    alignmentBaseline?: 'top' | 'center' | 'bottom';
    /** Multiplies each source point radius into native screen-pixel label offsets. */
    radiusPixelScale?: number;
    radiusPadding?: number;
    /** Native outline width ratio (after the Kepler multiplier); needs an SDF font atlas. */
    outlineWidth?: number;
    outlineColor?: Color;
    /** Draws one padded rectangle per label row; needs batches with `rowBounds`. */
    background?: boolean;
    backgroundColor?: Color;
    /** Pixels: [horizontal, vertical] or [left, top, right, bottom]. */
    backgroundPadding?: readonly [number, number] | readonly [number, number, number, number];
    /** Pixels; rounds the background rectangle corners (clamped to half the smaller side). */
    backgroundBorderRadius?: number;
    /** Additional native vertical offset, before applying the effective text-size multiplier. */
    verticalPadding?: number;
    /**
     * Optional CPU-computed label visibility, one bit per source row (global row index, little-endian
     * within each word), for example from collision avoidance. Combined with `selectionMask`.
     * Rows beyond the supplied words are hidden. Replace the array identity to update it.
     */
    visibilityWords?: Uint32Array;
  };

type LabelBatch = {
  model: Model;
  backgroundModel?: Model;
  styleBuffer: Buffer;
  selection: GPUGeometrySelectionBinding;
  chunkIndex: number;
};
type LabelState = {
  batches: LabelBatch[];
  defaultRadius: Buffer | null;
  boundBatches: GPUPointLabelLayerProps['batches'];
  boundResources: GPUPointLabelResources;
  boundBrushing: boolean;
  boundBackground: boolean;
  visibilityConfig: Buffer | null;
  visibilityBuffer: Buffer | null;
  appliedVisibilityWords: Uint32Array | undefined;
};
const STYLE_BYTES = 176;

const LABEL_STYLE_WGSL = /* wgsl */ `
struct LabelStyle {
  color: vec4<f32>,
  size: f32, radius: f32, radiusPixelScale: f32, fontSize: f32,
  rowIndexBase: u32, positionOffset: u32, positionStride: u32, positionComponents: u32,
  radiusOffset: u32, radiusStride: u32, useRadii: u32, glyphLayoutOffset: u32,
  glyphRowsOffset: u32, anchorOffset: f32, baselineOffset: f32, verticalPadding: f32,
  offsetDirection: vec2<f32>, radiusPadding: f32, positionLowOffset: u32,
  renderMode: u32, sdfThreshold: f32, sdfSmoothing: f32, distanceRange: f32,
  outlineColor: vec4<f32>, backgroundColor: vec4<f32>, backgroundPadding: vec4<f32>,
  outlineThreshold: f32, rowBoundsOffset: u32, backgroundBorderRadius: f32, positionLowStride: u32,
};
@group(0) @binding(auto) var<uniform> labelStyle: LabelStyle;
`;

const LABEL_VISIBILITY_WGSL = /* wgsl */ `
struct LabelVisibilityConfig { enabled: u32, wordCount: u32, _padding0: u32, _padding1: u32 };
@group(0) @binding(auto) var<uniform> labelVisibilityConfig: LabelVisibilityConfig;
@group(0) @binding(auto) var<storage, read> labelVisibility: array<u32>;
fn isLabelRowVisible(rowIndex: u32) -> bool {
  if (labelVisibilityConfig.enabled == 0u) { return true; }
  let wordIndex = rowIndex >> 5u;
  if (wordIndex >= labelVisibilityConfig.wordCount) { return false; }
  return ((labelVisibility[wordIndex] >> (rowIndex & 31u)) & 1u) != 0u;
}
`;

const SHADER = /* wgsl */ `
${GPU_GLOBE_WGSL}
${GPU_GEOMETRY_SELECTION_SHADER}
${LABEL_VISIBILITY_WGSL}
@group(0) @binding(auto) var fontAtlasTexture: texture_2d_array<f32>;
@group(0) @binding(auto) var fontAtlasTextureSampler: sampler;
@group(0) @binding(auto) var<storage, read> labelGlyphLayouts: array<vec4<f32>>;
@group(0) @binding(auto) var<storage, read> labelGlyphRows: array<vec2<u32>>;
@group(0) @binding(auto) var<storage, read> labelPositions: array<f32>;
@group(0) @binding(auto) var<storage, read> labelPositionsLow: array<f32>;
@group(0) @binding(auto) var<storage, read> labelRadii: array<f32>;
${LABEL_STYLE_WGSL}
struct FragmentInputs {
  @builtin(position) position: vec4<f32>,
  @location(0) atlasUV: vec2<f32>,
  @location(1) @interpolate(flat) atlasPage: u32,
  @location(2) @interpolate(flat) rowIndex: u32,
  @location(3) @interpolate(flat) visible: u32,
  ${WGSL_SHADOW_VARYING_FIELDS(4)}
};
@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> FragmentInputs {
  let corners = array<vec2<f32>, 6>(vec2<f32>(0.0,0.0),vec2<f32>(1.0,0.0),vec2<f32>(1.0,1.0),vec2<f32>(0.0,0.0),vec2<f32>(1.0,1.0),vec2<f32>(0.0,1.0));
  let corner = corners[vertexIndex];
  let rowAndPage = labelGlyphRows[labelStyle.glyphRowsOffset + instanceIndex];
  let localRow = rowAndPage.x;
  let rowIndex = labelStyle.rowIndexBase + localRow;
  let glyphLayout = labelGlyphLayouts[labelStyle.glyphLayoutOffset + instanceIndex * 2u];
  let frame = labelGlyphLayouts[labelStyle.glyphLayoutOffset + instanceIndex * 2u + 1u];
  let positionIndex = labelStyle.positionOffset + localRow * labelStyle.positionStride;
  var world = vec3<f32>(labelPositions[positionIndex], labelPositions[positionIndex + 1u], 0.0);
  if (labelStyle.positionComponents == 3u) { world.z = labelPositions[positionIndex + 2u]; }
  var positionLow = vec3<f32>(0.0);
  if (labelStyle.positionLowStride != 0u) {
    let lowIndex = labelStyle.positionLowOffset + localRow * labelStyle.positionLowStride;
    positionLow = vec3<f32>(labelPositionsLow[lowIndex], labelPositionsLow[lowIndex + 1u], 0.0);
    if (labelStyle.positionComponents == 3u) { positionLow.z = labelPositionsLow[lowIndex + 2u]; }
  }
  var radius = labelStyle.radius;
  if (labelStyle.useRadii != 0u) { radius = labelRadii[labelStyle.radiusOffset + localRow * labelStyle.radiusStride]; }
  let radiusPixels = radius * labelStyle.radiusPixelScale;
  let pixelOffset = labelStyle.offsetDirection * (vec2<f32>(radiusPixels + labelStyle.radiusPadding) + vec2<f32>(0.0, labelStyle.verticalPadding));
  let aligned = glyphLayout.xy + corner * frame.zw + vec2<f32>(glyphLayout.z * labelStyle.anchorOffset, glyphLayout.w * labelStyle.baselineOffset);
  let pixelPosition = aligned * (labelStyle.size / labelStyle.fontSize) + pixelOffset;
  geometry.worldPosition = world;
  let projection = project_position_to_clipspace_and_commonspace(world, positionLow, vec3<f32>(0.0));
  geometry.position = projection.commonPosition;
  var output: FragmentInputs;
  // Like Deck, the shadow map swap happens on the anchor before the pixel offset is added.
  output.position = shadow_setVertexPosition(projection.commonPosition, project_globe_billboard_clipspace(projection.clipPosition, projection.commonPosition.xyz));
  output.shadowPosition0 = shadow_getVertexPosition(0u, projection.commonPosition);
  output.shadowPosition1 = shadow_getVertexPosition(1u, projection.commonPosition);
  output.position = vec4<f32>(output.position.xy + project_pixel_size_to_clipspace(vec2<f32>(pixelPosition.x, -pixelPosition.y)), output.position.zw);
  // GlobeView has no occluder: hide whole labels anchored on the far hemisphere.
  if (gpuGlobeIsHidden(world)) { output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0); }
  output.atlasUV = (frame.xy + corner * frame.zw) / vec2<f32>(textureDimensions(fontAtlasTexture));
  output.atlasPage = rowAndPage.y;
  output.rowIndex = rowIndex;
  output.visible = select(0u, 1u, isGeometryRowSelected(rowIndex) && isLabelRowVisible(rowIndex) && labelStyle.size > 0.0);
  return output;
}
${makeTextGlyphAlphaWgsl({functionName: 'getTextGlyphOutlineAlpha', textureCoordinate: 'inputs.atlasUV', atlasPage: 'inputs.atlasPage', settings: {renderMode: {expression: 'labelStyle.renderMode', kind: 'uint'}, sdfThreshold: 'labelStyle.outlineThreshold', sdfSmoothing: 'labelStyle.sdfSmoothing', msdfDistanceRange: 'labelStyle.distanceRange'}})}
${makeTextGlyphAlphaWgsl({textureCoordinate: 'inputs.atlasUV', atlasPage: 'inputs.atlasPage', settings: {renderMode: {expression: 'labelStyle.renderMode', kind: 'uint'}, sdfThreshold: 'labelStyle.sdfThreshold', sdfSmoothing: 'labelStyle.sdfSmoothing', msdfDistanceRange: 'labelStyle.distanceRange'}})}
@fragment fn fragmentMain(inputs: FragmentInputs) -> @location(0) vec4<f32> {
  // Texture sampling and derivatives must precede selection/picking discards.
  let fillAlpha = getTextGlyphAlpha(inputs) * labelStyle.color.a * layer.opacity;
  let outlineAlpha = getTextGlyphOutlineAlpha(inputs) * labelStyle.outlineColor.a * layer.opacity * select(0.0, 1.0, labelStyle.outlineThreshold >= 0.0);
  let alpha = fillAlpha + outlineAlpha * (1.0 - fillAlpha);
  if (inputs.visible == 0u || alpha < 0.001) { discard; }
  if (picking.isActive > 0.5) { return vec4<f32>(picking_getPickingColorFromIndex(inputs.rowIndex), 1.0); }
  // Outline sits behind the fill, as native draws the outline pass first.
  let rgb = (labelStyle.color.rgb * fillAlpha + labelStyle.outlineColor.rgb * outlineAlpha * (1.0 - fillAlpha)) / alpha;
  return shadow_filterPremultiplied(vec4<f32>(rgb, alpha), inputs.position.z, inputs.shadowPosition0, inputs.shadowPosition1);
}`;

const BACKGROUND_SHADER = /* wgsl */ `
${GPU_GLOBE_WGSL}
${GPU_GEOMETRY_SELECTION_SHADER}
${LABEL_VISIBILITY_WGSL}
@group(0) @binding(auto) var<storage, read> labelRowBounds: array<vec4<f32>>;
@group(0) @binding(auto) var<storage, read> labelPositions: array<f32>;
@group(0) @binding(auto) var<storage, read> labelPositionsLow: array<f32>;
@group(0) @binding(auto) var<storage, read> labelRadii: array<f32>;
${LABEL_STYLE_WGSL}
struct FragmentInputs {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) rowIndex: u32,
  @location(1) @interpolate(flat) visible: u32,
  @location(2) localPixel: vec2<f32>,
  @location(3) @interpolate(flat) halfSize: vec2<f32>,
  ${WGSL_SHADOW_VARYING_FIELDS(4)}
};
@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> FragmentInputs {
  let corners = array<vec2<f32>, 6>(vec2<f32>(0.0,0.0),vec2<f32>(1.0,0.0),vec2<f32>(1.0,1.0),vec2<f32>(0.0,0.0),vec2<f32>(1.0,1.0),vec2<f32>(0.0,1.0));
  let corner = corners[vertexIndex];
  let localRow = instanceIndex;
  let rowIndex = labelStyle.rowIndexBase + localRow;
  let bounds = labelRowBounds[labelStyle.rowBoundsOffset + localRow];
  let positionIndex = labelStyle.positionOffset + localRow * labelStyle.positionStride;
  var world = vec3<f32>(labelPositions[positionIndex], labelPositions[positionIndex + 1u], 0.0);
  if (labelStyle.positionComponents == 3u) { world.z = labelPositions[positionIndex + 2u]; }
  var positionLow = vec3<f32>(0.0);
  if (labelStyle.positionLowStride != 0u) {
    let lowIndex = labelStyle.positionLowOffset + localRow * labelStyle.positionLowStride;
    positionLow = vec3<f32>(labelPositionsLow[lowIndex], labelPositionsLow[lowIndex + 1u], 0.0);
    if (labelStyle.positionComponents == 3u) { positionLow.z = labelPositionsLow[lowIndex + 2u]; }
  }
  var radius = labelStyle.radius;
  if (labelStyle.useRadii != 0u) { radius = labelRadii[labelStyle.radiusOffset + localRow * labelStyle.radiusStride]; }
  let radiusPixels = radius * labelStyle.radiusPixelScale;
  let pixelOffset = labelStyle.offsetDirection * (vec2<f32>(radiusPixels + labelStyle.radiusPadding) + vec2<f32>(0.0, labelStyle.verticalPadding));
  let origin = vec2<f32>(bounds.x * labelStyle.anchorOffset, bounds.y * labelStyle.baselineOffset);
  // Padding: left, top, right, bottom in screen pixels (y grows downward).
  let padding = labelStyle.backgroundPadding;
  let size = vec2<f32>(bounds.x, bounds.y) * (labelStyle.size / labelStyle.fontSize);
  let pixelPosition = origin * (labelStyle.size / labelStyle.fontSize) + pixelOffset
    + corner * size + vec2<f32>(mix(-padding.x, padding.z, corner.x), mix(-padding.y, padding.w, corner.y));
  geometry.worldPosition = world;
  let projection = project_position_to_clipspace_and_commonspace(world, positionLow, vec3<f32>(0.0));
  geometry.position = projection.commonPosition;
  var output: FragmentInputs;
  // Like Deck, the shadow map swap happens on the anchor before the pixel offset is added.
  output.position = shadow_setVertexPosition(projection.commonPosition, project_globe_billboard_clipspace(projection.clipPosition, projection.commonPosition.xyz));
  output.shadowPosition0 = shadow_getVertexPosition(0u, projection.commonPosition);
  output.shadowPosition1 = shadow_getVertexPosition(1u, projection.commonPosition);
  output.position = vec4<f32>(output.position.xy + project_pixel_size_to_clipspace(vec2<f32>(pixelPosition.x, -pixelPosition.y)), output.position.zw);
  if (gpuGlobeIsHidden(world)) { output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0); }
  output.rowIndex = rowIndex;
  let paddedSize = size + vec2<f32>(padding.x + padding.z, padding.y + padding.w);
  output.localPixel = (corner - vec2<f32>(0.5)) * paddedSize;
  output.halfSize = paddedSize * 0.5;
  output.visible = select(0u, 1u, isGeometryRowSelected(rowIndex) && isLabelRowVisible(rowIndex) && labelStyle.size > 0.0 && bounds.x > 0.0);
  return output;
}
@fragment fn fragmentMain(inputs: FragmentInputs) -> @location(0) vec4<f32> {
  var alpha = labelStyle.backgroundColor.a * layer.opacity;
  if (inputs.visible == 0u || alpha < 0.001) { discard; }
  if (labelStyle.backgroundBorderRadius > 0.0) {
    // Rounded-rectangle signed distance in screen pixels; one pixel of antialiasing.
    let cornerRadius = min(labelStyle.backgroundBorderRadius, min(inputs.halfSize.x, inputs.halfSize.y));
    let cornerDelta = abs(inputs.localPixel) - inputs.halfSize + vec2<f32>(cornerRadius);
    let distance = length(max(cornerDelta, vec2<f32>(0.0))) + min(max(cornerDelta.x, cornerDelta.y), 0.0) - cornerRadius;
    alpha = alpha * clamp(0.5 - distance, 0.0, 1.0);
    if (alpha < 0.001) { discard; }
  }
  if (picking.isActive > 0.5) { return vec4<f32>(picking_getPickingColorFromIndex(inputs.rowIndex), 1.0); }
  return shadow_filterPremultiplied(vec4<f32>(labelStyle.backgroundColor.rgb, alpha), inputs.position.z, inputs.shadowPosition0, inputs.shadowPosition1);
}`;

/** Deck-projected billboard glyphs with per-source-row selection; default picking remains off. */
export class GPUPointLabelLayer extends Layer<GPUPointLabelLayerProps> {
  static override layerName = 'GPUPointLabelLayer';
  override getAttributeManager() {
    return null;
  }
  override initializeState({device}: LayerContext): void {
    assert(device.type === 'webgpu');
    this.validateInputs();
    const defaultRadius = device.createBuffer({data: new Float32Array([0]), usage: Buffer.STORAGE});
    const visibilityConfig = device.createBuffer({
      byteLength: 16,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const visibilityBuffer = device.createBuffer({
      byteLength: 4,
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    const batches: LabelBatch[] = [];
    try {
      for (const [chunkIndex, batch] of this.props.batches.entries()) {
        if (!batch.glyphRows.length) continue;
        const selection = new GPUGeometrySelectionBinding(device, `${this.id}-${chunkIndex}`);
        const styleBuffer = device.createBuffer({
          byteLength: STYLE_BYTES,
          usage: Buffer.UNIFORM | Buffer.COPY_DST
        });
        try {
          const model = new Model(device, {
            ...this.getShaders({
              modules: [
                project32,
                color,
                picking,
                wgslShadow,
                ...(this.props.brushingEnabled ? [gpuBrushing] : [])
              ],
              // Hide each label with its anchor row, as native label layers share the point brush.
              source: replaceGPUBrushingMarker(
                SHADER,
                Boolean(this.props.brushingEnabled),
                'isGeometryRowSelected(rowIndex) &&',
                'isGeometryRowSelected(rowIndex) && brushing_isPointInRange(world.xy) &&'
              )
            }),
            id: `${this.id}-${chunkIndex}`,
            topology: 'triangle-list',
            vertexCount: 6,
            instanceCount: batch.glyphRows.length,
            bindings: {
              labelStyle: styleBuffer,
              labelGlyphLayouts: getGPUDataBuffer(batch.glyphLayouts),
              labelGlyphRows: getGPUDataBuffer(batch.glyphRows),
              labelPositions: getGPUDataBuffer(this.props.getPosition.data[chunkIndex]),
              labelPositionsLow: this.props.getPositionLow
                ? getGPUDataBuffer(this.props.getPositionLow.data[chunkIndex])
                : defaultRadius,
              labelRadii: defaultRadius,
              labelVisibilityConfig: visibilityConfig,
              labelVisibility: visibilityBuffer,
              fontAtlasTexture: this.props.resources.atlasTexture,
              ...selection.update(this.props.selectionMask, chunkIndex, batch.rowIndexBase)
            }
          });
          let backgroundModel: Model | undefined;
          if (this.props.background && batch.rowBounds) {
            backgroundModel = new Model(device, {
              ...this.getShaders({
                modules: [
                  project32,
                  color,
                  picking,
                  wgslShadow,
                  ...(this.props.brushingEnabled ? [gpuBrushing] : [])
                ],
                source: replaceGPUBrushingMarker(
                  BACKGROUND_SHADER,
                  Boolean(this.props.brushingEnabled),
                  'isGeometryRowSelected(rowIndex) &&',
                  'isGeometryRowSelected(rowIndex) && brushing_isPointInRange(world.xy) &&'
                )
              }),
              id: `${this.id}-background-${chunkIndex}`,
              topology: 'triangle-list',
              vertexCount: 6,
              instanceCount: batch.rowCount,
              bindings: {
                labelStyle: styleBuffer,
                labelRowBounds: getGPUDataBuffer(batch.rowBounds),
                labelPositions: getGPUDataBuffer(this.props.getPosition.data[chunkIndex]),
                labelPositionsLow: this.props.getPositionLow
                  ? getGPUDataBuffer(this.props.getPositionLow.data[chunkIndex])
                  : defaultRadius,
                labelRadii: defaultRadius,
                labelVisibilityConfig: visibilityConfig,
                labelVisibility: visibilityBuffer,
                ...selection.update(this.props.selectionMask, chunkIndex, batch.rowIndexBase)
              }
            });
          }
          batches.push({model, backgroundModel, styleBuffer, selection, chunkIndex});
        } catch (error) {
          selection.destroy();
          styleBuffer.destroy();
          throw error;
        }
      }
    } catch (error) {
      for (const batch of batches) this.destroyBatch(batch);
      defaultRadius.destroy();
      visibilityConfig.destroy();
      visibilityBuffer.destroy();
      throw error;
    }
    this.setState({
      batches,
      defaultRadius,
      boundBatches: this.props.batches,
      boundResources: this.props.resources,
      boundBrushing: Boolean(this.props.brushingEnabled),
      boundBackground: Boolean(this.props.background),
      visibilityConfig,
      visibilityBuffer,
      appliedVisibilityWords: undefined
    } satisfies LabelState);
  }
  override updateState(_parameters: UpdateParameters<this>): void {
    this.validateInputs();
    const state = this.state as LabelState;
    if (
      state.boundBatches !== this.props.batches ||
      state.boundResources !== this.props.resources ||
      state.boundBrushing !== Boolean(this.props.brushingEnabled) ||
      state.boundBackground !== Boolean(this.props.background)
    ) {
      this.destroyModels();
      this.initializeState(this.context);
    }
  }
  override getModels(): Model[] {
    return (
      (this.state as LabelState | undefined)?.batches.flatMap(batch =>
        batch.backgroundModel ? [batch.backgroundModel, batch.model] : [batch.model]
      ) ?? []
    );
  }
  override draw({renderPass}: {renderPass: RenderPass}): void {
    this.validateInputs();
    if (!this.props.resources.atlasTexture.isReady) return;
    const state = this.state as LabelState;
    this.applyVisibility(state);
    ensureWGSLShadowInputs(this.getModels(), this.context.device);
    for (const batch of state.batches) {
      const source = this.props.batches[batch.chunkIndex];
      const position = this.props.getPosition.data[batch.chunkIndex];
      const positionLow = this.props.getPositionLow?.data[batch.chunkIndex];
      const radii =
        typeof this.props.getRadius === 'object'
          ? this.props.getRadius.data[batch.chunkIndex]
          : undefined;
      const values = new ArrayBuffer(STYLE_BYTES);
      const floats = new Float32Array(values);
      const integers = new Uint32Array(values);
      const labelColor = this.props.getColor ?? [255, 255, 255, 255];
      floats.set([
        labelColor[0] / 255,
        labelColor[1] / 255,
        labelColor[2] / 255,
        (labelColor[3] ?? 255) / 255
      ]);
      floats.set(
        [
          this.props.size ?? 12,
          typeof this.props.getRadius === 'number' ? this.props.getRadius : 0,
          this.props.radiusPixelScale ?? 0,
          this.props.fontSize ?? 64
        ],
        4
      );
      integers.set(
        [
          source.rowIndexBase,
          position.byteOffset / 4,
          position.byteStride / 4,
          this.props.getPosition.format === 'float32x3' ? 3 : 2,
          radii ? radii.byteOffset / 4 : 0,
          radii ? radii.byteStride / 4 : 1,
          radii ? 1 : 0,
          source.glyphLayouts.byteOffset / 16,
          source.glyphRows.byteOffset / 8
        ],
        8
      );
      const anchor = this.props.textAnchor ?? 'middle';
      const alignment = this.props.alignmentBaseline ?? 'center';
      floats.set(
        [
          anchor === 'start' ? 0 : anchor === 'end' ? -1 : -0.5,
          alignment === 'top' ? 0 : alignment === 'bottom' ? -1 : -0.5,
          this.props.verticalPadding ?? 0,
          anchor === 'start' ? 1 : anchor === 'end' ? -1 : 0,
          alignment === 'top' ? -1 : alignment === 'bottom' ? 1 : 0,
          this.props.radiusPadding ?? 0,
          0
        ],
        17
      );
      const settings = this.props.resources.fontAtlas.renderSettings;
      integers[24] = settings.mode === 'msdf' ? 2 : settings.mode === 'sdf' ? 1 : 0;
      floats.set([settings.threshold, settings.smoothing, settings.distanceRange ?? 0], 25);
      const outlineColor = this.props.outlineColor ?? [0, 0, 0, 255];
      const backgroundColor = this.props.backgroundColor ?? [0, 0, 200, 255];
      const padding = this.props.backgroundPadding ?? [0, 0, 0, 0];
      const [paddingLeft, paddingTop, paddingRight, paddingBottom] =
        padding.length === 2 ? [padding[0], padding[1], padding[0], padding[1]] : padding;
      const outlineWidth = this.props.outlineWidth ?? 0;
      floats.set(
        [
          outlineColor[0] / 255,
          outlineColor[1] / 255,
          outlineColor[2] / 255,
          (outlineColor[3] ?? 255) / 255,
          backgroundColor[0] / 255,
          backgroundColor[1] / 255,
          backgroundColor[2] / 255,
          (backgroundColor[3] ?? 255) / 255,
          paddingLeft,
          paddingTop,
          paddingRight,
          paddingBottom,
          // Native Deck: max(smoothing, threshold * (1 - outlineWidth)); negative disables.
          outlineWidth > 0 && settings.mode === 'sdf'
            ? Math.max(settings.smoothing, settings.threshold * (1 - outlineWidth))
            : -1
        ],
        28
      );
      integers[23] = positionLow ? positionLow.byteOffset / 4 : 0;
      integers[43] = positionLow ? positionLow.byteStride / 4 : 0;
      integers[41] = source.rowBounds ? source.rowBounds.byteOffset / 16 : 0;
      floats[42] = this.props.backgroundBorderRadius ?? 0;
      batch.styleBuffer.write(new Uint8Array(values));
      batch.model.setBindings({
        labelPositions: getGPUDataBuffer(position),
        labelPositionsLow: positionLow ? getGPUDataBuffer(positionLow) : state.defaultRadius!,
        labelRadii: radii ? getGPUDataBuffer(radii) : state.defaultRadius!,
        labelVisibilityConfig: state.visibilityConfig!,
        labelVisibility: state.visibilityBuffer!,
        ...batch.selection.update(this.props.selectionMask, batch.chunkIndex, source.rowIndexBase)
      });
      setGPUBrushingProps(batch.model, this.context, this.props);
      if (batch.backgroundModel && source.rowBounds) {
        batch.backgroundModel.setBindings({
          labelPositions: getGPUDataBuffer(position),
          labelPositionsLow: positionLow ? getGPUDataBuffer(positionLow) : state.defaultRadius!,
          labelRadii: radii ? getGPUDataBuffer(radii) : state.defaultRadius!,
          labelVisibilityConfig: state.visibilityConfig!,
          labelVisibility: state.visibilityBuffer!,
          ...batch.selection.update(this.props.selectionMask, batch.chunkIndex, source.rowIndexBase)
        });
        setGPUBrushingProps(batch.backgroundModel, this.context, this.props);
        batch.backgroundModel.draw(renderPass);
      }
      batch.model.draw(renderPass);
    }
  }
  override getPickingInfo({info}: {info: PickingInfo}): PickingInfo {
    const result = info as GPUVectorLayerPickingInfo;
    result.gpuVector = getGPUVectorPickingProvenance(this.props.getPosition, result.index);
    return result;
  }
  override finalizeState(context: LayerContext): void {
    this.destroyModels();
    super.finalizeState(context);
  }
  private validateInputs(): void {
    const {device} = this.context;
    const {getPosition, getPositionLow, batches, resources} = this.props;
    assert(resources.device === device && batches.length === getPosition.data.length);
    assert(getPosition.format === 'float32x2' || getPosition.format === 'float32x3');
    validateGPUVectorPickingCapacity(getPosition.length, Boolean(this.props.pickable));
    validateGPUGeometrySelection(device, getPosition, this.props.selectionMask);
    const radii = typeof this.props.getRadius === 'object' ? this.props.getRadius : undefined;
    validateGPUGeometryChunks(getPosition, [radii, getPositionLow]);
    if (radii) assert(radii.format === 'float32');
    // The shader reads both position parts with the component count of getPosition.
    if (getPositionLow) assert(getPositionLow.format === getPosition.format);
    let rowIndexBase = 0;
    for (const [index, batch] of batches.entries()) {
      assert(
        batch.rowCount === getPosition.data[index].length && batch.rowIndexBase === rowIndexBase
      );
      assert(batch.glyphLayouts.format === 'float32x4' && batch.glyphRows.format === 'uint32x2');
      assert(batch.glyphLayouts.length === batch.glyphRows.length * 2);
      assert(batch.glyphLayouts.byteStride === 16 && batch.glyphRows.byteStride === 8);
      assert(batch.glyphLayouts.byteOffset % 16 === 0 && batch.glyphRows.byteOffset % 8 === 0);
      for (const chunk of [
        batch.glyphLayouts,
        batch.glyphRows,
        getPosition.data[index],
        radii?.data[index],
        getPositionLow?.data[index]
      ]) {
        if (!chunk) continue;
        const buffer = getGPUDataBuffer(chunk);
        assert(buffer.device === device && (buffer.usage & Buffer.STORAGE) !== 0);
        assert(buffer.byteLength <= device.limits.maxStorageBufferBindingSize);
        assert(chunk.byteOffset % 4 === 0 && chunk.byteStride % 4 === 0);
      }
      rowIndexBase += batch.rowCount;
    }
  }
  /** Uploads the collision visibility words when their identity changes; grows the buffer on demand. */
  private applyVisibility(state: LabelState): void {
    const words = this.props.visibilityWords;
    if (words === state.appliedVisibilityWords) return;
    state.appliedVisibilityWords = words;
    const requiredBytes = Math.max(4, words ? words.byteLength : 0);
    if (state.visibilityBuffer!.byteLength < requiredBytes) {
      state.visibilityBuffer!.destroy();
      state.visibilityBuffer = this.context.device.createBuffer({
        byteLength: requiredBytes,
        usage: Buffer.STORAGE | Buffer.COPY_DST
      });
    }
    if (words?.length) state.visibilityBuffer!.write(words);
    state.visibilityConfig!.write(new Uint32Array([words ? 1 : 0, words?.length ?? 0, 0, 0]));
  }
  private destroyBatch(batch: LabelBatch) {
    batch.model.destroy();
    batch.backgroundModel?.destroy();
    batch.styleBuffer.destroy();
    batch.selection.destroy();
  }
  private destroyModels() {
    const state = this.state as LabelState | undefined;
    state?.batches.forEach(batch => this.destroyBatch(batch));
    state?.defaultRadius?.destroy();
    state?.visibilityConfig?.destroy();
    state?.visibilityBuffer?.destroy();
    if (state)
      this.setState({
        batches: [],
        defaultRadius: null,
        visibilityConfig: null,
        visibilityBuffer: null,
        appliedVisibilityWords: undefined
      });
  }
}
