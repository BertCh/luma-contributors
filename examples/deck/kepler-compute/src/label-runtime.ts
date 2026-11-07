// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import {assert, Buffer, type Device} from '@luma.gl/core';
import {GPUData, type GPUVector} from '@luma-contributors/graph/gpu-data';
import {buildBitmapFontAtlas, buildSdfFontAtlas, GPUTextResources} from '@luma.gl/engine/text';
import {getDistanceScales, zoomToScale} from '@math.gl/web-mercator';
import type {GPUPointLabelBatch, GPUPointLabelLayerProps} from '@deck.gl-community/gpu-layers';
import type {KeplerDatasetRegistry, KeplerMemoryReservation} from '@luma-contributors/kepler';
import {getPointLabelFormatKey} from './label-format';
import {preparePointLabelLayout, type PreparedPointLabelSource} from './label-source';
import {
  computeLabelVisibilityWords,
  getAnchorValidityWords,
  type LabelAnchorSource
} from './label-collision-runtime';

export type NativePointLabelConfig = {
  field?: {name?: string; type?: string; displayFormat?: unknown} | null;
  format?: unknown;
  color?: readonly number[];
  size?: number;
  weight?: number;
  anchor?: string;
  alignment?: string;
  /** Raw Kepler outlineWidth; resolve() applies the native TEXT_OUTLINE_MULTIPLIER. */
  outlineWidth?: number;
  outlineColor?: readonly number[] | null;
  background?: boolean;
  backgroundColor?: readonly number[] | null;
  backgroundPadding?: readonly [number, number] | readonly [number, number, number, number];
  /** Extension beyond native: rounded background corners in pixels. */
  backgroundBorderRadius?: number;
  collisionEnabled?: boolean;
};
export type NativePointLabelMapState = {
  longitude: number;
  latitude: number;
  zoom: number;
  scale?: number;
  globe?: {enabled?: boolean};
  /** Viewport size and orientation; required for collision avoidance. */
  width?: number;
  height?: number;
  pitch?: number;
  bearing?: number;
};
export type PointLabelResolution =
  | {layerProps: Omit<GPUPointLabelLayerProps, 'id'>}
  | {diagnostic: string};
export type PointLabelRuntimeProps = {
  source: PreparedPointLabelSource;
  positions: GPUVector<'float32x2' | 'float32x3'>;
  fontWeight?: number;
  /** Display format the source text was prepared with; resolve() requires the same format. */
  formatKey?: string;
  /** Rasterize a signed-distance atlas so SDF outlines can render (native fontSettings.sdf). */
  sdf?: boolean;
  /** Upload per-row paragraph bounds so label backgrounds can render. */
  rowBounds?: boolean;
  /** GPU atlas, glyph payload and one borrowing renderer's fixed control buffers. */
  workingMemoryBudgetBytes: number;
  memoryRegistry?: KeplerDatasetRegistry;
  /**
   * CPU anchors (longitude/latitude per row). Enables collision avoidance and hides rows whose
   * anchor is invalid. Costs 8 bytes per row plus 16 bytes of retained row bounds.
   */
  anchors?: LabelAnchorSource;
};
export type ResolvePointLabelOptions = {
  label: NativePointLabelConfig;
  selectionMask: GPUVector<'uint32'>;
  radius?: number | GPUVector<'float32'>;
  radiusScale: number;
  mapState: NativePointLabelMapState;
};

function toRgba(color: readonly number[]): [number, number, number, number] {
  return [color[0], color[1], color[2], color[3] ?? 255];
}

/** Explicit field/font variant preparation; resolve() never reads source rows or reshapes text. */
export function makePointLabelRuntime(
  device: Device,
  props: PointLabelRuntimeProps
): PointLabelRuntime {
  return new PointLabelRuntime(device, props);
}

/** Everything resolvePointLabelLayerProps needs from a prepared label variant. */
export type PointLabelLayerContext = {
  fieldName: string;
  fontWeight: number;
  formatKey: string;
  sdf: boolean;
  hasRowBounds: boolean;
  hasCollisionAnchors: boolean;
  batches: readonly GPUPointLabelBatch[];
  resources: GPUTextResources;
  positions: GPUVector<'float32x2' | 'float32x3'>;
};

/** Validates the native label controls and builds the label layer props for one prepared variant. */
export function resolvePointLabelLayerProps(
  context: PointLabelLayerContext,
  options: ResolvePointLabelOptions,
  getVisibilityWords?: (style: {
    anchor: 'start' | 'middle' | 'end';
    alignment: 'top' | 'center' | 'bottom';
    size: number;
    radiusPixelScale: number;
  }) => Uint32Array | undefined
): PointLabelResolution {
  const {label, mapState} = options;
  if (!label.field?.name || label.field.name !== context.fieldName)
    return {diagnostic: 'Point label field requires explicit source preparation'};
  if ((label.weight ?? 400) !== context.fontWeight)
    return {diagnostic: 'Point label font weight requires explicit variant preparation'};
  const type = label.field.type?.toLowerCase();
  // Timestamps are formatted during explicit admission; dates keep their diagnostic.
  if (type === 'date' || type === 'datetime')
    return {diagnostic: 'Point labels do not yet support date fields'};
  if (getPointLabelFormatKey(label) !== context.formatKey)
    return {diagnostic: 'Point label display format requires explicit source preparation'};
  if (label.outlineWidth && !context.sdf)
    return {diagnostic: 'Point label SDF outlines require explicit SDF variant preparation'};
  if (label.background && !context.hasRowBounds)
    return {diagnostic: 'Point label backgrounds require explicit row-bounds preparation'};
  if (label.collisionEnabled && !context.hasCollisionAnchors)
    return {diagnostic: 'Point label collision requires explicit CPU anchor preparation'};
  const anchor = label.anchor ?? 'start';
  const alignment = label.alignment ?? 'center';
  if (anchor !== 'start' && anchor !== 'middle' && anchor !== 'end')
    return {diagnostic: 'Unsupported point label anchor'};
  if (alignment !== 'top' && alignment !== 'center' && alignment !== 'bottom')
    return {diagnostic: 'Unsupported point label alignment'};
  const size = label.size ?? 18;
  if (!Number.isFinite(size) || size < 0 || !Number.isFinite(options.radiusScale))
    return {diagnostic: 'Invalid point label size or radius scale'};
  const color = label.color ?? [255, 255, 255];
  if (
    color.length < 3 ||
    color.length > 4 ||
    color.some(component => !Number.isFinite(component) || component < 0 || component > 255)
  )
    return {diagnostic: 'Invalid point label color'};
  if (
    ![mapState.longitude, mapState.latitude, mapState.zoom, mapState.scale ?? 1].every(
      Number.isFinite
    )
  )
    return {diagnostic: 'Invalid point label map scale'};
  const pixelsPerMeter =
    getDistanceScales(mapState).unitsPerMeter[0] * (mapState.scale ?? zoomToScale(mapState.zoom));
  const radiusPixelScale = options.radiusScale * pixelsPerMeter;
  const visibilityWords = getVisibilityWords?.({
    anchor,
    alignment,
    size: size * (2 / 3),
    radiusPixelScale
  });
  return {
    layerProps: {
      batches: context.batches,
      resources: context.resources,
      getPosition: context.positions,
      selectionMask: options.selectionMask,
      getRadius: options.radius ?? 0,
      getColor: [color[0], color[1], color[2], color[3] ?? 255],
      size: size * (2 / 3),
      fontSize: 64,
      textAnchor: anchor,
      alignmentBaseline: alignment,
      radiusPixelScale,
      radiusPadding: 20,
      ...(visibilityWords ? {visibilityWords} : {}),
      verticalPadding: size,
      ...(label.outlineWidth
        ? {
            outlineWidth: Math.min(1, label.outlineWidth * 5), // Native TEXT_OUTLINE_MULTIPLIER
            outlineColor: toRgba(label.outlineColor ?? [0, 0, 0, 255])
          }
        : {}),
      ...(label.background
        ? {
            background: true,
            backgroundColor: toRgba(label.backgroundColor ?? [0, 0, 200, 255]),
            backgroundPadding: label.backgroundPadding ?? [0, 0, 0, 0],
            ...(label.backgroundBorderRadius
              ? {backgroundBorderRadius: label.backgroundBorderRadius}
              : {})
          }
        : {}),
      // Kepler ordinary point-label sharedProps never enables picking or inherits point opacity.
      pickable: false,
      opacity: 1,
      parameters: {depthCompare: 'always', depthWriteEnabled: false, cullMode: 'none'}
    }
  };
}

/** One source/font variant. Retire borrowing Deck layers before submitting the destruction fence. */
export class PointLabelRuntime {
  readonly gpuByteLength: number;
  readonly cpuByteLength: number;
  readonly source: PreparedPointLabelSource;
  readonly fontWeight: number;
  readonly ready: Promise<void>;
  private readonly resources: GPUTextResources;
  private readonly batches: readonly GPUPointLabelBatch[];
  private readonly buffers: Buffer[] = [];
  private readonly reservations: KeplerMemoryReservation[] = [];
  private readonly deviceLoss: Device['lost'];
  private readonly rowBounds?: Float32Array;
  private readonly invalidAnchorWords?: Uint32Array;
  private collisionCache?: {key: string; words: Uint32Array};
  private current = true;
  private destruction?: Promise<void>;

  constructor(
    device: Device,
    private readonly props: PointLabelRuntimeProps
  ) {
    this.source = props.source;
    this.deviceLoss = device.lost;
    this.fontWeight = props.fontWeight ?? 400;
    if (!Number.isFinite(this.fontWeight) || this.fontWeight < 1 || this.fontWeight > 1000)
      throw new Error('Invalid label font weight');
    if (!Number.isSafeInteger(props.workingMemoryBudgetBytes) || props.workingMemoryBudgetBytes < 0)
      throw new Error('Invalid label GPU memory budget');
    if (props.source.characterSet.length > 512)
      throw new KeplerAllocationError('capacity', {
        message: 'Point label atlas exceeds the admitted 512-character vocabulary'
      });
    assert(
      props.positions.length === props.source.rowCount &&
        props.positions.data.length === props.source.chunks.length
    );
    props.positions.data.forEach((chunk, index) => {
      assert(
        chunk.length === props.source.chunks[index].rowCount && chunk.buffer.device === device
      );
    });
    const glyphBytes = props.source.glyphCount * 40;
    if (glyphBytes > props.workingMemoryBudgetBytes)
      throw new KeplerAllocationError('memory-budget', {
        message: 'Point label glyphs exceed their admitted GPU budget'
      });
    const atlasSettings = {
      fontFamily: 'Monaco, monospace',
      fontWeight: this.fontWeight,
      fontSize: 64,
      buffer: 4,
      characterSet: props.source.characterSet.length ? [...props.source.characterSet] : [' ']
    };
    const fontAtlas = props.sdf
      ? buildSdfFontAtlas(atlasSettings)
      : buildBitmapFontAtlas({cache: false, ...atlasSettings});
    if (
      fontAtlas.width > device.limits.maxTextureDimension2D ||
      fontAtlas.height > device.limits.maxTextureDimension2D
    )
      throw new Error('Point label atlas exceeds device texture limits');
    const atlasBytes = fontAtlas.width * fontAtlas.height * 4 * fontAtlas.pages.length;
    // Native TextLayer uses a full trilinear mip chain for bitmap minification.
    let atlasGPUBytes = 0;
    for (let width = fontAtlas.width, height = fontAtlas.height; ; ) {
      atlasGPUBytes += width * height * 4 * fontAtlas.pages.length;
      if (width === 1 && height === 1) break;
      width = Math.max(1, Math.floor(width / 2));
      height = Math.max(1, Math.floor(height / 2));
    }
    const batchBytes = props.source.chunks.reduce(
      (bytes, chunk) =>
        bytes +
        Math.max(16, chunk.codePoints.length * 32) +
        Math.max(8, chunk.codePoints.length * 8) +
        (props.rowBounds ? Math.max(16, chunk.rowCount * 16) : 0),
      0
    );
    const rendererBytes =
      4 + props.source.chunks.filter(chunk => chunk.codePoints.length).length * (112 + 32 + 4);
    this.gpuByteLength = atlasGPUBytes + batchBytes + rendererBytes;
    const anchorBytes = props.anchors
      ? props.anchors.positions.byteLength +
        (props.anchors.validity?.byteLength ?? 0) +
        props.source.rowCount * 16 +
        Math.ceil(props.source.rowCount / 32) * 8
      : 0;
    this.cpuByteLength = props.source.byteLength + atlasBytes + anchorBytes;
    if (this.gpuByteLength > props.workingMemoryBudgetBytes)
      throw new KeplerAllocationError('memory-budget', {
        message: 'Point label atlas and glyphs exceed their admitted GPU budget'
      });
    let resources: GPUTextResources | undefined;
    let transient: KeplerMemoryReservation | undefined;
    try {
      if (props.memoryRegistry) {
        this.reservations.push(
          props.memoryRegistry.reserveMemory(this.cpuByteLength, 'cpu', 'source')
        );
        this.reservations.push(
          props.memoryRegistry.reserveMemory(this.gpuByteLength, 'gpu', 'source')
        );
        transient = props.memoryRegistry.reserveMemory(glyphBytes, 'cpu', 'scratch');
      }
      const layouts = preparePointLabelLayout(props.source, fontAtlas);
      resources = new GPUTextResources(device, {
        fontAtlas,
        mipmaps: true,
        id: `point-label-${props.source.fieldName}`
      });
      const allocate = (data: Float32Array | Uint32Array, minimum: number) => {
        const buffer = device.createBuffer({
          ...(data.byteLength ? {data} : {byteLength: minimum}),
          usage: Buffer.STORAGE | Buffer.COPY_DST
        });
        this.buffers.push(buffer);
        return buffer;
      };
      if (props.anchors) {
        const flatBounds = new Float32Array(props.source.rowCount * 4);
        for (const layout of layouts) flatBounds.set(layout.rowBounds, layout.rowIndexBase * 4);
        this.rowBounds = flatBounds;
        if (props.anchors.validity)
          this.invalidAnchorWords = getAnchorValidityWords(
            props.source.rowCount,
            props.anchors.validity
          );
      }
      this.batches = layouts.map(layout => ({
        rowCount: layout.rowCount,
        rowIndexBase: layout.rowIndexBase,
        glyphLayouts: new GPUData({
          format: 'float32x4',
          buffer: allocate(layout.glyphLayouts, 16),
          length: layout.glyphCount * 2,
          ownsBuffer: false
        }),
        glyphRows: new GPUData({
          format: 'uint32x2',
          buffer: allocate(layout.glyphRows, 8),
          length: layout.glyphCount,
          ownsBuffer: false
        }),
        ...(props.rowBounds
          ? {
              rowBounds: new GPUData({
                format: 'float32x4' as const,
                buffer: allocate(layout.rowBounds, 16),
                length: layout.rowCount,
                ownsBuffer: false
              })
            }
          : {})
      }));
      this.resources = resources;
      this.ready = resources.atlasTexture.ready.then(
        () => undefined,
        error => {
          this.current = false;
          throw error;
        }
      );
      transient?.release();
      transient = undefined;
    } catch (error) {
      transient?.release();
      for (const buffer of this.buffers) buffer.destroy();
      resources?.destroy();
      this.reservations.forEach(reservation => reservation.release());
      throw error;
    }
    void this.deviceLoss?.then(() => {
      this.current = false;
    });
  }

  isCurrent(): boolean {
    return this.current;
  }

  resolve(options: ResolvePointLabelOptions): PointLabelResolution {
    if (!this.current)
      return {diagnostic: 'Point label resources have retired or the device was lost'};
    return resolvePointLabelLayerProps(
      {
        fieldName: this.source.fieldName,
        fontWeight: this.fontWeight,
        formatKey: this.props.formatKey ?? '',
        sdf: Boolean(this.props.sdf),
        hasRowBounds: Boolean(this.props.rowBounds),
        hasCollisionAnchors: Boolean(this.props.anchors),
        batches: this.batches,
        resources: this.resources,
        positions: this.props.positions
      },
      options,
      style => this.resolveVisibility(options, style)
    );
  }

  /**
   * Collision avoidance runs on the CPU against the current viewport, per label group, and is
   * cached until the viewport or label style changes. Without a viewport size labels stay unfiltered.
   * Variable per-row radii are treated as radius zero for placement.
   */
  private resolveVisibility(
    options: ResolvePointLabelOptions,
    style: {
      anchor: 'start' | 'middle' | 'end';
      alignment: 'top' | 'center' | 'bottom';
      size: number;
      radiusPixelScale: number;
    }
  ): Uint32Array | undefined {
    const {label, mapState} = options;
    const anchors = this.props.anchors;
    if (!anchors || !this.rowBounds) return undefined;
    const {width, height} = mapState;
    // The CPU projector is Web Mercator; globe views keep labels unfiltered.
    if (!label.collisionEnabled || !(width && height) || mapState.globe?.enabled)
      return this.invalidAnchorWords;
    const radius = typeof options.radius === 'number' ? options.radius : 0;
    const key = [
      mapState.longitude,
      mapState.latitude,
      mapState.zoom,
      width,
      height,
      mapState.pitch ?? 0,
      mapState.bearing ?? 0,
      style.anchor,
      style.alignment,
      style.size,
      style.radiusPixelScale,
      radius
    ].join(',');
    if (this.collisionCache?.key === key) return this.collisionCache.words;
    const words = computeLabelVisibilityWords({
      rowCount: this.source.rowCount,
      anchors,
      rowBounds: this.rowBounds,
      viewport: {
        longitude: mapState.longitude,
        latitude: mapState.latitude,
        zoom: mapState.zoom,
        width,
        height,
        pitch: mapState.pitch,
        bearing: mapState.bearing
      },
      size: style.size,
      fontSize: 64,
      anchor: style.anchor,
      alignment: style.alignment,
      radius,
      radiusPixelScale: style.radiusPixelScale,
      radiusPadding: 20,
      verticalPadding: label.size ?? 18
    });
    this.collisionCache = {key, words};
    return words;
  }

  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (!this.destruction) {
      this.current = false;
      const fence = this.deviceLoss
        ? Promise.race([Promise.resolve(completion), this.deviceLoss])
        : Promise.resolve(completion);
      this.destruction = fence
        .then(() => undefined)
        .finally(() => {
          this.buffers.forEach(buffer => buffer.destroy());
          this.resources.destroy();
          this.reservations.forEach(reservation => reservation.release());
        });
    }
    return this.destruction;
  }
}
