// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import type {KeplerDatasetRegistry, KeplerMemoryReservation} from '@luma-contributors/kepler';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, type GPUVector} from '@luma.gl/graph/gpu-data';
import {buildBitmapFontAtlas, buildSdfFontAtlas, GPUTextResources} from '@luma.gl/engine/text';
import type {FontAtlas} from '@luma.gl/engine/text';
import type {GPUPointLabelBatch} from '@deck.gl-community/gpu-layers';
import {
  resolvePointLabelLayerProps,
  type PointLabelResolution,
  type ResolvePointLabelOptions
} from './label-runtime';
import {createGpuLabelCollision, type GpuLabelCollision} from './label-collision-gpu';
import {
  STREAMED_LABEL_LIMITS,
  collectVisibleRows,
  layoutVisibleLabels,
  measureLabelBounds,
  scanLabelVocabulary,
  type StreamedLabelTextChunk
} from './label-streamed';

export type StreamedPointLabelRuntimeProps = {
  chunks: readonly StreamedLabelTextChunk[];
  fieldName: string;
  /** Layer positions; chunks must align one to one with the label chunks. */
  positions: GPUVector<'float32x2' | 'float32x3'>;
  /** Collision anchors (the positions unless the geometry has separate anchors). */
  collisionPositions: GPUVector<'float32x2' | 'float32x3'>;
  validity?: GPUVector<'uint32'>;
  fontWeight?: number;
  formatKey?: string;
  sdf?: boolean;
  memoryRegistry?: KeplerDatasetRegistry;
  /** Called when a placement finished and the layer should be rebuilt. */
  onChange: () => void;
};

export type StreamedPointLabelStats = {
  rowCount: number;
  vocabularySize: number;
  distinctCharacters: number;
  /** Rows hidden because a character did not fit the shared atlas vocabulary. */
  hiddenRows: number;
  visibleLabels: number;
  visibleGlyphs: number;
  /** True when the visible-glyph budget dropped accepted labels. */
  truncated: boolean;
  collisionRounds: number;
  collisionAccepted: number;
};

type Generation = {
  batches: GPUPointLabelBatch[];
  buffers: Buffer[];
};

type PlacementUpdate = Parameters<GpuLabelCollision['update']>[0];

/**
 * One streamed label variant for sources above the glyph cap. The atlas and per-row bounds are
 * built once; every viewport change reruns GPU collision (converged rounds) and replaces only the
 * visible rows' glyph instances. Two generations stay alive so Deck can finish the previous frame.
 */
export class StreamedPointLabelRuntime {
  readonly ready: Promise<void>;
  readonly stats: StreamedPointLabelStats;
  private readonly fontWeight: number;
  private readonly rowCount: number;
  private readonly reservations: KeplerMemoryReservation[] = [];
  private readonly boundsBuffers: Buffer[] = [];
  private readonly deviceLoss: Device['lost'];
  private resources?: GPUTextResources;
  private fontAtlas?: FontAtlas;
  private collision?: GpuLabelCollision;
  private generations: Generation[] = [];
  private placementKey?: string;
  private queuedPlacement?: {key: string; update: PlacementUpdate};
  private placing = false;
  private current = true;
  private destruction?: Promise<void>;

  constructor(
    private readonly device: Device,
    private readonly props: StreamedPointLabelRuntimeProps
  ) {
    this.fontWeight = props.fontWeight ?? 400;
    this.rowCount = props.chunks.reduce((total, chunk) => total + chunk.rowCount, 0);
    this.deviceLoss = device.lost;
    this.stats = {
      rowCount: this.rowCount,
      vocabularySize: 0,
      distinctCharacters: 0,
      hiddenRows: 0,
      visibleLabels: 0,
      visibleGlyphs: 0,
      truncated: false,
      collisionRounds: 0,
      collisionAccepted: 0
    };
    if (this.rowCount > STREAMED_LABEL_LIMITS.maximumRows)
      throw new KeplerAllocationError('capacity', {
        message: 'Streamed point labels exceed the GPU collision row cap.'
      });
    if (
      props.positions.data.length !== props.chunks.length ||
      props.collisionPositions.data.length !== props.chunks.length ||
      props.chunks.some(
        (chunk, index) =>
          props.positions.data[index].length !== chunk.rowCount ||
          props.collisionPositions.data[index].length !== chunk.rowCount
      )
    )
      throw new Error('Streamed point label chunks do not match the resident positions.');
    this.ready = this.initialize();
    // Callers attach their own handlers; this keeps a retired runtime from reporting twice.
    this.ready.catch(() => undefined);
    void this.deviceLoss?.then(() => {
      this.current = false;
    });
  }

  isCurrent(): boolean {
    return this.current;
  }

  private assertCurrent(): void {
    // Stale: the label variant was retired while it was still streaming its source.
    if (!this.current) throw new Error('Streamed point labels are stale.');
  }

  private async initialize(): Promise<void> {
    const {props} = this;
    const registry = props.memoryRegistry;
    const boundsBytes = this.rowCount * 16;
    const glyphBytes = STREAMED_LABEL_LIMITS.maximumVisibleGlyphs * 40;
    if (registry) {
      this.reservations.push(registry.reserveMemory(boundsBytes + glyphBytes * 2, 'gpu', 'source'));
    }
    const isCurrent = () => this.current;
    const vocabulary = await scanLabelVocabulary(props.chunks, {isCurrent});
    this.stats.vocabularySize = vocabulary.characterSet.length;
    this.stats.distinctCharacters = vocabulary.distinctCount;
    const atlasSettings = {
      fontFamily: 'Monaco, monospace',
      fontWeight: this.fontWeight,
      fontSize: 64,
      buffer: 4,
      characterSet: vocabulary.characterSet.length ? vocabulary.characterSet : [' ']
    };
    const fontAtlas = props.sdf
      ? buildSdfFontAtlas(atlasSettings)
      : buildBitmapFontAtlas({cache: false, ...atlasSettings});
    if (
      fontAtlas.width > this.device.limits.maxTextureDimension2D ||
      fontAtlas.height > this.device.limits.maxTextureDimension2D
    )
      throw new Error('Point label atlas exceeds device texture limits');
    // Native TextLayer uses a full trilinear mip chain for bitmap minification.
    let atlasBytes = 0;
    for (let width = fontAtlas.width, height = fontAtlas.height; ; ) {
      atlasBytes += width * height * 4 * fontAtlas.pages.length;
      if (width === 1 && height === 1) break;
      width = Math.max(1, Math.floor(width / 2));
      height = Math.max(1, Math.floor(height / 2));
    }
    if (registry) this.reservations.push(registry.reserveMemory(atlasBytes, 'gpu', 'source'));
    this.fontAtlas = fontAtlas;
    this.assertCurrent();
    this.resources = new GPUTextResources(this.device, {
      fontAtlas,
      mipmaps: true,
      id: `streamed-point-label-${props.fieldName}`
    });
    for (const chunk of props.chunks) {
      this.boundsBuffers.push(
        this.device.createBuffer({
          id: `streamed-point-label-bounds-${chunk.id}`,
          byteLength: Math.max(16, chunk.rowCount * 16),
          usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
        })
      );
    }
    const measured = await measureLabelBounds(
      props.chunks,
      fontAtlas,
      new Set(vocabulary.characterSet),
      (chunkIndex, firstRow, bounds) => this.boundsBuffers[chunkIndex].write(bounds, firstRow * 16),
      {isCurrent}
    );
    this.stats.hiddenRows = measured.hiddenRows;
    await this.resources.atlasTexture.ready;
    this.assertCurrent();
    this.collision = createGpuLabelCollision({
      device: this.device,
      positions: props.collisionPositions,
      rowBounds: this.boundsBuffers.map(
        (buffer, index) =>
          new GPUData({
            format: 'float32x4',
            buffer,
            length: props.chunks[index].rowCount,
            ownsBuffer: false
          })
      ),
      validity: props.validity,
      cellSize: STREAMED_LABEL_LIMITS.collisionCellSize
    });
    this.generations.push(this.makeGeneration(props.chunks.map(() => new Uint32Array(0))));
  }

  /** Allocates glyph instances for the given visible rows; bounds buffers are shared. */
  private makeGeneration(visibleRows: readonly Uint32Array[]): Generation {
    const fontAtlas = this.fontAtlas!;
    const buffers: Buffer[] = [];
    const allocate = (data: Float32Array | Uint32Array, minimum: number) => {
      const buffer = this.device.createBuffer({
        ...(data.byteLength ? {data} : {byteLength: minimum}),
        usage: Buffer.STORAGE | Buffer.COPY_DST
      });
      buffers.push(buffer);
      return buffer;
    };
    let remainingGlyphs = STREAMED_LABEL_LIMITS.maximumVisibleGlyphs;
    let visibleLabels = 0;
    let visibleGlyphs = 0;
    let requestedLabels = 0;
    let rowIndexBase = 0;
    const batches = this.props.chunks.map((chunk, index) => {
      requestedLabels += visibleRows[index].length;
      const layout = layoutVisibleLabels(chunk, visibleRows[index], fontAtlas, remainingGlyphs);
      remainingGlyphs -= layout.glyphCount;
      visibleLabels += layout.rowCount;
      visibleGlyphs += layout.glyphCount;
      const batch: GPUPointLabelBatch = {
        rowCount: chunk.rowCount,
        rowIndexBase,
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
        rowBounds: new GPUData({
          format: 'float32x4',
          buffer: this.boundsBuffers[index],
          length: chunk.rowCount,
          ownsBuffer: false
        })
      };
      rowIndexBase += chunk.rowCount;
      return batch;
    });
    this.stats.visibleLabels = visibleLabels;
    this.stats.visibleGlyphs = visibleGlyphs;
    this.stats.truncated = visibleLabels < requestedLabels;
    return {batches, buffers};
  }

  private applyVisibilityWords(words: Uint32Array): void {
    const generation = this.makeGeneration(collectVisibleRows(words, this.props.chunks));
    this.generations.push(generation);
    // Keep the generation Deck may still be drawing; older ones are no longer referenced.
    while (this.generations.length > 2) {
      for (const buffer of this.generations.shift()!.buffers) buffer.destroy();
    }
  }

  private requestPlacement(update: PlacementUpdate): void {
    const key = JSON.stringify(update);
    if (key === this.placementKey || key === this.queuedPlacement?.key) return;
    if (this.placing) {
      this.queuedPlacement = {key, update};
      return;
    }
    void this.runPlacement(key, update);
  }

  private async runPlacement(key: string, update: PlacementUpdate): Promise<void> {
    this.placing = true;
    try {
      let next: {key: string; update: PlacementUpdate} | undefined = {key, update};
      while (next && this.current && this.collision) {
        this.placementKey = next.key;
        this.queuedPlacement = undefined;
        const result = await this.collision.updateUntilStable(next.update);
        const words = await this.collision.readVisibilityWords();
        if (!this.current) return;
        this.stats.collisionRounds = result.rounds;
        this.stats.collisionAccepted = result.accepted;
        this.applyVisibilityWords(words);
        this.props.onChange();
        next = this.queuedPlacement;
      }
    } catch {
      // A retired runtime or lost device ends placement; the previous labels stay as they were.
    } finally {
      this.placing = false;
    }
  }

  resolve(options: ResolvePointLabelOptions): PointLabelResolution {
    if (!this.current || !this.resources || !this.collision || !this.generations.length)
      return {diagnostic: 'Point label resources have retired or the device was lost'};
    const result = resolvePointLabelLayerProps(
      {
        fieldName: this.props.fieldName,
        fontWeight: this.fontWeight,
        formatKey: this.props.formatKey ?? '',
        sdf: Boolean(this.props.sdf),
        hasRowBounds: true,
        hasCollisionAnchors: true,
        batches: this.generations[this.generations.length - 1].batches,
        resources: this.resources,
        positions: this.props.positions
      },
      // Visibility comes from the glyph set itself, not a per-row mask.
      {...options, label: {...options.label, collisionEnabled: false}}
    );
    const {mapState} = options;
    if ('layerProps' in result && mapState.width && mapState.height) {
      const layerProps = result.layerProps;
      this.requestPlacement({
        viewport: {
          longitude: mapState.longitude,
          latitude: mapState.latitude,
          zoom: mapState.zoom,
          width: mapState.width,
          height: mapState.height,
          pitch: mapState.pitch,
          bearing: mapState.bearing,
          globe: Boolean(mapState.globe?.enabled)
        },
        size: layerProps.size ?? 18,
        fontSize: layerProps.fontSize ?? 64,
        anchor: layerProps.textAnchor ?? 'start',
        alignment: layerProps.alignmentBaseline ?? 'center',
        // Variable per-row radii are treated as radius zero, as on the CPU path.
        radius: typeof options.radius === 'number' ? options.radius : 0,
        radiusPixelScale: layerProps.radiusPixelScale ?? 0,
        radiusPadding: layerProps.radiusPadding ?? 0,
        verticalPadding: layerProps.verticalPadding ?? 0
      });
    }
    return result;
  }

  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (!this.destruction) {
      this.current = false;
      const fence = this.deviceLoss
        ? Promise.race([Promise.resolve(completion), this.deviceLoss])
        : Promise.resolve(completion);
      this.destruction = fence
        .then(() => undefined)
        .finally(async () => {
          // Initialization may still be streaming; it stops at its next stale check.
          await this.ready.catch(() => undefined);
          this.collision?.destroy();
          for (const generation of this.generations)
            for (const buffer of generation.buffers) buffer.destroy();
          this.generations = [];
          this.boundsBuffers.forEach(buffer => buffer.destroy());
          this.resources?.destroy();
          this.reservations.forEach(reservation => reservation.release());
        });
    }
    return this.destruction;
  }
}
