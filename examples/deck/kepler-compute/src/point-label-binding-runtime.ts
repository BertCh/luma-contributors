// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPUPointLabelLayer} from '@deck.gl-community/gpu-layers';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetLease,
  KeplerDatasetSnapshot
} from '@luma-contributors/kepler';
import {
  makePointLabelRuntime,
  type NativePointLabelConfig,
  type NativePointLabelMapState,
  type PointLabelResolution,
  type PointLabelRuntime
} from './label-runtime';
import {
  POINT_LABEL_LIMITS,
  assertPointLabelRowCount,
  prepareResidentPointLabels,
  prepareStreamedPointLabels
} from './point-label-admission';
import {StreamedPointLabelRuntime} from './label-streamed-runtime';
import {getPointLabelFormatKey, makePointLabelFormatter} from './label-format';
import {materializeResidentRows} from './resident-materialization';
import type {LabelAnchorSource} from './label-collision-runtime';
import {createGpuLabelCollision, type GpuLabelCollision} from './label-collision-gpu';

/**
 * Where CPU label anchors come from. `columns` reads longitude/latitude from the retained source
 * table (needed only for collision); `gpu` reads back a resident per-row anchor vector (for
 * non-point geometry such as GeoJSON polygon and line labels), with optional validity.
 */
export type PointLabelAnchorRequest =
  | {kind: 'columns'; longitude: string; latitude: string}
  | {
      kind: 'gpu';
      positions: GPUVector<'float32x2'>;
      validity?: GPUVector<'uint32'>;
    };

async function readGPUVectorBytes(vector: GPUVector<'float32x2' | 'uint32'>): Promise<ArrayBuffer> {
  const elementBytes = vector.format === 'uint32' ? 4 : 8;
  const output = new Uint8Array(vector.length * elementBytes);
  let offset = 0;
  for (const chunk of vector.data) {
    // Anchor vectors are tightly packed; the readback copies bytes in chunk order.
    const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * elementBytes);
    output.set(bytes, offset);
    offset += chunk.length * elementBytes;
  }
  return output.buffer;
}

async function readGPUAnchors(
  request: Extract<PointLabelAnchorRequest, {kind: 'gpu'}>
): Promise<LabelAnchorSource> {
  const positions = new Float32Array(await readGPUVectorBytes(request.positions));
  const validity = request.validity
    ? new Uint32Array(await readGPUVectorBytes(request.validity))
    : undefined;
  return {positions, validity};
}

function readColumnAnchors(
  dataset: KeplerDatasetSnapshot<KeplerArrowDataset>,
  request: Extract<PointLabelAnchorRequest, {kind: 'columns'}>
): LabelAnchorSource | undefined {
  const source = dataset.value.source;
  const positions = new Float64Array(source.numRows * 2);
  const longitudes = source.getChild(request.longitude);
  const latitudes = source.getChild(request.latitude);
  if (!longitudes || !latitudes) return undefined;
  const validity = new Uint32Array(source.numRows);
  for (let rowIndex = 0; rowIndex < source.numRows; rowIndex++) {
    const longitude = Number(longitudes.get(rowIndex));
    const latitude = Number(latitudes.get(rowIndex));
    positions[rowIndex * 2] = longitude;
    positions[rowIndex * 2 + 1] = latitude;
    validity[rowIndex] = Number.isFinite(longitude + latitude) ? 1 : 0;
  }
  return {positions, validity};
}

type LabelOwner = {
  datasetId: string;
  sourceRevision: number;
  layerId: string;
  key: string;
  positions: GPUVector<'float32x2' | 'float32x3'>;
  runtime?: PointLabelRuntime;
  /** Sources above the glyph cap stream their labels and run GPU collision per viewport. */
  streamed?: StreamedPointLabelRuntime;
  /** GPU collision state; set when collision is enabled without CPU anchors (or above the CPU cap). */
  gpuCollision?: {collision?: GpuLabelCollision; key?: string; words?: Uint32Array};
  lease?: KeplerDatasetLease;
  ready: boolean;
  error?: string;
};

/** Owns explicit source/font variants; filter and display controls only replace GPU bindings. */
export class PointLabelBindingRuntime {
  private readonly entries = new Map<string, LabelOwner>();
  private readonly pendingRetirements = new Set<LabelOwner>();
  private readonly retirements = new Set<Promise<void>>();
  private readonly failures: unknown[] = [];
  private readonly listeners = new Set<() => void>();
  private destroyed = false;

  constructor(
    private readonly session: KeplerComputeSession,
    private readonly requestUpdate: () => void
  ) {}

  resolve(options: {
    dataset: KeplerDatasetSnapshot<KeplerArrowDataset>;
    layerId: string;
    labels: readonly NativePointLabelConfig[];
    positions: GPUVector<'float32x2' | 'float32x3'>;
    selectionMask: GPUVector<'uint32'>;
    radius: number | GPUVector<'float32'>;
    radiusScale: number;
    mapState: NativePointLabelMapState;
    visible?: boolean;
    /** Required for collision avoidance, and for geometry whose anchors are not the point positions. */
    anchors?: PointLabelAnchorRequest;
  }): {layers: Layer[]; reason?: string; pending?: boolean} {
    if (
      this.destroyed ||
      this.session.deviceLost ||
      this.session.registry.getDataset(options.dataset.id) !== options.dataset
    )
      return {layers: [], reason: 'The point label source is stale or unavailable.'};
    const layers: Layer[] = [];
    const activeKeys = new Set<string>();
    let reason: string | undefined;
    let pending = false;
    for (const [labelIndex, label] of options.labels.entries()) {
      if (!label.field?.name) continue;
      const entryId = `${options.layerId}:label:${labelIndex}`;
      activeKeys.add(entryId);
      const key = JSON.stringify([
        options.dataset.id,
        options.dataset.revision,
        label.field.name,
        label.weight ?? 400,
        getPointLabelFormatKey(label),
        Boolean(label.outlineWidth),
        Boolean(label.background),
        Boolean(label.collisionEnabled),
        options.anchors?.kind ?? null
      ]);
      let entry = this.entries.get(entryId);
      if (entry && (entry.key !== key || entry.positions !== options.positions)) {
        this.entries.delete(entryId);
        this.pendingRetirements.add(entry);
        entry = undefined;
      }
      if (!entry) {
        entry = {
          datasetId: options.dataset.id,
          sourceRevision: options.dataset.revision,
          layerId: options.layerId,
          key,
          positions: options.positions,
          ready: false,
          ...(label.collisionEnabled &&
          (!options.anchors ||
            options.dataset.value.source.numRows > POINT_LABEL_LIMITS.maximumRows)
            ? {gpuCollision: {}}
            : {})
        };
        this.entries.set(entryId, entry);
        const owner = entry;
        let admission;
        try {
          owner.lease = this.session.registry.acquireDataset(options.dataset.id);
          admission = this.session.registry.reserveMemory(
            POINT_LABEL_LIMITS.maximumSourceBytes,
            'cpu',
            'source'
          );
          if (
            owner.gpuCollision &&
            options.dataset.value.source.numRows > POINT_LABEL_LIMITS.maximumRows
          ) {
            // Above the glyph cap only GPU collision can run, and only visible rows get glyphs.
            this.startStreamedOwner(
              entryId,
              owner,
              options,
              label,
              Boolean(options.dataset.value.resident)
            );
          } else if (options.dataset.value.resident) {
            // Over-cap results page their label rows from the GPU asynchronously; the row limit
            // is checked first so unreadable sizes fail before any readback.
            assertPointLabelRowCount(options.dataset.value.source.numRows);
            this.startResidentOwner(entryId, owner, options, label, true);
          } else if (options.anchors?.kind === 'gpu') {
            this.startResidentOwner(entryId, owner, options, label, false);
          } else {
            this.startOwner(entryId, owner, options, label);
          }
        } catch (error) {
          owner.error = error instanceof Error ? error.message : String(error);
          if (!owner.runtime && !owner.streamed && owner.lease) {
            const lease = owner.lease;
            owner.lease = undefined;
            this.track(lease.release());
          }
        } finally {
          admission?.release();
        }
      }
      if (entry.error) {
        reason ??= entry.error;
        continue;
      }
      const runtime = entry.streamed ?? entry.runtime;
      if (!entry.ready || !runtime) {
        pending = true;
        continue;
      }
      // The GPU collision path owns visibility, so the runtime must not require CPU anchors.
      const result = runtime.resolve({
        ...options,
        label: entry.gpuCollision ? {...label, collisionEnabled: false} : label
      });
      if ('diagnostic' in result) {
        reason ??= result.diagnostic;
        continue;
      }
      if (entry.gpuCollision && !entry.streamed) {
        const words = this.resolveGpuCollision(entry, options, result.layerProps);
        if (words) result.layerProps.visibilityWords = words;
      }
      layers.push(
        new GPUPointLabelLayer({
          id: `${options.layerId}-resident-point-label-${labelIndex}`,
          ...result.layerProps,
          visible: options.visible ?? true
        })
      );
    }
    for (const [entryId, entry] of this.entries) {
      if (entry.layerId === options.layerId && !activeKeys.has(entryId)) {
        this.entries.delete(entryId);
        this.pendingRetirements.add(entry);
      }
    }
    return {layers, ...(reason ? {reason} : {}), ...(pending ? {pending} : {})};
  }

  /**
   * Runs GPU collision for the current viewport and returns the last read-back visibility words.
   * Rows stay hidden until the first mask arrives; arrival triggers a layer update. Needs the
   * label batches' GPU rowBounds and chunk-aligned anchors; otherwise labels stay unfiltered.
   */
  private resolveGpuCollision(
    entry: LabelOwner,
    options: Parameters<PointLabelBindingRuntime['resolve']>[0],
    layerProps: Extract<PointLabelResolution, {layerProps: unknown}>['layerProps']
  ): Uint32Array | undefined {
    const state = entry.gpuCollision;
    const {mapState} = options;
    if (!state || !mapState.width || !mapState.height) return undefined;
    const request = options.anchors;
    const positions = (
      request?.kind === 'gpu' ? request.positions : options.positions
    ) as GPUVector<'float32x2' | 'float32x3'>;
    const validity = request?.kind === 'gpu' ? request.validity : undefined;
    const rowBounds = layerProps.batches.map(batch => batch.rowBounds);
    if (
      !rowBounds.every(Boolean) ||
      positions.data.length !== rowBounds.length ||
      positions.data.some((chunk, index) => chunk.length !== layerProps.batches[index].rowCount) ||
      (validity &&
        validity.data.some((chunk, index) => chunk.length !== positions.data[index]?.length))
    )
      return undefined;
    state.collision ??= createGpuLabelCollision({
      device: this.session.device,
      positions,
      rowBounds: rowBounds as NonNullable<(typeof rowBounds)[number]>[],
      validity
    });
    const update = {
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
    };
    const key = JSON.stringify(update);
    if (state.key !== key) {
      state.key = key;
      const collision = state.collision;
      collision.update(update);
      void collision.readVisibilityWords().then(
        words => {
          if (!this.isOwned(entry)) return;
          state.words = words;
          this.changed();
        },
        () => undefined
      );
    }
    return state.words ?? new Uint32Array(state.collision.wordCount);
  }

  /** Streams the source in windows (no glyph cap), then places labels on the GPU per viewport. */
  private startStreamedOwner(
    entryId: string,
    owner: LabelOwner,
    options: Parameters<PointLabelBindingRuntime['resolve']>[0],
    label: NativePointLabelConfig,
    materialize: boolean
  ): void {
    const isOwned = () => this.entries.get(entryId) === owner && !this.destroyed;
    void (async () => {
      try {
        if (materialize)
          await materializeResidentRows(options.dataset.value, undefined, {
            assertCurrent: () => {
              if (!isOwned()) throw new Error('The point label source is stale or unavailable.');
            }
          });
        if (!isOwned()) throw new Error('The point label source is stale or unavailable.');
        const request = options.anchors;
        const chunks = prepareStreamedPointLabels(
          options.dataset.value.source,
          options.dataset.batches,
          label.field?.name ?? '',
          makePointLabelFormatter(label)
        );
        const streamed = new StreamedPointLabelRuntime(this.session.device, {
          chunks,
          fieldName: label.field?.name ?? '',
          positions: options.positions,
          collisionPositions: (request?.kind === 'gpu'
            ? request.positions
            : options.positions) as GPUVector<'float32x2' | 'float32x3'>,
          validity: request?.kind === 'gpu' ? request.validity : undefined,
          fontWeight: label.weight,
          formatKey: getPointLabelFormatKey(label),
          sdf: Boolean(label.outlineWidth),
          memoryRegistry: this.session.registry,
          onChange: () => {
            if (isOwned()) this.changed();
          }
        });
        owner.streamed = streamed;
        await streamed.ready;
        if (!isOwned()) return;
        owner.ready = true;
        this.changed();
      } catch (error) {
        if (!isOwned()) return;
        owner.error = error instanceof Error ? error.message : String(error);
        if (!owner.streamed && owner.lease) {
          const lease = owner.lease;
          owner.lease = undefined;
          this.track(lease.release());
        }
        this.changed();
      }
    })();
  }

  private isOwned(entry: LabelOwner): boolean {
    for (const candidate of this.entries.values()) if (candidate === entry) return true;
    return false;
  }

  private startOwner(
    entryId: string,
    owner: LabelOwner,
    options: Parameters<PointLabelBindingRuntime['resolve']>[0],
    label: NativePointLabelConfig,
    gpuAnchors?: LabelAnchorSource
  ): void {
    const request = options.anchors;
    const anchors = owner.gpuCollision
      ? undefined
      : (gpuAnchors ??
        (request?.kind === 'columns' && label.collisionEnabled && !owner.gpuCollision
          ? readColumnAnchors(options.dataset, request)
          : undefined));
    const source = prepareResidentPointLabels(
      options.dataset.value.source,
      options.dataset.batches,
      label.field?.name ?? '',
      makePointLabelFormatter(label)
    );
    owner.runtime = makePointLabelRuntime(this.session.device, {
      source,
      positions: options.positions,
      fontWeight: label.weight,
      formatKey: getPointLabelFormatKey(label),
      sdf: Boolean(label.outlineWidth),
      rowBounds: Boolean(label.background) || Boolean(owner.gpuCollision),
      anchors,
      workingMemoryBudgetBytes: POINT_LABEL_LIMITS.workingMemoryBudgetBytes,
      memoryRegistry: this.session.registry
    });
    void owner.runtime.ready.then(
      () => {
        if (this.entries.get(entryId) !== owner || !owner.lease?.isValid()) return;
        owner.ready = true;
        this.changed();
      },
      error => {
        if (this.entries.get(entryId) !== owner) return;
        owner.error = error instanceof Error ? error.message : String(error);
        this.changed();
      }
    );
  }

  /** Streams the resident label rows into the accounted lazy view, then admits them as usual. */
  private startResidentOwner(
    entryId: string,
    owner: LabelOwner,
    options: Parameters<PointLabelBindingRuntime['resolve']>[0],
    label: NativePointLabelConfig,
    materialize: boolean
  ): void {
    const isOwned = () => this.entries.get(entryId) === owner && !this.destroyed;
    void (async () => {
      let admission;
      try {
        if (materialize)
          await materializeResidentRows(options.dataset.value, undefined, {
            assertCurrent: () => {
              if (!isOwned()) throw new Error('The point label source is stale or unavailable.');
            }
          });
        const request = options.anchors;
        const gpuAnchors = request?.kind === 'gpu' ? await readGPUAnchors(request) : undefined;
        if (!isOwned()) throw new Error('The point label source is stale or unavailable.');
        admission = this.session.registry.reserveMemory(
          POINT_LABEL_LIMITS.maximumSourceBytes,
          'cpu',
          'source'
        );
        this.startOwner(entryId, owner, options, label, gpuAnchors);
      } catch (error) {
        if (!isOwned()) return;
        owner.error = error instanceof Error ? error.message : String(error);
        if (!owner.runtime && owner.lease) {
          const lease = owner.lease;
          owner.lease = undefined;
          this.track(lease.release());
        }
        this.changed();
      } finally {
        admission?.release();
      }
    })();
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Invoke after Deck submits the frame that replaces old label bindings. */
  afterRender(): void {
    if (!this.pendingRetirements.size) return;
    const fence = this.session.device.isLost ? undefined : this.session.device.createFence();
    const completion = fence
      ? Promise.race([fence.signaled, this.session.device.lost]).finally(() => fence.destroy())
      : Promise.resolve();
    for (const entry of this.pendingRetirements) this.retire(entry, completion);
    this.pendingRetirements.clear();
  }

  async retainLayers(layerIds: readonly string[], completion: PromiseLike<unknown>): Promise<void> {
    const retained = new Set(layerIds);
    for (const [entryId, entry] of this.entries) {
      if (!retained.has(entry.layerId)) {
        this.entries.delete(entryId);
        this.retire(entry, completion);
      }
    }
    for (const entry of this.pendingRetirements) this.retire(entry, completion);
    this.pendingRetirements.clear();
    await this.drain();
  }

  async retireDataset(datasetId: string, completion: PromiseLike<unknown>): Promise<void> {
    for (const [entryId, entry] of this.entries) {
      if (entry.datasetId === datasetId) {
        this.entries.delete(entryId);
        this.retire(entry, completion);
      }
    }
    for (const entry of this.pendingRetirements) {
      if (entry.datasetId === datasetId) {
        this.pendingRetirements.delete(entry);
        this.retire(entry, completion);
      }
    }
    await this.drain();
  }

  async destroy(completion: PromiseLike<unknown>): Promise<void> {
    this.destroyed = true;
    this.listeners.clear();
    await this.retainLayers([], completion);
  }

  private changed(): void {
    this.requestUpdate();
    for (const listener of this.listeners) listener();
  }

  private retire(entry: LabelOwner, completion: PromiseLike<unknown>): void {
    this.track(
      (async () => {
        try {
          await entry.runtime?.destroy(completion);
          await entry.streamed?.destroy(completion);
        } finally {
          entry.gpuCollision?.collision?.destroy();
          await entry.lease?.release(completion);
        }
      })()
    );
  }

  private track(retirement: Promise<void>): void {
    this.retirements.add(retirement);
    void retirement
      .catch(error => this.failures.push(error))
      .finally(() => this.retirements.delete(retirement));
  }

  private async drain(): Promise<void> {
    await Promise.allSettled([...this.retirements]);
    if (this.failures.length)
      throw new AggregateError(this.failures.splice(0), 'Point label cleanup failed.');
  }
}
