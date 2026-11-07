// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import type {KeplerDatasetRegistry, KeplerMemoryReservation} from '@luma-contributors/kepler';
import {
  computeKeplerClusterSummaryAppearance,
  type KeplerClusterAppearance,
  type KeplerClusterSummary,
  type KeplerClusterView
} from '@luma-contributors/kepler';
import {
  createKeplerClusterMessageEngine,
  createKeplerInlineClusterEngine,
  type KeplerClusterEngine
} from '@luma-contributors/kepler';
import type {KeplerClusterRanges} from '@luma-contributors/kepler';

import {clamp} from '@math.gl/core';

// Mirrors KEPLER_CLUSTER_BUDGET_MESSAGE in modules/kepler/src/cluster/kepler-compact-cluster.ts,
// which is not yet exported from the @luma.gl/kepler package entry point.
const KEPLER_CLUSTER_BUDGET_MESSAGE = 'The cluster hierarchy exceeds its memory budget.';

/** Native cluster controls that this adapter interprets. */
export type ResidentClusterSettings = {
  clusterRadius: number;
  minClusterSize: number;
  radiusRange: readonly [number, number];
  colorRange: readonly (readonly number[])[];
  colorScaleType: 'quantize' | 'quantile' | 'ordinal' | 'threshold';
  getThresholds?: (sortedValues: readonly number[]) => readonly number[];
};
/** Numeric resident field plus the native aggregation of its member values. */
export type ResidentClusterField = {
  name: string;
  values: GPUVector<'float32' | 'uint32' | 'sint32'>;
  validity?: GPUVector<'uint32'>;
  /** Native aggregate() of every supported aggregation; null entries are invalid source rows. */
  aggregate(values: (number | null)[]): number;
};
export type ResidentClusterMembers = {rows: readonly number[]; total: number; truncated: boolean};
export type ResidentClusterBinding = {
  centers: GPUVector<'float32x2'>;
  radii: GPUVector<'float32'>;
  colors: GPUVector<'unorm8x4'>;
  clusterCount: number;
  revision: number;
  /** Bounded picking: member rows in native supercluster merge order. */
  getMembers(clusterIndex: number, limit: number): Promise<ResidentClusterMembers>;
  /** Native color measure of one cluster: member count, or the aggregated field value. */
  getValue(clusterIndex: number): number;
};
type Upload = {
  buffers: Buffer[];
  binding: Pick<ResidentClusterBinding, 'centers' | 'radii' | 'colors'>;
};

/**
 * Fixed CPU bytes reserved per source row while building: positions (8) and their readback staging
 * (8), mask (4), a float64 field (8), and the builder's per-level scratch and KD tree (about 64).
 */
const CLUSTER_BUILD_BYTES_PER_ROW = 96;
/** Settled bytes per row once built: worker positions, main-thread mask and field, query KD tree. */
const CLUSTER_SETTLED_BYTES_PER_ROW = 8 + 4 + 8 + 12;
/**
 * The typed-array hierarchy (about 20 bytes per row and distinct zoom level, levels without merges
 * shared) is admitted by measurement: the builder aborts above the current per-row budget, which
 * doubles up to the maximum, each step admitted through the memory registry before the retry.
 */
const CLUSTER_INITIAL_HIERARCHY_BYTES_PER_ROW = 64;
const CLUSTER_MAXIMUM_HIERARCHY_BYTES_PER_ROW = 512;
/** Readbacks are split so staging memory stays bounded for any row count. */
const CLUSTER_READ_CHUNK_BYTES = 8 * 1024 * 1024;

type LayerState = {
  layerId: string;
  sourceRevision: number;
  reservation: KeplerMemoryReservation;
  /** Per-row retained hierarchy budget currently admitted by the reservation. */
  hierarchyBytesPerRow: number;
  /** Hierarchy bytes the reservation currently covers; coarser levels built by queries add to it. */
  admittedHierarchyBytes: number;
  engine: KeplerClusterEngine;
  /** Incremented when the state is released, so late async results are discarded. */
  generation: number;
  /** Read positions awaiting their one transfer to the engine. */
  pendingPositions?: Float32Array;
  positionsRead: boolean;
  mask?: Uint32Array;
  maskRevision: number;
  readingMask?: number;
  field?: Float64Array;
  fieldKey?: string;
  readingField?: string;
  /** Build key `${maskRevision}:${clusterRadius}` currently inside the engine. */
  builtKey?: string;
  buildingKey?: string;
  queryingKey?: string;
  ranges?: KeplerClusterRanges;
  rangesKey?: string;
  appearanceKey?: string;
  failure?: string;
  upload?: Upload;
  binding?: ResidentClusterBinding;
  revision: number;
};

type ResolveRequest = {
  layerId: string;
  sourceRevision: number;
  mask: GPUVector<'uint32'>;
  maskRevision: number;
  settings: ResidentClusterSettings;
  view: KeplerClusterView;
  field?: ResidentClusterField;
};
type ResolveResult = {binding?: ResidentClusterBinding; reason?: string};
type MembersRead = {members: ResidentClusterMembers; value: number} | null;

/** Builds the supercluster engine in a module Web Worker; inline when workers are unavailable. */
export function createClusterEngine(): KeplerClusterEngine {
  if (typeof Worker === 'undefined') return createKeplerInlineClusterEngine();
  try {
    const worker = new Worker(new URL('./cluster.worker.ts', import.meta.url), {type: 'module'});
    return createKeplerClusterMessageEngine({
      post: (message, transfer) => worker.postMessage(message, transfer),
      subscribe: listener => {
        worker.onmessage = event => listener(event.data);
      },
      subscribeError: listener => {
        worker.onerror = event => listener(event.message || 'The cluster worker failed.');
      },
      close: () => worker.terminate()
    });
  } catch {
    return createKeplerInlineClusterEngine();
  }
}

/**
 * Exact native supercluster for every source size. Native Kepler clusters with supercluster, a
 * serial, source-order-dependent greedy hierarchy, so it is built from the real positions and
 * queried by viewport. Positions, the point filter mask and the aggregation field are read back in
 * bounded chunks (reserved as CPU memory; only a failed reservation refuses). Positions move to a
 * Web Worker that builds the index and answers viewport queries with transferred typed arrays; the
 * main thread then applies the native aggregate() and appearance. While a rebuild or query is
 * pending the previous binding stays current. Stale results are discarded.
 */
export class ResidentClusterRuntime {
  private readonly states = new Map<string, LayerState>();
  private destroyed = false;
  constructor(
    private readonly props: {
      device: Device;
      registry: KeplerDatasetRegistry;
      positions: GPUVector<'float32x2'>;
      requestUpdate: () => void;
      createEngine?: () => KeplerClusterEngine;
    }
  ) {}

  /** Returns a binding when current; `{}` while the first bounded build is pending. */
  resolve(request: ResolveRequest): ResolveResult {
    if (this.destroyed) return {reason: 'The cluster runtime is closed.'};
    let state = this.states.get(request.layerId);
    if (state?.failure) {
      if (state.sourceRevision === request.sourceRevision) return {reason: state.failure};
    }
    if (!state || state.sourceRevision !== request.sourceRevision) {
      if (state) this.releaseState(state);
      let reservation: KeplerMemoryReservation;
      try {
        reservation = this.props.registry.reserveMemory(
          this.props.positions.length *
            (CLUSTER_BUILD_BYTES_PER_ROW + CLUSTER_INITIAL_HIERARCHY_BYTES_PER_ROW),
          'cpu',
          'scratch'
        );
      } catch (error) {
        return {reason: error instanceof Error ? error.message : String(error)};
      }
      state = {
        layerId: request.layerId,
        sourceRevision: request.sourceRevision,
        reservation,
        hierarchyBytesPerRow: CLUSTER_INITIAL_HIERARCHY_BYTES_PER_ROW,
        admittedHierarchyBytes: 0,
        engine: (this.props.createEngine ?? createClusterEngine)(),
        generation: 0,
        positionsRead: false,
        maskRevision: -1,
        revision: (state?.revision ?? 0) + 1
      };
      this.states.set(request.layerId, state);
    }
    const current = state;
    const buildKey = `${request.maskRevision}:${request.settings.clusterRadius}`;
    if (current.maskRevision !== request.maskRevision) {
      current.mask = undefined;
      current.maskRevision = request.maskRevision;
    }
    // The color field is read once per source revision and field; invalid rows become NaN.
    const field = request.field;
    if (field && current.fieldKey !== field.name) {
      current.field = undefined;
      current.fieldKey = field.name;
    }
    if (field && !current.field) this.startFieldRead(current, field);
    if (current.builtKey !== buildKey && current.buildingKey === undefined)
      this.startBuild(current, request, buildKey);
    if (current.builtKey === buildKey) {
      const rangesKey = JSON.stringify([buildKey, request.view, request.settings.minClusterSize]);
      if (current.rangesKey !== rangesKey && current.queryingKey === undefined)
        this.startQuery(current, request, buildKey, rangesKey);
      if (current.ranges && current.rangesKey === rangesKey && (!field || current.field)) {
        const appearanceKey = JSON.stringify([rangesKey, request.settings, field?.name ?? null]);
        if (current.appearanceKey !== appearanceKey || !current.binding)
          this.applyAppearance(current, request, current.ranges, appearanceKey);
      }
    }
    return current.binding ? {binding: current.binding} : {};
  }

  /** Bounded hover read of the current binding; stale or unresolved clusters return null. */
  getMembers(layerId: string, clusterIndex: number, limit: number): Promise<MembersRead> {
    return readBindingMembers(this.states.get(layerId)?.binding, clusterIndex, limit);
  }

  getBinding(layerId: string): ResidentClusterBinding | undefined {
    return this.states.get(layerId)?.binding;
  }

  hasLayer(layerId: string): boolean {
    return this.states.has(layerId);
  }

  remove(layerId: string): void {
    const state = this.states.get(layerId);
    if (state) this.releaseState(state);
    this.states.delete(layerId);
  }

  destroy(): void {
    this.destroyed = true;
    for (const state of this.states.values()) this.releaseState(state);
    this.states.clear();
  }

  private isCurrent(state: LayerState, generation: number): boolean {
    return (
      !this.destroyed && state.generation === generation && this.states.get(state.layerId) === state
    );
  }

  /** Reads positions once (handing them to the worker), the mask, then rebuilds the index. */
  private startBuild(state: LayerState, request: ResolveRequest, buildKey: string): void {
    const {generation} = state;
    const maskRevision = request.maskRevision;
    // A rebuild after settling re-admits the build footprint before any allocation.
    const admitted =
      this.props.positions.length * (CLUSTER_BUILD_BYTES_PER_ROW + state.hierarchyBytesPerRow);
    if (state.reservation.byteLength < admitted) {
      try {
        state.reservation.resize(admitted);
      } catch (error) {
        this.fail(state, generation, error);
        return;
      }
    }
    state.buildingKey = buildKey;
    const run = async () => {
      if (!state.positionsRead && !state.pendingPositions) {
        const positions = new Float32Array(this.props.positions.length * 2);
        await readVectorBytes(this.props.positions, new Uint8Array(positions.buffer), () =>
          this.isCurrent(state, generation)
        );
        state.pendingPositions = positions;
      }
      if (!state.mask || state.maskRevision !== maskRevision) {
        const mask = new Uint32Array(request.mask.length);
        await readVectorBytes(request.mask, new Uint8Array(mask.buffer), () =>
          this.isCurrent(state, generation)
        );
        if (state.maskRevision === maskRevision) state.mask = mask;
      }
      // A newer filter revision supersedes this build; the next render starts it.
      if (!this.isCurrent(state, generation) || state.maskRevision !== maskRevision) return;
      const positions = state.pendingPositions;
      state.pendingPositions = undefined;
      if (positions) state.positionsRead = true;
      return await state.engine.build({
        positions,
        selectionMask: state.mask,
        clusterRadius: request.settings.clusterRadius,
        maximumHierarchyBytes: state.hierarchyBytesPerRow * this.props.positions.length,
        // Only the levels the view needs are built now; zooming out extends the index in the worker.
        lowestZoom: Math.round(request.view.zoom)
      });
    };
    run().then(
      built => {
        if (!this.isCurrent(state, generation)) return;
        // Settle the reservation to what the build measurably retains.
        if (built) this.settleReservation(state, built.hierarchyBytes, state.hierarchyBytesPerRow);
        state.buildingKey = undefined;
        // A superseded build leaves the key unmatched, so the next render rebuilds.
        if (state.maskRevision === maskRevision) state.builtKey = buildKey;
        this.props.requestUpdate();
      },
      error => {
        if (
          this.isCurrent(state, generation) &&
          error instanceof Error &&
          error.message === KEPLER_CLUSTER_BUDGET_MESSAGE &&
          state.hierarchyBytesPerRow < CLUSTER_MAXIMUM_HIERARCHY_BYTES_PER_ROW
        ) {
          try {
            // Admit the doubled budget before the worker is asked to retry.
            state.hierarchyBytesPerRow *= 2;
            state.reservation.resize(
              this.props.positions.length *
                (CLUSTER_BUILD_BYTES_PER_ROW + state.hierarchyBytesPerRow)
            );
            state.buildingKey = undefined;
            this.startBuild(state, request, buildKey);
            return;
          } catch (retryError) {
            error = retryError;
          }
        }
        this.fail(state, generation, error);
      }
    );
  }

  /** Shrinks the build-time reservation to the settled footprint after a successful build. */
  private settleReservation(
    state: LayerState,
    hierarchyBytes: number,
    hierarchyBytesPerRow: number
  ): void {
    const rowCount = this.props.positions.length;
    state.admittedHierarchyBytes = hierarchyBytes;
    const settled = rowCount * CLUSTER_SETTLED_BYTES_PER_ROW + hierarchyBytes;
    const admitted = rowCount * (CLUSTER_BUILD_BYTES_PER_ROW + hierarchyBytesPerRow);
    if (settled < admitted) state.reservation.resize(settled);
  }

  private startQuery(
    state: LayerState,
    request: ResolveRequest,
    buildKey: string,
    rangesKey: string
  ): void {
    const {generation} = state;
    state.queryingKey = rangesKey;
    state.engine.query(request.view, request.settings.minClusterSize).then(
      ranges => {
        if (!this.isCurrent(state, generation)) return;
        state.queryingKey = undefined;
        // Discard results of an index that has since been rebuilt.
        if (state.builtKey === buildKey) {
          // A zoom-out query may have built coarser levels; admit their bytes before keeping them.
          if (ranges.hierarchyBytes > state.admittedHierarchyBytes) {
            try {
              state.reservation.resize(
                this.props.positions.length * CLUSTER_SETTLED_BYTES_PER_ROW + ranges.hierarchyBytes
              );
              state.admittedHierarchyBytes = ranges.hierarchyBytes;
            } catch (error) {
              this.fail(state, generation, error);
              return;
            }
          }
          state.ranges = ranges;
          state.rangesKey = rangesKey;
        }
        this.props.requestUpdate();
      },
      error => {
        if (
          this.isCurrent(state, generation) &&
          error instanceof Error &&
          error.message === KEPLER_CLUSTER_BUDGET_MESSAGE &&
          state.hierarchyBytesPerRow < CLUSTER_MAXIMUM_HIERARCHY_BYTES_PER_ROW
        ) {
          // The coarser levels do not fit the build-time bound: rebuild with a doubled budget.
          try {
            state.hierarchyBytesPerRow *= 2;
            state.reservation.resize(
              this.props.positions.length *
                (CLUSTER_BUILD_BYTES_PER_ROW + state.hierarchyBytesPerRow)
            );
            state.queryingKey = undefined;
            state.builtKey = undefined;
            this.props.requestUpdate();
            return;
          } catch (retryError) {
            this.fail(state, generation, retryError);
            return;
          }
        }
        this.fail(state, generation, error);
      }
    );
  }

  private startFieldRead(state: LayerState, field: ResidentClusterField): void {
    if (state.readingField === field.name) return;
    const {generation} = state;
    state.readingField = field.name;
    readScalar(field.values, field.validity, () => this.isCurrent(state, generation)).then(
      values => {
        if (!this.isCurrent(state, generation)) return;
        if (state.fieldKey === field.name) state.field = values;
        this.props.requestUpdate();
      },
      () => {
        // Allow a retry on the next render after a failed or lost-device readback.
        if (state.readingField === field.name) state.readingField = undefined;
      }
    );
  }

  private fail(state: LayerState, generation: number, error: unknown): void {
    if (!this.isCurrent(state, generation)) return;
    state.buildingKey = undefined;
    state.queryingKey = undefined;
    state.failure = error instanceof Error ? error.message : String(error);
    this.props.requestUpdate();
  }

  private applyAppearance(
    state: LayerState,
    request: ResolveRequest,
    ranges: KeplerClusterRanges,
    appearanceKey: string
  ): void {
    const fieldValues = state.field;
    const field = request.field;
    const summaries: KeplerClusterSummary[] = [];
    for (let index = 0; index < ranges.clusterCount; index++) {
      const start = ranges.offsets[index];
      const end = ranges.offsets[index + 1];
      let colorValue = end - start;
      if (field && fieldValues) {
        const entries: (number | null)[] = [];
        for (let position = start; position < end; position++) {
          const value = fieldValues[ranges.rows[position]];
          entries.push(Number.isNaN(value) ? null : value);
        }
        colorValue = field.aggregate(entries);
      }
      summaries.push({
        count: end - start,
        colorValue,
        longitude: ranges.centers[index * 2],
        latitude: ranges.centers[index * 2 + 1]
      });
    }
    const appearance = computeKeplerClusterSummaryAppearance(summaries, request.settings);
    const colorValues = appearance.colorValues;
    const previous = state.upload;
    state.upload = uploadClusterAppearance(this.props.device, request.layerId, appearance);
    state.revision++;
    state.binding = {
      ...state.upload.binding,
      clusterCount: ranges.clusterCount,
      revision: state.revision,
      getMembers: (clusterIndex, limit) => {
        const start = ranges.offsets[clusterIndex];
        const total = ranges.offsets[clusterIndex + 1] - start;
        const count = clamp(limit, 0, total);
        return Promise.resolve({
          rows: Array.from(ranges.rows.subarray(start, start + count)),
          total,
          truncated: total > limit
        });
      },
      getValue: clusterIndex => colorValues[clusterIndex]
    };
    state.appearanceKey = appearanceKey;
    if (previous) retireClusterUpload(this.props.device, previous);
  }

  private releaseState(state: LayerState): void {
    state.generation++;
    state.engine.destroy();
    if (state.upload) retireClusterUpload(this.props.device, state.upload);
    state.upload = undefined;
    state.binding = undefined;
    state.ranges = undefined;
    state.mask = undefined;
    state.field = undefined;
    state.reservation.release();
  }
}

/**
 * Reads a vector into the byte target in CLUSTER_READ_CHUNK_BYTES slices, so GPU staging memory
 * stays bounded. Stops early when the caller no longer needs the result.
 */
async function readVectorBytes(
  vector: GPUVector,
  target: Uint8Array,
  shouldContinue: () => boolean
): Promise<void> {
  let offset = 0;
  for (const chunk of vector.data) {
    const chunkBytes = chunk.length * chunk.byteStride;
    for (let position = 0; position < chunkBytes; position += CLUSTER_READ_CHUNK_BYTES) {
      if (!shouldContinue()) throw new Error('The cluster readback was superseded.');
      const byteLength = Math.min(CLUSTER_READ_CHUNK_BYTES, chunkBytes - position);
      target.set(await chunk.buffer.readAsync(chunk.byteOffset + position, byteLength), offset);
      offset += byteLength;
    }
  }
}

async function readScalar(
  vector: GPUVector<'float32' | 'uint32' | 'sint32'>,
  validity: GPUVector<'uint32'> | undefined,
  shouldContinue: () => boolean
): Promise<Float64Array> {
  const raw =
    vector.format === 'float32'
      ? new Float32Array(vector.length)
      : vector.format === 'uint32'
        ? new Uint32Array(vector.length)
        : new Int32Array(vector.length);
  await readVectorBytes(vector, new Uint8Array(raw.buffer), shouldContinue);
  const result = Float64Array.from(raw);
  if (validity) {
    const flags = new Uint32Array(validity.length);
    await readVectorBytes(validity, new Uint8Array(flags.buffer), shouldContinue);
    for (let row = 0; row < result.length; row++) if (flags[row] === 0) result[row] = Number.NaN;
  }
  return result;
}

async function readBindingMembers(
  binding: ResidentClusterBinding | undefined,
  clusterIndex: number,
  limit: number
): Promise<MembersRead> {
  return binding && clusterIndex >= 0 && clusterIndex < binding.clusterCount
    ? {
        members: await binding.getMembers(clusterIndex, limit),
        value: binding.getValue(clusterIndex)
      }
    : null;
}

function uploadClusterAppearance(
  device: Device,
  layerId: string,
  appearance: KeplerClusterAppearance
): Upload {
  const buffers: Buffer[] = [];
  const make = <Format extends 'float32x2' | 'float32' | 'unorm8x4'>(
    name: string,
    format: Format,
    data: Float32Array | Uint8Array,
    length: number
  ): GPUVector<Format> => {
    const buffer = device.createBuffer({
      id: `${layerId}-cluster-${name}`,
      data: data.byteLength ? data : new Uint8Array(4),
      usage: Buffer.VERTEX | Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
    });
    buffers.push(buffer);
    return new GPUVector({
      type: 'data',
      name: `${layerId}-cluster-${name}`,
      format,
      data: [new GPUData({buffer, format, length, ownsBuffer: false})],
      ownsData: false
    });
  };
  const length = appearance.counts.length;
  return {
    buffers,
    binding: {
      centers: make('centers', 'float32x2', appearance.centers, length),
      radii: make('radii', 'float32', appearance.radii, length),
      colors: make('colors', 'unorm8x4', appearance.colors, length)
    }
  };
}

/** Retires small cluster buffers once the GPU has finished any frame that may still read them. */
function retireClusterUpload(device: Device, upload: Upload): void {
  if (device.isLost) return;
  try {
    const fence = device.createFence();
    fence.signaled
      .catch(() => undefined)
      .finally(() => {
        fence.destroy();
        for (const buffer of upload.buffers) buffer.destroy();
      });
  } catch {
    // A device torn down with its session already released these buffers.
    for (const buffer of upload.buffers) buffer.destroy();
  }
}
