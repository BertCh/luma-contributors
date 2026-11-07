// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {GPUData, GPUVector, type VertexList} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph, GPUGather, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  isTabularTrips,
  type GeoJSONTripGPUVectors,
  type PreparedGeoJSONTrips
} from '@luma.gl/engine/arrow';
import {TripMembershipFilter} from './trip-membership';
import {estimateTripOutputBytes, resolveTripWorkingBudget} from './trip-working-budget';
import type {TripGrouping} from './trip-source-options';
import type {KeplerDatasetRegistry, KeplerMemoryReservation} from '@luma-contributors/kepler';

export type ResidentTripBinding = {
  column: string;
  getPath: GPUVector<VertexList<'float32x3'>>;
  getTimestamps: GPUVector<VertexList<'float32'>>;
  selectionMask: GPUVector<'uint32'>;
  getColor?: GPUVector<'unorm8x4'>;
  getWidth?: GPUVector<'float32'>;
  /**
   * Grouped tabular trips only: per-vertex marker channels gathered through the member rows, in
   * the path's flattened vertex order. `vertexSelection` flags selected member rows; unselected
   * vertices repeat their nearest selected vertex in the filtered path, and consumers apply the
   * same rule to the other channels.
   */
  markerChannels?: {
    vertexSelection: GPUVector<'uint32'>;
    colors?: GPUVector<'unorm8x4'>;
    roll?: GPUVector<'float32'>;
    pitch?: GPUVector<'float32'>;
    yaw?: GPUVector<'float32'>;
  };
  /** Why a configured roll, pitch or yaw field has no resident per-vertex values. */
  markerChannelReason?: string;
  /** Present for grouped tabular trips: native table-mode column roles. */
  grouping?: TripGrouping;
  originMilliseconds: number;
  maximumTimeErrorMilliseconds: number;
  animationDomainMilliseconds: readonly [number, number] | null;
  resolveSourceRow(primitiveIndex: number): number;
  /**
   * Native table-mode hover: the member row at `bisectRight(timestamps, currentTime)` while the
   * absolute time is inside the trip, else -1. Uses retained CPU timestamps; no GPU readback.
   */
  resolveHoverSourceRow?(primitiveIndex: number, currentTimeMilliseconds: number): number;
};
export type TripRuntimeProps = {
  device: Device;
  column: string;
  trips: GeoJSONTripGPUVectors;
  grouping?: TripGrouping;
  selectionMask: GPUVector<'uint32'>;
  colors?: GPUVector<'unorm8x4'>;
  widths?: GPUVector<'float32'>;
  /** Native roll, pitch and yaw datum fields: one float32 value per source row. */
  orientation?: {
    roll?: GPUVector<'float32'>;
    pitch?: GPUVector<'float32'>;
    yaw?: GPUVector<'float32'>;
  };
  /** Admission for owned outputs plus compiled transient storage; source uploads are separate. */
  workingMemoryBudgetBytes: number;
  /**
   * Let a trip whose exact outputs exceed `workingMemoryBudgetBytes` grow its budget up to the
   * registry bytes still available (see {@link resolveTripWorkingBudget}).
   */
  growWorkingMemoryToRegistry?: boolean;
  /** Omit when the caller accounts the working reservation externally. */
  memoryRegistry?: KeplerDatasetRegistry;
};

/** Borrowed source vectors feed stable per-trip outputs; interaction only records GPU gathers. */
export class TripRuntime {
  readonly binding: ResidentTripBinding;
  readonly trips: GeoJSONTripGPUVectors;
  readonly reason?: string;
  private readonly graph: GPUCommandGraph;
  private readonly buffers: Buffer[] = [];
  private readonly outputs: GPUVector[] = [];
  private reservation?: KeplerMemoryReservation;
  private compiled?: CompiledGPUCommandGraph;
  private readonly workingMemoryBudgetBytes: number;
  private destroyed = false;
  private released = false;
  private destruction?: Promise<void>;

  constructor(private readonly props: TripRuntimeProps) {
    this.trips = props.trips;
    if (!props.column) throw new Error('Trip source column is required');
    if (!Number.isSafeInteger(props.workingMemoryBudgetBytes) || props.workingMemoryBudgetBytes < 0)
      throw new Error('Trip working budget must be nonnegative integral bytes');
    const {source, getPath, getTimestamps, sourceRowIds} = props.trips;
    const orientation = [props.orientation?.roll, props.orientation?.pitch, props.orientation?.yaw];
    const channels = [props.selectionMask, props.colors, props.widths];
    for (const vector of channels) {
      if (vector && vector.length !== source.rowCount)
        throw new Error('Trip mask/style source row count mismatch');
    }
    for (const vector of [...channels, ...orientation]) {
      if (vector && vector.length !== source.rowCount)
        throw new Error('Trip orientation source row count mismatch');
    }
    for (const vector of [...channels, ...orientation, getPath, getTimestamps, sourceRowIds]) {
      for (const chunk of vector?.data ?? [])
        if (chunk.buffer.device !== props.device) throw new Error('Trip source device mismatch');
    }
    if (
      [getPath, getTimestamps, sourceRowIds].some(
        vector => vector.data.length !== source.batches.length || vector.length !== source.tripCount
      )
    )
      throw new Error('Trip geometry/time/provenance chunks must align');
    for (let index = 0; index < source.batches.length; index++) {
      const batch = source.batches[index];
      if (
        getPath.data[index].length !== batch.sourceRows.length ||
        getTimestamps.data[index].length !== batch.sourceRows.length ||
        sourceRowIds.data[index].length !== batch.sourceRows.length ||
        getPath.data[index].valueLength !== getTimestamps.data[index].valueLength
      )
        throw new Error('Trip batch geometry/time/provenance lengths must align');
    }
    const {membership} = props.trips;
    if (
      membership &&
      (membership.memberOffsets.data.length !== source.batches.length ||
        membership.memberRows.data.length !== source.batches.length ||
        membership.memberOffsets.data.some(
          (chunk, index) => chunk.length !== source.batches[index].sourceRows.length + 1
        ))
    )
      throw new Error('Trip membership chunks must align with trip chunks');
    const outputBytes = estimateTripOutputBytes(props.trips, {
      selectionMask: props.selectionMask,
      colors: props.colors,
      widths: props.widths,
      orientation
    });
    this.workingMemoryBudgetBytes = resolveTripWorkingBudget(
      props.workingMemoryBudgetBytes,
      outputBytes,
      props.growWorkingMemoryToRegistry
        ? props.memoryRegistry?.memoryUsage?.availableBytes
        : undefined
    );
    if (outputBytes > this.workingMemoryBudgetBytes)
      throw new KeplerAllocationError('memory-budget', {
        message: 'Trip outputs exceed working memory budget'
      });
    this.reason =
      source.crs !== undefined &&
      source.crs !== null &&
      source.crs !== 'EPSG:4326' &&
      source.crs !== 'OGC:CRS84'
        ? 'Trip CRS requires explicit transformation to longitude/latitude'
        : undefined;
    this.graph = new GPUCommandGraph(props.device, {id: `trips-${props.column}`});
    try {
      this.reservation = props.memoryRegistry?.reserveMemory(
        this.workingMemoryBudgetBytes,
        'gpu',
        'scratch'
      );
      const filtered = membership ? this.filterMembers(props.selectionMask, membership) : undefined;
      const markerChannels =
        membership && filtered
          ? {
              vertexSelection: filtered.memberMask,
              colors: props.colors
                ? this.gatherMembers('vertex-colors', props.colors, membership.memberRows)
                : undefined,
              roll: props.orientation?.roll
                ? this.gatherMembers('vertex-roll', props.orientation.roll, membership.memberRows)
                : undefined,
              pitch: props.orientation?.pitch
                ? this.gatherMembers('vertex-pitch', props.orientation.pitch, membership.memberRows)
                : undefined,
              yaw: props.orientation?.yaw
                ? this.gatherMembers('vertex-yaw', props.orientation.yaw, membership.memberRows)
                : undefined
            }
          : undefined;
      this.binding = {
        column: props.column,
        getPath: filtered?.getPath ?? getPath,
        getTimestamps: filtered?.getTimestamps ?? getTimestamps,
        selectionMask: filtered?.selectionMask ?? this.gather('mask', props.selectionMask),
        grouping: props.grouping,
        markerChannels,
        getColor: props.colors ? this.gather('colors', props.colors) : undefined,
        getWidth: props.widths ? this.gather('widths', props.widths) : undefined,
        originMilliseconds: source.originMilliseconds,
        maximumTimeErrorMilliseconds: source.maximumTimeErrorMilliseconds,
        animationDomainMilliseconds: source.animationDomainMilliseconds,
        resolveSourceRow: primitiveIndex => props.trips.resolveSourceRow(primitiveIndex),
        resolveHoverSourceRow: isTabularTrips(source)
          ? (primitiveIndex, currentTimeMilliseconds) =>
              resolveTabularHoverRow(source, primitiveIndex, currentTimeMilliseconds)
          : undefined
      };
    } catch (error) {
      this.releaseResources();
      throw error;
    }
  }

  resolve(column: string): ResidentTripBinding | undefined {
    return !this.destroyed && !this.reason && column === this.props.column
      ? this.binding
      : undefined;
  }
  prepare(): void {
    this.assertAlive();
    if (this.compiled) return;
    const compiled = this.graph.compile();
    if (
      this.outputByteLength + compiled.stats.physicalTransientResourceBytes >
      this.workingMemoryBudgetBytes
    ) {
      compiled.destroy();
      throw new KeplerAllocationError('memory-budget', {
        message: 'Compiled trip gather exceeds working memory budget'
      });
    }
    this.compiled = compiled;
  }
  encode(encoder: CommandEncoder): void {
    this.assertAlive();
    if (encoder.device !== this.props.device) throw new Error('Trip encoder device mismatch');
    this.prepare();
    this.compiled!.encode(encoder, {parameters: undefined});
  }
  get accountedResidentByteLength(): number {
    // Pending destruction still owns memory until its submitted work completes.
    return this.released
      ? 0
      : this.outputByteLength + (this.compiled?.stats.physicalTransientResourceBytes ?? 0);
  }
  get readbackByteLength(): number {
    return 0;
  }
  get resources(): readonly Buffer[] {
    return this.buffers;
  }

  /** Invalidates bindings now; releases only owned resources after caller-submitted work settles. */
  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.destruction = Promise.resolve(completion)
      .then(() => undefined)
      .finally(() => this.releaseResources());
    return this.destruction;
  }

  private get outputByteLength(): number {
    return this.buffers.reduce((total, buffer) => total + buffer.byteLength, 0);
  }
  /**
   * Native tabular filtering: a trip is visible while any member row is selected, and unselected
   * vertices are dropped from its path (see {@link TripMembershipFilter}).
   */
  private filterMembers(
    source: GPUVector<'uint32'>,
    membership: NonNullable<GeoJSONTripGPUVectors['membership']>
  ) {
    const {getPath, getTimestamps} = this.trips;
    const memberMask = this.allocate('member-mask', membership.memberRows, 'uint32');
    this.graph.add(
      new GPUGather({
        id: 'trip-member-mask',
        source: this.graph.importGPUVector('trip-member-mask-source', source),
        indices: this.graph.importGPUVector('trip-member-rows', membership.memberRows),
        output: this.graph.importGPUVector('trip-member-mask-output', memberMask)
      })
    );
    const selectionMask = this.allocate('mask', this.trips.sourceRowIds, 'uint32');
    const makeList = <Format extends 'vertex-list<float32x3>' | 'vertex-list<float32>'>(
      name: string,
      format: Format,
      wordsPerVertex: number,
      template: GPUVector<Format>
    ) => {
      const words: GPUData<'float32'>[] = [];
      const list = new GPUVector<Format>({
        type: 'data',
        name: `trip-${name}`,
        format,
        ownsData: false,
        data: template.data.map((chunk, index) => {
          const buffer = this.props.device.createBuffer({
            id: `trip-${name}-${index}`,
            byteLength: Math.max(4, chunk.valueLength * wordsPerVertex * 4),
            usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
          });
          this.buffers.push(buffer);
          words.push(
            new GPUData({
              format: 'float32',
              length: chunk.valueLength * wordsPerVertex,
              buffer,
              ownsBuffer: false
            })
          );
          return new GPUData({
            format,
            length: chunk.length,
            valueLength: chunk.valueLength,
            valueOffsets: chunk.valueOffsets,
            buffer,
            ownsBuffer: false
          });
        })
      });
      this.outputs.push(list);
      return {list, words: this.wrapWords(`${name}-words`, words)};
    };
    const positions = makeList('filtered-path', 'vertex-list<float32x3>', 3, getPath);
    const times = makeList('filtered-times', 'vertex-list<float32>', 1, getTimestamps);
    const alias = (name: string, template: GPUVector, wordsPerVertex: number) =>
      this.wrapWords(
        name,
        template.data.map(
          chunk =>
            new GPUData({
              format: 'float32',
              length: chunk.valueLength * wordsPerVertex,
              buffer: chunk.buffer,
              byteOffset: chunk.byteOffset,
              ownsBuffer: false
            })
        )
      );
    this.graph.add(
      new TripMembershipFilter({
        id: 'trip-membership-filter',
        offsets: this.graph.importGPUVector('trip-member-offsets', membership.memberOffsets),
        memberValues: this.graph.importGPUVector('trip-member-mask-values', memberMask),
        output: this.graph.importGPUVector('trip-mask-output', selectionMask),
        positions: this.graph.importGPUVector(
          'trip-source-position-words',
          alias('source-position-words', getPath, 3)
        ),
        times: this.graph.importGPUVector(
          'trip-source-time-words',
          alias('source-time-words', getTimestamps, 1)
        ),
        filteredPositions: this.graph.importGPUVector(
          'trip-filtered-position-words',
          positions.words
        ),
        filteredTimes: this.graph.importGPUVector('trip-filtered-time-words', times.words)
      })
    );
    return {
      selectionMask,
      memberMask,
      getPath: positions.list,
      getTimestamps: times.list
    };
  }
  private wrapWords(name: string, data: GPUData<'float32'>[]): GPUVector<'float32'> {
    const vector = new GPUVector<'float32'>({
      type: 'data',
      name: `trip-${name}`,
      format: 'float32',
      ownsData: false,
      data
    });
    this.outputs.push(vector);
    return vector;
  }
  private allocate<Format extends 'uint32' | 'float32' | 'unorm8x4'>(
    name: string,
    layout: GPUVector,
    format: Format
  ): GPUVector<Format> {
    const output = new GPUVector<Format>({
      type: 'data',
      name: `trip-${name}`,
      format,
      ownsData: false,
      data: layout.data.map((chunk, index) => {
        const buffer = this.props.device.createBuffer({
          id: `trip-${name}-${index}`,
          byteLength: Math.max(4, chunk.length * 4),
          usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        this.buffers.push(buffer);
        return new GPUData({format, length: chunk.length, buffer, ownsBuffer: false});
      })
    });
    this.outputs.push(output);
    return output;
  }
  private gather<Format extends 'uint32' | 'float32' | 'unorm8x4'>(
    name: string,
    source: GPUVector<Format>
  ): GPUVector<Format> {
    const indices = this.trips.sourceRowIds;
    const output = this.allocate(name, indices, source.format!);
    this.graph.add(
      new GPUGather({
        id: `trip-${name}`,
        source: this.graph.importGPUVector(`trip-${name}-source`, source),
        indices: this.graph.importGPUVector(`trip-${name}-indices`, indices),
        output: this.graph.importGPUVector(`trip-${name}-output`, output)
      })
    );
    return output;
  }
  /** Gathers one source column per member row, aligned with the flattened trip vertices. */
  private gatherMembers<Format extends 'uint32' | 'float32' | 'unorm8x4'>(
    name: string,
    source: GPUVector<Format>,
    memberRows: GPUVector<'uint32'>
  ): GPUVector<Format> {
    const output = this.allocate(name, memberRows, source.format!);
    this.graph.add(
      new GPUGather({
        id: `trip-${name}`,
        source: this.graph.importGPUVector(`trip-${name}-source`, source),
        indices: this.graph.importGPUVector(`trip-${name}-indices`, memberRows),
        output: this.graph.importGPUVector(`trip-${name}-output`, output)
      })
    );
    return output;
  }
  private releaseResources(): void {
    if (this.released) return;
    this.released = true;
    this.compiled?.destroy();
    for (const vector of this.outputs) vector.destroy();
    for (const buffer of this.buffers) buffer.destroy();
    this.reservation?.release();
  }
  private assertAlive(): void {
    if (this.destroyed) throw new Error('Trip runtime has been destroyed');
  }
}

/** Mirrors native `_findDatumForFeatureByTime` datum selection over unfiltered trip vertices. */
function resolveTabularHoverRow(
  source: PreparedGeoJSONTrips,
  primitiveIndex: number,
  currentTime: number
): number {
  if (!Number.isSafeInteger(primitiveIndex) || primitiveIndex < 0 || !Number.isFinite(currentTime))
    return -1;
  if (!isTabularTrips(source)) return -1;
  for (const batch of source.batches) {
    if (primitiveIndex >= batch.sourceRows.length) {
      primitiveIndex -= batch.sourceRows.length;
      continue;
    }
    const first = batch.valueOffsets[primitiveIndex];
    const end = batch.valueOffsets[primitiveIndex + 1];
    if (
      currentTime < batch.epochMilliseconds[first] ||
      currentTime > batch.epochMilliseconds[end - 1]
    )
      return -1;
    // bisectRight: first vertex strictly after the current time, clamped to the last vertex.
    let low = first;
    let high = end;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (batch.epochMilliseconds[middle] <= currentTime) low = middle + 1;
      else high = middle;
    }
    return batch.memberRows[Math.min(low, end - 1)];
  }
  return -1;
}
