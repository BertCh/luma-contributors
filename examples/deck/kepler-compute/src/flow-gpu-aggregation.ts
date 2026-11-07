// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure (CPU, no GPU import) half of the GPU-resident Flow aggregation.
 *
 * Everything here is deterministic and runs in Node: the float32 key packing shared with the WGSL
 * kernels, the streaming Arrow column reader used for the single explicit upload pass, and a CPU
 * reference implementation of location dedupe and origin/destination aggregation. The GPU kernels in
 * `flow-gpu-pipeline.ts` are specified by (and should be compared against) the reference here.
 *
 * Deliberate differences from native Kepler Flow, documented once:
 * - Locations are deduplicated by float32 `(latitude bits, longitude bits)`, not by float64
 *   `lat:lon` strings. Distinct float64 coordinates that round to one float32 merge into one
 *   location (the first-seen location keeps its exact float64 coordinates and name).
 * - Flows are aggregated per `(sourceId, targetId)` pair: duplicate origin-destination rows become
 *   one datum carrying the summed count and a representative first row. Magnitudes, location
 *   totals and clusters are equivalent; picking returns the aggregate.
 * - Counts are summed in float32 on the GPU.
 */

import * as arrow from 'apache-arrow';

/** Word that marks an invalid endpoint or pair key; it sorts after every real float32 key. */
export const FLOW_INVALID_WORD = 0xffffffff;

/** Rejection limit on unique locations after dedupe (replaces the former row cap). */
export const MAXIMUM_FLOW_LOCATIONS = 262_144;
/** Rejection limit on unique origin-destination pairs after aggregation (replaces the row cap). */
export const MAXIMUM_FLOW_AGGREGATED_PAIRS = 4_194_304;

/** Estimated GPU bytes per row that stay resident for the runtime lifetime. */
export const FLOW_RETAINED_BYTES_PER_ROW = 12;
/** Estimated peak GPU bytes per row of the location dedupe phase (endpoints are 2 per row). */
export const FLOW_LOCATION_WORKING_BYTES_PER_ROW = 320;
/** Estimated peak GPU bytes per row of one origin-destination aggregation. */
export const FLOW_PAIR_WORKING_BYTES_PER_ROW = 112;

/** Aggregated origin-destination pairs, ordered by `(sourceId, targetId)`. */
export type FlowPairs = {
  sourceIds: Uint32Array;
  targetIds: Uint32Array;
  /** Summed count per pair. */
  counts: Float64Array;
  /** Representative (smallest selected) source row per pair. */
  firstRows: Uint32Array;
};

/** Selection mask handle; the aggregator reads it (GPU vector or test stand-in). */
export type FlowSelectionMask = unknown;

/**
 * Resident aggregation of one Flow table. Locations are fixed at creation; pair aggregation runs
 * per selection revision. Readback counters are cumulative.
 */
export type FlowAggregateSource = {
  rowCount: number;
  /** Endpoint index (`2 * row + (isTarget ? 1 : 0)`) of each location's first occurrence, in first-seen order. */
  firstEndpoints: Uint32Array;
  bounds: [number, number, number, number] | null;
  /** Reserved resident GPU bytes. */
  gpuByteLength: number;
  readbackCount: number;
  readbackBytes: number;
  /** Aggregates selected rows; `undefined` means all rows. */
  aggregatePairs(mask?: FlowSelectionMask): Promise<FlowPairs>;
  destroy(): void;
};

/**
 * Float32 key words of one endpoint, or `null` when either coordinate is not finite in float32.
 * Negative zero is normalized so that `-0` and `0` match, like the native `lat:lon` string key.
 */
export function getFlowEndpointKey(
  longitude: number,
  latitude: number
): {latitudeBits: number; longitudeBits: number} | null {
  const words = new Uint32Array(2);
  const floats = new Float32Array(words.buffer);
  floats[0] = latitude;
  floats[1] = longitude;
  const [latitudeBits, longitudeBits] = words;
  if ((latitudeBits & 0x7f800000) === 0x7f800000 || (longitudeBits & 0x7f800000) === 0x7f800000)
    return null;
  return {
    latitudeBits: (latitudeBits & 0x7fffffff) === 0 ? 0 : latitudeBits,
    longitudeBits: (longitudeBits & 0x7fffffff) === 0 ? 0 : longitudeBits
  };
}

/** Result of location dedupe over `2 * rowCount` endpoints. */
export type FlowEndpointLocations = {
  /** Location id per endpoint (`2 * row` source, `2 * row + 1` target); 0 when not finite. */
  endpointIds: Uint32Array;
  /** First endpoint index per location; location `k` has id `k + 1`. */
  firstEndpoints: Uint32Array;
};

/**
 * CPU reference for the GPU location dedupe: first-seen ids from 1, source before target within a
 * row, a non-finite endpoint gets id 0 and creates no location.
 */
export function deduplicateFlowEndpoints(columns: {
  sourceLongitude: ArrayLike<number>;
  sourceLatitude: ArrayLike<number>;
  targetLongitude: ArrayLike<number>;
  targetLatitude: ArrayLike<number>;
}): FlowEndpointLocations {
  const rowCount = columns.sourceLongitude.length;
  const endpointIds = new Uint32Array(rowCount * 2);
  const firstEndpoints: number[] = [];
  const idsByKey = new Map<string, number>();
  for (let endpoint = 0; endpoint < rowCount * 2; endpoint++) {
    const row = endpoint >> 1;
    const isTarget = endpoint & 1;
    const key = getFlowEndpointKey(
      (isTarget ? columns.targetLongitude : columns.sourceLongitude)[row],
      (isTarget ? columns.targetLatitude : columns.sourceLatitude)[row]
    );
    if (!key) continue;
    const text = `${key.latitudeBits}:${key.longitudeBits}`;
    let id = idsByKey.get(text);
    if (id === undefined) {
      id = idsByKey.size + 1;
      idsByKey.set(text, id);
      firstEndpoints.push(endpoint);
    }
    endpointIds[endpoint] = id;
  }
  return {endpointIds, firstEndpoints: Uint32Array.from(firstEndpoints)};
}

/**
 * CPU reference for the GPU origin-destination aggregation. Rows with an unresolved endpoint or a
 * zero selection are skipped; output is ordered by `(sourceId, targetId)`.
 */
export function aggregateFlowPairs(
  endpointIds: Uint32Array,
  counts: ArrayLike<number>,
  selected?: ArrayLike<number>
): FlowPairs {
  const pairs = new Map<
    string,
    {sourceId: number; targetId: number; count: number; firstRow: number}
  >();
  for (let row = 0; row < counts.length; row++) {
    if (selected && !selected[row]) continue;
    const sourceId = endpointIds[row * 2];
    const targetId = endpointIds[row * 2 + 1];
    if (!sourceId || !targetId) continue;
    const key = `${sourceId}:${targetId}`;
    const pair = pairs.get(key);
    if (pair) pair.count += counts[row];
    else pairs.set(key, {sourceId, targetId, count: counts[row], firstRow: row});
  }
  const ordered = Array.from(pairs.values()).sort(
    (first, second) => first.sourceId - second.sourceId || first.targetId - second.targetId
  );
  return {
    sourceIds: Uint32Array.from(ordered, pair => pair.sourceId),
    targetIds: Uint32Array.from(ordered, pair => pair.targetId),
    counts: Float64Array.from(ordered, pair => pair.count),
    firstRows: Uint32Array.from(ordered, pair => pair.firstRow)
  };
}

/** Bounds over endpoints whose own two coordinates are finite, like native `addLocation`. */
export function getFlowBounds(columns: {
  sourceLongitude: ArrayLike<number>;
  sourceLatitude: ArrayLike<number>;
  targetLongitude: ArrayLike<number>;
  targetLatitude: ArrayLike<number>;
}): [number, number, number, number] | null {
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  const include = (longitude: number, latitude: number) => {
    if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) return;
    bounds[0] = Math.min(bounds[0], longitude);
    bounds[1] = Math.min(bounds[1], latitude);
    bounds[2] = Math.max(bounds[2], longitude);
    bounds[3] = Math.max(bounds[3], latitude);
  };
  for (let row = 0; row < columns.sourceLongitude.length; row++) {
    include(columns.sourceLongitude[row], columns.sourceLatitude[row]);
    include(columns.targetLongitude[row], columns.targetLatitude[row]);
  }
  return bounds.every(Number.isFinite) ? bounds : null;
}

/**
 * Sequential float32 reader over one Arrow column. It walks the column's own chunks without
 * concatenating them: typed-array chunks without nulls are converted directly, other chunks fall
 * back to per-value reads. Non-numeric values and nulls read as `NaN`.
 */
export class FlowColumnReader {
  private chunkIndex = 0;
  private chunkOffset = 0;

  constructor(private readonly column: arrow.Vector | null) {}

  /** Reads the next `length` rows as float32. A missing column reads as `missingValue`. */
  readFloat32(length: number, missingValue = Number.NaN): Float32Array {
    const output = new Float32Array(length);
    if (!this.column) return output.fill(missingValue);
    let written = 0;
    while (written < length) {
      const chunk = this.column.data[this.chunkIndex];
      const available = chunk.length - this.chunkOffset;
      if (available <= 0) {
        this.chunkIndex++;
        this.chunkOffset = 0;
        continue;
      }
      const span = Math.min(available, length - written);
      const start = chunk.offset + this.chunkOffset;
      const values: unknown = chunk.values;
      if (
        chunk.nullCount === 0 &&
        ArrayBuffer.isView(values) &&
        (arrow.DataType.isFloat(chunk.type) || arrow.DataType.isInt(chunk.type)) &&
        !(values instanceof BigInt64Array) &&
        !(values instanceof BigUint64Array)
      ) {
        output.set((values as Float32Array).subarray(start, start + span), written);
      } else {
        const view = new arrow.Vector([chunk]);
        for (let index = 0; index < span; index++) {
          const value: unknown = view.get(this.chunkOffset + index);
          output[written + index] = typeof value === 'number' ? value : Number.NaN;
        }
      }
      this.chunkOffset += span;
      written += span;
    }
    return output;
  }
}

/** Replaces null and non-finite counts by zero in place. */
export function sanitizeFlowCounts(counts: Float32Array): Float32Array {
  for (let index = 0; index < counts.length; index++)
    if (!Number.isFinite(counts[index])) counts[index] = 0;
  return counts;
}

/**
 * CPU reference aggregator: the oracle for the GPU pipeline and the Node-testable stand-in.
 * It runs the same streaming column pass but keeps the arrays on the CPU.
 */
export class FlowReferenceAggregator implements FlowAggregateSource {
  readonly rowCount: number;
  readonly firstEndpoints: Uint32Array;
  readonly bounds: [number, number, number, number] | null;
  readonly gpuByteLength = 0;
  readbackCount = 0;
  readbackBytes = 0;
  private readonly endpointIds: Uint32Array;
  private readonly counts: Float32Array;

  constructor(
    columns: {
      sourceLongitude: arrow.Vector | null;
      sourceLatitude: arrow.Vector | null;
      targetLongitude: arrow.Vector | null;
      targetLatitude: arrow.Vector | null;
      count: arrow.Vector | null;
    },
    rowCount: number,
    private readonly readMask: (mask: FlowSelectionMask) => Uint32Array | undefined = () =>
      undefined
  ) {
    this.rowCount = rowCount;
    const coordinates = {
      sourceLongitude: new FlowColumnReader(columns.sourceLongitude).readFloat32(rowCount),
      sourceLatitude: new FlowColumnReader(columns.sourceLatitude).readFloat32(rowCount),
      targetLongitude: new FlowColumnReader(columns.targetLongitude).readFloat32(rowCount),
      targetLatitude: new FlowColumnReader(columns.targetLatitude).readFloat32(rowCount)
    };
    this.counts = sanitizeFlowCounts(new FlowColumnReader(columns.count).readFloat32(rowCount, 1));
    this.bounds = getFlowBounds(coordinates);
    const locations = deduplicateFlowEndpoints(coordinates);
    this.endpointIds = locations.endpointIds;
    this.firstEndpoints = locations.firstEndpoints;
  }

  async aggregatePairs(mask?: FlowSelectionMask): Promise<FlowPairs> {
    this.readbackCount++;
    const pairs = aggregateFlowPairs(
      this.endpointIds,
      this.counts,
      mask === undefined ? undefined : this.readMask(mask)
    );
    this.readbackBytes += pairs.sourceIds.length * 16 + 8;
    return pairs;
  }

  destroy(): void {}
}
