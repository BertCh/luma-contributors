// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Planning and CPU reference for lifting Group By bounds: key-slot chunking, hashed key slots,
 * measure batching and the extra measures median, standard deviation, count-distinct plus
 * compensated (Neumaier) sums. These are the pure contract the GPU hashing/chunk kernels must match;
 * no GPU kernel is implemented here.
 */

export const GROUP_BY_KEY_SLOTS_PER_CHUNK = 4096;
export const GROUP_BY_MEASURES_PER_BATCH = 16;

export type GroupByChunkPlan = {
  chunkCount: number;
  /** [firstSlot, endSlot) per chunk; the null group occupies slot 0 of chunk 0 when present. */
  ranges: [number, number][];
};

/** Partition distinct keys (plus an optional null group) into chunks of at most the slot capacity. */
export function planGroupByKeyChunks(
  distinctKeyCount: number,
  hasNullKey: boolean,
  slotsPerChunk = GROUP_BY_KEY_SLOTS_PER_CHUNK
): GroupByChunkPlan {
  const total = distinctKeyCount + (hasNullKey ? 1 : 0);
  const ranges: [number, number][] = [];
  for (let start = 0; start < total; start += slotsPerChunk)
    ranges.push([start, Math.min(total, start + slotsPerChunk)]);
  return {chunkCount: ranges.length, ranges};
}

/** Split measures into batches that each fit the per-pass measure limit, preserving order. */
export function planGroupByMeasureBatches<T>(
  measures: readonly T[],
  perBatch = GROUP_BY_MEASURES_PER_BATCH
): T[][] {
  const batches: T[][] = [];
  for (let index = 0; index < measures.length; index += perBatch)
    batches.push(measures.slice(index, index + perBatch));
  return batches;
}

/** FNV-1a 32-bit over UTF-16 code units (deterministic, shader-portable). */
export function hashGroupKey(key: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < key.length; index++) {
    hash ^= key.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

/**
 * Open-addressing assignment of distinct keys to slots in first-occurrence order of insertion.
 * Returns the slot of each input row (-1 for null) and the distinct key list (first occurrence
 * order), or null when the table would exceed 50 percent load (caller grows capacity or chunks).
 */
export function assignGroupSlots(
  keys: ArrayLike<string | null>,
  slotCapacity: number
): {rowSlots: Int32Array; distinctKeys: string[]; probes: number} | null {
  const table = new Int32Array(slotCapacity).fill(-1);
  const distinctKeys: string[] = [];
  const rowSlots = new Int32Array(keys.length);
  let probes = 0;
  for (let row = 0; row < keys.length; row++) {
    const key = keys[row];
    if (key === null) {
      rowSlots[row] = -1;
      continue;
    }
    let slot = hashGroupKey(key) % slotCapacity;
    for (;;) {
      probes++;
      const occupant = table[slot];
      if (occupant === -1) {
        if ((distinctKeys.length + 1) * 2 > slotCapacity) return null;
        table[slot] = distinctKeys.length;
        distinctKeys.push(key);
        rowSlots[row] = distinctKeys.length - 1;
        break;
      }
      if (distinctKeys[occupant] === key) {
        rowSlots[row] = occupant;
        break;
      }
      slot = (slot + 1) % slotCapacity;
    }
  }
  return {rowSlots, distinctKeys, probes};
}

/** Neumaier compensated summation in Float32 arithmetic (Math.fround each step). */
export class CompensatedFloat32Sum {
  sum = 0;
  compensation = 0;
  add(value: number): void {
    const next = Math.fround(this.sum + value);
    if (Math.abs(this.sum) >= Math.abs(value))
      this.compensation = Math.fround(
        this.compensation + Math.fround(Math.fround(this.sum - next) + value)
      );
    else
      this.compensation = Math.fround(
        this.compensation + Math.fround(Math.fround(value - next) + this.sum)
      );
    this.sum = next;
  }
  get value(): number {
    return Math.fround(this.sum + this.compensation);
  }
}

export type ExtendedGroupMeasure =
  | 'median'
  | 'standardDeviation'
  | 'countDistinct'
  | 'compensatedSum';

/** Reference per-group reduction over finite non-null values (nulls skipped, as native). */
export function reduceGroupMeasure(
  operation: ExtendedGroupMeasure,
  values: readonly (number | null)[]
): number | null {
  const present = values.filter(
    (value): value is number => value !== null && Number.isFinite(value)
  );
  if (operation === 'countDistinct') return new Set(present).size;
  if (!present.length) return null;
  if (operation === 'compensatedSum') {
    const accumulator = new CompensatedFloat32Sum();
    for (const value of present) accumulator.add(Math.fround(value));
    return accumulator.value;
  }
  if (operation === 'median') {
    const sorted = [...present].sort((first, second) => first - second);
    const middle = sorted.length >> 1;
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }
  // Population standard deviation by Welford's algorithm.
  let mean = 0;
  let m2 = 0;
  for (const [index, value] of present.entries()) {
    const delta = value - mean;
    mean += delta / (index + 1);
    m2 += delta * (value - mean);
  }
  return Math.sqrt(m2 / present.length);
}
