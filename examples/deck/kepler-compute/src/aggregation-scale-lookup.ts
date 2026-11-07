// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Data-dependent aggregate color scales (Kepler `jenks` and deck.gl `ordinal`). Both classify a
 * per-cell value by a small lookup that depends on the whole set of aggregated cell values, so
 * the resident pipeline reads back the per-cell color measure once per selection revision
 * (bounded by the cell budget, never source rows), derives the lookup on the CPU exactly like
 * native Kepler/deck.gl, and uploads it into a storage buffer that the style kernel searches.
 */

/** Lookup buffer header: element 0 = ready flag (1 once derived), element 1 = entry count. */
export const SCALE_LOOKUP_HEADER_LENGTH = 2;
/** Maximum jenks thresholds: palettes hold at most 64 colors. */
export const MAXIMUM_JENKS_THRESHOLDS = 63;

/** Float32 length of the lookup buffer for a cell count: jenks needs 65, ordinal 2 + 2 * cells. */
export function getScaleLookupLength(cellCount: number): number {
  return Math.max(
    SCALE_LOOKUP_HEADER_LENGTH + MAXIMUM_JENKS_THRESHOLDS,
    SCALE_LOOKUP_HEADER_LENGTH + 2 * cellCount
  );
}

function hashJenksSeed(input: number): number {
  let value = input;
  value += (value << 10) >>> 0;
  if (value > 4294967295) value = value & 4294967295;
  value ^= value >> 6;
  value += (value << 3) >>> 0;
  if (value > 4294967295) value = value & 4294967295;
  value ^= value >> 11;
  value += (value << 15) >>> 0;
  if (value > 4294967295) value = value & 4294967295;
  return value;
}

function createMulberry32(seed: number): () => number {
  let state = seed;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Port of Kepler's `naturalBreaks` (@kepler.gl/utils jenks): the same seeded permutation search, so
 * resident jenks thresholds match native for identical aggregated values. Returns k - 1 thresholds
 * (padded with the last break when there are fewer distinct values than classes).
 */
export function getJenksBreaks(data: readonly number[], classCount: number): number[] {
  const dataSize = data.length;
  const sortedData = Float64Array.from(data).sort();
  const uniquePositions: number[] = [];
  for (let index = 0; index < dataSize; index++)
    if (index === 0 || sortedData[index] !== sortedData[index - 1]) uniquePositions.push(index);
  const uniqueCount = uniquePositions.length;
  let permutationCount = Math.floor(44000000 / Math.max(dataSize, 1));
  if (permutationCount < 10) permutationCount = 10;
  if (permutationCount > 1000) permutationCount = 1000;
  if (classCount <= 1 || uniqueCount <= 0) return [];
  if (classCount >= uniqueCount) {
    let breaks = uniquePositions.map(position => sortedData[position]);
    if (uniqueCount > 1) breaks = breaks.slice(1);
    const lastBreak = breaks[breaks.length - 1];
    for (let index = breaks.length; index < classCount - 1; index++) breaks.push(lastBreak);
    return breaks;
  }
  let sum = 0;
  for (let index = 0; index < dataSize; index++) sum += sortedData[index];
  const mean = dataSize === 0 ? 0 : sum / dataSize;
  let globalSquaredDeviation = 0;
  for (let index = 0; index < dataSize; index++) {
    const difference = sortedData[index] - mean;
    globalSquaredDeviation += difference * difference;
  }
  let bestGoodnessOfVarianceFit = 0;
  let bestBreaks: number[] = [];
  for (let permutation = 0; permutation < permutationCount; permutation++) {
    const random = createMulberry32(hashJenksSeed(permutation));
    const candidates = uniquePositions.slice(1);
    const positions = new Set<number>();
    while (positions.size < classCount - 1)
      positions.add(candidates[Math.floor(random() * candidates.length)]);
    const sortedPositions = Array.from(positions).sort((first, second) => first - second);
    let totalSquaredDeviation = 0;
    for (let classIndex = 0; classIndex < classCount; classIndex++) {
      const start = classIndex === 0 ? 0 : sortedPositions[classIndex - 1];
      const end = classIndex === classCount - 1 ? dataSize : sortedPositions[classIndex];
      let localMean = 0;
      for (let index = start; index < end; index++) localMean += sortedData[index];
      localMean /= end - start;
      for (let index = start; index < end; index++) {
        const difference = sortedData[index] - localMean;
        totalSquaredDeviation += difference * difference;
      }
    }
    const goodnessOfVarianceFit =
      globalSquaredDeviation === 0 ? 1 : 1 - totalSquaredDeviation / globalSquaredDeviation;
    if (goodnessOfVarianceFit > bestGoodnessOfVarianceFit) {
      bestGoodnessOfVarianceFit = goodnessOfVarianceFit;
      bestBreaks = sortedPositions.map(position => sortedData[position]);
    }
  }
  return bestBreaks;
}

/** Native classifyBinsByCustomBreaks for jenks: index of the first threshold above the value. */
export function getReferenceJenksIndex(value: number, breaks: readonly number[]): number {
  if (!Number.isFinite(value)) return NaN;
  for (let index = 0; index < breaks.length; index++) if (value < breaks[index]) return index;
  return breaks.length;
}

export type OrdinalLookup = {
  /** Distinct finite values, ascending numerically (binary-searchable on the GPU). */
  values: number[];
  /** Deck rank of each value: its index in the default-sorted (string order) distinct values. */
  ranks: number[];
};

/**
 * deck.gl `applyScaleOrdinal`: distinct finite values are ranked by `Array.prototype.sort()` with no
 * comparator (string order), so the rank table is kept beside a numerically sorted value list.
 */
export function getOrdinalLookup(data: ArrayLike<number>): OrdinalLookup {
  const distinct = new Set<number>();
  for (let index = 0; index < data.length; index++)
    if (Number.isFinite(data[index])) distinct.add(data[index]);
  const stringOrder = Array.from(distinct).sort();
  const rankByValue = new Map<number, number>();
  stringOrder.forEach((value, rank) => rankByValue.set(value, rank));
  const values = Array.from(distinct).sort((first, second) => first - second);
  return {values, ranks: values.map(value => rankByValue.get(value)!)};
}

/**
 * CPU oracle of native `wrapOrdinalAccessor` for categorical mode given cells already in native bin
 * order (first-appearance order of the cell's first source row): each distinct label maps to its
 * index of first appearance, and deck then ranks those integer indices in string order. The
 * resident pipeline no longer calls this: the GPU computes the first-appearance indices exactly
 * (minimum source row per cell, see aggregation-statistics-gpu.ts) and `getOrdinalLookup` ranks them.
 * Kept for specs and as the reference for `getFirstAppearanceIndices`.
 */
export function getCategoricalOrdinalLookup(codes: ArrayLike<number>): OrdinalLookup {
  const firstAppearance = new Map<number, number>();
  for (let index = 0; index < codes.length; index++) {
    const code = codes[index];
    if (Number.isFinite(code) && !firstAppearance.has(code))
      firstAppearance.set(code, firstAppearance.size);
  }
  const rankByIndex = new Map<number, number>();
  Array.from(firstAppearance.values())
    .sort()
    .forEach((nativeIndex, rank) => rankByIndex.set(nativeIndex, rank));
  const values = Array.from(firstAppearance.keys()).sort((first, second) => first - second);
  return {values, ranks: values.map(code => rankByIndex.get(firstAppearance.get(code)!)!)};
}

export function getReferenceOrdinalIndex(value: number, lookup: OrdinalLookup): number {
  const index = lookup.values.indexOf(value);
  return Number.isFinite(value) && index >= 0 ? lookup.ranks[index] : NaN;
}

/** Float32 buffer contents: header, then jenks thresholds or ordinal values followed by ranks. */
export function packScaleLookup(
  kind: 'jenks' | 'ordinal',
  source: readonly number[] | OrdinalLookup,
  length: number
): Float32Array {
  const packed = new Float32Array(length);
  packed[0] = 1;
  if (kind === 'jenks') {
    const breaks = source as readonly number[];
    packed[1] = breaks.length;
    packed.set(breaks, SCALE_LOOKUP_HEADER_LENGTH);
  } else {
    const lookup = source as OrdinalLookup;
    packed[1] = lookup.values.length;
    packed.set(lookup.values, SCALE_LOOKUP_HEADER_LENGTH);
    packed.set(lookup.ranks, SCALE_LOOKUP_HEADER_LENGTH + lookup.values.length);
  }
  return packed;
}

/** WGSL helpers over `scaleLookup` (f32 storage) and its element offset `scaleLookupOffset`. */
export const SCALE_LOOKUP_WGSL = /* wgsl */ `fn getJenksIndex(v: f32) -> f32 {
  if (scaleLookup[scaleLookupOffset] < 0.5) { return getNaN(); }
  let count = u32(scaleLookup[scaleLookupOffset + 1u]);
  for (var i = 0u; i < count; i++) {
    if (v < scaleLookup[scaleLookupOffset + 2u + i]) { return f32(i); }
  }
  return f32(count);
}
fn getOrdinalIndex(v: f32) -> f32 {
  if (scaleLookup[scaleLookupOffset] < 0.5) { return getNaN(); }
  let count = u32(scaleLookup[scaleLookupOffset + 1u]);
  var low = 0u;
  var high = count;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (scaleLookup[scaleLookupOffset + 2u + middle] < v) { low = middle + 1u; } else { high = middle; }
  }
  if (low < count && scaleLookup[scaleLookupOffset + 2u + low] == v) {
    return scaleLookup[scaleLookupOffset + 2u + count + low];
  }
  return getNaN();
}`;

/**
 * Reference for the GPU remap: given each cell's first source row and dictionary code (NaN code or
 * unused cell = skipped), returns code -> native first-appearance index. Cells are visited in
 * ascending first-row order, which is the native deck.gl CPU aggregator bin order.
 */
export function getFirstAppearanceIndices(
  firstRows: ArrayLike<number>,
  codes: ArrayLike<number>
): Map<number, number> {
  const order = Array.from({length: codes.length}, (_, cell) => cell)
    .filter(cell => Number.isFinite(codes[cell]) && firstRows[cell] !== 0xffffffff)
    .sort((first, second) => firstRows[first] - firstRows[second]);
  const indices = new Map<number, number>();
  for (const cell of order) if (!indices.has(codes[cell])) indices.set(codes[cell], indices.size);
  return indices;
}
