// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The small data the building-blocks figures step through. Real rows come from building-blocks.json
// (see generate-data.ts); the radix keys and grid points are seeded so the prerendered HTML matches
// hydration. Every intermediate (prefix sums, histograms, offsets) is computed here on the CPU with
// the same definitions the GPU kernels implement, because the records only capture final values.

import recorded from './building-blocks.json';

export const RECORDED = recorded;

/** Exclusive prefix sum: `output[i]` is the sum of `values[0..i-1]`; one extra entry holds the total. */
export function getExclusiveScan(values: readonly number[]): number[] {
  const output = [0];
  for (const value of values) output.push(output[output.length - 1] + value);
  return output;
}

/** Deterministic 32-bit generator (mulberry32). */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

// Compaction: twenty recorded rows of the time-window filter.

export type CompactionRow = {
  row: number;
  timestamp: number;
  flag: number;
  /** Exclusive scan of the flags over the toy rows: where a kept row writes. */
  position: number;
  /** Slot of the row in the recorded `ids` of the full run, or -1. */
  recordedSlot: number;
};

const compactionPositions = getExclusiveScan(recorded.timeWindow.rows.map(row => row.flag));
export const COMPACTION_ROWS: readonly CompactionRow[] = recorded.timeWindow.rows.map(
  (row, index) => ({...row, position: compactionPositions[index]})
);
export const COMPACTION_KEPT = compactionPositions[compactionPositions.length - 1];

// Reduction: the recorded mask summed over 16 slices; each level halves the count.

export function getReductionLevels(leaves: readonly number[]): number[][] {
  const levels = [leaves.slice()];
  while (levels[levels.length - 1].length > 1) {
    const previous = levels[levels.length - 1];
    const next: number[] = [];
    for (let index = 0; index < previous.length; index += 2) {
      next.push(previous[index] + (previous[index + 1] ?? 0));
    }
    levels.push(next);
  }
  return levels;
}
export const REDUCTION_LEVELS = getReductionLevels(recorded.reductionLeaves.map(leaf => leaf.kept));

// Radix sort: twelve 4-bit keys, two passes of 2-bit digits.

export const RADIX_DIGIT_BITS = 2;
export const RADIX_BUCKETS = 1 << RADIX_DIGIT_BITS;
const radixRandom = createRandom(20261110);
export const RADIX_KEYS: readonly number[] = Array.from({length: 12}, () =>
  Math.floor(radixRandom() * 16)
);

export type RadixItem = {key: number; source: number};
export type RadixPass = {
  bitOffset: number;
  input: RadixItem[];
  histogram: number[];
  /** Exclusive scan of the histogram: the first slot of each digit. */
  starts: number[];
  /** Target slot of every input position. */
  targets: number[];
  output: RadixItem[];
};

export function getDigit(key: number, bitOffset: number): number {
  return (key >> bitOffset) & (RADIX_BUCKETS - 1);
}

function getRadixPasses(keys: readonly number[]): RadixPass[] {
  let items: RadixItem[] = keys.map((key, source) => ({key, source}));
  const passes: RadixPass[] = [];
  for (let bitOffset = 0; bitOffset < 4; bitOffset += RADIX_DIGIT_BITS) {
    const histogram = new Array<number>(RADIX_BUCKETS).fill(0);
    for (const item of items) histogram[getDigit(item.key, bitOffset)]++;
    const starts = getExclusiveScan(histogram).slice(0, RADIX_BUCKETS);
    const cursors = starts.slice();
    // Stable: equal digits keep their order, which is what lets the next pass build on this one.
    const targets = items.map(item => cursors[getDigit(item.key, bitOffset)]++);
    const output = new Array<RadixItem>(items.length);
    items.forEach((item, index) => {
      output[targets[index]] = item;
    });
    passes.push({bitOffset, input: items, histogram, starts, targets, output});
    items = output;
  }
  return passes;
}
export const RADIX_PASSES = getRadixPasses(RADIX_KEYS);

// Grid index: sixty seeded points on a 6 x 6 grid.

export const GRID_SIZE = 6;
export const GRID_CELL_COUNT = GRID_SIZE * GRID_SIZE;
export type GridPoint = {id: number; x: number; y: number; cell: number};

const gridRandom = createRandom(73);
// One point in three falls near a cluster, the rest anywhere, so cell counts vary.
const CLUSTERS = [{x: 0.68, y: 0.64, spread: 0.1}];
export const GRID_POINTS: readonly GridPoint[] = Array.from({length: 60}, (_, id) => {
  const cluster = id % 3 === 0 ? CLUSTERS[0] : undefined;
  let x: number;
  let y: number;
  if (cluster) {
    // Sums of uniforms, roughly normal. Arithmetic only: transcendental functions may differ in the
    // last bit between the server and the browser and break hydration.
    x = cluster.x + (gridRandom() + gridRandom() + gridRandom() - 1.5) * cluster.spread * 2;
    y = cluster.y + (gridRandom() + gridRandom() + gridRandom() - 1.5) * cluster.spread * 2;
  } else {
    x = gridRandom();
    y = gridRandom();
  }
  x = Math.round(Math.min(0.995, Math.max(0.005, x)) * 1000) / 1000;
  y = Math.round(Math.min(0.995, Math.max(0.005, y)) * 1000) / 1000;
  const column = Math.floor(x * GRID_SIZE);
  const row = Math.floor(y * GRID_SIZE);
  return {id, x, y, cell: row * GRID_SIZE + column};
});

export const GRID_COUNTS: readonly number[] = (() => {
  const counts = new Array<number>(GRID_CELL_COUNT).fill(0);
  for (const point of GRID_POINTS) counts[point.cell]++;
  return counts;
})();
/** `cellCount + 1` exclusive offsets, like the recorded `kernel-density-cell-offsets`. */
export const GRID_OFFSETS = getExclusiveScan(GRID_COUNTS);
/** Point ids grouped by cell, ascending inside a cell (what `sort-cells` restores). */
export const GRID_SORTED: readonly GridPoint[] = GRID_POINTS.slice().sort(
  (left, right) => left.cell - right.cell || left.id - right.id
);

/** The ranges of the sorted ids a query cell reads: one per grid row of its neighbourhood. */
export function getQueryRanges(column: number, row: number, radius = 1) {
  const ranges: {row: number; first: number; last: number; begin: number; end: number}[] = [];
  const first = Math.max(0, column - radius);
  const last = Math.min(GRID_SIZE - 1, column + radius);
  for (
    let gridRow = Math.max(0, row - radius);
    gridRow <= Math.min(GRID_SIZE - 1, row + radius);
    gridRow++
  ) {
    ranges.push({
      row: gridRow,
      first,
      last,
      begin: GRID_OFFSETS[gridRow * GRID_SIZE + first],
      end: GRID_OFFSETS[gridRow * GRID_SIZE + last + 1]
    });
  }
  return ranges;
}
