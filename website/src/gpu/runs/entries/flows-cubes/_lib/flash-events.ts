// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The timestamped events behind the cube and period runs: Midwest GLM flashes with measured times.
// Times are Int64 epoch-millisecond words, the way an Arrow timestamp column reaches the GPU, and a
// cell id per flash from a coarse lattice (a table would carry one from a spatial join).

import {getInt64TimeWords} from '@luma-contributors/analysis/dataframe-analysis';
import {sampleFlashes} from '../../../../data/lightning';

/** Flashes in the checked-in extract. */
export const FLASH_COUNT = 4096;
/** Hours the extract spans (the first flash is at 12:00 UTC on May 21, 2024). */
export const FLASH_HOURS = 18;
/** Start of the extract in milliseconds after the Unix epoch. */
export const FLASH_ORIGIN_MILLISECONDS = BigInt(Date.UTC(2024, 4, 21, 12));
export const MILLISECONDS_PER_HOUR = 3_600_000;

/** The temporal context of the Int64 millisecond words; the period index and relative times read it. */
export const FLASH_TEMPORAL_CONTEXT = {
  unit: 'millisecond',
  epoch: 'unix',
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
} as const;

export type FlashEvents = ReturnType<typeof createFlashEvents>;

/**
 * Flash positions in unit coordinates (x east, y north, both 0 to 1), hours since the extract
 * start, flash energy in femtojoules, and the epoch-millisecond Int64 words of each time.
 */
export function createFlashEvents() {
  const flashes = sampleFlashes({width: 1, height: 1}, FLASH_COUNT);
  const milliseconds = BigInt64Array.from(
    flashes.hours,
    hours => FLASH_ORIGIN_MILLISECONDS + BigInt(Math.round(hours * MILLISECONDS_PER_HOUR))
  );
  return {
    count: FLASH_COUNT,
    positions: flashes.positions,
    hours: flashes.hours,
    energy: flashes.energy,
    timestampWords: getInt64TimeWords(milliseconds)
  };
}

/** Row-major lattice cell of each position, row 0 at the north edge so it matches a heatmap's rows. */
export function createLatticeCells(
  positions: Float32Array,
  columns: number,
  rows: number
): Uint32Array {
  const cells = new Uint32Array(positions.length / 2);
  for (let row = 0; row < cells.length; row++) {
    const column = Math.min(columns - 1, Math.floor(positions[row * 2] * columns));
    const north = Math.min(rows - 1, Math.floor((1 - positions[row * 2 + 1]) * rows));
    cells[row] = north * columns + column;
  }
  return cells;
}

/** Per-bin counts, sums and means of energy in one-hour periods, computed on the CPU for fixed color ranges. */
export function summarizeCube(
  events: Pick<FlashEvents, 'hours' | 'energy'>,
  cells: Uint32Array,
  cellCount: number,
  periodCount: number
) {
  const counts = new Float64Array(cellCount * periodCount);
  const sums = new Float64Array(cellCount * periodCount);
  for (let row = 0; row < cells.length; row++) {
    const period = Math.floor(events.hours[row]);
    if (period >= periodCount) continue;
    counts[cells[row] * periodCount + period]++;
    sums[cells[row] * periodCount + period] += events.energy[row];
  }
  let busiest = 0;
  let maximumSum = 1;
  let maximumMean = 1;
  for (let bin = 0; bin < counts.length; bin++) {
    if (counts[bin] > counts[busiest]) busiest = bin;
    maximumSum = Math.max(maximumSum, sums[bin]);
    if (counts[bin] > 0) maximumMean = Math.max(maximumMean, sums[bin] / counts[bin]);
  }
  return {
    busiestCell: Math.floor(busiest / periodCount),
    busiestPeriod: busiest % periodCount,
    maximumCount: counts[busiest],
    maximumSum,
    maximumMean
  };
}
