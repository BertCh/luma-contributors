// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * Value of the low bits of a 64-bit DGGS cell key that carry no information at the resolution of
 * the cells, and so can hold a period instead.
 *
 * - `'ones'`: H3 (the digits below the cell resolution are 7, so `3 * (15 - resolution)` low bits
 *   are all ones) and Quadbin (the footer below the tile bits is all ones, `52 - 2 * zoom` bits).
 * - `'zeros'`: S2 (the cell id is the position bits, a lone sentinel one, then zeros, so
 *   `60 - 2 * level` low bits are zero).
 */
export type GPUCellPeriodKeyFill = 'ones' | 'zeros';

/** Explicit persisted key layout. Identity requires exactly one period and replaces no cell bits. */
export type GPUCellPeriodKeyLayout = 'legacy-one-bit' | 'zero-period-bit-identity';

/** Most period bits a packed key can hold: the period occupies the low 32-bit word only. */
export const GPU_CELL_PERIOD_KEY_MAXIMUM_PERIOD_BITS = 32;

/** Low bits for the explicit layout: legacy uses at least one; one-period identity uses zero. */
export function getGPUCellPeriodKeyPeriodBits(
  periodCount: number,
  layout: GPUCellPeriodKeyLayout
): number {
  validateCapacity('cell-period-key-packing', 'periodCount', periodCount, 1, 2 ** 32);
  validateRequirement(
    'cell-period-key-packing',
    layout === 'legacy-one-bit' || layout === 'zero-period-bit-identity',
    `layout must be 'legacy-one-bit' or 'zero-period-bit-identity'; received ${layout}`
  );
  if (layout === 'zero-period-bit-identity') {
    validateRequirement(
      'cell-period-key-packing',
      periodCount === 1,
      `identity layout requires periodCount = 1; received ${periodCount}`
    );
    return 0;
  }
  return Math.max(1, (periodCount - 1).toString(2).length);
}

/**
 * Free low bits of a DGGS cell key of a given resolution, the largest `periodBits` for which
 * `GPUCellPeriodKeys` is lossless: H3 `3 * (15 - resolution)`, Quadbin `52 - 2 * zoom`, S2
 * `60 - 2 * level`.
 */
export function getGPUCellKeyFreeBits(family: 'h3' | 'quadbin' | 's2', resolution: number): number {
  validateRequirement(
    'cell-period-key-packing',
    ['h3', 'quadbin', 's2'].includes(family),
    `family must be 'h3', 'quadbin' or 's2'; received ${family}`
  );
  if (family === 'h3') {
    return 3 * (15 - resolution);
  }
  if (family === 'quadbin') {
    return 52 - 2 * resolution;
  }
  return 60 - 2 * resolution;
}

/** The all-ones 64-bit key: "no cell" on input, "not packed" on output. */
const MISSING_CELL_KEY = 0xffffffffffffffffn;

/**
 * CPU packing of one (cell key, period) using the same explicit layout as `GPUCellPeriodKeys`.
 * Returns `undefined` for every row the kernel does not pack (it writes the reserved all-ones key
 * there): the all-ones "no cell" key, a period that is not an integer in `[0, periodCount)` (the
 * `0xffffffff` "excluded upstream" period included), a cell whose period bits are not `fill`,
 * or a nonmissing cell/period whose encoded key would equal the reserved all-ones sentinel.
 */
export function packGPUCellPeriodKey(
  cellKey: bigint,
  period: number,
  periodCount: number,
  fill: GPUCellPeriodKeyFill,
  layout: GPUCellPeriodKeyLayout
): bigint | undefined {
  const periodBits = BigInt(getGPUCellPeriodKeyPeriodBits(periodCount, layout));
  const mask = (1n << periodBits) - 1n;
  validateCellPeriodKey(cellKey, fill);
  const expected = fill === 'ones' ? mask : 0n;
  if (
    cellKey === MISSING_CELL_KEY ||
    !Number.isInteger(period) ||
    period < 0 ||
    period >= periodCount ||
    period === 0xffffffff ||
    (cellKey & mask) !== expected
  ) {
    return undefined;
  }
  const key = (cellKey & ~mask) | BigInt(period);
  return key === MISSING_CELL_KEY ? undefined : key;
}

/**
 * Restores cell fill bits and reads the period. The all-ones missing sentinel returns undefined;
 * other keys with a period outside [0, periodCount), or that reconstruct a missing cell, throw
 * instead of inventing a group.
 * Use the same explicit layout as the producer.
 */
export function unpackGPUCellPeriodKeyChecked(
  key: bigint,
  periodCount: number,
  fill: GPUCellPeriodKeyFill,
  layout: GPUCellPeriodKeyLayout
): {cellKey: bigint; period: number} | undefined {
  const periodBits = BigInt(getGPUCellPeriodKeyPeriodBits(periodCount, layout));
  validateCellPeriodKey(key, fill);
  if (key === MISSING_CELL_KEY) return undefined;
  const mask = (1n << periodBits) - 1n;
  const period = Number(key & mask);
  validateRequirement(
    'cell-period-key-packing',
    period < periodCount && period !== 0xffffffff,
    `packed period ${period} must be less than periodCount ${periodCount} and not the excluded period 4294967295`
  );
  const cellKey = (key & ~mask) | (fill === 'ones' ? mask : 0n);
  validateRequirement(
    'cell-period-key-packing',
    cellKey !== MISSING_CELL_KEY,
    `packed key ${key} reconstructs the reserved missing cell`
  );
  return {cellKey, period};
}

function validateCellPeriodKey(key: bigint, fill: GPUCellPeriodKeyFill): void {
  validateRequirement(
    'cell-period-key-packing',
    fill === 'ones' || fill === 'zeros',
    `fill must be 'ones' or 'zeros'; received ${fill}`
  );
  validateRequirement(
    'cell-period-key-packing',
    typeof key === 'bigint' && key >= 0n && key <= MISSING_CELL_KEY,
    `cell-period key must be an unsigned 64-bit bigint; received ${key}`
  );
}
