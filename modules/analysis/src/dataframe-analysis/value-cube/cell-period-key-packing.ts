// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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

/** Most period bits a packed key can hold: the period occupies the low 32-bit word only. */
export const GPU_CELL_PERIOD_KEY_MAXIMUM_PERIOD_BITS = 32;

/**
 * Number of low key bits that hold the period for `periodCount` periods: the bit length of
 * `periodCount - 1`, at least 1. Periods `0 .. periodCount - 1` fit exactly.
 */
export function getGPUCellPeriodKeyPeriodBits(periodCount: number): number {
  if (!Number.isInteger(periodCount) || periodCount < 1 || periodCount > 2 ** 32) {
    throw new Error('periodCount must be an integer in [1, 2^32]');
  }
  return Math.max(1, (periodCount - 1).toString(2).length);
}

/**
 * Free low bits of a DGGS cell key of a given resolution, the largest `periodBits` for which
 * `GPUCellPeriodKeys` is lossless: H3 `3 * (15 - resolution)`, Quadbin `52 - 2 * zoom`, S2
 * `60 - 2 * level`.
 */
export function getGPUCellKeyFreeBits(family: 'h3' | 'quadbin' | 's2', resolution: number): number {
  if (family === 'h3') {
    return 3 * (15 - resolution);
  }
  if (family === 'quadbin') {
    return 52 - 2 * resolution;
  }
  return 60 - 2 * resolution;
}

/** CPU packing of one (cell key, period): the exact function `GPUCellPeriodKeys` applies. */
export function packGPUCellPeriodKey(
  cellKey: bigint,
  period: number,
  periodCount: number,
  fill: GPUCellPeriodKeyFill
): bigint | undefined {
  const periodBits = BigInt(getGPUCellPeriodKeyPeriodBits(periodCount));
  const mask = (1n << periodBits) - 1n;
  const expected = fill === 'ones' ? mask : 0n;
  if (period >= periodCount || (cellKey & mask) !== expected) {
    return undefined;
  }
  return (cellKey & ~mask) | BigInt(period);
}

/** Inverse of {@link packGPUCellPeriodKey}: restores the fill bits of the cell and extracts the period. */
export function unpackGPUCellPeriodKey(
  key: bigint,
  periodCount: number,
  fill: GPUCellPeriodKeyFill
): {cellKey: bigint; period: number} {
  const periodBits = BigInt(getGPUCellPeriodKeyPeriodBits(periodCount));
  const mask = (1n << periodBits) - 1n;
  return {cellKey: (key & ~mask) | (fill === 'ones' ? mask : 0n), period: Number(key & mask)};
}
