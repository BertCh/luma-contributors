// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** `2^149`: every finite float32 is an integer multiple of `2^-149`. */
const SCALE = 1n << 149n;

/** Exact integer (in units of `2^-149`) of a finite float32 value. */
export function getFloat32Units(value: number): bigint {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, value);
  const bits = view.getUint32(0);
  const exponentBits = (bits >>> 23) & 0xff;
  const significand = BigInt(bits & 0x7fffff) | (exponentBits === 0 ? 0n : 0x800000n);
  const shift = BigInt(Math.max(exponentBits, 1) - 1);
  const magnitude = significand << shift;
  return bits >>> 31 ? -magnitude : magnitude;
}

/**
 * Exactly rounded float32 of the exact sum of float32 values, by arbitrary precision integers: the
 * result no float accumulation order can promise. Independent of the fixed-point accumulator.
 */
export function getExactFloat32Sum(values: readonly number[]): number {
  let units = 0n;
  for (const value of values) {
    units += getFloat32Units(value);
  }
  // Number(bigint) rounds to nearest even in binary64; the scaling by 2^-149 is exact.
  return Math.fround(Number(units) / Number(SCALE));
}

export type ValueCubeEvent = {
  /** Cell index, may be out of range. */
  cell: number;
  /** Period, a sentinel or out of range for a row that is not binned. */
  period: number;
  value: number;
  selected: boolean;
};

export type ValueCubeOracle = {
  counts: number[];
  sums: number[];
  means: number[];
  invalidCount: number;
};

/** Brute-force dense cube of events: exact sums and counts per (cell, period), cell-major. */
export function computeValueCubeOracle(
  events: readonly ValueCubeEvent[],
  cellCount: number,
  periodCount: number
): ValueCubeOracle {
  const binValues: number[][] = Array.from({length: cellCount * periodCount}, () => []);
  let invalidCount = 0;
  for (const event of events) {
    if (!event.selected || event.period === 0xffffffff) {
      continue;
    }
    if (event.cell >= cellCount || event.period >= periodCount || !Number.isFinite(event.value)) {
      invalidCount++;
      continue;
    }
    binValues[event.cell * periodCount + event.period].push(event.value);
  }
  const counts = binValues.map(values => values.length);
  const sums = binValues.map(values => getExactFloat32Sum(values));
  const means = binValues.map((values, bin) =>
    values.length > 0 ? Math.fround(sums[bin] / values.length) : NaN
  );
  return {counts, sums, means, invalidCount};
}
