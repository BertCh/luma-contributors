// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
  return roundFloat32Units(
    values.reduce((total, value) => total + getFloat32Units(value), 0n),
    1
  );
}

/** Rounds an exact numerator in 2^-149 units divided by count, directly to binary32 ties-to-even. */
export function roundFloat32Units(numerator: bigint, count: number): number {
  if (numerator === 0n) return 0;
  const negative = numerator < 0n;
  const magnitude = negative ? -numerator : numerator;
  const denominator = BigInt(count);
  let leading = magnitude.toString(2).length - denominator.toString(2).length;
  if (
    leading >= 0
      ? magnitude < denominator << BigInt(leading)
      : magnitude << BigInt(-leading) < denominator
  )
    leading--;
  // One unit is the subnormal spacing. Normal values retain 24 significant bits.
  const shift = Math.max(0, leading - 23);
  const divisor = denominator << BigInt(shift);
  let significand = magnitude / divisor;
  const remainder = magnitude % divisor;
  if (2n * remainder > divisor || (2n * remainder === divisor && (significand & 1n) !== 0n))
    significand++;
  // The at-most-24-bit significand and power of two are exact binary64 operands. The only
  // remaining binary32 rounding is overflow to infinity; no wide-total double rounding occurs.
  const result = Math.fround(Number(significand) * 2 ** (shift - 149));
  return negative ? -result : result;
}

export function getExactFloat32Mean(values: readonly number[]): number {
  return values.length
    ? roundFloat32Units(
        values.reduce((total, value) => total + getFloat32Units(value), 0n),
        values.length
      )
    : NaN;
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
  const means = binValues.map(getExactFloat32Mean);
  return {counts, sums, means, invalidCount};
}
