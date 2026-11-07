// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {DataType, type Vector} from 'apache-arrow';

/** True for numeric Arrow types that only reach Float32 through the explicit Float64 precision policy. */
export function isWideNumericType(type: DataType): boolean {
  return (
    (DataType.isInt(type) && type.bitWidth === 64) ||
    (DataType.isDecimal(type) && type.bitWidth >= 64)
  );
}

/**
 * Converts one Int64, Uint64 or Decimal value to the nearest JavaScript number. The caller applies
 * the same absolute/relative error budget it uses for Float64, so lossy values stay rejected
 * unless the policy allows them.
 */
export function convertWideNumericValue(type: DataType, value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value === 'bigint') {
    const rounded = Number(value);
    // Float64 cannot hold this integer; do not hide the loss from the Float32 error budget.
    if (BigInt(rounded) !== value) throw new Error('Float64 column exceeds its precision policy');
    return rounded;
  }
  if (typeof value === 'number') return value;
  // Decimal values are little-endian two's-complement words scaled by 10^-scale.
  const words = value as ArrayLike<number>;
  let integer = BigInt(0);
  for (let index = words.length - 1; index >= 0; index--)
    integer = (integer << BigInt(32)) | BigInt(words[index] >>> 0);
  const bitWidth = words.length * 32;
  if (integer >= BigInt(1) << BigInt(bitWidth - 1)) integer -= BigInt(1) << BigInt(bitWidth);
  const scale = DataType.isDecimal(type) ? type.scale : 0;
  return scale === 0 ? Number(integer) : Number(integer) / 10 ** scale;
}

/** Lazily maps every row of a wide numeric chunk through the shared Float64 policy converter. */
export function convertWideNumericVector(
  type: DataType,
  vector: Vector,
  convertNumber: (value: number | null) => number | null
): (number | null)[] {
  return Array.from(vector, value => convertNumber(convertWideNumericValue(type, value)));
}
