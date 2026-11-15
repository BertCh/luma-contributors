// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type WideNumericFormat = 'float32' | 'float64' | 'int32' | 'uint32' | 'int64' | 'uint64';

export type NumericNarrowingReport = {
  format: WideNumericFormat;
  rowCount: number;
  nullCount: number;
  /** Values whose Float32 image differs from the source value. */
  inexactCount: number;
  /** Finite values whose magnitude overflows Float32 (become infinity). */
  overflowCount: number;
  nonFiniteCount: number;
  /**
   * bigint for Int64/Uint64 (exact subtraction before conversion); number for other formats.
   * CPU numbers are exact JS differences. GPU Float64 numbers are lower bounds within 2^-31
   * relative error. Overflow and nonfinite values are counted separately, not in this maximum.
   */
  maximumAbsoluteError: number | bigint;
  /** exact: lossless; lossy: representable with rounding; refused: overflow present. */
  verdict: 'exact' | 'lossy' | 'refused';
};

/** Scan a column and report what a Float32 narrowing would do. 64-bit integers arrive as bigint. */
export function diagnoseFloat32Narrowing(
  format: WideNumericFormat,
  values: ArrayLike<number | bigint | null>
): NumericNarrowingReport {
  let nullCount = 0;
  let inexactCount = 0;
  let overflowCount = 0;
  let nonFiniteCount = 0;
  const integer64 = format === 'int64' || format === 'uint64';
  let maximumAbsoluteError: number | bigint = integer64 ? 0n : 0;
  for (let index = 0; index < values.length; index++) {
    const raw = values[index];
    if (raw === null || raw === undefined) {
      nullCount++;
      continue;
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      nonFiniteCount++;
      continue;
    }
    const narrowed = typeof raw === 'bigint' ? roundIntegerToFloat32(raw) : Math.fround(value);
    if (!Number.isFinite(narrowed)) {
      overflowCount++;
      continue;
    }
    // A bigint above 2^53 is already inexact as a double; compare through BigInt when possible.
    const exact =
      typeof raw === 'bigint'
        ? Number.isInteger(narrowed) && BigInt(narrowed) === raw
        : narrowed === value;
    if (!exact) {
      inexactCount++;
      const error =
        integer64 || typeof raw === 'bigint'
          ? absoluteBigInt(BigInt(narrowed) - BigInt(raw))
          : Math.abs(narrowed - value);
      if (error > maximumAbsoluteError) maximumAbsoluteError = error;
    }
  }
  return {
    format,
    rowCount: values.length,
    nullCount,
    inexactCount,
    overflowCount,
    nonFiniteCount,
    maximumAbsoluteError,
    verdict: overflowCount ? 'refused' : inexactCount ? 'lossy' : 'exact'
  };
}

/** Combine reports of consecutive chunks of one column into the report of the whole column. */
export function mergeNumericNarrowingReports(
  reports: readonly NumericNarrowingReport[],
  format: WideNumericFormat
): NumericNarrowingReport {
  const total: NumericNarrowingReport = {
    format,
    rowCount: 0,
    nullCount: 0,
    inexactCount: 0,
    overflowCount: 0,
    nonFiniteCount: 0,
    maximumAbsoluteError: format === 'int64' || format === 'uint64' ? 0n : 0,
    verdict: 'exact'
  };
  for (const report of reports) {
    total.rowCount += report.rowCount;
    total.nullCount += report.nullCount;
    total.inexactCount += report.inexactCount;
    total.overflowCount += report.overflowCount;
    total.nonFiniteCount += report.nonFiniteCount;
    if (report.maximumAbsoluteError > total.maximumAbsoluteError)
      total.maximumAbsoluteError = report.maximumAbsoluteError;
  }
  total.verdict = total.overflowCount ? 'refused' : total.inexactCount ? 'lossy' : 'exact';
  return total;
}

function absoluteBigInt(value: bigint): bigint {
  return value < 0n ? -value : value;
}

/** Round with integer arithmetic to avoid binary64 double rounding at Float32 halfway values. */
function roundIntegerToFloat32(value: bigint): number {
  const magnitude = absoluteBigInt(value);
  const shift = Math.max(0, magnitude.toString(2).length - 24);
  if (shift === 0) return Number(value);
  const shiftBits = BigInt(shift);
  let mantissa = magnitude >> shiftBits;
  const remainder = magnitude - (mantissa << shiftBits);
  const halfway = 1n << (shiftBits - 1n);
  if (remainder > halfway || (remainder === halfway && (mantissa & 1n) !== 0n)) mantissa++;
  const rounded = Number(mantissa) * 2 ** shift;
  return value < 0n ? -rounded : rounded;
}
