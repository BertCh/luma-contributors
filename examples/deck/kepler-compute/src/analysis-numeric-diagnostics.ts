// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Numeric storage-format narrowing diagnostics and cause-specific rate invalid-value diagnostics. */

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
  maximumAbsoluteError: number;
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
  let maximumAbsoluteError = 0;
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
    const narrowed = Math.fround(value);
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
      maximumAbsoluteError = Math.max(maximumAbsoluteError, Math.abs(narrowed - value));
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
    maximumAbsoluteError: 0,
    verdict: 'exact'
  };
  for (const report of reports) {
    total.rowCount += report.rowCount;
    total.nullCount += report.nullCount;
    total.inexactCount += report.inexactCount;
    total.overflowCount += report.overflowCount;
    total.nonFiniteCount += report.nonFiniteCount;
    total.maximumAbsoluteError = Math.max(total.maximumAbsoluteError, report.maximumAbsoluteError);
  }
  total.verdict = total.overflowCount ? 'refused' : total.inexactCount ? 'lossy' : 'exact';
  return total;
}

/** Explicit policy: Float64/Int64 columns narrow only when exact, or when the user accepts rounding. */
export function decideNarrowing(
  report: NumericNarrowingReport,
  acceptLossy: boolean
): {allowed: boolean; message?: string} {
  if (report.verdict === 'exact') return {allowed: true};
  if (report.verdict === 'refused')
    return {
      allowed: false,
      message: `${report.overflowCount} ${report.format} values overflow Float32`
    };
  return acceptLossy
    ? {
        allowed: true,
        message: `${report.inexactCount} values round (max error ${report.maximumAbsoluteError})`
      }
    : {
        allowed: false,
        message: `${report.inexactCount} ${report.format} values are not exact in Float32; accept rounding to continue`
      };
}

export const RATE_INVALID_CAUSES = [
  'valid',
  'masked',
  'nullEvent',
  'nullPopulation',
  'nonFiniteEvent',
  'nonFinitePopulation',
  'negativeEvent',
  'negativePopulation',
  'zeroPopulation'
] as const;
export type RateInvalidCause = (typeof RATE_INVALID_CAUSES)[number];

export type RateInvalidDiagnostics = {
  causes: Uint8Array;
  counts: Record<RateInvalidCause, number>;
};

/**
 * Attribute each excluded row to exactly one cause, in documented priority: masked, null event,
 * null population, non-finite event, non-finite population, negative event, negative population,
 * zero population. counts sum to the row count.
 */
export function classifyRateInvalidCauses(
  events: ArrayLike<number | null>,
  populations: ArrayLike<number | null>,
  mask?: ArrayLike<number | boolean>
): RateInvalidDiagnostics {
  const causes = new Uint8Array(events.length);
  const counts = Object.fromEntries(RATE_INVALID_CAUSES.map(cause => [cause, 0])) as Record<
    RateInvalidCause,
    number
  >;
  for (let row = 0; row < events.length; row++) {
    const event = events[row];
    const population = populations[row];
    let cause: RateInvalidCause = 'valid';
    if (mask && !mask[row]) cause = 'masked';
    else if (event === null || event === undefined) cause = 'nullEvent';
    else if (population === null || population === undefined) cause = 'nullPopulation';
    else if (!Number.isFinite(event)) cause = 'nonFiniteEvent';
    else if (!Number.isFinite(population)) cause = 'nonFinitePopulation';
    else if (event < 0) cause = 'negativeEvent';
    else if (population < 0) cause = 'negativePopulation';
    else if (population === 0) cause = 'zeroPopulation';
    causes[row] = RATE_INVALID_CAUSES.indexOf(cause);
    counts[cause]++;
  }
  return {causes, counts};
}
