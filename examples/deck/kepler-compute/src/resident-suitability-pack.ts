// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  ResidentSuitabilityColumnFormat,
  ResidentSuitabilityMissingValuePolicy
} from './resident-suitability-plan';

/** Smallest integer magnitude a Float32 can no longer distinguish from its neighbour. */
export const RESIDENT_SUITABILITY_INEXACT_INTEGER = 16_777_216;

/**
 * The only values that return from the GPU before the result is published: per weighted column
 * the extent and valid-row count of its defined values, and one flag raised by any row that is
 * missing in some but not all weighted columns. No row-sized data is involved.
 */
export type ResidentSuitabilityGPUStatistics = {
  names: readonly string[];
  formats: readonly ResidentSuitabilityColumnFormat[];
  /** `[minimum, maximum]` per column over its defined values. */
  extents: Float32Array;
  /** Defined (valid and finite) rows per column. */
  counts: Uint32Array;
  /** Rows missing in a strict subset of the weighted columns. */
  mixedMissingRows: number;
  /** Default `refuse`: such rows stop the run. `exclude` and `renormalize` score them by policy. */
  missingValues?: ResidentSuitabilityMissingValuePolicy;
  /** Raw data standardization leaves a constant column as its own values, which is exact. */
  isConstantAllowed?: boolean;
};

/**
 * Refuses inputs the GPU scorer cannot reproduce exactly:
 * - integers a Float32 cannot hold exactly;
 * - rows missing in only some weighted fields, unless the user chose a missing-value policy
 *   (exclude nulls those rows like native; renormalize scores them over their defined fields);
 * - constant columns, which native standardization leaves unchanged while the kernels yield 0.
 * An empty defined set scores every row null, as native does.
 */
export function assertResidentSuitabilityStatistics(
  statistics: ResidentSuitabilityGPUStatistics
): void {
  const {names, formats, extents, counts} = statistics;
  // Lengths come from the producer's own plan, so a mismatch is an internal invariant failure.
  if (extents.length !== names.length * 2 || counts.length !== names.length || !names.length)
    throw new Error('Suitability statistics do not match the weighted fields');
  for (const [index, name] of names.entries()) {
    if (
      formats[index] !== 'float32' &&
      counts[index] > 0 &&
      (Math.abs(extents[index * 2]) >= RESIDENT_SUITABILITY_INEXACT_INTEGER ||
        Math.abs(extents[index * 2 + 1]) >= RESIDENT_SUITABILITY_INEXACT_INTEGER)
    )
      throw new Error(
        `Suitability field "${name}" has integers at or beyond +/-16777216 that Float32 cannot hold exactly`
      );
  }
  if (statistics.mixedMissingRows && (statistics.missingValues ?? 'refuse') === 'refuse')
    throw new Error(
      'Suitability weighted fields are missing in different rows; choose a missing-value policy (exclude those rows or renormalize the weights) or use rows missing in every weighted field'
    );
  if (!statistics.isConstantAllowed)
    for (const [index, name] of names.entries())
      if (counts[index] > 0 && extents[index * 2] === extents[index * 2 + 1])
        throw new Error(
          `Suitability field "${name}" is constant over the scored rows; native scoring leaves it unstandardized, which the resident scorer cannot reproduce. Deselect it or use SQL`
        );
}
