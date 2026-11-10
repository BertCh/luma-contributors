// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUInferenceAlternative,
  GPUMultipleTestingMethod
} from '../../analysis-core/contracts/index';
import {resolveGPUMultipleTestingMethod} from './gpu-multiple-testing-adjustment';

/** Settings of {@link getPermutationRejectionFloor}. */
export type PermutationRejectionFloorProps = {
  /** Permutations `P` of the test (the per-frame `permutationCount`). */
  permutationCount: number;
  /** Tail convention of the test. Defaults to `'two-sided'`, the default of the permutation tests. */
  alternative?: GPUInferenceAlternative;
  /** Rows in the tested family (rows with a p-value). */
  rowCount: number;
  /** Significance level. */
  alpha: number;
  /** Correction applied to the p-values. Defaults to `'none'`. */
  method?: GPUMultipleTestingMethod;
};

/** Result of {@link getPermutationRejectionFloor}. */
export type PermutationRejectionFloor = {
  /** Smallest attainable pseudo p-value, `m / (P + 1)` (`m` is 2 for `'two-sided'`, else 1). */
  pValueFloor: number;
  /** True when some row can be rejected at `alpha` after the correction, even if every row sits at the floor. */
  canReject: boolean;
  /**
   * Fewest rows that must sit at the floor before one can be rejected: 1 for `'none'`,
   * `'bonferroni'` and `'holm'`, `ceil(pValueFloor rowCount c / alpha)` for the Benjamini
   * procedures (`c` is 1 for Hochberg, the harmonic sum of `rowCount` for Yekutieli). Infinity when
   * `canReject` is false.
   */
  minimumRowsAtFloor: number;
  /** `'bonferroni'` and `'holm'` only: the largest family size that can still reject, `floor(alpha / pValueFloor)`. */
  maximumRowCount?: number;
};

// Relative slack so exact boundaries such as 25 * 0.002 = 0.05 are not lost to float64 rounding.
const TOLERANCE = 1e-12;

/**
 * Whether a permutation test can reject at all. A permutation p-value is at least
 * `m / (P + 1)`, so at the default 999 permutations the two-sided floor is 0.002. Then
 * Bonferroni and Holm cannot reject in a family of more than 25 rows at `alpha = 0.05`, and
 * Benjamini-Hochberg with 10,000 rows needs 400 of them at the floor. Pure CPU arithmetic for
 * choosing `permutationCount` or interpreting an empty result; nothing is warned at run time.
 * Sequential stopping (`stopAfterExceedances`) does not change the floor.
 */
export function getPermutationRejectionFloor(
  props: PermutationRejectionFloorProps
): PermutationRejectionFloor {
  const {permutationCount, rowCount, alpha} = props;
  const method = resolveGPUMultipleTestingMethod('getPermutationRejectionFloor', props.method);
  const multiplier = (props.alternative ?? 'two-sided') === 'two-sided' ? 2 : 1;
  const pValueFloor = Math.min(multiplier / (permutationCount + 1), 1);
  const limit = alpha * (1 + TOLERANCE);
  let minimumRowsAtFloor = 1;
  let maximumRowCount: number | undefined;
  let canReject = rowCount >= 1 && pValueFloor <= limit;
  if (method === 'bonferroni' || method === 'holm') {
    // The smallest adjusted p-value is rowCount * pValueFloor, so rejection needs it at most alpha.
    maximumRowCount = Math.floor(limit / pValueFloor);
    canReject = rowCount >= 1 && rowCount <= maximumRowCount;
  } else if (method === 'benjamini-hochberg' || method === 'benjamini-yekutieli') {
    let harmonicSum = 1;
    if (method === 'benjamini-yekutieli') {
      harmonicSum = 0;
      for (let rank = rowCount; rank >= 1; rank--) {
        harmonicSum += 1 / rank;
      }
    }
    // Rank k rejects when pValueFloor * rowCount * harmonicSum / k <= alpha, with k at most rowCount.
    minimumRowsAtFloor = Math.max(
      1,
      Math.ceil((pValueFloor * rowCount * harmonicSum) / limit - TOLERANCE)
    );
    canReject = rowCount >= 1 && minimumRowsAtFloor <= rowCount;
  }
  return {
    pValueFloor,
    canReject,
    minimumRowsAtFloor: canReject ? minimumRowsAtFloor : Infinity,
    ...(maximumRowCount === undefined ? {} : {maximumRowCount})
  };
}
