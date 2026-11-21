// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUInferenceAlternative,
  GPUMultipleTestingMethod
} from '../../analysis-core/contracts/index';
import {
  resolveGPUMultipleTestingMethod,
  getFamilyHarmonicNumber
} from './gpu-multiple-testing-adjustment';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {MAXIMUM_PERMUTATION_COUNT} from '../permutation-inference/permutation-parameters';
import {MAXIMUM_GLOBAL_PERMUTATION_PAIR_BLOCK_COUNT} from '../permutation-inference/permutation-limits';
import {resolvePermutationAlternative} from '../permutation-inference/permutation-alternative';
import {isHarmonicRatioAtMost} from './harmonic-ratio-comparison';

/** Settings of {@link getPermutationRejectionFloor}. */
export type PermutationRejectionFloorProps = {
  /** Permutations `P` of the test (the per-frame `permutationCount`). */
  permutationCount: number;
  /** Tail convention of the test. Defaults to `'two-sided'`, the default of the permutation tests. */
  alternative?: GPUInferenceAlternative;
  /** Counting formula; omission retains the existing alternative-based convention. */
  pValueConstruction?: 'upper-tail-count' | 'doubled-minimum-tail';
  /** Contributor draw ceiling (at most 2^20), optionally constrained by the global row-block route. */
  contributorBudget?: {
    maximumPermutations: number;
    /** Sum of ceil(chunkRowCount / GLOBAL_PERMUTATION_PAIR_BLOCK_ROWS), including partial chunks. */
    globalRowBlockCount?: number;
  };
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
  /** Fewest requested draws that permit some rejection in this family; Infinity for an empty family. */
  requiredPermutations: number;
  /** Whether requiredPermutations fits the supported (and optionally narrower contributor) budget. */
  attainableWithinBudget: boolean;
  /** Effective draw ceiling after applying contributor and global row-block limits. */
  maximumSupportedPermutations: number;
  /** True when some row can be rejected at `alpha` after the correction, even if every row sits at the floor. */
  canReject: boolean;
  /**
   * Fewest rows at the floor sufficient for rejection when all other p-values are one: 1 for `'none'`,
   * `'bonferroni'` and `'holm'`, `ceil(pValueFloor rowCount c / alpha)` for the Benjamini
   * procedures (`c` is 1 for Hochberg, the harmonic sum of `rowCount` for Yekutieli). Infinity when
   * `canReject` is false. At alpha=1 this is zero: even all-one families reject without floor rows.
   */
  minimumRowsAtFloor: number;
  /** Bonferroni and Holm: largest family that can reject; Infinity at alpha=1 because adjustments cap at one. */
  maximumRowCount?: number;
};

/**
 * Whether a permutation test can reject at all. A permutation p-value is at least
 * `m / (P + 1)`, so at the default 999 permutations the two-sided floor is 0.002. Then
 * Bonferroni and Holm cannot reject in a family of more than 25 rows at `alpha = 0.05`, and
 * Benjamini-Hochberg with 10,000 rows can reject with 400 at the floor and all others at one.
 * This sufficient scenario is not a necessary condition on a general family. Pure CPU arithmetic for
 * choosing `permutationCount` or interpreting an empty result; nothing is warned at run time.
 * Sequential stopping (`stopAfterExceedances`) does not change the floor.
 */
export function getPermutationRejectionFloor(
  props: PermutationRejectionFloorProps
): PermutationRejectionFloor {
  const {permutationCount, rowCount, alpha} = props;
  const id = 'getPermutationRejectionFloor';
  validateCapacity(id, 'permutationCount', permutationCount, 1, 2 ** 31 - 1);
  validateCapacity(id, 'rowCount', rowCount, 0);
  validateRequirement(
    id,
    Number.isFinite(alpha) && alpha > 0 && alpha <= 1,
    'alpha must be finite and in (0, 1] in host precision'
  );
  const method = resolveGPUMultipleTestingMethod(id, props.method);
  const alternative = resolvePermutationAlternative(id, props.alternative);
  validateRequirement(
    id,
    props.pValueConstruction === undefined ||
      props.pValueConstruction === 'upper-tail-count' ||
      props.pValueConstruction === 'doubled-minimum-tail',
    `pValueConstruction must be 'upper-tail-count' or 'doubled-minimum-tail'; received ${String(props.pValueConstruction)}`
  );
  const multiplier =
    props.pValueConstruction === 'upper-tail-count'
      ? 1
      : props.pValueConstruction === 'doubled-minimum-tail'
        ? 2
        : alternative === 'two-sided'
          ? 2
          : 1;
  let maximumSupportedPermutations =
    props.contributorBudget?.maximumPermutations ?? MAXIMUM_PERMUTATION_COUNT;
  validateCapacity(
    id,
    'contributorBudget.maximumPermutations',
    maximumSupportedPermutations,
    1,
    MAXIMUM_PERMUTATION_COUNT
  );
  const rowBlocks = props.contributorBudget?.globalRowBlockCount;
  if (rowBlocks !== undefined) {
    validateCapacity(id, 'contributorBudget.globalRowBlockCount', rowBlocks, 1);
    maximumSupportedPermutations = Math.min(
      maximumSupportedPermutations,
      Math.max(0, Math.floor(MAXIMUM_GLOBAL_PERMUTATION_PAIR_BLOCK_COUNT / rowBlocks) - 1)
    );
  }
  const correction =
    method === 'bonferroni' || method === 'holm'
      ? rowCount
      : method === 'benjamini-yekutieli'
        ? getFamilyHarmonicNumber(rowCount)
        : 1;
  let requiredPermutations =
    rowCount === 0
      ? Infinity
      : alpha === 1
        ? 1
        : Math.max(1, Math.ceil((multiplier * correction) / alpha) - 1);
  const canRejectAt = (draws: number): boolean => {
    if (rowCount === 0 || draws < 1) return false;
    if (alpha === 1) return true;
    if (method === 'benjamini-yekutieli')
      return isHarmonicRatioAtMost(
        rowCount,
        BigInt(multiplier),
        BigInt(draws + 1),
        alpha,
        correction
      );
    const floor = Math.min(multiplier / (draws + 1), 1);
    const adjustedFloor =
      method === 'bonferroni' || method === 'holm'
        ? floor * rowCount
        : method === 'benjamini-hochberg'
          ? (floor * rowCount * correction) / rowCount
          : floor;
    return Math.min(adjustedFloor, 1) <= alpha;
  };
  // Correct the inverse estimate and its neighbor using the actual forward host predicate.
  // E.g. .05 * 3 is .15000000000000002, so alpha=.15 requires forty, not 39 draws.
  if (Number.isFinite(requiredPermutations)) {
    if (requiredPermutations > 1 && canRejectAt(requiredPermutations - 1)) requiredPermutations--;
    if (!canRejectAt(requiredPermutations)) requiredPermutations++;
  }
  const pValueFloor = Math.min(multiplier / (permutationCount + 1), 1);
  let minimumRowsAtFloor = 1;
  let maximumRowCount: number | undefined;
  let canReject = rowCount >= 1 && pValueFloor <= alpha;
  if (method === 'bonferroni' || method === 'holm') {
    // The smallest adjusted p-value is min(rowCount * pValueFloor, 1).
    maximumRowCount = alpha === 1 ? Infinity : Math.floor(alpha / pValueFloor);
    if (Number.isFinite(maximumRowCount)) {
      if ((maximumRowCount + 1) * pValueFloor <= alpha) maximumRowCount++;
      if (maximumRowCount * pValueFloor > alpha) maximumRowCount--;
    }
    canReject = rowCount >= 1 && rowCount <= maximumRowCount;
  } else if (method === 'benjamini-hochberg' || method === 'benjamini-yekutieli') {
    const harmonicSum = method === 'benjamini-yekutieli' ? getFamilyHarmonicNumber(rowCount) : 1;
    // Rank k rejects when pValueFloor * rowCount * harmonicSum / k <= alpha, with k at most rowCount.
    const scaledFloor = pValueFloor * rowCount * harmonicSum;
    const canRejectAtRank = (rank: number) =>
      Number.isFinite(rank) &&
      (method === 'benjamini-yekutieli'
        ? isHarmonicRatioAtMost(
            rowCount,
            BigInt(multiplier) * BigInt(rowCount),
            BigInt(permutationCount + 1) * BigInt(rank),
            alpha,
            harmonicSum
          )
        : scaledFloor / rank <= alpha);
    minimumRowsAtFloor = alpha === 1 ? 1 : Math.max(1, Math.ceil(scaledFloor / alpha));
    // Correct the candidate with the forward predicate. BY resolves harmonic
    // boundaries before rounding; alpha itself is never moved by a comparison slack.
    if (alpha < 1) {
      if (minimumRowsAtFloor > 1 && canRejectAtRank(minimumRowsAtFloor - 1)) minimumRowsAtFloor--;
      if (!canRejectAtRank(minimumRowsAtFloor)) minimumRowsAtFloor++;
    }
    canReject = rowCount >= 1 && minimumRowsAtFloor <= rowCount;
  }
  return {
    pValueFloor,
    requiredPermutations,
    attainableWithinBudget: canRejectAt(maximumSupportedPermutations),
    maximumSupportedPermutations,
    canReject,
    minimumRowsAtFloor: canReject ? (alpha === 1 ? 0 : minimumRowsAtFloor) : Infinity,
    ...(maximumRowCount === undefined ? {} : {maximumRowCount})
  };
}
