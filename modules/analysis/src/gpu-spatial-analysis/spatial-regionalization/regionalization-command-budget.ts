// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Fixed transitive allowance for adjacency transpose, standardization, published sort/scan and
 * publication nodes. Published primitives do not expose their node count; constructor tests
 * compare these upper bounds with the complete emitted graphs at multiple row counts.
 */
const REGIONALIZATION_FIXED_NODE_ALLOWANCE = 256;
/** Final partition evaluation includes one published sort and a fixed sequence of reductions. */
const PARTITION_EVALUATION_NODE_ALLOWANCE = 128;
/** Largest mutual-nearest input using the cooperative selection node instead of radix selection. @internal */
export const WARD_MUTUAL_COOPERATIVE_MAXIMUM_ROW_COUNT = 4096;

/** Upper bound including initial contiguity, adjacency transpose, sort and final objective. @internal */
export function estimateAZPCommandNodeCount(rowCount: number, maximumIterations: number): number {
  const logarithm = Math.ceil(Math.log2(Math.max(rowCount, 2)));
  return (
    REGIONALIZATION_FIXED_NODE_ALLOWANCE +
    (2 * logarithm + 8) * (logarithm + 4) +
    logarithm +
    6 * maximumIterations
  );
}

/** Upper bound including active-graph contraction, mutual radix selection and label jumps. @internal */
export function estimateWardCommandNodeCount(
  rowCount: number,
  maximumIterations: number,
  mutual: boolean
): number {
  const logarithm = Math.ceil(Math.log2(Math.max(rowCount, 2)));
  // Eight contraction nodes. Cooperative mutual selection/update needs ten more; the large
  // route needs 37, including 12 radix digits with two nodes each.
  const mutualNodeCount = rowCount <= WARD_MUTUAL_COOPERATIVE_MAXIMUM_ROW_COUNT ? 10 : 37;
  const roundCount = 8 + (mutual ? mutualNodeCount : 3 + Math.ceil(logarithm / 8));
  return REGIONALIZATION_FIXED_NODE_ALLOWANCE + logarithm + roundCount * maximumIterations;
}

/** Upper bound including symmetric adjacency, scan and nested refinement/evaluation. @internal */
export function estimateMaxPCommandNodeCount(
  rowCount: number,
  growthRounds: number,
  enclaveRounds: number,
  maximumIterations: number
): number {
  return (
    REGIONALIZATION_FIXED_NODE_ALLOWANCE +
    9 * growthRounds +
    3 * enclaveRounds +
    (maximumIterations > 0
      ? estimateAZPCommandNodeCount(rowCount, maximumIterations)
      : PARTITION_EVALUATION_NODE_ALLOWANCE)
  );
}
