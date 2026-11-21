// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Properties of {@link getPairwiseSumWGSL}. */
export type PairwiseSumWGSLProps = {
  /** WGSL `u32` expression of the number of terms. */
  count: string;
  /**
   * WGSL `f32` expression of one term. It may use the loop variable `${name}Item` (the term index,
   * `u32`) and anything declared by `setup`.
   */
  term: string;
  /** Name of the declared `f32` total. Default `sum`. */
  name?: string;
  /** Optional WGSL statements run for each term before `term` is evaluated. */
  setup?: string;
};

/**
 * WGSL statements that declare `var ${name}: f32`, the sum of `count` terms in a fixed pairwise
 * (cascade) order: term `i` is merged with the running blocks of equal size, as a binary counter
 * carries, and the at most 32 leftover blocks are folded from the smallest to the largest.
 * Row-tile merges above 4096 tiles apply this order separately to consecutive groups of at most
 * 4096 partials, then apply it again to the group totals in group order (see getRowTileMergeNodes).
 *
 * Why not a compensated sum: a float Kahan or Neumaier correction `(next - sum) - term` is zero
 * in real arithmetic, and the shader compiler of at least one supported GPU (Metal with fast
 * math) folds it to zero, so a "compensated" merge there is bit for bit the plain serial sum
 * (measured in `math-review-tile-merge-sums.spec.ts`: `[2^24, 1 x 999]` summed to `2^24`). The
 * pairwise order uses no such identity. Its error is at most about `log2(count)` roundings of
 * the partial totals, against `count` roundings for the serial sum; it is not exact, and a
 * single term far below the spacing of its neighbour can still be lost (one term of the 999
 * above). The order depends only on `count`, so the result is reproducible for a given adapter.
 *
 * The statements also declare `${name}Levels`, `${name}Depth`, `${name}Item`, `${name}Block` and
 * `${name}Carry`.
 *
 * @internal
 */
export function getPairwiseSumWGSL(props: PairwiseSumWGSLProps): string {
  const name = props.name ?? 'sum';
  return `var ${name}Levels: array<f32, 32>;
  var ${name}Depth = 0u;
  for (var ${name}Item = 0u; ${name}Item < ${props.count}; ${name}Item++) {
    ${props.setup ?? ''}
    var ${name}Block = ${props.term};
    // One merge per trailing one bit of the number of terms already taken.
    var ${name}Carry = ${name}Item;
    for (var level = 0u; level < 32u; level++) {
      if ((${name}Carry & 1u) == 0u) {
        break;
      }
      ${name}Depth--;
      ${name}Block = ${name}Levels[${name}Depth] + ${name}Block;
      ${name}Carry = ${name}Carry >> 1u;
    }
    ${name}Levels[${name}Depth] = ${name}Block;
    ${name}Depth++;
  }
  var ${name} = 0.0;
  for (var level = 0u; level < 32u; level++) {
    if (${name}Depth == 0u) {
      break;
    }
    ${name}Depth--;
    ${name} = ${name}Levels[${name}Depth] + ${name};
  }`;
}
