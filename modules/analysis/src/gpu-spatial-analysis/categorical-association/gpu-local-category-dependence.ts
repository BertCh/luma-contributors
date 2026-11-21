// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  type GPUSpatialWeights,
  getGPUSpatialWeightsNeighborLoopWGSL,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT} from '../spatial-entropy/gpu-category-pair-counts';

import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  snapshotCategoricalAssociationProps,
  createCategoricalSuppressionNodes,
  validateCategoricalCompleteness
} from './categorical-association-contract';

const OPERATION = 'GPULocalCategoryDependence';

/** Column of each statistic in `output.statistics` (row `i`, column `c` is at `i * length + c`). */
export const GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS = {
  /** Neighbours of the row with the row's own category. */
  likeCount: 0,
  /** Neighbours with a valid category (the row itself excluded). */
  neighborCount: 1,
  /** `likeCount / neighborCount`. */
  proportion: 2,
  /** Share of the row's category among all rows with a category, `N_c / N`. */
  expectedProportion: 3,
  /**
   * Binomial upper tail `P(X >= likeCount)` for `X ~ Binomial(neighborCount, expectedProportion)`:
   * small where the row's neighbourhood holds more of its own category than chance (local
   * clustering of the category).
   */
  upperTailProbability: 4,
  /** `(likeCount - n p) / sqrt(n p (1 - p))`; NaN when `p` is 1. */
  standardizedScore: 5,
  /** Number of columns. */
  length: 6
} as const;

/** Properties for {@link GPULocalCategoryDependence}. */
export type GPULocalCategoryDependenceProps = {
  /** Prefix for generated node IDs. Defaults to `'local-category-dependence'`. */
  id?: string;
  /**
   * Square weights; every active slot (positive weight, another row: the neighbor rule shared by
   * the spatial statistics) to a row with a valid category is one neighbour. Self slots and
   * zero-weight slots are not neighbours; the size of a positive weight is not used (binary
   * composition).
   */
  weights: GPUSpatialWeights;
  /** Category per row; `0xffffffff` and values at or above `categoryCount` leave the row out. */
  categories: GraphDataView<'uint32'>;
  /** Number of categories, at most {@link GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT}. */
  categoryCount: number;
  /** Caller-owned outputs. */
  output: GPULocalCategoryDependenceOutput;
};

/** Caller-owned outputs of {@link GPULocalCategoryDependence}. */
export type GPULocalCategoryDependenceOutput = {
  /**
   * Statistics per row, `rows * GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS.length` entries. A row without
   * a category, or without any neighbour, has NaN in every column except the counts, which are 0.
   */
  statistics: GraphDataView<'float32'>;
  /** Optional rows per category (`categoryCount` entries). */
  categoryRowCounts?: GraphDataView<'uint32'>;
};

/**
 * Local indicators for categorical data (LICD), local composition part: for each row, how many of
 * its neighbours share its category, and how unusual that is against the category's global share.
 *
 * **Definition.** For row `i` with category `c` and `k` neighbours that have a category (the row
 * itself is never counted, neither as a neighbour nor as a like):
 * - `likeCount` is the number of those neighbours of category `c`, `neighborCount` is `k`;
 * - `proportion = likeCount / k` is the local share of the row's category;
 * - `expectedProportion = N_c / N` is the global share of the category over all rows with one;
 * - `upperTailProbability = P(X >= likeCount)` for `X ~ Binomial(k, N_c / N)`, the upper tail;
 * - `standardizedScore` is the z score of the same binomial.
 *
 * **Reference.** Boots (2003), "Developing local measures of spatial association for categorical
 * data", Journal of Geographical Systems 5, defines the LICD family, and this class computes its
 * local composition idea with the definition above. It does not reproduce the columns of
 * `licd_multi` of the R package spdep (not installed here, so not checked): spdep counts the focal
 * row among its own neighbours (`k + 1` trials, the focal row always a like), reports the global
 * share as `prop_i`, and its `pbinom_like_BW` is the lower tail `P(X <= like)`. For 4 neighbours of
 * which 3 are like and a share of 0.5 this class gives `P(X >= 3) = 0.3125` on `Binomial(4, 0.5)`,
 * where spdep's `pbinom_like_BW` is `P(X <= 4) = 0.96875` on `Binomial(5, 0.5)`. The specs
 * compare with hand-computed values and with a binary64 evaluation of the formulas in
 * {@link GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS}.
 *
 * Not implemented: the local configuration part (local multi-category join counts and their
 * chi-square), the conditional permutation p-values and the unlike-neighbour tails.
 *
 * The binomial tail sums relative probabilities outward from the mode, normalizing by their
 * total in float32. Both recurrences are bounded by the row's degree. The 2,190-neighbour
 * regression checks an absolute tolerance of 2e-7 against SciPy; it is not a general error bound.
 *
 * Compile-time: row count, `categoryCount`, which outputs exist.
 * Accepts 1–32 categories.
 * Producer overflow/candidateOverflow flags remain live: either nonzero flag suppresses all
 * six statistics per row to NaN. Category row counts and trailing storage are preserved.
 * A requiredCount view is accepted only with the producer's overflow flag.
 * Verify CSR completeness before supplying it and omit upstream convergence, iteration-limit
 * and unresolved-work views.
 * When supplied, status.complete is also live: zero suppresses the same dependent outputs,
 * even when both overflow flags are clear; a nonzero certificate never overrides either loss.
 */
export class GPULocalCategoryDependence implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULocalCategoryDependenceProps;
  /** Rows of the weights graph. */
  readonly rowCount: number;

  constructor(props: GPULocalCategoryDependenceProps) {
    props = snapshotCategoricalAssociationProps(props, ['statistics', 'categoryRowCounts']);
    const id = props.id ?? 'local-category-dependence';
    this.id = id;
    this.props = props;
    this.rowCount = validateGPUSpatialWeights(id, props.weights);
    validateCategoricalCompleteness(id, props.weights);
    validateCapacity(
      id,
      'categoryCount',
      props.categoryCount,
      1,
      GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT
    );
    validateSinglePackedView(props.categories, ['uint32'], `${id} categories`);
    validateRequirement(
      id,
      props.categories.length === this.rowCount,
      'categories length must equal the weights row count'
    );
    validateSinglePackedView(props.output.statistics, ['float32'], `${id} output.statistics`);
    validateRequirement(
      id,
      props.output.statistics.length >=
        this.rowCount * GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS.length,
      'output.statistics must hold rows * GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS.length entries'
    );
    if (props.output.categoryRowCounts) {
      validateSinglePackedView(
        props.output.categoryRowCounts,
        ['uint32'],
        `${id} output.categoryRowCounts`
      );
      validateRequirement(
        id,
        props.output.categoryRowCounts.length >= props.categoryCount,
        'output.categoryRowCounts must hold categoryCount entries'
      );
    }
    validateDisjointWritableViews(id, props.output);
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.statistics, props.output.categoryRowCounts],
      [
        props.categories,
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        ...Object.values(props.weights.status ?? {})
      ]
    );
  }

  /** Returns the category count nodes and the per-row statistics node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount} = this;
    validateSpatialStatisticsBindings(graph, id, props, {
      'category counters': props.categoryCount * 4
    });
    const {weights, output, categoryCount} = props;
    const columns = GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS;
    validateGraphViewsBelongToGraph(id, graph, [
      props.categories,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      output.statistics,
      output.categoryRowCounts
    ]);
    const rowCounts =
      output.categoryRowCounts ??
      createTransientView(graph, `${id}-category-row-counts`, 'uint32', categoryCount);
    return [
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-rows`,
        operation: OPERATION,
        view: rowCounts,
        type: 'u32',
        value: '0u',
        componentCount: categoryCount
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count-rows`,
        operation: OPERATION,
        variant: 'count-rows',
        bindings: [
          {name: 'categories', view: props.categories, type: 'u32', access: 'read'},
          {name: 'rowCounts', view: rowCounts, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: `const CATEGORIES: u32 = ${categoryCount}u;`,
        body: `let own = categories[categoriesOffset + index];
  if (own < CATEGORIES) {
    atomicAdd(&rowCounts[rowCountsOffset + own], 1u);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-statistics`,
        operation: OPERATION,
        variant: 'statistics',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'categories', view: props.categories, type: 'u32', access: 'read'},
          {name: 'rowCounts', view: rowCounts, type: 'u32', access: 'read'},
          {name: 'statistics', view: output.statistics, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: `const ROWS: u32 = ${rowCount}u;
const CAPACITY: u32 = ${weights.neighbors.length}u;
const CATEGORIES: u32 = ${categoryCount}u;
const COLUMNS: u32 = ${columns.length}u;
${GET_NAN_WGSL}
// P(X >= like), X ~ Binomial(count, p). Normalize relative probabilities from the
// mode, avoiding cancellation of O(count) log terms in a large neighborhood.
fn getUpperTail(like: u32, count: u32, p: f32) -> f32 {
  if (like == 0u || p >= 1.0) {
    return 1.0;
  }
  if (p <= 0.0 || like > count) { return 0.0; }
  let mode = min(count, u32(floor((f32(count) + 1.0) * p)));
  let odds = p / (1.0 - p);
  let inverseOdds = (1.0 - p) / p;
  var normalizer = 1.0;
  var tail = select(0.0, 1.0, mode >= like);
  var probability = 1.0;
  for (var k = mode; k < count; k++) {
    probability *= (f32(count - k) / f32(k + 1u)) * odds;
    normalizer += probability;
    if (k + 1u >= like) { tail += probability; }
  }
  probability = 1.0;
  for (var k = mode; k > 0u; k--) {
    probability *= (f32(k) / f32(count - k + 1u)) * inverseOdds;
    normalizer += probability;
    if (k - 1u >= like) { tail += probability; }
  }
  return clamp(tail / normalizer, 0.0, 1.0);
}`,
        body: `let own = categories[categoriesOffset + index];
  let base = statisticsOffset + index * COLUMNS;
  let nan = getNaN();
  var likeCount = 0u;
  var neighborCount = 0u;
  if (own < CATEGORIES) {
    ${getGPUSpatialWeightsNeighborLoopWGSL(
      'categories[categoriesOffset + neighbor] < CATEGORIES',
      `neighborCount++;
      likeCount += select(0u, 1u, categories[categoriesOffset + neighbor] == own);`
    )}
  }
  statistics[base + ${columns.likeCount}u] = f32(likeCount);
  statistics[base + ${columns.neighborCount}u] = f32(neighborCount);
  var total = 0u;
  for (var b = 0u; b < CATEGORIES; b++) {
    total += rowCounts[rowCountsOffset + b];
  }
  if (own >= CATEGORIES || neighborCount == 0u || total == 0u) {
    statistics[base + ${columns.proportion}u] = nan;
    statistics[base + ${columns.expectedProportion}u] = nan;
    statistics[base + ${columns.upperTailProbability}u] = nan;
    statistics[base + ${columns.standardizedScore}u] = nan;
    return;
  }
  let p = f32(rowCounts[rowCountsOffset + own]) / f32(total);
  let n = f32(neighborCount);
  let variance = n * p * (1.0 - p);
  statistics[base + ${columns.proportion}u] = f32(likeCount) / n;
  statistics[base + ${columns.expectedProportion}u] = p;
  statistics[base + ${columns.upperTailProbability}u] = getUpperTail(likeCount, neighborCount, p);
  statistics[base + ${columns.standardizedScore}u] = select(nan, (f32(likeCount) - n * p) / sqrt(variance), variance > 0.0);`
      }),
      // Incomplete weights (status.overflow or status.candidateOverflow): NaN statistics.
      ...createCategoricalSuppressionNodes(
        graph,
        id,
        OPERATION,
        weights,
        output.statistics,
        rowCount * columns.length
      )
    ];
  }
}
