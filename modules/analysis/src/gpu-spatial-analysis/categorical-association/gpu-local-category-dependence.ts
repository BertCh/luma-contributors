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
  validatePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT} from '../spatial-entropy/gpu-category-pair-counts';

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
   * Square weights; every listed slot to another row with a valid category is one neighbour.
   * Weight values are ignored (binary composition, the default of spdep's `licd_multi`).
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
 * **Reference.** Boots (2003), "Developing local measures of spatial association for categorical
 * data", Journal of Geographical Systems 5, defines the LICD family. Its software form is
 * `licd_multi` of the R package spdep (GPL, not installed here, so no R fixture); its local
 * composition output (`count_like_i`, `count_nbs_i`, `prop_i`, `pbinom_like_BW`) is what this
 * class reproduces: the binomial test uses the global category share `N_c / N` as the success
 * probability. The specs compare with an independent binary64 evaluation of the formulas in
 * {@link GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS}.
 *
 * Not implemented, and not claimed: the local configuration part (local multi-category join
 * counts and their chi-square), the conditional permutation p-values, the unlike-neighbour
 * tails, and the `standardizedScore` is this class's own column (the z score of the same
 * binomial), not an spdep column.
 *
 * The binomial tail is a bounded sum over the row's own neighbourhood (at most its degree),
 * evaluated in log space in float32 (relative error about 1e-5).
 *
 * Compile-time: row count, `categoryCount`, which outputs exist.
 */
export class GPULocalCategoryDependence implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULocalCategoryDependenceProps;
  /** Rows of the weights graph. */
  readonly rowCount: number;

  constructor(props: GPULocalCategoryDependenceProps) {
    const id = props.id ?? 'local-category-dependence';
    this.id = id;
    this.props = props;
    this.rowCount = validateGPUSpatialWeights(id, props.weights);
    validateCapacity(
      id,
      'categoryCount',
      props.categoryCount,
      1,
      GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT
    );
    validatePackedView(props.categories, ['uint32'], `${id} categories`);
    validateRequirement(
      id,
      props.categories.length === this.rowCount,
      'categories length must equal the weights row count'
    );
    validatePackedView(props.output.statistics, ['float32'], `${id} output.statistics`);
    validateRequirement(
      id,
      props.output.statistics.length >=
        this.rowCount * GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS.length,
      'output.statistics must hold rows * GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS.length entries'
    );
    if (props.output.categoryRowCounts) {
      validatePackedView(
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
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.statistics, props.output.categoryRowCounts],
      [props.categories, props.weights.offsets, props.weights.neighbors]
    );
  }

  /** Returns the category count nodes and the per-row statistics node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount} = this;
    const {weights, output, categoryCount} = props;
    const columns = GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS;
    validateGraphViewsBelongToGraph(id, graph, [
      props.categories,
      weights.offsets,
      weights.neighbors,
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
          {name: 'categories', view: props.categories, type: 'u32', access: 'read'},
          {name: 'rowCounts', view: rowCounts, type: 'u32', access: 'read'},
          {name: 'statistics', view: output.statistics, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: `const ROWS: u32 = ${rowCount}u;
const SLOTS: u32 = ${weights.neighbors.length}u;
const CATEGORIES: u32 = ${categoryCount}u;
const COLUMNS: u32 = ${columns.length}u;
${GET_NAN_WGSL}
// P(X >= like), X ~ Binomial(count, p), by the log-space pmf recurrence over k = 0..count.
fn getUpperTail(like: u32, count: u32, p: f32) -> f32 {
  if (like == 0u || p >= 1.0) {
    return 1.0;
  }
  let logP = log(p);
  let logQ = log(1.0 - p);
  var logChoose = 0.0;
  var tail = 0.0;
  for (var k = 0u; k <= count; k++) {
    if (k >= like) {
      tail += exp(logChoose + f32(k) * logP + f32(count - k) * logQ);
    }
    logChoose += log(f32(count - k) / f32(k + 1u));
  }
  return min(tail, 1.0);
}`,
        body: `let own = categories[categoriesOffset + index];
  let base = statisticsOffset + index * COLUMNS;
  let nan = getNaN();
  var likeCount = 0u;
  var neighborCount = 0u;
  if (own < CATEGORIES) {
    let first = min(offsets[offsetsOffset + index], SLOTS);
    let last = min(offsets[offsetsOffset + index + 1u], SLOTS);
    for (var slot = first; slot < last; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor < ROWS && neighbor != index) {
        let other = categories[categoriesOffset + neighbor];
        if (other < CATEGORIES) {
          neighborCount++;
          likeCount += select(0u, 1u, other == own);
        }
      }
    }
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
      })
    ];
  }
}
