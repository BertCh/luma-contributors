// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
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
import {
  type GPUSpatialWeights,
  getGPUSpatialWeightsNeighborLoopWGSL,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';

const OPERATION = 'GPUCategoryPairCounts';

/**
 * Largest `categoryCount`. The pair matrix holds `categoryCount ^ 2` counters, each updated with
 * an integer atomic, and the summary kernels of the consumers loop over the matrix in one
 * invocation, so it is bounded: 32 categories give 1024 counters.
 */
export const GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT = 32;

/** Category value of a row without a category; the row and its links are left out. */
export const GPU_CATEGORY_PAIR_COUNTS_NONE = 0xffffffff;

/** Properties for {@link GPUCategoryPairCounts}. */
export type GPUCategoryPairCountsProps = {
  /** Prefix for generated node IDs. Defaults to `'category-pair-counts'`. */
  id?: string;
  /**
   * Square weights. Every active slot is one ordered link `i -> j`: a slot with positive weight to
   * another row, the neighbor rule shared by the spatial statistics. Self slots (for example from
   * `GPUSpatialWeightsAlgebra` `'selfWeight'`) and zero-weight slots are not links; the size of a
   * positive weight is not used.
   */
  weights: GPUSpatialWeights;
  /**
   * Category per row. `0xffffffff` and values at or above `categoryCount` leave the row out of the
   * row counts and out of every link it ends.
   */
  categories: GraphDataView<'uint32'>;
  /** Number of categories, at most {@link GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT}. */
  categoryCount: number;
  /** Caller-owned outputs. */
  output: GPUCategoryPairCountsOutput;
};

/** Caller-owned outputs of {@link GPUCategoryPairCounts}. */
export type GPUCategoryPairCountsOutput = {
  /**
   * Ordered link counts, `categoryCount * categoryCount` entries: entry `a * categoryCount + b`
   * counts the links from a row of category `a` to a row of category `b`.
   */
  pairCounts: GraphDataView<'uint32'>;
  /** Rows per category (`categoryCount` entries). */
  categoryRowCounts: GraphDataView<'uint32'>;
};

/**
 * Counts, over the links of a weights graph, how often a row of category `a` is linked to a row of
 * category `b`, and the rows per category. It is the join-count table of a categorical map and
 * the common input of the spatial entropy and the colocation quotient.
 *
 * Links are the active slots of the shared neighbor rule (positive weight, another row), so
 * weights with self slots give the same table as the same weights without them.
 *
 * One invocation per row walks its own links (a bounded neighbourhood) and adds to the counters
 * with integer atomics, so the counts are exact and independent of thread order (below 2^32).
 */
export class GPUCategoryPairCounts implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCategoryPairCountsProps;

  constructor(props: GPUCategoryPairCountsProps) {
    const id = props.id ?? 'category-pair-counts';
    this.id = id;
    this.props = props;
    const rows = validateGPUSpatialWeights(id, props.weights);
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
      props.categories.length === rows,
      'categories length must equal the weights row count'
    );
    validateSinglePackedView(props.output.pairCounts, ['uint32'], `${id} output.pairCounts`);
    validateRequirement(
      id,
      props.output.pairCounts.length >= props.categoryCount ** 2,
      'output.pairCounts must hold categoryCount * categoryCount entries'
    );
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
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.pairCounts, props.output.categoryRowCounts],
      [props.categories, props.weights.offsets, props.weights.neighbors, props.weights.weights]
    );
  }

  /** Returns the clear and count nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.categories,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      output.pairCounts,
      output.categoryRowCounts
    ]);
    const rows = weights.offsets.length - 1;
    const slots = weights.neighbors.length;
    return [
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-pairs`,
        operation: OPERATION,
        view: output.pairCounts,
        type: 'u32',
        value: '0u',
        componentCount: props.categoryCount ** 2
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-rows`,
        operation: OPERATION,
        view: output.categoryRowCounts,
        type: 'u32',
        value: '0u',
        componentCount: props.categoryCount
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        variant: 'count',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'categories', view: props.categories, type: 'u32', access: 'read'},
          {name: 'pairCounts', view: output.pairCounts, type: 'atomic<u32>', access: 'read_write'},
          {
            name: 'categoryRowCounts',
            view: output.categoryRowCounts,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        invocationCount: rows,
        declarations: `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${slots}u;
const CATEGORIES: u32 = ${props.categoryCount}u;`,
        body: `let own = categories[categoriesOffset + index];
  if (own >= CATEGORIES) {
    return;
  }
  atomicAdd(&categoryRowCounts[categoryRowCountsOffset + own], 1u);
  ${getGPUSpatialWeightsNeighborLoopWGSL(
    'categories[categoriesOffset + neighbor] < CATEGORIES',
    `atomicAdd(
        &pairCounts[pairCountsOffset + own * CATEGORIES + categories[categoriesOffset + neighbor]],
        1u
      );`
  )}`
      })
    ];
  }
}
