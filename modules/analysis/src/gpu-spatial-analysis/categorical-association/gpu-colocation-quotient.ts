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
  validateGPUStorageBindingByteLength,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createFixedPointClearNode,
  createFixedPointDecodeNode,
  FIXED_POINT_WGSL,
  FIXED_POINT_WORD_COUNT,
  getFixedPointAtomicAddWGSL,
  getFixedPointHeadroomBits
} from '../../analysis-core/kernels/fixed-point-accumulator';
import {GET_NAN_WGSL, IS_NAN_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT} from '../spatial-entropy/gpu-category-pair-counts';

const OPERATION = 'GPUColocationQuotient';

/**
 * Biased exponent of 1.0. Every per-row neighbour fraction is in `[0, 1]`, so the fixed-point
 * unit of the global sums is known without a scale reduction.
 */
const FRACTION_SCALE_EXPONENT_BITS = 127;

/** Properties for {@link GPUColocationQuotient}. */
export type GPUColocationQuotientProps = {
  /** Prefix for generated node IDs. Defaults to `'colocation-quotient'`. */
  id?: string;
  /**
   * Square weights. The neighbours of a row are its listed slots with a finite positive weight,
   * a neighbour other than the row itself, and a valid category. Weight values act as shares of
   * the row's neighbourhood (a Gaussian kernel gives the adaptive form of Wang et al. 2017;
   * binary weights on k nearest neighbours give Leslie and Kronenfeld's).
   */
  weights: GPUSpatialWeights;
  /** Category per row; `0xffffffff` and values at or above `categoryCount` leave the row out. */
  categories: GraphDataView<'uint32'>;
  /** Number of categories, at most {@link GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT}. */
  categoryCount: number;
  /** Caller-owned outputs. */
  output: GPUColocationQuotientOutput;
};

/** Caller-owned outputs of {@link GPUColocationQuotient}. */
export type GPUColocationQuotientOutput = {
  /**
   * Global quotients, `categoryCount * categoryCount` entries: entry `a * categoryCount + b` is
   * `CLQ(a -> b)`, how much a row of category `a` has neighbours of category `b` compared with
   * chance. 1 is independence, above 1 attraction, below 1 avoidance. NaN when no row of `a`
   * has a neighbour, or when the chance share of `b` seen from `a` is 0.
   */
  globalQuotients: GraphDataView<'float32'>;
  /**
   * Optional local quotients, `rows * categoryCount` entries: entry `i * categoryCount + b` is
   * the quotient of row `i` toward category `b`. NaN for rows without a category or neighbour.
   */
  localQuotients?: GraphDataView<'float32'>;
  /** Optional rows per category (`categoryCount` entries). */
  categoryRowCounts?: GraphDataView<'uint32'>;
};

/**
 * Colocation quotient of a categorical point (or area) set over a weights graph, after Leslie and
 * Kronenfeld (2011), "The colocation quotient: a new measure of spatial association between
 * categorical subsets of points", Geographical Analysis 43(3).
 *
 * For a row `i` of category `a` with neighbour shares `f(i, b)` (the share of its neighbourhood
 * weight on rows of category `b`), the local quotient is `f(i, b) / ratio(a, b)` and the global
 * quotient is `mean_{i in a} f(i, b) / ratio(a, b)`, where the chance share is
 * `ratio(a, b) = (N_b - [a = b]) / (N - 1)`, `N_b` the rows of category `b` and `N` all rows
 * with a category. The mean runs over rows of `a` that have at least one neighbour. With one
 * neighbour per row this is exactly `(N_{a->b} / N_a) / (N_b / (N - 1))`.
 *
 * One invocation per row walks its own links (a bounded neighbourhood) and accumulates its shares
 * into the shared deterministic fixed-point accumulator, so the global quotients are exact sums of
 * the float32 shares in any thread order.
 *
 * Not implemented: the permutation significance test of Leslie and Kronenfeld, and a column
 * limit beyond 32 categories.
 *
 * **Reference.** Leslie and Kronenfeld (2011) define the statistic. No pinned package
 * implements it (Esri's tool is closed); the expectation in the specs is an independent binary64
 * evaluation of the definition above.
 *
 * Compile-time: row count, `categoryCount`, which outputs exist. There are no per-frame numbers.
 */
export class GPUColocationQuotient implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUColocationQuotientProps;
  /** Rows of the weights graph. */
  readonly rowCount: number;

  constructor(props: GPUColocationQuotientProps) {
    const id = props.id ?? 'colocation-quotient';
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
    validateSinglePackedView(props.categories, ['uint32'], `${id} categories`);
    validateRequirement(
      id,
      props.categories.length === this.rowCount,
      'categories length must equal the weights row count'
    );
    validateSinglePackedView(
      props.output.globalQuotients,
      ['float32'],
      `${id} output.globalQuotients`
    );
    validateRequirement(
      id,
      props.output.globalQuotients.length >= props.categoryCount ** 2,
      'output.globalQuotients must hold categoryCount * categoryCount entries'
    );
    if (props.output.localQuotients) {
      validateSinglePackedView(
        props.output.localQuotients,
        ['float32'],
        `${id} output.localQuotients`
      );
      validateRequirement(
        id,
        props.output.localQuotients.length >= this.rowCount * props.categoryCount,
        'output.localQuotients must hold rows * categoryCount entries'
      );
    }
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
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.globalQuotients, props.output.localQuotients, props.output.categoryRowCounts],
      [props.categories, props.weights.offsets, props.weights.neighbors, props.weights.weights]
    );
  }

  /** Returns the clear, share accumulation, decode and quotient nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount} = this;
    const {weights, output, categoryCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.categories,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      output.globalQuotients,
      output.localQuotients,
      output.categoryRowCounts
    ]);
    if (output.localQuotients) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'output.localQuotients',
        rowCount * categoryCount * 4,
        'reduce the rows or the categoryCount'
      );
    }
    const slots = weights.neighbors.length;
    const cellCount = categoryCount ** 2;
    const rowCounts =
      output.categoryRowCounts ??
      createTransientView(graph, `${id}-category-row-counts`, 'uint32', categoryCount);
    const linkedCounts = createTransientView(graph, `${id}-linked-counts`, 'uint32', categoryCount);
    const sumWords = createTransientView(
      graph,
      `${id}-share-words`,
      'uint32',
      cellCount * FIXED_POINT_WORD_COUNT
    );
    const sums = createTransientView(graph, `${id}-share-sums`, 'float32', cellCount);
    const scale = createTransientView(graph, `${id}-share-scale`, 'uint32', 1);
    const localShares = output.localQuotients;
    const constants = {FIXED_POINT_HEADROOM_BITS: getFixedPointHeadroomBits(rowCount)};
    const nodes: GPUCommandNode<Parameters>[] = [
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-rows`,
        operation: OPERATION,
        view: rowCounts,
        type: 'u32',
        value: '0u',
        componentCount: categoryCount
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-linked`,
        operation: OPERATION,
        view: linkedCounts,
        type: 'u32',
        value: '0u'
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-scale`,
        operation: OPERATION,
        view: scale,
        type: 'u32',
        value: `${FRACTION_SCALE_EXPONENT_BITS}u`
      }),
      createFixedPointClearNode(graph, {
        id: `${id}-clear-shares`,
        operation: OPERATION,
        words: sumWords
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-shares`,
        operation: OPERATION,
        variant: localShares ? 'shares-local' : 'shares',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'linkWeights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'categories', view: props.categories, type: 'u32', access: 'read'},
          {name: 'rowCounts', view: rowCounts, type: 'atomic<u32>', access: 'read_write'},
          {name: 'linkedCounts', view: linkedCounts, type: 'atomic<u32>', access: 'read_write'},
          {name: 'shareWords', view: sumWords, type: 'atomic<u32>', access: 'read_write'},
          ...(localShares
            ? [{name: 'localValues', view: localShares, type: 'f32', access: 'read_write'} as const]
            : [])
        ],
        invocationCount: rowCount,
        declarations: `override FIXED_POINT_HEADROOM_BITS: u32;
const ROWS: u32 = ${rowCount}u;
const SLOTS: u32 = ${slots}u;
const CATEGORIES: u32 = ${categoryCount}u;
${GET_NAN_WGSL}
${FIXED_POINT_WGSL}
${getFixedPointAtomicAddWGSL('shareWords')}`,
        constants,
        body: `let own = categories[categoriesOffset + index];
  if (own >= CATEGORIES) {
    ${localShares ? 'for (var b = 0u; b < CATEGORIES; b++) { localValues[localValuesOffset + index * CATEGORIES + b] = getNaN(); }' : ''}
    return;
  }
  atomicAdd(&rowCounts[rowCountsOffset + own], 1u);
  var shares: array<f32, ${categoryCount}>;
  for (var b = 0u; b < CATEGORIES; b++) {
    shares[b] = 0.0;
  }
  var total = 0.0;
  let first = min(offsets[offsetsOffset + index], SLOTS);
  let last = min(offsets[offsetsOffset + index + 1u], SLOTS);
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let weight = linkWeights[linkWeightsOffset + slot];
    if (neighbor < ROWS && neighbor != index && weight > 0.0 && weight < 3.0e38) {
      let other = categories[categoriesOffset + neighbor];
      if (other < CATEGORIES) {
        shares[other] += weight;
        total += weight;
      }
    }
  }
  if (!(total > 0.0)) {
    ${localShares ? 'for (var b = 0u; b < CATEGORIES; b++) { localValues[localValuesOffset + index * CATEGORIES + b] = getNaN(); }' : ''}
    return;
  }
  atomicAdd(&linkedCounts[linkedCountsOffset + own], 1u);
  let unitExponent = getFixedPointUnitExponent(${FRACTION_SCALE_EXPONENT_BITS}u, FIXED_POINT_HEADROOM_BITS);
  for (var b = 0u; b < CATEGORIES; b++) {
    let share = shares[b] / total;
    addFixedPointToShareWords(own * CATEGORIES + b, quantizeFixedPoint(share, unitExponent));
    ${localShares ? 'localValues[localValuesOffset + index * CATEGORIES + b] = share;' : ''}
  }`
      }),
      createFixedPointDecodeNode(graph, {
        id: `${id}-decode-shares`,
        operation: OPERATION,
        words: sumWords,
        scale,
        maximumTermCount: rowCount,
        output: sums
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-global`,
        operation: OPERATION,
        variant: 'global',
        bindings: [
          {name: 'sums', view: sums, type: 'f32', access: 'read'},
          {name: 'rowCounts', view: rowCounts, type: 'u32', access: 'read'},
          {name: 'linkedCounts', view: linkedCounts, type: 'u32', access: 'read'},
          {name: 'globalQuotients', view: output.globalQuotients, type: 'f32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: `const CATEGORIES: u32 = ${categoryCount}u;
${GET_NAN_WGSL}`,
        body: `let own = index / CATEGORIES;
  let other = index % CATEGORIES;
  var total = 0u;
  for (var b = 0u; b < CATEGORIES; b++) {
    total += rowCounts[rowCountsOffset + b];
  }
  let otherRows = rowCounts[rowCountsOffset + other] - select(0u, 1u, own == other);
  let linked = linkedCounts[linkedCountsOffset + own];
  let valid = linked > 0u && total > 1u && rowCounts[rowCountsOffset + other] >= select(0u, 1u, own == other) && otherRows > 0u;
  let chance = f32(otherRows) / f32(total - 1u);
  globalQuotients[globalQuotientsOffset + index] = select(
    getNaN(),
    sums[sumsOffset + index] / f32(linked) / chance,
    valid
  );`
      })
    ];
    if (localShares) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-local`,
          operation: OPERATION,
          variant: 'local',
          bindings: [
            {name: 'categories', view: props.categories, type: 'u32', access: 'read'},
            {name: 'rowCounts', view: rowCounts, type: 'u32', access: 'read'},
            {name: 'localValues', view: localShares, type: 'f32', access: 'read_write'}
          ],
          invocationCount: rowCount * categoryCount,
          declarations: `const CATEGORIES: u32 = ${categoryCount}u;
${GET_NAN_WGSL}${IS_NAN_BITS_WGSL}`,
          body: `let row = index / CATEGORIES;
  let other = index % CATEGORIES;
  let own = categories[categoriesOffset + row];
  let share = localValues[localValuesOffset + index];
  if (own >= CATEGORIES || isNanBits(share)) {
    return;
  }
  var total = 0u;
  for (var b = 0u; b < CATEGORIES; b++) {
    total += rowCounts[rowCountsOffset + b];
  }
  let otherRows = rowCounts[rowCountsOffset + other] - select(0u, 1u, own == other);
  let valid = total > 1u && otherRows > 0u && otherRows < 0x80000000u;
  localValues[localValuesOffset + index] = select(getNaN(), share * f32(total - 1u) / f32(otherRows), valid);`
        })
      );
    }
    return nodes;
  }
}
