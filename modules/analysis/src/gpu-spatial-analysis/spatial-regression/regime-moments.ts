// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createSortedSegmentOffsetsNode} from '../../analysis-core/kernels/sorted-segment-offsets';
import {getSortKeyBits} from '../../analysis-core/kernels/sorted-segment-sums';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';

/** Threads of one regime workgroup: strided row partials, then a fixed binary tree. */
const REGIME_WORKGROUP_SIZE = 256;

/** Float32 words per regime in the moments view of {@link getRegimeMomentNodes}. */
export function getRegimeMomentStride(predictorCount: number): number {
  const dimension = predictorCount + 1;
  return dimension + (dimension * (dimension + 1)) / 2 + dimension;
}

/** Properties of {@link getRegimeMomentNodes}. */
export type RegimeMomentProps = {
  /** Node and transient ID prefix. */
  id: string;
  /** Operation name reported in workload estimates. */
  operation: string;
  /** Row-major predictors without an intercept column. */
  predictors: GraphDataView<'float32'>;
  predictorCount: number;
  response: GraphDataView<'float32'>;
  /** Regime per row; rows at or above `regimeCount` are excluded. */
  regimes: GraphDataView<'uint32'>;
  regimeCount: number;
  /** Optional row mask; zero excludes the row. */
  mask?: GraphDataView<'uint32'>;
};

/** Transient views written by {@link getRegimeMomentNodes}. */
export type RegimeMomentViews = {
  /** `regimeCount + 1` offsets into the regime-sorted rows: regime `g` has `offsets[g+1]-offsets[g]` rows. */
  segmentOffsets: GraphDataView<'uint32'>;
  /**
   * Per regime, `getRegimeMomentStride(predictorCount)` float32 words: the means of the
   * predictors and of the response (`D = predictorCount + 1`), then the packed upper-triangular
   * centered cross-products of `[predictors, response]` (row `first`, column `second >= first`,
   * slot `first * (2D - first + 1) / 2 + second - first`), then the `D` sums of the deviations
   * `value - mean` of each column. Empty regimes hold zeros.
   *
   * The means are float32, so they are the true means only to half a float32 step. The deviation
   * sums are what is left: `mean + deviationSum / count` is the mean to the precision of the
   * deviations. A consumer that needs levels finer than a float32 step of a column mean (a
   * response whose mean is large against its residuals) treats the stored mean as an exact shift
   * and carries the deviation sums through its moments.
   */
  moments: GraphDataView<'float32'>;
};

/**
 * Per-regime means and centered cross-product moments of `[predictors, response]` over the
 * complete rows, in parallel and in a fixed order:
 * 1. One invocation per row writes the sort key (the regime of a complete row, otherwise
 *    `regimeCount`, which sorts last) and its row index.
 * 2. Published stable `GPUSort` groups the rows by regime, keeping row order inside a regime.
 * 3. One binary search per regime gives the segment offsets (no atomics).
 * 4. One 256-thread workgroup per regime: each thread sums differences from the segment's first
 *    complete row, a fixed binary tree combines the threads and adds the origin back (means),
 *    then the same for the centered
 *    cross-products about those means and for the sums of the deviations from them. The order
 *    depends only on the segment bounds, so the result is bitwise reproducible.
 *
 * Work is `O(n log n)` for the sort plus `O(n D^2)`; the depth is the sort plus one dispatch.
 */
export function getRegimeMomentNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: RegimeMomentProps
): {nodes: GPUCommandNode<Parameters>[]; views: RegimeMomentViews} {
  const {id, operation, predictorCount, regimeCount} = props;
  const rowCount = props.response.length;
  const dimension = predictorCount + 1;
  const momentCount = (dimension * (dimension + 1)) / 2;
  const stride = getRegimeMomentStride(predictorCount);
  const sortKeys = createTransientView(graph, `${id}-regime-keys`, 'uint32', rowCount);
  const sortRows = createTransientView(graph, `${id}-regime-rows`, 'uint32', rowCount);
  const sortedKeys = createTransientView(graph, `${id}-sorted-regime-keys`, 'uint32', rowCount);
  const sortedRows = createTransientView(graph, `${id}-sorted-regime-rows`, 'uint32', rowCount);
  const segmentOffsets = createTransientView(
    graph,
    `${id}-regime-offsets`,
    'uint32',
    regimeCount + 1
  );
  const moments = createTransientView(
    graph,
    `${id}-regime-moments`,
    'float32',
    regimeCount * stride
  );
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-regime-keys`,
      operation,
      variant: 'regime-keys',
      invocationCount: rowCount,
      bindings: [
        {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
        {name: 'response', view: props.response, type: 'f32', access: 'read'},
        {name: 'regimes', view: props.regimes, type: 'u32', access: 'read'},
        ...(props.mask
          ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
        {name: 'sortRows', view: sortRows, type: 'u32', access: 'read_write'}
      ],
      declarations: IS_FINITE_BITS_WGSL,
      body: `var included = regimes[regimesOffset + index] < ${regimeCount}u &&
    isFiniteBits(response[responseOffset + index]);
  ${props.mask ? 'included = included && mask[maskOffset + index] != 0u;' : ''}
  for (var column = 0u; column < ${predictorCount}u; column++) {
    included = included && isFiniteBits(predictors[predictorsOffset + index * ${predictorCount}u + column]);
  }
  sortKeys[sortKeysOffset + index] = select(${regimeCount}u, regimes[regimesOffset + index], included);
  sortRows[sortRowsOffset + index] = index;`
    }),
    ...new GPUSort({
      id: `${id}-regime-sort`,
      keys: sortKeys,
      values: sortRows,
      outputKeys: sortedKeys,
      outputValues: sortedRows,
      keyBits: getSortKeyBits(regimeCount)
    }).getCommandNodes(graph),
    createSortedSegmentOffsetsNode<Parameters>(graph, {
      id: `${id}-regime-offsets`,
      operation,
      variant: 'regime-offsets',
      segmentCount: regimeCount,
      sortedKeys,
      segmentOffsets
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-regime-moments`,
      operation,
      variant: 'regime-moments',
      workgroupSize: REGIME_WORKGROUP_SIZE,
      invocationCount: regimeCount * REGIME_WORKGROUP_SIZE,
      guardIndex: false,
      bindings: [
        {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
        {name: 'response', view: props.response, type: 'f32', access: 'read'},
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'segmentOffsets', view: segmentOffsets, type: 'u32', access: 'read'},
        {name: 'moments', view: moments, type: 'f32', access: 'read_write'}
      ],
      declarations: `const K: u32 = ${predictorCount}u;
const D: u32 = ${dimension}u;
const MOMENT_COUNT: u32 = ${momentCount}u;
const STRIDE: u32 = ${stride}u;
const REGIME_COUNT: u32 = ${regimeCount}u;
${ROUNDED_DIFFERENCE_WGSL}
var<workgroup> partialSums: array<f32, ${REGIME_WORKGROUP_SIZE}>;
var<workgroup> regimeMeans: array<f32, ${dimension}>;

fn getValue(row: u32, column: u32) -> f32 {
  if (column == K) {
    return response[responseOffset + row];
  }
  return predictors[predictorsOffset + row * K + column];
}

// Fixed binary tree over the workgroup (strides 128 to 1); every thread must call it.
fn reduceWorkgroup(localIndex: u32, value: f32) -> f32 {
  partialSums[localIndex] = value;
  workgroupBarrier();
  for (var stride = ${REGIME_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localIndex < stride) {
      partialSums[localIndex] = partialSums[localIndex] + partialSums[localIndex + stride];
    }
    workgroupBarrier();
  }
  let total = partialSums[0];
  workgroupBarrier();
  return total;
}`,
      // No early return: every thread of the workgroup reaches the barriers.
      body: `let regime = index / ${REGIME_WORKGROUP_SIZE}u;
  let isInRange = regime < REGIME_COUNT;
  var begin = 0u;
  var end = 0u;
  if (isInRange) {
    begin = segmentOffsets[segmentOffsetsOffset + regime];
    end = segmentOffsets[segmentOffsetsOffset + regime + 1u];
  }
  let count = f32(end - begin);
  let runtimeZero = segmentOffsets[segmentOffsetsOffset];
  var origins: array<f32, ${dimension}>;
  if (end > begin) {
    let originRow = sortedRows[sortedRowsOffset + begin];
    for (var column = 0u; column < D; column++) { origins[column] = getValue(originRow, column); }
  }
  var sums: array<f32, ${dimension}>;
  for (var slot = begin + localInvocationIndex; slot < end; slot += ${REGIME_WORKGROUP_SIZE}u) {
    let row = sortedRows[sortedRowsOffset + slot];
    for (var column = 0u; column < D; column++) {
      sums[column] += getRoundedDifference(getValue(row, column), origins[column], runtimeZero);
    }
  }
  for (var column = 0u; column < D; column++) {
    let total = reduceWorkgroup(localInvocationIndex, sums[column]);
    if (localInvocationIndex == 0u) {
      regimeMeans[column] = select(0.0, origins[column] + total / max(count, 1.0), end > begin);
    }
  }
  workgroupBarrier();
  var means: array<f32, ${dimension}>;
  for (var column = 0u; column < D; column++) {
    means[column] = regimeMeans[column];
  }
  var products: array<f32, ${momentCount}>;
  var centered: array<f32, ${dimension}>;
  var deviationSums: array<f32, ${dimension}>;
  // The first segment starts at sorted row 0, so this buffer word is a zero the compiler cannot
  // know. It fixes each deviation as written: summed as (sum of values) - count * mean, the
  // deviation sums and the centered products would lose everything below a float32 step.
  for (var slot = begin + localInvocationIndex; slot < end; slot += ${REGIME_WORKGROUP_SIZE}u) {
    let row = sortedRows[sortedRowsOffset + slot];
    for (var column = 0u; column < D; column++) {
      centered[column] = getRoundedDifference(getValue(row, column), means[column], runtimeZero);
      deviationSums[column] = deviationSums[column] + centered[column];
    }
    var moment = 0u;
    for (var first = 0u; first < D; first++) {
      for (var second = first; second < D; second++) {
        products[moment] = products[moment] + centered[first] * centered[second];
        moment = moment + 1u;
      }
    }
  }
  let base = momentsOffset + min(regime, REGIME_COUNT - 1u) * STRIDE;
  if (isInRange && localInvocationIndex == 0u) {
    for (var column = 0u; column < D; column++) {
      moments[base + column] = means[column];
    }
  }
  for (var moment = 0u; moment < MOMENT_COUNT; moment++) {
    let total = reduceWorkgroup(localInvocationIndex, products[moment]);
    if (isInRange && localInvocationIndex == 0u) {
      moments[base + D + moment] = total;
    }
  }
  for (var column = 0u; column < D; column++) {
    let total = reduceWorkgroup(localInvocationIndex, deviationSums[column]);
    if (isInRange && localInvocationIndex == 0u) {
      moments[base + D + MOMENT_COUNT + column] = total;
    }
  }`
    })
  ];
  return {nodes, views: {segmentOffsets, moments}};
}
