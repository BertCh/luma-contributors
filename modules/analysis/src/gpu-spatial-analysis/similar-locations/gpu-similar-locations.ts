// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {type WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createStatisticsKernelNode as createWGSLKernelNode} from '../spatial-weights/statistics-kernel-node';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateRequirement,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSimilarLocationsParameterLength,
  GPU_SIMILAR_LOCATIONS_CONTROL_LENGTH
} from './similar-locations-parameters';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUCardinalityPort} from '../../analysis-core/contracts/status';

import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {ROUNDED_DIFFERENCE_WGSL} from '../spatial-regression/rounded-difference-wgsl';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {MULTIPLICATION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/multiplication-rounding-error-wgsl';

const OPERATION = 'GPUSimilarLocations';
/** Lanes of the column-reduction workgroups (one workgroup per attribute). */
const REDUCTION_LANES = 256;

/**
 * Largest `columnCount`: a reduction kernel runs one 256-lane workgroup per column, and the
 * dispatch holds at most `2^32 - 1` invocations, so `floor((2^32 - 1) / 256)`.
 */
export const GPU_SIMILAR_LOCATIONS_MAXIMUM_COLUMN_COUNT = Math.floor(0xffffffff / REDUCTION_LANES);

/**
 * Largest `values` length (`rowCount * columnCount`) of {@link GPUSimilarLocations} on `device`: the
 * attribute matrix (and its materialized copy on the rank route) holds 4 bytes per entry in one storage binding, so
 * 33554432 on a default device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPUSimilarLocationsMaximumValueCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}

/**
 * Row count from which `'rank'` standardization sorts each column (O(n log n)) instead of the
 * all-pairs count (O(n^2) per column). Below it the single all-pairs dispatch per column is
 * cheaper than a radix sort per column.
 */
const RANK_SORT_MINIMUM_ROWS = 4096;

/** WGSL tree reductions of one workgroup. Every lane of the workgroup must call them together. */
const REDUCTION_WGSL = /* wgsl */ `
const LANES: u32 = ${REDUCTION_LANES}u;
var<workgroup> floatScratch: array<f32, ${REDUCTION_LANES}>;
var<workgroup> errorScratch: array<f32, ${REDUCTION_LANES}>;
var<workgroup> countScratch: array<u32, ${REDUCTION_LANES}>;
fn addReferenceSum(total: vec2<f32>, value: f32, zero: u32) -> vec2<f32> {
  let next = getRoundedDifference(total.x + value, 0.0, zero);
  let remainder = total.y + getAdditionRoundingError(total.x, value, next);
  let high = getRoundedDifference(next + remainder, 0.0, zero);
  return vec2<f32>(high, getAdditionRoundingError(next, remainder, high));
}
fn reduceReferenceSum(value: vec2<f32>, lane: u32, zero: u32) -> vec2<f32> {
  workgroupBarrier();
  floatScratch[lane] = value.x; errorScratch[lane] = value.y;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      var total = addReferenceSum(vec2<f32>(floatScratch[lane], errorScratch[lane]), floatScratch[lane + stride], zero);
      total = addReferenceSum(total, errorScratch[lane + stride], zero);
      floatScratch[lane] = total.x; errorScratch[lane] = total.y;
    }
    workgroupBarrier();
  }
  return vec2<f32>(floatScratch[0], errorScratch[0]);
}
fn reduceMaximum(value: f32, lane: u32) -> f32 {
  workgroupBarrier();
  floatScratch[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) { floatScratch[lane] = max(floatScratch[lane], floatScratch[lane + stride]); }
    workgroupBarrier();
  }
  return floatScratch[0];
}
fn reduceSum(value: f32, lane: u32) -> f32 {
  workgroupBarrier();
  floatScratch[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      floatScratch[lane] += floatScratch[lane + stride];
    }
    workgroupBarrier();
  }
  return floatScratch[0];
}
fn reduceCount(value: u32, lane: u32) -> u32 {
  workgroupBarrier();
  countScratch[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      countScratch[lane] += countScratch[lane + stride];
    }
    workgroupBarrier();
  }
  return countScratch[0];
}
fn reduceFirstRow(value: u32, lane: u32) -> u32 {
  workgroupBarrier();
  countScratch[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) { countScratch[lane] = min(countScratch[lane], countScratch[lane + stride]); }
    workgroupBarrier();
  }
  return countScratch[0];
}`;

/** Order-preserving u32 key of a finite f32 (-0 and +0 share a key); WGSL for the rank sort. */
const RANK_KEY_WGSL = /* wgsl */ `
const INVALID_KEY: u32 = 0xffffffffu;
fn getOrderedKey(value: f32) -> u32 {
  // Integer test: a float compare of -0 against 0 can be folded away by the shader compiler.
  let raw = bitcast<u32>(value);
  let bits = select(raw, 0u, (raw & 0x7fffffffu) == 0u);
  return select(bits | 0x80000000u, ~bits, (bits >> 31u) == 1u);
}`;

/**
 * How attribute columns are standardized before distances are measured:
 * - `'z-score'`: `(x - mean) / standard deviation` over the valid rows (population deviation); a
 *   constant column contributes zero.
 * - `'rank'`: average-tie percentile rank in `[0, 1]`, `(rank - 1) / (n - 1)`. Below 4096 rows
 *   the ranking is an all-pairs count per column; from 4096 rows each column is radix sorted and
 *   the counts come from binary searches (O(n log n)). Both give the same ranks.
 */
export type GPUSimilarLocationsStandardization = 'z-score' | 'rank';

/** Caller-owned outputs of {@link GPUSimilarLocations}. */
export type GPUSimilarLocationsOutput = {
  /**
   * Similarity rank per row, uint32. 0 is the most similar (or, with `direction: 'least'`, the
   * least similar). Rows that cannot be ranked hold `GPU_SIMILAR_LOCATIONS_NO_RANK`.
   */
  ranks: GraphDataView<'uint32'>;
  /** Weighted Euclidean distance to the reference in standardized space; NaN when unranked. */
  distances: GraphDataView<'float32'>;
  /**
   * Row IDs of the best `min(resultCount, eligible rows)` rows in rank order, at least
   * `resultCapacity` uint32 rows; unused rows hold `GPU_SIMILAR_LOCATIONS_NO_RANK`.
   */
  topIds: GraphDataView<'uint32'>;
  /** Status words: `count` is the number of valid entries of `topIds`. */
  status: Pick<GPUCardinalityPort, 'count'>;
};

/**
 * Properties for {@link GPUSimilarLocations}.
 *
 * Per-frame (no recompile): the contents of `parameters`, `referenceMask`, `mask` and `values`.
 * Topology (needs a new graph): `rowCount`, `columnCount`, `standardization` and
 * `resultCapacity`, and whether `mask` is present.
 */
export type GPUSimilarLocationsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'similar-locations'`. */
  id?: string;
  /**
   * Row-major packed float32 attribute matrix, `rowCount * columnCount` elements. A row with
   * any non-finite attribute is invalid and is neither ranked nor used in the statistics. At most
   * {@link getGPUSimilarLocationsMaximumValueCount} elements (33554432 on a default device).
   */
  values: GraphDataView<'float32'>;
  /**
   * Number of attribute columns, at least 1 and at most
   * {@link GPU_SIMILAR_LOCATIONS_MAXIMUM_COLUMN_COUNT}.
   */
  columnCount: number;
  /** Optional packed `uint32` row mask; zero makes the row invalid. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Packed `uint32` reference selection, one per row; non-zero rows form the reference. The
   * reference vector is the mean of their standardized attributes, so a single selected row
   * compares everything to that row. With no valid selected row every row is unranked.
   */
  referenceMask: GraphDataView<'uint32'>;
  /** Per-frame float32 view written with `getGPUSimilarLocationsParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Standardization of the attribute columns. Defaults to `'z-score'`. */
  standardization?: GPUSimilarLocationsStandardization;
  /** Compile-time capacity of `output.topIds`, at most `rowCount`. Defaults to `min(rowCount, 32)`. */
  resultCapacity?: number;
  /** Caller-owned outputs. */
  output: GPUSimilarLocationsOutput;
};

/**
 * Feature-space nearest neighbors: ranks every row by its weighted Euclidean distance in
 * standardized attribute space to a reference row, or to the mean of several reference rows
 * (the CARTO `FIND_SIMILAR_LOCATIONS` and ArcGIS Similarity Search model).
 *
 * The graph validates rows, standardizes the columns (z-score or rank), reduces the reference
 * vector, measures distances and orders them with the stable `GPUSort` on the f32 bit pattern of
 * the distance, so ties keep the lowest row ID. No float atomics are used and results are
 * deterministic. Rows are ranked only when valid, unmasked and not excluded as references.
 *
 * Matches a float64 CPU oracle to f32 rounding; rows whose distances differ by less than that
 * rounding may swap ranks against the oracle.
 */
export class GPUSimilarLocations implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSimilarLocationsProps;
  /** Number of rows. */
  readonly rowCount: number;
  /** Capacity of `output.topIds`. */
  readonly resultCapacity: number;

  constructor(props: GPUSimilarLocationsProps) {
    props = {...props, output: {...props.output, status: {...props.output.status}}};
    this.id = props.id ?? 'similar-locations';
    this.props = props;
    const {id} = this;
    validateCapacity(
      id,
      'columnCount',
      props.columnCount,
      1,
      GPU_SIMILAR_LOCATIONS_MAXIMUM_COLUMN_COUNT
    );
    validateSinglePackedView(props.values, ['float32'], `${id} values`);
    if (props.values.length % props.columnCount !== 0) {
      throw new Error(`${id} values length must be a multiple of columnCount`);
    }
    this.rowCount = props.values.length / props.columnCount;
    if (this.rowCount < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    validateRequirement(
      id,
      props.standardization === undefined || ['z-score', 'rank'].includes(props.standardization),
      `standardization must be 'z-score' or 'rank'; received ${props.standardization}`
    );
    this.resultCapacity = props.resultCapacity ?? Math.min(this.rowCount, 32);
    validateCapacity(id, 'resultCapacity', this.resultCapacity, 1, this.rowCount);
    if (props.mask) {
      validateSinglePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== this.rowCount) {
        throw new Error(`${id} mask length must equal the row count`);
      }
    }
    validateSinglePackedView(props.referenceMask, ['uint32'], `${id} referenceMask`);
    if (props.referenceMask.length !== this.rowCount) {
      throw new Error(`${id} referenceMask length must equal the row count`);
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    const parameterLength = getGPUSimilarLocationsParameterLength(props.columnCount);
    if (props.parameters.length < parameterLength) {
      throw new Error(`${id} parameters must hold ${parameterLength} float32 values`);
    }
    validateSinglePackedView(props.output.ranks, ['uint32'], `${id} output.ranks`);
    validateSinglePackedView(props.output.distances, ['float32'], `${id} output.distances`);
    validateSinglePackedView(props.output.topIds, ['uint32'], `${id} output.topIds`);
    validateSinglePackedView(props.output.status.count, ['uint32'], `${id} output.status.count`);
    if (
      props.output.ranks.length < this.rowCount ||
      props.output.distances.length < this.rowCount
    ) {
      throw new Error(`${id} output.ranks and output.distances must hold one row per input row`);
    }
    if (props.output.topIds.length < this.resultCapacity) {
      throw new Error(`${id} output.topIds must hold resultCapacity rows`);
    }
    if (props.output.status.count.length < 1) {
      throw new Error(`${id} output.status.count must hold one row`);
    }
    validateDisjointWritableViews(id, {...props.output, status: undefined, ...props.output.status});
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.ranks, props.output.distances, props.output.topIds, props.output.status.count],
      [props.values, props.mask, props.referenceMask, props.parameters]
    );
  }

  /** Returns validation, standardization, reference, distance, sort and rank nodes, in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, resultCapacity} = this;
    validateSpatialStatisticsBindings(graph, id, props, {
      ...(props.standardization === 'rank'
        ? {'standardized matrix': props.values.length * 4, 'rank reference': props.columnCount * 4}
        : {'column moments': props.columnCount * 16}),
      'sort keys and IDs': rowCount * 4
    });
    const {output, columnCount: attributeCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.mask,
      props.referenceMask,
      props.parameters,
      output.ranks,
      output.distances,
      output.topIds,
      output.status.count
    ]);
    validateMaximumCount(
      id,
      'values rows',
      props.values.length,
      getGPUSimilarLocationsMaximumValueCount(graph.device)
    );
    const isRank = props.standardization === 'rank';
    const matrixLength = rowCount * attributeCount;
    const validity = createTransientView(graph, `${id}-validity`, 'uint32', rowCount);
    const standardized = isRank
      ? createTransientView(graph, `${id}-standardized`, 'float32', matrixLength)
      : undefined;
    const reference = createTransientView(
      graph,
      `${id}-reference`,
      'float32',
      attributeCount * (isRank ? 1 : 4)
    );
    let columnStatistics: GraphDataView<'float32'> | undefined;
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', rowCount);
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rowCount);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', rowCount);
    const sortedIds = createTransientView(graph, `${id}-sorted-ids`, 'uint32', rowCount);
    const declarations = `const ROW_COUNT: u32 = ${rowCount}u;
const ATTRIBUTE_COUNT: u32 = ${attributeCount}u;
const MAX_RESULTS: u32 = ${resultCapacity}u;
const NO_RANK: u32 = 0xffffffffu;
${IS_FINITE_VALUE_WGSL}
${GET_NAN_WGSL}
${ROUNDED_DIFFERENCE_WGSL}
${ADDITION_ROUNDING_ERROR_WGSL}
${MULTIPLICATION_ROUNDING_ERROR_WGSL}`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const kernel = (
      name: string,
      bindings: WGSLKernelBinding[],
      invocationCount: number,
      body: string,
      options: {reduction?: boolean; declarations?: string} = {}
    ) =>
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${name}`,
          operation: OPERATION,
          variant: name,
          bindings,
          // A reduction kernel is one workgroup per output; every lane reaches the barriers.
          invocationCount: options.reduction ? invocationCount * REDUCTION_LANES : invocationCount,
          workgroupSize: options.reduction ? REDUCTION_LANES : 64,
          guardIndex: options.reduction ? false : undefined,
          declarations: `${declarations}\n${options.reduction ? REDUCTION_WGSL : ''}\n${options.declarations ?? ''}`,
          body
        })
      );
    const read = (name: string, view: GraphDataView, type: 'f32' | 'u32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (name: string, view: GraphDataView, type: 'f32' | 'u32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read_write'
    });

    kernel(
      'validate',
      [
        read('attributes', props.values, 'f32'),
        ...(props.mask ? [read('rowMask', props.mask, 'u32')] : []),
        write('validity', validity, 'u32')
      ],
      rowCount,
      `var isValid = true;
  ${props.mask ? 'isValid = rowMask[rowMaskOffset + index] != 0u;' : ''}
  for (var channel = 0u; channel < ATTRIBUTE_COUNT; channel++) {
    isValid = isValid && isFiniteValue(attributes[attributesOffset + index * ATTRIBUTE_COUNT + channel]);
  }
  validity[validityOffset + index] = select(0u, 1u, isValid);`
    );

    if (isRank && rowCount < RANK_SORT_MINIMUM_ROWS) {
      kernel(
        'standardize-rank',
        [
          read('attributes', props.values, 'f32'),
          read('validity', validity, 'u32'),
          write('standardized', standardized!, 'f32')
        ],
        matrixLength,
        `let row = index / ATTRIBUTE_COUNT;
  let channel = index % ATTRIBUTE_COUNT;
  if (validity[validityOffset + row] == 0u) {
    standardized[standardizedOffset + index] = getNaN();
    return;
  }
  let value = attributes[attributesOffset + index];
  var less = 0u;
  var equal = 0u;
  var valid = 0u;
  for (var other = 0u; other < ROW_COUNT; other++) {
    if (validity[validityOffset + other] == 0u) {
      continue;
    }
    valid += 1u;
    let otherValue = attributes[attributesOffset + other * ATTRIBUTE_COUNT + channel];
    less += select(0u, 1u, otherValue < value);
    equal += select(0u, 1u, otherValue == value);
  }
  let denominator = max(f32(valid) - 1.0, 1.0);
  standardized[standardizedOffset + index] = (f32(less) + 0.5 * (f32(equal) - 1.0)) / denominator;`
      );
    } else if (isRank) {
      // Average-tie percentile ranks from one stable sort per column: the exact less-than and
      // equal counts are two binary searches on the sorted keys, the same integers (so the same
      // floats) as the all-pairs count.
      const rankKeys = createTransientView(graph, `${id}-rank-keys`, 'uint32', rowCount);
      const rankRowIds = createTransientView(graph, `${id}-rank-row-ids`, 'uint32', rowCount);
      const rankSortedKeys = createTransientView(
        graph,
        `${id}-rank-sorted-keys`,
        'uint32',
        rowCount
      );
      const rankSortedIds = createTransientView(graph, `${id}-rank-sorted-ids`, 'uint32', rowCount);
      // Column cursor: a one-thread node advances it before each column, so the column index never
      // reaches the WGSL text and every column shares one pipeline per kernel (rule 12).
      const rankColumn = createTransientView(graph, `${id}-rank-column`, 'uint32', 1);
      kernel(
        'rank-column-clear',
        [write('rankColumn', rankColumn, 'u32')],
        1,
        'rankColumn[rankColumnOffset] = 0xffffffffu;'
      );
      for (let channel = 0; channel < attributeCount; channel++) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-rank-column-${channel}`,
            operation: OPERATION,
            variant: 'rank-column',
            bindings: [write('rankColumn', rankColumn, 'u32')],
            invocationCount: 1,
            body: 'rankColumn[rankColumnOffset] = rankColumn[rankColumnOffset] + 1u;'
          })
        );
        kernel(
          `rank-keys-${channel}`,
          [
            read('attributes', props.values, 'f32'),
            read('validity', validity, 'u32'),
            write('rankKeys', rankKeys, 'u32'),
            write('rankRowIds', rankRowIds, 'u32'),
            read('rankColumn', rankColumn, 'u32')
          ],
          rowCount,
          `let value = attributes[attributesOffset + index * ATTRIBUTE_COUNT + rankColumn[rankColumnOffset]];
  rankKeys[rankKeysOffset + index] =
    select(INVALID_KEY, getOrderedKey(value), validity[validityOffset + index] != 0u);
  rankRowIds[rankRowIdsOffset + index] = index;`,
          {declarations: RANK_KEY_WGSL}
        );
        nodes.push(
          ...new GPUSort({
            id: `${id}-rank-sort-${channel}`,
            keys: rankKeys,
            values: rankRowIds,
            outputKeys: rankSortedKeys,
            outputValues: rankSortedIds,
            keyBits: 32
          }).getCommandNodes(graph)
        );
        kernel(
          `rank-assign-${channel}`,
          [
            read('sortedKeys', rankSortedKeys, 'u32'),
            read('sortedIds', rankSortedIds, 'u32'),
            write('standardized', standardized!, 'f32'),
            read('rankColumn', rankColumn, 'u32')
          ],
          rowCount,
          `let key = sortedKeys[sortedKeysOffset + index];
  let row = sortedIds[sortedIdsOffset + index];
  let column = rankColumn[rankColumnOffset];
  if (key == INVALID_KEY) {
    standardized[standardizedOffset + row * ATTRIBUTE_COUNT + column] = getNaN();
    return;
  }
  let below = lowerBound(key);
  let notAbove = lowerBound(key + 1u);
  let valid = lowerBound(INVALID_KEY);
  let denominator = max(f32(valid) - 1.0, 1.0);
  standardized[standardizedOffset + row * ATTRIBUTE_COUNT + column] =
    (f32(below) + 0.5 * (f32(notAbove - below) - 1.0)) / denominator;`,
          {
            declarations: `${RANK_KEY_WGSL}
fn lowerBound(bound: u32) -> u32 {
  var low = 0u;
  var high = ROW_COUNT;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < bound) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`
          }
        );
      }
    } else {
      const stats = createTransientView(graph, `${id}-stats`, 'float32', 4 * attributeCount);
      columnStatistics = stats;
      kernel(
        'column-statistics',
        [
          read('attributes', props.values, 'f32'),
          read('validity', validity, 'u32'),
          write('stats', stats, 'f32')
        ],
        attributeCount,
        `let column = index / LANES;
  let lane = localInvocationIndex;
  var firstRow = ROW_COUNT;
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u) { firstRow = row; break; }
  }
  let anchorRow = min(reduceFirstRow(firstRow, lane), ROW_COUNT - 1u);
  let anchor = attributes[attributesOffset + anchorRow * ATTRIBUTE_COUNT + column];
  let runtimeZero = validity[validityOffset + anchorRow] - 1u;
  // A common binary frame keeps squares normal when attribute units are tiny or huge.
  var maximumPartial = 0.0;
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u) {
      maximumPartial = max(maximumPartial, abs(attributes[attributesOffset + row * ATTRIBUTE_COUNT + column]));
    }
  }
  let maximum = reduceMaximum(maximumPartial, lane);
  let exponent = (bitcast<u32>(maximum) >> 23u) & 255u;
  // Keep the exponent, not a tiny multiplier whose reciprocal/product can flush or be
  // reassociated into the final normalization on Metal. ldexp fixes the normalized frame.
  let scaleExponent = 127 - i32(clamp(exponent, 1u, 254u));
  var validPartial = 0u;
  var sumPartial = 0.0;
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u) {
      validPartial += 1u;
      sumPartial += getRoundedDifference(ldexp(attributes[attributesOffset + row * ATTRIBUTE_COUNT + column], scaleExponent), ldexp(anchor, scaleExponent), runtimeZero);
    }
  }
  let valid = reduceCount(validPartial, lane);
  let mean = select(0.0, reduceSum(sumPartial, lane) / f32(valid), valid > 0u);
  var squaresPartial = 0.0;
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u) {
      let centered = getRoundedDifference(ldexp(attributes[attributesOffset + row * ATTRIBUTE_COUNT + column], scaleExponent), ldexp(anchor, scaleExponent), runtimeZero);
      let delta = getRoundedDifference(centered, mean, runtimeZero);
      squaresPartial += delta * delta;
    }
  }
  let squares = reduceSum(squaresPartial, lane);
  if (lane == 0u) {
    let deviation = select(0.0, sqrt(squares / f32(valid)), valid > 0u);
    stats[statsOffset + 4u * column] = anchor;
    stats[statsOffset + 4u * column + 3u] = f32(scaleExponent);
    stats[statsOffset + 4u * column + 1u] = mean;
    stats[statsOffset + 4u * column + 2u] = select(0.0, 1.0 / deviation, deviation > 0.0);
  }`,
        {reduction: true}
      );
    }

    kernel(
      'reference',
      [
        isRank
          ? read('standardized', standardized!, 'f32')
          : read('attributes', props.values, 'f32'),
        ...(!isRank ? [read('stats', columnStatistics!, 'f32')] : []),
        read('validity', validity, 'u32'),
        read('selection', props.referenceMask, 'u32'),
        write('reference', reference, 'f32')
      ],
      attributeCount,
      `let column = index / LANES;
  let lane = localInvocationIndex;
  ${
    isRank
      ? ''
      : `var firstRow = ROW_COUNT;
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u && selection[selectionOffset + row] != 0u) { firstRow = row; break; }
  }
  let anchorRow = min(reduceFirstRow(firstRow, lane), ROW_COUNT - 1u);
  let scaleExponent = i32(stats[statsOffset + 4u * column + 3u]);
  let anchor = ldexp(attributes[attributesOffset + anchorRow * ATTRIBUTE_COUNT + column], scaleExponent);
  let zero = validity[validityOffset + anchorRow] - 1u;`
  }
  var countPartial = 0u;
  var sumPartial = ${isRank ? '0.0' : 'vec2<f32>(0.0)'};
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u && selection[selectionOffset + row] != 0u) {
      countPartial += 1u;
      ${
        isRank
          ? 'sumPartial += standardized[standardizedOffset + row * ATTRIBUTE_COUNT + column];'
          : `let value = ldexp(attributes[attributesOffset + row * ATTRIBUTE_COUNT + column], scaleExponent);
      let centered = getRoundedDifference(value, anchor, zero);
      sumPartial = addReferenceSum(sumPartial, centered, zero);
      sumPartial = addReferenceSum(sumPartial, getAdditionRoundingError(value, -anchor, centered), zero);`
      }
    }
  }
  let count = reduceCount(countPartial, lane);
  let sum = ${isRank ? 'reduceSum(sumPartial, lane)' : 'reduceReferenceSum(sumPartial, lane, zero)'};
  if (lane == 0u) {
    ${
      isRank
        ? 'reference[referenceOffset + column] = select(getNaN(), sum / f32(count), count > 0u);'
        : `let divisor = f32(max(count, 1u));
    let mean = sum.x / divisor;
    let product = getRoundedDifference(mean * divisor, 0.0, zero);
    let remainder = (getRoundedDifference(sum.x, product, zero) - getMultiplicationRoundingError(mean, divisor, product) + sum.y) / divisor;
    var center = addReferenceSum(vec2<f32>(anchor, 0.0), mean, zero);
    center = addReferenceSum(center, remainder, zero);
    reference[referenceOffset + 4u * column] = select(getNaN(), center.x, count > 0u);
    reference[referenceOffset + 4u * column + 1u] = center.y;
    reference[referenceOffset + 4u * column + 3u] = stats[statsOffset + 4u * column + 3u];
    reference[referenceOffset + 4u * column + 2u] = stats[statsOffset + 4u * column + 2u];`
    }
  }`,
      {reduction: true}
    );

    kernel(
      'distance',
      [
        isRank
          ? read('standardized', standardized!, 'f32')
          : read('attributes', props.values, 'f32'),
        read('validity', validity, 'u32'),
        read('selection', props.referenceMask, 'u32'),
        read('reference', reference, 'f32'),
        read('params', props.parameters, 'f32'),
        write('keys', keys, 'u32'),
        write('rowIds', rowIds, 'u32'),
        write('distances', output.distances, 'f32')
      ],
      rowCount,
      `rowIds[rowIdsOffset + index] = index;
  let isLeast = params[paramsOffset + 1u] > 0.5;
  let excludeReference = params[paramsOffset + 2u] > 0.5;
  var eligible = validity[validityOffset + index] != 0u &&
    !(excludeReference && selection[selectionOffset + index] != 0u);
  var distanceSquared = 0.0;
  for (var channel = 0u; channel < ATTRIBUTE_COUNT; channel++) {
    let weight = params[paramsOffset + ${GPU_SIMILAR_LOCATIONS_CONTROL_LENGTH}u + channel];
    ${
      isRank
        ? `let delta = standardized[standardizedOffset + index * ATTRIBUTE_COUNT + channel] -
      reference[referenceOffset + channel];`
        : `let value = ldexp(attributes[attributesOffset + index * ATTRIBUTE_COUNT + channel], i32(reference[referenceOffset + 4u * channel + 3u]));
    let center = reference[referenceOffset + 4u * channel];
    let centered = getRoundedDifference(value, center, validity[validityOffset + index] - 1u);
    let differenceError = getAdditionRoundingError(value, -center, centered);
    let delta = (getRoundedDifference(centered, reference[referenceOffset + 4u * channel + 1u], validity[validityOffset + index] - 1u) + differenceError) * reference[referenceOffset + 4u * channel + 2u];`
    }
    distanceSquared += weight * delta * delta;
  }
  eligible = eligible && isFiniteValue(distanceSquared);
  let distance = sqrt(distanceSquared);
  // Non-negative f32 bit patterns order like the values; the sentinel sorts after every key.
  let bits = bitcast<u32>(distance);
  keys[keysOffset + index] = select(NO_RANK, select(bits, 0xfffffffeu - bits, isLeast), eligible);
  distances[distancesOffset + index] = select(getNaN(), distance, eligible);`
    );

    nodes.push(
      ...new GPUSort({
        id: `${id}-sort`,
        keys,
        values: rowIds,
        outputKeys: sortedKeys,
        outputValues: sortedIds,
        keyBits: 32
      }).getCommandNodes(graph)
    );

    if (output.topIds.length > resultCapacity) {
      kernel(
        'top-ids-tail',
        [write('topIds', output.topIds, 'u32')],
        output.topIds.length - resultCapacity,
        'topIds[topIdsOffset + MAX_RESULTS + index] = NO_RANK;'
      );
    }
    kernel(
      'rank',
      [
        read('sortedKeys', sortedKeys, 'u32'),
        read('sortedIds', sortedIds, 'u32'),
        read('params', props.parameters, 'f32'),
        write('ranks', output.ranks, 'u32'),
        write('topIds', output.topIds, 'u32'),
        write('resultCount', output.status.count, 'u32')
      ],
      rowCount,
      `let key = sortedKeys[sortedKeysOffset + index];
  let isEligible = key != NO_RANK;
  let row = sortedIds[sortedIdsOffset + index];
  ranks[ranksOffset + row] = select(NO_RANK, index, isEligible);
  let requested = params[paramsOffset];
  let limit = min(u32(clamp(select(0.0, requested, isFiniteValue(requested)), 0.0, f32(MAX_RESULTS))), MAX_RESULTS);
  // Eligible rows sort first, so the boundary row owns the eligible count.
  let isLast = index + 1u == ROW_COUNT || sortedKeys[sortedKeysOffset + index + 1u] == NO_RANK;
  if (isEligible && isLast) {
    resultCount[resultCountOffset] = min(index + 1u, limit);
  }
  if (index == 0u && !isEligible) {
    resultCount[resultCountOffset] = 0u;
  }
  if (index < MAX_RESULTS) {
    topIds[topIdsOffset + index] = select(NO_RANK, row, isEligible && index < limit);
  }`
    );
    return nodes;
  }
}
