// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createStatisticsKernelNode as createWGSLKernelNode} from './statistics-kernel-node';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import {
  getGPUSpatialWeightsSummaryParameterValues,
  GPU_SPATIAL_WEIGHTS_SUMMARY_PARAMETER_LENGTH
} from './spatial-weights-parameters';
import {
  validateCapacity,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getSlotGroupingNodes} from './slot-grouping';
import {getGPUSpatialWeightsNeighborLoopWGSL, type GPUSpatialWeights} from './spatial-weights';
import {
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsOrChunksViews,
  validateGPUSpatialWeightsOrChunks,
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeightsChunkDispatch
} from './gpu-chunked-spatial-weights';

const OPERATION = 'GPUSpatialWeightsSummary';

/** Rows reduced by one workgroup in the first level of the global reductions. */
const SUMMARY_BLOCK_ROWS = 4096;
/** Threads of the reduction workgroups. */
const SUMMARY_WORKGROUP_SIZE = 256;

/** Element positions of the `statistics` and `counts` outputs of {@link GPUSpatialWeightsSummary}. */
export const GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT = {
  /** Float32 `statistics`: `S0`, `S1`, `S2`. */
  statistics: {s0: 0, s1: 1, s2: 2, length: 3},
  /**
   * Uint32 `counts`: used slots, asymmetric slots, isolates (rows with no active slot), minimum
   * and maximum cardinality (active slots per row) and empty rows (rows with no slot at all).
   */
  counts: {
    slots: 0,
    asymmetricSlots: 1,
    isolates: 2,
    minimumCardinality: 3,
    maximumCardinality: 4,
    emptyRows: 5,
    length: 6
  }
} as const;

/** Caller-owned outputs of {@link GPUSpatialWeightsSummary}. */
export type GPUSpatialWeightsSummaryOutput = {
  /** Float32 output of at least 3 values, see {@link GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT}. */
  statistics: GraphDataView<'float32'>;
  /** Uint32 output of at least 6 values, see {@link GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT}. */
  counts: GraphDataView<'uint32'>;
  /** Optional per-row cardinality (active neighbor count), one uint32 per row. */
  cardinality?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUSpatialWeightsSummary}. */
export type GPUSpatialWeightsSummaryProps = {
  /** Prefix for generated node IDs. Defaults to `'spatial-weights-summary'`. */
  id?: string;
  /**
   * Square weights to summarize. Chunked weights run the row kernels once per chunk and the
   * reverse-slot (symmetry) lookup once per chunk pair; the results equal the whole matrix.
   */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /** Caller-owned outputs. */
  output: GPUSpatialWeightsSummaryOutput;
  /**
   * Absolute tolerance for calling `w_ij` and `w_ji` equal when counting asymmetric slots.
   * Per-frame: a one-float parameter view packed by getGPUSpatialWeightsSummaryParameterValues,
   * or a numeric default uploaded through GPUParameterBuffer. Defaults to 0 (exact).
   */
  symmetryTolerance?: number | GraphDataView<'float32'>;
};

/**
 * Per-frame: weights and symmetryTolerance. Compile-time: view shapes and chunk layout.
 *
 * Descriptive statistics of a {@link GPUSpatialWeights} on the GPU, the libpysal `W.s0/s1/s2`,
 * `cardinalities`, `asymmetry()` and `islands` summary:
 *
 * - `S0 = sum_ij w_ij`
 * - `S1 = 1/2 sum_ij (w_ij + w_ji)^2`, a missing `w_ji` counting as 0
 * - `S2 = sum_i (sum_j w_ij + sum_j w_ji)^2`
 * - used slots (`offsets[rows]`), minimum and maximum cardinality, and per-row cardinality
 * - cardinality: the number of ACTIVE slots of a row (positive weight, neighbor below the row
 *   count, not the row itself; the neighbor definition of every statistic, see
 *   `getGPUSpatialWeightsNeighborLoopWGSL`). libpysal `cardinalities` counts every listed
 *   neighbor, zero weights included. We deviate on purpose: a zero-weight slot is not a
 *   neighbor, so the counts agree with libpysal for a W built without zero-weight entries. Use
 *   the used slots for the listed total.
 * - asymmetric slots: slots `(i, j)` whose reverse slot `(j, i)` is missing or differs by more
 *   than `symmetryTolerance` (a pair of different weights counts twice, a one-way link once)
 * - isolates (the one island definition): rows with no active slot, which is libpysal `islands`
 *   for a W without zero-weight entries. A row whose only slots are zero-weight or self slots
 *   is an isolate here, as it is for every statistic.
 * - empty rows: rows with no slot at all (`offsets[i + 1] == offsets[i]`), at most the isolates
 *
 * Sums are fixed-order (row sums in slot order, column sums over a stable sort, a tree
 * reduction), so the results are bitwise reproducible. Slots with a neighbor ID `>= rows` are
 * ignored by the column sums and counted as one-way links. Self slots are included as in libpysal.
 */
export class GPUSpatialWeightsSummary implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialWeightsSummaryProps;
  private readonly toleranceValues?: Float32Array;

  constructor(props: GPUSpatialWeightsSummaryProps) {
    const id = props.id ?? 'spatial-weights-summary';
    this.id = id;
    this.props = props;
    const rows = validateGPUSpatialWeightsOrChunks(id, props.weights);
    validateSinglePackedView(props.output.statistics, ['float32'], `${id} statistics`);
    if (props.output.statistics.length < GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.statistics.length) {
      throw new Error(`${id} statistics must hold at least 3 values`);
    }
    validateSinglePackedView(props.output.counts, ['uint32'], `${id} counts`);
    if (props.output.counts.length < GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.counts.length) {
      throw new Error(`${id} counts must hold at least 6 values`);
    }
    if (props.output.cardinality) {
      validateSinglePackedView(props.output.cardinality, ['uint32'], `${id} cardinality`);
      if (props.output.cardinality.length !== rows) {
        throw new Error(`${id} cardinality length must equal the weights row count`);
      }
    }
    const tolerance = props.symmetryTolerance ?? 0;
    if (typeof tolerance === 'number') {
      validateFiniteNumber(id, 'symmetryTolerance', tolerance, 0);
      this.toleranceValues = getGPUSpatialWeightsSummaryParameterValues({
        symmetryTolerance: tolerance
      });
    } else {
      validateSinglePackedView(tolerance, ['float32'], `${id} symmetryTolerance`);
      validateCapacity(
        id,
        'symmetryTolerance.length',
        tolerance.length,
        GPU_SPATIAL_WEIGHTS_SUMMARY_PARAMETER_LENGTH
      );
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.statistics, props.output.counts, props.output.cardinality],
      [
        ...getGPUSpatialWeightsOrChunksViews(props.weights),
        typeof props.symmetryTolerance === 'object' ? props.symmetryTolerance : undefined
      ]
    );
  }

  /** Returns the summary nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {statistics, counts} = props.output;
    const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights);
    const isChunked = !dispatches[0].isWhole;
    const weights = dispatches[0].weights;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUSpatialWeightsOrChunksViews(props.weights),
      statistics,
      counts,
      props.output.cardinality,
      typeof props.symmetryTolerance === 'object' ? props.symmetryTolerance : undefined
    ]);
    if (typeof props.symmetryTolerance === 'object') {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'symmetryTolerance binding',
        getViewBindingRange(props.symmetryTolerance).size,
        'use a smaller packed symmetryTolerance view'
      );
    }
    const rows = dispatches.reduce((total, dispatch) => total + dispatch.rowCount, 0);
    const capacity = weights.neighbors.length;
    // A leading word stages the tolerance beside row mismatch counts, keeping reverse-chunk
    // kernels within WebGPU's guaranteed eight-storage-buffer limit.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'symmetry state',
      (rows + 1) * 4,
      'use fewer weights rows'
    );
    const runtime = createSpatialStatisticsParameterNodes(
      graph,
      `${id}-parameters`,
      this.toleranceValues ?? (props.symmetryTolerance as GraphDataView<'float32'>)
    );
    const tolerance = runtime.parameters;
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const columnSums = view('column-sums', 'float32', rows);
    const rowSums = view('row-sums', 'float32', rows);
    const s1Partials = view('s1-partials', 'float32', rows);
    const s2Partials = view('s2-partials', 'float32', rows);
    const symmetryState = view('symmetry-state', 'uint32', rows + 1);
    const asymmetric = graph.createDataView(symmetryState.buffer, {
      format: 'uint32',
      length: rows,
      byteOffset: 4
    });
    const isolates = view('isolates', 'uint32', rows);
    const cardinality = props.output.cardinality ?? view('cardinality', 'uint32', rows);
    const chunkSlotCounts = view('chunk-slot-counts', 'uint32', dispatches.length);
    const constants = `const ROWS: u32 = ${rows}u;`;
    const getWholeNodes = (): GPUCommandNode<Parameters>[] => {
      const slotKeys = view('slot-keys', 'uint32', capacity);
      const slotIndices = view('slot-indices', 'uint32', capacity);
      const columnOffsets = view('column-offsets', 'uint32', rows + 1);
      const grouping = getSlotGroupingNodes<Parameters>(graph, {
        id: `${id}-columns`,
        operation: OPERATION,
        slotKeys,
        slotIndices,
        columns: rows,
        columnOffsets
      });
      return [
        // Column sums: group the used slots by neighbor, in slot order, so the sums are reproducible.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-slot-keys`,
          operation: OPERATION,
          variant: 'slot-keys',
          bindings: [
            {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
            {name: 'slotKeys', view: slotKeys, type: 'u32', access: 'read_write'},
            {name: 'slotIndices', view: slotIndices, type: 'u32', access: 'read_write'}
          ],
          invocationCount: capacity,
          declarations: constants,
          body: `let neighbor = neighbors[neighborsOffset + index];
  let used = index < offsets[offsetsOffset + ROWS] && neighbor < ROWS;
  slotKeys[slotKeysOffset + index] = select(ROWS, neighbor, used);
  slotIndices[slotIndicesOffset + index] = index;`
        }),
        ...grouping.nodes,
        // One invocation per column: short segments (the degree) do not deserve a workgroup each.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-column-sums`,
          operation: OPERATION,
          variant: 'column-sums',
          bindings: [
            {name: 'columnOffsets', view: columnOffsets, type: 'u32', access: 'read'},
            {name: 'sortedSlots', view: grouping.sortedSlots, type: 'u32', access: 'read'},
            {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
            {name: 'columnSums', view: columnSums, type: 'f32', access: 'read_write'}
          ],
          invocationCount: rows,
          // Slots of one column are in ascending slot order (stable sort), so the sum order is fixed.
          body: `var sum = 0.0;
  for (var position = columnOffsets[columnOffsetsOffset + index];
      position < columnOffsets[columnOffsetsOffset + index + 1u]; position++) {
    sum += weights[weightsOffset + sortedSlots[sortedSlotsOffset + position]];
  }
  columnSums[columnSumsOffset + index] = sum;`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-rows`,
          operation: OPERATION,
          variant: 'rows',
          bindings: [
            {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
            {name: 'rowSums', view: rowSums, type: 'f32', access: 'read_write'},
            {name: 's1Partials', view: s1Partials, type: 'f32', access: 'read_write'},
            {name: 'asymmetric', view: asymmetric, type: 'u32', access: 'read_write'},
            {name: 'cardinality', view: cardinality, type: 'u32', access: 'read_write'},
            {name: 'tolerance', view: tolerance, type: 'f32', access: 'read'}
          ],
          invocationCount: rows,
          declarations: `${constants}
const CAPACITY: u32 = ${capacity}u;
const MISSING: f32 = -1.0;

// Weight w_row,column by binary search of the ascending neighbor IDs of row, or MISSING.
fn findWeight(row: u32, column: u32) -> f32 {
  var low = offsets[offsetsOffset + row];
  let end = offsets[offsetsOffset + row + 1u];
  var high = end;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (neighbors[neighborsOffset + middle] < column) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && neighbors[neighborsOffset + low] == column) {
    return weights[weightsOffset + low];
  }
  return MISSING;
}`,
          body: `var sum = 0.0;
  var s1 = 0.0;
  var unmatched = 0u;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let weight = weights[weightsOffset + slot];
    sum += weight;
    var reverse = MISSING;
    if (neighbor < ROWS) {
      reverse = findWeight(neighbor, index);
    }
    if (reverse == MISSING) {
      s1 += 2.0 * weight * weight;
      unmatched++;
    } else {
      s1 += (weight + reverse) * (weight + reverse);
      if (abs(weight - reverse) > tolerance[toleranceOffset]) {
        unmatched++;
      }
    }
  }
  // Cardinality counts active slots only (the shared neighbor definition).
  var activeCount = 0u;
  ${getGPUSpatialWeightsNeighborLoopWGSL('true', 'activeCount++;')}
  cardinality[cardinalityOffset + index] = activeCount;
  rowSums[rowSumsOffset + index] = sum;
  s1Partials[s1PartialsOffset + index] = s1;
  asymmetric[asymmetricOffset + index] = unmatched;`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-rows-final`,
          operation: OPERATION,
          variant: 'rows-final',
          bindings: [
            {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
            {name: 'rowSums', view: rowSums, type: 'f32', access: 'read'},
            {name: 'columnSums', view: columnSums, type: 'f32', access: 'read'},
            {name: 's2Partials', view: s2Partials, type: 'f32', access: 'read_write'},
            {name: 'cardinality', view: cardinality, type: 'u32', access: 'read'},
            {name: 'isolates', view: isolates, type: 'u32', access: 'read_write'}
          ],
          invocationCount: rows,
          // The active cardinality was written by the rows kernel.
          body: `let count = offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index];
  let total = rowSums[rowSumsOffset + index] + columnSums[columnSumsOffset + index];
  s2Partials[s2PartialsOffset + index] = total * total;
  // 0: has an active slot, 1: isolate that lists slots, 2: empty row (no slot at all).
  let isolate = cardinality[cardinalityOffset + index] == 0u;
  isolates[isolatesOffset + index] = select(0u, select(1u, 2u, count == 0u), isolate);`
        })
      ];
    };
    const rowNodes: GPUCommandNode<Parameters>[] = isChunked
      ? this.getChunkedRowNodes(graph, {
          dispatches,
          rows,
          tolerance,
          symmetryState,
          view,
          columnSums,
          rowSums,
          s1Partials,
          s2Partials,
          asymmetric,
          isolates,
          cardinality,
          chunkSlotCounts
        })
      : getWholeNodes();
    // The eight global reductions (S0, S1, S2, asymmetric slots, isolates, empty rows, minimum and
    // maximum cardinality) share one blocked pass over the per-row values and one combining workgroup,
    // instead of seven separate reduction chains. Fixed-shape trees keep the sums reproducible.
    const blockCount = Math.ceil(rows / SUMMARY_BLOCK_ROWS);
    const floatPartials = view('float-partials', 'float32', 3 * blockCount);
    const countPartials = view('count-partials', 'uint32', 5 * blockCount);
    const nodes: GPUCommandNode<Parameters>[] = [...runtime.nodes, ...rowNodes];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-blocks`,
        operation: OPERATION,
        variant: 'blocks',
        bindings: [
          {name: 'rowSums', view: rowSums, type: 'f32', access: 'read'},
          {name: 's1Partials', view: s1Partials, type: 'f32', access: 'read'},
          {name: 's2Partials', view: s2Partials, type: 'f32', access: 'read'},
          {name: 'asymmetric', view: asymmetric, type: 'u32', access: 'read'},
          {name: 'isolates', view: isolates, type: 'u32', access: 'read'},
          {name: 'cardinality', view: cardinality, type: 'u32', access: 'read'},
          {name: 'floatPartials', view: floatPartials, type: 'f32', access: 'read_write'},
          {name: 'countPartials', view: countPartials, type: 'u32', access: 'read_write'}
        ],
        workgroupSize: SUMMARY_WORKGROUP_SIZE,
        invocationCount: blockCount * SUMMARY_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${constants}
const BLOCK_ROWS: u32 = ${SUMMARY_BLOCK_ROWS}u;
const BLOCK_COUNT: u32 = ${blockCount}u;
const WORKGROUP_SIZE: u32 = ${SUMMARY_WORKGROUP_SIZE}u;
var<workgroup> partialS0: array<f32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialS1: array<f32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialS2: array<f32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialAsymmetric: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialIsolates: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialEmpty: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialMinimum: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialMaximum: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;`,
        // No early return: every invocation of a workgroup must reach the barriers.
        body: `let block = index / WORKGROUP_SIZE;
  let local = localInvocationIndex;
  let isInRange = index < INVOCATION_COUNT;
  var s0 = 0.0;
  var s1 = 0.0;
  var s2 = 0.0;
  var asymmetricCount = 0u;
  var isolateCount = 0u;
  var emptyCount = 0u;
  var minimum = 0xffffffffu;
  var maximum = 0u;
  if (isInRange) {
    let end = min((block + 1u) * BLOCK_ROWS, ROWS);
    for (var row = block * BLOCK_ROWS + local; row < end; row += WORKGROUP_SIZE) {
      s0 += rowSums[rowSumsOffset + row];
      s1 += s1Partials[s1PartialsOffset + row];
      s2 += s2Partials[s2PartialsOffset + row];
      asymmetricCount += asymmetric[asymmetricOffset + row];
      let isolateCode = isolates[isolatesOffset + row];
      isolateCount += select(0u, 1u, isolateCode != 0u);
      emptyCount += select(0u, 1u, isolateCode == 2u);
      let degree = cardinality[cardinalityOffset + row];
      minimum = min(minimum, degree);
      maximum = max(maximum, degree);
    }
  }
  partialS0[local] = s0;
  partialS1[local] = s1;
  partialS2[local] = s2;
  partialAsymmetric[local] = asymmetricCount;
  partialIsolates[local] = isolateCount;
  partialEmpty[local] = emptyCount;
  partialMinimum[local] = minimum;
  partialMaximum[local] = maximum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      partialS0[local] += partialS0[local + stride];
      partialS1[local] += partialS1[local + stride];
      partialS2[local] += partialS2[local + stride];
      partialAsymmetric[local] += partialAsymmetric[local + stride];
      partialIsolates[local] += partialIsolates[local + stride];
      partialEmpty[local] += partialEmpty[local + stride];
      partialMinimum[local] = min(partialMinimum[local], partialMinimum[local + stride]);
      partialMaximum[local] = max(partialMaximum[local], partialMaximum[local + stride]);
    }
    workgroupBarrier();
  }
  if (isInRange && local == 0u) {
    floatPartials[floatPartialsOffset + block] = partialS0[0];
    floatPartials[floatPartialsOffset + BLOCK_COUNT + block] = partialS1[0];
    floatPartials[floatPartialsOffset + 2u * BLOCK_COUNT + block] = partialS2[0];
    countPartials[countPartialsOffset + block] = partialAsymmetric[0];
    countPartials[countPartialsOffset + BLOCK_COUNT + block] = partialIsolates[0];
    countPartials[countPartialsOffset + 2u * BLOCK_COUNT + block] = partialMinimum[0];
    countPartials[countPartialsOffset + 3u * BLOCK_COUNT + block] = partialMaximum[0];
    countPartials[countPartialsOffset + 4u * BLOCK_COUNT + block] = partialEmpty[0];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-totals`,
        operation: OPERATION,
        variant: 'totals',
        bindings: [
          isChunked
            ? {name: 'chunkSlotCounts', view: chunkSlotCounts, type: 'u32', access: 'read'}
            : {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'floatPartials', view: floatPartials, type: 'f32', access: 'read'},
          {name: 'countPartials', view: countPartials, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'},
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        workgroupSize: SUMMARY_WORKGROUP_SIZE,
        invocationCount: SUMMARY_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${constants}
const BLOCK_COUNT: u32 = ${blockCount}u;
const WORKGROUP_SIZE: u32 = ${SUMMARY_WORKGROUP_SIZE}u;
var<workgroup> partialS0: array<f32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialS1: array<f32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialS2: array<f32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialAsymmetric: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialIsolates: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialEmpty: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialMinimum: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialMaximum: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;`,
        body: `let local = localInvocationIndex;
  var s0 = 0.0;
  var s1 = 0.0;
  var s2 = 0.0;
  var asymmetricCount = 0u;
  var isolateCount = 0u;
  var emptyCount = 0u;
  var minimum = 0xffffffffu;
  var maximum = 0u;
  for (var block = local; block < BLOCK_COUNT; block += WORKGROUP_SIZE) {
    s0 += floatPartials[floatPartialsOffset + block];
    s1 += floatPartials[floatPartialsOffset + BLOCK_COUNT + block];
    s2 += floatPartials[floatPartialsOffset + 2u * BLOCK_COUNT + block];
    asymmetricCount += countPartials[countPartialsOffset + block];
    isolateCount += countPartials[countPartialsOffset + BLOCK_COUNT + block];
    minimum = min(minimum, countPartials[countPartialsOffset + 2u * BLOCK_COUNT + block]);
    maximum = max(maximum, countPartials[countPartialsOffset + 3u * BLOCK_COUNT + block]);
    emptyCount += countPartials[countPartialsOffset + 4u * BLOCK_COUNT + block];
  }
  partialS0[local] = s0;
  partialS1[local] = s1;
  partialS2[local] = s2;
  partialAsymmetric[local] = asymmetricCount;
  partialIsolates[local] = isolateCount;
  partialEmpty[local] = emptyCount;
  partialMinimum[local] = minimum;
  partialMaximum[local] = maximum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      partialS0[local] += partialS0[local + stride];
      partialS1[local] += partialS1[local + stride];
      partialS2[local] += partialS2[local + stride];
      partialAsymmetric[local] += partialAsymmetric[local + stride];
      partialIsolates[local] += partialIsolates[local + stride];
      partialEmpty[local] += partialEmpty[local + stride];
      partialMinimum[local] = min(partialMinimum[local], partialMinimum[local + stride]);
      partialMaximum[local] = max(partialMaximum[local], partialMaximum[local + stride]);
    }
    workgroupBarrier();
  }
  if (local == 0u) {
    statistics[statisticsOffset] = partialS0[0];
    statistics[statisticsOffset + 1u] = 0.5 * partialS1[0];
    statistics[statisticsOffset + 2u] = partialS2[0];
    ${
      isChunked
        ? `var slotTotal = 0u;
    for (var chunk = 0u; chunk < ${dispatches.length}u; chunk++) {
      slotTotal += chunkSlotCounts[chunkSlotCountsOffset + chunk];
    }
    counts[countsOffset] = slotTotal;`
        : 'counts[countsOffset] = offsets[offsetsOffset + ROWS];'
    }
    counts[countsOffset + 1u] = partialAsymmetric[0];
    counts[countsOffset + 2u] = partialIsolates[0];
    counts[countsOffset + 3u] = partialMinimum[0];
    counts[countsOffset + 4u] = partialMaximum[0];
    counts[countsOffset + 5u] = partialEmpty[0];
  }`
      })
    );
    return nodes;
  }

  /**
   * Chunked form of the per-row nodes. Column sums continue one running sum across the chunks in
   * slot order (so they equal the whole-matrix sums); the reverse-slot lookup of the symmetry
   * statistics runs once per (row chunk, target chunk) pair, targets in ascending order, so every
   * row visits its slots in the same order as the whole matrix.
   */
  private getChunkedRowNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      dispatches: GPUSpatialWeightsChunkDispatch[];
      rows: number;
      tolerance: GraphDataView<'float32'>;
      symmetryState: GraphDataView<'uint32'>;
      view: <Format extends 'uint32' | 'float32'>(
        name: string,
        format: Format,
        length: number
      ) => GraphDataView<Format>;
      columnSums: GraphDataView<'float32'>;
      rowSums: GraphDataView<'float32'>;
      s1Partials: GraphDataView<'float32'>;
      s2Partials: GraphDataView<'float32'>;
      asymmetric: GraphDataView<'uint32'>;
      isolates: GraphDataView<'uint32'>;
      cardinality: GraphDataView<'uint32'>;
      chunkSlotCounts: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id} = this;
    const {dispatches, rows, tolerance, symmetryState, view} = context;
    const {columnSums, rowSums, s1Partials, s2Partials} = context;
    const {asymmetric, isolates, cardinality, chunkSlotCounts} = context;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-stage-tolerance`,
        operation: OPERATION,
        variant: 'stage-tolerance',
        invocationCount: 1,
        bindings: [
          {name: 'tolerance', view: tolerance, type: 'f32', access: 'read'},
          {name: 'symmetryState', view: symmetryState, type: 'u32', access: 'read_write'}
        ],
        body: 'symmetryState[symmetryStateOffset] = bitcast<u32>(tolerance[toleranceOffset]);'
      })
    ];
    // Weight w_row,column by binary search of the ascending neighbor IDs of a target chunk row.
    const findWeightWGSL = `
fn findWeight(row: u32, column: u32) -> f32 {
  var low = targetOffsets[targetOffsetsOffset + row];
  let end = targetOffsets[targetOffsetsOffset + row + 1u];
  var high = end;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (targetNeighbors[targetNeighborsOffset + middle] < column) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && targetNeighbors[targetNeighborsOffset + low] == column) {
    return targetWeights[targetWeightsOffset + low];
  }
  return MISSING;
}`;
    // Row finals read the column sums, which are complete only after every chunk has contributed.
    const finalNodes: GPUCommandNode<Parameters>[] = [];
    for (const dispatch of dispatches) {
      const chunkWeights = dispatch.weights;
      const chunkCapacity = chunkWeights.neighbors.length;
      const slotKeys = view(`slot-keys${dispatch.idSuffix}`, 'uint32', chunkCapacity);
      const slotIndices = view(`slot-indices${dispatch.idSuffix}`, 'uint32', chunkCapacity);
      const columnOffsets = view(`column-offsets${dispatch.idSuffix}`, 'uint32', rows + 1);
      const grouping = getSlotGroupingNodes<Parameters>(graph, {
        id: `${id}-columns${dispatch.idSuffix}`,
        operation: OPERATION,
        slotKeys,
        slotIndices,
        columns: rows,
        columnOffsets
      });
      const constants = `const ROWS: u32 = ${rows}u;
const CHUNK_ROWS: u32 = ${dispatch.rowCount}u;
const CHUNK_INDEX: u32 = ${dispatch.chunkIndex}u;
${dispatch.constantsWGSL}`;
      finalNodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-rows-final${dispatch.idSuffix}`,
          operation: OPERATION,
          variant: 'rows-final-chunk',
          bindings: [
            {name: 'offsets', view: chunkWeights.offsets, type: 'u32', access: 'read'},
            {name: 'rowSums', view: rowSums, type: 'f32', access: 'read'},
            {name: 'columnSums', view: columnSums, type: 'f32', access: 'read'},
            {name: 's2Partials', view: s2Partials, type: 'f32', access: 'read_write'},
            {name: 'cardinality', view: cardinality, type: 'u32', access: 'read'},
            {name: 'isolates', view: isolates, type: 'u32', access: 'read_write'}
          ],
          invocationCount: dispatch.rowCount,
          declarations: constants,
          body: `let globalRow = index + WEIGHTS_ROW_START;
  let count = offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index];
  let total = rowSums[rowSumsOffset + globalRow] + columnSums[columnSumsOffset + globalRow];
  s2Partials[s2PartialsOffset + globalRow] = total * total;
  let isolate = cardinality[cardinalityOffset + globalRow] == 0u;
  isolates[isolatesOffset + globalRow] = select(0u, select(1u, 2u, count == 0u), isolate);`
        })
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-slot-keys${dispatch.idSuffix}`,
          operation: OPERATION,
          variant: 'slot-keys',
          bindings: [
            {name: 'offsets', view: chunkWeights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: chunkWeights.neighbors, type: 'u32', access: 'read'},
            {name: 'slotKeys', view: slotKeys, type: 'u32', access: 'read_write'},
            {name: 'slotIndices', view: slotIndices, type: 'u32', access: 'read_write'}
          ],
          invocationCount: chunkCapacity,
          declarations: constants,
          body: `let neighbor = neighbors[neighborsOffset + index];
  let used = index < offsets[offsetsOffset + CHUNK_ROWS] && neighbor < ROWS;
  slotKeys[slotKeysOffset + index] = select(ROWS, neighbor, used);
  slotIndices[slotIndicesOffset + index] = index;`
        }),
        ...grouping.nodes,
        // The running sum continues from the previous chunks, so slots are added in global order.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-column-sums${dispatch.idSuffix}`,
          operation: OPERATION,
          variant: 'column-sums-chunk',
          bindings: [
            {name: 'columnOffsets', view: columnOffsets, type: 'u32', access: 'read'},
            {name: 'sortedSlots', view: grouping.sortedSlots, type: 'u32', access: 'read'},
            {name: 'weights', view: chunkWeights.weights, type: 'f32', access: 'read'},
            {name: 'columnSums', view: columnSums, type: 'f32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: `const CHUNK_INDEX: u32 = ${dispatch.chunkIndex}u;`,
          body: `var sum = 0.0;
  if (CHUNK_INDEX > 0u) {
    sum = columnSums[columnSumsOffset + index];
  }
  for (var position = columnOffsets[columnOffsetsOffset + index];
      position < columnOffsets[columnOffsetsOffset + index + 1u]; position++) {
    sum += weights[weightsOffset + sortedSlots[sortedSlotsOffset + position]];
  }
  columnSums[columnSumsOffset + index] = sum;`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-rows${dispatch.idSuffix}`,
          operation: OPERATION,
          variant: 'rows-chunk',
          bindings: [
            {name: 'offsets', view: chunkWeights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: chunkWeights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: chunkWeights.weights, type: 'f32', access: 'read'},
            {name: 'rowSums', view: rowSums, type: 'f32', access: 'read_write'},
            {name: 's1Partials', view: s1Partials, type: 'f32', access: 'read_write'},
            {name: 'asymmetric', view: asymmetric, type: 'u32', access: 'read_write'},
            {name: 'cardinality', view: cardinality, type: 'u32', access: 'read_write'}
          ],
          invocationCount: dispatch.rowCount,
          declarations: `${constants}
const CAPACITY: u32 = ${chunkCapacity}u;`,
          // The symmetry statistics are accumulated by the pair kernels that follow.
          body: `let globalRow = index + WEIGHTS_ROW_START;
  var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    sum += weights[weightsOffset + slot];
  }
  var activeCount = 0u;
  ${getGPUSpatialWeightsNeighborLoopWGSL('true', 'activeCount++;', 'globalRow', 'WEIGHTS_ROW_START')}
  cardinality[cardinalityOffset + globalRow] = activeCount;
  rowSums[rowSumsOffset + globalRow] = sum;
  s1Partials[s1PartialsOffset + globalRow] = 0.0;
  asymmetric[asymmetricOffset + globalRow] = 0u;`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-slot-count${dispatch.idSuffix}`,
          operation: OPERATION,
          variant: 'slot-count-chunk',
          bindings: [
            {name: 'offsets', view: chunkWeights.offsets, type: 'u32', access: 'read'},
            {name: 'chunkSlotCounts', view: chunkSlotCounts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          declarations: constants,
          body: 'chunkSlotCounts[chunkSlotCountsOffset + CHUNK_INDEX] = offsets[offsetsOffset + CHUNK_ROWS];'
        })
      );
    }
    nodes.push(...finalNodes);
    // Reverse-slot lookup per (row chunk, target chunk) pair. Targets ascend, so each row's slots
    // are visited in slot order; the last target also takes neighbors beyond the row count.
    for (const rowDispatch of dispatches) {
      for (const targetDispatch of dispatches) {
        const isLastTarget = targetDispatch.chunkIndex === dispatches.length - 1;
        const rowWeights = rowDispatch.weights;
        const targetWeights = targetDispatch.weights;
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-reverse-${rowDispatch.chunkIndex}-${targetDispatch.chunkIndex}`,
            operation: OPERATION,
            variant: 'reverse-chunk',
            bindings: [
              {name: 'offsets', view: rowWeights.offsets, type: 'u32', access: 'read'},
              {name: 'neighbors', view: rowWeights.neighbors, type: 'u32', access: 'read'},
              {name: 'weights', view: rowWeights.weights, type: 'f32', access: 'read'},
              {name: 'targetOffsets', view: targetWeights.offsets, type: 'u32', access: 'read'},
              {
                name: 'targetNeighbors',
                view: targetWeights.neighbors,
                type: 'u32',
                access: 'read'
              },
              {name: 'targetWeights', view: targetWeights.weights, type: 'f32', access: 'read'},
              {name: 's1Partials', view: s1Partials, type: 'f32', access: 'read_write'},
              {name: 'symmetryState', view: symmetryState, type: 'u32', access: 'read_write'}
            ],
            invocationCount: rowDispatch.rowCount,
            declarations: `const ROWS: u32 = ${rows}u;
const ROW_START: u32 = ${rowDispatch.rowStart}u;
const TARGET_START: u32 = ${targetDispatch.rowStart}u;
const TARGET_END: u32 = ${targetDispatch.rowStart + targetDispatch.rowCount}u;
const IS_LAST_TARGET: bool = ${isLastTarget};
const MISSING: f32 = -1.0;
${findWeightWGSL}`,
            body: `let globalRow = index + ROW_START;
  var s1 = s1Partials[s1PartialsOffset + globalRow];
  var unmatched = 0u;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < TARGET_START || (neighbor >= TARGET_END && !IS_LAST_TARGET)) {
      continue;
    }
    let weight = weights[weightsOffset + slot];
    var reverse = MISSING;
    if (neighbor < ROWS) {
      reverse = findWeight(neighbor - TARGET_START, globalRow);
    }
    if (reverse == MISSING) {
      s1 += 2.0 * weight * weight;
      unmatched++;
    } else {
      s1 += (weight + reverse) * (weight + reverse);
      if (abs(weight - reverse) > bitcast<f32>(symmetryState[symmetryStateOffset])) {
        unmatched++;
      }
    }
  }
  s1Partials[s1PartialsOffset + globalRow] = s1;
  symmetryState[symmetryStateOffset + 1u + globalRow] += unmatched;`
          })
        );
      }
    }
    return nodes;
  }
}
