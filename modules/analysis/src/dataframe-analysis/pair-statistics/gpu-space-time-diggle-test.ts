// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  GPU_CORE_MAX_STORAGE_BUFFER_BINDING_SIZE,
  GPU_MAXIMUM_COMMAND_NODE_COUNT,
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/index';
import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {PERMUTATION_RANDOM_WGSL} from '../../analysis-core/kernels/permutation-random';
import {getPermutationShuffleWGSL} from '../../analysis-core/kernels/permutation-shuffle';
import {resolveGPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
import {
  getPairHistogramNodes,
  getPairHistogramReadWGSL,
  PAIR_HISTOGRAM_MAXIMUM_ACCUMULATORS
} from './pair-histogram';
import {PAIR_STATISTICS_FLOAT_WGSL, validatePairStatisticsInputs} from './pair-statistics-grid';
import {
  createSpaceTimeKNodes,
  GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH,
  GPU_SPACE_TIME_K_MAXIMUM_CLASS_COUNT,
  validateSpaceTimeKRowCount,
  type GPUSpaceTimeKEdgeCorrection
} from './space-time-k-function-nodes';
import {
  GPU_SPACE_TIME_K_WEIGHT_CAP,
  SPACE_TIME_K_EDGE_WEIGHT_WGSL,
  SPACE_TIME_K_WEIGHT_SCALE
} from './space-time-k-function-wgsl';

const OPERATION = 'GPUSpaceTimeDiggleTest';

/**
 * Largest `permutationCount`. The statistics kernels loop over the permutations inside one
 * invocation per class, which is bounded by this capacity.
 */
export const GPU_SPACE_TIME_DIGGLE_MAXIMUM_PERMUTATIONS = 999;

/** Number of uint32 words in the `randomParameters` view: the seed as `[low, high]`. */
export const GPU_SPACE_TIME_DIGGLE_RANDOM_PARAMETER_LENGTH = 2;

/** Number of float32 words in the `summary` output. */
export const GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH = 11;

/** Indices of the `summary` output of {@link GPUSpaceTimeDiggleTest}. */
export const GPU_SPACE_TIME_DIGGLE_SUMMARY = {
  /** Observed statistic `U`, the sum of standardised `D` residuals over the usable classes. */
  observed: 0,
  /** Mean of the `P` permuted statistics. */
  permutationMean: 1,
  /** Sample variance (`P - 1` denominator) of the permuted statistics. */
  permutationVariance: 2,
  /** Permutations with a statistic `>=` the observed one. */
  greaterCount: 3,
  /** Permutations with a statistic `<=` the observed one. */
  lesserCount: 4,
  /** `(greaterCount + 1) / (P + 1)`: pseudo p-value for space-time interaction (clustering). */
  pseudoPGreater: 5,
  /** `(lesserCount + 1) / (P + 1)`: regularity. */
  pseudoPLesser: 6,
  /** `min(1, 2 min(pseudoPGreater, pseudoPLesser))`. */
  pseudoPTwoSided: 7,
  /** `(observed - permutationMean) / sqrt(permutationVariance)`, NaN when the variance is 0. */
  zScore: 8,
  /** Classes with a positive, finite permutation standard deviation (the classes in `U`). */
  usedClasses: 9,
  /** Included events `n`. */
  includedCount: 10
} as const;

/**
 * Command nodes of the observed pass (row filter, pair index, pair histogram, temporal marginal,
 * time sort, finish), counted with the allowance of the published sort and scan primitives, which
 * do not expose their own counts. A spec checks that the real count does not exceed the estimate.
 */
const OBSERVED_COMMAND_NODE_ALLOWANCE = 160;

/** Command nodes of one permutation batch, with the same allowance. */
const BATCH_COMMAND_NODE_ALLOWANCE = 40;

/** Caller-owned outputs of {@link GPUSpaceTimeDiggleTest}. */
export type GPUSpaceTimeDiggleTestOutput = {
  /** Observed `D(u_a, v_b)`, at least `distanceClassCount * timeClassCount` rows, distance-major like `GPUSpaceTimeKFunction`. */
  d: GraphDataView<'float32'>;
  /**
   * `permutationCount + 1` float32 rows: row 0 is the observed statistic `U`, rows `1..P` the
   * statistic of each permutation. NaN when no class is usable or fewer than two events are included.
   */
  statistics: GraphDataView<'float32'>;
  /** `GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH` float32 words, see {@link GPU_SPACE_TIME_DIGGLE_SUMMARY}. */
  summary: GraphDataView<'float32'>;
  /** Optional mean over the permutations of `D` per class, laid out like `d`. */
  permutationMean?: GraphDataView<'float32'>;
  /** Optional sample standard deviation over the permutations of `D` per class. */
  permutationStandardDeviation?: GraphDataView<'float32'>;
  /** Optional observed `K`, laid out like `d`. */
  k?: GraphDataView<'float32'>;
  /** Optional observed spatial marginal `K_s`, `distanceClassCount` rows. */
  spatialK?: GraphDataView<'float32'>;
  /** Optional observed temporal marginal `K_t`, `timeClassCount` rows. */
  temporalK?: GraphDataView<'float32'>;
  /** Optional one-row uint32 status of the observed pass: 0 valid, 1 undefined, 3 overflow. */
  status?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUSpaceTimeDiggleTest}.
 *
 * Per-frame (no recompile): the contents of `positions`, `times`, `mask`, `parameters` and
 * `randomParameters` (the seed). Compile-time: the row count, `gridSize`, both class counts,
 * `edgeCorrection`, `permutationCount` and which optional outputs are present.
 */
export type GPUSpaceTimeDiggleTestProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'space-time-diggle-test'`. */
  id?: string;
  /** Planar coordinate contract of `positions`. */
  spatialContext?: GPUSpatialContext;
  /** Optional declaration of `times`, as for `GPUSpaceTimeKFunction`. */
  temporalContext?: GPUTemporalContext;
  /** Packed planar event positions. */
  positions: GraphDataView<'float32x2'>;
  /** Event times, relative float32 in one unit. */
  times: GraphDataView<'float32'>;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /** Per-frame parameters written with `getGPUSpaceTimeKFunctionParameterValues` (bandwidths are ignored). */
  parameters: GraphDataView<'float32'>;
  /** Per-frame uint32 view of 2 words, `[seedLow, seedHigh]` from `getGPUSpaceTimeDiggleRandomParameterValues`. */
  randomParameters: GraphDataView<'uint32'>;
  /** Maximum `[columns, rows]` of the pair-search lattice; affects speed, not results. */
  gridSize: readonly [number, number];
  /** Number of distance classes `U`. */
  distanceClassCount: number;
  /** Number of time-lag classes `V`. */
  timeClassCount: number;
  /** `'isotropic'` (default) or `'none'`, as for `GPUSpaceTimeKFunction`. */
  edgeCorrection?: GPUSpaceTimeKEdgeCorrection;
  /** Number of time permutations `P`, 1 to {@link GPU_SPACE_TIME_DIGGLE_MAXIMUM_PERMUTATIONS}. Compile-time. */
  permutationCount: number;
  /** Caller-owned outputs. */
  output: GPUSpaceTimeDiggleTestOutput;
};

/**
 * Packs the seed of a {@link GPUSpaceTimeDiggleTest} into the 2-word uint32 `randomParameters`
 * layout `[seedLow, seedHigh]`.
 *
 * @param seed Random seed, an integer in `[0, 2^53)`. The result is a pure function of the seed
 * and the inputs.
 */
export function getGPUSpaceTimeDiggleRandomParameterValues(seed: number): Uint32Array {
  if (!Number.isSafeInteger(seed) || seed < 0) {
    throw new Error(
      'getGPUSpaceTimeDiggleRandomParameterValues seed must be a non-negative safe integer'
    );
  }
  return Uint32Array.from([seed % 2 ** 32, Math.floor(seed / 2 ** 32)]);
}

/**
 * Diggle, Chetwynd, Haggkvist and Morris (1995) test for space-time interaction: the Monte Carlo
 * test of `D(u, v) = K(u, v) - K_s(u) K_t(v)` against random relabelling of the event times over
 * the event locations (splancs `stmctest`, surveillance `stKtest`), on the indexed pair histogram.
 * `K`, `K_s`, `K_t` and the edge corrections are those of {@link GPUSpaceTimeKFunction}
 * (homogeneous estimator, `edgeCorrection` `'isotropic'` or `'none'`).
 *
 * Under a time permutation the locations and the multiset of times are fixed, so `K_s` and `K_t`
 * do not change and only `K` is recomputed. Each permutation is a swap-or-not shuffle of the
 * included events (the shuffle of `GPUKnoxTest`, on its own Philox stream), applied to the times.
 *
 * Statistic: with `m` and `s` the mean and sample standard deviation over the `P` permutations
 * of `D` in each class (classes with `s = 0` or non-finite values are skipped), the statistic is
 * `U = sum_classes (D - m) / s`, the sum of standardised residuals, for the observed pattern
 * and, with the same `m` and `s`, for each permutation. The pseudo p-value for interaction is
 * `(1 + #{U_p >= U}) / (P + 1)`. This is the definition of Diggle et al. (1995) as splancs
 * documents it; splancs is R-only (GPL), so its fixture is pending and the spec compares with a
 * binary64 reference that follows this definition. Whether splancs includes the observed pattern
 * in `m` and `s` was not verified.
 *
 * Permutations run in batches, `B = floor(1024 / (U V))` at a time (and as many as the time table
 * of `B * rows` floats fits one storage binding): a pair visited once accumulates the weight of
 * every permutation of the batch into its own `(permutation, distance, lag)` slot of the shared
 * deterministic pair histogram, so the result is exact and reproducible. The command node count
 * grows with `ceil(P / B)` and is exposed as `estimatedCommandNodeCount`.
 *
 * Not implemented: the inhomogeneous variant, the other stpp corrections, sequential stopping.
 */
export class GPUSpaceTimeDiggleTest implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpaceTimeDiggleTestProps;
  /** Resolved coordinate contract (planar/native). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved declaration of `times`, when the props gave one. */
  readonly temporalContext?: GPUTemporalContext;
  /** Resolved edge correction. */
  readonly edgeCorrection: GPUSpaceTimeKEdgeCorrection;
  /** Permutations per batch, before the row-count limit of the device is applied. */
  readonly batchSize: number;
  /** Upper bound of the number of command nodes `getCommandNodes` returns. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUSpaceTimeDiggleTestProps) {
    this.id = props.id ?? 'space-time-diggle-test';
    this.props = props;
    const {id} = this;
    const {output} = props;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'area'
    });
    this.temporalContext =
      props.temporalContext &&
      resolveGPUTemporalContext(id, props.temporalContext, {encodings: ['float32-relative']});
    this.edgeCorrection = props.edgeCorrection ?? 'isotropic';
    validateRequirement(
      id,
      ['none', 'isotropic'].includes(this.edgeCorrection),
      "edgeCorrection must be 'none' or 'isotropic'"
    );
    const rows = validatePairStatisticsInputs(id, {
      positions: props.positions,
      parameters: props.parameters,
      parameterLength: GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH,
      gridSize: props.gridSize,
      mask: props.mask
    });
    validatePackedView(props.times, ['float32'], `${id} times`);
    validateRequirement(
      id,
      props.times.length === rows,
      'times length must equal positions length'
    );
    validatePackedView(props.randomParameters, ['uint32'], `${id} randomParameters`);
    validateRequirement(
      id,
      props.randomParameters.length >= GPU_SPACE_TIME_DIGGLE_RANDOM_PARAMETER_LENGTH,
      `randomParameters must hold ${GPU_SPACE_TIME_DIGGLE_RANDOM_PARAMETER_LENGTH} uint32 words`
    );
    validateCapacity(
      id,
      'distanceClassCount',
      props.distanceClassCount,
      1,
      GPU_SPACE_TIME_K_MAXIMUM_CLASS_COUNT
    );
    validateCapacity(
      id,
      'timeClassCount',
      props.timeClassCount,
      1,
      GPU_SPACE_TIME_K_MAXIMUM_CLASS_COUNT
    );
    validateCapacity(
      id,
      'permutationCount',
      props.permutationCount,
      1,
      GPU_SPACE_TIME_DIGGLE_MAXIMUM_PERMUTATIONS
    );
    const bins = props.distanceClassCount * props.timeClassCount;
    // Two accumulator channels per slot; the observed pass needs the classes plus the marginal.
    validateRequirement(
      id,
      bins * 2 <= PAIR_HISTOGRAM_MAXIMUM_ACCUMULATORS &&
        (bins + props.distanceClassCount) * 3 <= PAIR_HISTOGRAM_MAXIMUM_ACCUMULATORS,
      `distanceClassCount * timeClassCount is too large for the pair histogram (${bins} classes)`
    );
    // The time table of one batch is `batchSize * rows` floats in one binding of at least the core limit.
    this.batchSize = Math.min(
      props.permutationCount,
      Math.floor(PAIR_HISTOGRAM_MAXIMUM_ACCUMULATORS / 2 / bins),
      Math.floor(GPU_CORE_MAX_STORAGE_BUFFER_BINDING_SIZE / 4 / rows)
    );
    validateCapacity(id, 'permutation batch size for this row count', this.batchSize, 1);
    const batchCount = Math.ceil(props.permutationCount / this.batchSize);
    this.estimatedCommandNodeCount =
      OBSERVED_COMMAND_NODE_ALLOWANCE + batchCount * BATCH_COMMAND_NODE_ALLOWANCE;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount, GPU_MAXIMUM_COMMAND_NODE_COUNT);
    const required: Record<string, number> = {
      d: bins,
      statistics: props.permutationCount + 1,
      summary: GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH,
      permutationMean: bins,
      permutationStandardDeviation: bins,
      k: bins,
      spatialK: props.distanceClassCount,
      temporalK: props.timeClassCount,
      status: 1
    };
    const outputViews: GraphDataView[] = [];
    for (const [name, view] of Object.entries(output)) {
      if (view) {
        validatePackedView(
          view,
          [name === 'status' ? 'uint32' : 'float32'],
          `${id} output.${name}`
        );
        validateRequirement(
          id,
          view.length >= required[name],
          `output.${name} must hold ${required[name]} rows`
        );
        outputViews.push(view);
      }
    }
    validateRequirement(
      id,
      new Set(outputViews.map(view => view.buffer)).size === outputViews.length,
      'outputs must use separate buffers'
    );
    validateGraphOutputsDisjointFromInputs(id, outputViews, [
      props.positions,
      props.times,
      props.mask,
      props.parameters,
      props.randomParameters
    ]);
  }

  /** Returns the observed pass, every permutation batch and the statistics kernels, in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output, distanceClassCount: U, timeClassCount: V, permutationCount} = props;
    const rows = props.positions.length;
    validateSpaceTimeKRowCount(id, graph.device, rows);
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.times,
      props.mask,
      props.parameters,
      props.randomParameters,
      ...Object.values(output)
    ]);
    const bins = U * V;
    const cellCount = props.gridSize[0] * props.gridSize[1];
    const isotropic = this.edgeCorrection === 'isotropic';
    // The time table of one batch holds `batchSize * rows` floats in one binding.
    const batchSize = Math.min(
      this.batchSize,
      Math.floor(getStorageBindingElementLimit(graph.device, 4) / rows)
    );
    validateCapacity(id, 'permutation batch size for this row count', batchSize, 1);
    // A device with smaller bindings than the core limit needs more batches than the estimate.
    validateCommandNodeBudget(
      id,
      OBSERVED_COMMAND_NODE_ALLOWANCE +
        Math.ceil(permutationCount / batchSize) * BATCH_COMMAND_NODE_ALLOWANCE
    );
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', Math.max(length, 1));
    const k = output.k ?? f32('k', bins);
    const spatialK = output.spatialK ?? f32('spatial-k', U);
    const temporalK = output.temporalK ?? f32('temporal-k', V);

    // Observed pass: K, K_s, K_t and D, with the pair index reused by the permutations.
    const observed = createSpaceTimeKNodes(graph, {
      id,
      operation: OPERATION,
      positions: props.positions,
      times: props.times,
      mask: props.mask,
      parameters: props.parameters,
      gridSize: props.gridSize,
      distanceClassCount: U,
      timeClassCount: V,
      edgeCorrection: this.edgeCorrection,
      k,
      spatialK,
      temporalK,
      d: output.d,
      status: output.status
    });
    const nodes: GPUCommandNode<Parameters>[] = [...observed.nodes];
    const {inputs} = observed;
    const constants = /* wgsl */ `
const DISTANCE_CLASS_COUNT: u32 = ${U}u;
const TIME_CLASS_COUNT: u32 = ${V}u;
const BINS: u32 = ${bins}u;
const PERMUTATION_COUNT: u32 = ${permutationCount}u;
const PERMUTATION_ROW_COUNT: u32 = ${rows}u;
const WEIGHT_SCALE: f32 = ${SPACE_TIME_K_WEIGHT_SCALE}.0;
const WEIGHT_CAP: f32 = ${GPU_SPACE_TIME_K_WEIGHT_CAP}.0;
const EDGE_ISOTROPIC: bool = ${isotropic};
`;

    // Rank of each included row among the included rows in row order: the domain of the shuffle,
    // so the permutations depend on the seed and the rows only, not on the pair lattice. A row is
    // included when the pair index holds it, which also needs a finite position inside `bounds`.
    const indexedRows = createTransientView(graph, `${id}-indexed-rows`, 'uint32', rows);
    const rowRanks = createTransientView(graph, `${id}-row-ranks`, 'uint32', rows);
    const rankRows = createTransientView(graph, `${id}-rank-rows`, 'uint32', rows);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-indexed-rows`,
        operation: OPERATION,
        variant: 'indexed-rows',
        bindings: [
          {name: 'cellKeys', view: inputs.cellKeys, type: 'u32', access: 'read'},
          {name: 'indexedRows', view: indexedRows, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `const CELL_COUNT: u32 = ${cellCount}u;`,
        body: `indexedRows[indexedRowsOffset + index] = select(0u, 1u, cellKeys[cellKeysOffset + index] < CELL_COUNT);`
      }),
      ...new GPUScan({
        id: `${id}-row-rank-scan`,
        input: indexedRows,
        output: rowRanks,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rank-rows`,
        operation: OPERATION,
        variant: 'rank-rows',
        bindings: [
          {name: 'indexedRows', view: indexedRows, type: 'u32', access: 'read'},
          {name: 'rowRanks', view: rowRanks, type: 'u32', access: 'read'},
          {name: 'rankRows', view: rankRows, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        body: `if (indexedRows[indexedRowsOffset + index] != 0u) {
    rankRows[rankRowsOffset + rowRanks[rowRanksOffset + index]] = index;
  }`
      })
    );

    const simulatedK = f32('simulated-k', permutationCount * bins);
    const groupsPerPermutation = Math.ceil(rows / 64);
    for (let batchStart = 0; batchStart < permutationCount; batchStart += batchSize) {
      const active = Math.min(batchSize, permutationCount - batchStart);
      const batchId = `${id}-batch-${batchStart}`;
      const batchConstants = `${constants}
const BATCH_START: u32 = ${batchStart}u;
const BATCH_SIZE: u32 = ${active}u;`;
      const permutedTimes = f32(`batch-${batchStart}-times`, active * rows);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${batchId}-times`,
          operation: OPERATION,
          variant: 'permuted-times',
          bindings: [
            {name: 'times', view: props.times, type: 'f32', access: 'read'},
            {name: 'indexedRows', view: indexedRows, type: 'u32', access: 'read'},
            {name: 'rowRanks', view: rowRanks, type: 'u32', access: 'read'},
            {name: 'rankRows', view: rankRows, type: 'u32', access: 'read'},
            {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
            {name: 'randomParameters', view: props.randomParameters, type: 'u32', access: 'read'},
            {name: 'permutedTimes', view: permutedTimes, type: 'f32', access: 'read_write'}
          ],
          invocationCount: active * groupsPerPermutation * 64,
          workgroupSize: 64,
          guardIndex: false,
          declarations: `${PERMUTATION_RANDOM_WGSL}
${getPermutationShuffleWGSL(PHILOX_PURPOSES.spaceTimeDiggleShuffle.id)}
${batchConstants}
const CELL_COUNT: u32 = ${cellCount}u;
const GROUPS_PER_PERMUTATION: u32 = ${groupsPerPermutation}u;
${PAIR_STATISTICS_FLOAT_WGSL}`,
          // The workgroup shares one permutation; every invocation reaches the barrier.
          body: `let workgroup = index / 64u;
  let localPermutation = workgroup / GROUPS_PER_PERMUTATION;
  let row = (workgroup - localPermutation * GROUPS_PER_PERMUTATION) * 64u + localInvocationIndex;
  let includedCount = cellOffsets[cellOffsetsOffset + CELL_COUNT];
  let running = localPermutation < BATCH_SIZE;
  if (running) {
    loadSwapOrNotRounds(vec2<u32>(randomParameters[randomParametersOffset], randomParameters[randomParametersOffset + 1u]),
      BATCH_START + localPermutation + 1u, includedCount, localInvocationIndex, 64u);
  }
  workgroupBarrier();
  if (running && row < PERMUTATION_ROW_COUNT) {
    var value = getQuietNaN(row);
    if (indexedRows[indexedRowsOffset + row] != 0u) {
      value = times[timesOffset + rankRows[rankRowsOffset + getSwapOrNotPermutationIndex(rowRanks[rowRanksOffset + row], includedCount)]];
    }
    permutedTimes[permutedTimesOffset + localPermutation * PERMUTATION_ROW_COUNT + row] = value;
  }`
        })
      );
      const histogram = getPairHistogramNodes<Parameters>(graph, {
        id: `${batchId}-histogram`,
        operation: OPERATION,
        positions: props.positions,
        parameters: props.parameters,
        gridSize: props.gridSize,
        sortedRows: inputs.sortedRows,
        cellOffsets: inputs.cellOffsets,
        slotCount: active * bins,
        channelCount: 2,
        pairOrder: 'ordered',
        extraBindings: [{name: 'permutedTimes', view: permutedTimes, type: 'f32', access: 'read'}],
        declarations: `${batchConstants}
${SPACE_TIME_K_EDGE_WEIGHT_WGSL}`,
        focusPrologue: `let timeStart = readParameter(7u);
    let timeEnd = readParameter(8u);
    let maximumDistance = readParameter(5u);
    let maximumTimeLag = readParameter(6u);
    let focusUsable = isFiniteFloat(maximumDistance) && maximumDistance > 0.0 &&
      isFiniteFloat(maximumTimeLag) && maximumTimeLag > 0.0;
    let left = x - lattice.minimumX;
    let bottom = y - lattice.minimumY;
    let right = lattice.maximumX - x;
    let top = lattice.maximumY - y;`,
        pairAction: `if (!focusUsable) { continue; }
            let scaledDistance = pairDistance / maximumDistance * f32(DISTANCE_CLASS_COUNT);
            if (!(scaledDistance <= f32(DISTANCE_CLASS_COUNT))) { continue; }
            let distanceClass = min(u32(max(ceil(scaledDistance) - 1.0, 0.0)), DISTANCE_CLASS_COUNT - 1u);
            let spatialWeight = select(1.0, getSpaceTimeSpatialWeight(left, bottom, right, top, pairDistance), EDGE_ISOTROPIC);
            for (var permutation = 0u; permutation < BATCH_SIZE; permutation++) {
              let timeBase = permutedTimesOffset + permutation * PERMUTATION_ROW_COUNT;
              let focusTime = permutedTimes[timeBase + focus];
              let timeLag = abs(permutedTimes[timeBase + neighbor] - focusTime);
              let scaledLag = timeLag / maximumTimeLag * f32(TIME_CLASS_COUNT);
              if (!(scaledLag <= f32(TIME_CLASS_COUNT))) { continue; }
              let lagClass = min(u32(max(ceil(scaledLag) - 1.0, 0.0)), TIME_CLASS_COUNT - 1u);
              let temporalWeight = select(1.0, getSpaceTimeTemporalWeight(timeLag, min(focusTime - timeStart, timeEnd - focusTime)), EDGE_ISOTROPIC);
              accumulateScaled((permutation * DISTANCE_CLASS_COUNT + distanceClass) * TIME_CLASS_COUNT + lagClass, 0u, 1u, spatialWeight * temporalWeight * WEIGHT_SCALE);
            }`
      });
      nodes.push(...histogram.nodes);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${batchId}-finish`,
          operation: OPERATION,
          variant: 'batch-finish',
          bindings: [
            {name: 'accumulators', view: histogram.accumulators, type: 'u32', access: 'read'},
            {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
            {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
            {name: 'simulatedK', view: simulatedK, type: 'f32', access: 'read_write'}
          ],
          invocationCount: active * bins,
          declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
${batchConstants}
const CELL_COUNT: u32 = ${cellCount}u;
const LARGE_UNIT: f32 = 4096.0;
${getPairHistogramReadWGSL('accumulators', 2, '', active * bins)}`,
          body: `let includedCount = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let area = (parameters[parametersOffset + 2u] - parameters[parametersOffset]) *
    (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]);
  let duration = parameters[parametersOffset + 8u] - parameters[parametersOffset + 7u];
  let defined = includedCount >= 2.0 && isFiniteFloat(area) && area > 0.0 &&
    isFiniteFloat(duration) && duration > 0.0 && !readPairOverflow();
  let localPermutation = index / BINS;
  let bin = index % BINS;
  let distanceIndex = bin / TIME_CLASS_COUNT;
  let lagIndex = bin % TIME_CLASS_COUNT;
  var sum = 0.0;
  for (var firstIndex = 0u; firstIndex <= distanceIndex; firstIndex++) {
    for (var secondIndex = 0u; secondIndex <= lagIndex; secondIndex++) {
      let slot = localPermutation * BINS + firstIndex * TIME_CLASS_COUNT + secondIndex;
      sum += readPairAccumulator(slot, 0u) + readPairAccumulator(slot, 1u) * LARGE_UNIT;
    }
  }
  simulatedK[simulatedKOffset + (BATCH_START + localPermutation) * BINS + bin] =
    select(getQuietNaN(index), area * duration / (includedCount * (includedCount - 1.0)) * sum / WEIGHT_SCALE, defined);`
        })
      );
    }

    // Mean and standard deviation of D over the permutations, per class.
    const permutationMean = output.permutationMean ?? f32('permutation-mean', bins);
    const permutationStandardDeviation =
      output.permutationStandardDeviation ?? f32('permutation-standard-deviation', bins);
    const statisticsConstants = `${PAIR_STATISTICS_FLOAT_WGSL}
${constants}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-permutation-moments`,
        operation: OPERATION,
        variant: 'permutation-moments',
        bindings: [
          {name: 'simulatedK', view: simulatedK, type: 'f32', access: 'read'},
          {name: 'spatialK', view: spatialK, type: 'f32', access: 'read'},
          {name: 'temporalK', view: temporalK, type: 'f32', access: 'read'},
          {name: 'permutationMean', view: permutationMean, type: 'f32', access: 'read_write'},
          {
            name: 'permutationStandardDeviation',
            view: permutationStandardDeviation,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: bins,
        declarations: statisticsConstants,
        body: `// Capacity-bounded loops over the permutations (at most ${GPU_SPACE_TIME_DIGGLE_MAXIMUM_PERMUTATIONS}).
  var sum = 0.0;
  for (var permutation = 0u; permutation < PERMUTATION_COUNT; permutation++) {
    sum += simulatedK[simulatedKOffset + permutation * BINS + index];
  }
  let mean = sum / f32(PERMUTATION_COUNT);
  var squares = 0.0;
  for (var permutation = 0u; permutation < PERMUTATION_COUNT; permutation++) {
    let deviation = simulatedK[simulatedKOffset + permutation * BINS + index] - mean;
    squares += deviation * deviation;
  }
  let product = spatialK[spatialKOffset + index / TIME_CLASS_COUNT] * temporalK[temporalKOffset + index % TIME_CLASS_COUNT];
  permutationMean[permutationMeanOffset + index] = mean - product;
  permutationStandardDeviation[permutationStandardDeviationOffset + index] =
    select(getQuietNaN(index), sqrt(squares / f32(PERMUTATION_COUNT - 1u)), PERMUTATION_COUNT > 1u);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-statistics`,
        operation: OPERATION,
        variant: 'statistics',
        bindings: [
          {name: 'd', view: output.d, type: 'f32', access: 'read'},
          {name: 'simulatedK', view: simulatedK, type: 'f32', access: 'read'},
          {name: 'spatialK', view: spatialK, type: 'f32', access: 'read'},
          {name: 'temporalK', view: temporalK, type: 'f32', access: 'read'},
          {name: 'permutationMean', view: permutationMean, type: 'f32', access: 'read'},
          {
            name: 'permutationStandardDeviation',
            view: permutationStandardDeviation,
            type: 'f32',
            access: 'read'
          },
          {name: 'statistics', view: output.statistics, type: 'f32', access: 'read_write'}
        ],
        invocationCount: permutationCount + 1,
        declarations: statisticsConstants,
        body: `// Row 0 is the observed pattern, row p the permutation p - 1.
  var total = 0.0;
  var used = 0u;
  for (var bin = 0u; bin < BINS; bin++) {
    let product = spatialK[spatialKOffset + bin / TIME_CLASS_COUNT] * temporalK[temporalKOffset + bin % TIME_CLASS_COUNT];
    var value = d[dOffset + bin];
    if (index > 0u) {
      value = simulatedK[simulatedKOffset + (index - 1u) * BINS + bin] - product;
    }
    let deviation = permutationStandardDeviation[permutationStandardDeviationOffset + bin];
    let residual = (value - permutationMean[permutationMeanOffset + bin]) / deviation;
    if (isFiniteFloat(residual) && deviation > 0.0) {
      total += residual;
      used++;
    }
  }
  statistics[statisticsOffset + index] = select(getQuietNaN(index), total, used > 0u);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'statistics', view: output.statistics, type: 'f32', access: 'read'},
          {
            name: 'permutationStandardDeviation',
            view: permutationStandardDeviation,
            type: 'f32',
            access: 'read'
          },
          {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
          {name: 'summary', view: output.summary, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `${statisticsConstants}
const CELL_COUNT: u32 = ${cellCount}u;`,
        body: `let observed = statistics[statisticsOffset];
  let defined = isFiniteFloat(observed);
  var greater = 0u;
  var lesser = 0u;
  var sum = 0.0;
  for (var permutation = 1u; permutation <= PERMUTATION_COUNT; permutation++) {
    let value = statistics[statisticsOffset + permutation];
    if (value >= observed) { greater++; }
    if (value <= observed) { lesser++; }
    sum += value;
  }
  let count = f32(PERMUTATION_COUNT);
  let mean = sum / count;
  var squares = 0.0;
  for (var permutation = 1u; permutation <= PERMUTATION_COUNT; permutation++) {
    let deviation = statistics[statisticsOffset + permutation] - mean;
    squares += deviation * deviation;
  }
  let variance = select(0.0, squares / (count - 1.0), PERMUTATION_COUNT > 1u);
  var usedClasses = 0u;
  for (var bin = 0u; bin < BINS; bin++) {
    let deviation = permutationStandardDeviation[permutationStandardDeviationOffset + bin];
    if (isFiniteFloat(deviation) && deviation > 0.0) { usedClasses++; }
  }
  let nan = getQuietNaN(0u);
  let pGreater = f32(greater + 1u) / (count + 1.0);
  let pLesser = f32(lesser + 1u) / (count + 1.0);
  summary[summaryOffset] = observed;
  summary[summaryOffset + 1u] = select(nan, mean, defined);
  summary[summaryOffset + 2u] = select(nan, variance, defined);
  summary[summaryOffset + 3u] = select(nan, f32(greater), defined);
  summary[summaryOffset + 4u] = select(nan, f32(lesser), defined);
  summary[summaryOffset + 5u] = select(nan, pGreater, defined);
  summary[summaryOffset + 6u] = select(nan, pLesser, defined);
  summary[summaryOffset + 7u] = select(nan, min(1.0, 2.0 * min(pGreater, pLesser)), defined);
  summary[summaryOffset + 8u] = select(nan, (observed - mean) / sqrt(variance), defined && variance > 0.0);
  summary[summaryOffset + 9u] = f32(usedClasses);
  summary[summaryOffset + 10u] = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);`
      })
    );
    return nodes;
  }
}
