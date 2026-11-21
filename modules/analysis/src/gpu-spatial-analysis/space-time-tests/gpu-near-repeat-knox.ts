// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateTrajectoryBindings,
  validateTrajectoryOutputRanges
} from '../trajectory-analysis/trajectory-binding-validation';
import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {createSpatialWeightsSuppressionNodes} from '../spatial-weights/spatial-weights-suppression';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotSpaceTimeTemporalContext} from './space-time-construction';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import type {GPUInferenceMetadata, GPUTemporalContext} from '../../analysis-core/contracts/index';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateRequirement,
  validateGraphPorts,
  validateViewLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateGPUStorageBindingByteLength,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {
  createSpaceTimePairMaskNode,
  getSpaceTimeBlocks,
  getSpaceTimeCommonWGSL,
  getSpaceTimeShuffleWGSL,
  SPACE_TIME_MAXIMUM_PERMUTATIONS,
  SPACE_TIME_MAXIMUM_ROWS,
  SPACE_TIME_WORKGROUP_SIZE
} from './space-time-kernels';
import {
  GPU_NEAR_REPEAT_MAXIMUM_CELLS,
  getSpaceTimeBandWGSL,
  resolveGPUSpaceTimeTemporalContext,
  SPACE_TIME_NO_BAND_WGSL,
  validateSpaceTimeBandEdges
} from './space-time-bands';
import {
  getGPUSpaceTimeInferenceMetadata,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  type GPUSpaceTimeSettings
} from './space-time-parameters';

const OPERATION = 'GPUNearRepeatKnox';

/** Number of float32 words per table cell in the `summary` output of {@link GPUNearRepeatKnox}. */
export const GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH = 8;

/**
 * Word indices within one cell of the `summary` output of {@link GPUNearRepeatKnox}; cell `c`
 * starts at word `c * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH`.
 */
export const GPU_NEAR_REPEAT_CELL_SUMMARY = {
  /** Observed number of pairs in the cell. */
  observed: 0,
  /** Mean of the cell count over the permutations. */
  permutationMean: 1,
  /** Sample variance (`P - 1` denominator) of the cell count over the permutations. */
  permutationVariance: 2,
  /** Permutations whose cell count is `>=` the observed. */
  greaterCount: 3,
  /** Permutations whose cell count is `<=` the observed. */
  lesserCount: 4,
  /** `(greaterCount + 1) / (P + 1)`: the near-repeat p-value of the cell (excess of pairs). */
  pseudoPGreater: 5,
  /** `(lesserCount + 1) / (P + 1)`. */
  pseudoPLesser: 6,
  /** Knox ratio: `observed / permutationMean`, NaN when the mean is 0. */
  knoxRatio: 7
} as const;

/** Caller-owned outputs of {@link GPUNearRepeatKnox}. */
export type GPUNearRepeatKnoxOutput = {
  /**
   * `(maximumPermutations + 1) * cells` uint32 pair counts, `counts[slot * cells + cell]` with
   * `cell = distanceBand * timeBands + timeBand`. Slot 0 is the observed table, slots `1..P` the
   * tables under permuted times (slots past `P` are 0).
   */
  counts: GraphDataView<'uint32'>;
  /**
   * `cells * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH` float32 words, per cell as in
   * {@link GPU_NEAR_REPEAT_CELL_SUMMARY}.
   */
  summary: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUNearRepeatKnox}.
 *
 * Per-frame (no recompile): the contents of `times`, `pairs` and `parameters` (seed, permutation
 * count). Compile-time: the band edges, the row count, `maximumPermutations` and the pair
 * capacity.
 */
export type GPUNearRepeatKnoxProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'near-repeat-knox'`. */
  id?: string;
  /**
   * Pairs close in space with their distances, as a self-join {@link GPUSpatialWeights} with
   * `distances`, the same pair semantics as `GPUKnoxTest`: each unordered pair that appears in
   * either direction is counted once, duplicates and self entries are ignored. List every pair up
   * to the last distance edge, for example a radius-mode `GPUNeighborSearch` with that radius;
   * pairs farther than the last edge are not counted. `distanceEdges` are in the length unit of
   * `distances` (the `spatialContext` of the pairs, when present).
   */
  pairs: GPUSpatialWeights;
  /**
   * Event time per row, finite float32, measured from a nearby origin in the unit of
   * `timeEdges`. Build it with `getGPUSpaceTimeGuardedRelativeTimes`, which refuses a span whose
   * float32 spacing is coarse against the narrowest time band.
   */
  times: GraphDataView<'float32'>;
  /**
   * Optional declaration of `times`: encoding `float32-relative`, with the unit of `timeEdges`
   * and the epoch the relative times count from (`getGPUSpaceTimeGuardedRelativeTimes` returns
   * it). Checked at construction and exposed as {@link GPUNearRepeatKnox.temporalContext};
   * nothing is converted.
   */
  temporalContext?: GPUTemporalContext;
  /**
   * Per-frame parameters, `GPU_SPACE_TIME_PARAMETER_LENGTH` uint32 words written with
   * `getGPUSpaceTimeParameterValues` (seed and permutation count; `timeThreshold` is ignored).
   */
  parameters: GraphDataView<'uint32'>;
  /**
   * Compile-time distance band edges, strictly ascending and non-negative, 2 to 17 edges. Band `k`
   * is `(edges[k], edges[k + 1]]` and band 0 also holds `edges[0]`. Distances are compared as
   * float32.
   */
  distanceEdges: readonly number[];
  /**
   * Compile-time time band edges in the unit of `times`, same convention applied to
   * `|t_i - t_j|` (the `<=` of `GPUKnoxTest`).
   */
  timeEdges: readonly number[];
  /** Compile-time upper bound on the per-frame permutation count, 1 to 65536. */
  maximumPermutations: number;
  /** Caller-owned outputs. */
  output: GPUNearRepeatKnoxOutput;
};

/**
 * Near-repeat Knox table: for every cell of distance band by time band, the number of event pairs
 * in the cell and a Monte Carlo p-value per cell. The pair counts are exact integers.
 *
 * Definition (the NearRepeat table of Ratcliffe and Rengert, as implemented by the NearRepeat R
 * package, GPL, used here as a definition only): count the event pairs by band of distance and
 * band of time difference, then compare each cell with the same table under random permutations of
 * the event times over the event locations. pointpats has no band table; with `<=` thresholds a
 * cell is an inclusion-exclusion of `pointpats.spacetime.Knox` runs, which is how the checked-in
 * pointpats fixture is made. The reference distribution is the one of `GPUKnoxTest` (the same
 * swap-or-not stream, so a given seed gives the same permutations): `P` permutations of the times
 * leave the pair list fixed, and the p-value of a cell is `(1 + #{count_perm >= count}) / (P + 1)`.
 *
 * Each permutation is one workgroup: its invocations walk the rows and add into a workgroup
 * histogram with integer atomics, so counts are exact and deterministic. Besag-Clifford stopping
 * is not offered: the cells stop at different permutations, so no single stopping index exists.
 *
 * Reported per cell: the Knox ratio against the permutation mean (the NearRepeat median ratio is
 * not computed). The p-values are 'greater' tests, one per cell, with no multiple-testing
 * correction; {@link GPUNearRepeatKnox.getInferenceMetadata} says so.
 *
 * Incomplete pairs: when `pairs.status.overflow` or `pairs.status.candidateOverflow` is nonzero
 * some pairs were dropped, so for that frame the owned `counts` words are 0 and the owned
 * `summary` words are quiet NaN, as in `GPUKnoxTest`. Spare output words are untouched.
 * Clearing the flags restores the ordinary tables next frame.
 */
export class GPUNearRepeatKnox implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNearRepeatKnoxProps;
  /** Number of events. */
  readonly rows: number;
  /** Distance bands. */
  readonly distanceBands: number;
  /** Time bands. */
  readonly timeBands: number;
  /** `distanceBands * timeBands`. */
  readonly cells: number;
  /** The validated context of `times`, when one was given. */
  readonly temporalContext?: GPUTemporalContext;

  constructor(props: GPUNearRepeatKnoxProps) {
    const captured = snapshotConstructionFields(props, [
      'id',
      'pairs',
      'times',
      'temporalContext',
      'parameters',
      'distanceEdges',
      'timeEdges',
      'maximumPermutations',
      'output'
    ]);
    const capturedId = captured.id ?? 'near-repeat-knox';
    validateRequirement(
      capturedId,
      Boolean(captured.pairs && typeof captured.pairs === 'object'),
      `pairs must be a port map; received ${String(captured.pairs)}`
    );
    validateRequirement(
      capturedId,
      Boolean(captured.output && typeof captured.output === 'object'),
      `output must be a port map; received ${String(captured.output)}`
    );
    validateRequirement(
      capturedId,
      Array.isArray(captured.distanceEdges),
      `distanceEdges must be an array of band edges; received ${String(captured.distanceEdges)}`
    );
    validateRequirement(
      capturedId,
      Array.isArray(captured.timeEdges),
      `timeEdges must be an array of band edges; received ${String(captured.timeEdges)}`
    );
    // This contributor accepts a packed CSR, not the chunked weights union. Capture only its
    // declared fields so unrelated caller properties cannot select different loss semantics.
    const pairs = snapshotConstructionFields(captured.pairs, [
      'offsets',
      'neighbors',
      'weights',
      'distances',
      'status',
      'spatialContext',
      'information'
    ]);
    const information =
      pairs.information &&
      snapshotConstructionFields(pairs.information, [
        'rowCount',
        'transform',
        'symmetric',
        'hasSelfWeights',
        'provenance'
      ]);
    const provenance = information?.provenance;
    props = Object.freeze({
      ...captured,
      pairs: Object.freeze({
        ...pairs,
        spatialContext:
          pairs.spatialContext && snapshotConstructionSpatialContext(pairs.spatialContext),
        information:
          information &&
          Object.freeze({
            ...information,
            provenance:
              provenance &&
              snapshotConstructionFields(provenance, [
                'source',
                'criterion',
                'k',
                'radius',
                'order'
              ])
          }),
        ...(pairs.status
          ? {
              status: snapshotConstructionFields(pairs.status, [
                'count',
                'requiredCount',
                'overflow',
                'candidateOverflow',
                'invalidCount',
                'uncertainCount',
                'iterationCount',
                'converged',
                'iterationLimitReached',
                'approximation'
              ])
            }
          : {})
      }),
      temporalContext:
        captured.temporalContext && snapshotSpaceTimeTemporalContext(captured.temporalContext),
      distanceEdges: Object.freeze([...captured.distanceEdges]),
      timeEdges: Object.freeze([...captured.timeEdges]),
      output: snapshotConstructionFields(captured.output, ['counts', 'summary'])
    });
    const id = props.id ?? 'near-repeat-knox';
    this.id = id;
    this.props = props;
    validateGraphPorts(id, props, {
      'pairs.distances': {required: true, viewType: 'data', formats: ['float32']},
      times: {required: true, viewType: 'data', formats: ['float32']},
      parameters: {required: true, viewType: 'data', formats: ['uint32']},
      'output.counts': {required: true, viewType: 'data', formats: ['uint32']},
      'output.summary': {required: true, viewType: 'data', formats: ['float32']}
    });
    this.rows = validateGPUSpatialWeights(id, props.pairs, 'pairs');
    validateSinglePackedView(props.times, ['float32'], `${id} times`);
    validateSinglePackedView(props.parameters, ['uint32'], `${id} parameters`);
    validateSinglePackedView(props.output.counts, ['uint32'], `${id} output.counts`);
    validateSinglePackedView(props.output.summary, ['float32'], `${id} output.summary`);
    this.temporalContext = resolveGPUSpaceTimeTemporalContext(id, props.temporalContext);
    validateCapacity(id, 'pairs rowCount', this.rows, 2, SPACE_TIME_MAXIMUM_ROWS);
    validateViewLength(id, 'times', props.times, this.rows);
    validateCapacity(
      id,
      'parameters.length',
      props.parameters.length,
      GPU_SPACE_TIME_PARAMETER_LENGTH
    );
    validateSpaceTimeBandEdges(id, 'distanceEdges', props.distanceEdges);
    validateSpaceTimeBandEdges(id, 'timeEdges', props.timeEdges);
    this.distanceBands = props.distanceEdges.length - 1;
    this.timeBands = props.timeEdges.length - 1;
    this.cells = this.distanceBands * this.timeBands;
    validateCapacity(id, 'cells', this.cells, 1, GPU_NEAR_REPEAT_MAXIMUM_CELLS);
    validateCapacity(
      id,
      'maximumPermutations',
      props.maximumPermutations,
      1,
      SPACE_TIME_MAXIMUM_PERMUTATIONS
    );
    validateCapacity(
      id,
      'output.counts.length',
      props.output.counts.length,
      (props.maximumPermutations + 1) * this.cells
    );
    validateCapacity(
      id,
      'output.summary.length',
      props.output.summary.length,
      this.cells * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH
    );
    validateTrajectoryOutputRanges(this.id, this.props.output);
    validateGraphOutputsDisjointFromInputs(
      id,
      {'output.counts': props.output.counts, 'output.summary': props.output.summary},
      {
        times: props.times,
        parameters: props.parameters,
        'pairs.offsets': props.pairs.offsets,
        'pairs.neighbors': props.pairs.neighbors,
        'pairs.weights': props.pairs.weights,
        'pairs.distances': props.pairs.distances,
        'pairs.status.overflow': props.pairs.status?.overflow,
        'pairs.status.candidateOverflow': props.pairs.status?.candidateOverflow
      }
    );
    validateRequirement(
      id,
      props.output.counts.buffer !== props.output.summary.buffer,
      'outputs must use separate buffers'
    );
  }

  /**
   * CPU-side description of how the cell p-values are produced: alternative `'greater'`, Monte
   * Carlo, one test per cell and no multiple-testing correction.
   *
   * @param settings The per-frame settings packed into `parameters`.
   */
  getInferenceMetadata(settings: GPUSpaceTimeSettings): GPUInferenceMetadata {
    return {
      ...getGPUSpaceTimeInferenceMetadata('knox', settings, this.props.maximumPermutations),
      testedRowsDefinition:
        'one test per cell of distance band by time band over all unordered event pairs listed in the spatial pair list'
    };
  }

  /**
   * Returns the nodes in order: `pair-mask`, `pair-band`, `histogram`, `summary`, then one
   * `incomplete` node per pair loss flag.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateTrajectoryBindings(graph, this.id, this.props);
    const {id, props, rows, cells, timeBands} = this;
    const {pairs, times, parameters, maximumPermutations} = props;
    const {counts, summary} = props.output;
    const distances = pairs.distances!;
    validateGraphViewsBelongToGraph(id, graph, [
      pairs.offsets,
      pairs.neighbors,
      pairs.weights,
      distances,
      times,
      parameters,
      counts,
      summary
    ]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'pair-band',
      pairs.neighbors.length * 4,
      'reduce the pair capacity'
    );
    const slotCount = maximumPermutations + 1;
    const [blocks, rowsPerBlock] = getSpaceTimeBlocks(rows, SPACE_TIME_WORKGROUP_SIZE);
    const common = getSpaceTimeCommonWGSL(rows, maximumPermutations, blocks, rowsPerBlock);
    const pairMask = createTransientView(
      graph,
      `${id}-pair-mask`,
      'uint32',
      pairs.neighbors.length
    );
    const pairBand = createTransientView(
      graph,
      `${id}-pair-band`,
      'uint32',
      pairs.neighbors.length
    );
    const bandWGSL = `${getSpaceTimeBandWGSL('Distance', props.distanceEdges)}${getSpaceTimeBandWGSL('Time', props.timeEdges)}`;
    const nodes = [
      createSpaceTimePairMaskNode<Parameters>(graph, {
        id,
        operation: OPERATION,
        rows,
        offsets: pairs.offsets,
        neighbors: pairs.neighbors,
        pairMask
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pair-band`,
        operation: OPERATION,
        variant: 'pair-band',
        bindings: [
          {name: 'pairMask', view: pairMask, type: 'u32', access: 'read'},
          {name: 'distances', view: distances, type: 'f32', access: 'read'},
          {name: 'pairBand', view: pairBand, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairs.neighbors.length,
        declarations: bandWGSL,
        // The distance band of each representative entry, fixed across the permutations.
        body: `pairBand[pairBandOffset + index] = select(
    ${SPACE_TIME_NO_BAND_WGSL},
    getDistanceBand(distances[distancesOffset + index]),
    pairMask[pairMaskOffset + index] != 0u
  );`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-histogram`,
        operation: OPERATION,
        variant: 'histogram',
        bindings: [
          {name: 'offsets', view: pairs.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: pairs.neighbors, type: 'u32', access: 'read'},
          {name: 'pairBand', view: pairBand, type: 'u32', access: 'read'},
          {name: 'times', view: times, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        workgroupSize: SPACE_TIME_WORKGROUP_SIZE,
        invocationCount: slotCount * SPACE_TIME_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${common}
${getSpaceTimeShuffleWGSL(blocks, PHILOX_PURPOSES.knoxTestShuffle.id)}
${bandWGSL}
const CELLS: u32 = ${cells}u;
const TIME_BANDS: u32 = ${timeBands}u;
const NO_BAND: u32 = ${SPACE_TIME_NO_BAND_WGSL};
var<workgroup> histogram: array<atomic<u32>, ${cells}>;`,
        // One workgroup is one permutation (slot 0 is the identity). Its invocations walk the rows
        // interleaved and add into the workgroup histogram; the barriers are reached by every
        // invocation of the workgroup, the slot being uniform across it.
        body: `let slot = index / SPACE_TIME_WORKGROUP_SIZE;
  let runsSlot = slot <= readPermutationCount();
  if (slot > 0u && runsSlot) {
    loadSwapOrNotRounds(readSeedKey(), slot, ROWS, localInvocationIndex, SPACE_TIME_WORKGROUP_SIZE);
  }
  for (var cell = localInvocationIndex; cell < CELLS; cell += SPACE_TIME_WORKGROUP_SIZE) {
    atomicStore(&histogram[cell], 0u);
  }
  workgroupBarrier();
  if (runsSlot) {
    for (var row = localInvocationIndex; row < ROWS; row += SPACE_TIME_WORKGROUP_SIZE) {
      let time = times[timesOffset + getPermutedRow(row, slot)];
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        let distanceBand = pairBand[pairBandOffset + entry];
        if (distanceBand != NO_BAND) {
          let otherTime = times[timesOffset + getPermutedRow(neighbors[neighborsOffset + entry], slot)];
          let timeBand = getTimeBand(abs(time - otherTime));
          if (timeBand != NO_BAND) {
            atomicAdd(&histogram[distanceBand * TIME_BANDS + timeBand], 1u);
          }
        }
      }
    }
  }
  workgroupBarrier();
  for (var cell = localInvocationIndex; cell < CELLS; cell += SPACE_TIME_WORKGROUP_SIZE) {
    counts[countsOffset + slot * CELLS + cell] = select(0u, atomicLoad(&histogram[cell]), runsSlot);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'summary', view: summary, type: 'f32', access: 'read_write'}
        ],
        invocationCount: cells,
        declarations: `${common}
const CELLS: u32 = ${cells}u;`,
        // Deviations from the observed count keep the sums exact in float32.
        body: `let permutations = readPermutationCount();
  let observed = counts[countsOffset + index];
  var greater = 0u;
  var lesser = 0u;
  var sumDeviation = 0.0;
  for (var slot = 1u; slot <= permutations; slot++) {
    let value = counts[countsOffset + slot * CELLS + index];
    if (value >= observed) { greater++; }
    if (value <= observed) { lesser++; }
    sumDeviation += f32(i32(value) - i32(observed));
  }
  let permutationCount = f32(permutations);
  let meanDeviation = sumDeviation / permutationCount;
  var sumSquares = 0.0;
  for (var slot = 1u; slot <= permutations; slot++) {
    let deviation = f32(i32(counts[countsOffset + slot * CELLS + index]) - i32(observed)) - meanDeviation;
    sumSquares += deviation * deviation;
  }
  let variance = select(0.0, sumSquares / (permutationCount - 1.0), permutations > 1u);
  let mean = f32(observed) + meanDeviation;
  let cellOffset = summaryOffset + index * ${GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH}u;
  summary[cellOffset + ${GPU_NEAR_REPEAT_CELL_SUMMARY.observed}u] = f32(observed);
  summary[cellOffset + ${GPU_NEAR_REPEAT_CELL_SUMMARY.permutationMean}u] = mean;
  summary[cellOffset + ${GPU_NEAR_REPEAT_CELL_SUMMARY.permutationVariance}u] = variance;
  summary[cellOffset + ${GPU_NEAR_REPEAT_CELL_SUMMARY.greaterCount}u] = f32(greater);
  summary[cellOffset + ${GPU_NEAR_REPEAT_CELL_SUMMARY.lesserCount}u] = f32(lesser);
  summary[cellOffset + ${GPU_NEAR_REPEAT_CELL_SUMMARY.pseudoPGreater}u] = f32(greater + 1u) / (permutationCount + 1.0);
  summary[cellOffset + ${GPU_NEAR_REPEAT_CELL_SUMMARY.pseudoPLesser}u] = f32(lesser + 1u) / (permutationCount + 1.0);
  summary[cellOffset + ${GPU_NEAR_REPEAT_CELL_SUMMARY.knoxRatio}u] = select(
    bitcast<f32>(parameters[parametersOffset + 3u] | 0x7fc00000u),
    f32(observed) / mean,
    mean > 0.0
  );`
      })
    ];
    return [
      ...nodes,
      // Truncated pairs give truncated cell tables: zero the counts and NaN the summary.
      ...createSpatialWeightsSuppressionNodes(
        graph,
        id,
        OPERATION,
        pairs,
        {counts, summary},
        {counts: slotCount * cells, summary: cells * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH}
      )
    ];
  }
}
