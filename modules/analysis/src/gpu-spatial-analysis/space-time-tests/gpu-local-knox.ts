// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
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
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateGPUStorageBindingByteLength,
  validatePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {
  getSpaceTimeBlocks,
  getSpaceTimeCommonWGSL,
  getSpaceTimeShuffleWGSL,
  SPACE_TIME_MAXIMUM_PERMUTATIONS,
  SPACE_TIME_MAXIMUM_ROWS,
  SPACE_TIME_WORKGROUP_SIZE
} from './space-time-kernels';
import {createSpaceTimeEntryMaskNode, resolveGPUSpaceTimeTemporalContext} from './space-time-bands';
import {
  getGPUSpaceTimeInferenceMetadata,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  type GPUSpaceTimeSettings
} from './space-time-parameters';

const OPERATION = 'GPULocalKnox';

/** Number of float32 words per event in the `summary` output of {@link GPULocalKnox}. */
export const GPU_LOCAL_KNOX_SUMMARY_LENGTH = 6;

/**
 * Word indices within one event of the `summary` output of {@link GPULocalKnox}; event `i` starts
 * at word `i * GPU_LOCAL_KNOX_SUMMARY_LENGTH`.
 */
export const GPU_LOCAL_KNOX_SUMMARY = {
  /** Observed local Knox count: spatial neighbors of the event that are also close in time. */
  observed: 0,
  /** Mean of the local count over the conditional permutations. */
  permutationMean: 1,
  /** Permutations whose local count is `>=` the observed. */
  greaterCount: 2,
  /** Permutations whose local count is `<=` the observed. */
  lesserCount: 3,
  /** `(greaterCount + 1) / (P + 1)`: pseudo p-value for an excess of close pairs at the event. */
  pseudoPGreater: 4,
  /** `(lesserCount + 1) / (P + 1)`. */
  pseudoPLesser: 5
} as const;

/** Caller-owned outputs of {@link GPULocalKnox}. */
export type GPULocalKnoxOutput = {
  /**
   * `rows * GPU_LOCAL_KNOX_SUMMARY_LENGTH` float32 words, per event as in
   * {@link GPU_LOCAL_KNOX_SUMMARY}.
   */
  summary: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPULocalKnox}.
 *
 * Per-frame (no recompile): the contents of `times`, `pairs` and `parameters` (seed, permutation
 * count, time threshold). Compile-time: the row count, `maximumPermutations` and the pair
 * capacity.
 */
export type GPULocalKnoxProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'local-knox'`. */
  id?: string;
  /**
   * Pairs close in space, as a self-join {@link GPUSpatialWeights} that lists **both directions**
   * of every pair, for example a radius-mode `GPUNeighborSearch`: row `i` lists every event within
   * the spatial threshold. The count of event `i` runs over its own row, ignoring self entries,
   * out-of-range IDs and repeats; an event listed only by a neighbor's row is not counted.
   */
  pairs: GPUSpatialWeights;
  /**
   * Event time per row, finite float32, measured from a nearby origin in the unit of
   * `timeThreshold`. Build it with `getGPUSpaceTimeGuardedRelativeTimes`.
   */
  times: GraphDataView<'float32'>;
  /**
   * Optional declaration of `times`: encoding `float32-relative`, the unit of `timeThreshold` and
   * the epoch the relative times count from. Checked at construction and exposed as
   * {@link GPULocalKnox.temporalContext}; nothing is converted.
   */
  temporalContext?: GPUTemporalContext;
  /**
   * Per-frame parameters, `GPU_SPACE_TIME_PARAMETER_LENGTH` uint32 words written with
   * `getGPUSpaceTimeParameterValues` (seed, permutation count and `timeThreshold`).
   */
  parameters: GraphDataView<'uint32'>;
  /** Compile-time upper bound on the per-frame permutation count, 1 to 65536. */
  maximumPermutations: number;
  /** Caller-owned outputs. */
  output: GPULocalKnoxOutput;
};

/**
 * Local Knox statistic (pointpats `KnoxLocal`): for every event, the number of its spatial
 * neighbors (the row of the `pairs` list) that are also close in time, `|t_i - t_j| <=
 * timeThreshold`, with conditional permutation inference per event.
 *
 * Inference follows pointpats: the focal event keeps its own time and the times of the other
 * events are permuted. Each permutation is one swap-or-not shuffle `pi` of the event times (the
 * stream of `GPUKnoxTest`, so a given seed gives the same permutations); for the focal event `i`
 * the neighbor `j` takes the time of event `pi(j)`, except that the neighbor with `pi(j) = i`
 * takes the time of `pi(i)`, so `i` never meets its own time. The same permutation serves every
 * focal event, as in pointpats. The p-value of event `i` is `(1 + #{count_perm >= count}) / (P +
 * 1)`. The counts are exact integers added with integer atomics, so the result is deterministic.
 *
 * Besag-Clifford stopping is not offered: events stop at different permutations. The p-values are
 * one 'greater' test per event with no multiple-testing correction, which matters for a map of
 * `n` tests; apply a correction to `summary` on the CPU or with the package's local-statistics
 * tools. The sum of permuted counts per event is a uint32, so `maximumPermutations` times the
 * largest row length must stay below 2^32.
 */
export class GPULocalKnox implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULocalKnoxProps;
  /** Number of events. */
  readonly rows: number;
  /** The validated context of `times`, when one was given. */
  readonly temporalContext?: GPUTemporalContext;

  constructor(props: GPULocalKnoxProps) {
    const id = props.id ?? 'local-knox';
    this.id = id;
    this.props = props;
    this.rows = validateGPUSpatialWeights(id, props.pairs, 'pairs');
    validatePackedView(props.times, ['float32'], `${id} times`);
    validatePackedView(props.parameters, ['uint32'], `${id} parameters`);
    validatePackedView(props.output.summary, ['float32'], `${id} output.summary`);
    this.temporalContext = resolveGPUSpaceTimeTemporalContext(id, props.temporalContext);
    if (this.rows < 2) {
      throw new Error(`${id} needs at least two rows`);
    }
    if (this.rows > SPACE_TIME_MAXIMUM_ROWS) {
      throw new Error(`${id} supports at most ${SPACE_TIME_MAXIMUM_ROWS} rows`);
    }
    if (props.times.length !== this.rows) {
      throw new Error(`${id} times needs one row per pairs row`);
    }
    if (props.parameters.length < GPU_SPACE_TIME_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters needs ${GPU_SPACE_TIME_PARAMETER_LENGTH} words`);
    }
    validateCapacity(
      id,
      'maximumPermutations',
      props.maximumPermutations,
      1,
      SPACE_TIME_MAXIMUM_PERMUTATIONS
    );
    if (props.output.summary.length < this.rows * GPU_LOCAL_KNOX_SUMMARY_LENGTH) {
      throw new Error(`${id} output.summary needs rows * ${GPU_LOCAL_KNOX_SUMMARY_LENGTH} rows`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.summary],
      [
        props.times,
        props.parameters,
        props.pairs.offsets,
        props.pairs.neighbors,
        props.pairs.weights
      ]
    );
  }

  /**
   * CPU-side description of how the event p-values are produced: alternative `'greater'`,
   * conditional Monte Carlo, no multiple-testing correction.
   *
   * @param settings The per-frame settings packed into `parameters`.
   */
  getInferenceMetadata(settings: GPUSpaceTimeSettings): GPUInferenceMetadata {
    return {
      ...getGPUSpaceTimeInferenceMetadata('knox', settings, this.props.maximumPermutations),
      testedRowsDefinition: 'every event, against its listed spatial neighbors'
    };
  }

  /** Returns the nodes in order: `entry-mask`, `observed`, `permutations` and `summary`. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rows} = this;
    const {pairs, times, parameters, maximumPermutations} = props;
    const {summary} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [
      pairs.offsets,
      pairs.neighbors,
      pairs.weights,
      times,
      parameters,
      summary
    ]);
    // Row-proportional scratch must fit one storage binding.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'accumulators',
      rows * 12,
      'reduce the rows'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'entry-mask',
      pairs.neighbors.length * 4,
      'reduce the pair capacity'
    );
    const [blocks, rowsPerBlock] = getSpaceTimeBlocks(rows, SPACE_TIME_WORKGROUP_SIZE);
    const common = getSpaceTimeCommonWGSL(rows, maximumPermutations, blocks, rowsPerBlock);
    const entryMask = createTransientView(
      graph,
      `${id}-entry-mask`,
      'uint32',
      pairs.neighbors.length
    );
    const observed = createTransientView(graph, `${id}-observed`, 'uint32', rows);
    // Per event: exceedances `>=`, exceedances `<=` and the sum of the permuted counts.
    const accumulators = createTransientView(graph, `${id}-accumulators`, 'uint32', rows * 3);
    return [
      createSpaceTimeEntryMaskNode<Parameters>(graph, {
        id,
        operation: OPERATION,
        rows,
        offsets: pairs.offsets,
        neighbors: pairs.neighbors,
        entryMask
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-observed`,
        operation: OPERATION,
        variant: 'observed',
        bindings: [
          {name: 'offsets', view: pairs.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: pairs.neighbors, type: 'u32', access: 'read'},
          {name: 'entryMask', view: entryMask, type: 'u32', access: 'read'},
          {name: 'times', view: times, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'observed', view: observed, type: 'u32', access: 'read_write'},
          {name: 'accumulators', view: accumulators, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: common,
        body: `let threshold = readTimeThreshold();
  let time = times[timesOffset + index];
  var count = 0u;
  for (var entry = offsets[offsetsOffset + index]; entry < offsets[offsetsOffset + index + 1u]; entry++) {
    if (entryMask[entryMaskOffset + entry] != 0u
      && abs(time - times[timesOffset + neighbors[neighborsOffset + entry]]) <= threshold) {
      count++;
    }
  }
  observed[observedOffset + index] = count;
  accumulators[accumulatorsOffset + index * 3u] = 0u;
  accumulators[accumulatorsOffset + index * 3u + 1u] = 0u;
  accumulators[accumulatorsOffset + index * 3u + 2u] = 0u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-permutations`,
        operation: OPERATION,
        variant: 'permutations',
        bindings: [
          {name: 'offsets', view: pairs.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: pairs.neighbors, type: 'u32', access: 'read'},
          {name: 'entryMask', view: entryMask, type: 'u32', access: 'read'},
          {name: 'times', view: times, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'observed', view: observed, type: 'u32', access: 'read'},
          {name: 'accumulators', view: accumulators, type: 'atomic<u32>', access: 'read_write'}
        ],
        workgroupSize: SPACE_TIME_WORKGROUP_SIZE,
        invocationCount: maximumPermutations * SPACE_TIME_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${common}
${getSpaceTimeShuffleWGSL(blocks, PHILOX_PURPOSES.knoxTestShuffle.id)}`,
        // One workgroup is one permutation 1..P; the barrier is reached by every invocation. The
        // focal event keeps its time: the neighbor mapped onto the focal event takes the time of
        // the event the focal event is mapped to (pointpats rids_i).
        body: `let slot = index / SPACE_TIME_WORKGROUP_SIZE + 1u;
  let runsSlot = slot <= readPermutationCount();
  if (runsSlot) {
    loadSwapOrNotRounds(readSeedKey(), slot, ROWS, localInvocationIndex, SPACE_TIME_WORKGROUP_SIZE);
  }
  workgroupBarrier();
  if (runsSlot) {
    let threshold = readTimeThreshold();
    for (var row = localInvocationIndex; row < ROWS; row += SPACE_TIME_WORKGROUP_SIZE) {
      let time = times[timesOffset + row];
      let focalImage = getPermutedRow(row, slot);
      var count = 0u;
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        if (entryMask[entryMaskOffset + entry] != 0u) {
          var image = getPermutedRow(neighbors[neighborsOffset + entry], slot);
          if (image == row) {
            image = focalImage;
          }
          if (abs(time - times[timesOffset + image]) <= threshold) {
            count++;
          }
        }
      }
      let observedCount = observed[observedOffset + row];
      if (count >= observedCount) {
        atomicAdd(&accumulators[accumulatorsOffset + row * 3u], 1u);
      }
      if (count <= observedCount) {
        atomicAdd(&accumulators[accumulatorsOffset + row * 3u + 1u], 1u);
      }
      atomicAdd(&accumulators[accumulatorsOffset + row * 3u + 2u], count);
    }
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'observed', view: observed, type: 'u32', access: 'read'},
          {name: 'accumulators', view: accumulators, type: 'u32', access: 'read'},
          {name: 'summary', view: summary, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: common,
        body: `let permutationCount = f32(readPermutationCount());
  let rowOffset = summaryOffset + index * ${GPU_LOCAL_KNOX_SUMMARY_LENGTH}u;
  let greater = accumulators[accumulatorsOffset + index * 3u];
  let lesser = accumulators[accumulatorsOffset + index * 3u + 1u];
  summary[rowOffset + ${GPU_LOCAL_KNOX_SUMMARY.observed}u] = f32(observed[observedOffset + index]);
  summary[rowOffset + ${GPU_LOCAL_KNOX_SUMMARY.permutationMean}u] = f32(accumulators[accumulatorsOffset + index * 3u + 2u]) / permutationCount;
  summary[rowOffset + ${GPU_LOCAL_KNOX_SUMMARY.greaterCount}u] = f32(greater);
  summary[rowOffset + ${GPU_LOCAL_KNOX_SUMMARY.lesserCount}u] = f32(lesser);
  summary[rowOffset + ${GPU_LOCAL_KNOX_SUMMARY.pseudoPGreater}u] = f32(greater + 1u) / (permutationCount + 1.0);
  summary[rowOffset + ${GPU_LOCAL_KNOX_SUMMARY.pseudoPLesser}u] = f32(lesser + 1u) / (permutationCount + 1.0);`
      })
    ];
  }
}
