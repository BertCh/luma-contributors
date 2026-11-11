// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {
  createTransientView,
  GPUSort,
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
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {
  getSpaceTimeBlocks,
  getSpaceTimeCommonWGSL,
  getSpaceTimeGroupsPerSlot,
  getSpaceTimeShuffleWGSL,
  getSpaceTimeSummaryBodyWGSL,
  SPACE_TIME_BATCH_PROLOGUE_WGSL,
  SPACE_TIME_MAXIMUM_PERMUTATIONS,
  SPACE_TIME_MAXIMUM_ROWS,
  SPACE_TIME_SLOT_PROLOGUE_WGSL,
  SPACE_TIME_WORKGROUP_SIZE
} from './space-time-kernels';
import {createSpaceTimeEntryMaskNode, resolveGPUSpaceTimeTemporalContext} from './space-time-bands';
import {
  getGPUSpaceTimeInferenceMetadata,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY_LENGTH,
  type GPUSpaceTimeSettings
} from './space-time-parameters';
import {
  createSpaceTimeStateView,
  createSpaceTimeStoppingNodes,
  getSpaceTimeStoppingBatchSize,
  type SpaceTimeBatchNodeProps,
  validateSpaceTimeStopping
} from './space-time-stopping';
import {GET_ORDERED_KEY_WGSL} from '../../analysis-core/kernels/wgsl-common';

const OPERATION = 'GPUJacquezTest';

/** Largest compile-time `k` of {@link GPUJacquezTest}. */
export const GPU_JACQUEZ_MAXIMUM_K = 256;

/** Caller-owned outputs of {@link GPUJacquezTest}. */
export type GPUJacquezTestOutput = {
  /**
   * Caller-owned statistics, `maximumPermutations + 1` rows: element 0 is the observed Jacquez
   * count, elements `1..P` the counts under permuted times (rows past `P` are 0).
   */
  statistics: GraphDataView<'uint32'>;
  /**
   * Caller-owned `GPU_SPACE_TIME_SUMMARY_LENGTH` float32 words, see `GPU_SPACE_TIME_SUMMARY`: here
   * `pairCount` is the number of listed spatial neighbor entries, `timeClosePairs` is `rows * k`
   * (the temporal neighbor entries) and `expected` is `pairCount * k / (rows - 1)`.
   */
  summary: GraphDataView<'float32'>;
  /**
   * One uint32 word: the number of permutations used, `L` when `stopAfterExceedances` stopped the
   * run and the permutation count `P` otherwise. Required with `stopAfterExceedances`, rejected
   * without it.
   */
  permutationCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUJacquezTest}.
 *
 * Per-frame (no recompile): the contents of `times`, `neighbors` and `parameters` (seed and
 * permutation count). Compile-time: `k`, the row count, `maximumPermutations` and the list
 * capacity.
 */
export type GPUJacquezTestProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'jacquez-test'`. */
  id?: string;
  /**
   * Spatial k-nearest-neighbor list, a self-join {@link GPUSpatialWeights}: row `i` lists the `k`
   * nearest events of event `i` in space, for example a k-nearest-neighbor `GPUNeighborSearch`.
   * Only the neighbor IDs are read; self entries, out-of-range IDs and repeats within a row are
   * ignored. The list is not symmetrized.
   */
  neighbors: GPUSpatialWeights;
  /**
   * Event time per row, finite float32 (NaN is not supported), measured from a nearby origin. Build
   * it with `getGPUSpaceTimeGuardedRelativeTimes`.
   */
  times: GraphDataView<'float32'>;
  /**
   * Optional declaration of `times`: encoding `float32-relative`, the unit of the times and the
   * epoch they count from. Checked at construction and exposed as
   * {@link GPUJacquezTest.temporalContext}; nothing is converted.
   */
  temporalContext?: GPUTemporalContext;
  /**
   * Per-frame parameters, `GPU_SPACE_TIME_PARAMETER_LENGTH` uint32 words written with
   * `getGPUSpaceTimeParameterValues` (seed and permutation count; `timeThreshold` is ignored).
   */
  parameters: GraphDataView<'uint32'>;
  /**
   * Compile-time number of nearest neighbors in time, 1 to {@link GPU_JACQUEZ_MAXIMUM_K} and below
   * the row count. Use the `k` of the spatial list.
   */
  k: number;
  /** Compile-time upper bound on the per-frame permutation count, 1 to 65536. */
  maximumPermutations: number;
  /**
   * Compile-time Besag-Clifford sequential stopping with an integer `h` in `[1,
   * maximumPermutations]`, exactly as {@link GPUKnoxTestProps.stopAfterExceedances}. Requires
   * `output.permutationCount`.
   */
  stopAfterExceedances?: number;
  /** Caller-owned outputs. */
  output: GPUJacquezTestOutput;
};

/**
 * Jacquez k-nearest-neighbor test for space-time interaction (pointpats `jacquez`): the number `J`
 * of (event, neighbor) pairs where the neighbor is among the `k` nearest events of the event in
 * space (the `neighbors` list) and among its `k` nearest events in time. Sums over the directed
 * lists, as pointpats does, so a mutual pair counts twice.
 *
 * Temporal neighbors are computed on the GPU from the sorted times (a stable `GPUSort`): the `k`
 * nearest events in time of an event form a contiguous interval of `k + 1` ranks around its own
 * rank, grown one rank at a time toward the nearer side. Ties in the time distance take the lower
 * rank (the earlier of equal times in row order), so with ties the chosen set can differ from
 * another library's arbitrary choice; without ties it is exactly the `k` nearest. Membership under
 * a permutation is then one rank comparison, `rank(pi(j))` inside `[lo(pi(i)), lo(pi(i)) + k]`.
 *
 * Reference distribution: `P` swap-or-not permutations of the event times (the stream of
 * `GPUKnoxTest`), the spatial lists fixed, so `E[J] = sum_i |N_i| k / (n - 1)` for `|N_i|` listed
 * neighbors; the pseudo p-value is `(1 + #{J_perm >= J}) / (P + 1)`, 'greater' only. Counts are
 * exact integers summed in fixed order and equal a CPU oracle exactly. Jacquez (1996) is the
 * source of the test; Malizia and Mack (2012) note its sensitivity to population shift. With
 * `stopAfterExceedances` the run stops as `GPUKnoxTest` does.
 */
export class GPUJacquezTest implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUJacquezTestProps;
  /** Number of events. */
  readonly rows: number;
  /** The validated context of `times`, when one was given. */
  readonly temporalContext?: GPUTemporalContext;

  constructor(props: GPUJacquezTestProps) {
    const id = props.id ?? 'jacquez-test';
    this.id = id;
    this.props = props;
    this.rows = validateGPUSpatialWeights(id, props.neighbors, 'neighbors');
    validateSinglePackedView(props.times, ['float32'], `${id} times`);
    validateSinglePackedView(props.parameters, ['uint32'], `${id} parameters`);
    validateSinglePackedView(props.output.statistics, ['uint32'], `${id} output.statistics`);
    validateSinglePackedView(props.output.summary, ['float32'], `${id} output.summary`);
    this.temporalContext = resolveGPUSpaceTimeTemporalContext(id, props.temporalContext);
    validateCapacity(id, 'k', props.k, 1, GPU_JACQUEZ_MAXIMUM_K);
    if (this.rows <= props.k) {
      throw new Error(`${id} needs more than k rows`);
    }
    if (this.rows > SPACE_TIME_MAXIMUM_ROWS) {
      throw new Error(`${id} supports at most ${SPACE_TIME_MAXIMUM_ROWS} rows`);
    }
    if (props.times.length !== this.rows) {
      throw new Error(`${id} times needs one row per neighbors row`);
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
    if (props.output.statistics.length < props.maximumPermutations + 1) {
      throw new Error(`${id} output.statistics needs maximumPermutations + 1 rows`);
    }
    validateSpaceTimeStopping(
      id,
      props.stopAfterExceedances,
      props.maximumPermutations,
      props.output.permutationCount
    );
    if (props.output.summary.length < GPU_SPACE_TIME_SUMMARY_LENGTH) {
      throw new Error(`${id} output.summary needs ${GPU_SPACE_TIME_SUMMARY_LENGTH} rows`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.statistics, props.output.summary, props.output.permutationCount],
      [
        props.times,
        props.parameters,
        props.neighbors.offsets,
        props.neighbors.neighbors,
        props.neighbors.weights
      ]
    );
    if (
      props.output.statistics.buffer === props.output.summary.buffer ||
      (props.output.permutationCount &&
        (props.output.permutationCount.buffer === props.output.statistics.buffer ||
          props.output.permutationCount.buffer === props.output.summary.buffer))
    ) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * CPU-side description of how the `output.summary` p-values are produced: alternative
   * `'greater'`, Monte Carlo, no multiple-testing correction.
   *
   * @param settings The per-frame settings packed into `parameters`.
   */
  getInferenceMetadata(settings: GPUSpaceTimeSettings): GPUInferenceMetadata {
    return {
      ...getGPUSpaceTimeInferenceMetadata('knox', settings, this.props.maximumPermutations),
      testedRowsDefinition: 'all listed (event, spatial neighbor) entries, counted once each'
    };
  }

  /**
   * Returns the nodes in order: `keys`, the time sort, `rank`, `window`, `entry-mask`, the
   * permutation partials and reduce (or the stopping batches) and `summary`.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rows} = this;
    const {neighbors, times, parameters, maximumPermutations, k} = props;
    const {statistics, summary} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [
      neighbors.offsets,
      neighbors.neighbors,
      neighbors.weights,
      times,
      parameters,
      statistics,
      summary,
      ...(props.output.permutationCount ? [props.output.permutationCount] : [])
    ]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'event-ranks',
      rows * 8,
      'reduce the rows'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'entry-mask',
      neighbors.neighbors.length * 4,
      'reduce the list capacity'
    );
    const [blocks, rowsPerBlock] = getSpaceTimeBlocks(rows);
    const slotCount = maximumPermutations + 1;
    const groupsPerSlot = getSpaceTimeGroupsPerSlot(blocks);
    const common = getSpaceTimeCommonWGSL(rows, maximumPermutations, blocks, rowsPerBlock);
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const stopAfterExceedances = props.stopAfterExceedances;
    const stopping = stopAfterExceedances !== undefined;
    const batchSize = getSpaceTimeStoppingBatchSize(slotCount);
    const tableSlots = stopping ? batchSize : slotCount;
    const state = stopping ? createSpaceTimeStateView(graph, id) : undefined;
    const stateBindings = state
      ? [{name: 'state', view: state, type: 'u32' as const, access: 'read' as const}]
      : [];
    const batchDeclarations = stopping ? `\nconst BATCH_SIZE: u32 = ${batchSize}u;` : '';
    const sortKeys = u32('sort-keys', rows);
    const sortIndices = u32('sort-indices', rows);
    const sortedKeys = u32('sorted-keys', rows);
    const sortedIndices = u32('sorted-indices', rows);
    // Two words per event: its rank in the time order, then the first rank of its temporal
    // neighborhood. One view keeps the partials kernel within eight storage bindings.
    const eventRanks = u32('event-ranks', rows * 2);
    const entryMask = u32('entry-mask', neighbors.neighbors.length);
    const partials = u32('partials', tableSlots * blocks);
    const pairPartials = u32('pair-partials', blocks);
    const createPartialsNode = (batch?: SpaceTimeBatchNodeProps<Parameters>) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: batch?.id ?? `${id}-partials`,
        operation: OPERATION,
        variant: 'partials',
        bindings: [
          {name: 'offsets', view: neighbors.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: neighbors.neighbors, type: 'u32', access: 'read'},
          {name: 'entryMask', view: entryMask, type: 'u32', access: 'read'},
          {name: 'eventRanks', view: eventRanks, type: 'u32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          ...stateBindings,
          {name: 'partials', view: partials, type: 'u32', access: 'read_write'},
          {name: 'pairPartials', view: pairPartials, type: 'u32', access: 'read_write'}
        ],
        condition: batch?.condition,
        extraResources: batch?.extraResources,
        workgroupSize: SPACE_TIME_WORKGROUP_SIZE,
        invocationCount: tableSlots * groupsPerSlot * SPACE_TIME_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${common}
${getSpaceTimeShuffleWGSL(blocks, PHILOX_PURPOSES.knoxTestShuffle.id)}
const SLOT_COUNT: u32 = ${tableSlots}u;
const NEAREST: u32 = ${k}u;${batchDeclarations}`,
        // As the Knox partials: a workgroup is one permutation, an invocation owns interleaved
        // rows. The row at location `row` carries the event `image = pi(row)`; its temporal
        // neighbors are the ranks `windowStart[image] .. windowStart[image] + NEAREST`, and a
        // spatial neighbor `other` counts when the event `pi(other)` has a rank in that interval.
        body: `${stopping ? SPACE_TIME_BATCH_PROLOGUE_WGSL : SPACE_TIME_SLOT_PROLOGUE_WGSL}
  var count = 0u;
  var entryCount = 0u;
  if (runsSlot) {
    for (var row = block; row < ROWS; row += BLOCKS) {
      let start = eventRanks[eventRanksOffset + getPermutedRow(row, slot) * 2u + 1u];
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        if (entryMask[entryMaskOffset + entry] != 0u) {
          entryCount++;
          let otherRank = eventRanks[eventRanksOffset + getPermutedRow(neighbors[neighborsOffset + entry], slot) * 2u];
          if (otherRank >= start && otherRank <= start + NEAREST) {
            count++;
          }
        }
      }
    }
  }
  if (block < BLOCKS) {
    partials[partialsOffset + block * SLOT_COUNT + ${stopping ? 'localSlot' : 'slot'}] = count;
  }
  if (slot == 0u && block < BLOCKS) {
    pairPartials[pairPartialsOffset + block] = entryCount;
  }`
      });
    const createReduceNode = (batch?: SpaceTimeBatchNodeProps<Parameters>) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: batch?.id ?? `${id}-reduce`,
        operation: OPERATION,
        variant: 'reduce',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          ...stateBindings,
          {name: 'partials', view: partials, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'u32', access: 'read_write'}
        ],
        condition: batch?.condition,
        extraResources: batch?.extraResources,
        ...(stopping
          ? {workgroupSize: 64, invocationCount: batchSize}
          : {invocationCount: statistics.length}),
        declarations: `${common}
const SLOT_COUNT: u32 = ${tableSlots}u;${batchDeclarations}`,
        body: stopping
          ? `let slot = state[stateOffset] * BATCH_SIZE + index;
  if (slot <= readPermutationCount()) {
    var total = 0u;
    for (var block = 0u; block < BLOCKS; block++) {
      total += partials[partialsOffset + block * SLOT_COUNT + index];
    }
    statistics[statisticsOffset + slot] = total;
  }`
          : `var total = 0u;
  if (index <= readPermutationCount()) {
    for (var block = 0u; block < BLOCKS; block++) {
      total += partials[partialsOffset + block * SLOT_COUNT + index];
    }
  }
  statistics[statisticsOffset + index] = total;`
      });
    const stoppingNodes =
      stopping && state
        ? createSpaceTimeStoppingNodes<Parameters>(graph, {
            id,
            operation: OPERATION,
            stopAfterExceedances,
            slotCount,
            groupsPerSlot,
            workgroupSize: SPACE_TIME_WORKGROUP_SIZE,
            statisticType: 'u32',
            parameters,
            statistics,
            state,
            common,
            createPartialsNode,
            createReduceNode
          })
        : [];
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-keys`,
        operation: OPERATION,
        variant: 'keys',
        bindings: [
          {name: 'times', view: times, type: 'f32', access: 'read'},
          {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
          {name: 'sortIndices', view: sortIndices, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: GET_ORDERED_KEY_WGSL,
        body: `sortKeys[sortKeysOffset + index] = getOrderedKey(times[timesOffset + index]);
  sortIndices[sortIndicesOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-sort`,
        keys: sortKeys,
        values: sortIndices,
        outputKeys: sortedKeys,
        outputValues: sortedIndices
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rank`,
        operation: OPERATION,
        variant: 'rank',
        bindings: [
          {name: 'sortedIndices', view: sortedIndices, type: 'u32', access: 'read'},
          {name: 'eventRanks', view: eventRanks, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        body: 'eventRanks[eventRanksOffset + sortedIndices[sortedIndicesOffset + index] * 2u] = index;'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-window`,
        operation: OPERATION,
        variant: 'window',
        bindings: [
          {name: 'times', view: times, type: 'f32', access: 'read'},
          {name: 'sortedIndices', view: sortedIndices, type: 'u32', access: 'read'},
          {name: 'eventRanks', view: eventRanks, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `const ROWS: u32 = ${rows}u;
const NEAREST: u32 = ${k}u;`,
        // Grows the interval [low, high] around the event's rank by the nearer neighbor (the lower
        // rank on a tie) until it holds NEAREST + 1 ranks; at an end of the order it grows the
        // other way. The interval start is the whole temporal neighborhood.
        body: `let time = times[timesOffset + index];
  var low = eventRanks[eventRanksOffset + index * 2u];
  var high = low;
  for (var step = 0u; step < NEAREST; step++) {
    var takeLow = false;
    if (low == 0u) {
      takeLow = false;
    } else if (high + 1u == ROWS) {
      takeLow = true;
    } else {
      let lowGap = time - times[timesOffset + sortedIndices[sortedIndicesOffset + low - 1u]];
      let highGap = times[timesOffset + sortedIndices[sortedIndicesOffset + high + 1u]] - time;
      takeLow = lowGap <= highGap;
    }
    if (takeLow) {
      low--;
    } else {
      high++;
    }
  }
  eventRanks[eventRanksOffset + index * 2u + 1u] = low;`
      }),
      createSpaceTimeEntryMaskNode<Parameters>(graph, {
        id,
        operation: OPERATION,
        rows,
        offsets: neighbors.offsets,
        neighbors: neighbors.neighbors,
        entryMask
      }),
      ...(stopping ? stoppingNodes : [createPartialsNode(), createReduceNode()]),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'u32', access: 'read'},
          {name: 'pairPartials', view: pairPartials, type: 'u32', access: 'read'},
          {name: 'summary', view: summary, type: 'f32', access: 'read_write'},
          ...stateBindings,
          ...(props.output.permutationCount
            ? [
                {
                  name: 'permutationsUsed',
                  view: props.output.permutationCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: 1,
        declarations: common,
        body: `var pairTotal = 0u;
  for (var block = 0u; block < BLOCKS; block++) {
    pairTotal += pairPartials[pairPartialsOffset + block];
  }
  let observedPairs = f32(pairTotal);
  let timeClosePairs = f32(ROWS) * ${k}.0;
  let expected = observedPairs * ${k}.0 / f32(ROWS - 1u);
  ${getSpaceTimeSummaryBodyWGSL('u32', stopAfterExceedances)}`
      })
    ];
  }
}
