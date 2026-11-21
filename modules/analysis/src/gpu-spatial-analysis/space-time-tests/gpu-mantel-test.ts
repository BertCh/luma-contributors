// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createSpatialWeightsSuppressionNodes} from '../spatial-weights/spatial-weights-suppression';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createMantelResidualPublicationNodes} from './mantel-residual-publication';
import {fp64arithmetic} from '@luma.gl/shaders';
import {snapshotSpaceTimeTestProps} from './space-time-construction';

import {
  validateTrajectoryBindings,
  validateTrajectoryWritableDispatches,
  validateTrajectoryScratch,
  validateTrajectoryOutputRanges
} from '../trajectory-analysis/trajectory-binding-validation';
import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement,
  validateGraphPorts,
  validateViewLength,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {
  createSpaceTimePairMaskNode,
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
import {
  getGPUSpaceTimeInferenceMetadata,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY_LENGTH,
  type GPUSpaceTimePValueField,
  type GPUSpaceTimeSettings
} from './space-time-parameters';
import {
  createSpaceTimeStateView,
  SPACE_TIME_STATE_LENGTH,
  createSpaceTimeStoppingNodes,
  getSpaceTimeStoppingBatchSize,
  SPACE_TIME_STATE,
  type SpaceTimeBatchNodeProps,
  validateSpaceTimeStopping
} from './space-time-stopping';
import type {GPUMantelTestInferenceMetadata} from './space-time-parameters';

import type {GPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
import {resolveGPUSpaceTimeTemporalContext} from './space-time-bands';

const OPERATION = 'GPUMantelTest';
/**
 * Float words per (permutation, block) partial: two words each for `sum y, sum y^2, sum x y`, transformed x extrema
 * and their associated transformed y values (four words for the exact two-pair ordering),
 * then transformed y extrema (two words to recognize zero temporal norm without cancellation).
 * The spatial moments
 * `sum x` and `sum x^2` (two words each) do not depend on the permutation, so they are computed once per block.
 */
const PARTIAL_STRIDE = 12;
/** Largest block count of the observed-moment pass, which has no per-permutation parallelism. */
const MAXIMUM_PAIR_STAT_BLOCKS = 4096;

/**
 * The pointpats `mantel` transform `(x + constant) ** power` applied to every pair distance or
 * absolute time difference before correlating. `scon`/`spow` (distance) and `tcon`/`tpow` (time)
 * of pointpats 2.6.0 map to `constant`/`power`. `x + constant` must stay positive when `power` is
 * not a non-negative integer (as in pointpats, the result is otherwise not finite).
 */
export type GPUMantelTransform = {
  /** Constant added to the value, finite. */
  constant: number;
  /** Power the shifted value is raised to, finite. */
  power: number;
};

/**
 * The defaults of pointpats `mantel` (`scon = tcon = 1`, `spow = tpow = -1`): the reciprocal
 * `1 / (x + 1)`. Pass it as both `distanceTransform` and `timeTransform` to reproduce pointpats.
 */
export const GPU_MANTEL_POINTPATS_TRANSFORM: Readonly<GPUMantelTransform> = {
  constant: 1,
  power: -1
};

/** Caller-owned outputs of {@link GPUMantelTest}. */
export type GPUMantelTestOutput = {
  /**
   * Caller-owned statistics, `maximumPermutations + 1` rows: element 0 is the observed correlation
   * `r`, elements `1..P` the correlation under permuted times (rows past `P` are 0).
   */
  statistics: GraphDataView<'float32'>;
  /** Caller-owned `GPU_SPACE_TIME_SUMMARY_LENGTH` float32 words, see `GPU_SPACE_TIME_SUMMARY`. */
  summary: GraphDataView<'float32'>;
  /**
   * One uint32 word: the number of permutations used, `L` when `stopAfterExceedances` stopped the
   * run and the permutation count `P` otherwise. It bounds the attempted statistics prefix,
   * including NaN null slots. Zero when the observation is undefined. Optional without stopping;
   * required with stopping. Use usablePermutationCount for the inference denominator.
   */
  permutationCount?: GraphDataView<'uint32'>;
  /** Defined null correlations used by inference; optional for either route. */
  usablePermutationCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUMantelTest}.
 *
 * Per-frame (no recompile): the contents of `times`, `pairs` and `parameters` (seed, permutation
 * count). Compile-time: the row count, `maximumPermutations` and the pair capacity.
 */
export type GPUMantelTestProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'mantel-test'`. */
  id?: string;
  /**
   * Pairs and their spatial distance, as a self-join {@link GPUSpatialWeights} with `distances`:
   * the Mantel statistic runs over every unordered pair that appears in either direction, counted
   * exactly once (taken from the lower row when both directions are listed, otherwise from the row
   * that lists it; duplicates and self entries are ignored), with `distances` as the spatial
   * distance. A radius-mode `GPUNeighborSearch` gives the distance band; a band wide enough to
   * list every pair gives the full Mantel test.
   * Pairs carry their own `spatialContext` from `GPUNeighborSearch`. No context check is
   * needed: the correlation is unit-free and no spatial distance parameter is supplied.
   */
  pairs: GPUSpatialWeights;
  /**
   * Event time per row, finite float32 (NaN is not supported), measured from a nearby origin: raw
   * epoch seconds do not fit float32 (128 s spacing near 1.7e9), so subtract an origin on the CPU
   * with `getGPUSpaceTimeRelativeTimes`.
   */
  times: GraphDataView<'float32'>;
  /** Optional metadata; float32-relative ticks, captured with the nested axis. */
  temporalContext?: GPUTemporalContext;
  /** Per-frame parameters: `GPU_SPACE_TIME_PARAMETER_LENGTH` uint32 words (`getGPUSpaceTimeParameterValues`). */
  parameters: GraphDataView<'uint32'>;
  /** Compile-time upper bound on the per-frame permutation count, 1 to 65536. */
  maximumPermutations: number;
  /**
   * Compile-time Besag-Clifford sequential stopping (Besag and Clifford 1991): an integer `h` in
   * `[1, maximumPermutations]`. Omitted (the default) runs all `P` permutations, unchanged. With
   * `h`, the run stops at the first permutation `L` whose running count of permuted correlations
   * `>=` the observed `r` equals `h`. Undefined null draws do not exceed. With `U` defined
   * correlations in that prefix, `pseudoPGreater = min(1, h / U)`; when fewer than `h` exceed,
   * it is `(greaterCount + 1) / (U + 1)`, or NaN when U is zero. Permutations are evaluated
   * in batches of 64 or more (the batch size grows so there are at most 256 batches), each batch
   * scanned in permutation order on the GPU, so `L` and the p-value do not depend on the batch
   * size, and a stopped run skips the remaining batches. The first `L` permutations equal those
   * of the full run. `statistics` rows past `L` are 0 and the `summary` mean, variance, counts and
   * z-score cover the first `L` permutations; `pseudoPLesser` and `pseudoPTwoSided` keep their
   * fixed-`U` formulas over defined draws and are not sequentially valid. Requires `output.permutationCount`, which
   * receives `L`.
   */
  stopAfterExceedances?: number;
  /**
   * Compile-time transform of the pair distances before correlating, `(distance + constant) **
   * power` like pointpats `scon` and `spow`. Omitted (the default) correlates the raw distances.
   */
  distanceTransform?: GPUMantelTransform;
  /**
   * Compile-time transform of the absolute time differences before correlating,
   * `(|t_i - t_j| + constant) ** power` like pointpats `tcon` and `tpow`. Omitted (the default)
   * correlates the raw differences.
   */
  timeTransform?: GPUMantelTransform;
  /** Caller-owned outputs. */
  output: GPUMantelTestOutput;
};

/** WGSL `fn name(value: f32) -> f32` applying a {@link GPUMantelTransform} (identity when omitted). */
function getTransformWGSL(name: string, transform: GPUMantelTransform | undefined): string {
  if (!transform) {
    return `fn ${name}(value: f32) -> f32 { return value; }`;
  }
  // toPrecision always yields a WGSL float literal (a decimal point or an exponent).
  const constant = `(${transform.constant.toPrecision(9)})`;
  const power = `(${transform.power.toPrecision(9)})`;
  return `fn ${name}(value: f32) -> f32 { return pow(value + ${constant}, ${power}); }`;
}

/**
 * Mantel test of space-time interaction (pointpats `Mantel`): the Pearson correlation `r` between
 * spatial distances `d_ij` and absolute time differences `|t_i - t_j|` over the listed pairs
 * `i < j`, with a Monte Carlo reference from `P` swap-or-not permutations of the times.
 *
 * Breaking change: the permutations used to be a Feistel bijection, which is measurably non-uniform
 * on small domains; the swap-or-not shuffle (Hoang-Morris-Rogaway) is uniform, so the permuted
 * correlations for a given seed differ from earlier versions. Permutation `s >= 1` is draw `s` of
 * the seed. Fewer than 2^24 events are supported.
 *
 * `r` is NaN with fewer than two usable pairs or when either side has no variance.
 * Undefined observations publish no inference and zero usable draws. Undefined null draws
 * remain NaN and are excluded from both tails, moments and their denominator. The pair list is a restriction of the full distance
 * matrices when a band is used: the statistic is the correlation over the listed pairs only (pairs
 * farther apart than the band are truncated away, so `r` is a distance-band Mantel statistic and
 * differs from the full Mantel `r`; report the band with the result), and a full-matrix Mantel test
 * needs a list of every pair (O(n^2) slots). By default the raw distances and time differences are
 * correlated (the scikit-bio `mantel` definition). pointpats `mantel` instead correlates
 * `(x + constant) ** power` of both, with the defaults `1` and `-1` (reciprocals): pass
 * `distanceTransform` and `timeTransform` (for example `GPU_MANTEL_POINTPATS_TRANSFORM`) to match
 * it. The time transform applies to `|t_i - t_j|`, which cannot be done by transforming `times`.
 *
 * Numerics: partial sums per (permutation, row block) are f32, centred on a selected observed
 * pair's transformed distance and each draw's own anchor time gap to avoid fractional-mean rounding, and
 * reduced in fixed order, so results are deterministic and match a double-precision CPU oracle to
 * about 1e-4 in `r` for up to 10^6 pairs. A permuted `r` within f32 rounding of the observed `r`
 * can fall on either side of the exceedance count.
 *
 * Alternative: the space-time interaction hypothesis is a positive correlation (events near in
 * space are near in time), the upper-tail convention of pointpats `mantel`, so
 * `getInferenceMetadata` states `'greater'` and `pseudoPGreater` is the p-value. `pseudoPTwoSided` is also in the `summary`.
 *
 * With maximumPermutations=M and storage binding limit B bytes, rowBlockCount is at most
 * min(256, rows, floor(B / (4 * (9 * (M + 1) + 2)))). Each draw/block has nine float32 words,
 * plus two spatial moment words per block. Both routes choose this bound; stopping stores one batch.
 *
 * Time units: pass times relative to an origin (`getGPUSpaceTimeRelativeTimes`); float32 times
 * resolve about `span * 2^-24`.
 */
export class GPUMantelTest implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUMantelTestProps;
  /** Number of events. */
  readonly rows: number;
  constructor(props: GPUMantelTestProps) {
    props = snapshotSpaceTimeTestProps(
      props,
      [
        'id',
        'pairs',
        'times',
        'temporalContext',
        'parameters',
        'maximumPermutations',
        'stopAfterExceedances',
        'distanceTransform',
        'timeTransform',
        'output'
      ],
      'pairs'
    );
    const id = props.id ?? 'mantel-test';
    this.id = id;
    this.props = props;
    validateGraphPorts(id, props, {
      'pairs.distances': {required: true, viewType: 'data', formats: ['float32']},
      times: {required: true, viewType: 'data', formats: ['float32']},
      parameters: {required: true, viewType: 'data', formats: ['uint32']},
      'output.statistics': {required: true, viewType: 'data', formats: ['float32']},
      'output.summary': {required: true, viewType: 'data', formats: ['float32']},
      'output.permutationCount': {required: false, viewType: 'data', formats: ['uint32']},
      'output.usablePermutationCount': {required: false, viewType: 'data', formats: ['uint32']}
    });
    for (const [name, view] of [
      ['times', props.times],
      ['output.statistics', props.output.statistics],
      ['output.summary', props.output.summary]
    ] as const)
      validateSinglePackedView(view, ['float32'], `${id} ${name}`);
    validateSinglePackedView(props.parameters, ['uint32'], `${id} parameters`);
    if (props.output.usablePermutationCount)
      validateSinglePackedView(
        props.output.usablePermutationCount,
        ['uint32'],
        `${id} output.usablePermutationCount`
      );
    if (props.output.permutationCount)
      validateSinglePackedView(
        props.output.permutationCount,
        ['uint32'],
        `${id} output.permutationCount`
      );
    this.rows = validateGPUSpatialWeights(id, props.pairs, 'pairs');
    resolveGPUSpaceTimeTemporalContext(id, props.temporalContext);
    for (const [name, transform] of [
      ['distanceTransform', props.distanceTransform],
      ['timeTransform', props.timeTransform]
    ] as const) {
      if (transform) {
        validateFiniteNumber(id, `${name}.constant`, transform.constant);
        validateFiniteNumber(id, `${name}.power`, transform.power);
        validateRequirement(
          id,
          Number.isFinite(Math.fround(transform.constant)) &&
            Number.isFinite(Math.fround(transform.power)),
          `${name} must encode as finite float32; received constant ${transform.constant}, power ${transform.power}`
        );
      }
    }
    validateCapacity(id, 'pairs rowCount', this.rows, 2, SPACE_TIME_MAXIMUM_ROWS);
    validateViewLength(id, 'times', props.times, this.rows);
    validateCapacity(
      id,
      'parameters.length',
      props.parameters.length,
      GPU_SPACE_TIME_PARAMETER_LENGTH
    );
    validateCapacity(
      id,
      'maximumPermutations',
      props.maximumPermutations,
      1,
      SPACE_TIME_MAXIMUM_PERMUTATIONS
    );
    validateCapacity(
      id,
      'output.statistics.length',
      props.output.statistics.length,
      props.maximumPermutations + 1
    );
    validateSpaceTimeStopping(
      id,
      props.stopAfterExceedances,
      props.maximumPermutations,
      props.stopAfterExceedances === undefined ? undefined : props.output.permutationCount
    );
    if (props.output.permutationCount)
      validateCapacity(
        id,
        'output.permutationCount.length',
        props.output.permutationCount.length,
        1
      );
    if (props.output.usablePermutationCount)
      validateCapacity(
        id,
        'output.usablePermutationCount.length',
        props.output.usablePermutationCount.length,
        1
      );
    validateCapacity(
      id,
      'output.summary.length',
      props.output.summary.length,
      GPU_SPACE_TIME_SUMMARY_LENGTH
    );
    validateTrajectoryOutputRanges(this.id, this.props.output);
    validateTrajectoryWritableDispatches(this.id, [
      {
        summary: props.output.summary,
        permutationCount: props.output.permutationCount,
        usablePermutationCount: props.output.usablePermutationCount
      }
    ]);
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.output.statistics,
        props.output.summary,
        props.output.permutationCount,
        props.output.usablePermutationCount
      ],
      [
        props.times,
        props.parameters,
        props.pairs.offsets,
        props.pairs.neighbors,
        props.pairs.weights,
        props.pairs.distances,
        props.pairs.status?.overflow,
        props.pairs.status?.candidateOverflow
      ]
    );
    validateRequirement(
      id,
      !(
        props.output.statistics.buffer === props.output.summary.buffer ||
        (props.output.permutationCount &&
          (props.output.permutationCount.buffer === props.output.statistics.buffer ||
            props.output.permutationCount.buffer === props.output.summary.buffer))
      ),
      'outputs must use separate buffers'
    );
  }

  /**
   * CPU-side description of how the `output.summary` p-values are produced, for building a
   * legend: alternative `'greater'`, Monte Carlo, no multiple-testing correction.
   * With stopping enabled, only `requestedPermutationCount` is known. Decode final metadata with
   * `decodeGPUSpaceTimeInferenceMetadata(metadata, attemptedCount, usableCount)` after caller readback.
   * Even without stopping, settings metadata cannot know the usable null count.
   *
   * @param settings The per-frame settings packed into `parameters` (validated as
   * `getGPUSpaceTimeParameterValues` does; the count is clamped to `maximumPermutations`).
   */
  getInferenceMetadata(
    settings: GPUSpaceTimeSettings,
    summaryField: GPUSpaceTimePValueField = 'pseudoPGreater'
  ): GPUMantelTestInferenceMetadata {
    return getGPUSpaceTimeInferenceMetadata(
      'mantel',
      settings,
      this.props.maximumPermutations,
      this.props.stopAfterExceedances,
      summaryField
    );
  }

  /** Returns `pair-mask`, `pair-stats`, `shift`, `partials`, `reduce` and `summary` nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateTrajectoryBindings(graph, this.id, this.props);
    const {id, props, rows} = this;
    const {pairs, times, parameters, maximumPermutations} = props;
    const {statistics, summary} = props.output;
    const distances = pairs.distances!;
    validateGraphViewsBelongToGraph(id, graph, [
      pairs.offsets,
      pairs.neighbors,
      pairs.weights,
      distances,
      times,
      parameters,
      statistics,
      summary,
      ...(props.output.permutationCount ? [props.output.permutationCount] : []),
      ...(props.output.usablePermutationCount ? [props.output.usablePermutationCount] : [])
    ]);
    const slotCount = maximumPermutations + 1;
    // The (permutation, block) partials and the spatial moments share one storage binding: at
    // many permutations fewer row blocks keep it within maxStorageBufferBindingSize (rule 16).
    const bindingWords = Math.floor(graph.device.limits.maxStorageBufferBindingSize / 4);
    const [blocks, rowsPerBlock] = getSpaceTimeBlocks(
      rows,
      Math.floor(bindingWords / (slotCount * PARTIAL_STRIDE + 4))
    );
    const groupsPerSlot = getSpaceTimeGroupsPerSlot(blocks);
    const stopAfterExceedances = props.stopAfterExceedances;
    const stopping = stopAfterExceedances !== undefined;
    // Stopping evaluates the slots in batches, so the (slot, block) table holds one batch. The
    // block count is unchanged so the sums, and with them the results, equal the full run's.
    const batchSize = getSpaceTimeStoppingBatchSize(slotCount);
    const tableSlots = stopping ? batchSize : slotCount;
    validateTrajectoryScratch(graph, id, {
      'permutation partials': (tableSlots * blocks * PARTIAL_STRIDE + blocks * 4) * 4,
      'residual statistics': statistics.length * 8,
      'pair statistic sums': Math.min(MAXIMUM_PAIR_STAT_BLOCKS, rows) * 16,
      'draw-local anchor rows': 16,
      'anchor block candidates': Math.min(MAXIMUM_PAIR_STAT_BLOCKS, rows) * 4,
      'pair mask': pairs.neighbors.length * 4,
      ...(stopping
        ? {
            'stopping state': SPACE_TIME_STATE_LENGTH * 4,
            'stopping dispatch': 24,
            'stopping comparisons': statistics.length * 4
          }
        : {})
    });
    const state = stopping ? createSpaceTimeStateView(graph, id) : undefined;
    const comparisonStatistics = stopping
      ? createTransientView(graph, `${id}-stop-comparisons`, 'float32', statistics.length)
      : undefined;
    const stateBindings = state
      ? [{name: 'state', view: state, type: 'u32' as const, access: 'read' as const}]
      : [];
    const batchDeclarations = stopping ? `\nconst BATCH_SIZE: u32 = ${batchSize}u;` : '';
    const common = `${GET_NAN_WGSL}
${getSpaceTimeCommonWGSL(rows, maximumPermutations, blocks, rowsPerBlock)}
${getTransformWGSL('transformDistance', props.distanceTransform)}
${getTransformWGSL('transformTime', props.timeTransform)}`;
    // pair-stats is one pass over the pairs with no permutation axis, so it takes more blocks
    // than the (permutation, block) kernels to fill the device.
    const pairStatBlocks = Math.min(MAXIMUM_PAIR_STAT_BLOCKS, rows);
    const pairCounts = createTransientView(graph, `${id}-pair-counts`, 'uint32', pairStatBlocks);
    // This resource retains its existing ID; each block now stores its first selected pair's
    // transformed distance/time anchor, rather than sums that introduce a fractional mean.
    const pairSums = createTransientView(graph, `${id}-pair-sums`, 'float32', pairStatBlocks * 4);
    const pairCount = createTransientView(graph, `${id}-pair-count`, 'uint32', 1);
    const anchorBlocks = createTransientView(
      graph,
      `${id}-anchor-blocks`,
      'uint32',
      pairStatBlocks
    );
    const firstAnchor = createTransientView(graph, `${id}-first-anchor`, 'uint32', 1);
    const pairMask = createTransientView(
      graph,
      `${id}-pair-mask`,
      'uint32',
      pairs.neighbors.length
    );
    // Stopping keeps the centring shift in the state words (as float32 bits): the partials kernel
    // is at the 8 storage bindings limit with the state binding.
    const shift = stopping ? undefined : createTransientView(graph, `${id}-shift`, 'float32', 4);
    const partials = createTransientView(
      graph,
      `${id}-partials`,
      'float32',
      // The permutation-invariant spatial moments (2 words per block) follow the partials, which
      // keeps the partials kernel at 8 storage bindings.
      tableSlots * blocks * PARTIAL_STRIDE + blocks * 4
    );
    const residualStatistics = createTransientView(
      graph,
      `${id}-residual-statistics`,
      'float32',
      statistics.length * 2
    );
    const pairBindings = [
      {name: 'offsets', view: pairs.offsets, type: 'u32' as const, access: 'read' as const},
      {name: 'neighbors', view: pairs.neighbors, type: 'u32' as const, access: 'read' as const},
      {name: 'pairMask', view: pairMask, type: 'u32' as const, access: 'read' as const},
      {name: 'distances', view: distances, type: 'f32' as const, access: 'read' as const},
      {name: 'times', view: times, type: 'f32' as const, access: 'read' as const}
    ];
    const createPartialsNode = (batch?: SpaceTimeBatchNodeProps<Parameters>) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: batch?.id ?? `${id}-partials`,
        operation: OPERATION,
        variant: 'partials',
        modules: [fp64arithmetic],
        bindings: [
          ...pairBindings,
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          // Stopping reads the shift from the state words; otherwise a separate view.
          ...(stopping
            ? stateBindings
            : [{name: 'shift', view: shift!, type: 'f32' as const, access: 'read' as const}]),
          {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
        ],
        condition: batch?.condition,
        extraResources: batch?.extraResources,
        workgroupSize: SPACE_TIME_WORKGROUP_SIZE,
        invocationCount: tableSlots * groupsPerSlot * SPACE_TIME_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${common}
${getSpaceTimeShuffleWGSL(blocks, PHILOX_PURPOSES.mantelTestShuffle.id)}
const SLOT_COUNT: u32 = ${tableSlots}u;${batchDeclarations}
const MOMENTS_BASE: u32 = partialsOffset + ${tableSlots * blocks * PARTIAL_STRIDE}u;`,
        // One permutation per workgroup with interleaved rows (see the Knox partials). Slot 0 also
        // writes the permutation-invariant moments. With stopping, the slot is the batch cursor (a
        // GPU word) times the batch size plus the workgroup's slot in the batch.
        body: `${stopping ? SPACE_TIME_BATCH_PROLOGUE_WGSL : SPACE_TIME_SLOT_PROLOGUE_WGSL}
  var sumX = vec2f(0.0);
  var sumXX = vec2f(0.0);
  var sumY = vec2f(0.0);
  var sumYY = vec2f(0.0);
  var sumXY = vec2f(0.0);
  var minimumDistance = 1.0;
  var maximumDistance = -1.0;
  var timeAtMinimum = 0.0;
  var timeAtMaximum = 0.0;
  var minimumTime = 1.0;
  var maximumTime = -1.0;
  if (runsSlot) {
    ${
      stopping
        ? `let shiftDistance = bitcast<f32>(state[stateOffset + ${SPACE_TIME_STATE.shiftDistance}u]);
    let anchorFirst = state[stateOffset + 6u];
    let anchorSecond = state[stateOffset + 7u];`
        : `let shiftDistance = shift[shiftOffset];
    let anchorFirst = u32(shift[shiftOffset + 2u]);
    let anchorSecond = u32(shift[shiftOffset + 3u]);`
    }
    // Center each draw on its own anchor gap, not the observed draw's gap.
    let shiftTime = transformTime(abs(times[timesOffset + getPermutedRow(anchorFirst, slot)] - times[timesOffset + getPermutedRow(anchorSecond, slot)]));
    for (var row = block; row < ROWS; row += BLOCKS) {
      let time = times[timesOffset + getPermutedRow(row, slot)];
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        let other = neighbors[neighborsOffset + entry];
        if (pairMask[pairMaskOffset + entry] != 0u) {
          let rawDistance = transformDistance(distances[distancesOffset + entry]);
          let rawTime = transformTime(abs(time - times[timesOffset + getPermutedRow(other, slot)]));
          let firstPair = minimumDistance > maximumDistance;
          if (firstPair || rawTime < minimumTime) { minimumTime = rawTime; }
          if (firstPair || rawTime > maximumTime) { maximumTime = rawTime; }
          if (firstPair || rawDistance < minimumDistance) {
            minimumDistance = rawDistance;
            timeAtMinimum = rawTime;
          }
          if (firstPair || rawDistance > maximumDistance) {
            maximumDistance = rawDistance;
            timeAtMaximum = rawTime;
          }
          let x = sub_fp64(vec2f(rawDistance, 0.0), vec2f(shiftDistance, 0.0));
          let y = sub_fp64(vec2f(rawTime, 0.0), vec2f(shiftTime, 0.0));
          if (slot == 0u) {
            sumX = sum_fp64(sumX, x);
            sumXX = sum_fp64(sumXX, mul_fp64(x, x));
          }
          sumY = sum_fp64(sumY, y);
          sumYY = sum_fp64(sumYY, mul_fp64(y, y));
          sumXY = sum_fp64(sumXY, mul_fp64(x, y));
        }
      }
    }
  }
  if (block < BLOCKS) {
    let base = partialsOffset + (block * SLOT_COUNT + ${stopping ? 'localSlot' : 'slot'}) * ${PARTIAL_STRIDE}u;
    partials[base] = sumY.x;
    partials[base + 1u] = sumY.y;
    partials[base + 2u] = sumYY.x;
    partials[base + 3u] = sumYY.y;
    partials[base + 4u] = sumXY.x;
    partials[base + 5u] = sumXY.y;
    partials[base + 6u] = minimumDistance;
    partials[base + 7u] = maximumDistance;
    partials[base + 8u] = timeAtMinimum;
    partials[base + 9u] = timeAtMaximum;
    partials[base + 10u] = minimumTime;
    partials[base + 11u] = maximumTime;
  }
  if (slot == 0u && block < BLOCKS) {
    partials[MOMENTS_BASE + 4u * block] = sumX.x;
    partials[MOMENTS_BASE + 4u * block + 1u] = sumX.y;
    partials[MOMENTS_BASE + 4u * block + 2u] = sumXX.x;
    partials[MOMENTS_BASE + 4u * block + 3u] = sumXX.y;
  }`
      });
    const createReduceNode = (batch?: SpaceTimeBatchNodeProps<Parameters>) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: batch?.id ?? `${id}-reduce`,
        operation: OPERATION,
        variant: 'reduce',
        modules: [fp64arithmetic],
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          ...stateBindings,
          {name: 'partials', view: partials, type: 'f32', access: 'read'},
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'},
          {name: 'residuals', view: residualStatistics, type: 'f32', access: 'read_write'}
        ],
        condition: batch?.condition,
        extraResources: batch?.extraResources,
        // Stopping reduces the slots of one batch; the arithmetic is the full run's.
        ...(stopping
          ? {workgroupSize: 64, invocationCount: batchSize}
          : {invocationCount: statistics.length}),
        declarations: `${common}
const SLOT_COUNT: u32 = ${tableSlots}u;${batchDeclarations}
const MOMENTS_BASE: u32 = partialsOffset + ${tableSlots * blocks * PARTIAL_STRIDE}u;`,
        body: `${
          stopping
            ? `let slot = state[stateOffset] * BATCH_SIZE + index;
  let localSlot = index;
  let runs = slot <= readPermutationCount();`
            : `let slot = index;
  let localSlot = index;
  let runs = index <= readPermutationCount();`
        }
  var result = 0.0;
  var preciseResult = vec2f(0.0);
  if (runs) {
    result = getNaN();
    preciseResult = vec2f(result, 0.0);
    var sumX = vec2f(0.0);
    var sumXX = vec2f(0.0);
    var sumY = vec2f(0.0);
    var sumYY = vec2f(0.0);
    var sumXY = vec2f(0.0);
    var minimumDistance = 1.0;
    var maximumDistance = -1.0;
    var timeAtMinimum = 0.0;
    var timeAtMaximum = 0.0;
    var minimumTime = 1.0;
    var maximumTime = -1.0;
    for (var block = 0u; block < BLOCKS; block++) {
      let base = partialsOffset + (block * SLOT_COUNT + localSlot) * ${PARTIAL_STRIDE}u;
      sumX = sum_fp64(sumX, vec2f(partials[MOMENTS_BASE + 4u * block], partials[MOMENTS_BASE + 4u * block + 1u]));
      sumXX = sum_fp64(sumXX, vec2f(partials[MOMENTS_BASE + 4u * block + 2u], partials[MOMENTS_BASE + 4u * block + 3u]));
      sumY = sum_fp64(sumY, vec2f(partials[base], partials[base + 1u]));
      sumYY = sum_fp64(sumYY, vec2f(partials[base + 2u], partials[base + 3u]));
      sumXY = sum_fp64(sumXY, vec2f(partials[base + 4u], partials[base + 5u]));
      let blockMinimum = partials[base + 6u];
      let blockMaximum = partials[base + 7u];
      if (blockMinimum <= blockMaximum) {
        let firstPair = minimumDistance > maximumDistance;
        if (firstPair || partials[base + 10u] < minimumTime) { minimumTime = partials[base + 10u]; }
        if (firstPair || partials[base + 11u] > maximumTime) { maximumTime = partials[base + 11u]; }
        if (firstPair || blockMinimum < minimumDistance) {
          minimumDistance = blockMinimum;
          timeAtMinimum = partials[base + 8u];
        }
        if (firstPair || blockMaximum > maximumDistance) {
          maximumDistance = blockMaximum;
          timeAtMaximum = partials[base + 9u];
        }
      }
    }
    let count = f32(pairCount[pairCountOffset]);
    // Scale the integer moment identity by a power of two: no fractional centering
    // mean, and no multiplication of an already large second moment by count.
    let countExponent = frexp(count).exp;
    let scaledCount = vec2f(ldexp(count, -countExponent), 0.0);
    let scaledX = vec2f(ldexp(sumX.x, -countExponent), ldexp(sumX.y, -countExponent));
    let scaledY = vec2f(ldexp(sumY.x, -countExponent), ldexp(sumY.y, -countExponent));
    let covariance = sub_fp64(mul_fp64(scaledCount, sumXY), mul_fp64(scaledX, sumY));
    let varianceX = sub_fp64(mul_fp64(scaledCount, sumXX), mul_fp64(scaledX, sumX));
    let varianceY = sub_fp64(mul_fp64(scaledCount, sumYY), mul_fp64(scaledY, sumY));
    if (count == 2.0) {
      // Compare raw transformed values before any cancellation-prone variance guard.
      if (minimumDistance < maximumDistance && timeAtMinimum != timeAtMaximum) {
        result = select(-1.0, 1.0, timeAtMaximum > timeAtMinimum);
        preciseResult = vec2f(result, 0.0);
      }
    } else if (count > 1.0 && minimumTime < maximumTime && varianceX.x > 0.0 && varianceY.x > 0.0) {
      // Keep both words through division and sqrt; normalize variance exponents before
      // multiplication so the denominator product cannot overflow.
      let firstExponent = frexp(varianceX.x).exp;
      let secondExponent = frexp(varianceY.x).exp;
      let exponentSum = firstExponent + secondExponent;
      let halfExponent = exponentSum / 2;
      let firstScaled = vec2f(ldexp(varianceX.x, -firstExponent), ldexp(varianceX.y, -firstExponent));
      let secondScaled = vec2f(ldexp(varianceY.x, -secondExponent), ldexp(varianceY.y, -secondExponent));
      let product = mul_fp64(firstScaled, secondScaled);
      let remainder = exponentSum - 2 * halfExponent;
      let denominator = sqrt_fp64(vec2f(ldexp(product.x, remainder), ldexp(product.y, remainder)));
      if (covariance.x == 0.0 && covariance.y == 0.0) {
        preciseResult = vec2f(0.0);
      } else {
        preciseResult = div_fp64(vec2f(ldexp(covariance.x, -halfExponent), ldexp(covariance.y, -halfExponent)), denominator);
        if (preciseResult.x > 1.0 || (preciseResult.x == 1.0 && preciseResult.y > 0.0)) { preciseResult = vec2f(1.0, 0.0); }
        if (preciseResult.x < -1.0 || (preciseResult.x == -1.0 && preciseResult.y < 0.0)) { preciseResult = vec2f(-1.0, 0.0); }
      }
      result = preciseResult.x;
    }
  }
  ${stopping ? 'if (runs) {' : ''}
  statistics[statisticsOffset + slot] = result;
  residuals[residualsOffset + 2u * slot] = preciseResult.x;
  residuals[residualsOffset + 2u * slot + 1u] = preciseResult.y;
  ${stopping ? '}' : ''}`
      });
    const createComparisonNode = (batch: SpaceTimeBatchNodeProps<Parameters>) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: batch.id,
        operation: OPERATION,
        variant: 'stop-compare',
        condition: batch.condition,
        extraResources: batch.extraResources,
        bindings: [
          {name: 'statistics', view: residualStatistics, type: 'f32', access: 'read'},
          {name: 'comparison', view: comparisonStatistics!, type: 'f32', access: 'read_write'},
          {name: 'state', view: state!, type: 'u32', access: 'read'}
        ],
        invocationCount: batchSize,
        workgroupSize: 64,
        declarations: `const BATCH_SIZE: u32 = ${batchSize}u;
fn isFiniteCorrelation(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }`,
        body: `let slot = state[stateOffset] * BATCH_SIZE + index;
  if (slot >= ${statistics.length}u) { return; }
  let observed = vec2f(statistics[statisticsOffset], statistics[statisticsOffset + 1u]);
  let value = vec2f(statistics[statisticsOffset + 2u * slot], statistics[statisticsOffset + 2u * slot + 1u]);
  var exceeds = false;
  if (isFiniteCorrelation(observed.x) && isFiniteCorrelation(value.x)) { exceeds = value.x > observed.x || (value.x == observed.x && value.y >= observed.y); }
  comparison[comparisonOffset + slot] = select(0.0, 1.0, slot == 0u || exceeds);`
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
            statisticType: 'f32',
            parameters,
            statistics,
            comparisonStatistics,
            createComparisonNode,
            state,
            common,
            createPartialsNode,
            createReduceNode
          })
        : [];
    return [
      createSpaceTimePairMaskNode<Parameters>(graph, {
        id,
        operation: OPERATION,
        rows,
        offsets: pairs.offsets,
        neighbors: pairs.neighbors,
        pairMask
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pair-stats`,
        operation: OPERATION,
        variant: 'pair-stats',
        bindings: [
          ...pairBindings,
          {name: 'pairCounts', view: pairCounts, type: 'u32', access: 'read_write'},
          {name: 'pairSums', view: pairSums, type: 'f32', access: 'read_write'},
          {name: 'anchorBlocks', view: anchorBlocks, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairStatBlocks,
        declarations: `const ROWS: u32 = ${rows}u;
const PAIR_STAT_BLOCKS: u32 = ${pairStatBlocks}u;
${getTransformWGSL('transformDistance', props.distanceTransform)}
${getTransformWGSL('transformTime', props.timeTransform)}`,
        // Rows interleave across blocks, so neighboring invocations read neighboring CSR rows.
        body: `var count = 0u;
  var sumDistance = 0.0;
  var sumTime = 0.0;
  var anchorFirst = 0u;
  var anchorSecond = 0u;
  for (var row = index; row < ROWS; row += PAIR_STAT_BLOCKS) {
    for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
      let other = neighbors[neighborsOffset + entry];
      if (pairMask[pairMaskOffset + entry] != 0u) {
        // A data-valued anchor keeps integral differences exact and avoids introducing
        // fractional mean rounding before permutation-invariant moments are accumulated.
        if (count == 0u) {
          anchorFirst = row;
          anchorSecond = other;
          sumDistance = transformDistance(distances[distancesOffset + entry]);
          sumTime = transformTime(abs(times[timesOffset + row] - times[timesOffset + other]));
        }
        count++;
      }
    }
  }
  pairCounts[pairCountsOffset + index] = count;
  anchorBlocks[anchorBlocksOffset+index]=select(0xffffffffu,index,count>0u);
  pairSums[pairSumsOffset + 2u * index] = sumDistance;
  pairSums[pairSumsOffset + 2u * index + 1u] = sumTime;
  pairSums[pairSumsOffset + ${2 * pairStatBlocks}u + 2u * index] = f32(anchorFirst);
  pairSums[pairSumsOffset + ${2 * pairStatBlocks}u + 2u * index + 1u] = f32(anchorSecond);`
      }),
      ...new GPUReduction({
        id: `${id}-pair-count-reduction`,
        input: pairCounts,
        output: pairCount,
        operation: 'sum'
      }).getCommandNodes(graph),
      ...new GPUReduction({
        id: `${id}-first-anchor-reduction`,
        input: anchorBlocks,
        output: firstAnchor,
        operation: 'min'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-shift`,
        operation: OPERATION,
        variant: 'shift',
        bindings: [
          {name: 'pairSums', view: pairSums, type: 'f32', access: 'read'},
          {name: 'firstAnchor', view: firstAnchor, type: 'u32', access: 'read'},
          stopping && state
            ? {name: 'state', view: state, type: 'u32', access: 'read_write'}
            : {name: 'shift', view: shift!, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const BLOCKS: u32 = ${pairStatBlocks}u;`,
        body: `var sumDistance = 0.0;
  var sumTime = 0.0;
  let block=firstAnchor[firstAnchorOffset];
  if(block<BLOCKS) {
    sumDistance = pairSums[pairSumsOffset + 2u * block];
    sumTime = pairSums[pairSumsOffset + 2u * block + 1u];
  }
${
  stopping
    ? `state[stateOffset + ${SPACE_TIME_STATE.shiftDistance}u] = bitcast<u32>(sumDistance);
  state[stateOffset + ${SPACE_TIME_STATE.shiftTime}u] = bitcast<u32>(sumTime);`
    : `shift[shiftOffset] = sumDistance;
  shift[shiftOffset + 1u] = sumTime;`
}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-anchor-rows`,
        operation: OPERATION,
        variant: 'anchor-rows',
        bindings: [
          {name: 'pairSums', view: pairSums, type: 'f32', access: 'read'},
          {name: 'firstAnchor', view: firstAnchor, type: 'u32', access: 'read'},
          stopping && state
            ? {name: 'state', view: state, type: 'u32', access: 'read_write'}
            : {name: 'shift', view: shift!, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 2,
        body: `let block = firstAnchor[firstAnchorOffset];
  var row = 0.0;
  if (block < ${pairStatBlocks}u) { row = pairSums[pairSumsOffset + ${2 * pairStatBlocks}u + 2u * block + index]; }
  ${stopping ? 'state[stateOffset + 6u + index] = u32(row);' : 'shift[shiftOffset + 2u + index] = row;'}`
      }),
      ...(stopping ? stoppingNodes : [createPartialsNode(), createReduceNode()]),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
          {name: 'summary', view: summary, type: 'f32', access: 'read_write'},
          ...stateBindings,
          ...(stopping && props.output.permutationCount
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
        body: `let observedPairs = f32(pairCount[pairCountOffset]);
  let timeClosePairs = 0.0;
  let expected = 0.0;
  ${getSpaceTimeSummaryBodyWGSL('f32', stopAfterExceedances)}`
      }),
      ...createMantelResidualPublicationNodes(graph, {
        residualStatistics,
        id,
        maximumPermutations,
        stopAfterExceedances,
        parameters,
        statistics,
        summary,
        state,
        permutationCount: props.output.permutationCount,
        usablePermutationCount: props.output.usablePermutationCount
      }),
      ...createSpatialWeightsSuppressionNodes(
        graph,
        id,
        OPERATION,
        props.pairs,
        {
          statistics: props.output.statistics,
          summary: props.output.summary,
          permutationCount: props.output.permutationCount,
          usablePermutationCount: props.output.usablePermutationCount
        },
        {
          summary: GPU_SPACE_TIME_SUMMARY_LENGTH,
          permutationCount: 1,
          usablePermutationCount: 1,
          statistics: props.maximumPermutations + 1
        }
      ),
      // The shared suppression has no usable-count field: zero it beside the attempted count,
      // so an incomplete neighborhood never reports usable draws without attempted ones.
      ...(props.output.usablePermutationCount
        ? createSpatialWeightsSuppressionNodes(
            graph,
            `${id}-usable`,
            OPERATION,
            props.pairs,
            {permutationCount: props.output.usablePermutationCount},
            {permutationCount: 1}
          )
        : [])
    ];
  }
}
