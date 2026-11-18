// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUCandidateStatusPort,
  type GPUInvalidStatusPort,
  type GPUIterationStatusPort,
  type GPUApproximationStatusPort,
  type GPUSpatialContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/index';
import {
  getStorageBindingElementLimit,
  validateGPUStorageBindingByteLength,
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateTrajectoryBindings,
  validateTrajectoryWritableDispatches
} from '../trajectory-analysis/trajectory-binding-validation';
import {GET_ORDERED_KEY_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getRegressionTileLayout,
  getRowTileSumDispatchShapes,
  getRowTileSumNodes,
  REGRESSION_TILE_MAXIMUM_ROW_COUNT
} from '../spatial-regression/regression-row-tiles';
import {resolveGPUSpaceTimeTemporalContext} from '../space-time-tests/space-time-bands';
import {
  GPU_SPACE_TIME_HAWKES_ESTIMATE_LENGTH,
  GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH
} from './space-time-hawkes-parameters';
import {
  getSpaceTimeHawkesEventWGSL,
  SPACE_TIME_HAWKES_COMPENSATOR_WGSL,
  GPU_SPACE_TIME_HAWKES_BRANCHING_RATIO_FLOOR,
  GPU_SPACE_TIME_HAWKES_INTENSITY_FLOOR,
  GPU_SPACE_TIME_HAWKES_MINIMUM_SPATIAL_VARIANCE,
  SPACE_TIME_HAWKES_STATE_LENGTH,
  SPACE_TIME_HAWKES_STATE_SLOTS,
  SPACE_TIME_HAWKES_SUM_COUNT
} from './space-time-hawkes-wgsl';

const OPERATION = 'GPUSpaceTimeHawkes';

/** Sort key of a row the fit ignores: above the ordered key of every finite float32. */
const IGNORED_ROW_KEY = 0xffffffff;

/** Default number of EM iterations that are unrolled into the graph. */
export const GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS = 100;

/**
 * Largest unrolled iteration count. Each iteration is three command nodes, so this keeps the graph
 * far below the shared command node budget.
 */
export const GPU_SPACE_TIME_HAWKES_MAXIMUM_ITERATIONS = 1000;

/** Default capacity of the candidate parents examined per event. */
export const GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_PARENTS = 256;

/**
 * Largest parent capacity. One thread walks this many parents per event and iteration; the limit
 * keeps a single event's work bounded and is the capacity named in the review (K up to 256 there,
 * 4096 here for dense catalogues).
 */
export const GPU_SPACE_TIME_HAWKES_MAXIMUM_PARENTS = 4096;

/**
 * Allowance for the command nodes of the published `GPUSort`, whose count is not exposed; a test
 * checks that the real count stays below the estimate.
 */
const SORT_COMMAND_NODE_ALLOWANCE = 64;

/** Command nodes outside the sort and the iterations: keys, gather, initialize, responsibilities, two publications. */
const FIXED_COMMAND_NODE_COUNT = 6;

/** Caller-owned outputs of {@link GPUSpaceTimeHawkes}. */
export type GPUSpaceTimeHawkesOutput = {
  /**
   * Four float32 rows: background rate `mu` (events per unit area and time unit), branching ratio
   * `omega` (expected offspring per event), time-kernel rate `beta` (per time unit) and the
   * standard deviation `sigma` of the Gaussian offspring displacement (position units). All NaN
   * when the fit could not start (see `status`).
   */
  estimates: GraphDataView<'float32'>;
  /**
   * Optional one-row log-likelihood of the catalogue at the parameters that entered the last
   * executed iteration, before its update. Use `fittedLogLikelihood` to compare fitted models.
   * The compensator uses the whole-plane spatial mass (no spatial edge term).
   */
  logLikelihood?: GraphDataView<'float32'>;
  /** Optional one-row log-likelihood at the published estimates, using the whole-plane spatial mass. */
  fittedLogLikelihood?: GraphDataView<'float32'>;
  /**
   * Optional per-event probability, in input row order, that the event is background (not
   * triggered), at the published estimates. NaN for an excluded row.
   */
  backgroundProbability?: GraphDataView<'float32'>;
  /** Optional diagnostic scalars. */
  status?: GPUIterationStatusPort &
    GPUCandidateStatusPort &
    GPUInvalidStatusPort &
    GPUApproximationStatusPort;
};

/**
 * Properties for {@link GPUSpaceTimeHawkes}.
 *
 * Per-frame (no recompile): the contents of `parameters` (window, parent bounds, tolerance,
 * starting values) and of `positions` and `times`. Compile-time (needs a new graph): the
 * event count, `maximumIterations`, `maximumParents`, and which optional outputs are present.
 */
export type GPUSpaceTimeHawkesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'space-time-hawkes'`. */
  id?: string;
  /** Coordinate and metric contract of `positions`: planar/native only. */
  spatialContext?: GPUSpatialContext;
  /**
   * Optional declaration of `times`: encoding `float32-relative`, counted from a declared origin in
   * the unit of the time kernel. Checked at construction; nothing is converted.
   */
  temporalContext?: GPUTemporalContext;
  /**
   * Packed event positions. Coordinates are float32: subtract a nearby origin in float64 before
   * upload when they are large (UTM metres), because differences of large float32 values lose
   * their low bits.
   */
  positions: GraphDataView<'float32x2'>;
  /**
   * Event time per row, float32, relative to the start of the observation period, in the unit of
   * the time kernel and in `[0, windowDuration]`. Rows with another value are excluded.
   */
  times: GraphDataView<'float32'>;
  /**
   * Per-frame float32 view of at least {@link GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH} values
   * written with `getGPUSpaceTimeHawkesParameterValues`.
   * The squared starting spatial deviation must be finite normal float32 (at least
   * {@link GPU_SPACE_TIME_HAWKES_MINIMUM_SPATIAL_VARIANCE}); otherwise the fit is invalid.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Number of EM iterations unrolled into the graph, `1` to
   * {@link GPU_SPACE_TIME_HAWKES_MAXIMUM_ITERATIONS}. Defaults to
   * {@link GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS}.
   */
  maximumIterations?: number;
  /**
   * Parent capacity: for each event the most recent `maximumParents` strictly earlier events inside the lag
   * bound are examined, and an event with more candidates is counted in
   * `output.status.candidateOverflow`. Defaults to
   * {@link GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_PARENTS}.
   */
  maximumParents?: number;
  /** Caller-owned outputs. */
  output: GPUSpaceTimeHawkesOutput;
};

/**
 * Space-time Hawkes process fitted by expectation-maximization with bounded parent windows
 * (Veen and Schoenberg 2008 for the EM; the exponential-Gaussian kernel is the fixed-kernel
 * special case of the ETAS family, without magnitudes, which are later work).
 *
 * Model: the conditional intensity at time `t` and place `s` is
 * `lambda(s, t) = mu + sum_{t_i < t} omega beta exp(-beta (t - t_i)) / (2 pi sigma^2) exp(-|s - s_i|^2 / (2 sigma^2))`
 * with a background rate `mu` that is constant in space and time, `omega` the expected number of
 * offspring per event, an exponential time kernel with rate `beta` and an isotropic Gaussian
 * offspring displacement. Each iteration computes, for every event `j`, the responsibilities
 * `p_jj = mu / lambda_j` and `p_ij = (offspring term of i) / lambda_j` (E-step) and updates
 * `mu = sum_j p_jj / (A T)`, `omega = sum_j (1 - p_jj) / sum_i (1 - exp(-beta (T - t_i)))`,
 * `beta = sum p_ij / sum p_ij lag_ij` and `sigma^2 = sum p_ij d_ij^2 / (2 sum p_ij)` (M-step), where
 * `A` is the window area and `T` the duration. Offspring that would fall after `T` are accounted
 * for in `omega`; offspring that would fall outside the spatial window are not, so the spatial
 * kernel is treated as having its whole mass in the window (keep `sigma` small against the window).
 * Pairs beyond `maximumLag` or `maximumDistance` are dropped and the kernel is not renormalized, so
 * choose them at several `1 / beta` and several `sigma`. Simultaneous events (equal times) do not
 * trigger each other.
 *
 * Implementation: events are stably sorted by time (radix sort of the order-preserving float key)
 * into a table of position, time. For each event a thread binary-searches the first candidate
 * parent inside the lag bound and walks the candidates up to `maximumParents` back from the event,
 * the ordered pair enumeration of a sorted list as in the Knox tests. The intensity is accumulated
 * in the log domain in a single pass with a running maximum, so the sum of terms of very different
 * size does not underflow (log-sum-exp, Ko, Suchard and Holbrook 2024), and the floor
 * {@link GPU_SPACE_TIME_HAWKES_INTENSITY_FLOOR} keeps the logarithms of the background rate and the
 * branching ratio finite. Event statistics are summed in tiles of consecutive events in plain
 * float32 and the tile partials by a fixed pairwise (cascade) sum (the `GPUOrdinaryLeastSquares`
 * pattern, `getRowTileSumNodes`), whose error is about `log2(tileCount)` roundings of the totals,
 * so results are bitwise repeatable on a device. (The merge is not a compensated sum: a float
 * Neumaier term is removed by fast-math shader compilers.) `maximumIterations` iterations are
 * unrolled; a one-thread node tests convergence (largest relative change of any estimate at most
 * `tolerance`) and disarms a GPU dispatch gate, so finished iterations launch nothing. If the budget is
 * exhausted first the latest estimates are still published with `iterationLimitReached` set.
 *
 * Cost per iteration is `events * min(maximumParents, candidates in the lag bound)`; memory is
 * 16 bytes per event plus the sort. Event counts are limited to 2^24 (float32 counts) and by the
 * 16-byte event table binding.
 */
export class GPUSpaceTimeHawkes implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpaceTimeHawkesProps;
  /** Resolved coordinate and metric contract (planar/native). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved declaration of `times`, when the props gave one. */
  readonly temporalContext?: GPUTemporalContext;
  /** Unrolled iteration count. */
  readonly maximumIterations: number;
  /** Parent capacity per event. */
  readonly maximumParents: number;
  /** Upper bound of the command nodes `getCommandNodes` returns. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUSpaceTimeHawkesProps) {
    this.id = props.id ?? 'space-time-hawkes';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'area'
    });
    this.temporalContext = resolveGPUSpaceTimeTemporalContext(id, props.temporalContext);
    this.maximumIterations =
      props.maximumIterations ?? GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS;
    this.maximumParents = props.maximumParents ?? GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_PARENTS;
    validateCapacity(
      id,
      'maximumIterations',
      this.maximumIterations,
      1,
      GPU_SPACE_TIME_HAWKES_MAXIMUM_ITERATIONS
    );
    validateCapacity(
      id,
      'maximumParents',
      this.maximumParents,
      1,
      GPU_SPACE_TIME_HAWKES_MAXIMUM_PARENTS
    );
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    validateSinglePackedView(props.times, ['float32'], `${id} times`);
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRequirement(id, props.positions.length >= 2, 'needs at least two events');
    // Counts are summed in float32, which is exact for integers up to 2^24.
    validateMaximumCount(
      id,
      'positions length',
      props.positions.length,
      REGRESSION_TILE_MAXIMUM_ROW_COUNT
    );
    validateRequirement(
      id,
      props.times.length === props.positions.length,
      'times length must equal positions length'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH,
      `parameters must hold ${GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH} float32 values`
    );
    const {output} = props;
    validateSinglePackedView(output.estimates, ['float32'], `${id} output.estimates`);
    validateRequirement(
      id,
      output.estimates.length >= GPU_SPACE_TIME_HAWKES_ESTIMATE_LENGTH,
      `output.estimates must hold ${GPU_SPACE_TIME_HAWKES_ESTIMATE_LENGTH} rows`
    );
    if (output.logLikelihood) {
      validateSinglePackedView(output.logLikelihood, ['float32'], `${id} output.logLikelihood`);
      validateRequirement(
        id,
        output.logLikelihood.length === 1,
        'output.logLikelihood must contain exactly one row'
      );
    }
    if (output.fittedLogLikelihood) {
      validateSinglePackedView(
        output.fittedLogLikelihood,
        ['float32'],
        `${id} output.fittedLogLikelihood`
      );
      validateRequirement(
        id,
        output.fittedLogLikelihood.length === 1,
        'output.fittedLogLikelihood must contain exactly one row'
      );
    }
    if (output.backgroundProbability) {
      validateSinglePackedView(
        output.backgroundProbability,
        ['float32'],
        `${id} output.backgroundProbability`
      );
      validateRequirement(
        id,
        output.backgroundProbability.length >= props.positions.length,
        'output.backgroundProbability must hold one row per event'
      );
    }
    for (const [name, view] of Object.entries(output.status ?? {})) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
        validateRequirement(
          id,
          view.length === 1,
          `output.status.${name} must contain exactly one row`
        );
      }
    }
    const outputViews = this.getOutputViews();
    validateTrajectoryWritableDispatches(id, [
      {
        estimates: output.estimates,
        logLikelihood: output.logLikelihood,
        fittedLogLikelihood: output.fittedLogLikelihood
      },
      {backgroundProbability: output.backgroundProbability},
      {...output.status}
    ]);
    validateGraphOutputsDisjointFromInputs(id, outputViews, [
      props.positions,
      props.times,
      props.parameters
    ]);
    this.estimatedCommandNodeCount =
      FIXED_COMMAND_NODE_COUNT +
      SORT_COMMAND_NODE_ALLOWANCE +
      3 * this.maximumIterations +
      (output.fittedLogLikelihood ? 2 : 0);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  private getOutputViews(): GraphDataView[] {
    const {output} = this.props;
    return [
      output.estimates,
      output.logLikelihood,
      output.fittedLogLikelihood,
      output.backgroundProbability,
      ...Object.values(output.status ?? {})
    ].filter(view => view !== undefined);
  }

  /**
   * Returns, in order: the row keys, the time sort, the sorted event table, the initialization, the
   * unrolled iterations (event sums, merge, update), the optional responsibilities and the
   * publication of estimates and status.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumIterations, maximumParents} = this;
    const {output} = props;
    const rowCount = props.positions.length;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.times,
      props.parameters,
      ...this.getOutputViews()
    ]);
    // The event table holds four floats per row in one binding.
    validateMaximumCount(
      id,
      'positions length',
      rowCount,
      getStorageBindingElementLimit(graph.device, 16)
    );
    const layout = getRegressionTileLayout(rowCount);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'state',
      SPACE_TIME_HAWKES_STATE_LENGTH * 4,
      'use a device with storage bindings'
    );
    for (const [name, view] of Object.entries({
      positions: props.positions,
      times: props.times,
      parameters: props.parameters,
      'output.estimates': output.estimates,
      'output.logLikelihood': output.logLikelihood,
      'output.backgroundProbability': output.backgroundProbability,
      ...Object.fromEntries(
        Object.entries(output.status ?? {}).map(([name, view]) => [`output.status.${name}`, view])
      )
    })) {
      if (view) {
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          name,
          getViewBindingRange(view).size,
          'reduce the view binding range'
        );
      }
    }
    validateTrajectoryBindings(graph, id, props);
    // Include sort histograms and all row-proportional intermediates before graph mutation.
    for (const [name, byteLength] of [
      ['event table', rowCount * 16],
      ['sort rows', rowCount * 4],
      ['sort histograms', 16 * Math.ceil(rowCount / 256) * 4],
      ['event partials', layout.tileCount * SPACE_TIME_HAWKES_SUM_COUNT * 4],
      ['iteration dispatch', 6 * 4]
    ] as const) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'reduce positions length'
      );
    }
    const slots = SPACE_TIME_HAWKES_STATE_SLOTS;
    const state = createTransientView(
      graph,
      `${id}-state`,
      'float32',
      SPACE_TIME_HAWKES_STATE_LENGTH
    );
    const nodes: GPUCommandNode<Parameters>[] = [];

    // 1. Row keys: the ordered time key of an included row, the ignored key otherwise.
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', rowCount);
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rowCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-row-keys`,
        operation: OPERATION,
        variant: 'row-keys',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'times', view: props.times, type: 'f32', access: 'read'},
          {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
          {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: `${IS_FINITE_VALUE_WGSL}\n${GET_ORDERED_KEY_WGSL}`,
        body: `let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let time = times[timesOffset + index];
  let included = isFiniteValue(position.x) && isFiniteValue(position.y) &&
    isFiniteValue(time) && time >= 0.0 && time <= params[paramsOffset + 1u];
  keys[keysOffset + index] = select(${IGNORED_ROW_KEY}u, getOrderedKey(time), included);
  rowIds[rowIdsOffset + index] = index;`
      })
    );

    // 2. Stable time sort of the row IDs (ties keep row order).
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', rowCount);
    const timeOrder = createTransientView(graph, `${id}-time-order`, 'uint32', rowCount);
    nodes.push(
      ...new GPUSort({
        id: `${id}-time-sort`,
        keys,
        values: rowIds,
        outputKeys: sortedKeys,
        outputValues: timeOrder,
        keyBits: 32
      }).getCommandNodes(graph)
    );

    // 3. Sorted event table: [x, y, time, 0]; an excluded row has a NaN time.
    const events = createTransientView(graph, `${id}-events`, 'float32', 4 * rowCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sorted-events`,
        operation: OPERATION,
        variant: 'sorted-events',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'times', view: props.times, type: 'f32', access: 'read'},
          {name: 'timeOrder', view: timeOrder, type: 'u32', access: 'read'},
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'events', view: events, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations:
          'fn getNaNBits() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }',
        body: `let row = timeOrder[timeOrderOffset + index];
  let kept = sortedKeys[sortedKeysOffset + index] != ${IGNORED_ROW_KEY}u;
  let nan = getNaNBits();
  events[eventsOffset + 4u * index] = select(nan, positions[positionsOffset + 2u * row], kept);
  events[eventsOffset + 4u * index + 1u] = select(nan, positions[positionsOffset + 2u * row + 1u], kept);
  events[eventsOffset + 4u * index + 2u] = select(nan, times[timesOffset + row], kept);
  events[eventsOffset + 4u * index + 3u] = 0.0;`
      })
    );

    // 4. Initialization from the parameters, and arming of the iteration gate.
    const termCount = SPACE_TIME_HAWKES_SUM_COUNT;
    const partial = createTransientView(
      graph,
      `${id}-partials`,
      'float32',
      layout.tileCount * termCount
    );
    const [tileShape, mergeShape] = getRowTileSumDispatchShapes(layout, termCount);
    const gate = createWGSLDispatchGate(graph, {
      id: `${id}-iteration-dispatch`,
      operation: OPERATION,
      shapes: [tileShape, mergeShape]
    });
    const stateConstants = Object.entries(slots)
      .map(
        ([name, slot]) =>
          `const S_${name.replace(/[A-Z]/g, letter => `_${letter}`).toUpperCase()}: u32 = ${slot}u;`
      )
      .join('\n');
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize`,
        operation: OPERATION,
        variant: 'initialize',
        invocationCount: 1,
        bindings: [
          {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'state', view: state, type: 'f32', access: 'read_write'},
          gate.binding
        ],
        declarations: `${stateConstants}\n${IS_FINITE_VALUE_WGSL}\n${getNaNDeclaration()}
const MINIMUM_SPATIAL_VARIANCE: f32 = ${getWGSLFloatLiteral(GPU_SPACE_TIME_HAWKES_MINIMUM_SPATIAL_VARIANCE)};`,
        body: `let area = params[paramsOffset];
  let duration = params[paramsOffset + 1u];
  let lag = params[paramsOffset + 2u];
  let distance = params[paramsOffset + 3u];
  let tolerance = params[paramsOffset + 4u];
  let background = params[paramsOffset + 5u];
  let branchingRatio = params[paramsOffset + 6u];
  let decayRate = params[paramsOffset + 7u];
  let deviation = params[paramsOffset + 8u];
  let variance = deviation * deviation;
  let valid = area > 0.0 && duration > 0.0 && lag > 0.0 && distance > 0.0 && tolerance > 0.0 &&
    background > 0.0 && branchingRatio > 0.0 && decayRate > 0.0 && deviation > 0.0 &&
    isFiniteValue(area) && isFiniteValue(duration) && isFiniteValue(lag) && isFiniteValue(distance) &&
    isFiniteValue(tolerance) && isFiniteValue(background) && isFiniteValue(branchingRatio) && isFiniteValue(decayRate) && isFiniteValue(deviation) &&
    isFiniteValue(variance) && variance >= MINIMUM_SPATIAL_VARIANCE;
  let nan = getNaN();
  state[stateOffset + S_BACKGROUND] = select(nan, background, valid);
  state[stateOffset + S_BRANCHING_RATIO] = select(nan, branchingRatio, valid);
  state[stateOffset + S_DECAY_RATE] = select(nan, decayRate, valid);
  state[stateOffset + S_SPATIAL_VARIANCE] = select(nan, variance, valid);
  state[stateOffset + S_DONE] = select(1.0, 0.0, valid);
  state[stateOffset + S_ITERATION_COUNT] = 0.0;
  state[stateOffset + S_CONVERGED] = 0.0;
  state[stateOffset + S_ITERATION_LIMIT_REACHED] = 0.0;
  state[stateOffset + S_LOG_LIKELIHOOD] = nan;
  // Reused compiled graphs must not publish statistics from a previous valid fit.
  ${Array.from({length: SPACE_TIME_HAWKES_SUM_COUNT}, (_, term) => `state[stateOffset + S_SUMS + ${term}u] = 0.0;`).join('\n  ')}
  ${gate.disarmWGSL}
  if (valid) {
    ${gate.armWGSL}
  }`
      })
    );

    // 5. The unrolled iterations.
    const eventWGSL = getSpaceTimeHawkesEventWGSL(maximumParents);
    const tileBindings: WGSLKernelBinding[] = [
      {name: 'events', view: events, type: 'f32', access: 'read'},
      {name: 'state', view: state, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'}
    ];
    const sumsSlot = slots.sums;
    for (let iteration = 0; iteration < maximumIterations; iteration++) {
      nodes.push(
        ...getRowTileSumNodes<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-sums`,
          operation: OPERATION,
          variant: 'event-sums',
          rowCount,
          layout,
          sumCount: termCount,
          bindings: tileBindings,
          declarations: `${stateConstants}\n${IS_FINITE_VALUE_WGSL}\n${eventWGSL}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  let time = events[eventsOffset + 4u * row + 2u];
  if (!isFiniteValue(time)) {
    return;
  }
  let statistics = evaluateEvent(row);
  let decayRate = state[stateOffset + S_DECAY_RATE];
  let exponent = decayRate * (params[paramsOffset + 1u] - time);
  // 1 - exp(-x), by its series for small x where the difference would cancel.
  let windowMass = select(1.0 - exp(-exponent), exponent - 0.5 * exponent * exponent, exponent < 1e-3);
  (*terms)[0] += statistics.background;
  (*terms)[1] += statistics.triggered;
  (*terms)[2] += statistics.lag;
  (*terms)[3] += statistics.distanceSquared;
  (*terms)[4] += statistics.logIntensity;
  (*terms)[5] += statistics.cut;
  (*terms)[6] += 1.0;
  (*terms)[7] += windowMass;
}`,
          partial,
          output: state,
          outputSlot: sumsSlot,
          tileGate: gate.getGate(0),
          mergeGate: gate.getGate(1)
        })
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-update`,
          operation: OPERATION,
          variant: 'update',
          invocationCount: 1,
          bindings: [
            {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
            {name: 'state', view: state, type: 'f32', access: 'read_write'},
            gate.binding
          ],
          declarations: `${stateConstants}\n${getNaNDeclaration()}
const INTENSITY_FLOOR: f32 = ${getWGSLFloatLiteral(GPU_SPACE_TIME_HAWKES_INTENSITY_FLOOR)};
const BRANCHING_RATIO_FLOOR: f32 = ${getWGSLFloatLiteral(GPU_SPACE_TIME_HAWKES_BRANCHING_RATIO_FLOOR)};
const MAXIMUM_ITERATIONS: f32 = ${maximumIterations}.0;
${SCALED_QUOTIENT_WGSL}
${SPACE_TIME_HAWKES_COMPENSATOR_WGSL}
fn sumAt(index: u32) -> f32 { return state[stateOffset + S_SUMS + index]; }
fn relativeChange(next: f32, previous: f32) -> f32 {
  return getScaledQuotient(abs(next - previous), max(abs(previous), INTENSITY_FLOOR), 1.0, 0);
}`,
          body: `if (state[stateOffset + S_DONE] != 0.0) {
    return;
  }
  let iterations = state[stateOffset + S_ITERATION_COUNT] + 1.0;
  state[stateOffset + S_ITERATION_COUNT] = iterations;
  let area = params[paramsOffset];
  let duration = params[paramsOffset + 1u];
  let tolerance = params[paramsOffset + 4u];
  let background = state[stateOffset + S_BACKGROUND];
  let branchingRatio = state[stateOffset + S_BRANCHING_RATIO];
  let decayRate = state[stateOffset + S_DECAY_RATE];
  let variance = state[stateOffset + S_SPATIAL_VARIANCE];
  let backgroundSum = sumAt(0u);
  let triggeredSum = sumAt(1u);
  let lagSum = sumAt(2u);
  let distanceSum = sumAt(3u);
  let eventCount = sumAt(6u);
  let windowMass = sumAt(7u);
  state[stateOffset + S_LOG_LIKELIHOOD] = sumAt(4u) - (getHawkesBackgroundCompensator(background, area, duration) + branchingRatio * windowMass);
  if (!(eventCount > 0.0) || !(windowMass > 0.0)) {
    let nan = getNaN();
    state[stateOffset + S_BACKGROUND] = nan;
    state[stateOffset + S_BRANCHING_RATIO] = nan;
    state[stateOffset + S_DECAY_RATE] = nan;
    state[stateOffset + S_SPATIAL_VARIANCE] = nan;
    state[stateOffset + S_DONE] = 1.0;
    ${gate.disarmWGSL}
    return;
  }
  let nextBackground = getScaledQuotient(backgroundSum, area, duration, 0);
  let nextBranchingRatio = max(getScaledQuotient(triggeredSum, windowMass, 1.0, 0), BRANCHING_RATIO_FLOOR);
  var nextDecayRate = decayRate;
  var nextVariance = variance;
  if (triggeredSum > 1e-6 && lagSum > 0.0 && distanceSum > 0.0) {
    nextDecayRate = getScaledQuotient(triggeredSum, lagSum, 1.0, 0);
    nextVariance = getScaledQuotient(distanceSum, triggeredSum, 1.0, -1);
  }
  let change = max(
    max(relativeChange(nextBackground, background), relativeChange(nextBranchingRatio, branchingRatio)),
    max(relativeChange(nextDecayRate, decayRate), relativeChange(nextVariance, variance))
  );
  state[stateOffset + S_BACKGROUND] = nextBackground;
  state[stateOffset + S_BRANCHING_RATIO] = nextBranchingRatio;
  state[stateOffset + S_DECAY_RATE] = nextDecayRate;
  state[stateOffset + S_SPATIAL_VARIANCE] = nextVariance;
  let converged = change <= tolerance;
  if (converged || iterations >= MAXIMUM_ITERATIONS) {
    state[stateOffset + S_DONE] = 1.0;
    state[stateOffset + S_CONVERGED] = select(0.0, 1.0, converged);
    state[stateOffset + S_ITERATION_LIMIT_REACHED] = select(1.0, 0.0, converged);
    ${gate.disarmWGSL}
  }`
        })
      );
    }

    // 6. Final diagnostics share one evaluation of each event at the published estimates.
    let fittedSums: GraphDataView<'float32'> | undefined;
    if (output.fittedLogLikelihood) {
      fittedSums = createTransientView(graph, `${id}-fitted-sums`, 'float32', 2);
      nodes.push(
        ...getRowTileSumNodes<Parameters>(graph, {
          id: `${id}-fitted-sums`,
          operation: OPERATION,
          variant: 'fitted-sums',
          rowCount,
          layout,
          sumCount: 2,
          partial,
          output: fittedSums,
          bindings: [
            ...tileBindings,
            ...(output.backgroundProbability
              ? [
                  {name: 'timeOrder', view: timeOrder, type: 'u32', access: 'read'} as const,
                  {
                    name: 'backgroundProbability',
                    view: output.backgroundProbability,
                    type: 'f32',
                    access: 'read_write'
                  } as const
                ]
              : [])
          ],
          declarations: `${stateConstants}\n${IS_FINITE_VALUE_WGSL}\n${getNaNDeclaration()}\n${eventWGSL}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  let time = events[eventsOffset + 4u * row + 2u];
  let hasEstimates = isFiniteValue(state[stateOffset + S_BACKGROUND]) &&
    isFiniteValue(state[stateOffset + S_BRANCHING_RATIO]) &&
    isFiniteValue(state[stateOffset + S_DECAY_RATE]) &&
    isFiniteValue(state[stateOffset + S_SPATIAL_VARIANCE]);
  var probability = getNaN();
  if (hasEstimates && isFiniteValue(time)) {
    let statistics = evaluateEvent(row);
    probability = statistics.background;
    let exponent = state[stateOffset + S_DECAY_RATE] * (params[paramsOffset + 1u] - time);
    let windowMass = select(1.0 - exp(-exponent), exponent - 0.5 * exponent * exponent, exponent < 1e-3);
    (*terms)[0] += statistics.logIntensity;
    (*terms)[1] += windowMass;
  }
  ${output.backgroundProbability ? 'backgroundProbability[backgroundProbabilityOffset + timeOrder[timeOrderOffset + row]] = probability;' : ''}
}`
        })
      );
    } else if (output.backgroundProbability) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-responsibilities`,
          operation: OPERATION,
          variant: 'responsibilities',
          bindings: [
            {name: 'events', view: events, type: 'f32', access: 'read'},
            {name: 'state', view: state, type: 'f32', access: 'read'},
            {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
            {name: 'timeOrder', view: timeOrder, type: 'u32', access: 'read'},
            {
              name: 'backgroundProbability',
              view: output.backgroundProbability,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: rowCount,
          declarations: `${stateConstants}\n${IS_FINITE_VALUE_WGSL}\n${getNaNDeclaration()}\n${eventWGSL}`,
          body: `var probability = getNaN();
  // A parentless event returns backgroundWeight / sum = 1 even for NaN estimates.
  // Check the stored bit patterns before evaluating; NaN arithmetic cannot mark invalid fits.
  let hasEstimates = isFiniteValue(state[stateOffset + S_BACKGROUND]) &&
    isFiniteValue(state[stateOffset + S_BRANCHING_RATIO]) &&
    isFiniteValue(state[stateOffset + S_DECAY_RATE]) &&
    isFiniteValue(state[stateOffset + S_SPATIAL_VARIANCE]);
  if (hasEstimates && isFiniteValue(events[eventsOffset + 4u * index + 2u])) {
    probability = evaluateEvent(index).background;
  }
  backgroundProbability[backgroundProbabilityOffset + timeOrder[timeOrderOffset + index]] = probability;`
        })
      );
    }

    // 7. Publish the estimates, then the status (two kernels keep each under the binding limit).
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-estimates`,
        operation: OPERATION,
        variant: 'publish-estimates',
        invocationCount: 1,
        bindings: [
          {name: 'state', view: state, type: 'f32', access: 'read'},
          {name: 'estimates', view: output.estimates, type: 'f32', access: 'read_write'},
          ...(output.logLikelihood
            ? [
                {
                  name: 'logLikelihood',
                  view: output.logLikelihood,
                  type: 'f32',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(output.fittedLogLikelihood && fittedSums
            ? [
                {name: 'fittedSums', view: fittedSums, type: 'f32', access: 'read'} as const,
                {name: 'params', view: props.parameters, type: 'f32', access: 'read'} as const,
                {
                  name: 'fittedLogLikelihood',
                  view: output.fittedLogLikelihood,
                  type: 'f32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        declarations: `${stateConstants}\n${IS_FINITE_VALUE_WGSL}\n${getNaNDeclaration()}\n${SPACE_TIME_HAWKES_COMPENSATOR_WGSL}`,
        body: `estimates[estimatesOffset] = state[stateOffset + S_BACKGROUND];
  estimates[estimatesOffset + 1u] = state[stateOffset + S_BRANCHING_RATIO];
  estimates[estimatesOffset + 2u] = state[stateOffset + S_DECAY_RATE];
  estimates[estimatesOffset + 3u] = sqrt(state[stateOffset + S_SPATIAL_VARIANCE]);
  ${output.logLikelihood ? 'logLikelihood[logLikelihoodOffset] = state[stateOffset + S_LOG_LIKELIHOOD];' : ''}
  ${
    output.fittedLogLikelihood
      ? `
  let valid = isFiniteValue(state[stateOffset + S_BACKGROUND]) &&
    isFiniteValue(state[stateOffset + S_BRANCHING_RATIO]) &&
    isFiniteValue(state[stateOffset + S_DECAY_RATE]) &&
    isFiniteValue(state[stateOffset + S_SPATIAL_VARIANCE]);
  fittedLogLikelihood[fittedLogLikelihoodOffset] = select(getNaN(),
    fittedSums[fittedSumsOffset] - (getHawkesBackgroundCompensator(state[stateOffset + S_BACKGROUND], params[paramsOffset], params[paramsOffset + 1u]) +
    state[stateOffset + S_BRANCHING_RATIO] * fittedSums[fittedSumsOffset + 1u]), valid);`
      : ''
  }`
      })
    );
    const status = output.status ?? {};
    const statusBindings = (
      [
        ['converged', status.converged],
        ['iterationLimitReached', status.iterationLimitReached],
        ['iterationCount', status.iterationCount],
        ['candidateOverflow', status.candidateOverflow],
        ['invalidCount', status.invalidCount],
        ['approximation', status.approximation]
      ] as const
    ).filter(([, view]) => view !== undefined);
    if (statusBindings.length > 0) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish-status`,
          operation: OPERATION,
          variant: 'publish-status',
          invocationCount: 1,
          bindings: [
            {name: 'state', view: state, type: 'f32', access: 'read'},
            ...statusBindings.map(
              ([name, view]) => ({name, view: view!, type: 'u32', access: 'read_write'}) as const
            )
          ],
          declarations: stateConstants,
          body: `${status.converged ? 'converged[convergedOffset] = u32(state[stateOffset + S_CONVERGED]);' : ''}
  ${status.iterationLimitReached ? 'iterationLimitReached[iterationLimitReachedOffset] = u32(state[stateOffset + S_ITERATION_LIMIT_REACHED]);' : ''}
  ${status.iterationCount ? 'iterationCount[iterationCountOffset] = u32(state[stateOffset + S_ITERATION_COUNT]);' : ''}
  ${status.candidateOverflow ? 'candidateOverflow[candidateOverflowOffset] = u32(state[stateOffset + S_SUMS + 5u]);' : ''}
  ${status.invalidCount ? `invalidCount[invalidCountOffset] = ${rowCount}u - min(u32(max(state[stateOffset + S_SUMS + 6u], 0.0)), ${rowCount}u);` : ''}
  ${status.approximation ? 'approximation[approximationOffset] = 1u;' : ''}`
        })
      );
    }
    return nodes;
  }
}

/** WGSL `getNaN()`: a quiet NaN built at run time, since WGSL rejects NaN constants. */
function getNaNDeclaration(): string {
  return 'fn getNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }';
}
