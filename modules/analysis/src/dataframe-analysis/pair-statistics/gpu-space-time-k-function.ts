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
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/index';
import {resolveGPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
import {validatePairHistogramShape} from './pair-histogram';
import {validatePairStatisticsInputs} from './pair-statistics-grid';
import {
  createSpaceTimeKNodes,
  GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH,
  GPU_SPACE_TIME_K_MAXIMUM_CLASS_COUNT,
  GPU_SPACE_TIME_K_MAXIMUM_LOCAL_BINS,
  validateSpaceTimeKRowCount,
  type GPUSpaceTimeKEdgeCorrection,
  type GPUSpaceTimePairCorrelationKernel
} from './space-time-k-function-nodes';

const OPERATION = 'GPUSpaceTimeKFunction';

/** Caller-owned outputs of {@link GPUSpaceTimeKFunction}. */
export type GPUSpaceTimeKFunctionOutput = {
  /**
   * Cumulative space-time K, at least `distanceClassCount * timeClassCount` float32 rows,
   * distance-major: row `a * timeClassCount + b` is `K(u_a, v_b)` with `u_a = maximumDistance *
   * (a + 1) / distanceClassCount` and `v_b = maximumTimeLag * (b + 1) / timeClassCount`. NaN where
   * undefined (fewer than two included rows, invalid parameters).
   */
  k: GraphDataView<'float32'>;
  /**
   * Optional purely spatial `K_s(u_a)` of all included events regardless of time, `distanceClassCount`
   * rows. Homogeneous only (rejected with `intensities`).
   */
  spatialK?: GraphDataView<'float32'>;
  /**
   * Optional purely temporal `K_t(v_b)` of all included events regardless of position,
   * `timeClassCount` rows. Homogeneous only.
   */
  temporalK?: GraphDataView<'float32'>;
  /** Optional Diggle `D(u_a, v_b) = K(u_a, v_b) - K_s(u_a) K_t(v_b)`, laid out like `k`. Homogeneous only. */
  d?: GraphDataView<'float32'>;
  /**
   * Optional space-time pair correlation `g(u_a, v_b)` at the same grid points, laid out like `k`
   * (needs positive `spatialBandwidth` and `temporalBandwidth` in the parameters, NaN otherwise).
   */
  pairCorrelation?: GraphDataView<'float32'>;
  /**
   * Optional uncorrected count of ordered pairs per class (not cumulative), laid out like `k`,
   * for pairs with `d <= maximumDistance` and `|dt| <= maximumTimeLag`. Saturates at `2^32 - 1`.
   */
  pairCounts?: GraphDataView<'uint32'>;
  /**
   * Optional local statistics per event (LISTA): `rows * distanceClassCount * timeClassCount`
   * float32, row `event * bins + a * timeClassCount + b` with `event` the input row and NaN for
   * excluded rows. At most {@link GPU_SPACE_TIME_K_MAXIMUM_LOCAL_BINS} classes.
   */
  local?: GraphDataView<'float32'>;
  /** Optional one-row uint32: 0 valid, 1 undefined (outputs are NaN), 3 integer accumulation overflowed. */
  status?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUSpaceTimeKFunction}.
 *
 * Per-frame (no recompile): the contents of `positions`, `times`, `intensities`, `mask` and
 * `parameters` (window, maximum distance and lag, time window, bandwidths). Compile-time: the row
 * count, `gridSize`, both class counts, `edgeCorrection`, the kernel, and which optional inputs and
 * outputs are present.
 */
export type GPUSpaceTimeKFunctionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'space-time-k-function'`. */
  id?: string;
  /** Planar coordinate contract of `positions`, as for `GPURipley`. */
  spatialContext?: GPUSpatialContext;
  /**
   * Optional declaration of `times`: `float32-relative`, in the unit of `maximumTimeLag`. Checked
   * at construction and exposed as {@link GPUSpaceTimeKFunction.temporalContext}.
   */
  temporalContext?: GPUTemporalContext;
  /** Packed planar event positions. */
  positions: GraphDataView<'float32x2'>;
  /** Event times, relative float32 in one unit (see `getGPUSpaceTimeGuardedRelativeTimes`). */
  times: GraphDataView<'float32'>;
  /**
   * Optional first-order intensity `lambda(x_i, t_i)` at each event (events per unit area per unit
   * time, positive). Selects the inhomogeneous estimator. Without it the intensity is the constant
   * `n / (|W| |T|)`. Rows with a non-finite or non-positive intensity are excluded.
   */
  intensities?: GraphDataView<'float32'>;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /** Per-frame float32 view of {@link GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH} values from `getGPUSpaceTimeKFunctionParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Maximum `[columns, rows]` of the pair-search lattice; affects speed, not results. */
  gridSize: readonly [number, number];
  /** Number of distance classes `U`, 1 to {@link GPU_SPACE_TIME_K_MAXIMUM_CLASS_COUNT}. */
  distanceClassCount: number;
  /** Number of time-lag classes `V`, 1 to {@link GPU_SPACE_TIME_K_MAXIMUM_CLASS_COUNT}. */
  timeClassCount: number;
  /** `'isotropic'` (default) or `'none'`. See the class documentation. */
  edgeCorrection?: GPUSpaceTimeKEdgeCorrection;
  /** Kernel of `output.pairCorrelation` along both axes: `'box'` (default, stpp's default) or `'epanechnikov'`. */
  pairCorrelationKernel?: GPUSpaceTimePairCorrelationKernel;
  /** Caller-owned outputs. */
  output: GPUSpaceTimeKFunctionOutput;
};

/**
 * Space-time K function, pair correlation, Diggle `D(u, v)` and the local statistics (LISTA) of a
 * space-time point pattern in a rectangle and a time interval, from one deterministic pass over
 * the indexed pairs. Definitions follow Gabriel and Diggle (2009, "Second-order analysis of
 * inhomogeneous spatio-temporal point process data", Statistica Neerlandica 63) as in stpp 2.0-8
 * `STIKhat` and `PCFhat`, and Diggle, Chetwynd, Haggkvist and Morris (1995) for `D`. stpp and
 * stopp are R-only (GPL) and R is not installed here, so their fixtures are pending; the spec
 * compares with a binary64 brute-force reference of the definitions below. Where stpp's exact
 * finite-sample factor or temporal weight differs from the definitions here it is not matched.
 *
 * Definitions, with `W` the rectangle (area `|W|`), `T = [timeStart, timeEnd]` (length `|T|`),
 * `n` the included events, ordered pairs `i != j`, `d_ij` the distance and `t_ij = |t_i - t_j|`:
 * - Included events: mask nonzero, finite position inside `W`, finite time inside `T`, and (with
 *   intensities) a finite positive intensity.
 * - Pair weight `w_ij = w^s_ij w^t_ij` (below), and homogeneous `K(u, v) = |W||T| / (n (n - 1))
 *   sum 1(d_ij <= u) 1(t_ij <= v) w_ij`. With `intensities`, `lambda'_i = lambda_i / (n / (|W||T|))` and
 *   `K(u, v) = |W||T| / n^2 sum 1(...) w_ij / (lambda'_i lambda'_j)`, which is
 *   `(1 / (|W||T|)) sum 1(...) w_ij / (lambda_i lambda_j)`. (The homogeneous `n (n - 1)` and the
 *   inhomogeneous `n^2` differ by the factor `(n - 1) / n`.)
 * - `edgeCorrection: 'isotropic'` (default): `w^s_ij` is Ripley's weight, the reciprocal of the
 *   fraction of the circle about `x_i` through `x_j` that lies inside `W` (exact for the
 *   rectangle, floored at 1/100 as in `GPURipley`); `w^t_ij` is 1 when both `t_i - t_ij` and
 *   `t_i + t_ij` lie in `T` and 2 otherwise (the one-dimensional Ripley weight). `'none'`: both 1.
 *   Not implemented: stpp's `border`, `modified.border` and `translate` corrections, polygon
 *   windows, and `infectious = TRUE` (pairs ordered in time).
 * - Spatial marginal `K_s(u) = |W| / (n (n - 1)) sum 1(d_ij <= u) w^s_ij` over all lags, temporal
 *   marginal `K_t(v) = |T| / (n (n - 1)) sum 1(t_ij <= v) w^t_ij` over all distances, and
 *   `D(u, v) = K(u, v) - K_s(u) K_t(v)`. Homogeneous only.
 * - Pair correlation `g(u, v) = |W||T| / (n (n - 1)) / (4 pi u) sum ks(u - d_ij) kt(v - t_ij) w_ij`
 *   (inhomogeneous: with the `n^2` weights), `ks` and `kt` normalised box or Epanechnikov kernels of half
 *   width `spatialBandwidth` and `temporalBandwidth`, evaluated at the same `(u_a, v_b)` grid.
 * - LISTA `K_i(u, v) = n s sum_{j != i} 1(...) w_ij / (lambda'_i lambda'_j)` with `s` the global
 *   scale above, so the mean of `K_i` over the included events is `K`.
 *
 * Classes are left-open, right-closed: a pair with scaled distance `d / u_max * U` in `(a, a + 1]`
 * is class `a` (distance 0 is class 0), the same convention as `GPURipley`, and likewise for lags.
 *
 * Exact ties follow stpp 2.0-8 (`src/stikfunction.f`, read from the CRAN source): a pair is
 * counted when `tij <= t` and `hij <= s`, the largest distance and lag included, and the
 * temporal weight is 1 when `t_i - t_ij >= start` and `t_i + t_ij <= end` (the time window is
 * closed, so a lag that exactly reaches a limit keeps weight 1). splancs `stkhat` decides both
 * the other way (weight 2 at such a lag; a pair at the largest distance or lag dropped), and
 * {@link GPUSpaceTimeDiggleTest} follows splancs, so on times recorded as whole days its `K`,
 * `K_s`, `K_t` and `D` can differ from the outputs of this class.
 * Pair counts and the temporal marginal are exact integers; weighted sums are accumulated as
 * fixed-point integers (2^-16 units) in the shared deterministic pair histogram, so results are
 * reproducible. LISTA sums add in float32 in the fixed neighbor order. The pair search is the
 * 3 x 3 lattice neighborhood within `maximumDistance` (plus `spatialBandwidth` for the pair
 * correlation); the temporal marginal costs `n V log n`. Cost is the number of indexed pairs, not
 * `n^2`.
 *
 * The permutation test of `D` is {@link GPUSpaceTimeDiggleTest}. A permutation test of the local
 * statistics is not implemented.
 */
export class GPUSpaceTimeKFunction implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpaceTimeKFunctionProps;
  /** Resolved coordinate contract (planar/native). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved declaration of `times`, when the props gave one. */
  readonly temporalContext?: GPUTemporalContext;
  /** Resolved edge correction. */
  readonly edgeCorrection: GPUSpaceTimeKEdgeCorrection;

  constructor(props: GPUSpaceTimeKFunctionProps) {
    this.id = props.id ?? 'space-time-k-function';
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
    validateRequirement(
      id,
      ['box', 'epanechnikov'].includes(props.pairCorrelationKernel ?? 'box'),
      "pairCorrelationKernel must be 'box' or 'epanechnikov'"
    );
    const rows = validatePairStatisticsInputs(id, {
      positions: props.positions,
      parameters: props.parameters,
      parameterLength: GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH,
      gridSize: props.gridSize,
      mask: props.mask
    });
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
    const bins = props.distanceClassCount * props.timeClassCount;
    // Slots are the classes plus the spatial marginal; channels are count, weight and its large
    // channel, plus two for the pair correlation.
    validatePairHistogramShape(
      id,
      bins + (output.spatialK || output.d ? props.distanceClassCount : 0),
      output.pairCorrelation ? 5 : 3
    );
    for (const [name, view] of [
      ['times', props.times],
      ['intensities', props.intensities]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} ${name}`);
        validateRequirement(id, view.length === rows, `${name} length must equal positions length`);
      }
    }
    validateRequirement(
      id,
      !(props.intensities && (output.spatialK || output.temporalK || output.d)),
      'spatialK, temporalK and d are defined for the homogeneous estimator only; omit intensities'
    );
    if (output.local) {
      validateRequirement(
        id,
        bins <= GPU_SPACE_TIME_K_MAXIMUM_LOCAL_BINS,
        `output.local needs distanceClassCount * timeClassCount <= ${GPU_SPACE_TIME_K_MAXIMUM_LOCAL_BINS}`
      );
    }
    const required: Record<string, number> = {
      k: bins,
      spatialK: props.distanceClassCount,
      temporalK: props.timeClassCount,
      d: bins,
      pairCorrelation: bins,
      pairCounts: bins,
      local: rows * bins,
      status: 1
    };
    const outputViews: GraphDataView[] = [];
    for (const [name, view] of Object.entries(output)) {
      if (view) {
        validateSinglePackedView(
          view,
          [name === 'pairCounts' || name === 'status' ? 'uint32' : 'float32'],
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
      props.intensities,
      props.mask,
      props.parameters
    ]);
  }

  /** Returns the nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateSpaceTimeKRowCount(id, graph.device, props.positions.length);
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.times,
      props.intensities,
      props.mask,
      props.parameters,
      ...Object.values(output)
    ]);
    if (output.local) {
      validateMaximumCount(
        id,
        'output.local rows',
        output.local.length,
        getStorageBindingElementLimit(graph.device, 4)
      );
    }
    const needsMarginals = Boolean(output.d);
    const spatialK =
      output.spatialK ??
      (needsMarginals
        ? createTransientView(graph, `${id}-spatial-k`, 'float32', props.distanceClassCount)
        : undefined);
    const temporalK =
      output.temporalK ??
      (needsMarginals
        ? createTransientView(graph, `${id}-temporal-k`, 'float32', props.timeClassCount)
        : undefined);
    return createSpaceTimeKNodes(graph, {
      id,
      operation: OPERATION,
      positions: props.positions,
      times: props.times,
      intensities: props.intensities,
      mask: props.mask,
      parameters: props.parameters,
      gridSize: props.gridSize,
      distanceClassCount: props.distanceClassCount,
      timeClassCount: props.timeClassCount,
      edgeCorrection: this.edgeCorrection,
      pairCorrelationKernel: props.pairCorrelationKernel,
      k: output.k,
      spatialK,
      temporalK,
      d: output.d,
      pairCorrelation: output.pairCorrelation,
      pairCounts: output.pairCounts,
      local: output.local,
      status: output.status
    }).nodes;
  }
}
