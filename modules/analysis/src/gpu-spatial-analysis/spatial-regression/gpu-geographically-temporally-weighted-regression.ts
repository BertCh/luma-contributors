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
  type WGSLDispatchGate,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateRequirement,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {type GPUIterationStatusPort} from '../../analysis-core/contracts/status';
import {
  GPU_GTWR_SEARCH_PARAMETER_LENGTH,
  createGTWRBandwidthSearchNodes,
  getGTWRBandwidthSearchNodeCount,
  getGTWRAdaptiveBandwidthWGSL,
  validateGTWRBandwidthSearch,
  type GPUGTWRBandwidthSearch
} from './gtwr-bandwidth-search';
import {validateRegressionOutputDispatches} from './regression-output-dispatches';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {
  resolveGPUTemporalContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/temporal-context';
import {createGTWRIndexedNodes, validateGTWRIndexedBindings} from './gtwr-indexed-nodes';
import {getCholeskyWGSL} from './spatial-regression-solve';
import {
  getEquilibratedCholeskyWGSL,
  getLocalKernelWeightWGSL
} from './local-weighted-least-squares-wgsl';
import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
} from './geographically-weighted-regression-parameters';

/** Most predictors of the space-time fit. */
export const GPU_GTWR_MAXIMUM_PREDICTOR_COUNT = 11;
/** Most rows: every focal row visits every row, `O(rows^2 x predictors^2)`. */
export const GPU_GTWR_MAXIMUM_ROW_COUNT = 32768;
/** Indexed fits keep summary row counts exactly representable in float32; bindings may cap rows sooner. */
export const GPU_GTWR_MAXIMUM_INDEXED_ROW_COUNT = 16777216;
/** Upper bound for the published grid aggregation, scan and stable radix sort plus fit/statistic passes. */
export const GPU_GTWR_INDEXED_COMMAND_NODE_ALLOWANCE = 256;
/** Number of float32 values of the per-frame parameters. */
export const GPU_GTWR_PARAMETER_LENGTH = 4;

/** Per-frame settings packed by {@link getGPUGTWRParameterValues}. */
export type GPUGTWRSettings = {
  /** Kernel bandwidth `h` in space-time distance units; positive and finite. */
  bandwidth: number;
  /**
   * Space-time scale `tau` of mgtwr: the distance between two observations is
   * `sqrt(ds^2 + tau dt^2)` with `ds` the planar distance and `dt` the difference of the times in
   * ticks. Zero gives a purely spatial fit. Non-negative and finite.
   */
  timeScale: number;
};

/**
 * Packs settings into the 4-element float32 layout read by {@link GPUGeographicallyTemporallyWeightedRegression}:
 * `[bandwidth, timeScale, 0, 0]`. Write it into a `GPUParameterBuffer` between encodings to move
 * through the (bandwidth, timeScale) plane without recompiling.
 *
 * @throws If `bandwidth` is not a positive finite number or `timeScale` is negative or not finite.
 */
export function getGPUGTWRParameterValues(
  settings: GPUGTWRSettings,
  target: Float32Array = new Float32Array(GPU_GTWR_PARAMETER_LENGTH)
): Float32Array {
  const bandwidth = Math.fround(settings.bandwidth);
  const timeScale = Math.fround(settings.timeScale);
  if (!(Number.isFinite(bandwidth) && bandwidth > 0)) {
    throw new Error('getGPUGTWRParameterValues bandwidth must be a positive finite number');
  }
  if (!(Number.isFinite(timeScale) && timeScale >= 0)) {
    throw new Error('getGPUGTWRParameterValues timeScale must be a non-negative finite number');
  }
  if (target.length < GPU_GTWR_PARAMETER_LENGTH) {
    throw new Error(`getGPUGTWRParameterValues target needs ${GPU_GTWR_PARAMETER_LENGTH} elements`);
  }
  target.set([bandwidth, timeScale, 0, 0]);
  return target;
}

/** Per-frame bounds for a nested bandwidth/time-scale profile search. */
export type GPUGTWRBandwidthSearchSettings = {
  /** Inclusive bandwidth bounds; integer neighbour counts when `adaptive` is enabled. */
  bandwidthBounds: readonly [number, number];
  /** Inclusive nonnegative time-scale bounds. */
  timeScaleBounds: readonly [number, number];
  /** Interval stopping tolerance, default 1e-3. */
  absoluteTolerance?: number;
  /** Relative interval stopping tolerance, default 1e-3. */
  relativeTolerance?: number;
};

/** Packs [minimumBandwidth, maximumBandwidth, minimumTimeScale, maximumTimeScale, absoluteTolerance, relativeTolerance]. */
export function getGPUGTWRBandwidthSearchParameterValues(
  settings: GPUGTWRBandwidthSearchSettings
): Float32Array {
  const values = new Float32Array([
    ...settings.bandwidthBounds,
    ...settings.timeScaleBounds,
    settings.absoluteTolerance ?? 1e-3,
    settings.relativeTolerance ?? 1e-3
  ]);
  validateRequirement(
    'getGPUGTWRBandwidthSearchParameterValues',
    values.every(Number.isFinite) &&
      values[0] > 0 &&
      values[0] <= values[1] &&
      values[2] >= 0 &&
      values[2] <= values[3] &&
      values[4] > 0 &&
      values[5] >= 0,
    'bounds must be finite and ordered, bandwidth > 0, timeScale >= 0, absoluteTolerance > 0 and relativeTolerance >= 0'
  );
  validateRequirement(
    'getGPUGTWRBandwidthSearchParameterValues',
    values.length === GPU_GTWR_SEARCH_PARAMETER_LENGTH,
    `bounds require ${GPU_GTWR_SEARCH_PARAMETER_LENGTH} values`
  );
  return values;
}

/** Caller-owned outputs of {@link GPUGeographicallyTemporallyWeightedRegression}. */
export type GPUGeographicallyTemporallyWeightedRegressionOutput = {
  /**
   * Local coefficients, row-major `row * (predictorCount + 1) + column`; column 0 is the
   * intercept. NaN for excluded or singular rows.
   */
  coefficients: GraphDataView<'float32'>;
  /**
   * Per-row status: `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS` `OK`, `SINGULAR` (the
   * equilibrated Cholesky factor of the local normal equations has a pivot below the OLS limit,
   * the local design is rank deficient, or the bandwidth or time scale is invalid) or `EXCLUDED`.
   */
  localStatus: GraphDataView<'uint32'>;
  /** Fitted value `x_i^T beta_i` per row; NaN for excluded or singular rows. */
  fitted: GraphDataView<'float32'>;
  /** Hat-matrix diagonal `S_ii` per row; NaN for excluded or singular rows. */
  hatDiagonal: GraphDataView<'float32'>;
  /** Optional residual `y_i - fitted_i` per row. */
  residuals?: GraphDataView<'float32'>;
  /**
   * Global summary with the layout of `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY`
   * (12 values): residual sum of squares, trace of the hat matrix, AICc, R squared, observation
   * count, singular row count, sigma squared, log-likelihood and included row count; the slots of
   * the significance-test fields are NaN. The AICc is NaN when `n - tr(S) - 2 <= 0` (the
   * correction is undefined there; see the class documentation for the formula).
   */
  summary: GraphDataView<'float32'>;
  /** Selected [bandwidth, timeScale], optional; requires bandwidthSearch. */
  selectedParameters?: GraphDataView<'float32'>;
  /** Profile-search iterations and convergence, optional. */
  status?: GPUIterationStatusPort;
};

/** Properties for {@link GPUGeographicallyTemporallyWeightedRegression}. */
export type GPUGeographicallyTemporallyWeightedRegressionProps = {
  id?: string;
  /** Planar `(x, y)` of every row. */
  positions: GraphDataView<'float32x2'>;
  /** Time of every row as float32 ticks after the origin of `temporalContext`. */
  times: GraphDataView<'float32'>;
  /** Declared meaning of `times`: `float32-relative` encoding with an origin epoch. */
  temporalContext: GPUTemporalContext;
  /** Planar native metric only; omitted keeps the legacy planar context. */
  spatialContext?: GPUSpatialContext;
  /** Predictors, `rows x predictorCount` row-major; the intercept is added. */
  predictors: GraphDataView<'float32'>;
  predictorCount: number;
  response: GraphDataView<'float32'>;
  /** Optional row filter: nonzero keeps the row. */
  mask?: GraphDataView<'uint32'>;
  /** Kernel on the space-time distance, fixed at compile time. Default `gaussian`. */
  kernel?: 'gaussian' | 'bisquare';
  /** Interpret bandwidth as an integer nearest-neighbour count, including the focal row, at most 128. */
  adaptive?: boolean;
  /**
   * Compile-time candidate route. Default `grid` for compact-support kernels, otherwise
   * `brute-force`. The reference route retains GPU_GTWR_MAXIMUM_ROW_COUNT. Grid requires
   * no bandwidthSearch; data and per-frame bandwidth/timeScale may change.
   */
  searchStructure?: 'grid' | 'brute-force';
  /**
   * Compile-time positive finite Gaussian support in bandwidths. Weights are exactly zero for
   * distance > gaussianCutoff * bandwidth on either route. Omit for the dense Gaussian reference.
   * This deliberately changes the kernel; it never silently approximates a dense Gaussian fit.
   */
  gaussianCutoff?: number;
  /** Nested golden-section profile search, compile-time budget with per-frame bounds. */
  bandwidthSearch?: GPUGTWRBandwidthSearch;
  /** At least 4 float32 values from {@link getGPUGTWRParameterValues}. */
  parameters: GraphDataView<'float32'>;
  output: GPUGeographicallyTemporallyWeightedRegressionOutput;
};

const OPERATION = 'GPUGeographicallyTemporallyWeightedRegression';
const STATUS = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS;

/**
 * Geographically and temporally weighted regression (Huang, Wu and Barry 2010) at a fixed or adaptive
 * bandwidth: a local weighted least squares fit at every row with kernel weights on the
 * space-time distance `sqrt(ds^2 + tau dt^2)`, as in the Python package mgtwr (its `GTWRKernel`
 * and `GTWR` with a fixed bandwidth, the Gaussian `exp(-z^2 / 2)` and bisquare `(1 - z^2)^2`
 * kernels with `z = d / h`).
 *
 * The local system is solved in coordinates centered on the focal row, with the diagonally
 * equilibrated Cholesky factorization shared with the GWR family, so a near-collinear local
 * design reports `SINGULAR` per row rather than a wild coefficient. Optional `bandwidthSearch`
 * profiles bandwidth inside a time-scale golden-section search; `selectedParameters` publishes
 * the winning pair. `adaptive` uses the kth usable space-time distance, including self, inflated
 * by 1.0000001 as in mgtwr. Counts above the usable row count or 128, and zero kth distances,
 * are singular. Search bounds/tolerances and fixed settings are per-frame; shapes, kernel,
 * adaptive mode and search budgets are compile-time. This reference-tier exhaustive search
 * assumes a unimodal profile, tests both endpoints, and keeps its best finite AICc at the limit.
 * Dense Gaussian and profile search use the bounded exhaustive reference.
 * Bisquare and explicit Gaussian-cutoff fits (fixed or adaptive) default to a stable space-time grid, rebuilding
 * it each encoding. Focal fits dispatch in parallel; only cells intersecting the support box are
 * visited, with the original space-time distance rejecting false positives. Adaptive bandwidths
 * find the exact kth distance by expanding cell shells with a conservative stopping bound. No neighbors are
 * truncated. Very dense neighborhoods can still require quadratic work. The grid accumulates in
 * cell order (ascending source row within a cell), so float32 rounding can differ from the dense
 * reference; compare well-conditioned coefficients at relative/absolute tolerance 1e-4, summaries
 * at 1e-3. Caller rows and output layouts are preserved. Grid inputs remain single packed views;
 * the device binding limit applies to each input, output and packed row scratch. Shapes, route
 * and gaussianCutoff are compile-time; buffer contents and bandwidth/timeScale are per-frame.
 *
 * The summary AICc is the one `GPUGeographicallyWeightedRegression` publishes (the mgwr formula),
 * `-2 llf + 2 n (tr(S) + 1) / (n - tr(S) - 2)`, so a GWR fit and a GTWR fit of the same data are
 * comparable: with `timeScale = 0` the two classes fit the same model and report the same AICc.
 * It is NaN when `n - tr(S) - 2 <= 0`, as in GWR, so a degenerate bandwidth whose hat trace
 * approaches the row count cannot win a profile search. mgtwr's own `aicc` is
 * `-2 llf + 2 (tr(S) + 1) (n - 1) / (n - tr(S) - 1)`; the value here is larger by
 * `2 n (k + 1) / (n - k - 2) - 2 (k + 1) (n - 1) / (n - k - 1)` with `k = tr(S)` (2.15 at
 * `n = 200`, `k = 60`; 0.05 at `n = 200`, `k = 10`). Residual sum of squares, trace, R squared,
 * sigma squared and log-likelihood are mgtwr's.
 */
export class GPUGeographicallyTemporallyWeightedRegression implements GPUCommandNodeProducer {
  readonly id: string;
  readonly rows: number;
  readonly estimatedCommandNodeCount: number;
  readonly searchStructure: 'grid' | 'brute-force';
  constructor(readonly props: GPUGeographicallyTemporallyWeightedRegressionProps) {
    this.id = props.id ?? 'geographically-temporally-weighted-regression';
    const {id} = this;
    const columns = props.predictorCount;
    const searchIterations = validateGTWRBandwidthSearch(
      id,
      props.bandwidthSearch,
      props.output.status
    );
    validateRequirement(
      id,
      props.kernel === undefined || ['gaussian', 'bisquare'].includes(props.kernel),
      'kernel must be gaussian or bisquare'
    );
    validateRequirement(
      id,
      props.searchStructure === undefined ||
        ['grid', 'brute-force'].includes(props.searchStructure),
      'searchStructure must be grid or brute-force'
    );
    if (props.gaussianCutoff !== undefined) {
      validateRequirement(
        id,
        Number.isFinite(Math.fround(props.gaussianCutoff)) && Math.fround(props.gaussianCutoff) > 0,
        'gaussianCutoff must be positive and finite in float32'
      );
    }
    const hasCompactSupport = props.kernel === 'bisquare' || props.gaussianCutoff !== undefined;
    this.searchStructure =
      props.searchStructure ??
      (hasCompactSupport && !props.bandwidthSearch ? 'grid' : 'brute-force');
    validateRequirement(
      id,
      this.searchStructure !== 'grid' || (hasCompactSupport && !props.bandwidthSearch),
      'searchStructure grid requires a compact kernel and no bandwidthSearch'
    );
    this.estimatedCommandNodeCount =
      this.searchStructure === 'grid'
        ? GPU_GTWR_INDEXED_COMMAND_NODE_ALLOWANCE
        : 3 +
          Number(Boolean(props.output.residuals)) +
          (searchIterations
            ? getGTWRBandwidthSearchNodeCount(searchIterations, 2) +
              Number(Boolean(props.output.selectedParameters))
            : 0);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    resolveGPUSpatialContext(id, props.spatialContext, {
      coordinateSpaces: ['planar'],
      metrics: ['native'],
      measure: 'distance'
    });
    validateCapacity(id, 'predictorCount', columns, 1, GPU_GTWR_MAXIMUM_PREDICTOR_COUNT);
    resolveGPUTemporalContext(id, props.temporalContext, {encodings: ['float32-relative']});
    validateSinglePackedView(props.response, ['float32'], `${id} response`);
    this.rows = props.response.length;
    validateCapacity(
      id,
      'rows',
      this.rows,
      2,
      this.searchStructure === 'grid'
        ? GPU_GTWR_MAXIMUM_INDEXED_ROW_COUNT
        : GPU_GTWR_MAXIMUM_ROW_COUNT
    );
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    validateViewLength(id, 'positions', props.positions, this.rows);
    validateSinglePackedView(props.times, ['float32'], `${id} times`);
    validateViewLength(id, 'times', props.times, this.rows);
    validateSinglePackedView(props.predictors, ['float32'], `${id} predictors`);
    validateViewLength(id, 'predictors', props.predictors, this.rows * columns);
    if (props.mask) {
      validateSinglePackedView(props.mask, ['uint32'], `${id} mask`);
      validateViewLength(id, 'mask', props.mask, this.rows);
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_GTWR_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters must have at least ${GPU_GTWR_PARAMETER_LENGTH} values`);
    }
    const {output} = props;
    for (const [name, view, format, length] of [
      ['coefficients', output.coefficients, 'float32', this.rows * (columns + 1)],
      ['localStatus', output.localStatus, 'uint32', this.rows],
      ['fitted', output.fitted, 'float32', this.rows],
      ['hatDiagonal', output.hatDiagonal, 'float32', this.rows],
      ['summary', output.summary, 'float32', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH]
    ] as const) {
      validateSinglePackedView(view, [format], `${id} output.${name}`);
      validateViewLength(id, `output.${name}`, view, length);
    }
    if (output.residuals) {
      validateSinglePackedView(output.residuals, ['float32'], `${id} output.residuals`);
      validateViewLength(id, 'output.residuals', output.residuals, this.rows);
    }
    if (output.selectedParameters) {
      validateSinglePackedView(
        output.selectedParameters,
        ['float32'],
        `${id} output.selectedParameters`
      );
      validateRequirement(
        id,
        Boolean(props.bandwidthSearch),
        'output.selectedParameters requires bandwidthSearch'
      );
      validateRequirement(
        id,
        output.selectedParameters.length >= 2,
        'output.selectedParameters must have at least 2 values'
      );
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
    validateRegressionOutputDispatches(
      id,
      {
        ...output,
        status: undefined,
        ...Object.fromEntries(
          Object.entries(output.status ?? {}).map(([name, view]) => [`status.${name}`, view])
        )
      },
      [
        ['coefficients', 'fitted', 'hatDiagonal', 'localStatus'],
        {read: ['fitted'], write: ['residuals']},
        ...(this.searchStructure === 'brute-force'
          ? [{read: ['fitted', 'hatDiagonal', 'localStatus'], write: ['summary']}]
          : []),
        Object.keys(output.status ?? {}).map(name => `status.${name}`)
      ]
    );
  }

  private getInputViews(): GraphDataView[] {
    const {props} = this;
    const views: (GraphDataView | undefined)[] = [
      props.positions,
      props.times,
      props.predictors,
      props.response,
      props.mask,
      props.bandwidthSearch ? undefined : props.parameters,
      props.bandwidthSearch?.parameters
    ];
    return views.filter((view): view is GraphDataView => view !== undefined);
  }

  private getOutputViews(): GraphDataView[] {
    const {status, ...output} = this.props.output;
    return [...Object.values(output), ...Object.values(status ?? {})].filter(
      (view): view is GraphDataView<'float32'> | GraphDataView<'uint32'> => Boolean(view)
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateGraphViewsBelongToGraph(this.id, graph, [
      ...this.getInputViews(),
      ...this.getOutputViews()
    ]);
    validateGraphChunkBindingRanges(graph, this.id, {
      ...Object.fromEntries(this.getInputViews().map((view, index) => [`input[${index}]`, view])),
      ...Object.fromEntries(this.getOutputViews().map((view, index) => [`output[${index}]`, view]))
    });
    validateGPUStorageBindingByteLength(
      graph.device,
      this.id,
      'space-time and coefficient scratch',
      this.rows * Math.max(3, this.props.predictorCount + 1) * 4,
      'reduce rows or predictorCount'
    );
    if (this.searchStructure === 'grid') {
      validateGTWRIndexedBindings(graph.device, this.id, this.rows, this.props.predictorCount);
      return createGTWRIndexedNodes(graph, this.id, this.props);
    }
    return this.createNodes(graph);
  }

  private createNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    gate?: WGSLDispatchGate<Parameters>,
    preparedPoints?: GraphDataView<'float32'>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rows} = this;
    const {output} = props;
    const columns = props.predictorCount;
    const size = columns + 1;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'space-time and coefficient scratch',
      rows * Math.max(3, size) * 4,
      'reduce rows or predictorCount'
    );
    if (props.bandwidthSearch) {
      const spaceTime = createSpaceTimePointNodes(graph, id, props, rows);
      const selectedParameters = createTransientView(graph, `${id}-selected`, 'float32', 4);
      const searchNodes = createGTWRBandwidthSearchNodes(graph, {
        id,
        rows,
        adaptive: props.adaptive ?? false,
        search: props.bandwidthSearch,
        parameters: selectedParameters,
        status: output.status,
        evaluate: (parameters, evaluationGate, evaluationId) => {
          const candidateOutput = {
            coefficients: createTransientView(
              graph,
              `${evaluationId}-coefficients`,
              'float32',
              rows * size
            ),
            localStatus: createTransientView(graph, `${evaluationId}-status`, 'uint32', rows),
            fitted: createTransientView(graph, `${evaluationId}-fitted`, 'float32', rows),
            hatDiagonal: createTransientView(graph, `${evaluationId}-hat`, 'float32', rows),
            summary: createTransientView(graph, `${evaluationId}-summary`, 'float32', 12)
          };
          const candidate = new GPUGeographicallyTemporallyWeightedRegression({
            ...props,
            id: evaluationId,
            bandwidthSearch: undefined,
            searchStructure: 'brute-force',
            parameters,
            output: candidateOutput
          });
          return {
            nodes: candidate.createNodes(graph, evaluationGate, spaceTime.points),
            summary: candidateOutput.summary
          };
        }
      });
      const {status, selectedParameters: selectedOutput, ...fitOutput} = output;
      const finalFit = new GPUGeographicallyTemporallyWeightedRegression({
        ...props,
        id: `${id}-selected-fit`,
        bandwidthSearch: undefined,
        searchStructure: 'brute-force',
        parameters: selectedParameters,
        output: fitOutput
      });
      if (selectedOutput) {
        searchNodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-publish-selected`,
            operation: OPERATION,
            invocationCount: 2,
            bindings: [
              {name: 'selected', view: selectedParameters, type: 'f32', access: 'read'},
              {name: 'output', view: selectedOutput, type: 'f32', access: 'read_write'}
            ],
            body: 'output[outputOffset + index] = selected[selectedOffset + index];'
          })
        );
      }
      return [
        ...spaceTime.nodes,
        ...searchNodes,
        ...finalFit.createNodes(graph, undefined, spaceTime.points)
      ];
    }
    const spaceTime = preparedPoints
      ? {points: preparedPoints, nodes: []}
      : createSpaceTimePointNodes(graph, id, props, rows, gate);
    const points = spaceTime.points;
    return [
      ...spaceTime.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-fit`,
        operation: OPERATION,
        variant: 'local-fit',
        invocationCount: rows,
        ...gate?.getGate(0),
        bindings: [
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'points', view: points, type: 'f32', access: 'read'},
          {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
          {name: 'response', view: props.response, type: 'f32', access: 'read'},
          {name: 'coefficients', view: output.coefficients, type: 'f32', access: 'read_write'},
          {name: 'fitted', view: output.fitted, type: 'f32', access: 'read_write'},
          {name: 'hatDiagonal', view: output.hatDiagonal, type: 'f32', access: 'read_write'},
          {name: 'localStatus', view: output.localStatus, type: 'u32', access: 'read_write'}
        ],
        declarations: `${IS_FINITE_BITS_WGSL}
${IS_FINITE_VALUE_WGSL}
${GET_NAN_WGSL}
${getCholeskyWGSL(size, size)}
${getEquilibratedCholeskyWGSL(size)}
${getLocalKernelWeightWGSL(props.gaussianCutoff ?? null)}
${props.adaptive ? getGTWRAdaptiveBandwidthWGSL(rows) : ''}`,
        body: `let timeScale = parameters[parametersOffset + 1u];
  let bandwidth = ${props.adaptive ? 'getAdaptiveBandwidth(index, parameters[parametersOffset], timeScale)' : 'parameters[parametersOffset]'};
  let originX = points[pointsOffset + 3u * index];
  var code = ${STATUS.SINGULAR}u;
  var coefficient: array<f32, ${size}>;
  for (var column = 0u; column < ${size}u; column++) {
    coefficient[column] = getNaN();
  }
  var fittedValue = getNaN();
  var hat = getNaN();
  if (!isFiniteBits(originX)) {
    code = ${STATUS.EXCLUDED}u;
  } else if (bandwidth > 0.0 && isFiniteBits(bandwidth) && timeScale >= 0.0 && isFiniteBits(timeScale)) {
    let originY = points[pointsOffset + 3u * index + 1u];
    let originTime = points[pointsOffset + 3u * index + 2u];
    let originBase = predictorsOffset + index * ${columns}u;
    var a: array<f32, ${size * size}>;
    var b: array<f32, ${size}>;
    var z: array<f32, ${size}>;
    for (var entry = 0u; entry < ${size * size}u; entry++) { a[entry] = 0.0; }
    for (var column = 0u; column < ${size}u; column++) { b[column] = 0.0; }
    z[0] = 1.0;
    for (var row = 0u; row < ${rows}u; row++) {
      let rowX = points[pointsOffset + 3u * row];
      if (!isFiniteBits(rowX)) { continue; }
      let deltaX = rowX - originX;
      let deltaY = points[pointsOffset + 3u * row + 1u] - originY;
      let deltaTime = points[pointsOffset + 3u * row + 2u] - originTime;
      let distance = sqrt(deltaX * deltaX + deltaY * deltaY + timeScale * deltaTime * deltaTime);
      let weight = getLocalKernelWeight(distance, bandwidth, ${props.kernel === 'bisquare' ? 'true' : 'false'});
      if (!(weight > 0.0)) { continue; }
      let rowBase = predictorsOffset + row * ${columns}u;
      for (var column = 0u; column < ${columns}u; column++) {
        z[column + 1u] = predictors[rowBase + column] - predictors[originBase + column];
      }
      let observed = response[responseOffset + row];
      for (var r = 0u; r < ${size}u; r++) {
        let weighted = weight * z[r];
        b[r] = b[r] + weighted * observed;
        for (var c = 0u; c <= r; c++) {
          a[r * ${size}u + c] = a[r * ${size}u + c] + weighted * z[c];
        }
      }
    }
    var scale: array<f32, ${size}>;
    if (factorEquilibrated_${size}(&a, &scale)) {
      let inverse00 = inverseDiagonalEquilibrated_${size}(&a, &scale, 0u);
      // b is X'Wy of the centered design; the solve is in place and unscales the coefficients.
      solveEquilibrated_${size}(&a, &scale, &b);
      var finite = isFiniteBits(inverse00) && inverse00 >= 0.0;
      for (var column = 0u; column < ${size}u; column++) {
        finite = finite && isFiniteBits(b[column]);
      }
      if (finite) {
        code = ${STATUS.OK}u;
        // The centered intercept is the fitted value; uncentering it gives the raw intercept.
        fittedValue = b[0];
        hat = inverse00;
        var intercept = b[0];
        for (var column = 0u; column < ${columns}u; column++) {
          coefficient[column + 1u] = b[column + 1u];
          intercept = intercept - b[column + 1u] * predictors[originBase + column];
        }
        coefficient[0] = intercept;
      }
    }
  }
  for (var column = 0u; column < ${size}u; column++) {
    coefficients[coefficientsOffset + index * ${size}u + column] = coefficient[column];
  }
  fitted[fittedOffset + index] = fittedValue;
  hatDiagonal[hatDiagonalOffset + index] = hat;
  localStatus[localStatusOffset + index] = code;`
      }),
      ...(output.residuals
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-residuals`,
              operation: OPERATION,
              variant: 'residuals',
              invocationCount: rows,
              ...gate?.getGate(0),
              bindings: [
                {name: 'response', view: props.response, type: 'f32', access: 'read'},
                {name: 'fitted', view: output.fitted, type: 'f32', access: 'read'},
                {name: 'residuals', view: output.residuals, type: 'f32', access: 'read_write'}
              ],
              body: `residuals[residualsOffset + index] = response[responseOffset + index] - fitted[fittedOffset + index];`
            })
          ]
        : []),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        invocationCount: 1,
        ...gate?.getGate(1),
        bindings: [
          {name: 'response', view: props.response, type: 'f32', access: 'read'},
          {name: 'fitted', view: output.fitted, type: 'f32', access: 'read'},
          {name: 'hatDiagonal', view: output.hatDiagonal, type: 'f32', access: 'read'},
          {name: 'localStatus', view: output.localStatus, type: 'u32', access: 'read'},
          {name: 'summary', view: output.summary, type: 'f32', access: 'read_write'}
        ],
        declarations: GET_NAN_WGSL,
        // Statistics over the non-singular included rows, as the GWR summary (mgwr / mgtwr formulas).
        body: `var count = 0.0;
  var singular = 0.0;
  var included = 0.0;
  var mean = 0.0;
  var residualSquares = 0.0;
  var trace = 0.0;
  for (var row = 0u; row < ${rows}u; row++) {
    let code = localStatus[localStatusOffset + row];
    if (code == ${STATUS.EXCLUDED}u) { continue; }
    included += 1.0;
    if (code != ${STATUS.OK}u) { singular += 1.0; continue; }
    count += 1.0;
    let observed = response[responseOffset + row];
    mean += (observed - mean) / count;
    let residual = observed - fitted[fittedOffset + row];
    residualSquares += residual * residual;
    trace += hatDiagonal[hatDiagonalOffset + row];
  }
  var totalSquares = 0.0;
  for (var row = 0u; row < ${rows}u; row++) {
    if (localStatus[localStatusOffset + row] == ${STATUS.OK}u) {
      let deviation = response[responseOffset + row] - mean;
      totalSquares += deviation * deviation;
    }
  }
  // Work in log space: 2 pi * RSS can overflow even when RSS and 2 pi * RSS / n are finite.
  let logVariance = log(residualSquares) - log(count);
  let logLikelihood = -0.5 * count * (logVariance + 1.8378770664093453 + 1.0);
  // The AICc of GPUGeographicallyWeightedRegression (mgwr): -2 llf + 2 n (tr + 1) / (n - tr - 2),
  // defined only while n - tr - 2 > 0. mgtwr's own aicc is 2 (tr + 1) (n - 1) / (n - tr - 1) in
  // the penalty and is not published here.
  let correctedDenominator = count - trace - 2.0;
  let corrected = -2.0 * logLikelihood + 2.0 * count * (trace + 1.0) / correctedDenominator;
  for (var slot = 0u; slot < ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH}u; slot++) {
    summary[summaryOffset + slot] = getNaN();
  }
  summary[summaryOffset] = residualSquares;
  summary[summaryOffset + 1u] = trace;
  summary[summaryOffset + 2u] = select(getNaN(), corrected, count > 0.0 && correctedDenominator > 0.0);
  summary[summaryOffset + 3u] = select(getNaN(), 1.0 - residualSquares / totalSquares, totalSquares > 0.0);
  summary[summaryOffset + 4u] = count;
  summary[summaryOffset + 6u] = singular;
  summary[summaryOffset + 7u] = select(getNaN(), residualSquares / (count - trace), count > trace);
  summary[summaryOffset + 10u] = select(getNaN(), logLikelihood, count > 0.0);
  summary[summaryOffset + 11u] = included;`
      })
    ];
  }
}

/**
 * Packs positions and times of every row into `(x, y, time)` triples, with a NaN `x` marking a
 * row that is masked out or has a non-finite position, time, response or predictor.
 *
 * @internal
 */
export function createSpaceTimePointNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: Pick<
    GPUGeographicallyTemporallyWeightedRegressionProps,
    'positions' | 'times' | 'predictors' | 'predictorCount' | 'response' | 'mask'
  >,
  rows: number,
  gate?: WGSLDispatchGate<Parameters>
): {nodes: GPUCommandNode<Parameters>[]; points: GraphDataView<'float32'>} {
  const columns = props.predictorCount;
  const points = createTransientView<'float32', Parameters>(
    graph,
    `${id}-points`,
    'float32',
    rows * 3
  );
  const node = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-points`,
    operation: OPERATION,
    variant: 'usable-rows',
    invocationCount: rows,
    ...gate?.getGate(0),
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'times', view: props.times, type: 'f32', access: 'read'},
      {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
      {name: 'response', view: props.response, type: 'f32', access: 'read'},
      ...(props.mask
        ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
        : []),
      {name: 'points', view: points, type: 'f32', access: 'read_write'}
    ],
    declarations: `${IS_FINITE_BITS_WGSL}\n${GET_NAN_WGSL}`,
    body: `var valid = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
  let x = positions[positionsOffset + 2u * index];
  let y = positions[positionsOffset + 2u * index + 1u];
  let time = times[timesOffset + index];
  valid = valid && isFiniteBits(x) && isFiniteBits(y) && isFiniteBits(time) && isFiniteBits(response[responseOffset + index]);
  for (var column = 0u; column < ${columns}u; column++) {
    valid = valid && isFiniteBits(predictors[predictorsOffset + index * ${columns}u + column]);
  }
  points[pointsOffset + 3u * index] = select(getNaN(), x, valid);
  points[pointsOffset + 3u * index + 1u] = y;
  points[pointsOffset + 3u * index + 2u] = time;`
  });
  return {nodes: [node], points};
}
