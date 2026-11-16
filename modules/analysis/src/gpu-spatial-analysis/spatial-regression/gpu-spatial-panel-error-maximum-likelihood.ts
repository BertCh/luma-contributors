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
import {getPanelDiagnosticNodes} from './panel-diagnostics';
import {validateRegressionOutputDispatches} from './regression-output-dispatches';
import {validateExactLogDeterminant} from './spatial-exact-log-determinant';
import {validateGPUSpatialWeightsTracesParameters} from './gpu-spatial-weights-traces';
import {
  GPU_SPATIAL_WEIGHTS_TRACES_EXACT_ROW_LIMIT,
  GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER
} from './spatial-weights-traces-parameters';
import {SPATIAL_SERIES_TAIL_MINIMUM_POWER} from './spatial-series-tail';
import {GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH} from './spatial-maximum-likelihood-parameters';
import {getOrdinaryLeastSquaresAllocationLengths} from './gpu-ordinary-least-squares';
import {REGRESSION_TILE_MAXIMUM_ROW_COUNT} from './regression-row-tiles';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateRequirement,
  validateGPUStorageBindingByteLength,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getPanelWithinNodes, getPanelWithinScratchLengths} from './panel-within-nodes';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  getSpatialErrorLikelihoodAllocationLengths,
  getSpatialErrorLikelihoodNodeCount,
  validateRegressionAllocations
} from './spatial-likelihood-allocations';
import {
  validateGPUSpatialWeights,
  type GPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {
  GPUSpatialErrorMaximumLikelihood,
  type GPUSpatialErrorMaximumLikelihoodProps
} from './gpu-spatial-error-maximum-likelihood';

/** Expanded CSR indices and float32 regression counts remain exact through 2^24 panel rows. */
export const GPU_PANEL_ERROR_MAXIMUM_ROW_COUNT = REGRESSION_TILE_MAXIMUM_ROW_COUNT;
/** Tiled period means need only bounded tile sums; the total panel row budget bounds periods. */
export const GPU_PANEL_ERROR_MAXIMUM_PERIOD_COUNT = GPU_PANEL_ERROR_MAXIMUM_ROW_COUNT;
/** The nested likelihood uses the existing sixteen-column Cholesky, including the intercept. */
const MAXIMUM_PREDICTOR_COUNT = 15;

export type GPUSpatialPanelErrorMaximumLikelihoodProps = Omit<
  GPUSpatialErrorMaximumLikelihoodProps,
  'weights'
> & {
  /** Whole entity-graph weights: the panel expansion kernels read one CSR, so chunked weights are not accepted. */
  weights: GPUSpatialWeights;
  /** Balanced period-major panel. weights contains only the time-invariant entity graph. */
  periodCount: number;
  /** Pooled-panel LM error, robust LM lag, robust LM error: statistic/p-value pairs. */
  diagnostics?: {tests: GraphDataView<'float32'>; status: GraphDataView<'uint32'>};
};

/**
 * Unit fixed-effects spatial-error Gaussian ML: y_it = alpha_i + X_it beta + u_it,
 * u_t = lambda W u_t + epsilon_t. Time demeaning profiles the unit intercepts; the block graph
 * I_T ⊗ W supplies T log|I-lambda W|. The objective and variance divisor use N*T, matching
 * spreg's Panel_FE_Error/ML_ErrorFE concentrated likelihood (not REML or a short-panel bias fix).
 * Reuses bounded trace-series likelihood search and its approximation/boundary diagnostics.
 * The absorbed intercept row and information criteria are NaN. No random/time effects,
 * serial correlation, clustered inference, Hausman tests or missing panels are supported. Optional pooled-panel LM error and robust LM lag/error diagnostics use the untransformed complete panel.
 * Those LM tests use the pooled OLS variance `RSS / (n - p)` with `n = N*T` and
 * `p = predictorCount + 1` (spreg `panel_LMerror`, `panel_rLMlag` and `panel_rLMerror` use
 * `ols.sig2`), where the cross-section tests of `GPUSpatialRegressionDiagnostics` use `RSS / n`
 * (spreg `LMtests`): on the same residuals a panel LM-error is `((n - p) / n)^2` times the
 * cross-section statistic.
 */
export class GPUSpatialPanelErrorMaximumLikelihood implements GPUCommandNodeProducer {
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  constructor(readonly props: GPUSpatialPanelErrorMaximumLikelihoodProps) {
    this.id = props.id ?? 'spatial-panel-error-maximum-likelihood';
    const entities = validateGPUSpatialWeights(this.id, props.weights);
    validateCapacity(
      this.id,
      'periodCount',
      props.periodCount,
      2,
      GPU_PANEL_ERROR_MAXIMUM_PERIOD_COUNT
    );
    validateCapacity(
      this.id,
      'rows',
      entities * props.periodCount,
      1,
      GPU_PANEL_ERROR_MAXIMUM_ROW_COUNT
    );
    validateCapacity(
      this.id,
      'edge capacity',
      props.weights.neighbors.length * props.periodCount,
      1,
      0xffffffff
    );
    const maximumPower = props.maximumPower ?? 128;
    validateCapacity(
      this.id,
      'maximumPower',
      maximumPower,
      2,
      GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER
    );
    validateCapacity(this.id, 'traceProbeCount', props.traceProbeCount ?? 64);
    validateRequirement(
      this.id,
      props.traceMode === undefined ||
        props.traceMode === 'exact' ||
        props.traceMode === 'stochastic',
      "traceMode must be 'exact' or 'stochastic'"
    );
    const rows = entities * props.periodCount;
    if (props.traceMode === 'exact')
      validateCapacity(
        this.id,
        'exact trace rows',
        rows,
        1,
        GPU_SPATIAL_WEIGHTS_TRACES_EXACT_ROW_LIMIT
      );
    if (props.tailExtrapolation)
      validateCapacity(
        this.id,
        'maximumPower',
        maximumPower,
        SPATIAL_SERIES_TAIL_MINIMUM_POWER,
        GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER
      );
    if (props.tileRowCount !== undefined)
      validateCapacity(this.id, 'tileRowCount', props.tileRowCount);
    validateExactLogDeterminant(this.id, props, rows);
    validateGPUSpatialWeightsTracesParameters(this.id, 'traceParameters', props.traceParameters);
    this.estimatedCommandNodeCount =
      getSpatialErrorLikelihoodNodeCount(
        entities * props.periodCount,
        maximumPower,
        props.traceMode,
        props.traceProbeCount
      ) + 6;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateCapacity(this.id, 'predictorCount', props.predictorCount, 1, MAXIMUM_PREDICTOR_COUNT);
    validateSinglePackedView(props.predictors, ['float32'], this.id);
    validateSinglePackedView(props.response, ['float32'], this.id);
    if (
      props.response.length !== entities * props.periodCount ||
      props.predictors.length !== props.response.length * props.predictorCount
    )
      throw new Error(`${this.id}: balanced period-major panel required`);
    if (props.diagnostics) {
      validateSinglePackedView(props.diagnostics.tests, ['float32'], this.id);
      validateSinglePackedView(props.diagnostics.status, ['uint32'], this.id);
      if (props.diagnostics.tests.length !== 6 || props.diagnostics.status.length !== 1)
        throw new Error(`${this.id}: diagnostic shape mismatch`);
    }
    for (const [name, view, format, minimum] of [
      ['output.table', props.output.table, 'float32', (props.predictorCount + 2) * 4],
      [
        'output.summary',
        props.output.summary,
        'float32',
        GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH
      ],
      ['output.residuals', props.output.residuals, 'float32', rows],
      ['output.status.code', props.output.status.code, 'uint32', 1],
      ['output.status.converged', props.output.status.converged, 'uint32', 1],
      [
        'output.powerTraceStandardErrors',
        props.output.powerTraceStandardErrors,
        'float32',
        maximumPower
      ]
    ] as const) {
      if (!view) continue;
      validateSinglePackedView(view, [format], `${this.id} ${name}`);
      validateRequirement(
        this.id,
        view.length >= minimum,
        `${name} must hold at least ${minimum} rows`
      );
    }
    validateRegressionOutputDispatches(
      this.id,
      {
        table: props.output.table,
        summary: props.output.summary,
        residuals: props.output.residuals,
        code: props.output.status.code,
        converged: props.output.status.converged,
        powerTraceStandardErrors: props.output.powerTraceStandardErrors,
        diagnosticTests: props.diagnostics?.tests,
        diagnosticStatus: props.diagnostics?.status
      },
      [
        ['table', 'summary', 'code', 'converged'],
        ['diagnosticTests', 'diagnosticStatus']
      ]
    );
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [
        props.output.table,
        props.output.summary,
        props.output.residuals,
        props.output.status.code,
        props.output.status.converged,
        props.output.powerTraceStandardErrors,
        ...Object.values(props.diagnostics ?? {})
      ],
      [
        props.predictors,
        props.response,
        props.eigenvalues,
        props.traceParameters,
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.weights.status?.overflow
      ]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const entities = props.weights.offsets.length - 1;
    const rows = props.response.length;
    const capacity = props.weights.neighbors.length;
    validateGraphChunkBindingRanges(graph, id, {
      predictors: props.predictors,
      response: props.response,
      offsets: props.weights.offsets,
      neighbors: props.weights.neighbors,
      weights: props.weights.weights,
      overflow: props.weights.status?.overflow,
      eigenvalues: props.eigenvalues,
      traceParameters: props.traceParameters,
      table: props.output.table,
      summary: props.output.summary,
      residuals: props.output.residuals,
      code: props.output.status.code,
      converged: props.output.status.converged,
      powerTraceStandardErrors: props.output.powerTraceStandardErrors,
      diagnosticTests: props.diagnostics?.tests,
      diagnosticStatus: props.diagnostics?.status
    });
    const meanScratch = getPanelWithinScratchLengths(
      entities,
      props.periodCount,
      props.predictorCount
    );
    for (const [name, length] of [
      ['expanded offsets', rows + 1],
      ['expanded neighbors/weights', capacity * props.periodCount],
      ['within predictors', props.predictors.length],
      ['within response', rows],
      ['panel mean partials', meanScratch.partialLength],
      ['panel means', meanScratch.meanLength],
      ...(props.diagnostics ? [['diagnostic lags', rows * 4] as const] : [])
    ] as const) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        length * 4,
        'reduce panel dimensions or edge capacity'
      );
    }
    validateRegressionAllocations(
      graph.device,
      id,
      getSpatialErrorLikelihoodAllocationLengths({
        ...props,
        rowCount: rows,
        edgeCount: capacity * props.periodCount
      })
    );
    if (props.diagnostics) {
      validateRegressionAllocations(
        graph.device,
        id,
        getOrdinaryLeastSquaresAllocationLengths(rows, props.predictorCount)
      );
    }
    const offsets = createTransientView(graph, `${id}-offsets`, 'uint32', rows + 1);
    const neighbors = createTransientView(
      graph,
      `${id}-neighbors`,
      'uint32',
      capacity * props.periodCount
    );
    const weights = createTransientView(
      graph,
      `${id}-weights`,
      'float32',
      capacity * props.periodCount
    );
    const predictors = createTransientView(
      graph,
      `${id}-within-predictors`,
      'float32',
      props.predictors.length
    );
    const response = createTransientView(graph, `${id}-within-response`, 'float32', rows);
    return [
      ...(props.diagnostics ? getPanelDiagnosticNodes(graph, this) : []),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-weights`,
        operation: 'GPUSpatialPanelErrorMaximumLikelihood',
        variant: 'time-block-weights',
        invocationCount: rows + 1,
        bindings: [
          {name: 'sourceOffsets', view: props.weights.offsets, type: 'u32', access: 'read'},
          {name: 'sourceNeighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'sourceWeights', view: props.weights.weights, type: 'f32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read_write'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'},
          {name: 'weights', view: weights, type: 'f32', access: 'read_write'}
        ],
        body: `let edges = min(sourceOffsets[sourceOffsetsOffset + ${entities}u], ${capacity}u);
if (index == ${rows}u) { offsets[offsetsOffset + index] = ${props.periodCount}u * edges; return; }
let period = index / ${entities}u; let entity = index % ${entities}u; let begin = min(sourceOffsets[sourceOffsetsOffset + entity], edges); let end = min(sourceOffsets[sourceOffsetsOffset + entity + 1u], edges);
offsets[offsetsOffset + index] = period * edges + begin;
for (var slot = begin; slot < end; slot++) { let destination = period * edges + slot; let neighbor = sourceNeighbors[sourceNeighborsOffset + slot]; neighbors[neighborsOffset + destination] = select(${rows}u, period * ${entities}u + neighbor, neighbor < ${entities}u); weights[weightsOffset + destination] = sourceWeights[sourceWeightsOffset + slot]; }`
      }),
      ...getPanelWithinNodes(graph, {
        id,
        entityCount: entities,
        periodCount: props.periodCount,
        predictorCount: props.predictorCount,
        predictors: props.predictors,
        response: props.response,
        overflow: props.weights.status?.overflow,
        output: {predictors, response}
      }),
      ...new GPUSpatialErrorMaximumLikelihood({
        ...props,
        id: `${id}-error`,
        predictors,
        response,
        weights: {offsets, neighbors, weights, status: props.weights.status}
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation: 'GPUSpatialPanelErrorMaximumLikelihood',
        variant: 'absorbed-intercept',
        invocationCount: 1,
        bindings: [
          {name: 'table', view: props.output.table, type: 'f32', access: 'read_write'},
          {name: 'summary', view: props.output.summary, type: 'f32', access: 'read_write'}
        ],
        declarations: GET_NAN_WGSL,
        body: `for (var slot = 0u; slot < 4u; slot++) { table[tableOffset + slot] = getNaN(); }
summary[summaryOffset + 4u] = getNaN(); summary[summaryOffset + 5u] = getNaN();`
      })
    ];
  }
}
