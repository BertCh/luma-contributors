import {
  getPanelPreciseDiagnosticAllocationLengths,
  getPanelPreciseDiagnosticNodeCount,
  PANEL_DIAGNOSTIC_SUM_COUNT
} from './panel-precise-diagnostics';
import {withPanelErrorSpectrum} from './panel-error-spectrum-context';
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
import {GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE} from './spatial-maximum-likelihood-parameters';
import {getRegressionTileLayout} from './regression-row-tiles';
import {getPanelDiagnosticNodes} from './panel-diagnostics';
import {
  getPanelAffineFitAllocationLengths,
  getPanelAffineFitNodes,
  PANEL_AFFINE_FIT_MAXIMUM_NODE_COUNT
} from './panel-affine-fit-nodes';
import {snapshotRegressionConstructionPropsWithCompleteness as snapshotPanelConstructionProps} from './regression-certificate-snapshot';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  getRegressionCompletenessViews,
  getRegressionLossFoldCount,
  getRegressionWeightsLossNodes
} from './regression-weights-completeness';
import {validateGraphViewsBelongToGraph} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateRegressionOutputDispatches} from './regression-output-dispatches';
import {validateExactLogDeterminant} from './spatial-exact-log-determinant';
import {validateGPUSpatialWeightsTracesParameters} from './gpu-spatial-weights-traces';
import {
  GPU_SPATIAL_WEIGHTS_TRACES_EXACT_MAXIMUM_ROWS,
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
export const GPU_PANEL_ERROR_MAXIMUM_PREDICTOR_COUNT = 15;

/** Pooled LM diagnostics are valid. */
export const GPU_PANEL_ERROR_DIAGNOSTIC_STATUS_OK = 0;
/** Pooled fit is unavailable, the panel is incomplete, or supplied weights lost candidates/rows. */
export const GPU_PANEL_ERROR_DIAGNOSTIC_STATUS_INCOMPLETE = 1;
/** Pooled LM variance, weight trace or residualized spatial fitted-value information is degenerate/nonfinite. */
export const GPU_PANEL_ERROR_DIAGNOSTIC_STATUS_DEGENERATE = 2;

export type GPUSpatialPanelErrorMaximumLikelihoodProps = Omit<
  GPUSpatialErrorMaximumLikelihoodProps,
  'weights'
> & {
  /** Whole entity-graph weights: the panel expansion kernels read one CSR, so chunked weights are not accepted. */
  weights: GPUSpatialWeights;
  /** Balanced period-major panel. weights contains only the time-invariant entity graph. */
  periodCount: number;
  /** Fixed effects to absorb; defaults to 'unit'. Time/twoWay require predictorCount + periodCount - 1 <= 15. Random effects are refused because their independent-unit covariance needs a second determinant, unsupported by this trace-series fit. */
  effects?: 'unit' | 'time' | 'twoWay' | 'random';
  /** Pooled-panel LM error, robust LM lag, robust LM error: statistic/p-value pairs. */
  diagnostics?: {tests: GraphDataView<'float32'>; status: GraphDataView<'uint32'>};
};

/**
 * Fixed-effects spatial-error Gaussian ML: y_it = alpha_i + X_it beta + u_it,
 * u_t = lambda W u_t + epsilon_t. Time demeaning profiles the unit intercepts; the block graph
 * I_T ⊗ W supplies T log|I-lambda W|. The objective and variance divisor use N*T, matching
 * the N*T concentrated likelihood (not REML or a short-panel bias fix):
 * logL = -N*T/2 * (log(2*pi*sigmaSquared) + 1) + T*log|I-lambda*W|.
 * spreg logll = published logL - N*T/2 * log(N*T) for this convention. At N*T=48,
 * the offset is -92.90882426 (-56.58435 published versus -149.49318 recorded);
 * use k=N+p+1 and df=N*T-N-p-1 for unit effects, with p=predictorCount.
 * Information criteria remain unavailable (NaN). Panel lag has a separate normalization.
 * Exact mode accepts N entity eigenvalues repeated T times or N*T expanded eigenvalues.
 * The GPU exact bound is 1,024 supplied eigenvalues (entities on the short representation,
 * total panel rows on the expanded representation). All determinant/trace sums repeat T times.
 * Reuses bounded trace-series likelihood search and its approximation/boundary diagnostics.
 * Time effects demean across entities; twoWay first demeans across periods, then entities.
 * Period contrasts are also profiled after spatial filtering: spatial weights need not commute
 * with entity centering. This bounded route requires predictorCount + periodCount - 1 <= 15.
 * An original-input affine witness y=a+b*x_j (including a constant response) rejects a zero-
 * variance likelihood with status 3, and pooled diagnostics with status 2. Integer collinearity
 * certification avoids a small-variance cutoff; unsupported coordinates are not a witness.
 * The absorbed intercept row and information criteria are NaN. No random effects,
 * serial correlation, clustered inference, Hausman tests or missing panels are supported. Optional pooled-panel LM error and robust LM lag/error diagnostics use the untransformed complete panel.
 * Those LM tests use the pooled OLS variance `RSS / (n - p)` with `n = N*T` and
 * `p = predictorCount + 1` (spreg `panel_LMerror`, `panel_rLMlag` and `panel_rLMerror` use
 * `ols.sig2`), where the cross-section tests of `GPUSpatialRegressionDiagnostics` use `RSS / n`
 * (spreg `LMtests`): on the same residuals a panel LM-error is `((n - p) / n)^2` times the
 * cross-section statistic.
 */
export class GPUSpatialPanelErrorMaximumLikelihood implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUSpatialPanelErrorMaximumLikelihoodProps;
  readonly estimatedCommandNodeCount: number;
  constructor(props: GPUSpatialPanelErrorMaximumLikelihoodProps) {
    validateRequirement(
      props.id ?? 'spatial-panel-error-maximum-likelihood',
      !('exactSpectrumRepetitionCount' in props),
      'exactSpectrumRepetitionCount is not a panel prop; supply entityCount or entityCount * periodCount eigenvalues'
    );
    props = snapshotPanelConstructionProps(
      props,
      [
        'id',
        'weights',
        'predictors',
        'response',
        'predictorCount',
        'periodCount',
        'effects',
        'maximumPower',
        'traceMode',
        'traceProbeCount',
        'tailExtrapolation',
        'logDeterminant',
        'eigenvalues',
        'traceParameters',
        'tileRowCount',
        'diagnostics',
        'output'
      ],
      ['table', 'summary', 'residuals', 'status', 'powerTraceStandardErrors'],
      'spatial-panel-error-maximum-likelihood'
    );
    props = Object.freeze({
      ...props,
      diagnostics:
        props.diagnostics && snapshotConstructionFields(props.diagnostics, ['tests', 'status'])
    });
    this.props = props;
    this.id = props.id ?? 'spatial-panel-error-maximum-likelihood';
    validateRequirement(
      this.id,
      ['unit', 'time', 'twoWay', 'random'].includes(props.effects ?? 'unit'),
      "effects must be 'unit', 'time', 'twoWay' or 'random'"
    );
    validateRequirement(
      this.id,
      props.effects !== 'random',
      "effects 'random' with independent unit effects needs log|I + (phi^-2 - 1) A A'|, outside this bounded W trace series; use a dense random-effects spatial-error estimator"
    );
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
        GPU_SPATIAL_WEIGHTS_TRACES_EXACT_MAXIMUM_ROWS
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
    if (props.eigenvalues)
      validateRequirement(
        this.id,
        props.eigenvalues.length === entities || props.eigenvalues.length === rows,
        `eigenvalues length ${props.eigenvalues.length} must equal entityCount ${entities} or total panel rows ${rows}`
      );
    validateExactLogDeterminant(
      this.id,
      props,
      props.eigenvalues?.length === entities ? entities : rows
    );
    validateGPUSpatialWeightsTracesParameters(this.id, 'traceParameters', props.traceParameters);
    this.estimatedCommandNodeCount =
      getSpatialErrorLikelihoodNodeCount(
        entities * props.periodCount,
        maximumPower,
        props.traceMode,
        props.traceProbeCount
      ) +
      6 +
      PANEL_AFFINE_FIT_MAXIMUM_NODE_COUNT +
      (props.effects === 'twoWay' ? 4 : 0) +
      (props.effects === 'time' || props.effects === 'twoWay' ? 1 : 0) +
      getRegressionLossFoldCount(props.weights) +
      2 +
      Number(Boolean(props.diagnostics));
    this.estimatedCommandNodeCount += props.diagnostics
      ? getPanelPreciseDiagnosticNodeCount(rows)
      : 0;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateCapacity(
      this.id,
      'predictorCount',
      props.predictorCount,
      1,
      GPU_PANEL_ERROR_MAXIMUM_PREDICTOR_COUNT
    );
    if (props.effects === 'time' || props.effects === 'twoWay') {
      validateRequirement(
        this.id,
        props.predictorCount + props.periodCount - 1 <= GPU_PANEL_ERROR_MAXIMUM_PREDICTOR_COUNT,
        `effects '${props.effects}' require predictorCount + periodCount - 1 <= ${GPU_PANEL_ERROR_MAXIMUM_PREDICTOR_COUNT} because period contrasts share the 16-column solve; use computeSpatialPanelErrorMaximumLikelihoodOnCPU for a larger period-effect design`
      );
    }
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
      [
        'output.table',
        props.output.table,
        'float32',
        (props.predictorCount + 2) * GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE
      ],
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
        ...getRegressionCompletenessViews(props.weights)
      ]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, getRegressionCompletenessViews(props.weights));
    const entities = props.weights.offsets.length - 1;
    const rows = props.response.length;
    const capacity = props.weights.neighbors.length;
    const periodEffects = props.effects === 'time' || props.effects === 'twoWay';
    const nestedPredictorCount = props.predictorCount + (periodEffects ? props.periodCount - 1 : 0);
    validateRegressionAllocations(
      graph.device,
      id,
      getPanelAffineFitAllocationLengths(rows, props.predictorCount)
    );
    validateGraphChunkBindingRanges(graph, id, {
      predictors: props.predictors,
      response: props.response,
      offsets: props.weights.offsets,
      neighbors: props.weights.neighbors,
      weights: props.weights.weights,
      overflow: props.weights.status?.overflow,
      candidateOverflow: props.weights.status?.candidateOverflow,
      ...Object.fromEntries(
        getRegressionCompletenessViews(props.weights).map((view, index) => [
          `completeness ${index}`,
          view
        ])
      ),
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
      props.predictorCount,
      props.effects === 'time' ? 'time' : 'unit'
    );
    const timeScratch = getPanelWithinScratchLengths(
      entities,
      props.periodCount,
      props.predictorCount,
      'time'
    );
    for (const [name, length] of [
      ['expanded offsets', rows + 1],
      ['expanded neighbors/weights', capacity * props.periodCount],
      ['within predictors', props.predictors.length],
      ['within response', rows],
      ['panel mean partials', meanScratch.partialLength],
      ['panel means', meanScratch.meanLength],
      ...(periodEffects
        ? [
            ['period mean partials', timeScratch.partialLength] as const,
            ['period means', timeScratch.meanLength] as const,
            ['period profile predictors', rows * nestedPredictorCount] as const,
            [
              'period profile table',
              (nestedPredictorCount + 2) * GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE
            ] as const
          ]
        : []),
      ...(props.diagnostics
        ? [
            ['diagnostic lags', rows * 4 + 1] as const,
            ...Object.entries(
              getPanelPreciseDiagnosticAllocationLengths(rows, props.predictorCount)
            ),
            [
              'diagnostic sum partials',
              getRegressionTileLayout(rows).tileCount * PANEL_DIAGNOSTIC_SUM_COUNT
            ] as const,
            ['diagnostic sums', PANEL_DIAGNOSTIC_SUM_COUNT] as const
          ]
        : [])
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
        predictorCount: nestedPredictorCount,
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
    const completeness = getRegressionWeightsLossNodes(graph, `${id}-entity`, props.weights);
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
    const firstPredictors =
      props.effects === 'twoWay'
        ? createTransientView(graph, `${id}-unit-predictors`, 'float32', props.predictors.length)
        : predictors;
    const firstResponse =
      props.effects === 'twoWay'
        ? createTransientView(graph, `${id}-unit-response`, 'float32', rows)
        : response;
    const profiledPredictors = periodEffects
      ? createTransientView(
          graph,
          `${id}-profile-predictors`,
          'float32',
          rows * nestedPredictorCount
        )
      : predictors;
    const profiledTable = periodEffects
      ? createTransientView(
          graph,
          `${id}-profile-table`,
          'float32',
          (nestedPredictorCount + 2) * GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE
        )
      : props.output.table;
    return [
      ...completeness.nodes,
      ...(props.diagnostics ? getPanelDiagnosticNodes(graph, this, completeness.loss) : []),
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
        effects: props.effects === 'time' ? 'time' : 'unit',
        predictors: props.predictors,
        response: props.response,
        overflow: completeness.loss,
        output: {predictors: firstPredictors, response: firstResponse}
      }),
      ...(props.effects === 'twoWay'
        ? getPanelWithinNodes(graph, {
            id: `${id}-time`,
            entityCount: entities,
            periodCount: props.periodCount,
            predictorCount: props.predictorCount,
            effects: 'time',
            predictors: firstPredictors,
            response: firstResponse,
            output: {predictors, response}
          })
        : []),
      ...(periodEffects
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-period-profile`,
              operation: 'GPUSpatialPanelErrorMaximumLikelihood',
              variant: 'period-profile',
              invocationCount: rows,
              bindings: [
                {name: 'source', view: predictors, type: 'f32', access: 'read'},
                {name: 'design', view: profiledPredictors, type: 'f32', access: 'read_write'}
              ],
              // The implicit intercept and T-1 contrasts span all period intercepts, including
              // the corrections needed when W and the cross-entity projection do not commute.
              body: `for (var column = 0u; column < ${nestedPredictorCount}u; column++) {
  var value = 0.0;
  if (column < ${props.predictorCount}u) { value = source[sourceOffset + index * ${props.predictorCount}u + column]; }
  else { value = select(0.0, 1.0, index / ${entities}u == column - ${props.predictorCount}u); }
  design[designOffset + index * ${nestedPredictorCount}u + column] = value;
}`
            })
          ]
        : []),
      ...new GPUSpatialErrorMaximumLikelihood(
        withPanelErrorSpectrum(
          {
            ...props,
            id: `${id}-error`,
            predictorCount: nestedPredictorCount,
            predictors: profiledPredictors,
            response,
            output: {...props.output, table: profiledTable},
            weights: {
              offsets,
              neighbors,
              weights,
              status: completeness.loss ? {overflow: completeness.loss} : undefined
            }
          },
          props.eigenvalues?.length === entities ? props.periodCount : 1
        )
      ).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation: 'GPUSpatialPanelErrorMaximumLikelihood',
        variant: 'absorbed-intercept',
        invocationCount: (props.predictorCount + 2) * GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE,
        bindings: [
          ...(periodEffects
            ? [{name: 'profileTable', view: profiledTable, type: 'f32', access: 'read'} as const]
            : []),
          {name: 'table', view: props.output.table, type: 'f32', access: 'read_write'},
          {name: 'summary', view: props.output.summary, type: 'f32', access: 'read_write'}
        ],
        declarations: GET_NAN_WGSL,
        body: `if (index < ${GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE}u) { table[tableOffset + index] = getNaN(); }
${
  periodEffects
    ? `else {
let source = select(index, ${(nestedPredictorCount + 1) * GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE}u + index % ${GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE}u, index >= ${(props.predictorCount + 1) * GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE}u);
table[tableOffset + index] = profileTable[profileTableOffset + source];
}`
    : ''
}
if (index == 0u) { summary[summaryOffset + 4u] = getNaN(); summary[summaryOffset + 5u] = getNaN(); }`
      }),
      ...getPanelAffineFitNodes(graph, id, props),
      ...(completeness.loss && props.diagnostics
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-diagnostic-loss-final`,
              operation: 'GPUSpatialPanelErrorMaximumLikelihood',
              invocationCount: 6,
              bindings: [
                {name: 'loss', view: completeness.loss, type: 'u32', access: 'read'},
                {name: 'tests', view: props.diagnostics.tests, type: 'f32', access: 'read_write'},
                {name: 'status', view: props.diagnostics.status, type: 'u32', access: 'read_write'}
              ],
              declarations: GET_NAN_WGSL,
              body: 'if (loss[lossOffset] != 0u) { tests[testsOffset + index] = getNaN(); if (index == 0u) { status[statusOffset] = 1u; } }'
            })
          ]
        : [])
    ];
  }
}
