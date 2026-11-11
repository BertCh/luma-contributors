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
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateGPUSpatialWeights,
  type GPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {
  GPUSpatialErrorMaximumLikelihood,
  type GPUSpatialErrorMaximumLikelihoodProps
} from './gpu-spatial-error-maximum-likelihood';

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
  constructor(readonly props: GPUSpatialPanelErrorMaximumLikelihoodProps) {
    this.id = props.id ?? 'spatial-panel-error-maximum-likelihood';
    const entities = validateGPUSpatialWeights(this.id, props.weights);
    validateCapacity(this.id, 'periodCount', props.periodCount, 2, 1024);
    validateCapacity(this.id, 'rows', entities * props.periodCount, 1, 65536);
    validateCapacity(
      this.id,
      'edge capacity',
      props.weights.neighbors.length * props.periodCount,
      1,
      0xffffffff
    );
    validateCapacity(this.id, 'predictorCount', props.predictorCount, 1, 15);
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
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [
        props.output.table,
        props.output.summary,
        props.output.residuals,
        props.output.status.code,
        props.output.status.converged,
        ...Object.values(props.diagnostics ?? {})
      ],
      [
        props.predictors,
        props.response,
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
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.response,
      props.weights.offsets,
      props.weights.neighbors,
      props.weights.weights,
      props.weights.status?.overflow
    ]);
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
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-within`,
        operation: 'GPUSpatialPanelErrorMaximumLikelihood',
        variant: 'unit-demean',
        invocationCount: rows,
        bindings: [
          {name: 'sourcePredictors', view: props.predictors, type: 'f32', access: 'read'},
          {name: 'sourceResponse', view: props.response, type: 'f32', access: 'read'},
          {name: 'predictors', view: predictors, type: 'f32', access: 'read_write'},
          {name: 'response', view: response, type: 'f32', access: 'read_write'},
          ...(props.weights.status?.overflow
            ? [
                {
                  name: 'overflow',
                  view: props.weights.status.overflow,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : [])
        ],
        declarations: GET_NAN_WGSL,
        body: `let entity = index % ${entities}u; var mean = 0.0;
for (var period = 0u; period < ${props.periodCount}u; period++) { mean += sourceResponse[sourceResponseOffset + period * ${entities}u + entity] / ${props.periodCount}.0; }
response[responseOffset + index] = sourceResponse[sourceResponseOffset + index] - mean;
${props.weights.status?.overflow ? 'if (overflow[overflowOffset] != 0u) { response[responseOffset + index] = getNaN(); }' : ''}
for (var column = 0u; column < ${props.predictorCount}u; column++) { var predictorMean = 0.0;
for (var period = 0u; period < ${props.periodCount}u; period++) { predictorMean += sourcePredictors[sourcePredictorsOffset + (period * ${entities}u + entity) * ${props.predictorCount}u + column] / ${props.periodCount}.0; }
predictors[predictorsOffset + index * ${props.predictorCount}u + column] = sourcePredictors[sourcePredictorsOffset + index * ${props.predictorCount}u + column] - predictorMean; }`
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
