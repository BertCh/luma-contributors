// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount} from './spatial-two-stage-least-squares-parameters';
import {validateRegressionOutputDispatches} from './regression-output-dispatches';

import {getRegressionTwoStageAllocationLengths} from './regression-binding-plans';
import {validateRegressionAllocations} from './spatial-likelihood-allocations';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {ERFC_VALUE_WGSL, GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGPUSpatialWeights, type GPUSpatialWeights} from '../spatial-weights/index';
import {
  GPUSpatialTwoStageLeastSquares,
  type GPUSpatialTwoStageLeastSquaresOutput
} from './gpu-spatial-two-stage-least-squares';

/** The serial reference within transform visits each period of one entity. */
const PANEL_REGRESSION_MAXIMUM_PERIOD_COUNT = 1024;
/** Preserve the existing strictly-below-2^24 capacity for exact float32 row counts. */
const PANEL_REGRESSION_MAXIMUM_ROW_COUNT = 2 ** 24 - 1;
/** Expanded CSR offsets and edge indices are uint32. */
const PANEL_REGRESSION_MAXIMUM_EDGE_COUNT = 0xffffffff;

/** Balanced period-major spatial lag panel, pooled or with unit fixed effects. */
export type GPUSpatialPanelRegressionProps = {
  id?: string;
  /** Time-invariant spatial weights for the entity axis. No cross-time edges are introduced. */
  weights: GPUSpatialWeights;
  periodCount: number;
  /** Default pooled; unit effects absorb a separate intercept for each entity. */
  effects?: 'pooled' | 'unit';
  predictorCount: number;
  /** `period * entityCount + entity`, then predictor column. No missing observations. */
  predictors: GraphDataView<'float32'>;
  response: GraphDataView<'float32'>;
  instrumentOrder?: 1 | 2;
  output: GPUSpatialTwoStageLeastSquaresOutput;
};

/**
 * Pooled spatial-lag panel 2SLS: y_t = rho W y_t + X_t beta + intercept + e_t. Builds I_T ⊗ W
 * on GPU and delegates estimation and completion diagnostics to GPUSpatialTwoStageLeastSquares.
 * The caller supplies a balanced period-major panel and a single, time-invariant entity graph.
 * Coefficients are common across periods and entities. This selected panel estimator does not
 * estimate time fixed effects, dynamic lags or random effects. With effects: unit, temporal
 * within transformation absorbs entity intercepts before spatial 2SLS. The intercept table row
 * and pooled AK diagnostics are then NaN; covariance uses n - entities - predictors - 1 degrees.
 * Inference is homoskedastic, without entity clustering or serial-correlation correction.
 * Pooled mode is directly comparable to spreg.GM_Lag on block-diagonal weights.
 * See docs/api/gpu-spatial-analysis/panel-fixed-effects.md.
 */
export class GPUSpatialPanelRegression implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUSpatialPanelRegressionProps) {
    this.id = props.id ?? 'spatial-panel-regression';
    const entities = validateGPUSpatialWeights(this.id, props.weights);
    if (props.effects !== undefined && props.effects !== 'pooled' && props.effects !== 'unit') {
      throw new Error(`${this.id}: unsupported panel effects`);
    }
    validateCapacity(
      this.id,
      'periodCount',
      props.periodCount,
      2,
      PANEL_REGRESSION_MAXIMUM_PERIOD_COUNT
    );
    validateCapacity(
      this.id,
      'entityCount * periodCount',
      entities * props.periodCount,
      1,
      PANEL_REGRESSION_MAXIMUM_ROW_COUNT
    );
    validateCapacity(
      this.id,
      'expanded edgeCount',
      props.weights.neighbors.length * props.periodCount,
      0,
      PANEL_REGRESSION_MAXIMUM_EDGE_COUNT
    );
    validateRequirement(
      this.id,
      props.instrumentOrder === undefined ||
        props.instrumentOrder === 1 ||
        props.instrumentOrder === 2,
      'instrumentOrder must be 1 or 2'
    );
    validateCapacity(
      this.id,
      'predictorCount',
      props.predictorCount,
      1,
      getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount(props.instrumentOrder ?? 1)
    );
    for (const [name, view, format, minimum] of [
      ['table', props.output.table, 'float32', (props.predictorCount + 2) * 4],
      ['summary', props.output.summary, 'float32', 16],
      ['status.code', props.output.status.code, 'uint32', 1],
      ['residuals', props.output.residuals, 'float32', entities * props.periodCount]
    ] as const) {
      if (!view) continue;
      validateSinglePackedView(view, [format], this.id);
      validateRequirement(
        this.id,
        view.length >= minimum,
        `output.${name} length must be at least ${minimum}`
      );
    }
    validateSinglePackedView(props.predictors, ['float32'], this.id);
    validateSinglePackedView(props.response, ['float32'], this.id);
    if (
      props.response.length !== entities * props.periodCount ||
      props.predictors.length !== props.response.length * props.predictorCount
    )
      throw new Error(`${this.id}: panel must be balanced and period-major`);
    validateRegressionOutputDispatches(
      this.id,
      {...props.output, status: undefined, code: props.output.status.code},
      [['table', 'summary', 'code']]
    );
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [props.output.table, props.output.summary, props.output.residuals, props.output.status.code],
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.weights.status?.overflow,
        props.predictors,
        props.response
      ]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const entities = props.weights.offsets.length - 1;
    const rows = entities * props.periodCount;
    const capacity = props.weights.neighbors.length;
    validateGraphChunkBindingRanges(graph, id, {
      predictors: props.predictors,
      response: props.response,
      offsets: props.weights.offsets,
      neighbors: props.weights.neighbors,
      weights: props.weights.weights,
      overflow: props.weights.status?.overflow,
      ...props.output,
      status: undefined,
      code: props.output.status.code
    });
    validateRegressionAllocations(graph.device, id, {
      ...getRegressionTwoStageAllocationLengths(
        rows,
        props.predictorCount,
        capacity * props.periodCount,
        props.instrumentOrder ?? 1
      ),
      'expanded offsets': rows + 1,
      'expanded edges': capacity * props.periodCount,
      'panel row scratch': rows,
      ...(props.effects === 'unit'
        ? {
            'within predictors': props.predictors.length,
            'unit means': entities * (props.predictorCount + 1)
          }
        : {})
    });
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
    validateGraphViewsBelongToGraph(id, graph, [
      props.weights.offsets,
      props.weights.neighbors,
      props.weights.weights,
      props.weights.status?.overflow,
      props.predictors,
      props.response
    ]);
    const nodes = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-expand-weights`,
        operation: 'GPUSpatialPanelRegression',
        variant: 'expand-weights',
        invocationCount: rows + 1,
        bindings: [
          {name: 'sourceOffsets', view: props.weights.offsets, type: 'u32', access: 'read'},
          {name: 'sourceNeighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'sourceWeights', view: props.weights.weights, type: 'f32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read_write'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'},
          {name: 'weights', view: weights, type: 'f32', access: 'read_write'}
        ],
        body: `let edgeCount = min(sourceOffsets[sourceOffsetsOffset + ${entities}u], ${capacity}u);
if (index == ${rows}u) { offsets[offsetsOffset + index] = edgeCount * ${props.periodCount}u; return; }
let period = index / ${entities}u; let entity = index % ${entities}u;
let start = min(sourceOffsets[sourceOffsetsOffset + entity], edgeCount); let end = min(sourceOffsets[sourceOffsetsOffset + entity + 1u], edgeCount);
offsets[offsetsOffset + index] = period * edgeCount + start;
for (var slot = start; slot < end; slot++) {
  let destination = period * edgeCount + slot;
  let neighbor = sourceNeighbors[sourceNeighborsOffset + slot];
  neighbors[neighborsOffset + destination] = select(${rows}u, period * ${entities}u + neighbor, neighbor < ${entities}u);
  weights[weightsOffset + destination] = sourceWeights[sourceWeightsOffset + slot];
}`
      })
    ];
    let response = props.response;
    if (props.weights.status?.overflow) {
      response = createTransientView(graph, `${id}-checked-response`, 'float32', rows);
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-check-weights`,
          operation: 'GPUSpatialPanelRegression',
          variant: 'check-weights',
          invocationCount: rows,
          bindings: [
            {name: 'source', view: props.response, type: 'f32', access: 'read'},
            {name: 'overflow', view: props.weights.status.overflow, type: 'u32', access: 'read'},
            {name: 'response', view: response, type: 'f32', access: 'read_write'}
          ],
          declarations: GET_NAN_WGSL,
          body: 'response[responseOffset + index] = select(source[sourceOffset + index], getNaN(), overflow[overflowOffset] != 0u);'
        })
      );
    }
    let predictors = props.predictors;
    if (props.effects === 'unit') {
      predictors = createTransientView(
        graph,
        `${id}-within-predictors`,
        'float32',
        props.predictors.length
      );
      const withinResponse = createTransientView(graph, `${id}-within-response`, 'float32', rows);
      const means = createTransientView(
        graph,
        `${id}-unit-means`,
        'float32',
        entities * (props.predictorCount + 1)
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-unit-means`,
          operation: 'GPUSpatialPanelRegression',
          variant: 'unit-means',
          invocationCount: entities,
          bindings: [
            {name: 'sourcePredictors', view: props.predictors, type: 'f32', access: 'read'},
            {name: 'sourceResponse', view: response, type: 'f32', access: 'read'},
            {name: 'means', view: means, type: 'f32', access: 'read_write'}
          ],
          body: `var responseMean = 0.0;
for (var period = 0u; period < ${props.periodCount}u; period++) {
  responseMean += sourceResponse[sourceResponseOffset + period * ${entities}u + index];
}
means[meansOffset + index * ${props.predictorCount + 1}u] = responseMean / ${props.periodCount}.0;
for (var column = 0u; column < ${props.predictorCount}u; column++) {
  var predictorMean = 0.0;
  for (var period = 0u; period < ${props.periodCount}u; period++) {
    predictorMean += sourcePredictors[sourcePredictorsOffset + (period * ${entities}u + index) * ${props.predictorCount}u + column];
  }
  means[meansOffset + index * ${props.predictorCount + 1}u + column + 1u] = predictorMean / ${props.periodCount}.0;
}`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-within`,
          operation: 'GPUSpatialPanelRegression',
          variant: 'unit-within',
          invocationCount: rows,
          bindings: [
            {name: 'sourcePredictors', view: props.predictors, type: 'f32', access: 'read'},
            {name: 'sourceResponse', view: response, type: 'f32', access: 'read'},
            {name: 'means', view: means, type: 'f32', access: 'read'},
            {name: 'predictors', view: predictors, type: 'f32', access: 'read_write'},
            {name: 'response', view: withinResponse, type: 'f32', access: 'read_write'}
          ],
          body: `let entity = index % ${entities}u;
response[responseOffset + index] = sourceResponse[sourceResponseOffset + index] - means[meansOffset + entity * ${props.predictorCount + 1}u];
for (var column = 0u; column < ${props.predictorCount}u; column++) {
  predictors[predictorsOffset + index * ${props.predictorCount}u + column] = sourcePredictors[sourcePredictorsOffset + index * ${props.predictorCount}u + column] - means[meansOffset + entity * ${props.predictorCount + 1}u + column + 1u];
}`
        })
      );
      response = withinResponse;
    }
    nodes.push(
      ...new GPUSpatialTwoStageLeastSquares({
        id: `${id}-pooled-lag`,
        weights: {offsets, neighbors, weights, status: props.weights.status},
        predictors,
        response,
        predictorCount: props.predictorCount,
        instrumentOrder: props.instrumentOrder,
        output: props.output
      }).getCommandNodes(graph)
    );
    if (props.effects === 'unit') {
      const degrees = rows - entities - props.predictorCount - 1;
      const instrumentCount = ((props.instrumentOrder ?? 1) + 1) * props.predictorCount;
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-within-finish`,
          operation: 'GPUSpatialPanelRegression',
          variant: 'unit-finish',
          invocationCount: 1,
          bindings: [
            {name: 'table', view: props.output.table, type: 'f32', access: 'read_write'},
            {name: 'summary', view: props.output.summary, type: 'f32', access: 'read_write'},
            {name: 'status', view: props.output.status.code, type: 'u32', access: 'read_write'}
          ],
          declarations: `${GET_NAN_WGSL}
${ERFC_VALUE_WGSL}`,
          body: `let nan = getNaN();
if (${degrees} <= 0) {
  status[statusOffset] = 2u;
  for (var slot = 0u; slot < ${(props.predictorCount + 2) * 4}u; slot++) { table[tableOffset + slot] = nan; }
  for (var slot = 1u; slot < 16u; slot++) { summary[summaryOffset + slot] = nan; }
} else if (status[statusOffset] == 0u) {
  let scale = sqrt(${rows}.0 / ${Math.max(degrees, 1)}.0);
  summary[summaryOffset + 1u] *= scale * scale;
  for (var row = 1u; row < ${props.predictorCount + 2}u; row++) {
    let standardError = table[tableOffset + row * 4u + 1u] * scale;
    let statistic = table[tableOffset + row * 4u] / standardError;
    table[tableOffset + row * 4u + 1u] = standardError;
    table[tableOffset + row * 4u + 2u] = statistic;
    table[tableOffset + row * 4u + 3u] = erfcValue(abs(statistic) * 0.70710678);
  }
  // First-stage residual degrees account for the absorbed entity intercepts.
  ${rows - entities - instrumentCount > 0 ? `summary[summaryOffset + 14u] *= ${rows - entities - instrumentCount}.0 / ${rows - instrumentCount - 1}.0;` : 'summary[summaryOffset + 14u] = nan;'}
}
for (var slot = 0u; slot < 4u; slot++) { table[tableOffset + slot] = nan; }
// Pooled AK inference does not account for the within projection.
for (var slot = 4u; slot < 7u; slot++) { summary[summaryOffset + slot] = nan; }`
        })
      );
      if (degrees <= 0 && props.output.residuals) {
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-within-clear-residuals`,
            operation: 'GPUSpatialPanelRegression',
            variant: 'unit-clear-residuals',
            invocationCount: rows,
            bindings: [
              {name: 'residuals', view: props.output.residuals, type: 'f32', access: 'read_write'}
            ],
            declarations: GET_NAN_WGSL,
            body: 'residuals[residualsOffset + index] = getNaN();'
          })
        );
      }
    }
    return nodes;
  }
}
