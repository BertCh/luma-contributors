// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_MORAN_I,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE,
  getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount
} from './spatial-two-stage-least-squares-parameters';
import {validateRegressionOutputDispatches} from './regression-output-dispatches';
import {snapshotRegressionConstructionPropsWithCompleteness as snapshotPanelConstructionProps} from './regression-certificate-snapshot';

import {
  getRegressionCompletenessViews,
  getRegressionWeightsLossNodes
} from './regression-weights-completeness';
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
import {getPanelWithinNodes, getPanelWithinScratchLengths} from './panel-within-nodes';

/** At most p+2=10 four-field coefficient rows; independent of the 16 instrument columns. */
export const GPU_PANEL_REGRESSION_MAXIMUM_TABLE_ROW_COUNT =
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT + 2;
/** Preserve the existing period limit while using the shared tiled within transform. */
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
  readonly props: GPUSpatialPanelRegressionProps;
  constructor(props: GPUSpatialPanelRegressionProps) {
    props = snapshotPanelConstructionProps(
      props,
      [
        'id',
        'weights',
        'periodCount',
        'effects',
        'predictorCount',
        'predictors',
        'response',
        'instrumentOrder',
        'output'
      ],
      ['table', 'summary', 'status', 'residuals'],
      'spatial-panel-regression'
    );
    this.props = props;
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
      [
        'table',
        props.output.table,
        'float32',
        (props.predictorCount + 2) * GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE
      ],
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
        ...getRegressionCompletenessViews(props.weights),
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
      candidateOverflow: props.weights.status?.candidateOverflow,
      requiredCount: props.weights.status?.requiredCount,
      count: props.weights.status?.count,
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
            'panel mean partials': getPanelWithinScratchLengths(
              entities,
              props.periodCount,
              props.predictorCount
            ).partialLength,
            'panel means': getPanelWithinScratchLengths(
              entities,
              props.periodCount,
              props.predictorCount
            ).meanLength
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
      ...getRegressionCompletenessViews(props.weights),
      props.predictors,
      props.response
    ]);
    const completeness = getRegressionWeightsLossNodes(graph, `${id}-entity`, props.weights);
    const nodes = [
      ...completeness.nodes,
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
    // Expansion clamps memory accesses, but clamping must not turn malformed CSR into a
    // different, apparently complete spatial operator. Check original bounds before fitting.
    let response = createTransientView(graph, `${id}-checked-response`, 'float32', rows);
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-check-weights`,
        operation: 'GPUSpatialPanelRegression',
        variant: 'check-weights',
        invocationCount: rows,
        bindings: [
          {name: 'source', view: props.response, type: 'f32', access: 'read'},
          {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
          ...(props.weights.status?.overflow
            ? [
                {
                  name: 'overflow',
                  view: props.weights.status.overflow,
                  type: 'u32',
                  access: 'read'
                } as const
              ]
            : []),
          ...(props.weights.status?.candidateOverflow
            ? [
                {
                  name: 'candidateOverflow',
                  view: props.weights.status.candidateOverflow,
                  type: 'u32',
                  access: 'read'
                } as const
              ]
            : []),
          {name: 'response', view: response, type: 'f32', access: 'read_write'}
        ],
        declarations: GET_NAN_WGSL,
        body: `let entity = index % ${entities}u;
let begin = offsets[offsetsOffset + entity]; let end = offsets[offsetsOffset + entity + 1u];
let edgeCount = offsets[offsetsOffset + ${entities}u];
var invalid = offsets[offsetsOffset] != 0u || edgeCount > ${capacity}u || end < begin || end > edgeCount${props.weights.status?.overflow ? ' || overflow[overflowOffset] != 0u' : ''}${props.weights.status?.candidateOverflow ? ' || candidateOverflow[candidateOverflowOffset] != 0u' : ''};
if(!invalid){for(var slot=begin;slot<end;slot++){invalid=invalid || neighbors[neighborsOffset+slot]>=${entities}u;}}
response[responseOffset + index] = select(source[sourceOffset + index], getNaN(), invalid);`
      })
    );
    let predictors = props.predictors;
    if (props.effects === 'unit') {
      predictors = createTransientView(
        graph,
        `${id}-within-predictors`,
        'float32',
        props.predictors.length
      );
      const withinResponse = createTransientView(graph, `${id}-within-response`, 'float32', rows);
      nodes.push(
        ...getPanelWithinNodes(graph, {
          id,
          entityCount: entities,
          periodCount: props.periodCount,
          predictorCount: props.predictorCount,
          predictors: props.predictors,
          response,
          output: {predictors, response: withinResponse}
        })
      );
      response = withinResponse;
    }
    nodes.push(
      ...new GPUSpatialTwoStageLeastSquares({
        id: `${id}-pooled-lag`,
        weights: {
          offsets,
          neighbors,
          weights,
          status: completeness.loss ? {overflow: completeness.loss} : undefined
        },
        predictors,
        response,
        predictorCount: props.predictorCount,
        instrumentOrder: props.instrumentOrder,
        output: props.output
      }).getCommandNodes(graph)
    );
    if (props.effects === 'unit') {
      validateCapacity(
        id,
        'table rows',
        props.predictorCount + 2,
        1,
        GPU_PANEL_REGRESSION_MAXIMUM_TABLE_ROW_COUNT
      );
      const degrees = rows - entities - props.predictorCount - 1;
      const instrumentCount = ((props.instrumentOrder ?? 1) + 1) * props.predictorCount;
      nodes.push(
        createWGSLKernelNode(graph, {
          // contract-allow: single-invocation -- GPU_PANEL_REGRESSION_MAXIMUM_TABLE_ROW_COUNT = 10 four-field rows (p+2); GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE = 4; GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH = 16 bounds summary endpoints; GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_CENTERED_INSTRUMENT_COLUMN_COUNT = 16 bounds instruments separately.
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
  for (var slot = 0u; slot < ${(props.predictorCount + 2) * GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE}u; slot++) { table[tableOffset + slot] = nan; }
  for (var slot = ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED}u; slot < ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH}u; slot++) { summary[summaryOffset + slot] = nan; }
} else if (status[statusOffset] == 0u) {
  let scale = sqrt(${rows}.0 / ${Math.max(degrees, 1)}.0);
  summary[summaryOffset + 1u] *= scale * scale;
  for (var row = 1u; row < ${props.predictorCount + 2}u; row++) {
    let standardError = table[tableOffset + row * ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE}u + 1u] * scale;
    let statistic = table[tableOffset + row * ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE}u] / standardError;
    table[tableOffset + row * ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE}u + 1u] = standardError;
    table[tableOffset + row * ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE}u + 2u] = statistic;
    table[tableOffset + row * ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE}u + 3u] = erfcValue(abs(statistic) * 0.70710678);
  }
  // First-stage residual degrees account for the absorbed entity intercepts.
  ${rows - entities - instrumentCount > 0 ? `summary[summaryOffset + 14u] *= ${rows - entities - instrumentCount}.0 / ${rows - instrumentCount - 1}.0;` : 'summary[summaryOffset + 14u] = nan;'}
}
for (var slot = 0u; slot < ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE}u; slot++) { table[tableOffset + slot] = nan; }
// Pooled AK inference does not account for the within projection.
for (var slot = ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_MORAN_I}u; slot < ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER}u; slot++) { summary[summaryOffset + slot] = nan; }`
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
