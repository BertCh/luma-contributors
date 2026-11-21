// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRegressionOutputDispatches} from './regression-output-dispatches';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import {validateRegressionAllocations} from './spatial-likelihood-allocations';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCholeskyWGSL, getPivotLimitedCholeskyWGSL} from './spatial-regression-solve';
import {getRegimeMomentNodes, getRegimeMomentStride} from './regime-moments';
import {REGRESSION_TILE_MAXIMUM_ROW_COUNT} from './regression-row-tiles';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';
import {
  CENTERED_REGRESSION_PREDICTION_WGSL,
  getCenteredRegressionContributionWGSL
} from './centered-regression-prediction';

/** Largest predictor count: the per-regime solve keeps an 11 x 11 Cholesky factor per thread. */
const MAXIMUM_PREDICTOR_COUNT = 11;
/** Largest regime count: bounds the per-regime outputs and the moment workgroups. */
const MAXIMUM_REGIME_COUNT = 4096;

export type GPUSpatialRegimeRegressionProps = {
  id?: string;
  predictors: GraphDataView<'float32'>;
  predictorCount: number;
  response: GraphDataView<'float32'>;
  /** Caller-defined contiguous spatial regime IDs in [0, regimeCount); other IDs excluded. */
  regimes: GraphDataView<'uint32'>;
  regimeCount: number;
  mask?: GraphDataView<'uint32'>;
  output: {
    /** Regime-major coefficients, intercept first. */
    coefficients: GraphDataView<'float32'>;
    fitted: GraphDataView<'float32'>;
    /** Per-regime: 0 fitted, 1 insufficient rows, 2 singular/nonfinite. */
    status: GraphDataView<'uint32'>;
    counts: GraphDataView<'uint32'>;
  };
};
/**
 * Fully varying spatial-regime OLS: separate intercept and slopes for each supplied region.
 * Complete cases are selected jointly. Fits center predictors within each regime before Cholesky,
 * then recover the intercept. Moments and fitted values retain the mean correction, and prediction
 * uses centered predictors; this avoids cancellation from large coordinate-like predictor means.
 * Equivalent to fully varying spreg.OLS_Regimes point estimates (no spatial lag/error term,
 * covariance estimates, pooled restrictions or Chow test). Empty/singular regimes return NaNs.
 * Counts, view identities and the output map are snapshotted at construction; buffer contents
 * remain per-frame data.
 * A regime is singular (status 2) when a centered predictor has no variance or a Cholesky pivot,
 * relative to its diagonal, is below `GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE` (`1e-3`): the
 * limit of `GPUOrdinaryLeastSquares`, a predictor within `1 - 1e-6` (squared correlation) of the
 * span of the earlier ones.
 *
 * The per-regime moments are accumulated in parallel and in a fixed order: the complete rows are
 * stably sorted by regime (published `GPUSort`), then one workgroup per regime sums its rows with
 * strided thread partials and a fixed binary tree (means, then centered cross-products), so the
 * result is bitwise reproducible. One invocation per regime then solves its small system. Rows are
 * limited by float32 counts (`2^24`) and by the predictor binding of the device.
 */
export class GPUSpatialRegimeRegression implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUSpatialRegimeRegressionProps) {
    const captured = snapshotConstructionFields(props, [
      'id',
      'predictors',
      'predictorCount',
      'response',
      'regimes',
      'regimeCount',
      'mask',
      'output'
    ]);
    this.props = Object.freeze({
      ...captured,
      output: snapshotConstructionFields(captured.output, [
        'coefficients',
        'fitted',
        'status',
        'counts'
      ])
    });
    props = this.props;
    this.id = props.id ?? 'spatial-regime-regression';
    const rows = props.response.length;
    validateCapacity(this.id, 'predictorCount', props.predictorCount, 1, MAXIMUM_PREDICTOR_COUNT);
    validateCapacity(this.id, 'regimeCount', props.regimeCount, 1, MAXIMUM_REGIME_COUNT);
    validateCapacity(this.id, 'response rows', rows, 1, REGRESSION_TILE_MAXIMUM_ROW_COUNT);
    for (const [name, view, format, length] of [
      ['predictors', props.predictors, 'float32', rows * props.predictorCount],
      ['response', props.response, 'float32', rows],
      ['regimes', props.regimes, 'uint32', rows],
      ['mask', props.mask, 'uint32', rows],
      [
        'output.coefficients',
        props.output.coefficients,
        'float32',
        props.regimeCount * (props.predictorCount + 1)
      ],
      ['output.fitted', props.output.fitted, 'float32', rows],
      ['output.status', props.output.status, 'uint32', props.regimeCount],
      ['output.counts', props.output.counts, 'uint32', props.regimeCount]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, [format], this.id);
        validateViewLength(this.id, name, view, length);
      }
    }
    validateRegressionOutputDispatches(this.id, props.output, [
      ['coefficients', 'status', 'counts'],
      {read: ['coefficients', 'status'], write: ['fitted']}
    ]);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.predictors,
      props.response,
      props.regimes,
      props.mask
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const rows = props.response.length;
    const columns = props.predictorCount;
    validateGraphChunkBindingRanges(graph, id, {
      predictors: props.predictors,
      response: props.response,
      regimes: props.regimes,
      mask: props.mask,
      ...props.output
    });
    validateRegressionAllocations(graph.device, id, {
      'regime row sorting and radix/scan scratch': rows,
      'regime offsets': props.regimeCount + 1,
      'regime moments': props.regimeCount * getRegimeMomentStride(columns)
    });
    // The predictor view is the largest binding: rows * predictorCount float32 values.
    if (rows > getStorageBindingElementLimit(graph.device, 4 * columns))
      throw new Error(`${id}: invalid regime regression shape`);
    const bindings = [
      {name: 'predictors', view: props.predictors, type: 'f32' as const, access: 'read' as const},
      {name: 'response', view: props.response, type: 'f32' as const, access: 'read' as const},
      {name: 'regimes', view: props.regimes, type: 'u32' as const, access: 'read' as const},
      ...(props.mask
        ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
        : [])
    ];
    validateGraphViewsBelongToGraph(id, graph, [
      ...bindings.map(binding => binding.view),
      ...Object.values(props.output)
    ]);
    const common = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
fn included(row: u32) -> bool {
  ${props.mask ? 'if (mask[maskOffset + row] == 0u) { return false; }' : ''}
  if (!isFiniteBits(response[responseOffset + row]) || regimes[regimesOffset + row] >= ${props.regimeCount}u) { return false; }
  for (var column = 0u; column < ${columns}u; column++) { if (!isFiniteBits(predictors[predictorsOffset + row * ${columns}u + column])) { return false; } }
  return true;
}`;
    const {nodes, views} = getRegimeMomentNodes<Parameters>(graph, {
      id,
      operation: 'GPUSpatialRegimeRegression',
      predictors: props.predictors,
      predictorCount: columns,
      response: props.response,
      regimes: props.regimes,
      regimeCount: props.regimeCount,
      mask: props.mask
    });
    const dimension = columns + 1;
    const stride = getRegimeMomentStride(columns);
    return [
      ...nodes,
      createWGSLKernelNode(graph, {
        id: `${id}-fit`,
        operation: 'GPUSpatialRegimeRegression',
        variant: 'fit',
        invocationCount: props.regimeCount,
        bindings: [
          {name: 'segmentOffsets', view: views.segmentOffsets, type: 'u32', access: 'read'},
          {name: 'moments', view: views.moments, type: 'f32', access: 'read'},
          {
            name: 'coefficients',
            view: props.output.coefficients,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${getCholeskyWGSL(columns)}\n${getPivotLimitedCholeskyWGSL(columns)}
const D: u32 = ${dimension}u;
${ROUNDED_DIFFERENCE_WGSL}
fn momentAt(base: u32, first: u32, second: u32, count: u32) -> f32 {
  let low = min(first, second);
  let high = max(first, second);
  let deviations = base + ${dimension + (dimension * (dimension + 1)) / 2}u;
  // Fix the quotient before multiplying: the product of two deviation sums can overflow
  // even when the corrected moment fits in f32. The first segment offset is a runtime zero.
  let secondCorrection = getRoundedDifference(moments[deviations + second] / max(f32(count), 1.0), 0.0, segmentOffsets[segmentOffsetsOffset]);
  return moments[base + D + low * (2u * D - low + 1u) / 2u + (high - low)]
    - moments[deviations + first] * secondCorrection;
}`,
        body: `let count = segmentOffsets[segmentOffsetsOffset + index + 1u] - segmentOffsets[segmentOffsetsOffset + index];
let base = momentsOffset + index * ${stride}u;
counts[countsOffset + index] = count;
for (var column = 0u; column < ${columns + 1}u; column++) { coefficients[coefficientsOffset + index * ${columns + 1}u + column] = getNaN(); }
status[statusOffset + index] = 1u; if (count <= ${columns}u) { return; }
var matrix: array<f32, ${columns * columns}>; var right: array<f32, ${columns}>;
for (var column = 0u; column < ${columns}u; column++) {
  right[column] = momentAt(base, column, ${columns}u, count);
  for (var other = 0u; other <= column; other++) { matrix[column * ${columns}u + other] = momentAt(base, column, other, count); }
}
let responseMean = moments[base + ${columns}u];
status[statusOffset + index] = 2u; if (!choleskyFactorLimited_${columns}(&matrix)) { return; }
choleskySolve_${columns}(&matrix, &right);
let deviations = base + ${dimension + (dimension * (dimension + 1)) / 2}u;
var intercept = responseMean;
var correction = moments[deviations + ${columns}u] / f32(count);
let runtimeZero = segmentOffsets[segmentOffsetsOffset];
for (var column = 0u; column < ${columns}u; column++) {
  if (!isFiniteBits(right[column])) { return; }
  intercept = getRoundedDifference(intercept, moments[base + column] * right[column], runtimeZero);
  correction -= right[column] * (moments[deviations + column] / f32(count));
}
intercept += correction;
if (!isFiniteBits(intercept)) { return; }
coefficients[coefficientsOffset + index * ${columns + 1}u] = intercept;
for (var column = 0u; column < ${columns}u; column++) { coefficients[coefficientsOffset + index * ${columns + 1}u + column + 1u] = right[column]; }
status[statusOffset + index] = 0u;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-predict`,
        operation: 'GPUSpatialRegimeRegression',
        variant: 'predict',
        invocationCount: rows,
        bindings: [
          ...bindings,
          {name: 'coefficients', view: props.output.coefficients, type: 'f32', access: 'read'},
          {name: 'moments', view: views.moments, type: 'f32', access: 'read'},
          {name: 'segmentOffsets', view: views.segmentOffsets, type: 'u32', access: 'read'},
          {name: 'fitted', view: props.output.fitted, type: 'f32', access: 'read_write'}
        ],
        declarations: `${common}\n${ROUNDED_DIFFERENCE_WGSL}\n${CENTERED_REGRESSION_PREDICTION_WGSL}`,
        body: `var value = getNaN();
if (included(index)) {
  let regime = regimes[regimesOffset + index];
  let base = coefficientsOffset + regime * ${columns + 1}u;
  if (!isFiniteBits(coefficients[base])) { fitted[fittedOffset + index] = value; return; }
  let momentBase = momentsOffset + regime * ${stride}u;
  let deviations = momentBase + ${dimension + (dimension * (dimension + 1)) / 2}u;
  let count = f32(segmentOffsets[segmentOffsetsOffset + regime + 1u] - segmentOffsets[segmentOffsetsOffset + regime]);
  var correction = moments[deviations + ${columns}u] / count;
  for (var column = 0u; column < ${columns}u; column++) {
    correction -= coefficients[base + column + 1u] * moments[deviations + column] / count;
  }
  // The first sorted segment starts at zero; reading it keeps differences rounded on Metal.
  ${getCenteredRegressionContributionWGSL({predictorCount: columns, predictor: 'predictors[predictorsOffset + index * ' + columns + 'u + column]', mean: 'moments[momentBase + column]', slope: 'coefficients[base + column + 1u]', runtimeZero: 'segmentOffsets[segmentOffsetsOffset]'})}
  value = getCenteredRegressionPrediction(moments[momentBase + ${columns}u], centeredContribution, correction, segmentOffsets[segmentOffsetsOffset]);
}
fitted[fittedOffset + index] = value;`
      })
    ];
  }
}
