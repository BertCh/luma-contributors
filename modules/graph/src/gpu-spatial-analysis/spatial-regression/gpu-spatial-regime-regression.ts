// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {getCholeskyWGSL} from './spatial-regression-solve';

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
 * then recover the intercept; this avoids cancellation from large coordinate-like predictor means.
 * Equivalent to fully varying spreg.OLS_Regimes point estimates (no spatial lag/error term,
 * covariance estimates, pooled restrictions or Chow test). Empty/singular regimes return NaNs.
 */
export class GPUSpatialRegimeRegression implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUSpatialRegimeRegressionProps) {
    this.id = props.id ?? 'spatial-regime-regression';
    const rows = props.response.length;
    if (
      !Number.isInteger(props.predictorCount) ||
      props.predictorCount < 1 ||
      props.predictorCount > 11 ||
      !Number.isInteger(props.regimeCount) ||
      props.regimeCount < 1 ||
      props.regimeCount > 4096 ||
      rows < 1 ||
      rows > 65536
    )
      throw new Error(`${this.id}: invalid regime regression shape`);
    for (const [view, format, length] of [
      [props.predictors, 'float32', rows * props.predictorCount],
      [props.response, 'float32', rows],
      [props.regimes, 'uint32', rows],
      [props.mask, 'uint32', rows],
      [props.output.coefficients, 'float32', props.regimeCount * (props.predictorCount + 1)],
      [props.output.fitted, 'float32', rows],
      [props.output.status, 'uint32', props.regimeCount],
      [props.output.counts, 'uint32', props.regimeCount]
    ] as const) {
      if (view) {
        validatePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: view length mismatch`);
      }
    }
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
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-fit`,
        operation: 'GPUSpatialRegimeRegression',
        variant: 'fit',
        invocationCount: props.regimeCount,
        bindings: [
          ...bindings,
          {
            name: 'coefficients',
            view: props.output.coefficients,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'}
        ],
        declarations: `${common}\n${getCholeskyWGSL(columns)}`,
        body: `var means: array<f32, ${columns}>; var responseMean = 0.0; var count = 0u;
for (var row = 0u; row < ${rows}u; row++) {
  if (!included(row) || regimes[regimesOffset + row] != index) { continue; }
  count++; responseMean += (response[responseOffset + row] - responseMean) / f32(count);
  for (var column = 0u; column < ${columns}u; column++) { means[column] += (predictors[predictorsOffset + row * ${columns}u + column] - means[column]) / f32(count); }
}
counts[countsOffset + index] = count;
for (var column = 0u; column < ${columns + 1}u; column++) { coefficients[coefficientsOffset + index * ${columns + 1}u + column] = getNaN(); }
status[statusOffset + index] = 1u; if (count <= ${columns}u) { return; }
var matrix: array<f32, ${columns * columns}>; var right: array<f32, ${columns}>;
for (var row = 0u; row < ${rows}u; row++) {
  if (!included(row) || regimes[regimesOffset + row] != index) { continue; }
  for (var column = 0u; column < ${columns}u; column++) {
    let value = predictors[predictorsOffset + row * ${columns}u + column] - means[column];
    right[column] += value * (response[responseOffset + row] - responseMean);
    for (var other = 0u; other <= column; other++) { matrix[column * ${columns}u + other] += value * (predictors[predictorsOffset + row * ${columns}u + other] - means[other]); }
  }
}
status[statusOffset + index] = 2u; if (!choleskyFactor_${columns}(&matrix)) { return; }
choleskySolve_${columns}(&matrix, &right); var intercept = responseMean;
for (var column = 0u; column < ${columns}u; column++) { if (!isFiniteBits(right[column])) { return; } intercept -= means[column] * right[column]; }
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
          {name: 'fitted', view: props.output.fitted, type: 'f32', access: 'read_write'}
        ],
        declarations: common,
        body: `var value = getNaN();
if (included(index)) { let base = coefficientsOffset + regimes[regimesOffset + index] * ${columns + 1}u; value = coefficients[base];
  for (var column = 0u; column < ${columns}u; column++) { value += predictors[predictorsOffset + index * ${columns}u + column] * coefficients[base + column + 1u]; }
}
fitted[fittedOffset + index] = value;`
      })
    ];
  }
}
