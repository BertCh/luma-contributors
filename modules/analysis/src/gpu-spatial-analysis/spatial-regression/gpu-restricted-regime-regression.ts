// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL,
  IS_NAN_BITS_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {GPU_UPPER_INCOMPLETE_GAMMA_WGSL} from '../distribution-dynamics/incomplete-gamma-wgsl';
import {getCholeskyWGSL} from './spatial-regression-solve';

export type GPURestrictedRegimeRegressionProps = {
  id?: string;
  predictors: GraphDataView<'float32'>;
  predictorCount: number;
  response: GraphDataView<'float32'>;
  regimes: GraphDataView<'uint32'>;
  regimeCount: number;
  /** Intercept first. True shares a coefficient across regimes, false estimates one per regime. */
  common: readonly boolean[];
  mask?: GraphDataView<'uint32'>;
  output: {
    /** Source-column-major: one slot if common, otherwise regimeCount slots. */
    coefficients: GraphDataView<'float32'>;
    /** Full covariance of the restricted model, pooled homoskedastic SSE/(n-parameterCount). */
    covariance: GraphDataView<'float32'>;
    fitted: GraphDataView<'float32'>;
    /** Joint Chow F statistic, numerator degrees, denominator degrees. NaN F if no restrictions. */
    chow: GraphDataView<'float32'>;
    /** Optional coefficient stability Wald tests [chi-square, degrees, upper-tail probability].
     * One row per source column (intercept first), then a joint row. Shared columns have df 0/NaN.
     * Tests equality of the varying coefficients under this model's pooled covariance, as spreg Chow.
     */
    stability?: GraphDataView<'float32'>;
    /** 0 fitted; 1 insufficient rows; 2 restricted singular; 3 unrestricted singular. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Restricted OLS with any common/varying intercept and slopes, plus the joint nested-model Chow
 * F statistic against fully varying regimes on exactly the same complete observations. The test
 * assumes independent homoskedastic normal errors; no spatial correction or p-value is implied.
 * Empty/singular regimes invalidate the unrestricted comparison explicitly. Total unrestricted
 * coefficients are bounded at 16. Design columns are norm-scaled, not implicitly ridge-regularized.
 */
export class GPURestrictedRegimeRegression implements GPUCommandNodeProducer {
  readonly id: string;
  readonly coefficientCount: number;
  readonly coefficientOffsets: readonly number[];
  constructor(readonly props: GPURestrictedRegimeRegressionProps) {
    this.id = props.id ?? 'restricted-regime-regression';
    validateCapacity(this.id, 'predictorCount', props.predictorCount, 1, 7);
    validateCapacity(this.id, 'regimeCount', props.regimeCount, 2, 8);
    validateCapacity(
      this.id,
      'unrestricted coefficients',
      (props.predictorCount + 1) * props.regimeCount,
      1,
      16
    );
    validateCapacity(this.id, 'rows', props.response.length, 1, 65536);
    if (
      props.common.length !== props.predictorCount + 1 ||
      !props.common.every(value => typeof value === 'boolean')
    )
      throw new Error(`${this.id}: common must include intercept and every predictor`);
    let count = 0;
    this.coefficientOffsets = props.common.map(common => {
      const offset = count;
      count += common ? 1 : props.regimeCount;
      return offset;
    });
    this.coefficientCount = count;
    const rows = props.response.length;
    for (const [view, format, length] of [
      [props.predictors, 'float32', rows * props.predictorCount],
      [props.response, 'float32', rows],
      [props.regimes, 'uint32', rows],
      [props.mask, 'uint32', rows],
      [props.output.coefficients, 'float32', count],
      [props.output.covariance, 'float32', count * count],
      [props.output.fitted, 'float32', rows],
      [props.output.chow, 'float32', 3],
      [props.output.stability, 'float32', (props.predictorCount + 2) * 3],
      [props.output.status, 'uint32', 1]
    ] as const)
      if (view) {
        validatePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
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
    const {id, props, coefficientCount: restricted} = this;
    const columns = props.predictorCount;
    const unrestricted = (columns + 1) * props.regimeCount;
    const rows = props.response.length;
    const common = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
const OFFSETS = array<u32, ${columns + 1}>(${this.coefficientOffsets.map(value => `${value}u`).join(', ')});
const COMMON = array<bool, ${columns + 1}>(${props.common.join(', ')});
fn included(row: u32) -> bool { if (regimes[regimesOffset + row] >= ${props.regimeCount}u || ${props.mask ? 'mask[maskOffset + row] == 0u || ' : ''}!isFiniteBits(response[responseOffset + row])) { return false; }
for (var column = 0u; column < ${columns}u; column++) { if (!isFiniteBits(predictors[predictorsOffset + row * ${columns}u + column])) { return false; } } return true; }
fn rawValue(row: u32, column: u32) -> f32 { if (column == 0u) { return 1.0; } return predictors[predictorsOffset + row * ${columns}u + column - 1u]; }
`;
    const fit = (name: string, size: number, constrained: boolean) => `
fn design_${name}(row: u32) -> array<f32, ${size}> { var values: array<f32, ${size}>; let regime = regimes[regimesOffset + row];
for (var column = 0u; column < ${columns + 1}u; column++) { let destination = ${constrained ? 'OFFSETS[column] + select(regime, 0u, COMMON[column])' : `column * ${props.regimeCount}u + regime`}; values[destination] = rawValue(row, column); } return values; }
fn fit_${name}(estimate: ptr<function, array<f32, ${size}>>, inverse: ptr<function, array<f32, ${size * size}>>, residualSquares: ptr<function, f32>) -> bool {
var normal: array<f32, ${size * size}>; var right: array<f32, ${size}>;
for (var row = 0u; row < ${rows}u; row++) { if (!included(row)) { continue; } let values = design_${name}(row); for (var first = 0u; first < ${size}u; first++) { right[first] += values[first] * response[responseOffset + row]; for (var second = 0u; second < ${size}u; second++) { normal[first * ${size}u + second] += values[first] * values[second]; } } }
var scales: array<f32, ${size}>;
for (var column = 0u; column < ${size}u; column++) { let diagonal = normal[column * ${size + 1}u]; if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) { return false; } scales[column] = inverseSqrt(diagonal); right[column] *= scales[column]; }
for (var first = 0u; first < ${size}u; first++) { for (var second = 0u; second < ${size}u; second++) { normal[first * ${size}u + second] *= scales[first] * scales[second]; } }
if (!choleskyFactor_${size}(&normal)) { return false; } choleskySolve_${size}(&normal, &right);
for (var column = 0u; column < ${size}u; column++) { (*estimate)[column] = right[column] * scales[column]; var unit: array<f32, ${size}>; unit[column] = scales[column]; choleskySolve_${size}(&normal, &unit); for (var row = 0u; row < ${size}u; row++) { (*inverse)[row * ${size}u + column] = unit[row] * scales[row]; } }
var squares = 0.0; for (var row = 0u; row < ${rows}u; row++) { if (!included(row)) { continue; } let values = design_${name}(row); var residual = response[responseOffset + row]; for (var column = 0u; column < ${size}u; column++) { residual -= values[column] * (*estimate)[column]; } squares += residual * residual; }
*residualSquares = squares; return isFiniteBits(squares);
}`;
    const inputs = [
      {name: 'predictors', view: props.predictors, type: 'f32' as const, access: 'read' as const},
      {name: 'response', view: props.response, type: 'f32' as const, access: 'read' as const},
      {name: 'regimes', view: props.regimes, type: 'u32' as const, access: 'read' as const},
      ...(props.mask
        ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
        : [])
    ];
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.response,
      props.regimes,
      props.mask,
      ...Object.values(props.output)
    ]);
    const nodes = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-fit`,
        operation: 'GPURestrictedRegimeRegression',
        variant: 'nested-models',
        invocationCount: 1,
        bindings: [
          ...inputs,
          {
            name: 'coefficients',
            view: props.output.coefficients,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'covariance', view: props.output.covariance, type: 'f32', access: 'read_write'},
          {name: 'chow', view: props.output.chow, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${common}\n${getCholeskyWGSL(restricted)}\n${unrestricted === restricted ? '' : getCholeskyWGSL(unrestricted)}\n${fit('restricted', restricted, true)}\n${fit('unrestricted', unrestricted, false)}`,
        body: `status[statusOffset] = 1u;
for (var slot = 0u; slot < ${restricted}u; slot++) { coefficients[coefficientsOffset + slot] = getNaN(); }
for (var slot = 0u; slot < ${restricted * restricted}u; slot++) { covariance[covarianceOffset + slot] = getNaN(); }
for (var slot = 0u; slot < 3u; slot++) { chow[chowOffset + slot] = getNaN(); }
var count = 0u; for (var row = 0u; row < ${rows}u; row++) { count += select(0u,1u,included(row)); } if (count <= ${unrestricted}u) { return; }
var estimate: array<f32, ${restricted}>; var inverse: array<f32, ${restricted * restricted}>; var squares = 0.0;
if (!fit_restricted(&estimate,&inverse,&squares)) { status[statusOffset] = 2u; return; }
let variance = squares / f32(count - ${restricted}u);
for (var slot = 0u; slot < ${restricted}u; slot++) { coefficients[coefficientsOffset + slot] = estimate[slot]; }
for (var slot = 0u; slot < ${restricted * restricted}u; slot++) { covariance[covarianceOffset + slot] = inverse[slot] * variance; }
var fullEstimate: array<f32, ${unrestricted}>; var fullInverse: array<f32, ${unrestricted * unrestricted}>; var fullSquares = 0.0;
if (!fit_unrestricted(&fullEstimate,&fullInverse,&fullSquares)) { status[statusOffset] = 3u; return; }
chow[chowOffset + 1u] = ${unrestricted - restricted}.0; chow[chowOffset + 2u] = f32(count - ${unrestricted}u);
${unrestricted > restricted ? `if (fullSquares > 0.0) { chow[chowOffset] = max(0.0, squares-fullSquares) / ${unrestricted - restricted}.0 / (fullSquares/f32(count-${unrestricted}u)); }` : ''}
status[statusOffset] = 0u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-predict`,
        operation: 'GPURestrictedRegimeRegression',
        variant: 'predict',
        invocationCount: rows,
        bindings: [
          ...inputs,
          {name: 'coefficients', view: props.output.coefficients, type: 'f32', access: 'read'},
          {name: 'fitted', view: props.output.fitted, type: 'f32', access: 'read_write'}
        ],
        declarations: common,
        body: `var prediction = getNaN(); if (included(index)) { prediction = 0.0; let regime = regimes[regimesOffset+index]; for (var column=0u; column<${columns + 1}u; column++) { let coefficient = OFFSETS[column]+select(regime,0u,COMMON[column]); prediction += rawValue(index,column)*coefficients[coefficientsOffset+coefficient]; } } fitted[fittedOffset+index]=prediction;`
      })
    ];

    if (props.output.stability) {
      const varying = props.common.flatMap((shared, column) => (shared ? [] : [column]));
      const contrasts = Math.max(1, varying.length * (props.regimeCount - 1));
      const contrastColumns = varying.flatMap(column =>
        Array.from({length: props.regimeCount - 1}, () => column)
      );
      const contrastOffsets = varying.flatMap(column =>
        Array.from(
          {length: props.regimeCount - 1},
          (_, regime) => this.coefficientOffsets[column] + regime + 1
        )
      );
      const contrastBases = varying.flatMap(column =>
        Array.from({length: props.regimeCount - 1}, () => this.coefficientOffsets[column])
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-stability`,
          operation: 'GPURestrictedRegimeRegression',
          variant: 'wald-stability',
          invocationCount: columns + 2,
          bindings: [
            {name: 'coefficients', view: props.output.coefficients, type: 'f32', access: 'read'},
            {name: 'covariance', view: props.output.covariance, type: 'f32', access: 'read'},
            {name: 'status', view: props.output.status, type: 'u32', access: 'read'},
            {name: 'stability', view: props.output.stability, type: 'f32', access: 'read_write'}
          ],
          declarations: `${GET_NAN_WGSL}\n${IS_NAN_BITS_WGSL}\n${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}\n${getCholeskyWGSL(contrasts)}
const COLUMNS = array<u32, ${contrasts}>(${(contrastColumns.length ? contrastColumns : [0]).map(value => `${value}u`).join(',')});
const OFFSETS = array<u32, ${contrasts}>(${(contrastOffsets.length ? contrastOffsets : [0]).map(value => `${value}u`).join(',')});
const BASES = array<u32, ${contrasts}>(${(contrastBases.length ? contrastBases : [0]).map(value => `${value}u`).join(',')});`,
          body: `stability[stabilityOffset + index * 3u] = getNaN(); stability[stabilityOffset + index * 3u + 1u] = 0.0; stability[stabilityOffset + index * 3u + 2u] = getNaN();
if (status[statusOffset] != 0u) { return; }
var selected: array<u32, ${contrasts}>; var count = 0u;
for (var contrast = 0u; contrast < ${contrastColumns.length}u; contrast++) { if (index == ${columns + 1}u || COLUMNS[contrast] == index) { selected[count] = contrast; count++; } }
if (count == 0u) { return; }
var normal: array<f32, ${contrasts * contrasts}>; var right: array<f32, ${contrasts}>; var difference: array<f32, ${contrasts}>;
for (var first = 0u; first < ${contrasts}u; first++) { normal[first * ${contrasts + 1}u] = 1.0; }
for (var first = 0u; first < count; first++) { let positive = OFFSETS[selected[first]]; let negative = BASES[selected[first]]; difference[first] = coefficients[coefficientsOffset + positive] - coefficients[coefficientsOffset + negative]; right[first] = difference[first];
for (var second = 0u; second < count; second++) { let otherPositive = OFFSETS[selected[second]]; let otherNegative = BASES[selected[second]];
normal[first * ${contrasts}u + second] = covariance[covarianceOffset + positive * ${restricted}u + otherPositive] - covariance[covarianceOffset + positive * ${restricted}u + otherNegative] - covariance[covarianceOffset + negative * ${restricted}u + otherPositive] + covariance[covarianceOffset + negative * ${restricted}u + otherNegative]; } }
if (!choleskyFactor_${contrasts}(&normal)) { return; } choleskySolve_${contrasts}(&normal, &right);
var statistic = 0.0; for (var contrast = 0u; contrast < count; contrast++) { statistic += difference[contrast] * right[contrast]; }
statistic = max(0.0, statistic); stability[stabilityOffset + index * 3u] = statistic; stability[stabilityOffset + index * 3u + 1u] = f32(count); stability[stabilityOffset + index * 3u + 2u] = getChiSquareSurvivalValue(statistic, f32(count));`
        })
      );
    }
    return nodes;
  }
}
