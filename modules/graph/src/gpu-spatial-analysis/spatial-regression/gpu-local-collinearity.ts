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
import {
  validateCapacity,
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import {
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsOrChunksViews,
  validateGPUSpatialWeightsOrChunks,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/gpu-chunked-spatial-weights';
import {wrapWeightsChunkRowBodyWGSL} from './chunked-weights-kernels';
import {
  getDesignSingularValueWGSL,
  LOCAL_COLLINEARITY_CONDITION_CAP,
  LOCAL_COLLINEARITY_SINGULAR_RATIO
} from './local-collinearity-wgsl';

const CONDITION_CAP = LOCAL_COLLINEARITY_CONDITION_CAP.toExponential();
const SINGULAR_RATIO = LOCAL_COLLINEARITY_SINGULAR_RATIO.toExponential();

export type GPULocalCollinearityProps = {
  id?: string;
  /** Explicit local estimation weights, including self weights if the local model uses them. */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  predictors: GraphDataView<'float32'>;
  predictorCount: number;
  mask?: GraphDataView<'uint32'>;
  output: {
    /** Focus-major inverse diagonal of the weighted predictor correlation matrix. */
    varianceInflationFactors: GraphDataView<'float32'>;
    /**
     * sqrt(maximum eigenvalue / minimum eigenvalue) of that correlation matrix, from singular
     * values of the weighted design. A numerically singular design reports a capped large value.
     */
    conditionNumbers: GraphDataView<'float32'>;
    /** Optional mgwr uncentered, intercept-inclusive design condition number. Requires varianceDecomposition. */
    designConditionNumbers?: GraphDataView<'float32'>;
    /** Optional proportion at the smallest singular value per intercept/predictor, focus-major.
     * Uses column-scaled X * normalized local weights (not sqrt(weights)), matching mgwr.
     */
    varianceDecomposition?: GraphDataView<'float32'>;
    /** 0 valid; 1 insufficient complete neighbors; 2 singular/nonfinite correlation/design; 3 singular value solver exhaustion. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Weighted, centered local predictor VIF surfaces. The intercept is excluded from the correlation
 * matrix. Weights are supplied explicitly so diagnostics use exactly the neighborhoods of the fit.
 * Every predictor uses the same complete rows. This is a correlation condition index, distinct
 * from mgwr's uncentered, intercept-inclusive variance-decomposition condition index.
 */
export class GPULocalCollinearity implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPULocalCollinearityProps) {
    this.id = props.id ?? 'local-collinearity';
    const rows = validateGPUSpatialWeightsOrChunks(this.id, props.weights);
    const columns = props.predictorCount;
    validateCapacity(this.id, 'predictorCount', columns, 1, 12);
    if (
      Boolean(props.output.designConditionNumbers) !== Boolean(props.output.varianceDecomposition)
    )
      throw new Error(`${this.id}: design diagnostics outputs must be paired`);
    for (const [view, format, length] of [
      [props.predictors, 'float32', rows * columns],
      [props.mask, 'uint32', rows],
      [props.output.varianceInflationFactors, 'float32', rows * columns],
      [props.output.conditionNumbers, 'float32', rows],
      [props.output.designConditionNumbers, 'float32', rows],
      [props.output.varianceDecomposition, 'float32', rows * (columns + 1)],
      [props.output.status, 'uint32', rows]
    ] as const)
      if (view) {
        validatePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.predictors,
      props.mask,
      ...getGPUSpatialWeightsOrChunksViews(props.weights)
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const columns = props.predictorCount;
    const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights);
    const rows = dispatches.reduce((total, dispatch) => total + dispatch.rowCount, 0);
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.mask,
      ...getGPUSpatialWeightsOrChunksViews(props.weights),
      ...Object.values(props.output)
    ]);
    const nodes = [
      ...dispatches.map(dispatch =>
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-fit${dispatch.idSuffix}`,
          operation: 'GPULocalCollinearity',
          variant: 'weighted-correlation',
          invocationCount: dispatch.rowCount,
          bindings: [
            {name: 'offsets', view: dispatch.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: dispatch.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: dispatch.weights.weights, type: 'f32', access: 'read'},
            {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
            ...(props.mask
              ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
              : []),
            {
              name: 'inflation',
              view: props.output.varianceInflationFactors,
              type: 'f32',
              access: 'read_write'
            },
            {
              name: 'condition',
              view: props.output.conditionNumbers,
              type: 'f32',
              access: 'read_write'
            },
            {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
          ],
          declarations: `${dispatch.constantsWGSL}
${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${getDesignSingularValueWGSL(columns)}
fn included(row: u32) -> bool {
  if (row >= ${rows}u) { return false; }
  ${props.mask ? 'if (mask[maskOffset + row] == 0u) { return false; }' : ''}
  for (var column = 0u; column < ${columns}u; column++) {
    if (!isFiniteBits(predictors[predictorsOffset + row * ${columns}u + column])) { return false; }
  }
  return true;
}`,
          body: wrapWeightsChunkRowBodyWGSL(
            dispatch,
            `status[statusOffset + index] = 1u;
condition[conditionOffset + index] = getNaN();
for (var column = 0u; column < ${columns}u; column++) {
  inflation[inflationOffset + index * ${columns}u + column] = getNaN();
}
let slotBegin = offsets[offsetsOffset + index];
let slotEnd = offsets[offsetsOffset + index + 1u];
var means: array<f32, ${columns}>;
var total = 0.0;
var count = 0u;
for (var slot = slotBegin; slot < slotEnd; slot++) {
  let neighbor = neighbors[neighborsOffset + slot];
  let weight = weights[weightsOffset + slot];
  if (!included(neighbor) || !(weight > 0.0)) { continue; }
  total += weight;
  count++;
  for (var column = 0u; column < ${columns}u; column++) {
    means[column] += weight / total * (predictors[predictorsOffset + neighbor * ${columns}u + column] - means[column]);
  }
}
if (count <= ${columns}u) { return; }
status[statusOffset + index] = 2u;
var correlation: array<f32, ${columns * columns}>;
for (var slot = slotBegin; slot < slotEnd; slot++) {
  let neighbor = neighbors[neighborsOffset + slot];
  let weight = weights[weightsOffset + slot];
  if (!included(neighbor) || !(weight > 0.0)) { continue; }
  for (var first = 0u; first < ${columns}u; first++) {
    let firstValue = predictors[predictorsOffset + neighbor * ${columns}u + first] - means[first];
    for (var second = 0u; second < ${columns}u; second++) {
      correlation[first * ${columns}u + second] += weight * firstValue * (predictors[predictorsOffset + neighbor * ${columns}u + second] - means[second]);
    }
  }
}
var scale: array<f32, ${columns}>;
for (var column = 0u; column < ${columns}u; column++) {
  let diagonal = correlation[column * ${columns + 1}u];
  if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) { return; }
  scale[column] = inverseSqrt(diagonal);
}
// Condition number from the singular values of the scaled, weighted, centered design (streaming QR).
var factor: array<f32, ${columns * columns}>;
for (var slot = slotBegin; slot < slotEnd; slot++) {
  let neighbor = neighbors[neighborsOffset + slot];
  let weight = weights[weightsOffset + slot];
  if (!included(neighbor) || !(weight > 0.0)) { continue; }
  var designRow: array<f32, ${columns}>;
  for (var column = 0u; column < ${columns}u; column++) {
    designRow[column] = sqrt(weight) * (predictors[predictorsOffset + neighbor * ${columns}u + column] - means[column]) * scale[column];
  }
  updateDesignFactor_${columns}(&factor, &designRow);
}
var vectors: array<f32, ${columns * columns}>;
var singularValues: array<f32, ${columns}>;
if (!getDesignSingularValues_${columns}(&factor, &vectors, &singularValues)) { status[statusOffset + index] = 3u; return; }
var smallest = 3.402823e38;
var largest = 0.0;
for (var column = 0u; column < ${columns}u; column++) {
  smallest = min(smallest, singularValues[column]);
  largest = max(largest, singularValues[column]);
}
if (!(smallest > largest * ${SINGULAR_RATIO}) || !isFiniteBits(largest)) {
  condition[conditionOffset + index] = ${CONDITION_CAP};
  return;
}
condition[conditionOffset + index] = largest / smallest;
// Columns of the scaled design have unit norm, so the inflation is the inverse diagonal of R^T R.
for (var column = 0u; column < ${columns}u; column++) {
  inflation[inflationOffset + index * ${columns}u + column] = getDesignFactorInverseDiagonal_${columns}(&factor, column);
}
status[statusOffset + index] = 0u;`
          )
        })
      )
    ];

    if (props.output.designConditionNumbers && props.output.varianceDecomposition) {
      const size = columns + 1;
      nodes.push(
        ...dispatches.map(dispatch =>
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-variance-decomposition${dispatch.idSuffix}`,
            operation: 'GPULocalCollinearity',
            variant: 'mgwr-design-variance-decomposition',
            invocationCount: dispatch.rowCount,
            bindings: [
              {name: 'offsets', view: dispatch.weights.offsets, type: 'u32', access: 'read'},
              {name: 'neighbors', view: dispatch.weights.neighbors, type: 'u32', access: 'read'},
              {name: 'weights', view: dispatch.weights.weights, type: 'f32', access: 'read'},
              {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
              ...(props.mask
                ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
                : []),
              {
                name: 'condition',
                view: props.output.designConditionNumbers!,
                type: 'f32',
                access: 'read_write'
              },
              {
                name: 'proportions',
                view: props.output.varianceDecomposition!,
                type: 'f32',
                access: 'read_write'
              },
              {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
            ],
            declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${getDesignSingularValueWGSL(size)}${dispatch.constantsWGSL ? `\n${dispatch.constantsWGSL}` : ''}
fn getDesignRow(neighbor: u32, values: ptr<function, array<f32, ${size}>>) -> bool {
  if (neighbor >= ${rows}u) { return false; }
  ${props.mask ? 'if (mask[maskOffset + neighbor] == 0u) { return false; }' : ''}
  (*values)[0] = 1.0;
  for (var column = 0u; column < ${columns}u; column++) {
    let value = predictors[predictorsOffset + neighbor * ${columns}u + column];
    if (!isFiniteBits(value)) { return false; }
    (*values)[column + 1u] = value;
  }
  return true;
}`,
            body: wrapWeightsChunkRowBodyWGSL(
              dispatch,
              `condition[conditionOffset + index] = getNaN();
for (var column = 0u; column < ${size}u; column++) {
  proportions[proportionsOffset + index * ${size}u + column] = getNaN();
}
if (status[statusOffset + index] != 0u) { return; }
let slotBegin = offsets[offsetsOffset + index];
let slotEnd = offsets[offsetsOffset + index + 1u];
// mgwr weights each design row by w (not sqrt(w)); columns are scaled to unit norm.
var scale: array<f32, ${size}>;
for (var slot = slotBegin; slot < slotEnd; slot++) {
  let weight = weights[weightsOffset + slot];
  var values: array<f32, ${size}>;
  if (!(weight > 0.0) || !getDesignRow(neighbors[neighborsOffset + slot], &values)) { continue; }
  for (var column = 0u; column < ${size}u; column++) { scale[column] += weight * weight * values[column] * values[column]; }
}
for (var column = 0u; column < ${size}u; column++) {
  if (!(scale[column] > 0.0) || !isFiniteBits(scale[column])) { status[statusOffset + index] = 2u; return; }
  scale[column] = inverseSqrt(scale[column]);
}
var factor: array<f32, ${size * size}>;
for (var slot = slotBegin; slot < slotEnd; slot++) {
  let weight = weights[weightsOffset + slot];
  var values: array<f32, ${size}>;
  if (!(weight > 0.0) || !getDesignRow(neighbors[neighborsOffset + slot], &values)) { continue; }
  for (var column = 0u; column < ${size}u; column++) { values[column] *= weight * scale[column]; }
  updateDesignFactor_${size}(&factor, &values);
}
var vectors: array<f32, ${size * size}>;
var singularValues: array<f32, ${size}>;
if (!getDesignSingularValues_${size}(&factor, &vectors, &singularValues)) { status[statusOffset + index] = 3u; return; }
var smallest = 0u;
var largest = 0.0;
for (var column = 0u; column < ${size}u; column++) {
  if (singularValues[column] < singularValues[smallest]) { smallest = column; }
  largest = max(largest, singularValues[column]);
}
let minimum = singularValues[smallest];
if (!(minimum > largest * ${SINGULAR_RATIO}) || !isFiniteBits(largest)) {
  condition[conditionOffset + index] = ${CONDITION_CAP};
  status[statusOffset + index] = 2u;
  return;
}
condition[conditionOffset + index] = largest / minimum;
// Belsley proportions of each coefficient's variance at the smallest singular value.
for (var column = 0u; column < ${size}u; column++) {
  var total = 0.0;
  for (var component = 0u; component < ${size}u; component++) {
    let value = vectors[component * ${size}u + column];
    total += value * value / (singularValues[component] * singularValues[component]);
  }
  let loading = vectors[smallest * ${size}u + column];
  proportions[proportionsOffset + index * ${size}u + column] = (loading * loading / (minimum * minimum)) / total;
}`
            )
          })
        )
      );
    }
    return nodes;
  }
}
