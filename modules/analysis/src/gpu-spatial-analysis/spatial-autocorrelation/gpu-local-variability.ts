// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {
  validateCapacity,
  validateViewLength,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsOrChunksViews,
  isGPUChunkedSpatialWeights,
  validateGPUSpatialWeightsOrChunks,
  wrapGPUSpatialWeightsChunkBody,
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeights,
  type GPUSpatialWeightsChunkDispatch
} from '../spatial-weights/index';

import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import * as variabilityParameters from './local-variability-parameters';

/** Per-row status: valid, excluded input, undefined statistic, incomplete weights. */
export const GPU_LOCAL_VARIABILITY_STATUS = {
  OK: 0,
  EXCLUDED: 1,
  UNDEFINED: 2,
  INCOMPLETE: 3
} as const;

type LocalVariabilityProps = {
  id?: string;
  /** Whole weights, or row chunks (one dispatch per chunk, same per-row result). */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /** Row-major values, `rowCount * variableCount`. Nonfinite rows are excluded jointly. */
  values: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  output: {
    statistic: GraphDataView<'float32'>;
    /** Required per-row completion evidence; see GPU_LOCAL_VARIABILITY_STATUS. */
    status: GraphDataView<'uint32'>;
  };
};

/** LOSH inputs. Positive weights are normalized over included neighbors. Islands are excluded. */
export type GPULocalSpatialHeteroscedasticityProps = LocalVariabilityProps & {
  /**
   * Absolute-deviation power; 2 (default) measures variance, 1 absolute deviation. A packed
   * float32 GPUParameterBuffer view from getGPULocalSpatialHeteroscedasticityParameterValues
   * allows the power to change between encodings without recompiling.
   */
  residualPower?: 1 | 2 | GraphDataView<'float32'>;
  output: LocalVariabilityProps['output'] & {
    /** Optional normalized residual powers for explicit residual-label randomization. */
    normalizedResiduals?: GraphDataView<'float32'>;
  };
};

/**
 * LOSH H_i = weightedMean_j(|y_j - weightedMean_k(y_k)|^a) / mean(residual^a).
 * The denominator includes finite residuals, excluding islands. A fixed-order maximum and sum
 * reduction form the mean from residuals scaled by their maximum, avoiding overflow of their
 * unscaled total. No invocation walks all rows. No inferential p-value is implied.
 * Compile-time: view shapes, optional views and chunk topology. Per-frame: input/mask contents
 * and a residual-power parameter view. Numeric powers are uploaded defaults.
 *
 * Masked rows are removed before local means; neighbors whose residual is undefined are removed
 * before the final weighted mean. An incomplete upstream weights graph invalidates every result.
 */
export class GPULocalSpatialHeteroscedasticity implements GPUCommandNodeProducer {
  readonly id: string;
  private readonly residualPowerValues?: Float32Array;
  constructor(readonly props: GPULocalSpatialHeteroscedasticityProps) {
    this.id = props.id ?? 'local-spatial-heteroscedasticity';
    validateProps(this.id, props, 1);
    const outputs = [props.output.statistic, props.output.status, props.output.normalizedResiduals];
    for (let index = 0; index < outputs.length; index++) {
      validateGraphOutputsDisjointFromInputs(this.id, [outputs[index]], outputs.slice(index + 1));
    }
    const residualPower = props.residualPower ?? 2;
    if (typeof residualPower === 'number') {
      validateCapacity(this.id, 'residualPower', residualPower, 1, 2);
      this.residualPowerValues =
        variabilityParameters.getGPULocalSpatialHeteroscedasticityParameterValues({
          residualPower
        });
    } else {
      validateSinglePackedView(residualPower, ['float32'], `${this.id} residualPower`);
      validateCapacity(
        this.id,
        'residualPower.length',
        residualPower.length,
        variabilityParameters.GPU_LOCAL_SPATIAL_HETEROSCEDASTICITY_PARAMETER_LENGTH
      );
      validateGraphOutputsDisjointFromInputs(this.id, outputs, [residualPower]);
    }
    if (props.output.normalizedResiduals) {
      validateSinglePackedView(
        props.output.normalizedResiduals,
        ['float32'],
        `${this.id} normalizedResiduals`
      );
      validateViewLength(
        this.id,
        'normalizedResiduals',
        props.output.normalizedResiduals,
        getRowCount(props)
      );
      validateGraphOutputsDisjointFromInputs(
        this.id,
        [props.output.normalizedResiduals],
        getAllInputViews(props)
      );
    }
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const rows = getRowCount(props);
    const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'residual scratch',
      rows * Float32Array.BYTES_PER_ELEMENT,
      'reduce the row count'
    );
    const finiteResiduals = createTransientView(graph, `${id}-finite-residuals`, 'float32', rows);
    const residualCounts = createTransientView(graph, `${id}-residual-counts`, 'uint32', rows);
    const maximumResidual = createTransientView(graph, `${id}-maximum-residual`, 'float32', 1);
    const normalizedSum = createTransientView(graph, `${id}-normalized-sum`, 'float32', 1);
    const residualCount = createTransientView(graph, `${id}-residual-count`, 'uint32', 1);
    const residuals = createTransientView(graph, `${id}-residuals`, 'float32', rows);
    const mean = createTransientView(graph, `${id}-mean`, 'float32', 1);
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.mask,
      ...getGPUSpatialWeightsOrChunksViews(props.weights),
      props.weights.status?.overflow,
      props.output.statistic,
      props.output.status,
      props.output.normalizedResiduals,
      typeof props.residualPower === 'object' ? props.residualPower : undefined
    ]);
    const runtime = createSpatialStatisticsParameterNodes(
      graph,
      `${id}-parameters`,
      this.residualPowerValues ?? (props.residualPower as GraphDataView<'float32'>)
    );
    const nodes: GPUCommandNode<Parameters>[] = [
      ...runtime.nodes,
      ...dispatches.map(dispatch =>
        createWGSLKernelNode(graph, {
          id: `${id}-residuals${dispatch.idSuffix}`,
          operation: 'GPULocalSpatialHeteroscedasticity',
          variant: 'residuals',
          invocationCount: dispatch.rowCount,
          bindings: [
            ...getBindings(props, dispatch.weights),
            {name: 'parameters', view: runtime.parameters, type: 'f32', access: 'read'},
            {name: 'residuals', view: residuals, type: 'f32', access: 'read_write'}
          ],
          declarations: getCommon(props, rows, 1, dispatch),
          body: wrapGPUSpatialWeightsChunkBody(
            dispatch,
            `var result = getNaN();
if (included(index)) {
  var weighted = 0.0;
  var total = 0.0;
  ${neighborLoop(`weighted += weight * values[valuesOffset + neighbor]; total += weight;`, dispatch)}
  if (total > 0.0) {
    let deviation = abs(values[valuesOffset + index] - weighted / total);
    result = select(deviation * deviation, deviation, parameters[parametersOffset] == 1.0);
  }
}
residuals[residualsOffset + index] = result;`
          )
        })
      ),
      createWGSLKernelNode(graph, {
        id: `${id}-finite-residuals`,
        operation: 'GPULocalSpatialHeteroscedasticity',
        variant: 'finite-residuals',
        invocationCount: rows,
        bindings: [
          {name: 'residuals', view: residuals, type: 'f32', access: 'read'},
          {name: 'finiteResiduals', view: finiteResiduals, type: 'f32', access: 'read_write'},
          {name: 'residualCounts', view: residualCounts, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_BITS_WGSL,
        body: `let value = residuals[residualsOffset + index];
let finite = isFiniteBits(value);
finiteResiduals[finiteResidualsOffset + index] = select(0.0, value, finite);
residualCounts[residualCountsOffset + index] = u32(finite);`
      }),
      ...new GPUReduction({
        id: `${id}-maximum-residual`,
        input: finiteResiduals,
        output: maximumResidual,
        operation: 'max'
      }).getCommandNodes(graph),
      ...new GPUReduction({
        id: `${id}-residual-count`,
        input: residualCounts,
        output: residualCount,
        operation: 'sum'
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-scale-residuals`,
        operation: 'GPULocalSpatialHeteroscedasticity',
        variant: 'scale-residuals',
        invocationCount: rows,
        bindings: [
          {name: 'finiteResiduals', view: finiteResiduals, type: 'f32', access: 'read_write'},
          {name: 'maximumResidual', view: maximumResidual, type: 'f32', access: 'read'}
        ],
        declarations: SCALED_QUOTIENT_WGSL,
        body: `let maximum = maximumResidual[maximumResidualOffset];
var scaled = 0.0;
if (maximum > 0.0) {
  // A direct division can flush the reciprocal of a large maximum to zero.
  scaled = getScaledQuotient(finiteResiduals[finiteResidualsOffset + index], maximum, 1.0, 0);
}
finiteResiduals[finiteResidualsOffset + index] = scaled;`
      }),
      ...new GPUReduction({
        id: `${id}-normalized-sum`,
        input: finiteResiduals,
        output: normalizedSum,
        operation: 'sum'
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-mean`,
        operation: 'GPULocalSpatialHeteroscedasticity',
        variant: 'mean',
        invocationCount: 1,
        bindings: [
          {name: 'maximumResidual', view: maximumResidual, type: 'f32', access: 'read'},
          {name: 'normalizedSum', view: normalizedSum, type: 'f32', access: 'read'},
          {name: 'residualCount', view: residualCount, type: 'u32', access: 'read'},
          {name: 'mean', view: mean, type: 'f32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}
// 2^126 is finite; splitting larger exponents avoids an overflowing ldexp scale.
const MAXIMUM_MEAN_SCALE_EXPONENT: i32 = 126;`,
        body: `let count = residualCount[residualCountOffset];
let maximum = frexp(maximumResidual[maximumResidualOffset]);
// Keep the maximum's exponent out of float products, including compiler reassociation.
let fraction = maximum.fract * (normalizedSum[normalizedSumOffset] / f32(count));
let firstExponent = min(maximum.exp, MAXIMUM_MEAN_SCALE_EXPONENT);
let average = ldexp(ldexp(fraction, firstExponent), maximum.exp - firstExponent);
mean[meanOffset] = select(getNaN(), average, count > 0u);`
      }),
      ...dispatches.map(dispatch =>
        createWGSLKernelNode(graph, {
          id: `${id}-finish${dispatch.idSuffix}`,
          operation: 'GPULocalSpatialHeteroscedasticity',
          variant: 'finish',
          invocationCount: dispatch.rowCount,
          // A compact scratch stage keeps the standard eight storage-binding limit.
          bindings: [
            {name: 'offsets', view: dispatch.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: dispatch.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: dispatch.weights.weights, type: 'f32', access: 'read'},
            {name: 'residuals', view: residuals, type: 'f32', access: 'read'},
            {name: 'mean', view: mean, type: 'f32', access: 'read'},
            {name: 'result', view: props.output.statistic, type: 'f32', access: 'read_write'},
            {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
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
          declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${SCALED_QUOTIENT_WGSL}\n${dispatch.constantsWGSL}`,
          body: wrapGPUSpatialWeightsChunkBody(
            dispatch,
            `var resultValue = getNaN(); var state = 1u;
if (isFiniteBits(residuals[residualsOffset + index])) {
  var total = 0.0; var weighted = 0.0;
  let rowBegin = offsets[offsetsOffset + ${getLocalRow(dispatch)}];
  let rowEnd = offsets[offsetsOffset + ${getLocalRow(dispatch)} + 1u];
  for (var slot = rowBegin; slot < min(rowEnd, ${dispatch.weights.neighbors.length}u); slot++) {
    let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
    if (neighbor >= ${rows}u || neighbor == index || !(weight > 0.0) || !isFiniteBits(weight)) { continue; }
    let residual = residuals[residualsOffset + neighbor];
    if (isFiniteBits(residual)) { weighted += weight * residual; total += weight; }
  }
  state = 2u;
  if (total > 0.0 && mean[meanOffset] > 0.0) {
    if (isFiniteBits(weighted) && isFiniteBits(total) && isFiniteBits(mean[meanOffset])) {
      resultValue = getScaledQuotient(weighted, total, mean[meanOffset], 0);
    } else {
      resultValue = weighted / total / mean[meanOffset];
    }
    state = 0u;
  }
}
${props.weights.status?.overflow ? 'if (overflow[overflowOffset] != 0u) { state = 3u; resultValue = getNaN(); }' : ''}
if (!isFiniteBits(resultValue) && state == 0u) { state = 2u; }
result[resultOffset + index] = resultValue; status[statusOffset + index] = state;`
          )
        })
      )
    ];
    if (props.output.normalizedResiduals) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-normalized-residuals`,
          operation: 'GPULocalSpatialHeteroscedasticity',
          variant: 'normalized-residuals',
          invocationCount: rows,
          bindings: [
            {name: 'residuals', view: residuals, type: 'f32', access: 'read'},
            {name: 'mean', view: mean, type: 'f32', access: 'read'},
            {
              name: 'normalized',
              view: props.output.normalizedResiduals,
              type: 'f32',
              access: 'read_write'
            }
          ],
          declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${SCALED_QUOTIENT_WGSL}`,
          body: `let average = mean[meanOffset];
let residual = residuals[residualsOffset + index];
var normalizedValue = getNaN();
if (average > 0.0 && isFiniteBits(average)) {
  normalizedValue = residual;
  if (isFiniteBits(residual)) {
    normalizedValue = getScaledQuotient(residual, average, 1.0, 0);
  }
}
normalized[normalizedOffset + index] = normalizedValue;`
        })
      );
    }
    return nodes;
  }
}

/** Multivariate local Geary inputs, with population standardization and joint complete-case rows. */
export type GPUMultivariateLocalGearyProps = LocalVariabilityProps & {variableCount: number};

/**
 * Mean across variables of row-standardized local Geary. Each variable uses its population
 * variance over joint complete-case rows, including islands; island results are undefined.
 * A constant variable invalidates the statistic. This is descriptive: no univariate permutation
 * adapter is reused because valid joint randomization must preserve row pairing across variables.
 */
export class GPUMultivariateLocalGeary implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUMultivariateLocalGearyProps) {
    this.id = props.id ?? 'multivariate-local-geary';
    validateProps(this.id, props, props.variableCount);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = getRowCount(props);
    const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights);
    const variance = createTransientView(graph, `${id}-variance`, 'float32', props.variableCount);
    const bindings = getBindings(props, dispatches[0].weights);
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.mask,
      ...getGPUSpatialWeightsOrChunksViews(props.weights),
      props.weights.status?.overflow,
      props.output.statistic,
      props.output.status
    ]);
    const common = getCommon(props, rows, props.variableCount, dispatches[0]);
    const valueBindings = bindings.filter(binding =>
      ['values', 'mask', 'overflow'].includes(binding.name)
    );
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-moments`,
        operation: 'GPUMultivariateLocalGeary',
        variant: 'moments',
        invocationCount: props.variableCount,
        bindings: [
          ...valueBindings,
          {name: 'variance', view: variance, type: 'f32', access: 'read_write'}
        ],
        declarations: common,
        body: `var count = 0.0; var mean = 0.0; var squared = 0.0;
for (var row = 0u; row < ${rows}u; row++) {
  if (!included(row)) { continue; }
  let value = values[valuesOffset + row * ${props.variableCount}u + index];
  count += 1.0; let difference = value - mean; mean += difference / count; squared += difference * (value - mean);
}
variance[varianceOffset + index] = select(getNaN(), squared / count, count > 0.0);`
      }),
      ...dispatches.map(dispatch =>
        createWGSLKernelNode(graph, {
          id: `${id}-statistic${dispatch.idSuffix}`,
          operation: 'GPUMultivariateLocalGeary',
          variant: 'statistic',
          invocationCount: dispatch.rowCount,
          bindings: [
            ...getBindings(props, dispatch.weights).filter(binding => binding.name !== 'overflow'),
            {name: 'variance', view: variance, type: 'f32', access: 'read'},
            {name: 'result', view: props.output.statistic, type: 'f32', access: 'read_write'},
            {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
          ],
          declarations: getCommon(
            {...props, weights: {...props.weights, status: undefined}},
            rows,
            props.variableCount,
            dispatch
          ),
          body: wrapGPUSpatialWeightsChunkBody(
            dispatch,
            `var state = 1u; var resultValue = getNaN();
if (included(index)) {
  var sum = 0.0; var total = 0.0; var defined = true;
  for (var column = 0u; column < ${props.variableCount}u; column++) { defined = defined && variance[varianceOffset + column] > 0.0 && isFiniteBits(variance[varianceOffset + column]); }
  if (defined) {
    ${neighborLoop(
      `var distance = 0.0;
      for (var column = 0u; column < ${props.variableCount}u; column++) {
        let difference = values[valuesOffset + index * ${props.variableCount}u + column] - values[valuesOffset + neighbor * ${props.variableCount}u + column];
        distance += difference * difference / variance[varianceOffset + column];
      }
      sum += weight * distance; total += weight;`,
      dispatch
    )}
  }
  state = 2u;
  if (defined && total > 0.0) { resultValue = sum / total / ${props.variableCount}.0; state = select(2u, 0u, isFiniteBits(resultValue)); }
}
result[resultOffset + index] = resultValue; status[statusOffset + index] = state;`
          )
        })
      )
    ];
    if (props.weights.status?.overflow)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-upstream-status`,
          operation: 'GPUMultivariateLocalGeary',
          variant: 'upstream-status',
          invocationCount: rows,
          bindings: [
            {name: 'overflow', view: props.weights.status.overflow, type: 'u32', access: 'read'},
            {name: 'result', view: props.output.statistic, type: 'f32', access: 'read_write'},
            {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
          ],
          declarations: GET_NAN_WGSL,
          body: 'if (overflow[overflowOffset] != 0u) { result[resultOffset + index] = getNaN(); status[statusOffset + index] = 3u; }'
        })
      );
    return nodes;
  }
}

function validateProps(id: string, props: LocalVariabilityProps, variableCount: number): void {
  const rows = validateGPUSpatialWeightsOrChunks(id, props.weights);
  if (rows >= 16777216) throw new Error(`${id}: float32 moments require fewer than 2^24 rows`);
  if (!Number.isInteger(variableCount) || variableCount < 1 || variableCount > 32)
    throw new Error(`${id}: variableCount must be 1 to 32`);
  const views = [props.values, props.output.statistic, props.output.status, props.mask];
  const lengths = [rows * variableCount, rows, rows, rows];
  for (let index = 0; index < views.length; index++) {
    const view = views[index];
    if (!view) continue;
    validateSinglePackedView(view, [index < 2 ? 'float32' : 'uint32'], id);
    if (view.length !== lengths[index]) throw new Error(`${id}: invalid row count`);
  }
  validateGraphOutputsDisjointFromInputs(
    id,
    [props.output.statistic, props.output.status],
    getAllInputViews(props)
  );
}
function getRowCount(props: LocalVariabilityProps): number {
  return isGPUChunkedSpatialWeights(props.weights)
    ? props.weights.rowCount
    : props.weights.offsets.length - 1;
}
function getAllInputViews(
  props: LocalVariabilityProps
): (GraphDataView<'uint32' | 'float32'> | undefined)[] {
  return [
    props.values,
    props.mask,
    props.weights.status?.overflow,
    ...getGPUSpatialWeightsOrChunksViews(props.weights)
  ];
}
function getLocalRow(dispatch: GPUSpatialWeightsChunkDispatch): string {
  return dispatch.isWhole ? 'index' : 'index - WEIGHTS_ROW_START';
}
function getBindings(
  props: LocalVariabilityProps,
  weights: GPUSpatialWeights
): WGSLKernelBinding[] {
  return [
    {name: 'values', view: props.values, type: 'f32', access: 'read'},
    {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
    {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
    {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
    ...(props.mask
      ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
      : []),
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
  ];
}
function getCommon(
  props: LocalVariabilityProps,
  rows: number,
  variableCount: number,
  dispatch: GPUSpatialWeightsChunkDispatch
): string {
  return `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
const ROWS: u32 = ${rows}u; const SLOTS: u32 = ${dispatch.weights.neighbors.length}u;
${dispatch.constantsWGSL}
fn included(row: u32) -> bool {
  if (row >= ROWS) { return false; }
  ${props.mask ? 'if (mask[maskOffset + row] == 0u) { return false; }' : ''}
  ${props.weights.status?.overflow ? 'if (overflow[overflowOffset] != 0u) { return false; }' : ''}
  for (var column = 0u; column < ${variableCount}u; column++) {
    if (!isFiniteBits(values[valuesOffset + row * ${variableCount}u + column])) { return false; }
  }
  return true;
}`;
}
function neighborLoop(action: string, dispatch: GPUSpatialWeightsChunkDispatch): string {
  const localRow = getLocalRow(dispatch);
  return `for (var slot = offsets[offsetsOffset + ${localRow}]; slot < min(offsets[offsetsOffset + ${localRow} + 1u], SLOTS); slot++) {
    let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
    if (neighbor == index || !included(neighbor) || !(weight > 0.0) || !isFiniteBits(weight)) { continue; }
    ${action}
  }`;
}
