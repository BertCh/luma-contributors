// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
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
  /** Absolute-deviation power; 2 (default) measures variance, 1 absolute deviation. */
  residualPower?: 1 | 2;
  output: LocalVariabilityProps['output'] & {
    /** Optional normalized residual powers for explicit residual-label randomization. */
    normalizedResiduals?: GraphDataView<'float32'>;
  };
};

/**
 * LOSH H_i = weightedMean_j(|y_j - weightedMean_k(y_k)|^a) / mean(residual^a).
 * The denominator includes finite residuals, excluding islands. No inferential p-value is implied.
 * Masked rows are removed before local means; neighbors whose residual is undefined are removed
 * before the final weighted mean. An incomplete upstream weights graph invalidates every result.
 */
export class GPULocalSpatialHeteroscedasticity implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPULocalSpatialHeteroscedasticityProps) {
    this.id = props.id ?? 'local-spatial-heteroscedasticity';
    validateProps(this.id, props, 1);
    const outputs = [props.output.statistic, props.output.status, props.output.normalizedResiduals];
    for (let index = 0; index < outputs.length; index++) {
      validateGraphOutputsDisjointFromInputs(this.id, [outputs[index]], outputs.slice(index + 1));
    }
    if (
      props.residualPower !== undefined &&
      props.residualPower !== 1 &&
      props.residualPower !== 2
    ) {
      throw new Error(`${this.id}: residualPower must be 1 or 2`);
    }
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const rows = getRowCount(props);
    const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights);
    const residuals = createTransientView(graph, `${id}-residuals`, 'float32', rows);
    const mean = createTransientView(graph, `${id}-mean`, 'float32', 1);
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.mask,
      ...getGPUSpatialWeightsOrChunksViews(props.weights),
      props.weights.status?.overflow,
      props.output.statistic,
      props.output.status
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...dispatches.map(dispatch =>
        createWGSLKernelNode(graph, {
          id: `${id}-residuals${dispatch.idSuffix}`,
          operation: 'GPULocalSpatialHeteroscedasticity',
          variant: 'residuals',
          invocationCount: dispatch.rowCount,
          bindings: [
            ...getBindings(props, dispatch.weights),
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
    result = ${props.residualPower === 1 ? 'deviation' : 'deviation * deviation'};
  }
}
residuals[residualsOffset + index] = result;`
          )
        })
      ),
      createWGSLKernelNode(graph, {
        id: `${id}-mean`,
        operation: 'GPULocalSpatialHeteroscedasticity',
        variant: 'mean',
        invocationCount: 1,
        bindings: [
          {name: 'residuals', view: residuals, type: 'f32', access: 'read'},
          {name: 'mean', view: mean, type: 'f32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
        body: `var count = 0.0; var average = 0.0;
for (var row = 0u; row < ${rows}u; row++) {
  let value = residuals[residualsOffset + row];
  if (isFiniteBits(value)) { count += 1.0; average += (value - average) / count; }
}
mean[meanOffset] = select(getNaN(), average, count > 0.0);`
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
          declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${dispatch.constantsWGSL}`,
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
  if (total > 0.0 && mean[meanOffset] > 0.0) { resultValue = weighted / total / mean[meanOffset]; state = 0u; }
}
${props.weights.status?.overflow ? 'if (overflow[overflowOffset] != 0u) { state = 3u; resultValue = getNaN(); }' : ''}
if (!isFiniteBits(resultValue) && state == 0u) { state = 2u; }
result[resultOffset + index] = resultValue; status[statusOffset + index] = state;`
          )
        })
      )
    ];
    if (props.output.normalizedResiduals) {
      validatePackedView(props.output.normalizedResiduals, ['float32'], id);
      if (props.output.normalizedResiduals.length !== rows)
        throw new Error(`${id}: invalid residual row count`);
      validateGraphOutputsDisjointFromInputs(
        id,
        [props.output.normalizedResiduals, props.output.statistic, props.output.status],
        getAllInputViews(props)
      );
      validateGraphViewsBelongToGraph(id, graph, [props.output.normalizedResiduals]);
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
          declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
          body: `let average = mean[meanOffset];
normalized[normalizedOffset + index] = select(getNaN(), residuals[residualsOffset + index] / average, average > 0.0 && isFiniteBits(average));`
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
    validatePackedView(view, [index < 2 ? 'float32' : 'uint32'], id);
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
