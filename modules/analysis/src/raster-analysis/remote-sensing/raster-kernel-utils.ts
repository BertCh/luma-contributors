// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GET_QUIET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  createTransientView,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

export type RasterScalarView =
  | GraphDataView<'float32'>
  | GraphDataView<'uint32'>
  | GraphDataView<'sint32'>;
/**
 * Throws unless `count` is a positive safe integer of at most `maximum`. The message carries no
 * contributor ID; contributors that own a prop use `validateCapacity(id, name, ...)` instead.
 */
export function validateRasterCount(count: number, maximum = 0xffffffff): void {
  if (!Number.isSafeInteger(count) || count <= 0 || count > maximum) {
    throw new Error(`raster count must be an integer in [1, ${maximum}]`);
  }
}
export function validateRasterViews(
  id: string,
  inputs: GraphDataView[],
  outputs: GraphDataView[]
): void {
  for (const view of [...inputs, ...outputs]) validatePackedView(view, [view.format], id);
  validateGraphOutputsDisjointFromInputs(id, outputs, inputs);
  for (let index = 0; index < outputs.length; index++)
    validateGraphOutputsDisjointFromInputs(id, [outputs[index]], outputs.slice(index + 1));
}
export function validateRasterGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  inputs: RasterScalarView[],
  outputs: RasterScalarView[]
): void {
  validateGraphViewsBelongToGraph(id, graph, [...inputs, ...outputs]);
}
export function rasterBinding(name: string, view: GraphDataView, write = false): WGSLKernelBinding {
  return {
    name,
    view,
    type: view.format.startsWith('uint32')
      ? 'u32'
      : view.format.startsWith('sint32')
        ? 'i32'
        : 'f32',
    access: write ? 'read_write' : 'read'
  };
}
export const RASTER_FINITE_WGSL = `${GET_QUIET_NAN_WGSL}${IS_FINITE_BITS_WGSL}
fn finite(value: f32) -> bool { return isFiniteBits(value); }
fn missing() -> f32 { return getQuietNaN(); }`;

/** Modified Gram-Schmidt with a second orthogonalization pass; arrays are column-major. */
export function getRasterLeastSquaresWGSL(
  rows: number,
  columns: number,
  variableColumns = false
): string {
  const columnCount = variableColumns ? 'columnCount' : `${columns}u`;
  return `
struct LeastSquaresResult { coefficients: array<f32, ${columns}>, residual: f32, valid: bool }
fn leastSquares(design: array<f32, ${rows * columns}>, response: array<f32, ${rows}>${variableColumns ? ', columnCount: u32' : ''}) -> LeastSquaresResult {
  var basis = design;
  var upper: array<f32, ${columns * columns}>;
  var projected: array<f32, ${columns}>;
  var result: LeastSquaresResult;
  result.valid = false;
  var maximumNorm = 0.0;
  for (var column = 0u; column < ${columnCount}; column++) {
    var originalNorm = 0.0;
    for (var row = 0u; row < ${rows}u; row++) { originalNorm += basis[column * ${rows}u + row] * basis[column * ${rows}u + row]; }
    maximumNorm = max(maximumNorm, sqrt(originalNorm));
    for (var orthogonalizationRound = 0u; orthogonalizationRound < 2u; orthogonalizationRound++) {
      for (var previous = 0u; previous < column; previous++) {
        var projection = 0.0;
        for (var row = 0u; row < ${rows}u; row++) { projection += basis[previous * ${rows}u + row] * basis[column * ${rows}u + row]; }
        upper[previous * ${columns}u + column] += projection;
        for (var row = 0u; row < ${rows}u; row++) { basis[column * ${rows}u + row] -= projection * basis[previous * ${rows}u + row]; }
      }
    }
    var norm = 0.0;
    for (var row = 0u; row < ${rows}u; row++) { norm += basis[column * ${rows}u + row] * basis[column * ${rows}u + row]; }
    norm = sqrt(norm);
    if (!finite(norm) || norm <= maximumNorm * 1e-6) { return result; }
    upper[column * ${columns}u + column] = norm;
    for (var row = 0u; row < ${rows}u; row++) {
      basis[column * ${rows}u + row] /= norm;
      projected[column] += basis[column * ${rows}u + row] * response[row];
    }
  }
  for (var reverse = 0u; reverse < ${columnCount}; reverse++) {
    let column = ${columnCount} - 1u - reverse;
    var value = projected[column];
    for (var following = column + 1u; following < ${columnCount}; following++) { value -= upper[column * ${columns}u + following] * result.coefficients[following]; }
    result.coefficients[column] = value / upper[column * ${columns}u + column];
    if (!finite(result.coefficients[column])) { return result; }
  }
  result.residual = 0.0;
  for (var row = 0u; row < ${rows}u; row++) {
    var prediction = 0.0;
    for (var column = 0u; column < ${columnCount}; column++) { prediction += design[column * ${rows}u + row] * result.coefficients[column]; }
    result.residual += (prediction - response[row]) * (prediction - response[row]);
  }
  result.valid = finite(result.residual);
  return result;
}`;
}

/**
 * Parallel replacement for single-thread counting loops: a clear node, then one kernel in which
 * each workgroup counts matching elements in workgroup memory and issues one global `atomicAdd`.
 * `condition` is a WGSL boolean expression over `values[valuesOffset + index]` (named `value`).
 * Integer counting is exact and order independent. The output is one `uint32` row.
 */
export function createRasterCountNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    values: GraphDataView;
    /** Number of leading elements of `values` to test. */
    count: number;
    /** WGSL boolean expression of the local `value`, e.g. `value != 1u`. */
    condition: string;
    output: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const nodes = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-count-clear`,
      operation: props.operation,
      bindings: [{name: 'counted', view: props.output, type: 'u32', access: 'read_write'}],
      invocationCount: 1,
      body: 'counted[countedOffset] = 0u;'
    })
  ];
  if (props.count > 0) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-count`,
        operation: props.operation,
        bindings: [
          rasterBinding('countValues', props.values),
          {name: 'counted', view: props.output, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: props.count,
        guardIndex: false,
        declarations: 'var<workgroup> workgroupCount: atomic<u32>;',
        // No early return: every invocation of a workgroup must reach the barriers.
        body: `if (localInvocationIndex == 0u) { atomicStore(&workgroupCount, 0u); }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let value = countValues[countValuesOffset + index];
    if (${props.condition}) { atomicAdd(&workgroupCount, 1u); }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let total = atomicLoad(&workgroupCount);
    if (total != 0u) { atomicAdd(&counted[countedOffset], total); }
  }`
      })
    );
  }
  return nodes;
}

/** Workgroup size of the argmax kernels; a power of two for the tree reduction. */
const RASTER_ARGMAX_WORKGROUP_SIZE = 256;

/**
 * Parallel argmax over `count` candidates, replacing a single-invocation scan loop. Pass one
 * reduces each workgroup of 256 candidates to a (value, index) pair in workgroup memory; pass two
 * is one workgroup that folds the per-group pairs, striding over them, and writes the result.
 *
 * The caller supplies `candidate`, WGSL statements over the thread's `index` that assign the
 * pre-declared `candidateValue: f32` and `candidateEligible: bool`. A candidate wins only if it is
 * eligible and `candidateValue > initialValue` (so NaN never wins). Larger values win and equal
 * values go to the lowest index, which is the result of a serial scan with a strict `>` test.
 * With no winner the value output is `initialValue` and the index output is 0.
 * The outputs are one `float32` row and one `uint32` row. Because the combine order is fixed by
 * index, the result is deterministic and independent of workgroup scheduling.
 */
export function createRasterArgmaxNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    count: number;
    /** Read-only bindings the `candidate` statements use. */
    bindings: WGSLKernelBinding[];
    declarations?: string;
    candidate: string;
    /** Smallest acceptable value, as a WGSL float literal, e.g. `-3.402823466e38`. */
    initialValue: string;
    outputValue: GraphDataView<'float32'>;
    outputIndex: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const groupCount = Math.ceil(props.count / RASTER_ARGMAX_WORKGROUP_SIZE);
  const groupValues = createTransientView(graph, `${props.id}-group-values`, 'float32', groupCount);
  const groupIndices = createTransientView(
    graph,
    `${props.id}-group-indices`,
    'uint32',
    groupCount
  );
  const sharedDeclarations = `var<workgroup> argmaxSharedValues: array<f32, ${RASTER_ARGMAX_WORKGROUP_SIZE}>;
var<workgroup> argmaxSharedIndices: array<u32, ${RASTER_ARGMAX_WORKGROUP_SIZE}>;`;
  const reduceTree = `workgroupBarrier();
  for (var argmaxStride = ${RASTER_ARGMAX_WORKGROUP_SIZE / 2}u; argmaxStride > 0u; argmaxStride = argmaxStride >> 1u) {
    if (localInvocationIndex < argmaxStride) {
      let argmaxOtherValue = argmaxSharedValues[localInvocationIndex + argmaxStride];
      let argmaxOtherIndex = argmaxSharedIndices[localInvocationIndex + argmaxStride];
      let argmaxValue = argmaxSharedValues[localInvocationIndex];
      if (argmaxOtherValue > argmaxValue || (argmaxOtherValue == argmaxValue && argmaxOtherIndex < argmaxSharedIndices[localInvocationIndex])) {
        argmaxSharedValues[localInvocationIndex] = argmaxOtherValue;
        argmaxSharedIndices[localInvocationIndex] = argmaxOtherIndex;
      }
    }
    workgroupBarrier();
  }`;
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-argmax-groups`,
      operation: props.operation,
      invocationCount: props.count,
      workgroupSize: RASTER_ARGMAX_WORKGROUP_SIZE,
      guardIndex: false,
      bindings: [
        ...props.bindings,
        rasterBinding('groupValues', groupValues, true),
        rasterBinding('groupIndices', groupIndices, true)
      ],
      declarations: `${props.declarations ?? ''}\n${sharedDeclarations}`,
      // No early return: every invocation of a workgroup must reach the barriers.
      body: `var argmaxBestValue = ${props.initialValue};
  var argmaxBestIndex = 0xffffffffu;
  if (index < INVOCATION_COUNT) {
    var candidateValue = ${props.initialValue};
    var candidateEligible = false;
    ${props.candidate}
    if (candidateEligible && candidateValue > ${props.initialValue}) {
      argmaxBestValue = candidateValue;
      argmaxBestIndex = index;
    }
  }
  argmaxSharedValues[localInvocationIndex] = argmaxBestValue;
  argmaxSharedIndices[localInvocationIndex] = argmaxBestIndex;
  ${reduceTree}
  if (localInvocationIndex == 0u && workgroupIndex < ${groupCount}u) {
    groupValues[groupValuesOffset + workgroupIndex] = argmaxSharedValues[0];
    groupIndices[groupIndicesOffset + workgroupIndex] = argmaxSharedIndices[0];
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-argmax-final`,
      operation: props.operation,
      invocationCount: RASTER_ARGMAX_WORKGROUP_SIZE,
      workgroupSize: RASTER_ARGMAX_WORKGROUP_SIZE,
      guardIndex: false,
      bindings: [
        rasterBinding('groupValues', groupValues),
        rasterBinding('groupIndices', groupIndices),
        rasterBinding('outputValue', props.outputValue, true),
        rasterBinding('outputIndex', props.outputIndex, true)
      ],
      declarations: sharedDeclarations,
      body: `var argmaxBestValue = ${props.initialValue};
  var argmaxBestIndex = 0xffffffffu;
  for (var argmaxGroup = localInvocationIndex; argmaxGroup < ${groupCount}u; argmaxGroup += ${RASTER_ARGMAX_WORKGROUP_SIZE}u) {
    let argmaxValue = groupValues[groupValuesOffset + argmaxGroup];
    let argmaxGroupIndex = groupIndices[groupIndicesOffset + argmaxGroup];
    if (argmaxValue > argmaxBestValue || (argmaxValue == argmaxBestValue && argmaxGroupIndex < argmaxBestIndex)) {
      argmaxBestValue = argmaxValue;
      argmaxBestIndex = argmaxGroupIndex;
    }
  }
  argmaxSharedValues[localInvocationIndex] = argmaxBestValue;
  argmaxSharedIndices[localInvocationIndex] = argmaxBestIndex;
  ${reduceTree}
  if (localInvocationIndex == 0u) {
    outputValue[outputValueOffset] = argmaxSharedValues[0];
    outputIndex[outputIndexOffset] = select(argmaxSharedIndices[0], 0u, argmaxSharedIndices[0] == 0xffffffffu);
  }`
    })
  ];
}
