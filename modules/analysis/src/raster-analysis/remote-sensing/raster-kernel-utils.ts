// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {GET_QUIET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GraphDataView, GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
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
export function validateRasterCount(count: number, maximum = 0xffffffff): void {
  assert(Number.isSafeInteger(count) && count > 0 && count <= maximum);
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
