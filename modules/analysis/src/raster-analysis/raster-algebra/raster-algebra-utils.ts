// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GraphDataView,
  type GraphVectorView,
  type GPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  isGraphDataView,
  validateCapacity,
  validateFiniteNumber,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';

type RasterAliasingViews =
  | readonly (GraphDataView | undefined)[]
  | Readonly<Record<string, GraphDataView | undefined>>;

/** Retained raster output ports must use distinct physical allocations. @internal */
export function validateRasterWritableBuffers(id: string, outputs: RasterAliasingViews): void {
  const ports = new Map<unknown, string>();
  for (const [name, view] of Object.entries(outputs)) {
    if (!view) continue;
    const port = Array.isArray(outputs) ? `outputs[${name}]` : name;
    const buffer = view.buffer.defaultBuffer
      ? getCoreDefaultBuffer(view.buffer.defaultBuffer)
      : view.buffer;
    const previous = ports.get(buffer);
    if (previous !== undefined)
      throw new Error(`${id} outputs must not share buffers: ${previous} conflicts with ${port}`);
    ports.set(buffer, port);
  }
}

/** Nodata description shared by raster-algebra contributors. @internal */
export type RasterAlgebraNoData = {
  /** Optional finite sentinel compared exactly against cell values. NaN is always nodata. */
  noDataValue?: number;
};

/**
 * WGSL helpers: `getNaN()`, `isFiniteValue(value)` and `isNoDataValue(value)`.
 *
 * NaN is always nodata (bit test, because WGSL may optimise `x != x` away); infinities are values;
 * an optional finite sentinel is compared exactly.
 *
 * @internal
 */
export function getRasterAlgebraValueWGSL(noDataValue: number | undefined): string {
  return /* wgsl */ `
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
${GET_NAN_WGSL}
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
${IS_FINITE_VALUE_WGSL}
fn isNoDataValue(value: f32) -> bool {
  return isNaNValue(value)${noDataValue !== undefined ? ` || value == ${getWGSLFloatLiteral(noDataValue)}` : ''};
}`;
}

/**
 * Throws unless `view` is a single packed float32 or uint32 view with at least `rowCount` rows.
 * An absent view passes.
 *
 * @internal
 */
export function validateRasterAlgebraView(
  id: string,
  name: string,
  view: GraphDataView | GraphVectorView | undefined,
  format: 'float32' | 'uint32',
  rowCount: number
): void {
  if (!view) {
    return;
  }
  validateSinglePackedView(view, [format], `${id} ${name}`);
  if (view.length < rowCount) {
    throw new Error(`${id} ${name} must hold at least ${rowCount} rows`);
  }
}

/** Throws unless `value` is an integer in `[minimum, maximum]`. @internal */
export function validateRasterAlgebraCount(
  id: string,
  name: string,
  value: number,
  minimum: number,
  maximum: number
): void {
  validateCapacity(id, name, value, minimum, maximum);
}

/** Throws unless the optional sentinel is finite. @internal */
export function validateRasterAlgebraNoData(id: string, noDataValue: number | undefined): void {
  if (noDataValue !== undefined) validateFiniteNumber(id, 'noDataValue', noDataValue);
}

/** Collects public output paths without walking inside graph views. @internal */
export function getNamedRasterOutputViews(
  output: object,
  prefix = 'output'
): Record<string, GraphDataView | undefined> {
  const views: Record<string, GraphDataView | undefined> = {};
  for (const [name, value] of Object.entries(output)) {
    const path = `${prefix}.${name}`;
    if (isGraphDataView(value)) views[path] = value;
    else if (value && typeof value === 'object')
      Object.assign(views, getNamedRasterOutputViews(value, path));
  }
  return views;
}

/**
 * Throws when outputs share buffers with each other or with inputs.
 *
 * @internal
 */
export function validateRasterAlgebraAliasing(
  id: string,
  outputs: RasterAliasingViews,
  inputs: RasterAliasingViews
): void {
  validateRasterWritableBuffers(id, outputs);
  validateGraphOutputsDisjointFromInputs(id, outputs, inputs);
}

/** Validates that every defined view belongs to `graph`. @internal */
export function validateRasterAlgebraGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  views: readonly (GraphDataView | undefined)[]
): void {
  validateGraphViewsBelongToGraph(id, graph, views);
}

/**
 * WGSL `fn ${functionName}(first: u32, count: u32, value: f32, closedRight: bool) -> u32` that
 * counts ascending breaks `binding[first .. first + count)` that are `<= value` (or `< value` when
 * `closedRight`) by binary search. The breaks must be sorted ascending; NaN breaks give an
 * unspecified but deterministic class.
 *
 * @internal
 */
export function getBreakSearchWGSL(functionName: string, bindingName: string): string {
  return /* wgsl */ `
fn ${functionName}(first: u32, count: u32, value: f32, closedRight: bool) -> u32 {
  var low = 0u;
  var high = count;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    let boundary = ${bindingName}[${bindingName}Offset + first + middle];
    let isBelow = select(boundary <= value, boundary < value, closedRight);
    if (isBelow) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`;
}
