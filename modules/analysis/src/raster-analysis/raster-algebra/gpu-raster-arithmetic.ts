// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGraphOutputViews as getRasterPreflightOutputViews} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateRasterStorageBindings} from '../raster-grid/raster-storage-preflight';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  getRasterAlgebraInputNodes,
  getRasterAlgebraInputViews,
  isRasterAlgebraBand,
  validateRasterAlgebraInput,
  validateRasterAlgebraInputBindings
} from './raster-algebra-input';
import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {validateRasterBandBelongsToGraph} from '../raster-band-utils';
import {GPU_RASTER_ARITHMETIC_PARAMETER_LENGTH} from './local-operations-parameters';
import {
  getRasterAlgebraValueWGSL,
  validateRasterAlgebraAliasing,
  validateRasterAlgebraCount,
  validateRasterAlgebraGraph,
  validateRasterAlgebraNoData,
  validateRasterAlgebraView
} from './raster-algebra-utils';

const OPERATION = 'GPURasterArithmetic';

/** Caller-owned output of {@link GPURasterArithmetic}. */
export type GPURasterArithmeticOutput = {
  /** Result per cell; NaN for nodata inputs and outside an operation's domain. */
  values: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPURasterArithmetic}.
 *
 * Per-frame (no recompile): input contents and `parameters` (operation, operand scales and
 * offsets, constant `b`, clamp). Topology: `cellCount`, `noDataValue`, and whether `b` is a view.
 */
export type GPURasterArithmeticProps = {
  /** Prefix for generated node IDs. Defaults to `'raster-arithmetic'`. */
  id?: string;
  /** Cells to evaluate. */
  cellCount: number;
  /** Raster grid, required for published bands. Band metadata and grid shape are compile-time. */
  grid?: GPURasterMetadata;
  /** First operand: published band or legacy float values. */
  a: GraphDataView<'float32'> | GPURasterBand;
  /** Optional second operand raster; the per-frame `constantB` is used when absent. */
  b?: GraphDataView<'float32'> | GPURasterBand;
  /**
   * Optional finite sentinel for legacy values. Bands use their own sentinel, validity and calibration. It never applies to the
   * per-frame `constantB`: `maximum(a, 0)` with `noDataValue: 0` is defined wherever `a` is.
   */
  noDataValue?: number;
  /**
   * Optional input row filter (`uint32`, at least `cellCount` rows): nonzero keeps the cell, zero
   * marks it no-data and the result is NaN. Combines with NaN and `noDataValue`.
   */
  mask?: GraphDataView<'uint32'>;
  /** Per-frame float32 view written with `getGPURasterArithmeticParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned output. */
  output: GPURasterArithmeticOutput;
};

/**
 * A small per-cell raster calculator: `op(a * scaleA + offsetA, b * scaleB + offsetB)` then an
 * optional clamp, with the operation chosen per frame.
 *
 * Binary operations: add, subtract, multiply, divide, minimum, maximum, power, absolute difference,
 * and normalized difference `(a - b) / (a + b)` (NDVI, NDWI). Unary operations on `a`: absolute
 * value, square root, natural logarithm, exponential, floor, ceil, and round (half to even).
 * Domain errors give NaN: division by zero (also `a + b == 0` for normalized difference), square
 * root of a negative, logarithm of a non-positive value, and power of a negative base or of zero
 * to a non-positive exponent. It is not an expression compiler: chain contributors for longer formulas.
 */
export class GPURasterArithmetic implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterArithmeticProps;

  constructor(props: GPURasterArithmeticProps) {
    this.id = props.id ?? 'raster-arithmetic';
    this.props = props;
    const {id} = this;
    const {cellCount, output} = props;
    validateRasterAlgebraCount(id, 'cellCount', cellCount, 0, 0xffffffff);
    validateRasterAlgebraNoData(id, props.noDataValue);
    if (!props.a) {
      throw new Error(`${id} needs a`);
    }
    validateRasterAlgebraInput(id, 'a', props.a, props.grid, cellCount);
    if (props.b) validateRasterAlgebraInput(id, 'b', props.b, props.grid, cellCount);
    validateRasterAlgebraView(id, 'mask', props.mask, 'uint32', cellCount);
    validateRasterAlgebraView(
      id,
      'parameters',
      props.parameters,
      'float32',
      GPU_RASTER_ARITHMETIC_PARAMETER_LENGTH
    );
    if (!output?.values) {
      throw new Error(`${id} needs output.values`);
    }
    validateRasterAlgebraView(id, 'output.values', output.values, 'float32', cellCount);
    validateRasterAlgebraAliasing(
      id,
      [output.values],
      [
        ...getRasterAlgebraInputViews(props.a),
        ...getRasterAlgebraInputViews(props.b),
        props.mask,
        props.parameters
      ]
    );
  }

  /** Returns one per-cell kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const storageProps = this.props;
    validateRasterAlgebraInputBindings(
      graph,
      this.id,
      [storageProps.a, storageProps.b],
      storageProps.grid
    );
    validateRasterStorageBindings(graph, this.id, [
      storageProps.mask,
      storageProps.parameters,
      ...getRasterPreflightOutputViews(storageProps.output)
    ]);

    const {id, props} = this;
    validateRasterAlgebraGraph(id, graph, [
      ...getRasterAlgebraInputViews(props.a),
      ...getRasterAlgebraInputViews(props.b),
      props.mask,
      props.parameters,
      props.output.values
    ]);
    // Buffer-view checks above do not include textures. Validate both before either conversion
    // allocates scratch, so a rejected operand leaves the graph available for a corrected retry.
    for (const input of [props.a, props.b]) {
      if (input && isRasterAlgebraBand(input))
        validateRasterBandBelongsToGraph(id, graph, input, []);
    }
    const aSource = getRasterAlgebraInputNodes(graph, `${id}-a`, props.a, props.grid);
    const bSource = props.b
      ? getRasterAlgebraInputNodes(graph, `${id}-b`, props.b, props.grid)
      : undefined;
    const bindings: WGSLKernelBinding[] = [
      {name: 'aValues', view: aSource.values, type: 'f32', access: 'read'},
      ...(props.b
        ? [{name: 'bValues', view: bSource!.values, type: 'f32', access: 'read'} as const]
        : []),
      ...(props.mask
        ? [{name: 'mask', view: props.mask, type: 'u32', access: 'read'} as const]
        : []),
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'valuesOut', view: props.output.values, type: 'f32', access: 'read_write'}
    ];
    return [
      ...aSource.nodes,
      ...(bSource?.nodes ?? []),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-evaluate`,
        operation: OPERATION,
        variant: 'evaluate',
        bindings,
        invocationCount: props.cellCount,
        declarations: `${getRasterAlgebraValueWGSL(props.noDataValue)}
fn getPower(base: f32, exponent: f32) -> f32 {
  if (base > 0.0) {
    return pow(base, exponent);
  }
  if (base == 0.0 && exponent > 0.0) {
    return 0.0;
  }
  return getNaN();
}`,
        body: `let operation = u32(params[paramsOffset]);
  let aRaw = aValues[aValuesOffset + index];
  let bRaw = ${props.b ? 'bValues[bValuesOffset + index]' : 'params[paramsOffset + 5u]'};
  let isUnary = operation >= 16u;
  // The sentinel marks raster cells only; a constant second operand that equals it is a value.
  let bMissing = ${props.b ? `${isRasterAlgebraBand(props.b) ? 'isNaNValue' : 'isNoDataValue'}(bRaw)` : 'isNaNValue(bRaw)'};
  let isDefined = !${isRasterAlgebraBand(props.a) ? 'isNaNValue' : 'isNoDataValue'}(aRaw) && (isUnary || !bMissing)${props.mask ? ' && mask[maskOffset + index] != 0u' : ''};
  let a = aRaw * params[paramsOffset + 1u] + params[paramsOffset + 2u];
  let b = bRaw * params[paramsOffset + 3u] + params[paramsOffset + 4u];
  let nan = getNaN();
  var result = nan;
  switch operation {
    case 0u: { result = a + b; }
    case 1u: { result = a - b; }
    case 2u: { result = a * b; }
    case 3u: { result = select(a / b, nan, b == 0.0); }
    case 4u: { result = min(a, b); }
    case 5u: { result = max(a, b); }
    case 6u: { result = getPower(a, b); }
    case 7u: { result = abs(a - b); }
    case 8u: {
      let total = a + b;
      result = select((a - b) / total, nan, total == 0.0);
    }
    case 16u: { result = abs(a); }
    case 17u: { result = select(sqrt(a), nan, a < 0.0); }
    case 18u: { result = select(log(a), nan, a <= 0.0); }
    case 19u: { result = exp(a); }
    case 20u: { result = floor(a); }
    case 21u: { result = ceil(a); }
    case 22u: { result = round(a); }
    default: { result = nan; }
  }
  if (isDefined && !isNaNValue(result)) {
    result = min(max(result, params[paramsOffset + 6u]), params[paramsOffset + 7u]);
  } else {
    result = nan;
  }
  valuesOut[valuesOutOffset + index] = result;`
      })
    ];
  }
}
