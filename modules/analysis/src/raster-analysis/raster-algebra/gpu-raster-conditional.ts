// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {GPU_RASTER_CONDITIONAL_PARAMETER_LENGTH} from './local-operations-parameters';
import {
  getRasterAlgebraValueWGSL,
  validateRasterAlgebraAliasing,
  validateRasterAlgebraCount,
  validateRasterAlgebraGraph,
  validateRasterAlgebraNoData,
  validateRasterAlgebraView
} from './raster-algebra-utils';

import {
  getRasterAlgebraInputNodes,
  getRasterAlgebraInputViews,
  isRasterAlgebraBand,
  validateRasterAlgebraInput
} from './raster-algebra-input';

const OPERATION = 'GPURasterConditional';

/** Caller-owned outputs of {@link GPURasterConditional}. */
export type GPURasterConditionalOutput = {
  /** `condition ? a : b` per cell; NaN where the condition or the chosen operand is nodata. */
  values: GraphDataView<'float32'>;
  /** Optional condition result (not a no-data flag): `1` true, `0` false or nodata. */
  conditionResult?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPURasterConditional}. Provide exactly one of `mask` and
 * `conditionValues`.
 *
 * Per-frame (no recompile): input contents and `parameters` (comparison, thresholds, constants).
 * Topology: `cellCount`, `noDataValue`, and which views exist (a missing `a` or `b` uses its
 * per-frame constant). Band metadata and grid dimensions are compile-time.
 */
export type GPURasterConditionalProps = {
  /** Prefix for generated node IDs. Defaults to `'raster-conditional'`. */
  id?: string;
  /** Cells to evaluate. */
  cellCount: number;
  /** Raster grid, required when any input is a published band. */
  grid?: GPURasterMetadata;
  /** Boolean condition raster: nonzero is true. */
  mask?: GraphDataView<'uint32'>;
  /** Published condition band or legacy float values compared against per-frame thresholds. */
  conditionValues?: GraphDataView<'float32'> | GPURasterBand;
  /** Published band or legacy values where the condition holds. Defaults to the per-frame `constantA`. */
  a?: GraphDataView<'float32'> | GPURasterBand;
  /** Published band or legacy values where the condition fails. Defaults to the per-frame `constantB`. */
  b?: GraphDataView<'float32'> | GPURasterBand;
  /**
   * Optional finite sentinel treated as nodata in the `conditionValues`, `a` and `b` legacy
   * rasters. Bands use their own native sentinel, validity and calibration. It never applies to the per-frame constants: `constantB: 0` with `noDataValue: 0` writes 0.
   */
  noDataValue?: number;
  /** Per-frame float32 view written with `getGPURasterConditionalParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPURasterConditionalOutput;
};

/**
 * Per-cell `where(condition, a, b)` (ArcGIS `Con`, numpy `where`).
 *
 * The condition is a `uint32` mask or a float raster compared against per-frame thresholds
 * (`<`, `<=`, `>`, `>=`, `==`, `!=`, inclusive `between`); `a` and `b` are rasters or per-frame
 * constants. Changing the comparison, thresholds, or constants never recompiles.
 */
export class GPURasterConditional implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterConditionalProps;

  constructor(props: GPURasterConditionalProps) {
    this.id = props.id ?? 'raster-conditional';
    this.props = props;
    const {id} = this;
    const {cellCount, output} = props;
    validateRasterAlgebraCount(id, 'cellCount', cellCount, 0, 0xffffffff);
    validateRasterAlgebraNoData(id, props.noDataValue);
    if (Boolean(props.mask) === Boolean(props.conditionValues)) {
      throw new Error(`${id} needs exactly one of mask and conditionValues`);
    }
    validateRasterAlgebraView(id, 'mask', props.mask, 'uint32', cellCount);
    if (props.conditionValues) {
      validateRasterAlgebraInput(
        id,
        'conditionValues',
        props.conditionValues,
        props.grid,
        cellCount
      );
    }
    if (props.a) {
      validateRasterAlgebraInput(id, 'a', props.a, props.grid, cellCount);
    }
    if (props.b) {
      validateRasterAlgebraInput(id, 'b', props.b, props.grid, cellCount);
    }
    validateRasterAlgebraView(
      id,
      'parameters',
      props.parameters,
      'float32',
      GPU_RASTER_CONDITIONAL_PARAMETER_LENGTH
    );
    if (!output?.values) {
      throw new Error(`${id} needs output.values`);
    }
    validateRasterAlgebraView(id, 'output.values', output.values, 'float32', cellCount);
    validateRasterAlgebraView(
      id,
      'output.conditionResult',
      output.conditionResult,
      'uint32',
      cellCount
    );
    validateRasterAlgebraAliasing(
      id,
      [output.values, output.conditionResult],
      [
        props.mask,
        ...getRasterAlgebraInputViews(props.conditionValues),
        ...getRasterAlgebraInputViews(props.a),
        ...getRasterAlgebraInputViews(props.b),
        props.parameters
      ]
    );
  }

  /** Returns one per-cell kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateRasterAlgebraGraph(id, graph, [
      props.mask,
      ...getRasterAlgebraInputViews(props.conditionValues),
      ...getRasterAlgebraInputViews(props.a),
      ...getRasterAlgebraInputViews(props.b),
      props.parameters,
      output.values,
      output.conditionResult
    ]);
    const conditionSource = props.conditionValues
      ? getRasterAlgebraInputNodes(graph, `${id}-condition`, props.conditionValues, props.grid)
      : undefined;
    const aSource = props.a
      ? getRasterAlgebraInputNodes(graph, `${id}-a`, props.a, props.grid)
      : undefined;
    const bSource = props.b
      ? getRasterAlgebraInputNodes(graph, `${id}-b`, props.b, props.grid)
      : undefined;
    const bindings: WGSLKernelBinding[] = [
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'}
    ];
    if (props.mask) {
      bindings.push({name: 'conditionMask', view: props.mask, type: 'u32', access: 'read'});
    }
    if (props.conditionValues) {
      bindings.push({
        name: 'conditionValues',
        view: conditionSource!.values,
        type: 'f32',
        access: 'read'
      });
    }
    if (props.a) {
      bindings.push({name: 'aValues', view: aSource!.values, type: 'f32', access: 'read'});
    }
    if (props.b) {
      bindings.push({name: 'bValues', view: bSource!.values, type: 'f32', access: 'read'});
    }
    bindings.push({name: 'valuesOut', view: output.values, type: 'f32', access: 'read_write'});
    if (output.conditionResult) {
      bindings.push({
        name: 'maskOut',
        view: output.conditionResult,
        type: 'u32',
        access: 'read_write'
      });
    }
    return [
      ...(conditionSource?.nodes ?? []),
      ...(aSource?.nodes ?? []),
      ...(bSource?.nodes ?? []),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-select`,
        operation: OPERATION,
        variant: 'select',
        bindings,
        invocationCount: props.cellCount,
        declarations: getRasterAlgebraValueWGSL(props.noDataValue),
        body: `var conditionDefined = true;
  var condition = false;
  ${
    props.mask
      ? 'condition = conditionMask[conditionMaskOffset + index] != 0u;'
      : `let conditionValue = conditionValues[conditionValuesOffset + index];
  conditionDefined = !${props.conditionValues && isRasterAlgebraBand(props.conditionValues) ? 'isNaNValue' : 'isNoDataValue'}(conditionValue);
  let comparison = u32(params[paramsOffset]);
  let threshold = params[paramsOffset + 1u];
  let upperThreshold = params[paramsOffset + 2u];
  switch comparison {
    case 0u: { condition = conditionValue < threshold; }
    case 1u: { condition = conditionValue <= threshold; }
    case 2u: { condition = conditionValue > threshold; }
    case 3u: { condition = conditionValue >= threshold; }
    case 4u: { condition = conditionValue == threshold; }
    case 5u: { condition = conditionValue != threshold; }
    default: { condition = conditionValue >= threshold && conditionValue <= upperThreshold; }
  }`
  }
  // The legacy sentinel marks bare raster cells only; each band resolves its own sentinel.
  // A per-frame constant that equals the sentinel is a value
  // (ArcGIS Con, numpy.where); a NaN constant still gives NaN through the select below.
  let aValue = ${props.a ? 'aValues[aValuesOffset + index]' : 'params[paramsOffset + 3u]'};
  let bValue = ${props.b ? 'bValues[bValuesOffset + index]' : 'params[paramsOffset + 4u]'};
  let aMissing = ${props.a && !isRasterAlgebraBand(props.a) ? 'isNoDataValue(aValue)' : 'isNaNValue(aValue)'};
  let bMissing = ${props.b && !isRasterAlgebraBand(props.b) ? 'isNoDataValue(bValue)' : 'isNaNValue(bValue)'};
  let chosen = select(bValue, aValue, condition);
  let isDefined = conditionDefined && !select(bMissing, aMissing, condition);
  valuesOut[valuesOutOffset + index] = select(getNaN(), chosen, isDefined);
  ${output.conditionResult ? 'maskOut[maskOutOffset + index] = select(0u, 1u, conditionDefined && condition);' : ''}`
      })
    ];
  }
}
