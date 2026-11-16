// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createClassificationKernelNode as createWGSLKernelNode} from './classification-kernel-node';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  validateCommandNodeBudget,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPU_SCALAR_SCALE_MAXIMUM_DOMAIN_COUNT,
  GPU_SCALAR_SCALE_PARAMETER_LENGTH
} from './scalar-scale-parameters';
import {SCALAR_SCALE_WGSL} from './scalar-scale-wgsl';

import {
  type ClassificationRows,
  type ClassificationRowView,
  validateClassificationBindings,
  validateClassificationRows,
  validateClassificationOutput,
  validateClassificationOutputs
} from './column-classification-rows';

export type GPUScalarScaleProps<
  Rows extends ClassificationRows<'float32' | 'uint32'> =
    | GraphDataView<'float32'>
    | GraphDataView<'uint32'>
> = {
  id?: string;
  /** Packed or chunked scalar f32 values, or exact uint32 category codes for point mode only. */
  values: Rows;
  /**
   * Optional row-aligned packed or chunked `uint32` mask, one entry per source row; nonzero keeps the row, zero maps
   * it to `nullValue` with output validity 0. Source validity only; display filtering is a separate
   * renderer selection mask.
   */
  mask?: ClassificationRowView<'uint32', Rows>;
  /** @deprecated Use `mask`; same meaning (nonzero keeps the row). `mask` wins when both are set. */
  validity?: ClassificationRowView<'uint32', Rows>;
  /** Numeric endpoint pair, or unique ordered finite point-domain entries. */
  domain: GraphDataView<'float32'>;
  /** Exactly the first two entries supply the output endpoints, including descending ranges. */
  range: GraphDataView<'float32'>;
  /** getGPUScalarScaleParameterValues layout; contents replay without recompilation. */
  parameters: GraphDataView<'float32'>;
  /** Capacity, at least two and no greater than 256; point lookup is O(capacity) per row. */
  maximumDomainCount: number;
  output: {
    values: ClassificationRowView<'float32', Rows>;
    /** One for a finite mapped result; zero for null/unknown/unsupported numerical results. */
    validity?: ClassificationRowView<'uint32', Rows>;
  };
};

/**
 * Native-style, source-aligned scalar mapping with no source readback or row compaction.
 * Linear and signed sqrt extrapolate; log accepts positive OR negative same-sign domains.
 * There is no log floor or implicit domain clamp. Uint32 sources are supported only for
 * point mode, with codes at most 2^24; larger codes are invalid before conversion. `identity`
 * implements Kepler's fixed linear scale with range=domain, including a constant domain.
 * Its optional invert control reverses that GPU-derived range, matching native fixed-height
 * inversion (domain[0]+domain[1]-value, not negation); a constant domain remains constant.
 * `point` matches d3 scalePoint's default padding=0 and align=0.5, using the supplied ordinal
 * domain order. Unknown categories map to nullValue; n=1 maps to the range midpoint.
 *
 * Numeric modes require two finite domain endpoints; point has 0..capacity unique finite
 * values. Uniqueness is checked by the host controls helper; GPU domain writers must guarantee
 * it. Null input, nonfinite input, malformed replay controls, log values with the wrong
 * sign, and nonfinite f32 output receive nullValue and output.validity=0. A valid degenerate
 * numeric domain maps every finite input to the range midpoint, matching d3 (even a log input
 * whose sign would otherwise be invalid). Controls helpers reject unsupported host metadata.
 * Overflow/collapsed transformed domains are invalid, never silently assigned a class.
 *
 * This is finite f32 arithmetic, not a claim of bitwise double-precision d3 equivalence.
 * Transcendental rounding and shader subnormal behavior require explicit caller tolerances.
 * Infinity-producing d3 extrapolation and subnormal-sensitive arithmetic are outside parity.
 * Consumers requiring a strict diagnostic should reduce output validity against source validity.
 *
 * Row inputs and outputs accept GraphVectorView with identical ordered chunk lengths.
 * Domain, range and parameters remain packed and are shared by every chunk. Results are bitwise
 * equal to packed input. Empty chunks generate no command. Borrowed resources
 * are never allocated/destroyed here. Writable outputs must have disjoint aligned bindings within each dispatch and disjoint logical
 * writes across chunks. Outputs must use separate buffers from inputs. Only view topology/capacity
 * changes require recompilation.
 */
export class GPUScalarScale<
  Rows extends ClassificationRows<'float32' | 'uint32'> =
    | GraphDataView<'float32'>
    | GraphDataView<'uint32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly props: GPUScalarScaleProps<Rows>;
  /** One command per nonempty source chunk. */
  readonly estimatedCommandNodeCount: number;
  /** Resolved row mask: `props.mask`, else the deprecated `props.validity`. */
  readonly mask: ClassificationRowView<'uint32', Rows> | undefined;
  constructor(props: GPUScalarScaleProps<Rows>) {
    this.id = props.id ?? 'scalar-scale';
    this.props = props;
    this.mask = props.mask ?? props.validity;
    const id = this.id;
    validateCapacity(
      id,
      'maximumDomainCount',
      props.maximumDomainCount,
      2,
      GPU_SCALAR_SCALE_MAXIMUM_DOMAIN_COUNT
    );
    validateClassificationRows(id, 'values', props.values, ['float32', 'uint32']);
    validateSinglePackedView(props.domain, ['float32'], `${id} domain`);
    validateSinglePackedView(props.range, ['float32'], `${id} range`);
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateClassificationOutput(id, 'output.values', props.output.values, 'float32', props.values);
    // Controls cover their declared capacity; output storage covers every source row.
    validateRequirement(
      id,
      props.domain.length >= props.maximumDomainCount,
      'domain must hold at least maximumDomainCount entries'
    );
    validateRequirement(id, props.range.length >= 2, 'range must hold at least 2 entries');
    validateRequirement(
      id,
      props.parameters.length >= GPU_SCALAR_SCALE_PARAMETER_LENGTH,
      `parameters must hold at least ${GPU_SCALAR_SCALE_PARAMETER_LENGTH} values`
    );
    validateRequirement(
      id,
      props.output.values.length >= props.values.length,
      'output.values must hold at least one value per source row'
    );
    if (this.mask) {
      validateClassificationRows(id, 'mask', this.mask, ['uint32'], props.values);
      // The row mask follows the exact source row topology.
      validateRequirement(
        id,
        this.mask.length === props.values.length,
        'mask must hold exactly one entry per source row'
      );
    }
    if (props.output.validity) {
      validateClassificationOutput(
        id,
        'output.validity',
        props.output.validity,
        'uint32',
        props.values
      );
      // Every source row receives an output validity value.
      validateRequirement(
        id,
        props.output.validity.length >= props.values.length,
        'output.validity must hold at least one value per source row'
      );
    }
    const outputs = [props.output.values, props.output.validity];
    validateClassificationOutputs(id, outputs);
    validateGraphOutputsDisjointFromInputs(id, outputs, [
      props.values,
      this.mask,
      props.domain,
      props.range,
      props.parameters
    ]);
    this.estimatedCommandNodeCount = getGraphViewChunks(props.values).filter(
      chunk => chunk.length > 0
    ).length;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;

    validateClassificationBindings(id, graph, {
      values: props.values,
      mask: this.mask,
      domain: props.domain,
      range: props.range,
      parameters: props.parameters,
      'output.values': props.output.values,
      'output.validity': props.output.validity
    });
    const nodes: GPUCommandNode<Parameters>[] = [];
    const valueChunks = getGraphViewChunks(props.values);
    for (let chunkIndex = 0; chunkIndex < valueChunks.length; chunkIndex++) {
      const values = valueChunks[chunkIndex];
      if (values.length === 0) continue;
      const props = {
        ...this.props,
        values,
        output: {
          values: getGraphViewChunks(this.props.output.values)[chunkIndex],
          validity: this.props.output.validity
            ? getGraphViewChunks(this.props.output.validity)[chunkIndex]
            : undefined
        }
      };
      const mask = this.mask ? getGraphViewChunks(this.mask)[chunkIndex] : undefined;
      const chunkId = valueChunks.length === 1 ? id : `${id}-chunk-${chunkIndex}`;
      const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
        name,
        view,
        type,
        access: 'read'
      });
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${chunkId}-map`,
          operation: 'GPUScalarScale',
          variant: 'finite-float32',
          invocationCount: props.values.length,
          bindings: [
            read('values', props.values, props.values.format === 'uint32' ? 'u32' : 'f32'),
            ...(mask ? [read('sourceMask', mask, 'u32')] : []),
            read('domain', props.domain, 'f32'),
            read('range', props.range, 'f32'),
            read('parameters', props.parameters, 'f32'),
            {name: 'outputValues', view: props.output.values, type: 'f32', access: 'read_write'},
            ...(props.output.validity
              ? [
                  {
                    name: 'outputValidity',
                    view: props.output.validity,
                    type: 'u32',
                    access: 'read_write'
                  } satisfies WGSLKernelBinding
                ]
              : [])
          ],
          declarations: `const MAXIMUM_DOMAIN_COUNT: u32 = ${props.maximumDomainCount}u;\n${SCALAR_SCALE_WGSL}`,
          body: `let nullValue = parameters[parametersOffset + 2u];
  outputValues[outputValuesOffset + index] = select(0.0, nullValue, isFiniteValue(nullValue));
  ${props.output.validity ? 'outputValidity[outputValidityOffset + index] = 0u;' : ''}
  ${mask ? 'if (sourceMask[sourceMaskOffset + index] == 0u) { return; }' : ''}
  let scaleParameter = parameters[parametersOffset];
  let domainCountParameter = parameters[parametersOffset + 1u];
  if (!isFiniteValue(scaleParameter) || !isFiniteValue(domainCountParameter) || !isFiniteValue(nullValue)) { return; }
  if (scaleParameter < 0.0 || scaleParameter > 4.0 || floor(scaleParameter) != scaleParameter || domainCountParameter < 0.0 || domainCountParameter > f32(MAXIMUM_DOMAIN_COUNT) || floor(domainCountParameter) != domainCountParameter) { return; }
  let invertParameter = parameters[parametersOffset + 3u];
  if (invertParameter != 0.0 && invertParameter != 1.0) { return; }
  if (invertParameter == 1.0 && scaleParameter != 3.0) { return; }
  let scale = u32(scaleParameter);
  let domainCount = u32(domainCountParameter);
  ${
    props.values.format === 'uint32'
      ? `if (scale != 4u || values[valuesOffset + index] > 16777216u) { return; }
  let value = f32(values[valuesOffset + index]);`
      : 'let value = values[valuesOffset + index];'
  }
  if (!isFiniteValue(value)) { return; }
  var mapped: f32;
  if (scale == 4u) {
    if (domainCount == 0u) { return; }
    let firstRange = range[rangeOffset];
    let secondRange = range[rangeOffset + 1u];
    if (!isFiniteValue(firstRange) || !isFiniteValue(secondRange)) { return; }
    var categoryIndex = MAXIMUM_DOMAIN_COUNT;
    for (var candidate = 0u; candidate < domainCount; candidate++) {
      let category = domain[domainOffset + candidate];
      if (!isFiniteValue(category)) { return; }
      if (category == value && categoryIndex == MAXIMUM_DOMAIN_COUNT) { categoryIndex = candidate; }
    }
    if (categoryIndex == MAXIMUM_DOMAIN_COUNT) { return; }
    let position = select(0.5, f32(categoryIndex) / f32(max(domainCount - 1u, 1u)), domainCount > 1u);
    mapped = scalarInterpolate(firstRange, secondRange, position);
  } else {
    if (domainCount != 2u) { return; }
    let firstDomain = domain[domainOffset];
    let secondDomain = domain[domainOffset + 1u];
    if (!isFiniteValue(firstDomain) || !isFiniteValue(secondDomain)) { return; }
    if (scale == 2u && !((firstDomain > 0.0 && secondDomain > 0.0) || (firstDomain < 0.0 && secondDomain < 0.0))) { return; }
    if (scale == 3u && invertParameter == 0.0) {
      mapped = select(value, firstDomain, firstDomain == secondDomain);
    } else {
      let firstRange = select(range[rangeOffset], secondDomain, scale == 3u);
      let secondRange = select(range[rangeOffset + 1u], firstDomain, scale == 3u);
      if (!isFiniteValue(firstRange) || !isFiniteValue(secondRange)) { return; }
      if (firstDomain == secondDomain) {
        mapped = select(scalarInterpolate(firstRange, secondRange, 0.5), firstDomain, scale == 3u);
      } else {
        if (scale == 2u && ((value <= 0.0 && firstDomain > 0.0) || (value >= 0.0 && firstDomain < 0.0))) { return; }
        let first = scalarTransform(firstDomain, scale);
        let second = scalarTransform(secondDomain, scale);
        let transformed = scalarTransform(value, scale);
        if (!isFiniteValue(first) || !isFiniteValue(second) || !isFiniteValue(transformed) || first == second) { return; }
        // Normalize large magnitudes before dividing: hardware may flush the reciprocal of
        // a near-maximum f32 denominator to zero even when the quotient is ordinary finite.
        let magnitude = max(max(abs(first), abs(second)), abs(transformed));
        let normalization = select(1.0, 1e-20, magnitude > 1e19);
        let scaledFirst = first * normalization;
        let scaledSecond = second * normalization;
        if (scaledFirst == scaledSecond) { return; }
        let position = (transformed * normalization - scaledFirst) / (scaledSecond - scaledFirst);
        if (!isFiniteValue(position)) { return; }
        if (scale == 3u) {
          // Keep both interpolation operands normalized too: GPU algebraic optimization may
          // form their difference, which overflows for opposite near-maximum f32 endpoints.
          mapped = scalarInterpolate(second * normalization, first * normalization, position) / normalization;
        } else {
          mapped = scalarInterpolate(firstRange, secondRange, position);
        }
      }
    }
  }
  if (!isFiniteValue(mapped)) { return; }
  outputValues[outputValuesOffset + index] = mapped;
  ${props.output.validity ? 'outputValidity[outputValidityOffset + index] = 1u;' : ''}`
        })
      );
    }
    return nodes;
  }
}
