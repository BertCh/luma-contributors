// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  GPU_SCALAR_SCALE_MAXIMUM_DOMAIN_COUNT,
  GPU_SCALAR_SCALE_PARAMETER_LENGTH
} from './scalar-scale-parameters';
import {SCALAR_SCALE_WGSL} from './scalar-scale-wgsl';

export type GPUScalarScaleProps = {
  id?: string;
  /** Prepared scalar f32 values, or exact uint32 category codes for point mode only. */
  values: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Source validity only; display filtering is a separate renderer selection mask. */
  validity?: GraphDataView<'uint32'>;
  /** Numeric endpoint pair, or unique ordered finite point-domain entries. */
  domain: GraphDataView<'float32'>;
  /** Exactly the first two entries supply the output endpoints, including descending ranges. */
  range: GraphDataView<'float32'>;
  /** getGPUScalarScaleParameterValues layout; contents replay without recompilation. */
  parameters: GraphDataView<'float32'>;
  /** Capacity, at least two and no greater than 256; point lookup is O(capacity) per row. */
  maximumDomainCount: number;
  output: {
    values: GraphDataView<'float32'>;
    /** One for a finite mapped result; zero for null/unknown/unsupported numerical results. */
    validity?: GraphDataView<'uint32'>;
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
 * Inputs are packed single views; a chunk-preserving runtime adds one contributor per source
 * chunk and shares domain/range controls. Empty chunks generate no command. Borrowed resources
 * are never allocated/destroyed here. Writable outputs must use separate buffers from each
 * other and all inputs, including disjoint ranges in the same buffer. Only view topology/capacity
 * changes require recompilation.
 */
export class GPUScalarScale implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUScalarScaleProps;
  constructor(props: GPUScalarScaleProps) {
    this.id = props.id ?? 'scalar-scale';
    this.props = props;
    const id = this.id;
    validateCapacity(
      id,
      'maximumDomainCount',
      props.maximumDomainCount,
      2,
      GPU_SCALAR_SCALE_MAXIMUM_DOMAIN_COUNT
    );
    validatePackedView(props.values, ['float32', 'uint32'], `${id} values`);
    validatePackedView(props.domain, ['float32'], `${id} domain`);
    validatePackedView(props.range, ['float32'], `${id} range`);
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    validatePackedView(props.output.values, ['float32'], `${id} output.values`);
    // Controls cover their declared capacity; output storage covers every source row.
    assert(
      props.domain.length >= props.maximumDomainCount &&
        props.range.length >= 2 &&
        props.parameters.length >= GPU_SCALAR_SCALE_PARAMETER_LENGTH &&
        props.output.values.length >= props.values.length
    );
    if (props.validity) {
      validatePackedView(props.validity, ['uint32'], `${id} validity`);
      // Source validity follows the exact source row topology.
      assert(props.validity.length === props.values.length);
    }
    if (props.output.validity) {
      validatePackedView(props.output.validity, ['uint32'], `${id} output.validity`);
      // Every source row receives an output validity value.
      assert(props.output.validity.length >= props.values.length);
    }
    const outputs = [props.output.values, props.output.validity].filter(
      (view): view is GraphDataView<'float32'> | GraphDataView<'uint32'> => Boolean(view)
    );
    // Writable bindings use distinct buffers, including when byte ranges would be disjoint.
    assert(new Set(outputs.map(view => view.buffer)).size === outputs.length);
    validateGraphOutputsDisjointFromInputs(id, outputs, [
      props.values,
      props.validity,
      props.domain,
      props.range,
      props.parameters
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.validity,
      props.domain,
      props.range,
      props.parameters,
      props.output.values,
      props.output.validity
    ]);
    if (props.values.length === 0) return [];
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-map`,
        operation: 'GPUScalarScale',
        variant: 'finite-float32',
        invocationCount: props.values.length,
        bindings: [
          read('values', props.values, props.values.format === 'uint32' ? 'u32' : 'f32'),
          ...(props.validity ? [read('sourceValidity', props.validity, 'u32')] : []),
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
  outputValues[outputValuesOffset + index] = select(0.0, nullValue, scalarIsFinite(nullValue));
  ${props.output.validity ? 'outputValidity[outputValidityOffset + index] = 0u;' : ''}
  ${props.validity ? 'if (sourceValidity[sourceValidityOffset + index] == 0u) { return; }' : ''}
  let scaleParameter = parameters[parametersOffset];
  let domainCountParameter = parameters[parametersOffset + 1u];
  if (!scalarIsFinite(scaleParameter) || !scalarIsFinite(domainCountParameter) || !scalarIsFinite(nullValue)) { return; }
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
  if (!scalarIsFinite(value)) { return; }
  var mapped: f32;
  if (scale == 4u) {
    if (domainCount == 0u) { return; }
    let firstRange = range[rangeOffset];
    let secondRange = range[rangeOffset + 1u];
    if (!scalarIsFinite(firstRange) || !scalarIsFinite(secondRange)) { return; }
    var categoryIndex = MAXIMUM_DOMAIN_COUNT;
    for (var candidate = 0u; candidate < domainCount; candidate++) {
      let category = domain[domainOffset + candidate];
      if (!scalarIsFinite(category)) { return; }
      if (category == value && categoryIndex == MAXIMUM_DOMAIN_COUNT) { categoryIndex = candidate; }
    }
    if (categoryIndex == MAXIMUM_DOMAIN_COUNT) { return; }
    let position = select(0.5, f32(categoryIndex) / f32(max(domainCount - 1u, 1u)), domainCount > 1u);
    mapped = scalarInterpolate(firstRange, secondRange, position);
  } else {
    if (domainCount != 2u) { return; }
    let firstDomain = domain[domainOffset];
    let secondDomain = domain[domainOffset + 1u];
    if (!scalarIsFinite(firstDomain) || !scalarIsFinite(secondDomain)) { return; }
    if (scale == 2u && !((firstDomain > 0.0 && secondDomain > 0.0) || (firstDomain < 0.0 && secondDomain < 0.0))) { return; }
    if (scale == 3u && invertParameter == 0.0) {
      mapped = select(value, firstDomain, firstDomain == secondDomain);
    } else {
      let firstRange = select(range[rangeOffset], secondDomain, scale == 3u);
      let secondRange = select(range[rangeOffset + 1u], firstDomain, scale == 3u);
      if (!scalarIsFinite(firstRange) || !scalarIsFinite(secondRange)) { return; }
      if (firstDomain == secondDomain) {
        mapped = select(scalarInterpolate(firstRange, secondRange, 0.5), firstDomain, scale == 3u);
      } else {
        if (scale == 2u && ((value <= 0.0 && firstDomain > 0.0) || (value >= 0.0 && firstDomain < 0.0))) { return; }
        let first = scalarTransform(firstDomain, scale);
        let second = scalarTransform(secondDomain, scale);
        let transformed = scalarTransform(value, scale);
        if (!scalarIsFinite(first) || !scalarIsFinite(second) || !scalarIsFinite(transformed) || first == second) { return; }
        // Normalize large magnitudes before dividing: hardware may flush the reciprocal of
        // a near-maximum f32 denominator to zero even when the quotient is ordinary finite.
        let magnitude = max(max(abs(first), abs(second)), abs(transformed));
        let normalization = select(1.0, 1e-20, magnitude > 1e19);
        let scaledFirst = first * normalization;
        let scaledSecond = second * normalization;
        if (scaledFirst == scaledSecond) { return; }
        let position = (transformed * normalization - scaledFirst) / (scaledSecond - scaledFirst);
        if (!scalarIsFinite(position)) { return; }
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
  if (!scalarIsFinite(mapped)) { return; }
  outputValues[outputValuesOffset + index] = mapped;
  ${props.output.validity ? 'outputValidity[outputValidityOffset + index] = 1u;' : ''}`
      })
    ];
  }
}
