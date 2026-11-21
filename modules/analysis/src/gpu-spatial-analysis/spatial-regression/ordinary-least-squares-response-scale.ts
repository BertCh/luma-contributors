// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUOrdinaryLeastSquaresProps} from './gpu-ordinary-least-squares';

/** Scale before subtraction; integer reconstruction also preserves subnormal output bit patterns. */
export const ORDINARY_LEAST_SQUARES_RESPONSE_SCALE_WGSL = /* wgsl */ `
fn getOLSResponseExponent(magnitude: u32) -> i32 {
  if (magnitude == 0u) { return 0; }
  var exponent = i32(magnitude >> 23u) - 127;
  if ((magnitude >> 23u) == 0u) { exponent = -126 - i32(countLeadingZeros(magnitude) - 8u); }
  // Keep ordinary response units unchanged; normalize extreme units before any differences.
  return select(exponent, 0, exponent >= -32 && exponent <= 32);
}
fn scaleOLSResponse(value: f32, power: i32) -> f32 {
  let bits = bitcast<u32>(value);
  let magnitude = bits & 0x7fffffffu;
  let sign = bits & 0x80000000u;
  if (magnitude == 0u || magnitude >= 0x7f800000u || power == 0) { return bitcast<f32>(bits); }
  var exponent = i32(magnitude >> 23u);
  var significand = (magnitude & 0x007fffffu) | 0x00800000u;
  if (exponent == 0) {
    let shift = countLeadingZeros(magnitude) - 8u;
    significand = magnitude << shift;
    exponent = 1 - i32(shift);
  }
  exponent += power;
  if (exponent >= 255) { return bitcast<f32>(sign | 0x7f800000u); }
  if (exponent > 0) { return bitcast<f32>(sign | (u32(exponent) << 23u) | (significand & 0x007fffffu)); }
  let shift = u32(1 - exponent);
  if (shift > 24u) { return bitcast<f32>(sign); }
  var rounded = significand >> shift;
  let remainder = significand & ((1u << shift) - 1u);
  let halfway = 1u << (shift - 1u);
  if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0u)) { rounded += 1u; }
  return bitcast<f32>(sign | rounded);
}
`;

/** Parallel preconditioning of response units before OLS forms centered differences. @internal */
export function getOrdinaryLeastSquaresResponseScaleNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUOrdinaryLeastSquaresProps
) {
  const id = props.id ?? 'ordinary-least-squares';
  const magnitudes = createTransientView(
    graph,
    `${id}-response-magnitudes`,
    'uint32',
    props.response.length
  );
  const maximum = createTransientView(graph, `${id}-response-maximum`, 'uint32', 1);
  const response = createTransientView(
    graph,
    `${id}-response-normalized`,
    'float32',
    props.response.length
  );
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-response-magnitudes`,
      operation: 'GPUOrdinaryLeastSquares',
      variant: 'response-magnitudes',
      invocationCount: props.response.length,
      bindings: [
        {name: 'response', view: props.response, type: 'f32', access: 'read'},
        {name: 'design', view: props.predictors, type: 'f32', access: 'read'},
        ...(props.mask
          ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'magnitudes', view: magnitudes, type: 'u32', access: 'read_write'}
      ],
      body: `magnitudes[magnitudesOffset + index] = 0u;
  ${props.mask ? 'if (mask[maskOffset + index] == 0u) { return; }' : ''}
  for (var column = 0u; column < ${props.predictorCount}u; column++) {
    if ((bitcast<u32>(design[designOffset + index * ${props.predictorCount}u + column]) & 0x7fffffffu) >= 0x7f800000u) { return; }
  }
  let magnitude = bitcast<u32>(response[responseOffset + index]) & 0x7fffffffu;
  if (magnitude < 0x7f800000u) { magnitudes[magnitudesOffset + index] = magnitude; }`
    }),
    ...new GPUReduction({
      id: `${id}-response-maximum`,
      input: magnitudes,
      output: maximum,
      operation: 'max'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-response-normalize`,
      operation: 'GPUOrdinaryLeastSquares',
      variant: 'response-normalize',
      invocationCount: props.response.length,
      bindings: [
        {name: 'source', view: props.response, type: 'f32', access: 'read'},
        {name: 'maximum', view: maximum, type: 'u32', access: 'read'},
        {name: 'response', view: response, type: 'f32', access: 'read_write'}
      ],
      declarations: ORDINARY_LEAST_SQUARES_RESPONSE_SCALE_WGSL,
      body: 'response[responseOffset + index] = scaleOLSResponse(source[sourceOffset + index], -getOLSResponseExponent(maximum[maximumOffset]));'
    })
  ];
  return {response, maximum, nodes};
}

/** Restore original response units after every diagnostic has consumed normalized residuals. @internal */
export function getOrdinaryLeastSquaresResponseRestoreNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUOrdinaryLeastSquaresProps,
  maximum: GraphDataView<'uint32'>
): GPUCommandNode<Parameters>[] {
  const id = props.id ?? 'ordinary-least-squares';
  const output = props.output;
  const rowOutputs = [output.residuals, output.fitted, output.predictions].filter(
    (view): view is GraphDataView<'float32'> => Boolean(view)
  );
  const nodes: GPUCommandNode<Parameters>[] = [];
  if (rowOutputs.length)
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-response-restore-rows`,
        operation: 'GPUOrdinaryLeastSquares',
        variant: 'response-restore-rows',
        invocationCount: props.response.length,
        bindings: [
          {name: 'maximum', view: maximum, type: 'u32', access: 'read'},
          ...rowOutputs.map((view, index) => ({
            name: `values${index}`,
            view,
            type: 'f32' as const,
            access: 'read_write' as const
          }))
        ],
        declarations: ORDINARY_LEAST_SQUARES_RESPONSE_SCALE_WGSL,
        body: `let exponent = getOLSResponseExponent(maximum[maximumOffset]);
  ${rowOutputs.map((_, index) => `values${index}[values${index}Offset + index] = scaleOLSResponse(values${index}[values${index}Offset + index], exponent);`).join('\n  ')}`
      })
    );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-response-restore-report`,
      operation: 'GPUOrdinaryLeastSquares',
      variant: 'response-restore-report',
      invocationCount: Math.max(3, props.predictorCount + 1),
      bindings: [
        {name: 'maximum', view: maximum, type: 'u32', access: 'read'},
        {name: 'coefficients', view: output.coefficients, type: 'f32', access: 'read_write'},
        {name: 'errors', view: output.standardErrors, type: 'f32', access: 'read_write'},
        {name: 'summary', view: output.summary, type: 'f32', access: 'read_write'}
      ],
      declarations: ORDINARY_LEAST_SQUARES_RESPONSE_SCALE_WGSL,
      body: `let exponent = getOLSResponseExponent(maximum[maximumOffset]);
  if (index < ${props.predictorCount + 1}u) {
    coefficients[coefficientsOffset + index] = scaleOLSResponse(coefficients[coefficientsOffset + index], exponent);
    errors[errorsOffset + index] = scaleOLSResponse(errors[errorsOffset + index], exponent);
  }
  if (index < 3u) {
    let slots = array<u32, 3>(3u, 11u, 12u);
    let slot = summaryOffset + slots[index];
    summary[slot] = scaleOLSResponse(summary[slot], 2 * exponent);
  }`
    })
  );
  return nodes;
}
