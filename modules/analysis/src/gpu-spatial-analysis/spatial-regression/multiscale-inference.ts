// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {validateGraphViewsBelongToGraph} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getRegressionBindings,
  getRegressionWGSL
} from './gpu-generalized-geographically-weighted-regression';
import type {GPUMultiscaleGeographicallyWeightedRegression} from './gpu-multiscale-geographically-weighted-regression';

/** Propagate every response derivative through the actual selected backfit history. @internal */
export function createMultiscaleInference<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUMultiscaleGeographicallyWeightedRegression,
  parameters: GraphDataView<'float32'>,
  state: GraphDataView<'uint32'>
) {
  const {id, props} = contributor;
  const inference = props.inference!;
  const rows = props.response.length,
    columns = props.predictorCount + 1;
  const length = rows * rows * columns;
  let source = createTransientView(graph, `${id}-response-smoother-first`, 'float32', length),
    destination = createTransientView(graph, `${id}-response-smoother-second`, 'float32', length);
  const bindings = getRegressionBindings({...props, parameters});
  const common = getRegressionWGSL(props, {sphereRadius: contributor.sphereRadius});
  validateGraphViewsBelongToGraph(id, graph, Object.values(inference.output));
  return {
    initialize: () =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-response-smoother`,
        operation: 'GPUMultiscaleGeographicallyWeightedRegression',
        variant: 'initialize-full-inference',
        invocationCount: length,
        bindings: [{name: 'smoother', view: source, type: 'f32', access: 'read_write'}],
        body: 'smoother[smootherOffset + index] = 0.0;'
      }),
    advance: (iteration: number, column: number) => {
      const node = createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-response-smoother-${iteration}-${column}`,
        operation: 'GPUMultiscaleGeographicallyWeightedRegression',
        variant: 'full-backfit-smoother',
        invocationCount: rows * rows,
        bindings: [
          ...bindings,
          {name: 'source', view: source, type: 'f32', access: 'read'},
          {name: 'destination', view: destination, type: 'f32', access: 'read_write'},
          {name: 'state', view: state, type: 'u32', access: 'read'}
        ],
        declarations: common,
        body: `let focus = index / ${rows}u; let responseRow = index % ${rows}u; let base = index * ${columns}u; for (var component = 0u; component < ${columns}u; component++) { destination[destinationOffset + base + component] = source[sourceOffset + base + component]; } if (state[stateOffset] != 3u || !included(focus) || !included(responseRow)) { return; }
let bandwidth = parameters[parametersOffset + ${column + 1}u]; var denominator = 0.0; var numerator = 0.0;
for (var row = 0u; row < ${rows}u; row++) { if (!included(row)) { continue; } let value = predictor(row, ${column}u); let weight = spatialWeight(focus,row,bandwidth); denominator += weight * value * value; var partial = select(0.0,1.0,row == responseRow); for (var other = 0u; other < ${columns}u; other++) { if (other != ${column}u) { partial -= predictor(row,other) * source[sourceOffset + (row * ${rows}u + responseRow) * ${columns}u + other]; } } numerator += weight * value * partial; }
destination[destinationOffset + base + ${column}u] = select(getNaN(),numerator / denominator,denominator > 0.0);`
      });
      [source, destination] = [destination, source];
      return node;
    },
    finish: () => [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-full-inference-summary`,
        operation: 'GPUMultiscaleGeographicallyWeightedRegression',
        variant: 'full-inference-summary',
        invocationCount: 1,
        bindings: [
          {name: 'smoother', view: source, type: 'f32', access: 'read'},
          {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
          {name: 'response', view: props.response, type: 'f32', access: 'read'},
          {name: 'fitted', view: props.output.fitted, type: 'f32', access: 'read'},
          {name: 'codes', view: props.output.status.codes, type: 'u32', access: 'read'},
          {name: 'summary', view: inference.output.summary, type: 'f32', access: 'read_write'},
          {name: 'status', view: inference.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
        body: `status[statusOffset] = 1u; for (var slot = 0u; slot < 4u; slot++) { summary[summaryOffset + slot] = getNaN(); } var trace = 0.0; var traceSquared = 0.0; var residualSquares = 0.0; var count = 0u;
for (var focus = 0u; focus < ${rows}u; focus++) { let code = codes[codesOffset + focus]; if (code == 1u) { continue; } if (code != 0u) { return; } count++; let residual = response[responseOffset + focus] - fitted[fittedOffset + focus]; residualSquares += residual * residual;
for (var responseRow = 0u; responseRow < ${rows}u; responseRow++) { var influence = 0.0; for (var column = 0u; column < ${columns}u; column++) { var value = 1.0; if (column > 0u) { value = predictors[predictorsOffset + focus * ${columns - 1}u + column - 1u]; } influence += value * smoother[smootherOffset + (focus * ${rows}u + responseRow) * ${columns}u + column]; } traceSquared += influence * influence; if (focus == responseRow) { trace += influence; } } }
let degrees = f32(count) - ${inference.varianceConvention === 'smoother-df' ? '(2.0 * trace - traceSquared)' : 'trace'}; let variance = residualSquares / degrees;
summary[summaryOffset] = trace; summary[summaryOffset + 1u] = traceSquared; summary[summaryOffset + 2u] = variance; summary[summaryOffset + 3u] = degrees; status[statusOffset] = select(2u,0u,count > 0u && degrees > 0.0 && variance >= 0.0 && isFiniteBits(variance) && isFiniteBits(traceSquared));`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-full-inference-publish`,
        operation: 'GPUMultiscaleGeographicallyWeightedRegression',
        variant: 'full-inference-publish',
        invocationCount: rows * columns,
        bindings: [
          {name: 'smoother', view: source, type: 'f32', access: 'read'},
          {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
          {name: 'codes', view: props.output.status.codes, type: 'u32', access: 'read'},
          {name: 'summary', view: inference.output.summary, type: 'f32', access: 'read'},
          {name: 'status', view: inference.output.status, type: 'u32', access: 'read'},
          {
            name: 'standardErrors',
            view: inference.output.standardErrors,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'effective',
            view: inference.output.effectiveParameters,
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: GET_NAN_WGSL,
        body: `let focus = index / ${columns}u; let column = index % ${columns}u; standardErrors[standardErrorsOffset + index] = getNaN(); if (focus == 0u) { effective[effectiveOffset + column] = getNaN(); } if (status[statusOffset] != 0u) { return; }
if (codes[codesOffset + focus] == 0u) { var sum = 0.0; for (var responseRow = 0u; responseRow < ${rows}u; responseRow++) { let derivative = smoother[smootherOffset + (focus * ${rows}u + responseRow) * ${columns}u + column]; sum += derivative * derivative; } standardErrors[standardErrorsOffset + index] = sqrt(max(0.0,sum * summary[summaryOffset + 2u])); }
if (focus == 0u) { var count = 0.0; for (var row = 0u; row < ${rows}u; row++) { if (codes[codesOffset + row] != 0u) { continue; } var value = 1.0; if (column > 0u) { value = predictors[predictorsOffset + row * ${columns - 1}u + column - 1u]; } count += value * smoother[smootherOffset + (row * ${rows}u + row) * ${columns}u + column]; } effective[effectiveOffset + column] = count; }`
      })
    ]
  };
}
