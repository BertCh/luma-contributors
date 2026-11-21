// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUTrajectoryKalmanResolvedProps} from './gpu-trajectory-kalman';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {MULTIPLICATION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/multiplication-rounding-error-wgsl';

/** Retain position high/low parts through predictions, measurement corrections and RTS. @internal */
export const KALMAN_LOCAL_ORIGIN_WGSL = `${ADDITION_ROUNDING_ERROR_WGSL}
${MULTIPLICATION_ROUNDING_ERROR_WGSL}
struct KalmanPosition { high: vec2<f32>, low: vec2<f32> }
fn kalmanAdditionError(first: vec2<f32>, second: vec2<f32>, sum: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(getAdditionRoundingError(first.x, second.x, sum.x), getAdditionRoundingError(first.y, second.y, sum.y));
}
fn kalmanPositionNormalize(high: vec2<f32>, low: vec2<f32>) -> KalmanPosition {
  let sum = high + low;
  return KalmanPosition(sum, kalmanAdditionError(high, low, sum));
}
fn kalmanPositionAdd(first: KalmanPosition, second: KalmanPosition) -> KalmanPosition {
  let high = first.high + second.high;
  let low = (first.low + second.low) + kalmanAdditionError(first.high, second.high, high);
  return kalmanPositionNormalize(high, low);
}
fn kalmanPositionSubtract(first: KalmanPosition, second: KalmanPosition) -> KalmanPosition {
  return kalmanPositionAdd(first, KalmanPosition(-second.high, -second.low));
}
fn kalmanPositionProduct(first: vec2<f32>, second: vec2<f32>) -> KalmanPosition {
  let high = first * second;
  return KalmanPosition(high, vec2<f32>(getMultiplicationRoundingError(first.x, second.x, high.x), getMultiplicationRoundingError(first.y, second.y, high.y)));
}
fn kalmanPositionScale(position: KalmanPosition, scale: vec2<f32>) -> KalmanPosition {
  return kalmanPositionAdd(kalmanPositionProduct(position.high, scale), kalmanPositionProduct(position.low, scale));
}
fn kalmanPositionMultiply(matrix: mat2x2<f32>, position: KalmanPosition) -> KalmanPosition {
  return kalmanPositionAdd(kalmanPositionScale(KalmanPosition(vec2<f32>(position.high.x), vec2<f32>(position.low.x)), matrix[0]), kalmanPositionScale(KalmanPosition(vec2<f32>(position.high.y), vec2<f32>(position.low.y)), matrix[1]));
}
fn kalmanPositionAdvance(position: KalmanPosition, velocity: vec2<f32>, delta: f32) -> KalmanPosition {
  return kalmanPositionAdd(position, kalmanPositionProduct(velocity, vec2<f32>(delta)));
}
fn kalmanPositionValue(position: KalmanPosition) -> vec2<f32> { return position.high + position.low; }
`;

/** Metadata shared by filter, smoother and absolute publication. @internal */
export type KalmanLocalMetadata<Parameters> = {
  parameters: GraphDataView<'float32'>;
  origins: GraphDataView<'float32x4'>;
  initialize: GPUCommandNode<Parameters>;
  originSource: string;
};

/** Share origins through the smoother's parameter binding to preserve its eight-buffer budget. @internal */
export function createKalmanLocalMetadata<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUTrajectoryKalmanResolvedProps
): KalmanLocalMetadata<Parameters> {
  const parameterCount = props.covarianceMode === 'full' ? 10 : 4;
  const parameters = createTransientView(
    graph,
    `${id}-local-metadata`,
    'float32',
    parameterCount + 4 * Math.max(props.positions.length, 1)
  );
  const origins = graph.createDataView(parameters.buffer, {
    format: 'float32x4',
    byteOffset: parameterCount * 4,
    length: props.positions.length
  });
  const initialize = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-initialize-local-metadata`,
    operation: 'GPUTrajectoryKalman',
    variant: 'initialize-local-metadata',
    invocationCount: parameters.length,
    bindings: [
      {name: 'inputParameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'metadata', view: parameters, type: 'f32', access: 'read_write'}
    ],
    body: `var value = 0.0;
if (index < ${parameterCount}u) { value = inputParameters[inputParametersOffset + index]; }
metadata[metadataOffset + index] = value;`
  });
  const originSource = `fn rowOrigin(row: u32) -> vec2<f32> { let base = parametersOffset + ${parameterCount}u + 4u * row; return vec2<f32>(parameters[base], parameters[base + 1u]); }
fn rowSmoothedOrigin(row: u32) -> vec2<f32> { let base = parametersOffset + ${parameterCount}u + 4u * row + 2u; return vec2<f32>(parameters[base], parameters[base + 1u]); }
fn storeSmoothedOrigin(row: u32, origin: vec2<f32>) { let base = parametersOffset + ${parameterCount}u + 4u * row + 2u; parameters[base] = origin.x; parameters[base + 1u] = origin.y; }`;
  return {parameters, origins, initialize, originSource};
}

/** Publish absolute positions only after every backward correction has consumed local states. @internal */
export function createKalmanAbsolutePublicationNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUTrajectoryKalmanResolvedProps,
  origins: GraphDataView<'float32x4'>
): GPUCommandNode<Parameters> {
  const smoothing = Boolean(props.output.smoothedState);
  return createWGSLKernelNode(graph, {
    id: `${id}-publish-absolute`,
    operation: 'GPUTrajectoryKalman',
    variant: 'publish-absolute',
    invocationCount: props.positions.length,
    bindings: [
      {name: 'origins', view: origins, type: 'f32', access: 'read'},
      {name: 'state', view: props.output.filteredState, type: 'f32', access: 'read_write'},
      ...(props.output.smoothedState
        ? [
            {
              name: 'smoothedState',
              view: props.output.smoothedState,
              type: 'f32' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ],
    body: `let origin = vec2<f32>(origins[originsOffset + 4u * index], origins[originsOffset + 4u * index + 1u]);
state[stateOffset + 4u * index] += origin.x;
state[stateOffset + 4u * index + 1u] += origin.y;
${smoothing ? 'smoothedState[smoothedStateOffset + 4u * index] += origins[originsOffset + 4u * index + 2u];\nsmoothedState[smoothedStateOffset + 4u * index + 1u] += origins[originsOffset + 4u * index + 3u];' : ''}`
  });
}
