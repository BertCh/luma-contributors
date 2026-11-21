// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUParameterBuffer,
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Float32 layout: [coefficient relative-change tolerance]. */
export const GPU_SUR_PARAMETER_LENGTH = 1;
/** Invalid runtime tolerance in a supplied parameters view; no GLS fit is published. */
export const GPU_SUR_STATUS_INVALID_PARAMETERS = 5;
/** Float32 normal range avoids underflow/flush-to-zero and overflow in a stopping threshold. */
const MINIMUM_TOLERANCE = 2 ** -126;
const MAXIMUM_TOLERANCE = 3.4028234663852886e38;
/** Per-frame convergence setting, independent of the construction-time iteration budget. */
export type GPUSeeminglyUnrelatedRegressionSettings = {tolerance?: number};

/** Packs the per-frame tolerance. Use GPUParameterBuffer.write between encodings.
 * GPU tolerance must remain a finite normal positive float32; use the CPU twin for a smaller
 * positive float64 tolerance. Layout: [tolerance], default 1e-5.
 */
export function getGPUSeeminglyUnrelatedRegressionParameterValues(
  settings: GPUSeeminglyUnrelatedRegressionSettings = {},
  target: Float32Array = new Float32Array(GPU_SUR_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPUSeeminglyUnrelatedRegressionParameterValues';
  const tolerance = settings.tolerance ?? 1e-5;
  validateRequirement(
    id,
    Number.isFinite(Math.fround(tolerance)) && Math.fround(tolerance) >= MINIMUM_TOLERANCE,
    `tolerance must round to a finite normal positive float32 in [${MINIMUM_TOLERANCE}, ${MAXIMUM_TOLERANCE}]; values outside this range overflow or can become zero on the GPU; use CPUSeeminglyUnrelatedRegressionOperation for positive finite float64 tolerances outside this range`
  );
  validateCapacity(id, 'target.length', target.length, GPU_SUR_PARAMETER_LENGTH);
  target[0] = tolerance;
  return target;
}

/** Upload convenience settings through runtime storage; compilation owns the source buffer.
 * No numeric tolerance is embedded in shader source. @internal
 */
export function createSURParameterNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  values: Float32Array
): {parameters: GraphDataView<'float32'>; nodes: GPUCommandNode<Parameters>[]} {
  const parameters = createTransientView(
    graph,
    `${id}-parameters`,
    'float32',
    GPU_SUR_PARAMETER_LENGTH,
    Buffer.STORAGE | Buffer.COPY_DST
  );
  return {
    parameters,
    nodes: [
      {
        id: `${id}-upload-parameters`,
        type: 'copy',
        resources: [{buffer: parameters, usage: 'copy-destination'}],
        workload: {
          operation: 'GPUSeeminglyUnrelatedRegression',
          commandCount: 1,
          writeByteLength: values.byteLength
        },
        compile: ({device}) => {
          const source = new GPUParameterBuffer(device, {
            id: `${id}-parameter-source`,
            format: 'float32',
            length: GPU_SUR_PARAMETER_LENGTH,
            values
          });
          return {
            encode: ({commandEncoder, getBuffer}) =>
              commandEncoder.copyBufferToBuffer({
                sourceBuffer: source.buffer,
                destinationBuffer: getBuffer(parameters),
                size: GPU_SUR_PARAMETER_LENGTH * 4
              }),
            destroy: () => source.destroy()
          };
        }
      }
    ]
  };
}
