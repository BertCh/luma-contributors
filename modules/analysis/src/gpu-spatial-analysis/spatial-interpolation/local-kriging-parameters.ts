// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Covariance settings retained independently of the lifetime of any compiled graph. @internal */
export type LocalKrigingParameterState = {
  values: Float32Array;
  buffers: Set<GPUParameterBuffer<'float32'>>;
};

/**
 * Upload the convenience covariance through the same runtime layout as caller-owned parameters.
 * Each compiled executable owns its parameter buffer. Graph destruction frees it; failed
 * compilation also destroys the executable. Rebuilding requires a new command graph.
 * @internal
 */
export function createLocalKrigingParameterNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  state: LocalKrigingParameterState
): {parameters: GraphDataView<'float32'>; nodes: GPUCommandNode<Parameters>[]} {
  const parameters = createTransientView(
    graph,
    `${id}-covariance`,
    'float32',
    state.values.length,
    Buffer.STORAGE | Buffer.COPY_DST
  );
  return {
    parameters,
    nodes: [
      {
        id: `${id}-upload-covariance`,
        type: 'copy',
        resources: [{buffer: parameters, usage: 'copy-destination'}],
        workload: {
          operation: 'GPULocalKriging',
          commandCount: 1,
          writeByteLength: state.values.byteLength
        },
        compile: ({device}) => {
          const buffer = new GPUParameterBuffer(device, {
            id: `${id}-covariance-source`,
            format: 'float32',
            length: state.values.length,
            values: state.values
          });
          state.buffers.add(buffer);
          return {
            encode: ({commandEncoder, getBuffer}) => {
              commandEncoder.copyBufferToBuffer({
                sourceBuffer: buffer.buffer,
                destinationBuffer: getBuffer(parameters),
                size: parameters.length * Float32Array.BYTES_PER_ELEMENT
              });
            },
            destroy: () => {
              state.buffers.delete(buffer);
              buffer.destroy();
            }
          };
        }
      }
    ]
  };
}
