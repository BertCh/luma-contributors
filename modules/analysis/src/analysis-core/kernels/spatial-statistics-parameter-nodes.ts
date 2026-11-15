// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createGPUCopyCommandNode,
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from './gpu-contributor-utils';

/**
 * Uses a borrowed runtime parameter view, or uploads validated numeric defaults with the same
 * GPUParameterBuffer layout. The compiled copy node owns the defaults and releases them on
 * destruction. Copying on every encoding respects transient aliasing and command ordering.
 * @internal
 */
export function createSpatialStatisticsParameterNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  values: GraphDataView<'float32'> | Float32Array
): {parameters: GraphDataView<'float32'>; nodes: GPUCommandNode<Parameters>[]};
export function createSpatialStatisticsParameterNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  values: GraphDataView<'uint32'> | Uint32Array
): {parameters: GraphDataView<'uint32'>; nodes: GPUCommandNode<Parameters>[]};
export function createSpatialStatisticsParameterNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  values: GraphDataView<'float32' | 'uint32'> | Float32Array | Uint32Array
): {parameters: GraphDataView<'float32' | 'uint32'>; nodes: GPUCommandNode<Parameters>[]} {
  if (!(values instanceof Float32Array) && !(values instanceof Uint32Array))
    return {parameters: values, nodes: []};
  const format = values instanceof Uint32Array ? 'uint32' : 'float32';
  const parameters = createTransientView(
    graph,
    id,
    format,
    values.length,
    Buffer.STORAGE | Buffer.COPY_DST
  );
  return {
    parameters,
    nodes: [
      createGPUCopyCommandNode<Parameters>({
        id: `${id}-upload`,
        resources: [{buffer: parameters, usage: 'copy-destination'}],
        compile: ({device}) => {
          const defaults = new GPUParameterBuffer(device, {
            id: `${id}-defaults`,
            format,
            length: values.length,
            values
          });
          return {
            encode: ({commandEncoder, getBuffer}) => {
              commandEncoder.copyBufferToBuffer({
                sourceBuffer: defaults.buffer,
                sourceOffset: 0,
                destinationBuffer: getBuffer(parameters),
                destinationOffset: parameters.byteOffset,
                size: values.byteLength
              });
            },
            destroy: () => defaults.destroy()
          };
        }
      })
    ]
  };
}
