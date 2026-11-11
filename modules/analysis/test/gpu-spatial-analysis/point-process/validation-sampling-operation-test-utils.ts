// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createTransientView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import type {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

/** Tests deferred contributor construction with packed views and no shader execution. @internal */
export function constructSamplingContributor(
  operation: GPUContributorOperation,
  inputs: Readonly<Record<string, {format: GPUGraphVectorFormat; length: number}>>
): void {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const views = Object.fromEntries(
      Object.entries({...inputs, ...operation.outputs}).map(([port, shape]) => [
        port,
        createTransientView(graph, port, shape.format, shape.length)
      ])
    );
    operation.createContributor(port => views[port] as never);
  } finally {
    operation.destroy();
    device.destroy();
  }
}

/** Wraps borrowed typed rows in the published CPU callback's input layout. @internal */
export function createSamplingCPUInputs(
  inputs: Readonly<
    Record<string, {format: GPUGraphVectorFormat; values: Float32Array | Uint32Array}>
  >
): Record<string, GPUCPUVector> {
  return Object.fromEntries(
    Object.entries(inputs).map(([port, input]) => {
      const length = input.values.length / getGPUVectorFormatInfo(input.format).components;
      return [
        port,
        {
          format: input.format,
          length,
          chunks: [{values: input.values, length}],
          pack: () => input.values
        }
      ];
    })
  );
}
