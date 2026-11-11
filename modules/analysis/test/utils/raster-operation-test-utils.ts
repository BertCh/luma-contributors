// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import type {GPUContributorOperation} from '../../src/analysis-core/operations/gpu-contributor-operation';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';

/** Constructs the underlying contributor with real packed views, without executing kernels. */
export function constructRasterOperationContributor(
  operation: GPUContributorOperation,
  inputs: Record<string, {format: GPUGraphVectorFormat; length: number}>
): void {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const views = Object.fromEntries(
      Object.entries({...inputs, ...operation.outputs}).map(([port, descriptor]) => [
        port,
        createTransientView(graph, port, descriptor.format, descriptor.length)
      ])
    );
    operation.createContributor(port => views[port] as never);
  } finally {
    device.destroy();
  }
}
