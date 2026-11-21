// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createTransientView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import type {GPUContributorOperation} from '../../src/analysis-core/operations/gpu-contributor-operation';
import {onTestFinished} from 'vitest';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';

/**
 * Invokes deferred contributor construction against packed null-device graph views. This tests
 * the GPU operation's validation at contributor construction, without executing a shader.
 * Input shapes are keyed by the operation's local port names. The fixture device stays alive
 * for deferred command construction and is released when the current test finishes.
 */
export function constructNetworkOperationContributor(
  operation: GPUContributorOperation,
  inputs: Readonly<Record<string, {format: GPUGraphVectorFormat; length: number}>>
) {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const views = Object.fromEntries(
      Object.entries({...inputs, ...operation.outputs}).map(([port, shape]) => [
        port,
        createTransientView(graph, port, shape.format, shape.length)
      ])
    );
    const contributor = operation.createContributor(port => views[port] as never);
    onTestFinished(() => device.destroy());
    return contributor;
  } catch (error) {
    device.destroy();
    throw error;
  }
}
