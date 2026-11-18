// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GPUCommandGraphCompileOptions
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {MeteredCommandGraph} from '../src/session/metered-command-graph';

it.each([
  'layout',
  'source'
] as const)('metered sync/async compile forwards %s metadata options and records scratch bytes', async kernelMetadata => {
  const device = createNullWebGPUDevice();
  const options: GPUCommandGraphCompileOptions = {kernelMetadata};
  const compiled = {stats: {physicalTransientResourceBytes: 4096}} as CompiledGPUCommandGraph;
  const synchronous = vi.spyOn(GPUCommandGraph.prototype, 'compile').mockReturnValue(compiled);
  const asynchronous = vi
    .spyOn(GPUCommandGraph.prototype, 'compileAsync')
    .mockResolvedValue(compiled);
  try {
    const graph = new MeteredCommandGraph(device);
    expect(graph.compile(options)).toBe(compiled);
    expect(synchronous).toHaveBeenCalledWith(options);
    expect(graph.transientByteLength).toBe(4096);
    graph.transientByteLength = 0;
    expect(await graph.compileAsync(options)).toBe(compiled);
    expect(asynchronous).toHaveBeenCalledWith(options);
    expect(graph.transientByteLength).toBe(4096);
  } finally {
    synchronous.mockRestore();
    asynchronous.mockRestore();
    device.destroy();
  }
});
