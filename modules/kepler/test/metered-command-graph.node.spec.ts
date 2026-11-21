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

it.each([
  'sync',
  'async'
] as const)('metered %s compilation preserves the published compile-once contract', async mode => {
  const device = createNullWebGPUDevice();
  const graph = new MeteredCommandGraph(device);
  const compiled =
    mode === 'sync'
      ? graph.compile({kernelMetadata: 'source'})
      : await graph.compileAsync({kernelMetadata: 'source'});
  try {
    expect(graph.transientByteLength).toBe(0);
    expect(() => graph.compile({kernelMetadata: 'layout'})).toThrow(/compiled/);
    await expect(graph.compileAsync({kernelMetadata: 'layout'})).rejects.toThrow(/compiled/);
    expect(graph.transientByteLength).toBe(0);
  } finally {
    compiled.destroy();
    device.destroy();
  }
});

for (const mode of ['sync', 'async'] as const) {
  it.each([
    undefined,
    'layout',
    'source'
  ] as const)(`metered ${mode} compile exposes the requested public metadata (%s)`, async kernelMetadata => {
    const device = createNullWebGPUDevice();
    const graph = new MeteredCommandGraph(device);
    const source = '@compute @workgroup_size(1) fn main() {}';
    graph.addComputePass({
      id: 'metadata-probe',
      resources: [],
      kernel: {entryPoint: 'main', workgroupSize: [1, 1, 1], bindings: [], source},
      // This checks compilation metadata; it creates no backend pipeline or GPU commands.
      compile: () => ({encode() {}})
    });
    const options = {kernelMetadata};
    const compiled = mode === 'sync' ? graph.compile(options) : await graph.compileAsync(options);
    try {
      const kernel = compiled.preflight.nodes[0].kernel;
      if (kernelMetadata === undefined) {
        expect(kernel).toBeUndefined();
      } else {
        expect(kernel?.entryPoint).toBe('main');
        expect(kernel?.workgroupSize).toEqual([1, 1, 1]);
        expect(kernel?.bindings).toEqual([]);
        expect(kernel?.source).toBe(kernelMetadata === 'source' ? source : undefined);
      }
    } finally {
      compiled.destroy();
      device.destroy();
    }
  });
}
