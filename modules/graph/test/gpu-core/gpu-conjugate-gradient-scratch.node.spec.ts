// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {Buffer} from '@luma.gl/core';
import {GPUData} from '@luma-contributors/graph/gpu-data';
import {GPUGraphCompiler, createGPUConjugateGradientGraph} from '@luma-contributors/graph/gpu-core';
import {NullDevice} from '@luma.gl/core/test-utils';
import {afterEach, expect, test, vi} from 'vitest';

afterEach(() => vi.restoreAllMocks());

function compileSolver(size: number, maximumIterations: number, recording = false) {
  const device = new NullDevice({_cacheShaders: false, _cachePipelines: false});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBuffersPerShaderStage: 8,
      maxStorageBufferBindingSize: 2 ** 30
    }
  });
  const sources: string[] = [];
  vi.spyOn(device, 'createComputePipeline').mockImplementation(props => {
    sources.push(props!.shader.source);
    return {id: props?.id, hash: '', destroy() {}} as never;
  });
  const submit = vi.spyOn(device, 'submit');
  const solver = createGPUConjugateGradientGraph({
    id: 'solver',
    size,
    nonZeros: size,
    maxIterations: maximumIterations,
    toleranceSquared: 1e-8
  });
  const inputs = [
    solver.rowOffsets,
    solver.columnIndices,
    solver.values,
    solver.rhs,
    solver.solution
  ];
  const data = inputs.map(
    input =>
      new GPUData({
        buffer: device.createBuffer({byteLength: input.length * 4, usage: Buffer.STORAGE}),
        ownsBuffer: true,
        format: input.format,
        length: input.length
      })
  );
  try {
    const compilation = new GPUGraphCompiler(device).compile(
      solver.program,
      {
        vectors: Object.fromEntries(inputs.map((input, index) => [input.id, data[index]]))
      },
      recording ? {recording: {sourceText: true}} : {}
    );
    const compiled = compilation.graph.compile();
    try {
      const description = compiled.describe();
      expect(submit).not.toHaveBeenCalled();
      return {
        sources,
        order: compiled.stats.nodeOrder,
        edges: description.graph.edges,
        resources: description.graph.resources ?? [],
        nodes: description.graph.nodes.map(node => ({
          id: node.id,
          kind: node.kind,
          reads: node.reads,
          writes: node.writes,
          workload: node.workload,
          condition: node.condition,
          workgroups: node.dispatch?.workgroups
        }))
      };
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const input of data) input.destroy();
    device.destroy();
  }
}

// Fingerprints cover compiled WGSL: reduction kernels take problem sizes as pipeline constants, so
// refreeze them when kernel text changes. Nodes, resources, edges and workloads must stay unchanged.
test.each([
  [1, 'c18063bfe59f119bd49f8de0f9f7091abadb786e665c8b0dab7bc926a8749682'],
  [256, '3026599dc702d3ce0bd9dc9d513af03ac6fa9d94ffdaaca2c1467330632f331e'],
  [257, 'db57ac52f06ff63794b0e03f88d8f01b314022fcad2519f77893397cf57494af'],
  [65537, '5bfdd04fc1e33e866bd289a66f40e4509c8620d50d2b00b83ab0cc5d118cb312']
] as const)('CG size %s preserves the previously working production graph', (size, fingerprint) => {
  // Freeze pre-fix nodes, resource layouts, dependencies, workloads and compiled WGSL.
  const result = compileSolver(size, size <= 256 ? 3 : 1);
  expect(createHash('sha256').update(JSON.stringify(result)).digest('hex')).toBe(fingerprint);
});

test.each([
  257, 65537
])('CG size %s allocates distinct reduction scratch for every iteration', size => {
  const result = compileSolver(size, 3);
  const scratch = result.resources.filter(resource => resource.id.includes('-partials-'));
  const levels = size === 257 ? 1 : 2;
  expect(scratch).toHaveLength(7 * levels);
  expect(new Set(scratch.map(resource => resource.id)).size).toBe(scratch.length);
  for (const resource of scratch) {
    expect(result.nodes.filter(node => node.writes?.includes(resource.id))).toHaveLength(1);
    expect(result.nodes.filter(node => node.reads?.includes(resource.id))).toHaveLength(1);
  }
  expect(compileSolver(size, 3, true)).toEqual(result);
  expect(compileSolver(size, 3)).toEqual(result);
});
