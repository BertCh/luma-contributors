// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import type {GPUNetworkVisibilityGraphCSRProps} from '@luma-contributors/analysis/gpu-network';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addVisibilityGraphRecipe} from '@luma-contributors/analysis/gpu-network';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  expectUrbanWGSLIdentifiers,
  expectUrbanWGSLNumericValues
} from './visibility-graph-wgsl-test-utils';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {publishVisibilityGraphOutput} from './visibility-graph-test-utils';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE} from '../../../src/gpu-spatial-analysis/urban-morphology/visibility-graph-csr';
import {addVisibilityGraphRecipe as addSpatialVisibilityGraphRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';

function getNodes(graph: GPUCommandGraph) {
  return (graph as unknown as {nodes: {id: string}[]}).nodes;
}
it('10k viewpoints compose a bounded CSR graph and the existing network centrality', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const result = addVisibilityGraphRecipe(graph, {
      mode: 'csr',
      positions: createTransientView(graph, 'positions', 'float32x2', 10000),
      segments: createTransientView(graph, 'segments', 'float32x4', 4),
      maximumDistance: 1.1,
      edgeCapacity: 50000,
      pivotCount: 4,
      laneCount: 4,
      maximumIterations: 8
    });
    expect(result.outputs.offsets.length).toBe(10001);
    expect(result.outputs.neighbors.length).toBe(50000);
    expect(result.outputs.closeness.length).toBe(4);
    const nodes = getNodes(graph);
    expect(nodes.some(node => node.id.includes('csr-centrality'))).toBe(true);
    expect(nodes.length).toBeLessThanOrEqual(result.estimatedCommandNodeCount);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    for (const kernel of kernels) {
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      expectUrbanWGSLIdentifiers(kernel.source, kernel.id);
    }
    expect(kernels.filter(kernel => kernel.id === 'visibility-graph-csr-count')).toHaveLength(1);
    expect(kernels.filter(kernel => kernel.id === 'visibility-graph-csr-scatter')).toHaveLength(1);
  } finally {
    device.destroy();
  }
});
it('CSR caller aliases and aligned size violations fail before nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const shared = createTransientView(graph, 'shared', 'uint32', 10001);
    const props = {
      mode: 'csr' as const,
      positions: createTransientView(graph, 'positions', 'float32x2', 10000),
      segments: createTransientView(graph, 'segments', 'float32x4', 4),
      maximumDistance: 1,
      edgeCapacity: 50000
    };
    expect(() =>
      addVisibilityGraphRecipe(graph, {
        ...props,
        output: {
          offsets: shared,
          degree: graph.createDataView(shared.buffer, {format: 'uint32', length: 10000})
        }
      })
    ).toThrow(/aligned binding byte ranges must not overlap/);
    expect(getNodes(graph)).toHaveLength(0);
    expect(() => addVisibilityGraphRecipe(graph, {...props, edgeCapacity: 16777216})).toThrow(
      /centrality adjacency needs/
    );
    expect(getNodes(graph)).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('query shader and pipeline counts stay constant as unequal chunk count grows', () => {
  const counts: number[][] = [];
  for (const queryChunkSize of [1024, 256, 128]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      addVisibilityGraphRecipe(graph, {
        mode: 'csr',
        positions: createTransientView(graph, 'points', 'float32x2', 769),
        segments: createTransientView(graph, 'walls', 'float32x4', 4),
        maximumDistance: 1,
        edgeCapacity: 4000,
        pivotCount: 4,
        maximumIterations: 8,
        queryChunkSize
      });
      const nodes = getNodes(graph).filter(node => /csr-(count|scatter)-chunk-/.test(node.id));
      const kernels = getGPUCommandNodeKernelSources(nodes);
      counts.push([
        new Set(kernels.map(kernel => kernel.source)).size,
        getGPUCommandNodePipelineKeys(nodes).size
      ]);
      expect(nodes).toHaveLength(2 * Math.ceil(769 / queryChunkSize));
      expect(kernels.every(kernel => Object.keys(kernel.constants).length === 0)).toBe(true);
      if (queryChunkSize < 769) {
        const last = nodes.filter(node => node.id.includes('csr-count')).at(-1) as unknown as {
          workload: {maximumInvocationCount: number; maximumWorkgroupCount: number};
        };
        expect(last.workload.maximumInvocationCount).toBe(1);
        expect(last.workload.maximumWorkgroupCount).toBe(1);
      }
    } finally {
      device.destroy();
    }
  }
  expect(counts).toEqual([
    [2, 2],
    [2, 2],
    [2, 2]
  ]);
});

it('composed centrality reuses its round pipelines as sampled pivot batches grow', () => {
  const pipelineCounts: number[] = [];
  for (const pivotCount of [4, 12]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      addVisibilityGraphRecipe(graph, {
        mode: 'csr',
        positions: createTransientView(graph, 'points', 'float32x2', 128),
        segments: createTransientView(graph, 'walls', 'float32x4', 4),
        maximumDistance: 1,
        edgeCapacity: 512,
        pivotCount,
        laneCount: 4,
        maximumIterations: 8
      });
      const nodes = getNodes(graph).filter(node =>
        /csr-centrality-batch-\d+-(sigma|delta)-\d+$/.test(node.id)
      );
      expect(nodes).toHaveLength((pivotCount / 4) * 16);
      pipelineCounts.push(getGPUCommandNodePipelineKeys(nodes).size);
    } finally {
      device.destroy();
    }
  }
  expect(pipelineCounts[0]).toBeGreaterThan(0);
  expect(pipelineCounts[1]).toBe(pipelineCounts[0]);
});

it.each([
  'positions',
  'segments',
  'parameters',
  'pivotNodes',
  'output.neighbors',
  'output.gridVisits'
] as const)('rejects missing STORAGE usage for %s without graph mutation, then permits retry', field => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const validProps = {
      mode: 'csr' as const,
      positions: createTransientView(graph, 'points', 'float32x2', 4),
      segments: createTransientView(graph, 'walls', 'float32x4', 1),
      maximumDistance: 1,
      edgeCapacity: 8,
      pivotCount: 1,
      maximumIterations: 8
    };
    const bad = graph.importBuffer({id: 'bad-usage', usage: Buffer.COPY_DST, byteLength: 64});
    const props: GPUNetworkVisibilityGraphCSRProps = {...validProps};
    switch (field) {
      case 'positions':
        props.positions = graph.createDataView(bad, {format: 'float32x2', length: 4});
        break;
      case 'segments':
        props.segments = graph.createDataView(bad, {format: 'float32x4', length: 1});
        break;
      case 'parameters':
        props.parameters = graph.createDataView(bad, {format: 'float32', length: 2});
        break;
      case 'pivotNodes':
        props.pivotNodes = graph.createDataView(bad, {format: 'uint32', length: 1});
        break;
      case 'output.neighbors':
        props.output = {neighbors: graph.createDataView(bad, {format: 'uint32', length: 8})};
        break;
      case 'output.gridVisits':
        props.output = {gridVisits: graph.createDataView(bad, {format: 'uint32', length: 4})};
        break;
    }
    const resourceCount = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    expect(() => addVisibilityGraphRecipe(graph, props)).toThrow(
      `visibility-graph ${field} chunk 0 buffer usage must include Buffer.STORAGE`
    );
    expect(getNodes(graph)).toHaveLength(0);
    expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(resourceCount);
    expect(() => addVisibilityGraphRecipe(graph, validProps)).not.toThrow();
    expect(getNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it.each([
  1e20, 1e21, 3.4028234663852886e38
])('every composed CSR shader has legal numeric values at radius %s', maximumDistance => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    addVisibilityGraphRecipe(graph, {
      mode: 'csr',
      positions: createTransientView(graph, 'points', 'float32x2', 769),
      segments: createTransientView(graph, 'walls', 'float32x4', 4),
      maximumDistance,
      tolerance: 1 / 64,
      edgeCapacity: 4000,
      pivotCount: 4,
      maximumIterations: 8,
      queryChunkSize: 128
    });
    const kernels = getGPUCommandNodeKernelSources(getNodes(graph));
    for (const kernel of kernels) expectUrbanWGSLIdentifiers(kernel.source, kernel.id);
    for (const stage of ['prepare', 'weights', 'mask-metrics']) {
      const kernel = kernels.find(value => value.id === `visibility-graph-csr-${stage}`)!;
      expect(kernel.source, stage).toContain(GET_NAN_WGSL);
      expect(kernel.source, stage).toContain('getNaN()');
      expect(kernel.source, stage).not.toContain('bitcast<f32>(0x7fc00000u)');
    }
  } finally {
    device.destroy();
  }
});

it.each([
  'bitcast<f32>(0x7fc00000u)',
  'bitcast<f32>(0x7f800000u)',
  'const bits:u32=0xff800000u; bitcast<f32>(bits)',
  'let value:f32=1e100;',
  'let value:f32=0x1p128;',
  'let value:f32=NaN;',
  'let value:f32=Infinity;',
  'let value=1.0/0.0;'
])('the numeric audit rejects shader-creation failures: %s', source => {
  expect(() => expectUrbanWGSLNumericValues(source, 'negative control')).toThrow();
});

it('the numeric audit accepts the shared runtime NaN helper and finite float boundaries', () => {
  expectUrbanWGSLNumericValues(GET_NAN_WGSL, 'runtime NaN');
  expectUrbanWGSLNumericValues(
    'let maximum=3.4028234663852886e38f; let hexadecimal=0x1.fffffep127; let minimum=1.401298464324817e-45f;',
    'finite f32 boundaries'
  );
});

it.each([
  'reference',
  'csr'
] as const)('derived ray tolerance is validated before graph mutation: %s', mode => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = {
      positions: createTransientView(graph, 'points', 'float32x2', 4),
      segments: createTransientView(graph, 'walls', 'float32x4', 4),
      maximumDistance: 1,
      edgeCapacity: 12,
      tolerance: VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE * 2
    };
    const bufferCount = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    expect(() =>
      mode === 'csr'
        ? addVisibilityGraphRecipe(graph, {...props, mode: 'csr'})
        : addVisibilityGraphRecipe(graph, {...props, mode: 'reference', obstacleIndex: 'bvh'})
    ).toThrow(/visibility-graph tolerance.*VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE/);
    expect(getNodes(graph)).toHaveLength(0);
    expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(bufferCount);
  } finally {
    device.destroy();
  }
});

it('the exhaustive obstacle reference keeps the full finite float32 tolerance range', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    addVisibilityGraphRecipe(graph, {
      mode: 'reference',
      positions: createTransientView(graph, 'points', 'float32x2', 4),
      segments: createTransientView(graph, 'walls', 'float32x4', 4),
      maximumDistance: 1e20,
      tolerance: 3.4028234663852886e38,
      edgeCapacity: 12,
      obstacleIndex: 'brute-force'
    });
    for (const kernel of getGPUCommandNodeKernelSources(getNodes(graph)))
      expectUrbanWGSLIdentifiers(kernel.source, kernel.id);
  } finally {
    device.destroy();
  }
});

it('the public spatial CSR recipe builds only adjacency and degree at ten thousand points', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const result = addSpatialVisibilityGraphRecipe(graph, {
      mode: 'csr',
      positions: createTransientView(graph, 'points', 'float32x2', 10000),
      segments: createTransientView(graph, 'walls', 'float32x4', 4),
      maximumDistance: 1.1,
      edgeCapacity: 50000
    });
    expect(Object.keys(result.outputs).sort()).toEqual([
      'degree',
      'gridVisits',
      'neighbors',
      'offsets',
      'status',
      'viewpointClassifications',
      'weights'
    ]);
    expect(Object.keys(result.status).sort()).toEqual([
      'candidateOverflow',
      'count',
      'invalidCount',
      'overflow',
      'requiredCount'
    ]);
    expect(result.outputs.offsets.length).toBe(10001);
    expect(result.outputs.degree.length).toBe(10000);
    expect(getNodes(graph).some(node => /centrality|pivot|mask-metrics/.test(node.id))).toBe(false);
    expect(getNodes(graph).length).toBeLessThanOrEqual(result.estimatedCommandNodeCount);
    for (const kernel of getGPUCommandNodeKernelSources(getNodes(graph)))
      expectUrbanWGSLIdentifiers(kernel.source, kernel.id);
  } finally {
    device.destroy();
  }
});

it('network-only failures leave adjacency undeclared, including invalid metric layouts and overlapping bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = {
      mode: 'csr' as const,
      positions: createTransientView(graph, 'points', 'float32x2', 4),
      segments: createTransientView(graph, 'walls', 'float32x4', 1),
      maximumDistance: 1,
      edgeCapacity: 8,
      pivotCount: 2,
      laneCount: 2,
      maximumIterations: 8
    };
    const metricBuffer = graph.createTransientBuffer({
      id: 'metrics',
      byteLength: 1024,
      usage: Buffer.STORAGE
    });
    const betweenness = graph.createDataView(metricBuffer, {format: 'float32', length: 4});
    const closeness = graph.createDataView(metricBuffer, {
      format: 'float32',
      byteOffset: 16,
      length: 2
    });
    const wrong = createTransientView(graph, 'wrong-metric', 'float32', 3);
    const bufferCount = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    for (const invalid of [
      {...props, output: {betweenness: wrong}},
      {...props, output: {betweenness, closeness}},
      {...props, laneCount: 3},
      {...props, maximumIterations: 1025}
    ]) {
      expect(() => addVisibilityGraphRecipe(graph, invalid)).toThrow();
      expect(getNodes(graph)).toHaveLength(0);
      expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(bufferCount);
    }
    expect(() => addVisibilityGraphRecipe(graph, props)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it.each([
  1, 255, 256
])('the headless publisher keeps all dense and CSR output IDs separate for %i viewpoints', count => {
  const device = createNullWebGPUDevice();
  const fixture = new RecipeTestFixture(device, `visibility-readback-${count}`);
  try {
    const graph = fixture.graph;
    const positions = createTransientView(graph, 'points', 'float32x2', count);
    const segments = createTransientView(graph, 'segments', 'float32x4', 1);
    const dense = addVisibilityGraphRecipe(graph, {
      id: 'dense',
      mode: 'reference',
      positions,
      segments,
      maximumDistance: 1,
      edgeCapacity: 8,
      searchStructure: 'brute-force',
      obstacleIndex: 'bvh'
    });
    const csr = addVisibilityGraphRecipe(graph, {
      mode: 'csr',
      positions,
      segments,
      maximumDistance: 1,
      edgeCapacity: 16,
      pivotCount: Math.min(count, 4),
      maximumIterations: 8,
      queryChunkSize: 128
    });
    expect(dense.outputs.adjacency.buffer.id).toBe('dense-adjacency');
    const captured = Object.entries({...dense.outputs, status: dense.status}).map(
      ([name, source]) => publishVisibilityGraphOutput(fixture, `dense-${name}`, source)
    );
    const {status, ...fields} = csr.outputs;
    captured.push(
      ...Object.entries({...fields, ...status}).map(([name, source]) =>
        publishVisibilityGraphOutput(fixture, `csr-read-${name}`, source)
      )
    );
    expect(captured[0].source).toBe(dense.outputs.adjacency);
    expect(captured[0].output.view.buffer.id).toBe('readback-dense-adjacency');
    expect(new Set(captured.map(value => value.output.view.buffer.id)).size).toBe(captured.length);
    expect(captured.every(value => value.output.view.buffer !== value.source.buffer)).toBe(true);
    for (const kernel of getGPUCommandNodeKernelSources(getNodes(graph)))
      expectUrbanWGSLIdentifiers(kernel.source, kernel.id);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('network visibility checks centrality allocations against maxBufferSize before declaring adjacency', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    Object.assign(device.limits, {
      maxStorageBufferBindingSize: 1073741824,
      maxBufferSize: 268435456
    });
    const props = {
      mode: 'csr' as const,
      positions: createTransientView(graph, 'points', 'float32x2', 1000000),
      segments: createTransientView(graph, 'walls', 'float32x4', 4),
      maximumDistance: 1,
      edgeCapacity: 16777216,
      pivotCount: 4,
      laneCount: 4,
      maximumIterations: 8
    };
    const bufferCount = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    expect(() => addVisibilityGraphRecipe(graph, props)).toThrow(
      /visibility-graph centrality adjacency allocation byteLength \(maxBufferSize\).*268435456/
    );
    expect(getNodes(graph)).toHaveLength(0);
    expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(bufferCount);
    // Spatial adjacency uses separate buffers that each fit this device; only the packed centrality allocation is refused.
    const adjacency = addSpatialVisibilityGraphRecipe(graph, props);
    expect(adjacency.outputs.neighbors.length).toBe(16777216);
    expect(getNodes(graph).length).toBeLessThanOrEqual(adjacency.estimatedCommandNodeCount);
  } finally {
    device.destroy();
  }
});
