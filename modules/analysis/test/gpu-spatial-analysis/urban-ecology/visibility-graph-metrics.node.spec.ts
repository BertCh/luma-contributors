// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSnapshotArena, GPUSnapshotTap} from '../../../src/computation-record';
import {addVisibilityGraphCSRRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/visibility-graph-csr';
import {getCoreDefaultBuffer} from '../../../src/analysis-core/contracts/network-topology';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  VISIBILITY_GRAPH_PIVOT_WGSL,
  createVisibilityGraphPivotsNode
} from '../../../src/gpu-network/visibility-graph/visibility-graph-pivots';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {addVisibilityGraphRecipe} from '../../../src/gpu-network/visibility-graph/visibility-graph';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  computeVisibilityMetricsOracle,
  VISIBILITY_METRIC_GRAPH_FIXTURES
} from './visibility-graph-metrics-oracle';
import {expectUrbanWGSLIdentifiers} from './visibility-graph-wgsl-test-utils';

function getNodes(graph: GPUCommandGraph) {
  return (graph as unknown as {nodes: {id: string}[]}).nodes;
}
function expectUniformStorageAccess(graph: GPUCommandGraph) {
  for (const node of getNodes(graph) as {
    id: string;
    resources?: {buffer?: GraphDataView; usage: string}[];
  }[]) {
    const usages = new Map<object, Set<string>>();
    for (const resource of node.resources ?? []) {
      if (!resource.buffer || !['storage-read', 'storage-read-write'].includes(resource.usage))
        continue;
      const buffer = resource.buffer.buffer;
      const physical = buffer.defaultBuffer ? getCoreDefaultBuffer(buffer.defaultBuffer) : buffer;
      const current = usages.get(physical) ?? new Set<string>();
      current.add(resource.usage);
      usages.set(physical, current);
    }
    for (const current of usages.values())
      expect(current.size, `${node.id}: one storage access mode per physical buffer`).toBe(1);
  }
}
function getProps(graph: GPUCommandGraph, count: number) {
  return {
    mode: 'csr' as const,
    positions: createTransientView(graph, 'points', 'float32x2', count),
    segments: createTransientView(graph, 'walls', 'float32x4', 1),
    maximumDistance: 1.1,
    edgeCapacity: count * 4,
    maximumIterations: 8
  };
}

it('all-source CSR integration exceeds the old 256-source cap without quadratic scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const result = addVisibilityGraphRecipe(graph, {...getProps(graph, 512), metricsMode: 'exact'});
    expect(result.pivotNodes.length).toBe(512);
    expect(result.outputs.metrics.length).toBe(512);
    expect(result.outputs.integration.length).toBe(512);
    expect(result.outputs.extendedIntegration.length).toBe(512);
    expect(result.outputs.normalizedChoice.length).toBe(512);
    expect(result.outputs.closeness.length).toBe(512);
    expect(getNodes(graph).length).toBeLessThanOrEqual(result.estimatedCommandNodeCount);
    const sources = getGPUCommandNodeKernelSources(getNodes(graph));
    const pivotKernel = sources.find(kernel => kernel.id.endsWith('csr-default-pivots'))!;
    // Ensure the composed recipe uses the selector covered by the independent BigInt oracle.
    expect(pivotKernel.source).toContain('selectVisibilityGraphPivot');
    expect(pivotKernel.source).not.toContain('divideUint64ByUint32');
    for (const kernel of sources) {
      expectUrbanWGSLIdentifiers(kernel.source, kernel.id);
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
    }
    expect(sources.find(kernel => kernel.id.endsWith('validate-pivots'))!.source).not.toContain(
      'previous'
    );
    expect(sources.find(kernel => kernel.id.endsWith('csr-approximation'))!.source).toContain(
      '=0u'
    );
  } finally {
    device.destroy();
  }
});

it('100k viewpoints automatically sample but publish integration at every viewpoint', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const result = addVisibilityGraphRecipe(graph, getProps(graph, 100000));
    expect(result.pivotNodes.length).toBe(32);
    expect(result.outputs.integration.length).toBe(100000);
    expect(result.outputs.metrics.length).toBe(100000);
    expect(result.outputs.normalizedChoice.length).toBe(100000);
    expect(getNodes(graph).length).toBeLessThanOrEqual(result.estimatedCommandNodeCount);
    const sources = getGPUCommandNodeKernelSources(getNodes(graph));
    expect(sources.find(kernel => kernel.id.endsWith('csr-approximation'))!.source).toContain(
      '=1u'
    );
  } finally {
    device.destroy();
  }
});

it('exact requests over the budget reject without mutation and permit a sampled retry', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = getProps(graph, 100000);
    const bufferCount = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    expect(() => addVisibilityGraphRecipe(graph, {...props, metricsMode: 'exact'})).toThrow(
      /estimated command node count/
    );
    expect(getNodes(graph)).toHaveLength(0);
    expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(bufferCount);
    expect(() => addVisibilityGraphRecipe(graph, props)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('integration and frontier pipeline counts stay constant as batch and round counts grow', () => {
  const counts: number[][] = [];
  for (const [pivotCount, maximumIterations] of [
    [4, 4],
    [12, 8],
    [24, 12]
  ]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      addVisibilityGraphRecipe(graph, {
        ...getProps(graph, 128),
        pivotCount,
        laneCount: 4,
        maximumIterations
      });
      const nodes = getNodes(graph).filter(node => /csr-integration-batch-/.test(node.id));
      counts.push([
        new Set(getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.source)).size,
        getGPUCommandNodePipelineKeys(nodes).size
      ]);
      for (const kernel of getGPUCommandNodeKernelSources(nodes))
        expectUrbanWGSLIdentifiers(kernel.source, kernel.id);
    } finally {
      device.destroy();
    }
  }
  expect(counts[0][0]).toBeGreaterThan(0);
  expect(counts).toEqual([counts[0], counts[0], counts[0]]);
});

it('aligned metric aliases and derived integration scratch limits fail before declarations', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = getProps(graph, 512);
    const shared = createTransientView(graph, 'shared', 'float32x4', 1024);
    const integration = graph.createDataView(shared.buffer, {
      format: 'float32x4',
      length: 512,
      byteOffset: 16
    });
    expect(() =>
      addVisibilityGraphRecipe(graph, {...props, output: {metrics: shared, integration}})
    ).toThrow(/aligned binding byte ranges must not overlap/);
    expect(getNodes(graph)).toHaveLength(0);
    // Adjacency and one-lane centrality fit; the new per-viewpoint vec4 storage does not.
    Object.assign(device.limits, {maxStorageBufferBindingSize: 8191});
    expect(() =>
      addVisibilityGraphRecipe(graph, {...props, laneCount: 1, pivotCount: 1, edgeCapacity: 1})
    ).toThrow(/integration totals and outputs/);
    expect(getNodes(graph)).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it.each([
  1, 3, 127, 251, 255, 256
])('automatic CSR defaults retain all %i reference sources and enough dependency rounds', count => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const {maximumIterations: _maximumIterations, ...props} = getProps(graph, count);
    const result = addVisibilityGraphRecipe(graph, props);
    expect(result.pivotNodes.length).toBe(count);
    const nodes = getNodes(graph);
    expect(nodes.filter(node => /csr-centrality-batch-0-delta-\d+$/.test(node.id))).toHaveLength(
      count
    );
    expect(nodes.length).toBeLessThanOrEqual(result.estimatedCommandNodeCount);
  } finally {
    device.destroy();
  }
});

it.each([
  'choice',
  'degree',
  'all'
] as const)('accepts validated disjoint %s outputs in one packed buffer', layout => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = getProps(graph, 8);
    const packed = createTransientView(graph, 'packed', 'uint32', 2048);
    let nextByteOffset = 256;
    const view = <Format extends 'float32' | 'float32x4' | 'uint32'>(
      format: Format,
      length: number
    ) => {
      const result = graph.createDataView(packed.buffer, {
        format,
        length,
        byteOffset: nextByteOffset
      });
      nextByteOffset += 256;
      return result;
    };
    const output =
      layout === 'choice'
        ? {
            betweenness: view('float32', 8),
            normalizedChoice: view('float32', 8)
          }
        : layout === 'degree'
          ? {
              degree: view('uint32', 8),
              metrics: view('float32x4', 8)
            }
          : {
              offsets: view('uint32', 9),
              neighbors: view('uint32', 32),
              weights: view('float32', 32),
              degree: view('uint32', 8),
              gridVisits: view('uint32', 8),
              viewpointClassifications: view('uint32', 8),
              betweenness: view('float32', 8),
              closeness: view('float32', 8),
              reachedCounts: view('uint32', 8),
              metrics: view('float32x4', 8),
              integration: view('float32x4', 8),
              extendedIntegration: view('float32x4', 8),
              normalizedChoice: view('float32', 8),
              status: {
                count: view('uint32', 1),
                requiredCount: view('uint32', 1),
                overflow: view('uint32', 1),
                candidateOverflow: view('uint32', 1),
                invalidCount: view('uint32', 1),
                approximation: view('uint32', 1),
                converged: view('uint32', 1),
                iterationLimitReached: view('uint32', 1),
                zeroWeightCycle: view('uint32', 1)
              }
            };
    const result = addVisibilityGraphRecipe(graph, {...props, output});
    expect(getNodes(graph).length).toBeLessThanOrEqual(result.estimatedCommandNodeCount);
    expectUniformStorageAccess(graph);
    for (const kernel of getGPUCommandNodeKernelSources(getNodes(graph)))
      expectUrbanWGSLIdentifiers(kernel.source, kernel.id);
  } finally {
    device.destroy();
  }
});

it.each([
  16, 32
])('rejects overlapping aligned output ranges at offset %i without declaring nodes or buffers', byteOffset => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = getProps(graph, 8);
    const packed = createTransientView(graph, 'packed', 'float32', 256);
    const betweenness = graph.createDataView(packed.buffer, {format: 'float32', length: 8});
    const normalizedChoice = graph.createDataView(packed.buffer, {
      format: 'float32',
      length: 8,
      byteOffset
    });
    const bufferCount = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    expect(() =>
      addVisibilityGraphRecipe(graph, {...props, output: {betweenness, normalizedChoice}})
    ).toThrow(/aligned binding byte ranges must not overlap/);
    expect(getNodes(graph)).toHaveLength(0);
    expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(bufferCount);
    expect(() =>
      addVisibilityGraphRecipe(graph, {
        ...props,
        output: {
          betweenness,
          normalizedChoice: graph.createDataView(packed.buffer, {
            format: 'float32',
            length: 8,
            byteOffset: 256
          })
        }
      })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it.each(
  VISIBILITY_METRIC_GRAPH_FIXTURES
)('CPU BFS/Brandes oracle matches independent $name choice values', fixture => {
  const result = computeVisibilityMetricsOracle(fixture.neighbors);
  result.betweenness.forEach((value, index) =>
    expect(value).toBeCloseTo(fixture.expectedChoice[index], 12)
  );
});

it.each([
  'metrics',
  'normalizedChoice',
  'status'
] as const)('rejects missing STORAGE on %s before graph mutation', field => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = getProps(graph, 8);
    const bad = graph.importBuffer({
      id: 'missing-storage',
      byteLength: 256,
      usage: Buffer.COPY_DST
    });
    const output =
      field === 'metrics'
        ? {metrics: graph.createDataView(bad, {format: 'float32x4', length: 8})}
        : field === 'normalizedChoice'
          ? {normalizedChoice: graph.createDataView(bad, {format: 'float32', length: 8})}
          : {status: {approximation: graph.createDataView(bad, {format: 'uint32', length: 1})}};
    const bufferCount = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    expect(() => addVisibilityGraphRecipe(graph, {...props, output})).toThrow(
      /buffer usage must include Buffer.STORAGE/
    );
    expect(getNodes(graph)).toHaveLength(0);
    expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(bufferCount);
  } finally {
    device.destroy();
  }
});

it('allows a disjoint classification output beside packed position input', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = getProps(graph, 8);
    const packed = createTransientView(graph, 'packed-position-classification', 'uint32', 256);
    const positions = graph.createDataView(packed.buffer, {
      format: 'float32x2',
      length: 8,
      byteOffset: 256
    });
    const viewpointClassifications = graph.createDataView(packed.buffer, {
      format: 'uint32',
      length: 8,
      byteOffset: 320
    });
    expect(() =>
      addVisibilityGraphRecipe(graph, {...props, positions, output: {viewpointClassifications}})
    ).not.toThrow();
    expectUniformStorageAccess(graph);
    for (const kernel of getGPUCommandNodeKernelSources(getNodes(graph)))
      expectUrbanWGSLIdentifiers(kernel.source, kernel.id);
  } finally {
    device.destroy();
  }
});

it('permits consecutive logical output ranges that are bound in separate dispatches', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = getProps(graph, 8);
    const packed = createTransientView(graph, 'consecutive', 'uint32', 128);
    const betweenness = graph.createDataView(packed.buffer, {format: 'float32', length: 8});
    const metrics = graph.createDataView(packed.buffer, {
      format: 'float32x4',
      length: 8,
      byteOffset: 32
    });
    expect(() =>
      addVisibilityGraphRecipe(graph, {...props, output: {betweenness, metrics}})
    ).not.toThrow();
    expectUniformStorageAccess(graph);
  } finally {
    device.destroy();
  }
});

it('rejects logical overlaps between outputs used by separate dispatches without mutation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = getProps(graph, 8);
    const packed = createTransientView(graph, 'overlapping', 'uint32', 128);
    const betweenness = graph.createDataView(packed.buffer, {format: 'float32', length: 8});
    const metrics = graph.createDataView(packed.buffer, {
      format: 'float32x4',
      length: 8,
      byteOffset: 16
    });
    const bufferCount = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    expect(() =>
      addVisibilityGraphRecipe(graph, {...props, output: {betweenness, metrics}})
    ).toThrow(/logical byte ranges must not overlap/);
    expect(getNodes(graph)).toHaveLength(0);
    expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(bufferCount);
  } finally {
    device.destroy();
  }
});

it('adjacency-only CSR ignores network output views it never binds', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = getProps(graph, 8);
    const unused = graph.importBuffer({
      id: 'unused-network-output',
      byteLength: 256,
      usage: Buffer.COPY_DST
    });
    const output = {
      degree: createTransientView(graph, 'degree', 'uint32', 8),
      betweenness: graph.createDataView(unused, {format: 'float32', length: 8})
    };
    expect(() => addVisibilityGraphCSRRecipe(graph, {...props, output})).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('systematic pivot arithmetic matches BigInt beyond uint32 products without general wide division', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const cases = [
      [68640, 68640, 1],
      [68640, 68640, 62573],
      [68640, 68640, 68639],
      [1000000, 68640, 62573],
      [1000000, 68640, 68639],
      [999983, 999979, 999978],
      [1000000, 32, 31],
      [65537, 65535, 65534],
      [1000000, 1, 0],
      [3, 2, 1],
      [1, 1, 0]
    ];
    const queries = createTransientView(graph, 'queries', 'uint32x4', cases.length);
    const output = createTransientView(graph, 'selected', 'uint32', cases.length);
    const node = createWGSLKernelNode(graph, {
      id: 'pivot-arithmetic',
      operation: 'visibility-pivot-test',
      invocationCount: cases.length,
      declarations: VISIBILITY_GRAPH_PIVOT_WGSL,
      bindings: [
        {name: 'queries', view: queries, type: 'u32', access: 'read'},
        {name: 'output', view: output, type: 'u32', access: 'read_write'}
      ],
      body: 'output[outputOffset+index]=selectVisibilityGraphPivot(queries[queriesOffset+4u*index+2u],queries[queriesOffset+4u*index],queries[queriesOffset+4u*index+1u]);'
    });
    const memory = createCPUGraphMemory();
    memory.write(queries, new Uint32Array(cases.flatMap(values => [...values, 0])));
    executeCommandNodesOnCPU([node], memory);
    expect(Array.from(memory.readUint32(output))).toEqual(
      cases.map(([pointCount, pivotCount, index]) =>
        Number((BigInt(index) * BigInt(pointCount)) / BigInt(pivotCount))
      )
    );
    const pivots = createTransientView(graph, 'all-source-pivots', 'uint32', 68640);
    const production = createVisibilityGraphPivotsNode(graph, {
      id: 'production',
      operation: 'visibility-pivots',
      pointCount: 68640,
      pivots
    });
    for (const kernel of getGPUCommandNodeKernelSources([node, production])) {
      expectUrbanWGSLIdentifiers(kernel.source, kernel.id);
      expect(kernel.source).not.toContain('divideUint64ByUint32');
      expect(kernel.source).not.toMatch(/(?:workgroupBarrier|storageBarrier)\s*\(/);
    }
  } finally {
    device.destroy();
  }
});

it('rejects a co-bound caller segment input and writable neighbor output sharing a physical buffer before declaration', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = getProps(graph, 8);
    const packed = createTransientView(graph, 'packed-input-output', 'uint32', 256);
    const segments = graph.createDataView(packed.buffer, {format: 'float32x4', length: 1});
    const neighbors = graph.createDataView(packed.buffer, {
      format: 'uint32',
      length: 32,
      byteOffset: 256
    });
    const bufferCount = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    expect(() =>
      addVisibilityGraphRecipe(graph, {...props, segments, output: {neighbors}})
    ).toThrow(/input segments must not share a physical buffer in one dispatch/);
    expect(getNodes(graph)).toHaveLength(0);
    expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(bufferCount);
  } finally {
    device.destroy();
  }
});

it.each([
  156, 312, 68640
])('composed visibility metrics reuse frontier allocations for %i pivots at the overflow dimensions', pivotCount => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const createBuffer = device.createBuffer.bind(device);
  // Exercise the real public compiler's schedule and allocator without allocating large CPU arrays
  // or compiling GPU shaders. NullBuffer otherwise reserves every requested byte on this machine.
  device.createBuffer = props => {
    const buffer = createBuffer({...props, byteLength: 4});
    buffer.byteLength = props?.byteLength ?? 4;
    return buffer;
  };
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    const result = addVisibilityGraphRecipe(graph, {
      ...getProps(graph, 68640),
      edgeCapacity: 1,
      pivotCount,
      laneCount: 156,
      maximumIterations: 1,
      metricsMode: pivotCount === 68640 ? 'exact' : 'auto'
    });
    const capturedPivots = createTransientView(graph, 'captured-pivots', 'uint32', pivotCount);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'capture-pivots',
        operation: 'visibility-pivot-test',
        invocationCount: pivotCount,
        bindings: [
          {name: 'pivots', view: result.pivotNodes, type: 'u32', access: 'read'},
          {name: 'captured', view: capturedPivots, type: 'u32', access: 'read_write'}
        ],
        body: 'captured[capturedOffset+index]=pivots[pivotsOffset+index];'
      })
    );
    const nodes = getNodes(graph) as {
      id: string;
      dependsOn?: readonly string[];
      compile: () => {encode: () => void};
      resources?: {buffer?: GraphDataView; usage: string}[];
    }[];
    const metricNodes = nodes.filter(node => /-csr-(?:centrality|integration)-/.test(node.id));
    for (let index = 1; index < metricNodes.length; index++)
      expect(metricNodes[index].dependsOn, metricNodes[index].id).toContain(
        metricNodes[index - 1].id
      );
    // Default pivots are written exactly once. No downstream batch may overwrite them.
    expect(
      nodes
        .filter(node =>
          node.resources?.some(
            resource =>
              resource.buffer?.buffer === result.pivotNodes.buffer &&
              resource.usage !== 'storage-read'
          )
        )
        .map(node => node.id)
    ).toEqual(['visibility-graph-csr-default-pivots']);
    for (const node of nodes) node.compile = () => ({encode: () => {}});
    compiled = graph.compile();
    // Before ordering the batches, the 440-batch case planned 35,685,939,844 physical bytes.
    // One fixed lane-expanded working set fits below 768 MiB regardless of the pivot batch count.
    expect(compiled.stats.physicalTransientBytes).toBeLessThan(768 * 1024 * 1024);
    expect(
      compiled.stats.nodeOrder.filter(id => /-csr-(?:centrality|integration)-/.test(id))
    ).toEqual(metricNodes.map(node => node.id));
  } finally {
    compiled?.destroy();
    device.destroy();
  }
});

it('new recipe-owned visibility metrics support the lens snapshot copy contract', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let arena: GPUSnapshotArena | undefined;
  try {
    const result = addVisibilityGraphRecipe(graph, {...getProps(graph, 3), laneCount: 3});
    const views = {
      metrics: result.outputs.metrics,
      integration: result.outputs.integration,
      extendedIntegration: result.outputs.extendedIntegration,
      normalizedChoice: result.outputs.normalizedChoice
    };
    arena = new GPUSnapshotArena(device, {
      slots: Object.entries(views).map(([name, view]) => ({
        name,
        byteLength: view.length * view.rowByteLength,
        elementType: 'f32'
      })),
      capacity: 1
    });
    expect(() =>
      graph.add(new GPUSnapshotTap({id: 'metric-snapshot', arena: arena!, views}))
    ).not.toThrow();
    for (const view of Object.values(views))
      expect(view.buffer.usage & Buffer.COPY_SRC).not.toBe(0);
    // The recipe's private scratch does not acquire observation-only copy usage.
    const scratch = (graph as unknown as {buffers: Map<string, {usage: number}>}).buffers.get(
      'visibility-graph-csr-invalid-pivot-rows'
    )!;
    expect(scratch.usage & Buffer.COPY_SRC).toBe(0);
  } finally {
    arena?.destroy();
    device.destroy();
  }
});

it('metric snapshot capability preserves caller-owned STORAGE-only packed views', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const packed = createTransientView(graph, 'provided-metrics', 'uint32', 512);
    const output = {
      metrics: graph.createDataView(packed.buffer, {
        format: 'float32x4',
        length: 3,
        byteOffset: 256
      }),
      integration: graph.createDataView(packed.buffer, {
        format: 'float32x4',
        length: 3,
        byteOffset: 512
      }),
      extendedIntegration: graph.createDataView(packed.buffer, {
        format: 'float32x4',
        length: 3,
        byteOffset: 768
      }),
      normalizedChoice: graph.createDataView(packed.buffer, {
        format: 'float32',
        length: 3,
        byteOffset: 1024
      })
    };
    const result = addVisibilityGraphRecipe(graph, {...getProps(graph, 3), laneCount: 3, output});
    for (const name of Object.keys(output) as (keyof typeof output)[])
      expect(result.outputs[name]).toBe(output[name]);
    expect(packed.buffer.usage).toBe(Buffer.STORAGE);
    expectUniformStorageAccess(graph);
  } finally {
    device.destroy();
  }
});
