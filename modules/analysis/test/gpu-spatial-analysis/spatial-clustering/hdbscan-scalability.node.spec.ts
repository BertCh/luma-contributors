// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUHierarchicalDensityClustering,
  GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_ROW_COUNT,
  GPU_HIERARCHICAL_DENSITY_CLUSTERING_REFERENCE_MAXIMUM_ROW_COUNT,
  type GPUHierarchicalDensityClusteringProps
} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  createVectorView
} from '../../utils/gpu-contributor-test-utils';
import {getHDBSCANDendrogramNodes} from '../../../src/gpu-spatial-analysis/spatial-clustering/hdbscan-tree';
import {
  findReservedIdentifiers,
  stripWGSLComments,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function getReservedWGSLIdentifiers(source: string) {
  return findReservedIdentifiers(stripWGSLComments(source));
}

function createProps(graph: GPUCommandGraph, rows: number): GPUHierarchicalDensityClusteringProps {
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', rows),
    minimumClusterSize: 2,
    minimumSamples: 32,
    output: {
      labels: createTransientView(graph, 'labels', 'uint32', rows),
      coreDistances: createTransientView(graph, 'core', 'float32', rows),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        requiredCount: createTransientView(graph, 'required-count', 'uint32', 1)
      }
    }
  };
}

it('declares four million rows at 32 samples with every actual binding within 128 MiB', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const rows of [1, 257, 1_048_577, GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_ROW_COUNT]) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph, rows);
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const clustering = new GPUHierarchicalDensityClustering(props);
      const nodes = clustering.getCommandNodes(graph);
      expect(nodes.length, `rows=${rows}`).toBeLessThanOrEqual(
        clustering.estimatedCommandNodeCount
      );
      const largestBuffer = Math.max(
        ...allocate.mock.calls.map(([descriptor]) => descriptor.byteLength)
      );
      expect(largestBuffer).toBeLessThanOrEqual(device.limits.maxStorageBufferBindingSize);
      expect(
        allocate.mock.calls.some(([descriptor]) =>
          /neighbor-(distances|weights|ids)/.test(descriptor.id)
        )
      ).toBe(false);
      for (const node of nodes) {
        for (const resource of node.resources ?? []) {
          if (!('buffer' in resource) || !resource.usage.startsWith('storage')) continue;
          const size =
            'buffer' in resource.buffer
              ? getViewBindingRange(resource.buffer).size
              : resource.buffer.byteLength;
          expect(size, node.id).toBeLessThanOrEqual(device.limits.maxStorageBufferBindingSize);
        }
      }
    }
  } finally {
    device.destroy();
  }
});

it('keeps the packed neighbour reference opt-in and its original named row cap', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(
      graph,
      GPU_HIERARCHICAL_DENSITY_CLUSTERING_REFERENCE_MAXIMUM_ROW_COUNT + 1
    );
    expect(
      () => new GPUHierarchicalDensityClustering({...props, coreDistanceSearch: 'neighbor-search'})
    ).toThrow(/positions rows for coreDistanceSearch neighbor-search.*1048576/);
    const referenceGraph = new GPUCommandGraph(device);
    const nodes = new GPUHierarchicalDensityClustering({
      ...createProps(referenceGraph, 64),
      coreDistanceSearch: 'neighbor-search'
    }).getCommandNodes(referenceGraph);
    expect(nodes.some(node => node.id.includes('-neighbors-'))).toBe(true);
    expect(nodes.some(node => node.id.endsWith('-bvh-core'))).toBe(false);
  } finally {
    device.destroy();
  }
});

it('rejects every overlapping pair of output fields at construction, while allowing adjacent ranges', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, 64);
    const shared = createTransientView(graph, 'shared', 'uint32', 129).buffer;
    const labels = graph.createDataView(shared, {format: 'uint32', length: 64});
    const coreDistances = graph.createDataView(shared, {
      format: 'float32',
      length: 64,
      byteOffset: 256
    });
    const count = graph.createDataView(shared, {format: 'uint32', length: 1, byteOffset: 512});
    const output = {labels, coreDistances, status: {count}};
    expect(() => new GPUHierarchicalDensityClustering({...props, output})).not.toThrow();
    const overlap = (format: 'uint32' | 'float32', byteOffset: number) =>
      graph.createDataView(shared, {format, length: format === 'float32' ? 64 : 1, byteOffset});
    expect(
      () =>
        new GPUHierarchicalDensityClustering({
          ...props,
          output: {
            ...output,
            coreDistances: overlap('float32', 252) as GraphDataView<'float32'>
          }
        })
    ).toThrow(/writable byte ranges/);
    for (const byteOffset of [252, 508]) {
      expect(
        () =>
          new GPUHierarchicalDensityClustering({
            ...props,
            output: {
              ...output,
              status: {count: overlap('uint32', byteOffset) as GraphDataView<'uint32'>}
            }
          })
      ).toThrow(/writable byte ranges/);
    }
  } finally {
    device.destroy();
  }
});

it('preflights aligned caller bindings and derived planes without changing the graph on failure', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const testCase of [
      'aligned-caller',
      'derived-tree',
      'reference-neighbors',
      'reference-parameters'
    ] as const) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(
        graph,
        testCase === 'aligned-caller'
          ? 64
          : testCase === 'derived-tree'
            ? 65536
            : testCase === 'reference-neighbors'
              ? 1024
              : 1
      );
      if (testCase === 'aligned-caller') {
        const storage = createTransientView(graph, 'offset-count', 'uint32', 1025);
        props.output.status = {
          count: graph.createDataView(storage.buffer, {
            format: 'uint32',
            length: 1024,
            byteOffset: 4
          })
        };
        device.limits.maxStorageBufferBindingSize = 4096;
      } else if (testCase === 'derived-tree') {
        device.limits.maxStorageBufferBindingSize = 1024 * 1024;
      } else {
        props.coreDistanceSearch = 'neighbor-search';
        props.minimumSamples = testCase === 'reference-parameters' ? 2 : 32;
        device.limits.maxStorageBufferBindingSize =
          testCase === 'reference-parameters' ? 32 : 65536;
      }
      const clustering = new GPUHierarchicalDensityClustering(props);
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const createView = vi.spyOn(graph, 'createDataView');
      const add = vi.spyOn(graph, 'add');
      expect(() => clustering.getCommandNodes(graph)).toThrow(
        testCase === 'aligned-caller'
          ? /count chunk 0 needs 4100 bytes/
          : testCase === 'derived-tree'
            ? /BVH node data/
            : testCase === 'reference-parameters'
              ? /neighbor parameter data/
              : /neighbor distances, IDs and weights/
      );
      expect(allocate).not.toHaveBeenCalled();
      expect(createView).not.toHaveBeenCalled();
      expect(add).not.toHaveBeenCalled();
    }
  } finally {
    device.destroy();
  }
});

it('adds query lanes and tree rounds without adding distinct iterative pipelines', () => {
  const device = createNullWebGPUDevice();
  try {
    const getNodes = (rows: number) => {
      const graph = new GPUCommandGraph(device);
      return new GPUHierarchicalDensityClustering(createProps(graph, rows)).getCommandNodes(graph);
    };
    const small = getNodes(1024);
    const large = getNodes(4097);
    const pattern =
      /-(bvh-core|gate|resolve|leaf-flags|search|tie-low|tie-high|add|tour-(gate|jump|copy))-?\d*$|-splice-(gate|round)-\d+|-(ancestors|depth|representatives)-(gate|jump|copy)-|selection-(level-gate|functions|chain-gate|chain-jump|chain-copy|best|advance)-/;
    expect(large.length).toBeGreaterThan(small.length);
    expect(getGPUCommandNodePipelineKeys(large, {idPattern: pattern}).size).toBe(
      getGPUCommandNodePipelineKeys(small, {idPattern: pattern}).size
    );
    // Audit all generated sources, including the reference route and the optional time binding.
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, 257);
    const timed = new GPUHierarchicalDensityClustering({
      ...props,
      times: createTransientView(graph, 'times', 'float32', 257),
      parameters: createTransientView(graph, 'parameters', 'float32', 4),
      temporalContext: {
        unit: 'second',
        epoch: 0n,
        encoding: 'float32-relative',
        axis: {kind: 'irregular'}
      }
    }).getCommandNodes(graph);
    const referenceGraph = new GPUCommandGraph(device);
    const reference = new GPUHierarchicalDensityClustering({
      ...createProps(referenceGraph, 257),
      coreDistanceSearch: 'neighbor-search'
    }).getCommandNodes(referenceGraph);
    for (const nodes of [small, large, timed, reference]) {
      const recordedKernelSources1 = getParsedKernelSources(nodes);
      for (const {
        id: kernelId,
        source: source,
        constants
      } of recordedKernelSources1.configurations) {
        expect(getReservedWGSLIdentifiers(source), kernelId).toEqual([]);
        expect(getStorageBindingCount(source), kernelId).toBeLessThanOrEqual(8);
        expect(getPrivateMemoryBytes(source, constants), kernelId).toBeLessThanOrEqual(8192);
      }
    }
  } finally {
    device.destroy();
  }
});

it('audits every WGSL reserved word and accepts sample as an identifier', () => {
  for (const word of WGSL_RESERVED_WORDS) {
    expect(getReservedWGSLIdentifiers(`let ${word} = 1u;`), word).toEqual([
      {check: 'reserved word', detail: expect.stringContaining(`\`${word}\``)}
    ]);
  }
  expect(WGSL_RESERVED_WORDS.has('sample')).toBe(false);
  expect(getReservedWGSLIdentifiers('let sample = 1u;')).toEqual([]);
});

it('rejects numeric props that cannot become finite uint32 WGSL literals at construction', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, 64);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    for (const name of ['minimumClusterSize', 'minimumSamples'] as const) {
      for (const value of [NaN, Infinity, -Infinity, 1.5, 2 ** 32, Number.MAX_VALUE]) {
        expect(
          () => new GPUHierarchicalDensityClustering({...props, [name]: value}),
          `${name}=${value}`
        ).toThrow(new RegExp(`^hierarchical-density-clustering ${name} must be an integer`));
      }
    }
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('validates requiredCount scalar shape and the whole-buffer usage scope of simultaneous status bindings', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, 64);
    const shared = createTransientView(graph, 'shared-status', 'uint32', 129).buffer;
    const statusView = (byteOffset: number, length = 1, byteStride = 4) =>
      graph.createDataView(shared, {
        format: 'uint32',
        byteOffset,
        byteStride,
        length
      });
    for (const requiredCount of [statusView(0, 0), statusView(0, 1, 8)]) {
      expect(
        () =>
          new GPUHierarchicalDensityClustering({
            ...props,
            output: {
              ...props.output,
              status: {requiredCount}
            }
          })
      ).toThrow(/requiredCount/);
    }
    for (const other of [
      props.output.labels,
      props.output.coreDistances!,
      props.output.status!.count!
    ]) {
      const requiredCount = graph.createDataView(other.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: other.byteOffset
      });
      expect(
        () =>
          new GPUHierarchicalDensityClustering({
            ...props,
            output: {
              ...props.output,
              status: {...props.output.status, requiredCount}
            }
          })
      ).toThrow(/writable byte ranges/);
    }
    const count = statusView(0);
    expect(
      () =>
        new GPUHierarchicalDensityClustering({
          ...props,
          output: {
            ...props.output,
            status: {count, requiredCount: statusView(4)}
          }
        })
    ).toThrow(/physical buffers/);
    expect(
      () =>
        new GPUHierarchicalDensityClustering({
          ...props,
          output: {
            ...props.output,
            status: {count, requiredCount: statusView(256)}
          }
        })
    ).toThrow(/physical buffers/);
    const nodes = new GPUHierarchicalDensityClustering({
      ...props,
      output: {
        ...props.output,
        status: {requiredCount: props.output.status!.requiredCount}
      }
    }).getCommandNodes(graph);
    expect(nodes.some(node => node.id.endsWith('-required-count'))).toBe(true);
    const sources = getParsedKernelSources(
      nodes.filter(node => node.id.endsWith('-required-count'))
    );
    expect([...sources.values()][0]).toContain(
      'requiredCount[requiredCountOffset] = selectedCount[selectedCountOffset]'
    );
  } finally {
    device.destroy();
  }
});

it('bounds each reconstruction dispatch and keeps splice shader/pipeline counts constant as rounds grow', () => {
  const device = createNullWebGPUDevice();
  try {
    const build = (rows: number) => {
      const graph = new GPUCommandGraph(device);
      return new GPUHierarchicalDensityClustering(createProps(graph, rows)).getCommandNodes(graph);
    };
    const small = build(1024);
    const large = build(8193);
    const pattern = /-dendrogram-splice-(round|gate)-/;
    const smallRounds = small.filter(node => /-splice-round-/.test(node.id));
    const largeRounds = large.filter(node => /-splice-round-/.test(node.id));
    expect(largeRounds.length).toBeGreaterThan(smallRounds.length);
    expect(getGPUCommandNodePipelineKeys(large, {idPattern: pattern}).size).toBe(
      getGPUCommandNodePipelineKeys(small, {idPattern: pattern}).size
    );
    const sources = [...getParsedKernelSources(largeRounds).values()];
    expect(new Set(sources).size).toBe(1);
    for (const node of largeRounds)
      expect(node.condition).toMatchObject({source: 'gpu', mode: 'indirect'});
    for (const source of sources) {
      expect(source).toMatch(/attempt < 2048u/);
      expect(source).not.toMatch(/\bloop\s*\{/);
      expect(getReservedWGSLIdentifiers(source)).toEqual([]);
    }
    const diagnosticGraph = new GPUCommandGraph(device);
    const diagnosticRows = 257;
    const view = (name: string, length: number) =>
      createTransientView(diagnosticGraph, name, 'uint32', length);
    const diagnostic = getHDBSCANDendrogramNodes(diagnosticGraph, {
      id: 'diagnostic-tree',
      shape: {
        rowCount: diagnosticRows,
        minimumSamples: 32,
        minimumClusterSize: 2,
        clusterCapacity: diagnosticRows + 1,
        leafCapacity: 512,
        internalNodeCount: 511,
        levelCount: 10
      },
      state: view('state', 2),
      edges: view('edges', diagnosticRows * 3),
      edgeOrder: view('order', diagnosticRows),
      dendrogram: view('dendrogram', diagnosticRows * 6),
      spliceMaximumWork: view('maximum-work', 1)
    });
    for (const {id: kernelId, source} of getParsedKernelSources(diagnostic.nodes).configurations) {
      expect(getReservedWGSLIdentifiers(source), kernelId).toEqual([]);
      expect(getStorageBindingCount(source), kernelId).toBeLessThanOrEqual(8);
    }
    const coreSources = getParsedKernelSources(large.filter(node => node.id.endsWith('-bvh-core')));
    expect([...coreSources.values()][0]).toContain('nearest[count - 1u] == 0.0) { break; }');
  } finally {
    device.destroy();
  }
});

it('rejects chunked inputs explicitly instead of specializing kernels or padding chunks', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const chunkLengths of [[64], [32, 32], [1, 31, 32]]) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph, 64);
      const chunks = chunkLengths.map((length, index) =>
        createTransientView(graph, `chunk-${index}`, 'float32x2', length)
      );
      const positions = createVectorView('chunked-positions', 'float32x2', chunks);
      expect(
        () =>
          new GPUHierarchicalDensityClustering({
            ...props,
            positions: positions as unknown as GraphDataView<'float32x2'>
          })
      ).toThrow(/one packed view/);
    }
  } finally {
    device.destroy();
  }
});
