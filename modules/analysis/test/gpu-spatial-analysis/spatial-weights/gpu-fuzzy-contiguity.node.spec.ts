// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {Kernel} from '@luma.gl/graph';
import {describe, expect, it, vi} from 'vitest';
import {GPUFuzzyContiguity} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity';
import {getFuzzyContiguityPairSortNodes} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity-pair-sort';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {getFuzzyContiguityOffsetNodes} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity-csr';
import {getReservedWGSLIdentifiers} from '../../gpu-dataframe/column-classification/wgsl-reserved-words';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

// wgsl_reflect's arrayLength reads the entire backing allocation, including the executor's
// extra 16 bytes. These cases use whole-buffer bindings, so trim that interpreter-only padding.
function createFuzzyCPUMemory() {
  const memory = createCPUGraphMemory();
  const getBuffer = memory.getBuffer;
  memory.getBuffer = logical => {
    const buffer = getBuffer(logical);
    if (buffer.backing.byteLength > logical.byteLength) {
      buffer.backing = buffer.backing.slice(0, logical.byteLength);
    }
    return buffer;
  };
  return memory;
}

function setup(rows: number, capacity: number, vertices = rows * 4, id = 'fuzzy-contiguity') {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const contributor = new GPUFuzzyContiguity({
    id,
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', vertices),
      featureOffsets: view('uint32', rows + 1),
      polygonOffsets: view('uint32', rows + 1),
      ringOffsets: view('uint32', rows + 1)
    },
    tolerance: 0.25,
    output: {
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', capacity),
      weights: view('float32', capacity),
      distances: view('float32', capacity),
      added: view('uint32', capacity),
      featureStatus: view('uint32', rows),
      status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
    }
  });
  return {device, graph, contributor};
}

it('declares full-width pair sorting above the former feature and vertex caps with eight bindings', () => {
  const {device, graph, contributor} = setup(65538, 131091);
  const declareBuffer = vi.spyOn(graph, 'createTransientBuffer');
  try {
    const nodes = contributor.getCommandNodes(graph);
    for (const [descriptor] of declareBuffer.mock.calls) {
      expect(descriptor.byteLength, descriptor.id).toBeLessThanOrEqual(
        device.limits.maxStorageBufferBindingSize
      );
    }
    expect(nodes.some(node => node.id.startsWith('fuzzy-contiguity-pair-sort-pass-1-sort'))).toBe(
      true
    );
    expect(new Set(nodes.map(node => node.id)).size).toBe(nodes.length);
    for (const {id, source} of getGPUCommandNodeKernelSources(nodes)) {
      expect(getReservedWGSLIdentifiers(source), id).toEqual([]);
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    }
    // Parse the contributor's kernels separately: wgsl_reflect does not parse the published
    // bounds-reduction primitive's override-sized workgroup arrays.
    const sources = getParsedKernelSources(
      nodes.filter(node =>
        /^fuzzy-contiguity-(validate|search-bounds|clear-pair-keys|pair-rows|edges-(counts|offsets|emit-.*)|finish)$/.test(
          node.id
        )
      )
    );
    expect(sources.get('fuzzy-contiguity-finish')).toContain('pairNeighborsOffset+pair');
  } finally {
    declareBuffer.mockRestore();
    device.destroy();
  }
});

it.each([
  {rows: 4, capacity: 33554433, name: 'pair scratch'},
  {rows: 4194304, capacity: 1, name: 'search bounds scratch'}
])('rejects $name over the binding limit before declaring scratch', ({rows, capacity, name}) => {
  const {device, graph, contributor} = setup(rows, capacity, 4);
  const declareBuffer = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      new RegExp(
        name === 'pair scratch'
          ? '^fuzzy-contiguity output.neighbors chunk 0 needs 134217732 bytes in one binding \\(limit 134217728\\)'
          : `^fuzzy-contiguity ${name} needs .*limit 134217728`
      )
    );
    expect(declareBuffer).not.toHaveBeenCalled();
  } finally {
    declareBuffer.mockRestore();
    device.destroy();
  }
});

it('rejects a BVH refit beyond the device dispatch bound before declaring scratch', () => {
  const {device, graph, contributor} = setup(16777217, 8, 3);
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 2 ** 31}
  });
  const declareBuffer = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /^fuzzy-contiguity features BVH leaf capacity must be at most 16777216$/
    );
    expect(declareBuffer).not.toHaveBeenCalled();
  } finally {
    declareBuffer.mockRestore();
    device.destroy();
  }
});

it('the largest accepted BVH leaf capacity refits within one dispatch dimension', () => {
  const {device, graph, contributor} = setup(16777216, 8, 3);
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 2 ** 31}
  });
  Object.defineProperty(device, 'createComputePipeline', {
    value: () => ({id: 'refit-probe', hash: '', destroy: () => {}})
  });
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  const dispatch = vi.spyOn(Kernel.prototype, 'dispatch').mockImplementation(() => {});
  let compiled:
    | {encode: (context: Record<string, unknown>) => void; destroy?: () => void}
    | undefined;
  try {
    const widest = contributor
      .getCommandNodes(graph)
      .find(node => node.id.endsWith('-refit-depth-23'));
    expect(widest).toBeDefined();
    const executable = widest as unknown as {
      compile: (context: {device: typeof device}) => NonNullable<typeof compiled>;
    };
    compiled = executable.compile({device});
    compiled.encode({computePass: {}, getBuffer: () => undefined});
    expect(dispatch).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({x: 32768}));
    expect(32768).toBeLessThanOrEqual(device.limits.maxComputeWorkgroupsPerDimension);
  } finally {
    dispatch.mockRestore();
    compiled?.destroy?.();
    device.destroy();
  }
});

it.each([
  'weights',
  'distances',
  'added'
] as const)('rejects overlapping output.neighbors and output.%s', field => {
  const {device, graph, contributor} = setup(4, 8);
  try {
    const output = {
      ...contributor.props.output,
      [field]: graph.createDataView(contributor.props.output.neighbors.buffer, {
        format: field === 'added' ? 'uint32' : 'float32',
        length: 8
      })
    };
    expect(() => new GPUFuzzyContiguity({...contributor.props, output})).toThrow(
      field === 'added'
        ? /output.added must not overlap writable bytes of output.neighbors/
        : new RegExp(
            `neighbors chunk 0 and ${field} chunk 0 aligned binding byte ranges must not overlap`
          )
    );
  } finally {
    device.destroy();
  }
});

it('rejects overlapping scalar status and feature outputs written by different kernels', () => {
  const {device, graph, contributor} = setup(4, 8);
  try {
    const output = {
      ...contributor.props.output,
      status: {
        ...contributor.props.output.status,
        requiredCount: graph.createDataView(contributor.props.output.featureStatus.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: 4
        })
      }
    };
    expect(() => new GPUFuzzyContiguity({...contributor.props, output})).toThrow(
      /output.status.requiredCount must not overlap writable bytes of output.featureStatus/
    );
  } finally {
    device.destroy();
  }
});

it('accepts disjoint aligned output ranges sharing one logical buffer', () => {
  const {device, graph, contributor} = setup(4, 8);
  try {
    const storage = createTransientView(graph, 'shared-output', 'uint32', 128);
    const output = {
      ...contributor.props.output,
      neighbors: graph.createDataView(storage.buffer, {format: 'uint32', length: 8}),
      added: graph.createDataView(storage.buffer, {format: 'uint32', byteOffset: 256, length: 8})
    };
    expect(() => new GPUFuzzyContiguity({...contributor.props, output})).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('checks the aligned prefix of an output view before declaring scratch', () => {
  const capacity = 33554432;
  const {device, graph, contributor} = setup(4, capacity, 4);
  const storage = createTransientView(graph, 'shifted-output', 'uint32', capacity + 1);
  const output = {
    ...contributor.props.output,
    neighbors: graph.createDataView(storage.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: capacity
    })
  };
  const shifted = new GPUFuzzyContiguity({...contributor.props, output});
  const declareBuffer = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => shifted.getCommandNodes(graph)).toThrow(/needs 134217732 bytes in one binding/);
    expect(declareBuffer).not.toHaveBeenCalled();
  } finally {
    declareBuffer.mockRestore();
    device.destroy();
  }
});

it('reuses shaders and pipelines as the host adds independent packed input partitions', () => {
  // The contributor accepts one packed geometry, so a host partitions through repeated producers.
  // Include source, overrides and layout: changing a partition ID must not specialize pipelines.
  const build = (partitionCount: number) => {
    const partitions = Array.from({length: partitionCount}, (_, index) =>
      setup(4, 11, 16, `partition-${index}`)
    );
    try {
      const nodes = partitions.flatMap(({graph, contributor}) =>
        contributor.getCommandNodes(graph)
      );
      return {
        pipelines: getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}),
        shaders: new Set(getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.source))
      };
    } finally {
      for (const {device} of partitions) device.destroy();
    }
  };
  const one = build(1);
  const many = build(4);
  expect(many.pipelines).toEqual(one.pipelines);
  expect(many.shaders).toEqual(one.shaders);
});

it.each([1, 4])('rejects vector geometry with %i chunks at construction', chunkCount => {
  const {device, contributor} = setup(4, 8);
  try {
    const positions = contributor.props.polygons.positions;
    const vector = new GraphVectorView({
      id: 'chunked-positions',
      name: 'positions',
      format: 'float32x2',
      length: positions.length * chunkCount,
      valueLength: positions.length * chunkCount * 2,
      stride: 2,
      byteStride: 8,
      rowByteLength: 8,
      data: new Array(chunkCount).fill(positions)
    });
    expect(
      () =>
        new GPUFuzzyContiguity({
          ...contributor.props,
          polygons: {
            ...contributor.props.polygons,
            positions: vector as unknown as typeof positions
          }
        })
    ).toThrow(/must be one packed view/);
  } finally {
    device.destroy();
  }
});

describe('multipart fuzzy contiguity against Shapely', () => {
  useCPUExecutorTestHooks();
  it('reduces distance over all polygon parts and preserves an empty feature row', () => {
    // multipart-polygon-contiguity.oracle.py: feature 0 has two disjoint unit squares;
    // its first part touches row 1, its second part lies exactly 0.25 from row 2.
    const {device, graph, contributor} = setup(4, 4, 16);
    const memory = createFuzzyCPUMemory();
    const {polygons, output} = contributor.props;
    memory.write(
      polygons.positions,
      Float32Array.from([0, 4, 1, 5.25].flatMap(x => [x, 0, x + 1, 0, x + 1, 1, x, 1]))
    );
    memory.write(polygons.featureOffsets, Uint32Array.of(0, 2, 3, 4, 4));
    memory.write(output.status.overflow, Uint32Array.of(99));
    memory.write(output.status.requiredCount!, Uint32Array.of(99));
    memory.write(polygons.polygonOffsets, Uint32Array.of(0, 1, 2, 3, 4));
    memory.write(polygons.ringOffsets, Uint32Array.of(0, 4, 8, 12, 16));
    try {
      executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
      expect(Array.from(memory.readUint32(output.featureStatus))).toEqual([0, 0, 0, 1]);
      expect(Array.from(memory.readUint32(output.offsets))).toEqual([0, 2, 3, 4, 4]);
      expect(Array.from(memory.readUint32(output.neighbors))).toEqual([1, 2, 0, 0]);
      expect(Array.from(memory.readFloat32(output.weights))).toEqual([1, 1, 1, 1]);
      expect(Array.from(memory.readFloat32(output.distances!))).toEqual([0, 0.25, 0, 0.25]);
      expect(Array.from(memory.readUint32(output.added))).toEqual([0, 1, 0, 1]);
      expect(Array.from(memory.readUint32(output.status.requiredCount!))).toEqual([4]);
      expect(Array.from(memory.readUint32(output.status.overflow))).toEqual([0]);
    } finally {
      device.destroy();
    }
  });
});

describe('fuzzy pair sorting executes the shared gathers around stable sort passes', () => {
  useCPUExecutorTestHooks();

  it('saturates wrapped prefixes before clamping them and reporting overflow', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const counts = createTransientView(graph, 'counts', 'uint32', 3);
    const offsets = createTransientView(graph, 'offsets', 'uint32', 4);
    const slots = createTransientView(graph, 'slots', 'uint32', 8);
    const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
    const requiredCount = createTransientView(graph, 'required', 'uint32', 1);
    const memory = createFuzzyCPUMemory();
    memory.write(counts, new Uint32Array([0xfffffffe, 2, 7]));
    try {
      executeCommandNodesOnCPU(
        getFuzzyContiguityOffsetNodes(graph, {
          id: 'synthetic',
          counts,
          offsets,
          slots,
          overflow,
          requiredCount
        }),
        memory
      );
      expect(Array.from(memory.readUint32(offsets))).toEqual([0, 8, 8, 8]);
      expect(Array.from(memory.readUint32(overflow))).toEqual([1]);
      expect(Array.from(memory.readUint32(requiredCount))).toEqual([0xffffffff]);
    } finally {
      device.destroy();
    }
  });

  it.each([
    3, 11
  ])('keeps every output aligned against a small polygon oracle with %i slots', capacity => {
    const {device, graph, contributor} = setup(4, capacity);
    const memory = createFuzzyCPUMemory();
    const {polygons, output} = contributor.props;
    const positions = [0, 1, 2.25, 20].flatMap(x => [x, 0, x + 1, 0, x + 1, 1, x, 1]);
    memory.write(polygons.positions, Float32Array.from(positions));
    memory.write(polygons.featureOffsets, new Uint32Array([0, 1, 2, 3, 4]));
    memory.write(polygons.polygonOffsets, new Uint32Array([0, 1, 2, 3, 4]));
    memory.write(polygons.ringOffsets, new Uint32Array([0, 4, 8, 12, 16]));
    try {
      executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
      expect(Array.from(memory.readUint32(output.featureStatus))).toEqual([0, 0, 0, 0]);
      expect(Array.from(memory.readUint32(output.status.requiredCount!))).toEqual([4]);
      expect(Array.from(memory.readUint32(output.status.overflow))).toEqual([capacity < 4 ? 1 : 0]);
      expect(Array.from(memory.readUint32(output.offsets))).toEqual(
        capacity < 4 ? [0, 1, 3, 3, 3] : [0, 1, 3, 4, 4]
      );
      const used = Math.min(capacity, 4);
      expect(Array.from(memory.readUint32(output.neighbors, used))).toEqual(
        [1, 0, 2, 1].slice(0, used)
      );
      expect(Array.from(memory.readFloat32(output.weights, used))).toEqual(new Array(used).fill(1));
      expect(Array.from(memory.readFloat32(output.distances!, used))).toEqual(
        [0, 0, 0.25, 0.25].slice(0, used)
      );
      expect(Array.from(memory.readUint32(output.added, used))).toEqual(
        [0, 0, 1, 1].slice(0, used)
      );
    } finally {
      device.destroy();
    }
  });

  it.each([65536, 65537, 0xfffffffe])('retains tuple ordering for %i possible rows', rows => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const highRow = rows - 1;
    const pairs = [
      [highRow, 0],
      [0, highRow],
      [7, highRow],
      [highRow, 7],
      [7, 0],
      [7, 0],
      [0xffffffff, 0xffffffff],
      [0xffffffff, 0xffffffff]
    ];
    const pairRows = createTransientView(graph, 'rows', 'uint32', pairs.length);
    const pairNeighbors = createTransientView(graph, 'neighbors', 'uint32', pairs.length);
    const memory = createCPUGraphMemory();
    memory.write(pairRows, Uint32Array.from(pairs.map(pair => pair[0])));
    memory.write(pairNeighbors, Uint32Array.from(pairs.map(pair => pair[1])));
    try {
      const sorted = getFuzzyContiguityPairSortNodes(graph, {
        id: 'pairs',
        rows,
        pairRows,
        pairNeighbors
      });
      executeCommandNodesOnCPU(sorted.nodes, memory);
      const indices = Array.from(memory.readUint32(sorted.sortedIndices));
      // A direct tuple comparison is independent of the shared least-to-most stable passes.
      const expected = pairs
        .map((_, index) => index)
        .sort(
          (left, right) => pairs[left][0] - pairs[right][0] || pairs[left][1] - pairs[right][1]
        );
      expect(indices).toEqual(expected);
      if (rows === 65536) {
        const legacy = pairs
          .map((pair, index) => ({key: ((pair[0] << 16) | pair[1]) >>> 0, index}))
          .sort((left, right) => left.key - right.key)
          .map(pair => pair.index);
        expect(indices).toEqual(legacy);
      }
    } finally {
      device.destroy();
    }
  });
});
