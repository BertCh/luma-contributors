// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {Buffer} from '@luma.gl/core';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {getReservedWGSLWords} from './wgsl-reserved-words';
import {getChunkDispatches} from './chunk-dispatch-test-utils';
import {expect, expectTypeOf, it, vi} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import type {GPUUint32Rows} from '../../../src/analysis-core/kernels/gpu-contributor-types';
import type {GPUPointGeometryPort} from '../../../src/analysis-core/contracts/index';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUPointInPolygonJoin,
  type GPUPointInPolygonJoinOutput,
  type GPUPointInPolygonJoinProps
} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {
  createNullWebGPUDevice,
  createVectorView,
  getParsedKernelSources,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function createFixture(lengths: number[]) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'join-chunks'});
  const rowCount = lengths.reduce((total, length) => total + length, 0);
  const pointChunks = lengths.map((length, index) =>
    createTransientView(graph, `points-${index}`, 'float32x2', length)
  );
  const outputChunks = lengths.map((length, index) =>
    createTransientView(graph, `assignments-${index}`, 'uint32', length)
  );
  const flagChunks = lengths.map((length, index) =>
    createTransientView(graph, `flags-${index}`, 'uint32', length)
  );
  // This fixture intentionally creates single-point chunks, also consumed by the nearest join.
  const props: GPUPointInPolygonJoinProps<GPUUint32Rows, unknown, GPUPointGeometryPort> = {
    id: 'join-test',
    points: {kind: 'points', positions: createVectorView('points', 'float32x2', pointChunks)},
    polygons: {
      kind: 'polygons',
      positions: createTransientView(graph, 'vertices', 'float32x2', 4),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2)
    },
    candidateCapacity: 16,
    output: {
      pointFeatureIds: createVectorView('assignments', 'uint32', outputChunks),
      candidateTruncated: createVectorView('flags', 'uint32', flagChunks),
      polygonAssignmentCounts: createTransientView(graph, 'feature-counts', 'uint32', 1),
      matches: {
        ids: createTransientView(graph, 'match-ids', 'uint32', 3),
        status: {
          count: createTransientView(graph, 'match-count', 'uint32', 1),
          overflow: createTransientView(graph, 'match-overflow', 'uint32', 1),
          requiredCount: createTransientView(graph, 'match-required', 'uint32', 1)
        }
      },
      status: {
        overflow: createTransientView(graph, 'overflow', 'uint32', 1),
        candidateOverflow: createTransientView(graph, 'candidate-overflow', 'uint32', 1)
      }
    }
  };
  return {device, graph, props, rowCount};
}

for (const edgeSlabs of [false, true]) {
  for (const querySort of [false, true]) {
    it(`GPUPointInPolygonJoin chunked truncation flags parse and preserve row offsets (slabs ${edgeSlabs}, sort ${querySort})`, () => {
      const {device, graph, props} = createFixture([2, 3, 5]);
      try {
        const contributor = new GPUPointInPolygonJoin({...props, edgeSlabs, querySort});
        const nodes = contributor.getCommandNodes(graph);
        expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
        expect(new Set(nodes.map(node => node.id)).size).toBe(nodes.length);
        const sources = getParsedKernelSources(nodes);
        expect(sources.has('join-test-probe-scan-saturate')).toBe(true);
        expect(sources.has('join-test-probe-total')).toBe(false);
        const copies = [...sources].filter(([id]) =>
          /^join-test-candidate-truncated-\d+$/.test(id)
        );
        expect(copies).toHaveLength(3);
        expect(new Set(copies.map(([, source]) => source)).size).toBe(1);
        expect(
          getGPUCommandNodePipelineKeys(nodes, {idPattern: /^join-test-candidate-truncated-\d+$/})
            .size
        ).toBe(1);
        for (const {source} of sources.configurations) {
          expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
          expect(getReservedWGSLWords(source)).toEqual([]);
        }
      } finally {
        device.destroy();
      }
    });
  }
}

it('GPUPointInPolygonJoin declares thirty million points in three uneven chunks within its node budget', () => {
  const {device, graph, props} = createFixture([5_000_000, 9_000_000, 16_000_000]);
  try {
    const contributor = new GPUPointInPolygonJoin({...props, querySort: true});
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin keeps accepting packed truncation flags with chunked points', () => {
  const {device, graph, props, rowCount} = createFixture([2, 3, 5]);
  try {
    const candidateTruncated = createTransientView(graph, 'packed-flags', 'uint32', rowCount);
    const contributor = new GPUPointInPolygonJoin({
      ...props,
      output: {...props.output, candidateTruncated}
    });
    expect(
      contributor
        .getCommandNodes(graph)
        .some(node => /^join-test-candidate-truncated-\d+$/.test(node.id))
    ).toBe(false);
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin rejects truncation vectors with different chunk boundaries', () => {
  const {device, graph, props} = createFixture([2, 3, 5]);
  try {
    const candidateTruncated = createVectorView(
      'wrong-flags',
      'uint32',
      [3, 2, 5].map((length, index) =>
        createTransientView(graph, `wrong-flags-${index}`, 'uint32', length)
      )
    );
    expect(
      () => new GPUPointInPolygonJoin({...props, output: {...props.output, candidateTruncated}})
    ).toThrow(/join-test output.candidateTruncated must match the chunk topology of points/);
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin checks row scratch binding sizes before declaring nodes', () => {
  const {device, graph, props} = createFixture([10_000_000, 10_000_000, 15_000_000]);
  try {
    const contributor = new GPUPointInPolygonJoin(props);
    const createScratch = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /join-test point assignment, probe and match scratch.*limit/
    );
    expect(createScratch).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin pages the wider slab candidate binding', () => {
  const {device, graph, props} = createFixture([2, 3, 5]);
  try {
    const contributor = new GPUPointInPolygonJoin({
      ...props,
      edgeSlabs: true,
      candidateCapacity: 9_000_000
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(contributor.candidatePages.pageCount).toBe(2);
    expect(contributor.candidatePages.capacity * 16).toBeLessThanOrEqual(
      device.limits.maxStorageBufferBindingSize
    );
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin rejects oversized BVH scratch before declaring any resources', () => {
  const {device, graph, props} = createFixture([2, 3, 1]);
  try {
    const contributor = new GPUPointInPolygonJoin({...props, leafCapacity: 16_777_216});
    const createScratch = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /join-test BVH node bounds and children.*268435448.*limit.*reduce leafCapacity/
    );
    expect(createScratch).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin reuses every chunk pipeline with uneven sizes and nonzero offsets', () => {
  const {device, graph, props} = createFixture([1, 255, 256]);
  try {
    const pointChunks = [1, 255, 256].map((length, index) => {
      const backing = createTransientView(
        graph,
        `offset-points-${index}`,
        'float32x2',
        length + 32
      );
      return graph.createDataView(backing.buffer, {
        format: 'float32x2',
        length,
        byteOffset: 4 + index * 4
      });
    });
    const flagChunks = [1, 255, 256].map((length, index) => {
      const backing = createTransientView(graph, `offset-flags-${index}`, 'uint32', length + 64);
      return graph.createDataView(backing.buffer, {
        format: 'uint32',
        length,
        byteOffset: 4 + index * 4
      });
    });
    const contributor = new GPUPointInPolygonJoin({
      ...props,
      querySort: true,
      points: {
        ...props.points,
        positions: createVectorView('offset-points', 'float32x2', pointChunks)
      },
      output: {
        ...props.output,
        candidateTruncated: createVectorView('offset-flags', 'uint32', flagChunks)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    const patterns = [
      /^join-test-probe-\d+$/,
      /^join-test-probe-\d+-write$/,
      /^join-test-probe-order-keys-\d+$/,
      /^join-test-assign-\d+$/,
      /^join-test-collect-matches-\d+$/,
      /^join-test-collect-matches-\d+-write$/,
      /^join-test-candidate-truncated-\d+$/
    ];
    for (const idPattern of patterns) {
      expect(nodes.filter(node => idPattern.test(node.id))).toHaveLength(3);
      expect(getGPUCommandNodePipelineKeys(nodes, {idPattern}).size).toBe(1);
    }
    for (const source of getParsedKernelSources(nodes).values())
      expect(getReservedWGSLWords(source)).toEqual([]);
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin preflights derived edge-slab parameters before allocating scratch', () => {
  const {device, graph, props} = createFixture([2, 3, 1]);
  try {
    const polygonOffsets = createTransientView(graph, 'many-parts', 'uint32', 6_000_001);
    const contributor = new GPUPointInPolygonJoin({
      ...props,
      edgeSlabs: true,
      polygons: {...props.polygons, polygonOffsets}
    });
    const createScratch = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /join-test edge slab parameters.*144000000.*limit/
    );
    expect(createScratch).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('the other counted-probe caller uses saturated offsets and valid WGSL', () => {
  const {device, graph, props} = createFixture([2, 3, 5]);
  try {
    const contributor = new GPUNearestFeatureJoin({
      id: 'nearest-shared-scan',
      queries: props.points,
      features: {
        kind: 'points',
        positions: createTransientView(graph, 'nearest-feature-point', 'float32x2', 1)
      },
      radius: createTransientView(graph, 'radius', 'float32', 1),
      candidateCapacity: 4,
      querySort: true,
      output: {
        nearestFeatureIds: props.output.pointFeatureIds,
        candidateTruncated: createTransientView(graph, 'nearest-flags', 'uint32', 10),
        matches: props.output.matches,
        featureCounts: props.output.polygonAssignmentCounts,
        status: {
          overflow: props.output.status.overflow,
          candidateOverflow: props.output.status.candidateOverflow
        }
      }
    });
    const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
    expect(sources.has('nearest-shared-scan-probe-scan-saturate')).toBe(true);
    for (const source of sources.values()) expect(getReservedWGSLWords(source)).toEqual([]);
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin keeps concrete packed output types and infers vector flags', () => {
  const {device, graph, props} = createFixture([1]);
  try {
    expectTypeOf<GPUPointInPolygonJoinOutput['candidateTruncated']>().toEqualTypeOf<
      GraphDataView<'uint32'> | undefined
    >();
    expectTypeOf<GPUPointInPolygonJoinProps['output']['candidateTruncated']>().toEqualTypeOf<
      GraphDataView<'uint32'> | undefined
    >();
    const flags = createTransientView(graph, 'type-packed-flags', 'uint32', 1);
    const packed = new GPUPointInPolygonJoin({
      ...props,
      output: {...props.output, candidateTruncated: flags}
    });
    expectTypeOf(packed.props.output.candidateTruncated).toEqualTypeOf<
      GraphDataView<'uint32'> | undefined
    >();
    const vector = createVectorView('type-vector-flags', 'uint32', [flags]);
    const chunked = new GPUPointInPolygonJoin({
      ...props,
      output: {...props.output, candidateTruncated: vector}
    });
    expectTypeOf(chunked.props.output.candidateTruncated).toEqualTypeOf<
      typeof vector | undefined
    >();
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin rejects co-bound output alignment overlap but allows consecutive output chunks', () => {
  const {device, graph, props} = createFixture([1, 1, 1]);
  try {
    const storage = createTransientView(graph, 'shared-output', 'uint32', 3);
    const chunks = [0, 4, 8].map(byteOffset =>
      graph.createDataView(storage.buffer, {format: 'uint32', length: 1, byteOffset})
    );
    const pointFeatureIds = createVectorView('consecutive-assignments', 'uint32', chunks);
    expect(
      () => new GPUPointInPolygonJoin({...props, output: {...props.output, pointFeatureIds}})
    ).not.toThrow();
    expect(
      () =>
        new GPUPointInPolygonJoin({
          ...props,
          output: {...props.output, pointFeatureIds, polygonAssignmentCounts: chunks[0]}
        })
    ).toThrow(/join-test.*must not overlap/);
    // Logical [0,4) and [4,8) are disjoint, but the aligned assignment binding is [0,8).
    const alignedAssignments = createVectorView('aligned-assignments', 'uint32', [
      chunks[1],
      createTransientView(graph, 'separate-assignment-1', 'uint32', 1),
      createTransientView(graph, 'separate-assignment-2', 'uint32', 1)
    ]);
    expect(
      () =>
        new GPUPointInPolygonJoin({
          ...props,
          output: {
            ...props.output,
            pointFeatureIds: alignedAssignments,
            polygonAssignmentCounts: chunks[0]
          }
        })
    ).toThrow(/join-test.*aligned binding byte ranges.*must not overlap/);
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin dispatches query keys at each chunk length and keeps pipeline counts constant as chunks grow', () => {
  const keySets: string[][] = [];
  const shaderSets: string[][] = [];
  for (const lengths of [[65_537], [65_536, 1], [32_768, 32_768, 1], [1, 255, 256, 65_025]]) {
    const {device, graph, props} = createFixture(lengths);
    try {
      const nodes = new GPUPointInPolygonJoin({
        ...props,
        querySort: true,
        edgeSlabs: false
      }).getCommandNodes(graph);
      const keyNodes = nodes.filter(node => /^join-test-probe-order-keys-\d+$/.test(node.id));
      expect(keyNodes.map(node => node.workload?.maximumInvocationCount)).toEqual(lengths);
      expect(getChunkDispatches(device, keyNodes)).toEqual(
        lengths.map(length => [Math.ceil(length / 256), 1, 1])
      );
      expect(
        nodes.filter(node => /probe-order-.*sort-radix-digit-\d+-histogram/.test(node.id))
      ).toHaveLength(16);
      keySets.push([...getGPUCommandNodePipelineKeys(nodes)].sort());
      const sources = getParsedKernelSources(nodes);
      shaderSets.push([...new Set(sources.values())].sort());
      for (const source of sources.values()) expect(getReservedWGSLWords(source)).toEqual([]);
    } finally {
      device.destroy();
    }
  }
  for (let index = 1; index < keySets.length; index++) {
    expect(keySets[index]).toEqual(keySets[0]);
    expect(shaderSets[index]).toEqual(shaderSets[0]);
  }
  const {device, graph, props} = createFixture([1, 1, 1]);
  try {
    const nodes = new GPUPointInPolygonJoin({...props, querySort: true}).getCommandNodes(graph);
    expect(nodes.some(node => node.id.startsWith('join-test-probe-order-'))).toBe(false);
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin preflights aligned point ranges before declaring resources', () => {
  const {device, graph, props} = createFixture([16]);
  try {
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 256}
    });
    const backing = createTransientView(graph, 'offset-input', 'float32x2', 64);
    const positions = createVectorView('offset-point-vector', 'float32x2', [
      graph.createDataView(backing.buffer, {
        format: 'float32x2',
        length: 16,
        byteOffset: 132
      })
    ]);
    const contributor = new GPUPointInPolygonJoin({...props, points: {...props.points, positions}});
    const createScratch = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /join-test points.positions chunk 0.*260.*limit.*256/
    );
    expect(createScratch).not.toHaveBeenCalled();
    // WGSL permits sample; the canonical reserved set must not reject this existing identifier.
    expect(getReservedWGSLWords('let sample = 0u;')).toEqual([]);
    expect(
      getReservedWGSLWords(
        'let target = 0u; let active = 1u; let filter = 2u; let module = 3u; let meta = 4u;'
      )
    ).toEqual(['target', 'active', 'filter', 'module', 'meta']);
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin refuses candidate and BVH allocations above maxBufferSize before scratch declaration', () => {
  for (const leafCapacity of [1, 16]) {
    const {device, graph, props} = createFixture([2, 3, 5]);
    try {
      Object.defineProperty(device, 'limits', {
        value: {...device.limits, maxBufferSize: leafCapacity === 1 ? 64 : 128}
      });
      const contributor = new GPUPointInPolygonJoin({...props, leafCapacity, edgeSlabs: false});
      const createScratch = vi.spyOn(graph, 'createTransientBuffer');
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        leafCapacity === 1
          ? /join-test point assignment, probe and match scratch byteLength.*maxBufferSize/
          : /join-test BVH node bounds and children byteLength.*maxBufferSize/
      );
      expect(createScratch).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  }
});

it('GPUPointInPolygonJoin rejects nonfinite and unrepresentable numeric props at construction', () => {
  const {device, graph, props} = createFixture([2, 3, 5]);
  try {
    for (const candidateCapacity of [NaN, Infinity, -Infinity, 1e100, 0x7fffffff, 1.5]) {
      expect(() => new GPUPointInPolygonJoin({...props, candidateCapacity})).toThrow(
        /join-test candidateCapacity.*2147483646/
      );
    }
    for (const leafCapacity of [NaN, Infinity, -Infinity, 1e100, 2 ** 32]) {
      expect(() => new GPUPointInPolygonJoin({...props, leafCapacity})).toThrow(
        /join-test leafCapacity/
      );
    }
    const invalidPositions = createTransientView(graph, 'oversized-count', 'float32x2', 2 ** 32);
    expect(
      () =>
        new GPUPointInPolygonJoin({
          ...props,
          points: {...props.points, positions: invalidPositions}
        })
    ).toThrow(/join-test points.positions.length.*4294967295/);
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin checks full backing allocation size even when the bound view fits', () => {
  const {device, graph, props} = createFixture([2]);
  try {
    const backing = createTransientView(graph, 'oversized-backing', 'float32x2', 8);
    const positions = createVectorView('short-positions', 'float32x2', [
      graph.createDataView(backing.buffer, {format: 'float32x2', length: 2})
    ]);
    // Declare the caller buffer first, then simulate a stricter compile-time device limit.
    Object.defineProperty(device, 'limits', {value: {...device.limits, maxBufferSize: 48}});
    const contributor = new GPUPointInPolygonJoin({...props, points: {...props.points, positions}});
    const declarations = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /join-test points.positions.buffer.byteLength.*maxBufferSize/
    );
    expect(declarations).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin preflights the complete default slab entry budget against maxBufferSize', () => {
  const {device, graph, props} = createFixture([2, 3, 5]);
  try {
    Object.defineProperty(device, 'limits', {value: {...device.limits, maxBufferSize: 1024}});
    const contributor = new GPUPointInPolygonJoin({...props, edgeSlabs: true});
    const declarations = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /join-test edge slab entries byteLength.*maxBufferSize/
    );
    expect(declarations).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('GPUPointInPolygonJoin writes candidate overflow even when every point chunk is empty', () => {
  const {device, graph, props} = createFixture([0, 0, 0]);
  try {
    const nodes = new GPUPointInPolygonJoin(props).getCommandNodes(graph);
    const status = nodes.find(node => node.id === 'join-test-probe-candidate-overflow');
    expect(status).toBeDefined();
    expect(nodes.indexOf(status!)).toBeGreaterThan(
      nodes.findIndex(node => node.id === 'join-test-clear')
    );
    expect(nodes.indexOf(status!)).toBeLessThan(
      nodes.findIndex(node => node.id === 'join-test-finalize')
    );
    const [kernel] = getGPUCommandNodeKernelSources([status!]);
    expect(getReservedWGSLWords(kernel.source)).toEqual([]);
    const values: Record<string, Uint32Array> = {
      total: new Uint32Array(4),
      bvhOverflow: new Uint32Array(1),
      candidateOverflow: new Uint32Array([0xffffffff])
    };
    const bindings = Object.fromEntries(
      new WgslReflect(kernel.source).storage.map(binding => [binding.binding, values[binding.name]])
    );
    for (const leafOverflow of [0, 1, 0]) {
      values['bvhOverflow'][0] = leafOverflow;
      new WgslExec(new WgslParser().parse(kernel.source)).dispatchWorkgroups(
        'main',
        [1, 1, 1],
        {0: bindings},
        {constants: kernel.constants}
      );
      expect(values['candidateOverflow'][0]).toBe(leafOverflow);
    }
  } finally {
    device.destroy();
  }
});

for (const edgeSlabs of [false, true]) {
  it(`point candidate pages reuse pipelines and source modules (slabs ${edgeSlabs})`, () => {
    const sizes: number[][] = [];
    for (const capacity of [7, 19, 31]) {
      const {device, graph, props} = createFixture([1, 255, 257]);
      try {
        const contributor = new GPUPointInPolygonJoin({
          ...props,
          candidateCapacity: capacity,
          candidateTileCapacity: 3,
          edgeSlabs,
          querySort: true
        });
        const nodes = contributor.getCommandNodes(graph);
        expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
        const sources = getParsedKernelSources(nodes);
        for (const source of sources.values()) expect(getReservedWGSLWords(source)).toEqual([]);
        const writes = nodes.filter(node => /tile-\d+-probe-\d+-write$/.test(node.id));
        expect(writes).toHaveLength(Math.ceil(capacity / 3) * 3);
        expect(getGPUCommandNodePipelineKeys(writes).size).toBe(1);
        expect(getChunkDispatches(device, writes)).toEqual(
          Array.from({length: Math.ceil(capacity / 3)}, () => [
            [1, 1, 1],
            [1, 1, 1],
            [2, 1, 1]
          ]).flat()
        );
        sizes.push([new Set(sources.values()).size, getGPUCommandNodePipelineKeys(nodes).size]);
      } finally {
        device.destroy();
      }
    }
    expect(sizes[1]).toEqual(sizes[0]);
    expect(sizes[2]).toEqual(sizes[0]);
  });
}

for (const candidateTileCapacity of [3, 16]) {
  it(`querySort retains Hilbert candidate-write order with tile capacity ${candidateTileCapacity}`, () => {
    const {device, graph, props} = createFixture([1, 255, 257]);
    try {
      const nodes = new GPUPointInPolygonJoin({
        ...props,
        querySort: true,
        candidateTileCapacity
      }).getCommandNodes(graph);
      const writers = nodes.filter(node => /probe(?:-\d+)?-write$/.test(node.id));
      expect(writers).toHaveLength(3 * Math.ceil(16 / candidateTileCapacity));
      expect(getGPUCommandNodePipelineKeys(writers).size).toBe(1);
      for (const {source} of getGPUCommandNodeKernelSources(writers)) {
        expect(source).toMatch(/queryOrder\[chunkSettings\[\d+\]\[\d+\] \+ index\]/);
        // Inactive source rows must return before the ordered point read or BVH traversal.
        expect(source.indexOf('if (globalStart >= pageEnd')).toBeLessThan(
          source.lastIndexOf('let point = vec2f(')
        );
        expect(getReservedWGSLWords(source)).toEqual([]);
      }
      expect(getChunkDispatches(device, writers)).toEqual(
        Array.from({length: Math.ceil(16 / candidateTileCapacity)}, () => [
          [1, 1, 1],
          [1, 1, 1],
          [2, 1, 1]
        ]).flat()
      );
    } finally {
      device.destroy();
    }
  });
}

it('point joins accept aligned packed outputs and ignore untouched planar status aliases', () => {
  const {device, graph, props} = createFixture([1]);
  try {
    const storage = createTransientView(graph, 'packed-output', 'uint32', 256);
    const field = (byteOffset: number) =>
      graph.createDataView(storage.buffer, {format: 'uint32', length: 1, byteOffset});
    const pointFeatureIds = field(0);
    const featureCounts = field(256);
    const overflow = field(512);
    const join = new GPUPointInPolygonJoin({
      ...props,
      output: {
        pointFeatureIds,
        polygonAssignmentCounts: featureCounts,
        status: {
          overflow,
          antimeridianUncertainCount: overflow,
          poleUncertainCount: overflow
        }
      }
    });
    expect(() => join.getCommandNodes(graph)).not.toThrow();
    expect(() =>
      new GPUPointInPolygonJoin({
        ...props,
        id: 'unused-source-ids',
        points: {...props.points, sourceIds: pointFeatureIds},
        output: join.props.output
      }).getCommandNodes(graph)
    ).not.toThrow();
    const matches = {
      ids: field(4),
      status: props.output.matches!.status
    };
    expect(() =>
      new GPUPointInPolygonJoin({
        ...props,
        id: 'separate-passes',
        output: {...join.props.output, matches}
      }).getCommandNodes(graph)
    ).not.toThrow();
    const invalid = new GPUPointInPolygonJoin({
      ...props,
      output: {
        pointFeatureIds: graph.createDataView(
          graph.createTransientBuffer({id: 'no-storage', byteLength: 4, usage: Buffer.COPY_DST}),
          {format: 'uint32', length: 1}
        ),
        status: {overflow}
      }
    });
    const allocation = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => invalid.getCommandNodes(graph)).toThrow(/STORAGE/);
    expect(allocation).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('point joins preflight nested prepared edge-slab STORAGE usage before allocating nodes', () => {
  const {device, graph, props} = createFixture([3]);
  const prepared = new GPUSpatialJoinPrepared({geometry: props.polygons, edgeSlabs: true});
  try {
    graph.add(prepared);
    const join = new GPUPointInPolygonJoin({...props, prepared, edgeSlabs: true});
    const buffer = prepared.edgeSlabStorage!.slabs.slabEntries.buffer;
    Object.defineProperty(buffer, 'usage', {configurable: true, value: Buffer.COPY_DST});
    const allocation = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => join.getCommandNodes(graph)).toThrow(/prepared.slabEntries.*STORAGE/);
    expect(allocation).not.toHaveBeenCalled();
    // The same optional prepared slabs are irrelevant when this join selects the direct edge walk.
    expect(() =>
      new GPUPointInPolygonJoin({
        ...props,
        id: 'without-slabs',
        prepared,
        edgeSlabs: false
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    prepared.destroy();
    device.destroy();
  }
});

it('point joins permit unused empty assignment views to alias live feature-count storage', () => {
  const {device, graph, props} = createFixture([0]);
  try {
    const pointFeatureIds = graph.createDataView(props.output.polygonAssignmentCounts!.buffer, {
      format: 'uint32',
      length: 0
    });
    expect(() =>
      new GPUPointInPolygonJoin({
        ...props,
        output: {...props.output, pointFeatureIds}
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('point joins reject mixed physical feature-ID bindings while ignoring them for empty queries', () => {
  for (const pointCount of [0, 1]) {
    const {device, graph, props} = createFixture([pointCount]);
    const storage = createTransientView(graph, 'feature-id-alias', 'uint32', 128);
    const pointFeatureIds = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: pointCount
    });
    const sourceIds = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 256
    });
    const create = () =>
      new GPUPointInPolygonJoin({
        ...props,
        polygons: {...props.polygons, sourceIds},
        output: {
          ...props.output,
          pointFeatureIds,
          polygonAssignmentCounts:
            pointCount === 0
              ? graph.createDataView(storage.buffer, {format: 'uint32', length: 1})
              : props.output.polygonAssignmentCounts
        }
      });
    try {
      if (pointCount === 0) expect(() => create().getCommandNodes(graph)).not.toThrow();
      else expect(create).toThrow(/physical buffer.*read-only and writable/);
    } finally {
      device.destroy();
    }
  }
});
