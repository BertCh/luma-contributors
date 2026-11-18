// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUBufferSelection,
  type GPUBufferSelectionProps
} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {
  GPUSpatialJoinPrepared,
  type GPUSpatialJoinPreparedStorage
} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {
  createNullWebGPUDevice,
  createVectorView,
  getParsedKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {getReservedWGSLWords} from './wgsl-reserved-words';
import {getChunkDispatches} from './chunk-dispatch-test-utils';
import {GPUSpatialJoinCandidates} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-candidates';
import {validateSpatialJoinBindings} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-binding-preflight';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', 1);
  const outputMask = createTransientView(graph, 'mask', 'uint32', 1);
  const props: GPUBufferSelectionProps = {
    id: 'binding-selection',
    points: {kind: 'points', positions},
    features: {kind: 'points', positions: createTransientView(graph, 'features', 'float32x2', 1)},
    distance: createTransientView(graph, 'distance', 'float32', 1),
    candidateCapacity: 1,
    spatialSort: false,
    querySort: false,
    output: {outputMask}
  };
  return {device, graph, props};
}

function recordDeclarations(graph: GPUCommandGraph) {
  const spies = [
    vi.spyOn(graph, 'createTransientBuffer'),
    vi.spyOn(graph, 'createDataView'),
    vi.spyOn(graph, 'importBuffer'),
    vi.spyOn(graph, 'add'),
    vi.spyOn(graph.device, 'createBuffer')
  ];
  return {
    expectUnchanged() {
      for (const spy of spies) expect(spy).not.toHaveBeenCalled();
    },
    restore() {
      for (const spy of spies) spy.mockRestore();
    }
  };
}

for (const kind of ['selection', 'prepared', 'candidates'] as const) {
  for (const defect of ['aligned range', 'STORAGE usage', 'scratch', 'sort scratch'] as const) {
    if (kind === 'candidates' && defect === 'sort scratch') continue;
    it(`${kind} preflights ${defect} without declaring graph/device resources and permits retry`, () => {
      const {device, graph, props} = createFixture();
      const limits = device.limits;
      let positions = props.points.positions as GraphDataView<'float32x2'>;
      let distance = props.distance;
      if (defect === 'aligned range') {
        const backing = createTransientView(graph, 'offset-input', 'float32', 32);
        positions = graph.createDataView(backing.buffer, {
          format: 'float32x2',
          length: 1,
          byteOffset: 60
        });
        distance = graph.createDataView(backing.buffer, {
          format: 'float32',
          length: 1,
          byteOffset: 60
        });
      }
      if (defect === 'STORAGE usage') {
        const handle = graph.createTransientBuffer({
          id: 'not-storage',
          byteLength: 8,
          usage: Buffer.COPY_DST
        });
        positions = graph.createDataView(handle, {format: 'float32x2', length: 1});
      }
      const sortedPositions = createTransientView(graph, 'sort-points', 'float32x2', 2);
      const contributor =
        kind === 'selection'
          ? new GPUBufferSelection({
              ...props,
              points: {kind: 'points', positions},
              distance,
              candidateCapacity: defect === 'scratch' ? 64 : 1,
              ...(defect === 'sort scratch'
                ? {features: {kind: 'points', positions: sortedPositions}, spatialSort: true}
                : {})
            })
          : kind === 'candidates'
            ? new GPUSpatialJoinCandidates({
                id: 'binding-candidates',
                left: {kind: 'points', positions},
                right: {kind: 'points', positions: sortedPositions},
                distance,
                leafCapacity: defect === 'scratch' ? 16 : undefined,
                output: {
                  leftRows: createTransientView(graph, 'left-ids', 'uint32', 1),
                  rightRows: createTransientView(graph, 'right-ids', 'uint32', 1),
                  status: {
                    count: createTransientView(graph, 'pair-count', 'uint32', 1),
                    overflow: createTransientView(graph, 'pair-overflow', 'uint32', 1)
                  }
                }
              })
            : new GPUSpatialJoinPrepared({
                id: 'binding-prepared',
                geometry: {
                  kind: 'points',
                  positions: defect === 'sort scratch' ? sortedPositions : positions
                },
                leafCapacity: defect === 'scratch' ? 16 : undefined,
                spatialSort: defect === 'sort scratch'
              });
      Object.defineProperty(device, 'limits', {
        configurable: true,
        value: {
          ...limits,
          maxStorageBufferBindingSize:
            defect === 'aligned range' ? 64 : defect === 'sort scratch' ? 32 : 128
        }
      });
      const declarations = recordDeclarations(graph);
      try {
        expect(() => contributor.getCommandNodes(graph)).toThrow(
          defect === 'STORAGE usage'
            ? /STORAGE/
            : defect === 'aligned range'
              ? /aligned|binding/
              : defect === 'sort scratch'
                ? /radix histogram/
                : /scratch|BVH/
        );
        declarations.expectUnchanged();
        if (contributor instanceof GPUSpatialJoinPrepared)
          expect(contributor.isDeclaredIn(graph)).toBe(false);
        Object.defineProperty(device, 'limits', {value: limits});
        if (defect === 'STORAGE usage')
          Object.defineProperty(positions.buffer, 'usage', {
            configurable: true,
            value: Buffer.STORAGE | Buffer.COPY_DST
          });
        expect(() => contributor.getCommandNodes(graph)).not.toThrow();
      } finally {
        declarations.restore();
        if (contributor instanceof GPUSpatialJoinPrepared) contributor.destroy();
        device.destroy();
      }
    });
  }
}

it('selection preflights the great-circle traversal node table before adapter allocations', () => {
  const {device, graph, props} = createFixture();
  const selection = new GPUBufferSelection({
    ...props,
    points: {
      kind: 'points',
      positions: createTransientView(graph, 'three-queries', 'float32x2', 3)
    },
    output: {outputMask: createTransientView(graph, 'three-mask', 'uint32', 3)},
    leafCapacity: 2,
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
  });
  const original = device.limits;
  Object.defineProperty(device, 'limits', {
    configurable: true,
    value: {...original, maxStorageBufferBindingSize: 64}
  });
  const declarations = recordDeclarations(graph);
  try {
    expect(() => selection.getCommandNodes(graph)).toThrow(/nodeData|topk|node table/);
    declarations.expectUnchanged();
    Object.defineProperty(device, 'limits', {value: original});
    expect(() => selection.getCommandNodes(graph)).not.toThrow();
  } finally {
    declarations.restore();
    device.destroy();
  }
});

function createPolygon(graph: GPUCommandGraph, polygonOffsetCount = 2) {
  return {
    kind: 'polygons' as const,
    positions: createTransientView(graph, 'vertices', 'float32x2', 4),
    featureOffsets: createTransientView(graph, 'feature-offsets', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygon-offsets', 'uint32', polygonOffsetCount),
    ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 2)
  };
}

for (const defect of ['owner parameters', 'persistent entries'] as const) {
  it(`prepared preflights nested edge-slab ${defect} before persistent allocation`, () => {
    const {device, graph} = createFixture();
    const geometry = createPolygon(graph, defect === 'owner parameters' ? 10 : 2);
    const prepared = new GPUSpatialJoinPrepared({geometry, edgeSlabs: true});
    Object.defineProperty(device, 'limits', {
      value: {
        ...device.limits,
        maxStorageBufferBindingSize: 64,
        maxBufferSize: defect === 'persistent entries' ? 48 : device.limits.maxBufferSize
      }
    });
    const declarations = recordDeclarations(graph);
    try {
      expect(() => prepared.getCommandNodes(graph)).toThrow(
        defect === 'owner parameters' ? /edge slab parameters/ : /edge slab entries.*maxBufferSize/
      );
      declarations.expectUnchanged();
      expect(prepared.isDeclaredIn(graph)).toBe(false);
    } finally {
      declarations.restore();
      prepared.destroy();
      device.destroy();
    }
  });
}

it('prepared accepts a zero optional slab-entry budget using the shared exact fallback', () => {
  const {device, graph} = createFixture();
  const geometry = {
    ...createPolygon(graph),
    positions: createTransientView(graph, 'six-vertices', 'float32x2', 6)
  };
  const prepared = new GPUSpatialJoinPrepared({geometry, edgeSlabs: true});
  Object.defineProperty(device, 'limits', {
    // Four nonempty snapshot views need 48 dispatch bytes; six vertices use all six 8-byte
    // slab slots, leaving zero optional entries without invalidating the snapshot binding.
    value: {...device.limits, maxStorageBufferBindingSize: 48}
  });
  try {
    const sources = getParsedKernelSources(prepared.getCommandNodes(graph));
    for (const source of sources.values()) expect(getReservedWGSLWords(source)).toEqual([]);
    expect(prepared.edgeSlabStorage!.slabs.entryCapacity).toBe(0);
  } finally {
    prepared.destroy();
    device.destroy();
  }
});

it('selection rejects physical read-only/write aliasing even for disjoint aligned subviews', () => {
  const {device, graph, props} = createFixture();
  const backing = createTransientView(graph, 'shared-outputs', 'uint32', 128);
  const nearestFeatureRows = graph.createDataView(backing.buffer, {format: 'uint32', length: 1});
  const outputMask = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 256
  });
  try {
    expect(
      () => new GPUBufferSelection({...props, output: {nearestFeatureRows, outputMask}})
    ).toThrow(/physical buffer.*read-only and writable/);
  } finally {
    device.destroy();
  }
});

it('binding preflight rejects duplicate imports of one physical buffer before graph declarations', () => {
  const {device, graph, props} = createFixture();
  const buffer = device.createBuffer({byteLength: 512, usage: Buffer.STORAGE});
  const positions = importGraphBuffer(graph, 'imported-position', buffer, 'float32x2', 1);
  const radius = importGraphBuffer(graph, 'imported-radius', buffer, 'float32', 1);
  const selection = new GPUBufferSelection({
    ...props,
    points: {kind: 'points', positions},
    distance: radius
  });
  const declarations = recordDeclarations(graph);
  try {
    expect(() => selection.getCommandNodes(graph)).toThrow(/physical buffer must be imported once/);
    declarations.expectUnchanged();
  } finally {
    declarations.restore();
    buffer.destroy();
    device.destroy();
  }
});

it('prepared accepts packed fused storage and rejects a mixed-access packed level hierarchy', () => {
  const {device, graph, props} = createFixture();
  const backing = createTransientView(graph, 'packed-tree', 'uint32', 512);
  const pair = (byteOffset: number) =>
    graph.createDataView(backing.buffer, {format: 'float32x2', length: 3, byteOffset});
  const word = (byteOffset: number, length = 1) =>
    graph.createDataView(backing.buffer, {format: 'uint32', length, byteOffset});
  const storage: GPUSpatialJoinPreparedStorage = {
    nodeMinima: pair(0),
    nodeMaxima: pair(256),
    nodeChildren: graph.createDataView(backing.buffer, {
      format: 'uint32x2',
      length: 3,
      byteOffset: 1280
    }),
    leafIds: word(512, 2),
    count: word(768),
    overflow: word(1024)
  };
  const prepared = new GPUSpatialJoinPrepared({
    geometry: props.features as {kind: 'points'; positions: GraphDataView<'float32x2'>},
    leafCapacity: 2,
    storage
  });
  try {
    expect(() => prepared.getCommandNodes(graph)).not.toThrow();
    const large = createTransientView(graph, 'large-tree', 'uint32', 4096).buffer;
    const largeStorage: GPUSpatialJoinPreparedStorage = {
      nodeMinima: graph.createDataView(large, {format: 'float32x2', length: 511}),
      nodeMaxima: graph.createDataView(large, {format: 'float32x2', length: 511, byteOffset: 4096}),
      nodeChildren: graph.createDataView(large, {
        format: 'uint32x2',
        length: 511,
        byteOffset: 8192
      }),
      leafIds: graph.createDataView(large, {format: 'uint32', length: 256, byteOffset: 12288}),
      count: graph.createDataView(large, {format: 'uint32', length: 1, byteOffset: 13568}),
      overflow: graph.createDataView(large, {format: 'uint32', length: 1, byteOffset: 13824})
    };
    expect(
      () =>
        new GPUSpatialJoinPrepared({
          geometry: prepared.geometry,
          leafCapacity: 256,
          storage: largeStorage
        })
    ).toThrow(/physical buffer.*read-only and writable/);
  } finally {
    prepared.destroy();
    device.destroy();
  }
});

for (const defect of ['aligned range', 'STORAGE usage'] as const) {
  it(`prepared preflights caller-owned tree ${defect} before persistent or transient declarations`, () => {
    const {device, graph, props} = createFixture();
    const backing = graph.createTransientBuffer({
      id: 'caller-children',
      byteLength: 128,
      usage: defect === 'STORAGE usage' ? Buffer.COPY_DST : Buffer.STORAGE
    });
    const storage: GPUSpatialJoinPreparedStorage = {
      nodeMinima: createTransientView(graph, 'minima', 'float32x2', 1),
      nodeMaxima: createTransientView(graph, 'maxima', 'float32x2', 1),
      nodeChildren: graph.createDataView(backing, {
        format: 'uint32x2',
        length: 1,
        byteOffset: defect === 'aligned range' ? 60 : 0
      }),
      leafIds: createTransientView(graph, 'leaves', 'uint32', 1),
      count: createTransientView(graph, 'count', 'uint32', 1),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1)
    };
    const prepared = new GPUSpatialJoinPrepared({
      geometry: {kind: 'points', positions: props.points.positions as GraphDataView<'float32x2'>},
      storage
    });
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 64}
    });
    const declarations = recordDeclarations(graph);
    try {
      expect(() => prepared.getCommandNodes(graph)).toThrow(
        defect === 'STORAGE usage'
          ? /storage.nodeChildren.*STORAGE/
          : /storage.nodeChildren.*binding/
      );
      declarations.expectUnchanged();
      expect(prepared.isDeclaredIn(graph)).toBe(false);
    } finally {
      declarations.restore();
      prepared.destroy();
      device.destroy();
    }
  });
}

it('selection validates compact-output STORAGE usage before building its nearest adapter', () => {
  const {device, graph, props} = createFixture();
  const output = graph.createTransientBuffer({
    id: 'invalid-compact-output',
    byteLength: 16,
    usage: Buffer.COPY_SRC
  });
  const selection = new GPUBufferSelection({
    ...props,
    output: {
      matches: {
        ids: graph.createDataView(output, {format: 'uint32', length: 4}),
        status: {
          count: createTransientView(graph, 'selected-count', 'uint32', 1),
          overflow: createTransientView(graph, 'selected-overflow', 'uint32', 1)
        }
      }
    }
  });
  const declarations = recordDeclarations(graph);
  try {
    expect(() => selection.getCommandNodes(graph)).toThrow(/STORAGE/);
    declarations.expectUnchanged();
  } finally {
    declarations.restore();
    device.destroy();
  }
});

it('selection ignores planar candidate storage in a great-circle context', () => {
  const {device, graph, props} = createFixture();
  const selection = new GPUBufferSelection({
    ...props,
    candidateCapacity: 0xffffffff,
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
  });
  try {
    const nodes = selection.getCommandNodes(graph);
    expect(nodes.some(node => node.id === 'binding-selection-join-traverse')).toBe(true);
    expect(nodes.some(node => node.id.includes('probe'))).toBe(false);
  } finally {
    device.destroy();
  }
});

it('selection mask pipelines stay constant across uneven chunks and use each chunk workgroup count', () => {
  const cardinalities: number[][] = [];
  for (const lengths of [[257], [1, 255, 1]]) {
    const {device, graph, props} = createFixture();
    const points = createVectorView(
      'query-chunks',
      'float32x2',
      lengths.map((length, index) =>
        createTransientView(graph, `query-${index}`, 'float32x2', length)
      )
    );
    const outputMask = createVectorView(
      'mask-chunks',
      'uint32',
      lengths.map((length, index) =>
        createTransientView(graph, `output-mask-${index}`, 'uint32', length)
      )
    );
    try {
      const nodes = new GPUBufferSelection({
        ...props,
        points: {kind: 'points', positions: points},
        output: {outputMask}
      }).getCommandNodes(graph);
      const masks = nodes.filter(node => /binding-selection-mask(?:-chunk-\d+)?$/.test(node.id));
      const sources = getParsedKernelSources(masks);
      cardinalities.push([
        new Set(sources.values()).size,
        getGPUCommandNodePipelineKeys(masks).size
      ]);
      for (const source of sources.values()) expect(getReservedWGSLWords(source)).toEqual([]);
      expect(getChunkDispatches(device, masks)).toEqual(
        lengths.map(length => [Math.ceil(length / 256), 1, 1])
      );
    } finally {
      device.destroy();
    }
  }
  expect(cardinalities).toEqual([
    [1, 1],
    [1, 1]
  ]);
});

it('binding preflight bounds flat word and copy-count addresses even on a larger future device', () => {
  const {device, graph} = createFixture();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 2 ** 36, maxBufferSize: 2 ** 36}
  });
  const declarations = recordDeclarations(graph);
  try {
    expect(() =>
      validateSpatialJoinBindings(
        graph,
        'word-address',
        {},
        {
          'copy count': 4 * 2 ** 32
        }
      )
    ).toThrow(/word-address.*copy count.*SPATIAL_JOIN_MAXIMUM_STORAGE_WORD_COUNT/);
    declarations.expectUnchanged();
  } finally {
    declarations.restore();
    device.destroy();
  }
});
