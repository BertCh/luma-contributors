// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  getPointCloudKernelNodes,
  validatePointCloudPackedViews
} from '../../src/pointcloud-analysis/operations/pointcloud-kernel-nodes';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPU_POINT_CLOUD_MAXIMUM_GRID_AXIS_CELLS,
  GPUPointCloudChange,
  GPUPointCloudSMRF,
  GPUPointCloudStatisticalFilter,
  GPUPointCloudNormalScale,
  getGPUPointCloudGridParameterValues,
  GPUPointCloudFeatures,
  GPUPointCloudGround,
  GPUPointCloudICP,
  GPUPointCloudVoxelMerge,
  GPUPointCloudHeight,
  GPUPointCloudNeighborhoods,
  GPUPointCloudSurfaces,
  getGPUPointCloudHeightParameterValues,
  getGPUPointCloudNeighborhoodsParameterValues,
  getGPUPointCloudSurfacesParameterValues
} from '../../src/pointcloud-analysis';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../utils/gpu-contributor-test-utils';
import {
  auditWGSLModule,
  WGSL_RESERVED_WORDS,
  stripWGSLComments
} from '../analysis-core/kernel-audit/kernel-audit-wgsl';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
afterEach(() => vi.restoreAllMocks());

it('checks shared declared point-cloud ports before layout and alias validation', () => {
  const graph = new GPUCommandGraph(device);
  const squaredDistances = createTransientView(graph, 'wrong-distance-port', 'uint32', 3);
  const counts = createTransientView(graph, 'count-port', 'uint32', 3);
  const descriptors = {
    squaredDistances: {required: true, viewType: 'data' as const, formats: ['float32' as const]},
    counts: {required: true, viewType: 'data' as const, formats: ['uint32' as const]}
  };
  expect(() =>
    validatePointCloudPackedViews('lidar', {squaredDistances, counts}, {}, [], descriptors)
  ).toThrow(/lidar squaredDistances.*float32.*received uint32/);
  expect(() =>
    validatePointCloudPackedViews(
      'lidar',
      {squaredDistances: createTransientView(graph, 'distance-port', 'float32', 3)},
      {},
      [],
      descriptors
    )
  ).toThrow(/lidar counts.*uint32.*received undefined/);
});
const GRID = {
  origin: [0, 0] as const,
  cellSize: 1,
  width: 3,
  height: 1,
  rowDirection: 'north' as const
};
let sequence = 0;
function createHeightProps(graph: GPUCommandGraph, lengths: number[], padded = false) {
  const identifier = sequence++;
  const rows = <Format extends 'float32x3' | 'float32' | 'uint32'>(name: string, format: Format) =>
    createVectorView(
      name,
      format,
      lengths.map((length, chunkIndex) => {
        const byteOffset = padded ? (chunkIndex % 63) * 4 : 0;
        const components = format === 'float32x3' ? 3 : 1;
        const backing = createTransientView(
          graph,
          `${identifier}-${name}-${chunkIndex}`,
          'uint32',
          Math.max(length, 1) * components + byteOffset / 4
        );
        return graph.createDataView(backing.buffer, {format, length, byteOffset});
      })
    );
  return {
    id: 'runtime-height',
    grid: GRID,
    positions: rows('positions', 'float32x3'),
    terrain: createTransientView(graph, `terrain-${identifier}`, 'float32', 3),
    coverage: createTransientView(graph, `coverage-${identifier}`, 'uint32', 3),
    parameters: createTransientView(graph, `parameters-${identifier}`, 'uint32', 5),
    interpolation: {method: 'idw' as const, radius: 1},
    output: {heights: rows('heights', 'float32'), status: rows('status', 'uint32')}
  };
}

it('packs exact uint32 settings and rejects IDW powers that round to zero', () => {
  expect([...getGPUPointCloudSurfacesParameterValues({groundClass: 0xffffffff})]).toEqual([
    0xffffffff
  ]);
  const words = getGPUPointCloudHeightParameterValues(
    {
      coverageBit: 0x80000000,
      interpolation: {method: 'idw', radius: 2, power: 4, minimumSupport: 0xffffffff}
    },
    GRID
  );
  expect(words[0]).toBe(0x80000000);
  expect([...new Float32Array(words.buffer).slice(1, 3)]).toEqual([4, 2]);
  expect([...words.slice(3)]).toEqual([0xffffffff, 3]);
  const graph = new GPUCommandGraph(device);
  const props = createHeightProps(graph, [1]);
  for (const power of [1e-50, NaN, Infinity, 0, -1, 9]) {
    const interpolation = {method: 'idw' as const, radius: 1, power};
    expect(() => getGPUPointCloudHeightParameterValues({interpolation}, GRID)).toThrow(
      /interpolation.power/
    );
    expect(
      () =>
        new GPUPointCloudHeight({
          ...props,
          interpolation
        })
    ).toThrow(/^runtime-height interpolation.power/);
  }
  expect(() =>
    getGPUPointCloudHeightParameterValues({interpolation: {method: 'idw', radius: 65}}, GRID)
  ).toThrow(/interpolation.radius in cells/);
  expect(
    () =>
      new GPUPointCloudHeight({
        ...props,
        parameters: createTransientView(graph, 'short', 'uint32', 4)
      })
  ).toThrow(/^runtime-height parameters/);
});

it('validates runtime radius squares and float32 support extents', () => {
  expect([
    ...getGPUPointCloudNeighborhoodsParameterValues({
      radius: 2,
      supportBounds: {minimum: [-1, -2, -3], maximum: [1, 2, 3]}
    })
  ]).toEqual([2, 4, -1, -2, -3, 1, 2, 3, 1]);
  for (const radius of [0, -1, 1e-30, 1e30, NaN, Infinity])
    expect(() => getGPUPointCloudNeighborhoodsParameterValues({radius})).toThrow(/radius/);
  expect(() =>
    getGPUPointCloudNeighborhoodsParameterValues({
      radius: 1,
      supportBounds: {minimum: [1, 0, 0], maximum: [1 + 1e-10, 1, 1]}
    })
  ).toThrow(/supportBounds/);
});

it('validates chunk topology and logical output overlap while allowing consecutive subviews', () => {
  const graph = new GPUCommandGraph(device);
  const props = createHeightProps(graph, [1, 1]);
  const backing = createTransientView(graph, 'consecutive', 'float32', 2);
  const first = graph.createDataView(backing.buffer, {format: 'float32', length: 1});
  const second = graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 1,
    byteOffset: 4
  });
  expect(
    () =>
      new GPUPointCloudHeight({
        ...props,
        output: {
          ...props.output,
          heights: createVectorView('consecutive', 'float32', [first, second])
        }
      })
  ).not.toThrow();
  expect(
    () =>
      new GPUPointCloudHeight({
        ...props,
        output: {...props.output, heights: createVectorView('overlap', 'float32', [first, first])}
      })
  ).toThrow(/byte ranges must not overlap/);
  const status = graph.createDataView(backing.buffer, {format: 'uint32', length: 1, byteOffset: 4});
  expect(
    () =>
      new GPUPointCloudHeight({
        ...createHeightProps(graph, [1]),
        output: {heights: first, status}
      })
  ).toThrow(/byte ranges must not overlap/);
  expect(
    () =>
      new GPUPointCloudHeight({
        ...props,
        output: {...props.output, heights: createHeightProps(graph, [2]).output.heights}
      })
  ).toThrow(/chunk topology/);
});

it('preflights later aligned binding ranges before any graph mutation', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUPointCloudHeight(createHeightProps(graph, [1, 11_184_811], true));
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /^runtime-height positions chunk 1 needs 134217736 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
});

it('supports 30 million height queries and skips empty chunks', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUPointCloudHeight(
    createHeightProps(graph, [0, ...new Array(10).fill(3_000_000), 0])
  );
  expect(contributor.getCommandNodes(graph)).toHaveLength(11);
  expect(new GPUPointCloudHeight(createHeightProps(graph, [0])).getCommandNodes(graph)).toEqual([]);
});

it('keeps one height shader and pipeline as chunk count, length and padding grow', () => {
  const pipelineKeys = new Set<string>();
  for (const chunkCount of [1, 3, 100]) {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUPointCloudHeight(
      createHeightProps(
        graph,
        Array.from({length: chunkCount}, (_, index) => [1, 257, 513][index % 3]),
        true
      )
    );
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(getParsedKernelSources(nodes).size).toBe(1);
    for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
      pipelineKeys.add(key);
    const dispatches = nodes.filter(node => node.workload?.maximumInvocationCount !== undefined);
    expect(dispatches.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
      Array.from({length: chunkCount}, (_, index) => [1, 2, 3][index % 3])
    );
    for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
      expect(Object.keys(constants)).toHaveLength(0);
      expect(
        (stripWGSLComments(source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(identifier =>
          WGSL_RESERVED_WORDS.has(identifier)
        )
      ).toEqual([]);
    }
  }
  expect(pipelineKeys.size).toBe(1);
});

function createPackedProps(graph: GPUCommandGraph) {
  const identifier = sequence++;
  const view = <Format extends 'uint32' | 'float32' | 'float32x3'>(
    format: Format,
    length: number
  ): GraphDataView<Format> =>
    createTransientView(graph, `packed-${identifier}-${sequence++}`, format, length);
  return {view, positions: view('float32x3', 4)};
}

it('preflights indexed scratch before allocating an index and rejects aliased outputs at construction', () => {
  const graph = new GPUCommandGraph(device);
  const {view, positions} = createPackedProps(graph);
  const props = {
    positions,
    classifications: view('uint32', 4),
    grid: GRID,
    output: {
      terrain: view('float32', 3),
      surface: view('float32', 3),
      canopy: view('float32', 3),
      coverage: view('uint32', 3)
    }
  };
  expect(
    () =>
      new GPUPointCloudSurfaces({
        ...props,
        output: {...props.output, canopy: props.output.surface}
      })
  ).toThrow(/byte ranges must not overlap/);
  const cells = 33_554_432;
  const contributor = new GPUPointCloudSurfaces({
    ...props,
    grid: {...GRID, width: 8192, height: 4096},
    output: {
      ...props.output,
      terrain: view('float32', cells),
      surface: view('float32', cells),
      canopy: view('float32', cells),
      coverage: view('uint32', cells)
    }
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /grid cell offsets needs 134217732 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
});

it('audits runtime neighbourhood and surface shaders within eight baseline bindings', () => {
  for (const strategy of ['serial', 'cooperative'] as const) {
    const graph = new GPUCommandGraph(device);
    const {view, positions} = createPackedProps(graph);
    const status = view('uint32', 4);
    const neighborhoods = {
      positions,
      queryRows: view('uint32', 4),
      radius: 1,
      capacity: 2,
      strategy,
      grid: {origin: [0, 0, 0] as const, cellSize: 1, dimensions: [2, 2, 2] as const},
      parameters: view('float32', 9),
      output: {
        indices: view('uint32', 8),
        squaredDistances: view('float32', 8),
        counts: view('uint32', 4),
        status
      }
    };
    expect(
      () =>
        new GPUPointCloudNeighborhoods({
          ...neighborhoods,
          output: {...neighborhoods.output, counts: status}
        })
    ).toThrow(/byte ranges must not overlap/);
    const nodes = [
      ...new GPUPointCloudNeighborhoods(neighborhoods).getCommandNodes(graph),
      ...new GPUPointCloudSurfaces({
        positions,
        classifications: view('uint32', 4),
        grid: GRID,
        parameters: view('uint32', 1),
        output: {
          terrain: view('float32', 3),
          surface: view('float32', 3),
          canopy: view('float32', 3),
          coverage: view('uint32', 3)
        }
      }).getCommandNodes(graph)
    ];
    getParsedKernelSources(nodes);
    for (const {source, constants} of getGPUCommandNodeKernelSources(nodes))
      expect(auditWGSLModule(source, constants)).toEqual([]);
    for (const {source} of getGPUCommandNodeKernelSources(nodes))
      expect(
        (stripWGSLComments(source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(identifier =>
          WGSL_RESERVED_WORDS.has(identifier)
        )
      ).toEqual([]);
  }
});

it('checks all M3C2 grid scratch before an alignment resource or node is declared', () => {
  const graph = new GPUCommandGraph(device);
  const {view, positions} = createPackedProps(graph);
  const props = {
    reference: positions,
    comparison: positions,
    cores: positions,
    normals: positions,
    alignment: {transform: view('float32', 16)},
    radius: 1,
    halfDepth: 1,
    grid: {origin: [0, 0, 0] as const, dimensions: [8192, 4096, 1] as const, cellSize: 1},
    output: {
      changes: view('float32x3', 4),
      roughness: view('float32x3', 4),
      counts: view('uint32', 8),
      status: view('uint32', 4)
    }
  };
  const contributor = new GPUPointCloudChange(props);
  expect(
    () =>
      new GPUPointCloudChange({
        ...props,
        output: {...props.output, roughness: props.output.changes}
      })
  ).toThrow(/byte ranges must not overlap/);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const add = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /grid cell offsets needs 134217732 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
});

it('checks aligned ground raster ranges before allocating morphology scratch', () => {
  const graph = new GPUCommandGraph(device);
  const {view, positions} = createPackedProps(graph);
  const cells = 33_554_432;
  const backing = view('float32', cells + 1);
  const props = {
    positions,
    classifications: view('uint32', 4),
    grid: {...GRID, width: 8192, height: 4096},
    radii: [1],
    initialThreshold: 0,
    slope: 0,
    maximumThreshold: 0,
    minimumSurface: graph.createDataView(backing.buffer, {
      format: 'float32',
      length: cells,
      byteOffset: 4
    }),
    coverage: view('uint32', cells),
    output: {
      groundMask: view('uint32', 4),
      outputClassifications: view('uint32', 4),
      status: view('uint32', 4)
    }
  };
  const contributor = new GPUPointCloudGround(props);
  expect(
    () =>
      new GPUPointCloudGround({
        ...props,
        output: {...props.output, status: props.output.groundMask}
      })
  ).toThrow(/byte ranges must not overlap/);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /minimumSurface chunk 0 needs 134217732 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
});

it('checks derived ICP moments against device binding size before allocating them', () => {
  const smallDevice = createNullWebGPUDevice();
  Object.defineProperty(smallDevice, 'limits', {
    value: {...smallDevice.limits, maxStorageBufferBindingSize: 64}
  });
  try {
    const graph = new GPUCommandGraph(smallDevice);
    const {view, positions} = createPackedProps(graph);
    const props = {
      source: positions,
      target: positions,
      maximumDistance: 1,
      maximumIterations: 1,
      output: {
        transform: view('float32', 16),
        history: view('float32x3', 1),
        inlierMask: view('uint32', 4),
        status: view('uint32', 1)
      }
    };
    const contributor = new GPUPointCloudICP(props);
    expect(
      () =>
        new GPUPointCloudICP({
          ...props,
          output: {
            ...props.output,
            inlierMask: graph.createDataView(props.output.transform.buffer, {
              format: 'uint32',
              length: 4
            })
          }
        })
    ).toThrow(/byte ranges must not overlap/);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/ICP block moments needs 72 bytes/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    smallDevice.destroy();
  }
});

it('checks voxel inputs and output binding ranges before allocating merge intermediates', () => {
  const graph = new GPUCommandGraph(device);
  const {view} = createPackedProps(graph);
  const rows = 11_184_811;
  const props = {
    partitionPolicy: 'disjoint-owned-rows' as const,
    parts: [
      {
        counts: view('uint32', rows),
        centroids: view('float32x3', rows),
        representatives: view('uint32', rows),
        sourceRowOffset: 0
      }
    ],
    output: {
      counts: view('uint32', rows),
      centroids: view('float32x3', rows),
      representatives: view('uint32', rows),
      status: view('uint32', rows)
    }
  };
  const contributor = new GPUPointCloudVoxelMerge(props);
  expect(
    () =>
      new GPUPointCloudVoxelMerge({
        ...props,
        output: {...props.output, status: props.output.counts}
      })
  ).toThrow(/byte ranges must not overlap/);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /centroids chunk 0 needs 134217732 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
});

it('keeps runtime numeric settings out of neighborhood, height and surface pipelines', () => {
  const keysByOperation = new Map<string, Set<string>>();
  for (const radius of [0.5, 2, 3]) {
    const graph = new GPUCommandGraph(device);
    const {view, positions} = createPackedProps(graph);
    const neighborhoods = new GPUPointCloudNeighborhoods({
      positions,
      radius,
      capacity: 2,
      parameters: view('float32', 9),
      grid: {origin: [-4, -4, -4], cellSize: 1, dimensions: [8, 8, 8]},
      output: {
        indices: view('uint32', 8),
        squaredDistances: view('float32', 8),
        counts: view('uint32', 4),
        status: view('uint32', 4)
      }
    });
    const surface = new GPUPointCloudSurfaces({
      positions,
      classifications: view('uint32', 4),
      groundClass: radius === 2 ? 0xffffffff : 2,
      grid: GRID,
      parameters: view('uint32', 1),
      output: {
        terrain: view('float32', 3),
        surface: view('float32', 3),
        canopy: view('float32', 3),
        coverage: view('uint32', 3)
      }
    });
    const height = new GPUPointCloudHeight({
      ...createHeightProps(graph, [1, 257], true),
      interpolation: {method: 'idw', radius, power: radius, minimumSupport: radius === 2 ? 3 : 1}
    });
    for (const [name, contributor] of [
      ['neighborhoods', neighborhoods],
      ['surface', surface],
      ['height', height]
    ] as const) {
      const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).filter(
        kernel => kernel.operation === 'GPUPointCloud'
      );
      const signatures = kernels.map(
        kernel => `${kernel.source}\n${JSON.stringify(kernel.constants)}`
      );
      const signature = [...new Set(signatures)].sort().join('\n');
      const keys = keysByOperation.get(name) ?? new Set<string>();
      keys.add(signature);
      keysByOperation.set(name, keys);
      for (const kernel of kernels)
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    }
  }
  expect([...keysByOperation.values()].map(keys => keys.size)).toEqual([1, 1, 1]);
});

it('rejects certified support bounds that collapse to equal float32 endpoints', () => {
  const graph = new GPUCommandGraph(device);
  const {view, positions} = createPackedProps(graph);
  expect(
    () =>
      new GPUPointCloudNeighborhoods({
        positions,
        radius: 1,
        capacity: 2,
        supportBounds: {minimum: [1, 0, 0], maximum: [1 + 1e-10, 1, 1]},
        output: {
          indices: view('uint32', 8),
          squaredDistances: view('float32', 8),
          counts: view('uint32', 4),
          status: view('uint32', 4)
        }
      })
  ).toThrow(/supportBounds.*float32/);
});

it('checks each raster axis at its named limit before declaring graph resources', () => {
  const graph = new GPUCommandGraph(device);
  const {view, positions} = createPackedProps(graph);
  const maximum = GPU_POINT_CLOUD_MAXIMUM_GRID_AXIS_CELLS;
  for (const axis of ['width', 'height'] as const) {
    const makeProps = (dimension: number) => ({
      positions,
      classifications: view('uint32', 4),
      grid: {...GRID, width: 1, height: 1, [axis]: dimension},
      output: {
        terrain: view('float32', dimension),
        surface: view('float32', dimension),
        canopy: view('float32', dimension),
        coverage: view('uint32', dimension)
      }
    });
    expect(() => new GPUPointCloudSurfaces(makeProps(maximum))).not.toThrow();
    const props = makeProps(maximum + 1);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const add = vi.spyOn(graph, 'add');
    expect(() => new GPUPointCloudSurfaces(props)).toThrow(new RegExp(`grid.${axis}.*${maximum}`));
    expect(allocate).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    vi.restoreAllMocks();
  }
});

it('accepts logical voxel fields sharing a binding prefix only across separate dispatches', () => {
  const graph = new GPUCommandGraph(device);
  const {view} = createPackedProps(graph);
  const backing = view('uint32', 4);
  const counts = graph.createDataView(backing.buffer, {format: 'uint32', length: 1});
  const centroids = graph.createDataView(backing.buffer, {
    format: 'float32x3',
    length: 1,
    byteOffset: 4
  });
  const parts = [0, 1].map(() => ({
    counts: view('uint32', 1),
    centroids: view('float32x3', 1),
    representatives: view('uint32', 1),
    sourceRowOffset: 0
  }));
  const props = {
    partitionPolicy: 'disjoint-owned-rows' as const,
    parts,
    output: {counts, centroids, representatives: view('uint32', 1), status: view('uint32', 1)}
  };
  expect(new GPUPointCloudVoxelMerge(props).getCommandNodes(graph).length).toBeGreaterThan(0);
  const representatives = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 4
  });
  expect(
    () =>
      new GPUPointCloudVoxelMerge({
        ...props,
        output: {...props.output, centroids: view('float32x3', 1), representatives}
      })
  ).toThrow(/aligned binding byte ranges must not overlap/);
  const overlap = graph.createDataView(backing.buffer, {format: 'float32x3', length: 1});
  expect(
    () =>
      new GPUPointCloudVoxelMerge({
        ...props,
        output: {...props.output, centroids: overlap}
      })
  ).toThrow(/logical byte ranges must not overlap/);
});

it('ignores unused point-to-point normals and permits disjoint ICP outputs across dispatches', () => {
  const graph = new GPUCommandGraph(device);
  const {view, positions} = createPackedProps(graph);
  const backing = view('uint32', 20);
  const props = {
    source: positions,
    target: positions,
    maximumDistance: 1,
    maximumIterations: 1,
    targetNormals: graph.createDataView(backing.buffer, {format: 'float32x3', length: 1}),
    output: {
      transform: graph.createDataView(backing.buffer, {format: 'float32', length: 16}),
      history: graph.createDataView(backing.buffer, {
        format: 'float32x3',
        length: 1,
        byteOffset: 64
      }),
      inlierMask: view('uint32', 4),
      status: view('uint32', 1)
    }
  };
  expect(() => new GPUPointCloudICP(props).getCommandNodes(graph)).not.toThrow();
});

it('keeps ICP shader and pipeline counts fixed across iterations and merge levels', () => {
  for (const metric of ['point-to-point', 'point-to-plane'] as const) {
    const pipelineKeys = new Set<string>();
    for (const iterations of [1, 3, 9]) {
      const graph = new GPUCommandGraph(device);
      const {view} = createPackedProps(graph);
      const contributor = new GPUPointCloudICP({
        source: view('float32x3', 68_000),
        target: view('float32x3', 6),
        targetNormals: view('float32x3', 6),
        metric,
        maximumDistance: 1,
        maximumIterations: iterations,
        output: {
          transform: view('float32', 16),
          history: view('float32x3', iterations),
          inlierMask: view('uint32', 68_000),
          status: view('uint32', 1)
        }
      });
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
        pipelineKeys.add(key);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      const reducers = kernels.filter(kernel => kernel.source.includes('let mergeBegin'));
      expect(reducers).toHaveLength(iterations * 2);
      expect(
        new Set(reducers.map(kernel => `${kernel.source}\n${JSON.stringify(kernel.constants)}`))
          .size
      ).toBe(1);
      for (const kernel of kernels) {
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
        expect(
          (stripWGSLComments(kernel.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(
            identifier => WGSL_RESERVED_WORDS.has(identifier)
          )
        ).toEqual([]);
      }
    }
    // Initialize, match, moments, bounded merge, solve, final mask and publish.
    expect(pipelineKeys.size).toBe(7);
  }
});

it('keeps voxel merge shader and pipeline counts fixed across differently padded parts', () => {
  const pipelineKeys = new Set<string>();
  for (const partCount of [1, 3, 9]) {
    const graph = new GPUCommandGraph(device);
    const {view} = createPackedProps(graph);
    const padded = <Format extends 'uint32' | 'float32x3'>(format: Format, index: number) => {
      const backing = view('uint32', 64);
      return graph.createDataView(backing.buffer, {
        format,
        length: 1,
        byteOffset: (index % 60) * 4
      });
    };
    const contributor = new GPUPointCloudVoxelMerge({
      partitionPolicy: 'disjoint-owned-rows',
      parts: Array.from({length: partCount}, (_, partIndex) => ({
        counts: padded('uint32', partIndex),
        centroids: padded('float32x3', partIndex + 1),
        representatives: padded('uint32', partIndex + 2),
        sourceRowOffset: partIndex * 13
      })),
      output: {
        counts: padded('uint32', 3),
        centroids: padded('float32x3', 5),
        representatives: padded('uint32', 7),
        status: padded('uint32', 9)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
      pipelineKeys.add(key);
    for (const kernel of getGPUCommandNodeKernelSources(nodes))
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
  }
  expect(pipelineKeys.size).toBe(3);
});

it('keeps feature pipelines fixed across strided chunks and accepts separate-pass packed outputs', () => {
  const keys = new Set<string>();
  for (const chunkCount of [1, 3, 9]) {
    const graph = new GPUCommandGraph(device);
    const {view, positions} = createPackedProps(graph);
    const lengths = Array.from({length: chunkCount}, (_, index) => [1, 255, 513][index % 3]);
    const rows = <Format extends 'uint32' | 'float32' | 'float32x3'>(
      format: Format,
      multiplier = 1
    ) =>
      createVectorView(
        `feature-${sequence++}`,
        format,
        lengths.map((length, index) => {
          const components = format === 'float32x3' ? 3 : 1;
          const backing = view('uint32', length * multiplier * components + index + 1);
          return graph.createDataView(backing.buffer, {
            format,
            length: length * multiplier,
            byteOffset: index * 4
          });
        })
      );
    const contributor = new GPUPointCloudFeatures({
      positions,
      capacity: 4,
      counts: rows('uint32'),
      indices: rows('uint32', 4),
      output: {
        normals: rows('float32x3'),
        eigenvalues: rows('float32x3'),
        shapes: rows('float32x3'),
        covariance: rows('float32', 6),
        status: rows('uint32')
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
      keys.add(key);
    for (const kernel of getGPUCommandNodeKernelSources(nodes))
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
  }
  expect(keys.size).toBe(2);
  const graph = new GPUCommandGraph(device);
  const {view, positions} = createPackedProps(graph);
  const backing = view('float32', 6);
  expect(() =>
    new GPUPointCloudFeatures({
      positions,
      capacity: 4,
      counts: view('uint32', 1),
      indices: view('uint32', 4),
      output: {
        normals: graph.createDataView(backing.buffer, {format: 'float32x3', length: 1}),
        shapes: graph.createDataView(backing.buffer, {
          format: 'float32x3',
          length: 1,
          byteOffset: 12
        }),
        eigenvalues: view('float32x3', 1),
        status: view('uint32', 1)
      }
    }).getCommandNodes(graph)
  ).not.toThrow();
});

it('keeps ground pipelines fixed across chunk and morphology round counts', () => {
  const keys = new Set<string>();
  for (const count of [1, 3, 5]) {
    const graph = new GPUCommandGraph(device);
    const {view} = createPackedProps(graph);
    const lengths = Array.from({length: count}, (_, index) => [1, 255, 513][index % 3]);
    const rows = <Format extends 'uint32' | 'float32x3'>(format: Format) =>
      createVectorView(
        `ground-${sequence++}`,
        format,
        lengths.map(length => view(format, length))
      );
    const contributor = new GPUPointCloudGround({
      positions: rows('float32x3'),
      classifications: rows('uint32'),
      grid: {...GRID, width: 9, height: 9},
      radii: Array.from({length: count}, (_, index) => index + 1),
      initialThreshold: 0,
      maximumThreshold: 1,
      slope: 0.1,
      minimumSurface: view('float32', 81),
      coverage: view('uint32', 81),
      output: {
        groundMask: rows('uint32'),
        outputClassifications: rows('uint32'),
        status: rows('uint32')
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
      keys.add(key);
    for (const kernel of getGPUCommandNodeKernelSources(nodes))
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
  }
  expect(keys.size).toBe(5);
});

it('audits runtime grid variants within baseline storage bindings', () => {
  for (const strategy of ['serial', 'cooperative'] as const) {
    const graph = new GPUCommandGraph(device);
    const {view, positions} = createPackedProps(graph);
    const gridParameters = view('float32', 4);
    const nodes = [
      ...new GPUPointCloudNeighborhoods({
        positions,
        queryRows: view('uint32', 4),
        grid: {origin: [0, 0, 0], dimensions: [3, 3, 3], cellSize: 1},
        gridParameters,
        radius: 1,
        parameters: view('float32', 9),
        capacity: 4,
        strategy,
        output: {
          indices: view('uint32', 16),
          squaredDistances: view('float32', 16),
          counts: view('uint32', 4),
          status: view('uint32', 4)
        }
      }).getCommandNodes(graph),
      ...new GPUPointCloudSurfaces({
        positions,
        classifications: view('uint32', 4),
        grid: GRID,
        gridParameters,
        parameters: view('uint32', 1),
        output: {
          terrain: view('float32', 3),
          surface: view('float32', 3),
          canopy: view('float32', 3),
          coverage: view('uint32', 3)
        }
      }).getCommandNodes(graph),
      ...new GPUPointCloudHeight({
        ...createHeightProps(graph, [1, 257], true),
        gridParameters
      }).getCommandNodes(graph),
      ...new GPUPointCloudGround({
        positions,
        classifications: view('uint32', 4),
        grid: {...GRID, width: 9, height: 9},
        gridParameters,
        parameters: view('float32', 2),
        radii: [1, 2],
        initialThreshold: 0,
        maximumThreshold: 1,
        slope: 0.1,
        minimumSurface: view('float32', 81),
        coverage: view('uint32', 81),
        output: {
          groundMask: view('uint32', 4),
          outputClassifications: view('uint32', 4),
          status: view('uint32', 4)
        }
      }).getCommandNodes(graph)
    ];
    getParsedKernelSources(nodes);
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
      expect(
        (stripWGSLComments(kernel.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(
          identifier => WGSL_RESERVED_WORDS.has(identifier)
        )
      ).toEqual([]);
    }
  }
  expect([...getGPUPointCloudGridParameterValues({origin: [1, 2], cellSize: 0.5})]).toEqual([
    1, 2, 0, 0.5
  ]);
  for (const cellSize of [0, -1, 1e-50, 1e40, NaN, Infinity])
    expect(() => getGPUPointCloudGridParameterValues({origin: [0, 0], cellSize})).toThrow(
      /cellSize/
    );
});

it('checks STORAGE usage and separates read-only buffers while allowing packed writable ranges', () => {
  const graph = new GPUCommandGraph(device);
  const props = createHeightProps(graph, [1]);
  const terrain = graph.createDataView(
    graph.importBuffer({id: 'copy-only-terrain', byteLength: 12, usage: Buffer.COPY_DST}),
    {format: 'float32', length: 3}
  );
  const contributor = new GPUPointCloudHeight({
    ...props,
    terrain
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /terrain.*usage must include Buffer.STORAGE/
  );
  expect(allocate).not.toHaveBeenCalled();
  vi.restoreAllMocks();
  const backing = createTransientView(graph, 'packed-layout', 'uint32', 129);
  const packed = {
    ...props,
    output: {
      ...props.output,
      heights: graph.createDataView(backing.buffer, {
        format: 'float32',
        length: 1,
        byteOffset: 256
      }),
      status: graph.createDataView(backing.buffer, {
        format: 'uint32' as const,
        length: 1,
        byteOffset: 512
      })
    }
  };
  expect(() => new GPUPointCloudHeight(packed).getCommandNodes(graph)).not.toThrow();
  expect(
    () =>
      new GPUPointCloudHeight({
        ...packed,
        positions: graph.createDataView(backing.buffer, {format: 'float32x3', length: 1})
      })
  ).toThrow(
    /^runtime-height input positions and writable outputs must not share a physical storage buffer in one dispatch$/
  );
});

it('normalizes extreme finite vectors using exponent-scaled quotients', () => {
  const graph = new GPUCommandGraph(device);
  const {view, positions} = createPackedProps(graph);
  const [kernel] = getGPUCommandNodeKernelSources(
    getPointCloudKernelNodes(graph, {
      id: 'normalization-scales',
      length: positions.length,
      inputs: {positions},
      outputs: {normals: view('float32x3', positions.length)},
      body: 'writeNormals(index, safeNormal(readPositions(index)));'
    })
  );
  expect(kernel.source).toContain('let scaled = getPointCloudQuotient3(value, scale);');
  expect(kernel.source).toContain('let numerator = frexp(value);');
  expect(kernel.source).not.toContain('value / scale');
  expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
  expect(
    (stripWGSLComments(kernel.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(identifier =>
      WGSL_RESERVED_WORDS.has(identifier)
    )
  ).toEqual([]);
});

it('keeps normal-scale pipelines fixed across scale rounds and preflights its nested index', () => {
  const keys = new Set<string>();
  for (const radii of [
    [1, 4],
    [1, 2, 4],
    [1, 2, 3, 4]
  ]) {
    const graph = new GPUCommandGraph(device);
    const {view, positions} = createPackedProps(graph);
    const contributor = new GPUPointCloudNormalScale({
      positions,
      radii,
      capacity: 4,
      parameters: view('float32', 5),
      output: {
        normals: view('float32x3', 4),
        normalScales: view('float32', 4),
        normalVariances: view('float32', 4),
        status: view('uint32', 4)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
      keys.add(key);
    for (const kernel of getGPUCommandNodeKernelSources(nodes))
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
  }
  expect(keys.size).toBe(7);
  const graph = new GPUCommandGraph(device);
  const {view, positions} = createPackedProps(graph);
  const contributor = new GPUPointCloudNormalScale({
    positions,
    radii: [1],
    capacity: 4,
    grid: {origin: [0, 0, 0], dimensions: [8192, 4096, 1], cellSize: 1},
    parameters: view('float32', 5),
    output: {
      normals: view('float32x3', 4),
      normalScales: view('float32', 4),
      normalVariances: view('float32', 4),
      status: view('uint32', 4)
    }
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /grid cell offsets needs 134217732 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
});

it('keeps voxel row-chunk pipelines fixed, including empty and padded chunks', () => {
  const keys = new Set<string>();
  for (const chunkCount of [1, 3, 9]) {
    const graph = new GPUCommandGraph(device);
    const {view} = createPackedProps(graph);
    const lengths = [
      0,
      ...Array.from({length: chunkCount}, (_, index) => [1, 255, 513][index % 3]),
      0
    ];
    const rows = <Format extends 'uint32' | 'float32x3'>(format: Format) =>
      createVectorView(
        `voxel-${sequence++}`,
        format,
        lengths.map((length, index) => {
          const components = format === 'float32x3' ? 3 : 1;
          const backing = view('uint32', Math.max(length, 1) * components + index);
          return graph.createDataView(backing.buffer, {format, length, byteOffset: index * 4});
        })
      );
    const contributor = new GPUPointCloudVoxelMerge({
      partitionPolicy: 'disjoint-owned-rows',
      parts: [0, 1].map(sourceRowOffset => ({
        counts: rows('uint32'),
        centroids: rows('float32x3'),
        representatives: rows('uint32'),
        sourceRowOffset
      })),
      output: {
        counts: rows('uint32'),
        centroids: rows('float32x3'),
        representatives: rows('uint32'),
        status: rows('uint32')
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
      keys.add(key);
    for (const kernel of getGPUCommandNodeKernelSources(nodes))
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
  }
  expect(keys.size).toBe(3);
});

it('preflights statistical-filter output flags and SMRF nested grid scratch before allocation', () => {
  const graph = new GPUCommandGraph(device);
  const {view, positions} = createPackedProps(graph);
  const validity = graph.createDataView(
    graph.importBuffer({id: 'filter-copy-only', byteLength: 16, usage: Buffer.COPY_DST}),
    {format: 'uint32', length: 4}
  );
  const statisticalFilter = new GPUPointCloudStatisticalFilter({
    counts: view('uint32', 4),
    squaredDistances: view('float32', 16),
    capacity: 4,
    output: {meanDistances: view('float32', 4), validity}
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => statisticalFilter.getCommandNodes(graph)).toThrow(
    /validity.*usage must include Buffer.STORAGE/
  );
  expect(allocate).not.toHaveBeenCalled();
  vi.restoreAllMocks();
  const cells = 33_554_432;
  const smrf = new GPUPointCloudSMRF({
    positions,
    classifications: view('uint32', 4),
    grid: {...GRID, width: 8192, height: 4096},
    window: 1,
    output: {
      groundSurface: view('float32', cells),
      groundMask: view('uint32', 4),
      outputClassifications: view('uint32', 4),
      status: view('uint32', 4)
    }
  });
  const allocateSMRF = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => smrf.getCommandNodes(graph)).toThrow(/grid cell offsets needs 134217732 bytes/);
  expect(allocateSMRF).not.toHaveBeenCalled();
});

it('keeps SMRF shader and pipeline counts fixed as the opening rounds grow', () => {
  const counts: number[] = [];
  for (const window of [1, 3, 5]) {
    const graph = new GPUCommandGraph(device);
    const {view, positions} = createPackedProps(graph);
    const contributor = new GPUPointCloudSMRF({
      positions,
      classifications: view('uint32', 4),
      grid: {...GRID, width: 9, height: 9},
      window,
      output: {
        groundSurface: view('float32', 81),
        groundMask: view('uint32', 4),
        outputClassifications: view('uint32', 4),
        status: view('uint32', 4)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    counts.push(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size);
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
      expect(
        (stripWGSLComments(kernel.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(
          identifier => WGSL_RESERVED_WORDS.has(identifier)
        )
      ).toEqual([]);
    }
  }
  expect(new Set(counts).size).toBe(1);
});

it('accepts disjoint packed status fields across actual neighborhood and ground dispatches', () => {
  const graph = new GPUCommandGraph(device);
  const {view, positions} = createPackedProps(graph);
  const shared = view('uint32', 12);
  expect(() =>
    new GPUPointCloudNeighborhoods({
      positions,
      radius: 1,
      capacity: 2,
      parameters: view('float32', 9),
      output: {
        indices: graph.createDataView(shared.buffer, {format: 'uint32', length: 8}),
        status: graph.createDataView(shared.buffer, {format: 'uint32', length: 4, byteOffset: 32}),
        counts: view('uint32', 4),
        squaredDistances: view('float32', 8)
      }
    }).getCommandNodes(graph)
  ).not.toThrow();
  const groundShared = view('uint32', 8);
  expect(() =>
    new GPUPointCloudGround({
      positions,
      classifications: view('uint32', 4),
      grid: {...GRID, width: 9, height: 9},
      radii: [1],
      initialThreshold: 0,
      slope: 0,
      maximumThreshold: 0,
      minimumSurface: view('float32', 81),
      coverage: view('uint32', 81),
      output: {
        groundMask: view('uint32', 4),
        status: graph.createDataView(groundShared.buffer, {format: 'uint32', length: 4}),
        outputClassifications: graph.createDataView(groundShared.buffer, {
          format: 'uint32',
          length: 4,
          byteOffset: 16
        })
      }
    }).getCommandNodes(graph)
  ).not.toThrow();
});

it('skips all empty row dispatches without validating unused storage bindings', () => {
  const graph = new GPUCommandGraph(device);
  const backing = graph.importBuffer({
    id: 'empty-copy-only',
    byteLength: 16,
    usage: Buffer.COPY_DST
  });
  const counts = graph.createDataView(backing, {format: 'uint32', length: 0});
  const centroids = graph.createDataView(backing, {format: 'float32x3', length: 0});
  const merge = new GPUPointCloudVoxelMerge({
    partitionPolicy: 'disjoint-owned-rows',
    parts: [{counts, centroids, representatives: counts, sourceRowOffset: 0}],
    output: {counts, centroids, representatives: counts, status: counts}
  });
  expect(merge.estimatedCommandNodeCount).toBe(0);
  expect(merge.getCommandNodes(graph)).toEqual([]);
  const height = new GPUPointCloudHeight({
    positions: centroids,
    grid: GRID,
    terrain: graph.createDataView(backing, {format: 'float32', length: 3}),
    coverage: graph.createDataView(backing, {format: 'uint32', length: 3}),
    output: {heights: graph.createDataView(backing, {format: 'float32', length: 0}), status: counts}
  });
  expect(height.getCommandNodes(graph)).toEqual([]);
});

it('permits normal-scale status to share an alignment prefix with separately read source positions', () => {
  const graph = new GPUCommandGraph(device);
  const {view} = createPackedProps(graph);
  const backing = view('uint32', 16);
  const positions = graph.createDataView(backing.buffer, {format: 'float32x3', length: 4});
  const status = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 4,
    byteOffset: 48
  });
  const contributor = new GPUPointCloudNormalScale({
    positions,
    radii: [1],
    capacity: 4,
    parameters: view('float32', 5),
    grid: {origin: [0, 0, 0], dimensions: [3, 3, 3], cellSize: 1},
    gridParameters: view('float32', 4),
    output: {
      status,
      normals: view('float32x3', 4),
      normalScales: view('float32', 4),
      normalVariances: view('float32', 4)
    }
  });
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  for (const kernel of getGPUCommandNodeKernelSources(nodes))
    expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
});
