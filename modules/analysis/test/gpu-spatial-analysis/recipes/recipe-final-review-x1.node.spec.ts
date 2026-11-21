// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {setFlagsFromString} from 'node:v8';
import {runInNewContext} from 'node:vm';
import {setImmediate} from 'node:timers/promises';
import {expect, it, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {addPointsInPolygonsChoroplethRecipe} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe';
import {addDriveTimeCatchmentRecipe} from '../../../src/gpu-network/recipes/drive-time-catchment-recipe';
import {addFacilityScenarioRecipe} from '../../../src/gpu-network/recipes/network-scenarios-recipe';
import {addLandscapeMetricsRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/landscape-metrics';
import {addRateClusterMapRecipe} from '../../../src/gpu-spatial-analysis/recipes/rate-cluster-map-recipe';
import {addRasterZonalDistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/raster-summary-recipes';
import {addSurfacePreparationRecipe} from '../../../src/gpu-spatial-analysis/recipes/surface-preparation-recipe';
import {GPUGroupStatistics} from '../../../src/dataframe-analysis/group-statistics/gpu-group-statistics';
import {
  computePointsInPolygonsChoroplethOnCPU,
  createCPUPointsInPolygonsChoroplethOperation
} from '../../../src/gpu-spatial-analysis/index';
import {createPointsInPolygonsChoroplethOperation} from '../../../src/gpu-spatial-analysis/index';
import {GPUNetworkIsochrones} from '../../../src/gpu-network/network-isochrones/gpu-network-isochrones';
import {GPUIsobandRings} from '../../../src/raster-analysis/isolines/index';
import {addZonalDistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/zonal-distribution-recipe';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {
  computePointInPolygonJoinOnCPU,
  createCPUPointInPolygonJoinOperation
} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-point-in-polygon-join-operation';

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let count = 0;
  const disposers: (() => void)[] = [];
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `seed-${count++}`, format, length);
  const polygons = () => ({
    kind: 'polygons' as const,
    positions: view('float32x2', 4),
    featureOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    ringOffsets: view('uint32', 2),
    candidateCapacity: 8,
    crossingCapacity: 8
  });
  const raster = () => ({
    id: 'retry',
    grid: {
      width: 2,
      height: 2,
      affine: [1, 0, 0, 0, 1, 0] as [number, number, number, number, number, number],
      pixelInterpretation: 'area' as const
    },
    input: {
      id: 'band',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: view('float32', 4)}
    },
    zones: view('uint32', 4),
    zoneCount: 2
  });
  const drive = () => ({
    id: 'retry',
    topology: {
      offsets: view('uint32', 4),
      neighbors: view('uint32', 4),
      weights: view('float32', 4),
      nodePositions: view('float32x2', 3)
    },
    facilities: view('float32x2', 2),
    demand: view('float32x2', 4),
    maximumSnapDistance: view('float32', 1),
    costBreaks: view('float32', 3)
  });
  const parameter = () => {
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: 8
    });
    disposers.push(() => parameters.destroy());
    return parameters;
  };
  const physicalAliases = () => {
    const buffer = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE});
    disposers.push(() => buffer.destroy());
    const aliases = [0, 1].map(() =>
      importGraphBuffer(graph, `seed-${count++}`, buffer, 'float32', 3)
    );
    return aliases;
  };
  const nonStorageView = () => {
    const buffer = device.createBuffer({byteLength: 8, usage: Buffer.COPY_DST});
    disposers.push(() => buffer.destroy());
    return importGraphBuffer(graph, `seed-${count++}`, buffer, 'uint32', 2);
  };
  const nonStorageMetrics = () => {
    const buffer = device.createBuffer({byteLength: 32, usage: Buffer.COPY_DST});
    disposers.push(() => buffer.destroy());
    return importGraphBuffer(graph, `seed-${count++}`, buffer, 'float32x4', 2);
  };
  return {
    device,
    graph,
    view,
    polygons,
    raster,
    drive,
    parameter,
    physicalAliases,
    nonStorageView,
    nonStorageMetrics,
    count: () => count,
    destroy: () => {
      for (const dispose of disposers) dispose();
      device.destroy();
    }
  };
}
type Context = ReturnType<typeof createContext>;

it('B1 inherited points source kind retains its assignment writer and loss status', () => {
  const context = createContext();
  try {
    const join = {
      points: {kind: 'points' as const, positions: context.view('float32x2', 2)},
      polygons: context.polygons(),
      candidateCapacity: 8
    };
    const source = Object.assign(Object.create({kind: 'points'}), {join}) as Parameters<
      typeof addZonalDistributionRecipe
    >[1]['source'];
    const result = addZonalDistributionRecipe(context.graph, {
      source,
      values: context.view('float32', 2)
    });
    expect(result.status.stages.map(stage => stage.stage)).toEqual([
      'zone-assignment',
      'zone-distributions'
    ]);
    expect(result.contributors.some(stage => stage instanceof GPUPointInPolygonJoin)).toBe(true);
  } finally {
    context.destroy();
  }
});

it.each([
  'zonal',
  'group'
] as const)('B2 %s preserves get-only multipart offsets in GPU declarations', backend => {
  const context = createContext();
  try {
    const featureOffsets = context.view('uint32', 2);
    let reads = 0;
    const points = new Proxy(
      {kind: 'points' as const, positions: context.view('float32x2', 2)},
      {
        get(target, key, receiver) {
          if (key === 'featureOffsets') {
            reads++;
            return featureOffsets;
          }
          return Reflect.get(target, key, receiver);
        }
      }
    );
    const result = addPointsInPolygonsChoroplethRecipe(context.graph, {
      backend,
      points,
      polygons: context.polygons()
    });
    const join = result.contributors.find(stage => stage instanceof GPUPointInPolygonJoin) as
      | GPUPointInPolygonJoin
      | undefined;
    expect(join?.props.points.featureOffsets).toBe(featureOffsets);
    expect(reads).toBe(1);
  } finally {
    context.destroy();
  }
});

it.each([
  'group',
  'distribution'
] as const)('N1 %s ordinary point chunks preserve [1,2] assignment topology', route => {
  const context = createContext();
  try {
    const positions = new GraphVectorView({
      id: 'chunked-points',
      name: 'chunked-points',
      format: 'float32x2',
      length: 3,
      valueLength: 6,
      stride: 2,
      byteStride: 8,
      rowByteLength: 8,
      data: [context.view('float32x2', 1), context.view('float32x2', 2)]
    });
    const points = {kind: 'points' as const, positions};
    const values = context.view('float32', 3);
    const result =
      route === 'group'
        ? addPointsInPolygonsChoroplethRecipe(context.graph, {
            backend: 'group',
            statistic: 'sum',
            points,
            polygons: context.polygons(),
            values
          })
        : addZonalDistributionRecipe(context.graph, {
            source: {
              kind: 'points',
              join: {points, polygons: context.polygons(), candidateCapacity: 8}
            },
            values
          });
    const join = result.contributors.find(
      stage => stage instanceof GPUPointInPolygonJoin
    ) as GPUPointInPolygonJoin;
    const assignments = join.props.output.pointFeatureRows;
    expect(assignments).toBeInstanceOf(GraphVectorView);
    if (!(assignments instanceof GraphVectorView)) throw new Error('expected chunked assignments');
    expect(assignments.data.map(chunk => chunk.length)).toEqual([1, 2]);
    const statistics = result.contributors.find(
      stage => stage instanceof GPUGroupStatistics
    ) as GPUGroupStatistics;
    const keys = statistics.props.keys;
    if (keys instanceof GraphVectorView) throw new Error('expected packed reducer keys');
    expect(assignments.data.map(chunk => chunk.byteOffset)).toEqual([
      keys.byteOffset,
      keys.byteOffset + 4
    ]);
    for (const chunk of assignments.data) expect(chunk.buffer).toBe(keys.buffer);
  } finally {
    context.destroy();
  }
});

for (const backend of ['zonal', 'group'] as const)
  for (const chunkCount of [0, 2])
    it(`W5-13 ${backend} accepts empty point vectors with ${chunkCount} chunks`, () => {
      const context = createContext();
      try {
        const positions = new GraphVectorView({
          id: 'empty-vector',
          name: 'empty-vector',
          format: 'float32x2',
          length: 0,
          valueLength: 0,
          stride: 2,
          byteStride: 8,
          rowByteLength: 8,
          data: Array.from({length: chunkCount}, () => context.view('float32x2', 0))
        });
        expect(() =>
          addPointsInPolygonsChoroplethRecipe(context.graph, {
            backend,
            points: {kind: 'points', positions},
            polygons: context.polygons()
          })
        ).not.toThrow();
      } finally {
        context.destroy();
      }
    });

it.each([
  'zonal',
  'group'
] as const)('UGE-01 %s captures inherited topology without reading unused IDs', backend => {
  const context = createContext();
  try {
    const topology = context.polygons();
    const geometry = Object.create(topology) as typeof topology;
    Object.defineProperty(geometry, 'sourceIds', {
      get() {
        throw new Error('unused polygon IDs');
      }
    });
    const points = {
      kind: 'points' as const,
      positions: context.view('float32x2', 1),
      get sourceIds(): never {
        throw new Error('unused point IDs');
      }
    };
    const result = addPointsInPolygonsChoroplethRecipe(context.graph, {
      backend,
      polygons: geometry,
      points
    });
    if (backend === 'group') {
      const join = result.contributors.find(
        stage => stage instanceof GPUPointInPolygonJoin
      ) as GPUPointInPolygonJoin;
      expect(join.props.output.pointFeatureRows).toBeDefined();
      expect(join.props.output.pointFeatureIds).toBeUndefined();
    }
    const distribution = addZonalDistributionRecipe(context.graph, {
      source: {kind: 'points', join: {points, polygons: geometry, candidateCapacity: 8}},
      values: context.view('float32', 1)
    });
    const join = distribution.contributors.find(
      stage => stage instanceof GPUPointInPolygonJoin
    ) as GPUPointInPolygonJoin;
    expect(join.props.output.pointFeatureRows).toBe(distribution.intermediates.zones);
    expect(join.props.output.pointFeatureIds).toBeUndefined();
  } finally {
    context.destroy();
  }
});

function assertAtomicRetry(
  create: (context: Context) => {invalid(): unknown; corrected(): unknown},
  error = /rows|alias|share|distinct|multipart/
) {
  for (const inspect of [true, false]) {
    const context = createContext();
    const calls = create(context);
    const allocate = vi.spyOn(context.graph, 'createTransientBuffer');
    const imported = vi.spyOn(context.graph, 'importBuffer');
    const add = vi.spyOn(context.graph, 'add');
    try {
      expect(calls.invalid).toThrow(error);
      expect(allocate).not.toHaveBeenCalled();
      expect(imported).not.toHaveBeenCalled();
      expect(add).not.toHaveBeenCalled();
      if (inspect) {
        const compiled = context.graph.compile();
        expect(compiled.stats.logicalBufferCount).toBe(context.count());
        expect(compiled.stats.nodeOrder).toEqual([]);
        compiled.destroy();
      } else expect(calls.corrected).not.toThrow();
    } finally {
      vi.restoreAllMocks();
      context.destroy();
    }
  }
}

it.each([
  undefined,
  'group'
] as const)('X1 rejects per-feature value lengths before either choropleth backend (%s) and retries', backend => {
  assertAtomicRetry(context => {
    const points = {kind: 'points' as const, positions: context.view('float32x2', 2)};
    const featureOffsets = context.view('uint32', 2);
    const props = {
      id: 'retry',
      backend,
      points: {...points, featureOffsets},
      polygons: context.polygons(),
      statistic: 'sum' as const
    };
    const partValues = context.view('float32', 2);
    const featureValues = context.view('float32', 1);
    return {
      invalid: () =>
        Reflect.apply(addPointsInPolygonsChoroplethRecipe, undefined, [
          context.graph,
          {
            ...props,
            values: partValues
          }
        ]),
      corrected: () =>
        addPointsInPolygonsChoroplethRecipe(context.graph, {...props, values: featureValues})
    };
  });
});

it.each([
  'demandTimes',
  'demandBands'
] as const)('X2 rejects short %s despite spare offset backing capacity and retries with four rows', port => {
  assertAtomicRetry(context => {
    const props = context.drive();
    const format = port === 'demandTimes' ? 'float32' : 'uint32';
    const backing = context.view(format, 8);
    const short = context.graph.createDataView(backing.buffer, {format, length: 1, byteOffset: 4});
    const corrected = context.graph.createDataView(backing.buffer, {
      format,
      length: 4,
      byteOffset: 4
    });
    return {
      invalid: () =>
        addDriveTimeCatchmentRecipe(context.graph, {...props, output: {[port]: short}}),
      corrected: () => {
        const result = addDriveTimeCatchmentRecipe(context.graph, {
          ...props,
          output: {[port]: corrected}
        });
        expect(result[port]).toBe(corrected);
      }
    };
  });
});

it.each([
  false,
  true
])('X3 rejects short landscape class output before defaults or reconciliation (reconcile=%s)', reconcilePatches => {
  assertAtomicRetry(context => {
    const props = {
      id: 'retry',
      patches: context.view('uint32', 2),
      patchClasses: context.view('uint32', 2),
      width: 2,
      height: 1,
      classCount: 2,
      cellSize: [1, 1] as const,
      connectivity: 4 as const,
      reconcilePatches
    };
    const short = context.view('float32x4', 1);
    const corrected = context.view('float32x4', 2);
    return {
      invalid: () =>
        addLandscapeMetricsRecipe(context.graph, {...props, output: {classMetrics: short}}),
      corrected: () =>
        addLandscapeMetricsRecipe(context.graph, {...props, output: {classMetrics: corrected}})
    };
  });
});

it('X5 rejects spatial-lag scratch overwriting retained smoothed rates and retries', () => {
  assertAtomicRetry(context => {
    const shared = context.view('float32', 4);
    const props = {
      id: 'retry',
      events: context.view('float32', 4),
      populations: context.view('float32', 4),
      polygons: {
        kind: 'polygons' as const,
        positions: context.view('float32x2', 16),
        featureOffsets: context.view('uint32', 5),
        polygonOffsets: context.view('uint32', 5),
        ringOffsets: context.view('uint32', 5)
      },
      parameters: context.parameter(),
      palette: context.view('uint32', 5),
      neighborCapacity: 16,
      pairCapacity: 16,
      output: {smoothedRates: shared}
    };
    return {
      invalid: () =>
        addRateClusterMapRecipe(context.graph, {...props, scratch: {spatialLag: shared}}),
      corrected: () =>
        addRateClusterMapRecipe(context.graph, {
          ...props,
          scratch: {spatialLag: context.view('float32', 4)}
        })
    };
  });
});

it.each([
  ['patchMetrics', 'float32x4', 2],
  ['classMetrics', 'float32x4', 2],
  ['classAdjacency', 'uint32', 4],
  ['aggregation', 'float32', 2],
  ['landscape', 'float32x4', 1],
  ['landscapeAggregation', 'float32', 1],
  ['landscapeCohesion', 'float32', 1],
  ['reconciledPatches', 'uint32', 2],
  ['patchParents', 'uint32', 2]
] as const)('X3 preflights supplied %s before any default or reconcile stage', (name, format, length) => {
  assertAtomicRetry(context => {
    const props = {
      id: 'retry',
      patches: context.view('uint32', 2),
      patchClasses: context.view('uint32', 2),
      width: 2,
      height: 1,
      classCount: 2,
      cellSize: [1, 1] as const,
      connectivity: 4 as const,
      reconcilePatches: true
    };
    const short = context.view(format, length === 1 ? 2 : 1);
    const corrected = context.view(format, length);
    return {
      invalid: () => addLandscapeMetricsRecipe(context.graph, {...props, output: {[name]: short}}),
      corrected: () =>
        addLandscapeMetricsRecipe(context.graph, {...props, output: {[name]: corrected}})
    };
  });
});

it('X5 rejects demand times overwriting original routing weights and retries', () => {
  assertAtomicRetry(context => {
    const props = context.drive();
    const corrected = context.view('float32', 4);
    return {
      invalid: () =>
        addDriveTimeCatchmentRecipe(context.graph, {
          ...props,
          output: {demandTimes: props.topology.weights}
        }),
      corrected: () =>
        addDriveTimeCatchmentRecipe(context.graph, {...props, output: {demandTimes: corrected}})
    };
  });
});

it('X5 rejects physically aliased facility outputs imported through different handles and retries', () => {
  assertAtomicRetry(context => {
    const [nearestCosts, accessibility] = context.physicalAliases();
    const corrected = context.view('float32', 3);
    const props = {
      id: 'retry',
      costs: context.view('float32', 6),
      demand: context.view('float32', 3),
      supply: context.view('float32', 2),
      facilityMask: context.view('uint32', 2),
      parameters: context.view('float32', 4)
    };
    return {
      invalid: () =>
        addFacilityScenarioRecipe(context.graph, {
          ...props,
          outputs: {nearestCosts, accessibility}
        }),
      corrected: () =>
        addFacilityScenarioRecipe(context.graph, {
          ...props,
          outputs: {nearestCosts, accessibility: corrected}
        })
    };
  });
});

it.each([
  'raster',
  'drive'
] as const)('X6 returns every scheduled %s stage as a contributor in declaration order', kind => {
  const context = createContext();
  const originalAdd = context.graph.add.bind(context.graph);
  const stages: unknown[] = [];
  let depth = 0;
  vi.spyOn(context.graph, 'add').mockImplementation((...args) => {
    if (depth === 0) stages.push(args[0]);
    depth++;
    try {
      return originalAdd(...args);
    } finally {
      depth--;
    }
  });
  try {
    const result =
      kind === 'raster'
        ? addRasterZonalDistributionRecipe(context.graph, context.raster())
        : addDriveTimeCatchmentRecipe(context.graph, context.drive());
    expect(stages).toEqual(result.contributors);
    expect(result.contributors.length).toBeGreaterThan(kind === 'raster' ? 1 : 4);
  } finally {
    vi.restoreAllMocks();
    context.destroy();
  }
});

it('X8 ignores enumerable auxiliary foreign views but rejects a foreign consumed raster input', () => {
  const context = createContext();
  const foreign = createContext();
  try {
    const props = context.raster();
    const preview = foreign.view('float32', 4);
    const configuration = {...props, preview};
    const foreignInput = {
      ...configuration,
      input: {...props.input, storage: {kind: 'buffer' as const, values: preview}}
    };
    expect(() => addRasterZonalDistributionRecipe(context.graph, foreignInput)).toThrow(/belong/);
    const result = addRasterZonalDistributionRecipe(context.graph, configuration);
    expect(result.outputs.medians.length).toBe(2);
  } finally {
    context.destroy();
    foreign.destroy();
  }
});

it('X9 retaining only committed grid metadata does not retain the preparation graph', async () => {
  setFlagsFromString('--expose-gc');
  const collect = runInNewContext('gc') as () => void;
  const retained = (() => {
    const context = createContext();
    const props = {...context.raster(), maximumDistance: 1};
    let preparation: WeakRef<GPUCommandGraph> | undefined;
    const original = GPUCommandGraph.prototype.createTransientBuffer;
    const spy = vi
      .spyOn(GPUCommandGraph.prototype, 'createTransientBuffer')
      .mockImplementation(function (this: GPUCommandGraph, descriptor) {
        if (this !== context.graph) preparation ??= new WeakRef(this);
        return original.call(this, descriptor);
      });
    try {
      const result = addSurfacePreparationRecipe(context.graph, props);
      expect(preparation).toBeDefined();
      return {
        grid: result.intermediates['grid'],
        preparation: preparation!,
        caller: new WeakRef(context.graph),
        props: new WeakRef(props),
        control: new WeakRef({})
      };
    } finally {
      spy.mockRestore();
      context.destroy();
    }
  })();
  async function isCollected(reference: WeakRef<object>) {
    for (let attempt = 0; attempt < 30; attempt++) {
      await setImmediate();
      collect();
      await setImmediate();
      if (!reference.deref()) return true;
    }
    return false;
  }
  expect(await isCollected(retained.control)).toBe(true);
  expect(await isCollected(retained.preparation)).toBe(true);
  expect(await isCollected(retained.caller)).toBe(true);
  expect(await isCollected(retained.props)).toBe(true);
  expect(retained.grid).toMatchObject({width: 2, height: 2});
});

it.each([
  'patches',
  'patchClasses'
] as const)('R1 rejects strided %s before mutation and retries', name => {
  assertAtomicRetry(context => {
    const packed = context.view('uint32', 2);
    const backing = context.view('uint32', 4);
    const strided = context.graph.createDataView(backing.buffer, {
      format: 'uint32',
      length: 2,
      byteStride: 8
    });
    const props = {
      id: 'retry',
      patches: packed,
      patchClasses: context.view('uint32', 2),
      width: 2,
      height: 1,
      classCount: 2,
      cellSize: [1, 1] as const,
      connectivity: 4 as const
    };
    return {
      invalid: () => addLandscapeMetricsRecipe(context.graph, {...props, [name]: strided}),
      corrected: () => addLandscapeMetricsRecipe(context.graph, {...props, [name]: packed})
    };
  }, /packed|stride/);
});

it.each([
  'zonal',
  'group'
] as const)('R2 %s counts own and inherited feature offsets once', backend => {
  // Hand-counted oracle: both interior positions belong to one feature in one square.
  const polygons = {
    kind: 'polygons' as const,
    positions: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1),
    featureOffsets: Uint32Array.of(0, 1),
    polygonOffsets: Uint32Array.of(0, 1),
    ringOffsets: Uint32Array.of(0, 4)
  };
  for (const inherited of [false, true]) {
    const featureOffsets = Uint32Array.of(0, 2);
    const points = Object.assign(
      Object.create(
        inherited
          ? {
              get featureOffsets() {
                return featureOffsets;
              }
            }
          : null
      ),
      {kind: 'points', positions: Float32Array.of(0.25, 0.25, 0.75, 0.75)},
      inherited ? {} : {featureOffsets}
    );
    const result = computePointsInPolygonsChoroplethOnCPU(
      {points, polygons},
      {backend, candidateCapacity: 8}
    );
    expect(Array.from(result.counts)).toEqual([1]);
    expect(Array.from(result.overflow)).toEqual([0]);
    expect(Array.from(result.uncertainCount)).toEqual([0]);
  }
});

it('R1a output usage preflight also preserves a corrected landscape retry', () => {
  assertAtomicRetry(context => {
    const shortUsage = context.nonStorageMetrics();
    const corrected = context.view('float32x4', 2);
    const props = {
      id: 'retry',
      patches: context.view('uint32', 2),
      patchClasses: context.view('uint32', 2),
      width: 2,
      height: 1,
      classCount: 2,
      cellSize: [1, 1] as const,
      connectivity: 4 as const
    };
    return {
      invalid: () =>
        addLandscapeMetricsRecipe(context.graph, {...props, output: {classMetrics: shortUsage}}),
      corrected: () =>
        addLandscapeMetricsRecipe(context.graph, {...props, output: {classMetrics: corrected}})
    };
  }, /usage|STORAGE/);
});

it.each([
  'patches',
  'patchClasses'
] as const)('R1a rejects COPY_DST-only %s before mutation and retries', name => {
  assertAtomicRetry(context => {
    const imported = context.nonStorageView();
    const packed = context.view('uint32', 2);
    const props = {
      id: 'retry',
      patches: packed,
      patchClasses: context.view('uint32', 2),
      width: 2,
      height: 1,
      classCount: 2,
      cellSize: [1, 1] as const,
      connectivity: 4 as const
    };
    return {
      invalid: () => addLandscapeMetricsRecipe(context.graph, {...props, [name]: imported}),
      corrected: () => addLandscapeMetricsRecipe(context.graph, {...props, [name]: packed})
    };
  }, /usage|STORAGE/);
});

it.each([
  'zonal',
  'group'
] as const)('R2a %s preserves inherited closed-boundary options', backend => {
  // Independent boundary oracle: the square covers its right edge, while half-open excludes it.
  const input = {
    points: {
      kind: 'points' as const,
      positions: Float32Array.of(1, 0.5),
      featureOffsets: Uint32Array.of(0, 1)
    },
    polygons: {
      kind: 'polygons' as const,
      positions: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1),
      featureOffsets: Uint32Array.of(0, 1),
      polygonOffsets: Uint32Array.of(0, 1),
      ringOffsets: Uint32Array.of(0, 4)
    }
  };
  const closed = {backend, candidateCapacity: 8, boundaryRule: 'closed' as const};
  for (const options of [
    closed,
    Object.assign(Object.create({boundaryRule: 'closed'}), {backend, candidateCapacity: 8}),
    Object.create(closed)
  ]) {
    const result = computePointsInPolygonsChoroplethOnCPU(input, options);
    expect(Array.from(result.counts)).toEqual([1]);
    expect(Array.from(result.overflow)).toEqual([0]);
    expect(Array.from(result.uncertainCount)).toEqual([0]);
  }
});

it.each([
  false,
  true
])('R4a captures and schedules own/inherited raster rings (inherited=%s)', inherited => {
  const context = createContext();
  const expand = vi.spyOn(GPUIsobandRings.prototype, 'getCommandNodes');
  try {
    const rings = {
      edgeCapacity: 8,
      output: {
        positions: context.view('float32x2', 8),
        ringOffsets: context.view('uint32', 3),
        status: {
          count: context.view('uint32', 1),
          overflow: context.view('uint32', 1),
          edgeOverflow: context.view('uint32', 1)
        }
      }
    };
    const fields = {width: 2, height: 2, output: {bandClasses: context.view('uint32', 4)}};
    const raster = inherited
      ? Object.assign(
          Object.create({
            get rings() {
              return rings;
            }
          }),
          fields
        )
      : {...fields, rings};
    const result = addDriveTimeCatchmentRecipe(context.graph, {
      ...context.drive(),
      isochrones: {
        costBreaks: context.view('float32', 3),
        parameters: context.view('float32', 12),
        raster
      }
    });
    const contributor = result.contributors.find(
      stage => stage instanceof GPUNetworkIsochrones
    ) as GPUNetworkIsochrones;
    expect(contributor.props.raster?.rings?.output.status.edgeOverflow).toBe(
      rings.output.status.edgeOverflow
    );
    expect(
      result.status.stages.find(stage => stage.stage === 'isochrone-band-rings')?.status
        .candidateOverflow
    ).toBe(rings.output.status.edgeOverflow);
    // The captured ring request and published stage must refer to the same scheduled producer.
    expect(contributor.props.raster?.rings).toBeDefined();
    expect(expand.mock.calls.some(([graph]) => graph === context.graph)).toBe(true);
  } finally {
    expand.mockRestore();
    context.destroy();
  }
});

it.each([
  false,
  true
])('R4a rejects own/inherited raster ring status alias (inherited=%s) and retries', inherited => {
  assertAtomicRetry(context => {
    const converged = context.view('uint32', 1);
    const separate = context.view('uint32', 1);
    const fields = {width: 2, height: 2, output: {bandClasses: context.view('uint32', 4)}};
    const rings = {
      edgeCapacity: 8,
      output: {
        positions: context.view('float32x2', 8),
        ringOffsets: context.view('uint32', 3),
        status: {
          count: context.view('uint32', 1),
          overflow: context.view('uint32', 1),
          edgeOverflow: converged
        }
      }
    };
    const props = {...context.drive(), output: {status: {serviceAreaConverged: converged}}};
    const parameters = context.view('float32', 12);
    const breaks = context.view('float32', 3);
    const declare = (edgeOverflow: typeof converged) => {
      const request = {
        ...rings,
        output: {...rings.output, status: {...rings.output.status, edgeOverflow}}
      };
      const raster = inherited
        ? Object.assign(
            Object.create({
              get rings() {
                return request;
              }
            }),
            fields
          )
        : {...fields, rings: request};
      return addDriveTimeCatchmentRecipe(context.graph, {
        ...props,
        isochrones: {costBreaks: breaks, parameters, raster}
      });
    };
    return {invalid: () => declare(converged), corrected: () => declare(separate)};
  });
});

it('R4b accepts disjoint nested isochrone outputs sharing one physical buffer', () => {
  const context = createContext();
  try {
    const backing = context.view('uint32', 68);
    const bandClasses = context.graph.createDataView(backing.buffer, {format: 'uint32', length: 4});
    const pixelFacilities = context.graph.createDataView(backing.buffer, {
      format: 'uint32',
      length: 4,
      byteOffset: 256
    });
    expect(() =>
      addDriveTimeCatchmentRecipe(context.graph, {
        ...context.drive(),
        isochrones: {
          costBreaks: context.view('float32', 3),
          parameters: context.view('float32', 12),
          raster: {width: 2, height: 2, output: {bandClasses, pixelFacilities}}
        }
      })
    ).not.toThrow();
  } finally {
    context.destroy();
  }
});

for (const backend of ['zonal', 'group'] as const)
  it.each([
    'overflow',
    'uncertainCount'
  ] as const)(`R5c ${backend} rejects an empty aggregate aliasing %s and retries`, port => {
    assertAtomicRetry(context => {
      const shared = context.view('uint32', 1);
      const featureValues = context.graph.createDataView(shared.buffer, {
        format: 'float32',
        length: 1
      });
      const separate = context.view('float32', 1);
      const props = {
        id: 'retry',
        points: {
          kind: 'points' as const,
          positions: context.view('float32x2', 0),
          featureOffsets: context.view('uint32', 1)
        },
        polygons: context.polygons(),
        values: context.view('float32', 0),
        statistic: 'mean' as const,
        backend
      };
      return {
        invalid: () =>
          addPointsInPolygonsChoroplethRecipe(context.graph, {
            ...props,
            output: {featureValues, status: {[port]: shared}}
          }),
        corrected: () =>
          addPointsInPolygonsChoroplethRecipe(context.graph, {
            ...props,
            output: {featureValues: separate, status: {[port]: shared}}
          })
      };
    });
  });

it.each([
  'zonal',
  'group'
] as const)('R5a %s declares empty ordinary points for every statistic', backend => {
  const context = createContext();
  try {
    const points = {kind: 'points' as const, positions: context.view('float32x2', 0)};
    const polygons = context.polygons();
    const values = context.view('float32', 0);
    for (const statistic of ['count', 'sum', 'mean', 'minimum', 'maximum'] as const)
      expect(() =>
        addPointsInPolygonsChoroplethRecipe(context.graph, {
          id: `empty-${statistic}`,
          backend,
          points,
          polygons,
          values,
          statistic
        })
      ).not.toThrow();
  } finally {
    context.destroy();
  }
});

it('R5b empty CPU aggregates preserve polygon leaf-capacity loss', () => {
  const result = computePointsInPolygonsChoroplethOnCPU(
    {
      points: {kind: 'points', positions: new Float32Array(0), featureOffsets: Uint32Array.of(0)},
      polygons: {
        kind: 'polygons',
        positions: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 2, 0, 3, 0, 3, 1, 2, 1),
        featureOffsets: Uint32Array.of(0, 1, 2),
        polygonOffsets: Uint32Array.of(0, 1, 2),
        ringOffsets: Uint32Array.of(0, 4, 8)
      }
    },
    {candidateCapacity: 8, leafCapacity: 1}
  );
  expect(Array.from(result.counts)).toEqual([0, 0]);
  expect(Array.from(result.overflow)).toEqual([1]);
  expect(Array.from(result.uncertainCount)).toEqual([0]);
});

it('R7a count-only CPU aggregation reuses a polygon row instead of a membership matrix', async () => {
  const input = {
    points: {
      kind: 'points' as const,
      positions: new Float32Array(0),
      featureOffsets: new Uint32Array(8)
    },
    polygons: {
      kind: 'polygons' as const,
      positions: Float32Array.of(
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        2,
        0,
        3,
        0,
        3,
        1,
        2,
        1,
        4,
        0,
        5,
        0,
        5,
        1,
        4,
        1
      ),
      featureOffsets: Uint32Array.of(0, 1, 2, 3),
      polygonOffsets: Uint32Array.of(0, 1, 2, 3),
      ringOffsets: Uint32Array.of(0, 4, 8, 12)
    }
  };
  const allocations: number[] = [];
  const Native = Uint32Array;
  vi.stubGlobal(
    'Uint32Array',
    new Proxy(Native, {
      construct(target, arguments_) {
        if (typeof arguments_[0] === 'number') allocations.push(arguments_[0]);
        return Reflect.construct(target, arguments_);
      }
    })
  );
  try {
    const result = computePointsInPolygonsChoroplethOnCPU(input, {candidateCapacity: 8});
    expect(Array.from(result.counts)).toEqual([0, 0, 0]);
    expect(allocations).not.toContain(21);
    allocations.length = 0;
    const operation = createCPUPointsInPolygonsChoroplethOperation({
      candidateCapacity: 8,
      points: {
        kind: 'points',
        positions: {value: 'points', length: 0},
        featureOffsets: {value: 'offsets', length: 8}
      },
      polygons: {
        kind: 'polygons',
        positions: {value: 'polygonPositions', length: 12},
        featureOffsets: {value: 'features', length: 4},
        polygonOffsets: {value: 'polygons', length: 4},
        ringOffsets: {value: 'rings', length: 4}
      },
      output: {counts: 'counts', overflow: 'overflow', uncertainCount: 'uncertainCount'}
    });
    const valuesByName = {
      points: input.points.positions,
      offsets: input.points.featureOffsets,
      polygonPositions: input.polygons.positions,
      features: input.polygons.featureOffsets,
      polygons: input.polygons.polygonOffsets,
      rings: input.polygons.ringOffsets
    };
    const rows = (await operation.execute({
      inputs: Object.fromEntries(
        Object.entries(operation.inputs).map(([port, name]) => {
          const values = valuesByName[name as keyof typeof valuesByName];
          const format =
            name === 'points' || name === 'polygonPositions'
              ? ('float32x2' as const)
              : ('uint32' as const);
          const length = values.length / (format === 'float32x2' ? 2 : 1);
          return [port, {format, length, chunks: [{values, length}], pack: () => values}];
        })
      )
    })) as Record<string, Uint32Array>;
    expect(Array.from(rows['counts'])).toEqual([0, 0, 0]);
    expect(Array.from(rows['overflow'])).toEqual([0]);
    expect(Array.from(rows['uncertainCount'])).toEqual([0]);
    expect(allocations).not.toContain(21);
    // Explicit membership callers still receive the complete 7 x 3 matrix.
    allocations.length = 0;
    const pointOperation = createCPUPointInPolygonJoinOperation({
      candidateCapacity: 8,
      matchCapacity: 7,
      points: {
        kind: 'points',
        positions: {value: 'points', length: 0},
        featureOffsets: {value: 'offsets', length: 8}
      },
      polygons: {
        kind: 'polygons',
        positions: {value: 'polygonPositions', length: 12},
        featureOffsets: {value: 'features', length: 4},
        polygonOffsets: {value: 'polygons', length: 4},
        ringOffsets: {value: 'rings', length: 4}
      },
      output: {
        pointFeatureIds: 'pointFeatureIds',
        polygonAssignmentCounts: 'featureCounts',
        candidateTruncated: 'candidateTruncated',
        ids: 'ids',
        count: 'count',
        requiredCount: 'requiredCount',
        matchOverflow: 'matchOverflow',
        matchCandidateOverflow: 'matchCandidateOverflow',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow',
        candidateCount: 'candidateCount',
        uncertainCount: 'uncertainCount'
      }
    });
    const pointRows = (await pointOperation.execute({
      inputs: Object.fromEntries(
        Object.entries(pointOperation.inputs).map(([port, name]) => {
          const values = valuesByName[name as keyof typeof valuesByName];
          const format =
            name === 'points' || name === 'polygonPositions'
              ? ('float32x2' as const)
              : ('uint32' as const);
          const length = values.length / (format === 'float32x2' ? 2 : 1);
          return [port, {format, length, chunks: [{values, length}], pack: () => values}];
        })
      )
    })) as Record<string, Uint32Array>;
    // Seven empty features have no assignments, matches or candidate diagnostics.
    expect(Array.from(pointRows['featureCounts'])).toEqual([0, 0, 0]);
    expect(Array.from(pointRows['pointFeatureIds'])).toEqual([
      0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff
    ]);
    for (const name of ['candidateTruncated', 'ids'])
      expect(Array.from(pointRows[name])).toEqual([0, 0, 0, 0, 0, 0, 0]);
    for (const name of [
      'count',
      'requiredCount',
      'matchOverflow',
      'matchCandidateOverflow',
      'overflow',
      'candidateOverflow',
      'candidateCount',
      'uncertainCount'
    ])
      expect(Array.from(pointRows[name])).toEqual([0]);
    expect(allocations).not.toContain(21);
    expect(
      computePointInPolygonJoinOnCPU(input, {candidateCapacity: 8, matchCapacity: 7})
        .featurePolygonMembership
    ).toHaveLength(21);
  } finally {
    vi.unstubAllGlobals();
  }
});

it('R3 raster validity is a value filter, not an assigned-row mask', () => {
  const context = createContext();
  try {
    const result = addRasterZonalDistributionRecipe(context.graph, context.raster());
    const group = result.contributors.find(
      stage => stage instanceof GPUGroupStatistics
    ) as GPUGroupStatistics;
    expect(group.props.mask).toBeUndefined();
    expect(group.props.keys).toBe(result.intermediates.zones);
    expect(group.props.output.counts).toBe(result.outputs.rowCounts);
  } finally {
    context.destroy();
  }
});

it('R4 rejects demand times aliasing nested isochrone values and retries', () => {
  assertAtomicRetry(context => {
    const shared = context.view('float32', 4);
    const separate = context.view('float32', 4);
    const props = {...context.drive(), output: {demandTimes: shared}};
    const isochrones = {
      costBreaks: context.view('float32', 3),
      parameters: context.view('float32', 12),
      raster: {
        width: 2,
        height: 2,
        output: {values: shared, bandClasses: context.view('uint32', 4)}
      }
    };
    return {
      invalid: () => addDriveTimeCatchmentRecipe(context.graph, {...props, isochrones}),
      corrected: () =>
        addDriveTimeCatchmentRecipe(context.graph, {
          ...props,
          isochrones: {
            ...isochrones,
            raster: {...isochrones.raster, output: {...isochrones.raster.output, values: separate}}
          }
        })
    };
  });
});

it.each(['zonal', 'group'] as const)('R5 %s declares all empty multipart statistics', backend => {
  const context = createContext();
  try {
    const points = {
      kind: 'points' as const,
      positions: context.view('float32x2', 0),
      featureOffsets: context.view('uint32', 1)
    };
    const polygons = context.polygons();
    const values = context.view('float32', 0);
    for (const statistic of ['count', 'sum', 'mean', 'minimum', 'maximum'] as const)
      expect(() =>
        addPointsInPolygonsChoroplethRecipe(context.graph, {
          id: `empty-${statistic}`,
          backend,
          points,
          polygons,
          values,
          statistic
        })
      ).not.toThrow();
  } finally {
    context.destroy();
  }
});

it('R6 raster provenance snapshots declared nested metadata before caller mutation', () => {
  const context = createContext();
  try {
    const props = context.raster();
    const coordinateReferenceSystem = {
      authority: 'fixture',
      projectionJson: {nested: {axis: ['east', 'north']}}
    };
    const grid = {
      ...props.grid,
      coordinateReferenceSystem,
      levelZeroOrigin: [3, 4] as [number, number]
    };
    const result = addSurfacePreparationRecipe(context.graph, {...props, grid, maximumDistance: 1});
    Object.assign(grid, {pixelInterpretation: 'point'});
    grid.affine[2] = 123;
    grid.levelZeroOrigin[0] = 99;
    coordinateReferenceSystem.projectionJson.nested.axis[0] = 'changed';
    expect(result.intermediates['grid']).toMatchObject({
      pixelInterpretation: 'area',
      affine: [1, 0, 0, 0, 1, 0],
      levelZeroOrigin: [3, 4],
      coordinateReferenceSystem: {projectionJson: {nested: {axis: ['east', 'north']}}}
    });
  } finally {
    context.destroy();
  }
});

it('R7 count-only multipart declaration does not allocate a dense membership matrix', () => {
  const context = createContext();
  try {
    const polygonCount = 4097;
    expect(() =>
      addPointsInPolygonsChoroplethRecipe(context.graph, {
        points: {
          kind: 'points',
          positions: context.view('float32x2', 8192),
          featureOffsets: context.view('uint32', 8193)
        },
        polygons: {
          kind: 'polygons',
          positions: context.view('float32x2', 4 * polygonCount),
          featureOffsets: context.view('uint32', polygonCount + 1),
          polygonOffsets: context.view('uint32', polygonCount + 1),
          ringOffsets: context.view('uint32', polygonCount + 1),
          candidateCapacity: 1
        }
      })
    ).not.toThrow();
  } finally {
    context.destroy();
  }
});

function assertOperationAtomicRetry(injectAllocation: boolean) {
  assertAtomicRetry(context => {
    const points = {
      kind: 'points' as const,
      positions: context.view('float32x2', 2),
      featureOffsets: context.view('uint32', 2)
    };
    const polygons = context.polygons();
    const views = {
      pointPositions: points.positions,
      pointOffsets: points.featureOffsets,
      polygonPositions: polygons.positions,
      features: polygons.featureOffsets,
      polygons: polygons.polygonOffsets,
      rings: polygons.ringOffsets,
      counts: context.view('uint32', 1),
      overflow: context.view('uint32', 1),
      uncertainCount: context.view('uint32', 1)
    };
    const short = context.view('uint32', 0);
    const value = (name: keyof typeof views) => ({value: name, length: views[name].length});
    const operation = createPointsInPolygonsChoroplethOperation({
      id: 'retry',
      points: {
        kind: 'points',
        positions: value('pointPositions'),
        featureOffsets: value('pointOffsets')
      },
      polygons: {
        kind: 'polygons',
        positions: value('polygonPositions'),
        featureOffsets: value('features'),
        polygonOffsets: value('polygons'),
        ringOffsets: value('rings')
      },
      candidateCapacity: 8,
      output: {counts: 'counts', overflow: 'overflow', uncertainCount: 'uncertainCount'}
    });
    const declare = (invalid: boolean) => {
      const contributor = operation.createContributor(
        port =>
          (invalid && port === 'counts'
            ? short
            : views[
                (operation.outputs[port]?.value ?? operation.inputs[port]) as keyof typeof views
              ]) as never
      );
      // Call the adapter directly: graph.add itself would count the attempted outer declaration.
      if (injectAllocation)
        context.graph.createTransientBuffer({
          id: 'injected-before-preflight',
          byteLength: 4,
          usage: Buffer.STORAGE
        });
      return contributor.getCommandNodes(context.graph);
    };
    return {invalid: () => declare(true), corrected: () => declare(false)};
  }, /rows|count|length|polygon(?:Assignment|Membership)Counts/);
}

it('R8 choropleth operation rejects an invalid resolved output without allocation and retries', () => {
  assertOperationAtomicRetry(false);
});

it('R8 rollback discriminator catches an injected allocation before recipe preflight', () => {
  expect(() => assertOperationAtomicRetry(true)).toThrow(/createTransientBuffer/);
});

it.each([
  'zonal',
  'group'
] as const)('R5 %s CPU operation publishes every empty aggregate and status', async backend => {
  const inputs = {
    points: {format: 'float32x2' as const, values: new Float32Array(0)},
    offsets: {format: 'uint32' as const, values: Uint32Array.of(0)},
    values: {format: 'float32' as const, values: new Float32Array(0)},
    polygonPositions: {
      format: 'float32x2' as const,
      values: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1)
    },
    features: {format: 'uint32' as const, values: Uint32Array.of(0, 1)},
    polygons: {format: 'uint32' as const, values: Uint32Array.of(0, 1)},
    rings: {format: 'uint32' as const, values: Uint32Array.of(0, 4)}
  };
  for (const statistic of ['count', 'sum', 'mean', 'minimum', 'maximum'] as const) {
    const operation = createCPUPointsInPolygonsChoroplethOperation({
      backend,
      statistic,
      candidateCapacity: 8,
      points: {
        kind: 'points',
        positions: {value: 'points', length: 0},
        featureOffsets: {value: 'offsets', length: 1}
      },
      polygons: {
        kind: 'polygons',
        positions: {value: 'polygonPositions', length: 4},
        featureOffsets: {value: 'features', length: 2},
        polygonOffsets: {value: 'polygons', length: 2},
        ringOffsets: {value: 'rings', length: 2}
      },
      ...(statistic === 'count' ? {} : {values: {value: 'values', length: 0}}),
      output: {
        counts: 'counts',
        ...(statistic === 'count' ? {} : {featureValues: 'featureValues'}),
        overflow: 'overflow',
        uncertainCount: 'uncertainCount'
      }
    });
    const result = (await operation.execute({
      inputs: Object.fromEntries(
        Object.entries(operation.inputs).map(([port, name]) => {
          const {values, format} = inputs[name as keyof typeof inputs];
          const length = values.length / (format === 'float32x2' ? 2 : 1);
          return [port, {format, length, chunks: [{values, length}], pack: () => values}];
        })
      )
    })) as Record<string, Uint32Array | Float32Array>;
    // Closed-form empty-set reduction; each output is checked independently of the GPU form.
    expect(Array.from(result['counts'])).toEqual([0]);
    expect(Array.from(result['overflow'])).toEqual([0]);
    expect(Array.from(result['uncertainCount'])).toEqual([0]);
    if (statistic === 'sum') expect(Array.from(result['featureValues'])).toEqual([0]);
    else if (statistic !== 'count') expect(Number.isNaN(result['featureValues'][0])).toBe(true);
  }
});

it.each([
  'count',
  'sum',
  'mean',
  'minimum',
  'maximum'
] as const)('RCP-108 public CPU nonempty %s matches independent tally', async statistic => {
  const inputs = {
    points: {
      format: 'float32x2' as const,
      values: Float32Array.of(0.25, 0.25, 0.75, 0.75, 1.5, 0.5)
    },
    values: {format: 'float32' as const, values: Float32Array.of(2, 4, 8)},
    polygonPositions: {
      format: 'float32x2' as const,
      values: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1)
    },
    features: {format: 'uint32' as const, values: Uint32Array.of(0, 1, 2)},
    polygons: {format: 'uint32' as const, values: Uint32Array.of(0, 1, 2)},
    rings: {format: 'uint32' as const, values: Uint32Array.of(0, 4, 8)}
  };
  const operation = createCPUPointsInPolygonsChoroplethOperation({
    statistic,
    candidateCapacity: 8,
    points: {kind: 'points', positions: {value: 'points', length: 3}},
    polygons: {
      kind: 'polygons',
      positions: {value: 'polygonPositions', length: 8},
      featureOffsets: {value: 'features', length: 3},
      polygonOffsets: {value: 'polygons', length: 3},
      ringOffsets: {value: 'rings', length: 3}
    },
    ...(statistic === 'count' ? {} : {values: {value: 'values', length: 3}}),
    output: {
      counts: 'counts',
      ...(statistic === 'count' ? {} : {featureValues: 'featureValues'}),
      overflow: 'overflow',
      uncertainCount: 'uncertainCount'
    }
  });
  const result = (await operation.execute({
    inputs: Object.fromEntries(
      Object.entries(operation.inputs).map(([port, name]) => {
        const {values, format} = inputs[name as keyof typeof inputs];
        const length = values.length / (format === 'float32x2' ? 2 : 1);
        return [port, {format, length, chunks: [{values, length}], pack: () => values}];
      })
    )
  })) as Record<string, Uint32Array | Float32Array>;
  expect(Array.from(result['counts'])).toEqual([2, 1]);
  expect(Array.from(result['overflow'])).toEqual([0]);
  expect(Array.from(result['uncertainCount'])).toEqual([0]);
  // Left: [2,4]; right: [8]. Independent exact integer arithmetic.
  const expected = {sum: [6, 8], mean: [3, 8], minimum: [2, 8], maximum: [4, 8]};
  if (statistic !== 'count') {
    expect(result['featureValues']).toHaveLength(2);
    Array.from(result['featureValues']).forEach((value, row) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBe(expected[statistic][row]);
    });
  }
});
