// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
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
  return {
    device,
    graph,
    view,
    polygons,
    raster,
    drive,
    parameter,
    physicalAliases,
    count: () => count,
    destroy: () => {
      for (const dispose of disposers) dispose();
      device.destroy();
    }
  };
}
type Context = ReturnType<typeof createContext>;

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
] as const)('X1 rejects multipart point features before either choropleth backend (%s) and retries', backend => {
  assertAtomicRetry(context => {
    const points = {kind: 'points' as const, positions: context.view('float32x2', 2)};
    const props = {id: 'retry', backend, points, polygons: context.polygons()};
    const featureOffsets = context.view('uint32', 2);
    return {
      invalid: () =>
        Reflect.apply(addPointsInPolygonsChoroplethRecipe, undefined, [
          context.graph,
          {
            ...props,
            points: {...points, featureOffsets}
          }
        ]),
      corrected: () => addPointsInPolygonsChoroplethRecipe(context.graph, props)
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
        addDriveTimeCatchmentRecipe(context.graph, {...props, outputs: {[port]: short}}),
      corrected: () => {
        const result = addDriveTimeCatchmentRecipe(context.graph, {
          ...props,
          outputs: {[port]: corrected}
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
      outputs: {smoothedRates: shared}
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
          outputs: {demandTimes: props.topology.weights}
        }),
      corrected: () =>
        addDriveTimeCatchmentRecipe(context.graph, {...props, outputs: {demandTimes: corrected}})
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
    const props = context.raster();
    let preparation: WeakRef<GPUCommandGraph> | undefined;
    const original = GPUCommandGraph.prototype.createTransientBuffer;
    const spy = vi
      .spyOn(GPUCommandGraph.prototype, 'createTransientBuffer')
      .mockImplementation(function (this: GPUCommandGraph, descriptor) {
        if (this !== context.graph) preparation ??= new WeakRef(this);
        return original.call(this, descriptor);
      });
    try {
      const result = addSurfacePreparationRecipe(context.graph, {...props, maximumDistance: 1});
      expect(preparation).toBeDefined();
      return {
        grid: result.intermediates['grid'],
        preparation: preparation!,
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
  expect(retained.grid).toMatchObject({width: 2, height: 2});
});
