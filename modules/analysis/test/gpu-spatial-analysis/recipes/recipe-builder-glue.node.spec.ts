// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {createRequire} from 'node:module';
import {runInNewContext} from 'node:vm';
import {
  addExcessRiskRecipe,
  addSpatialRateRecipe
} from '../../../src/gpu-spatial-analysis/recipes/rate-mapping-recipes';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  addRecipeCountNodes,
  resolveRecipeDestinations,
  createTransientSpatialWeights,
  RecipeBuilder
} from '../../../src/gpu-spatial-analysis/recipes/recipe-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

import {
  GPUParameterBuffer,
  captureGraphCommandNodes
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {addHotSpotAnalysisRecipe} from '../../../src/gpu-spatial-analysis/recipes/hot-spot-analysis-recipe';
import {addRateClusterMapRecipe} from '../../../src/gpu-spatial-analysis/recipes/rate-cluster-map-recipe';
import {addSpatialRegressionRecipe} from '../../../src/gpu-spatial-analysis/recipes/spatial-regression-recipe';
import {addPeriodComparisonRecipe} from '../../../src/gpu-spatial-analysis/recipes/period-comparison-recipe';
import {addClusterAndOutlineRecipe} from '../../../src/gpu-spatial-analysis/recipes/cluster-and-outline-recipe';
import {addChangeOfSupportRecipe} from '../../../src/gpu-spatial-analysis/recipes/change-of-support-recipe';
import {addStraightLineCatchmentsRecipe} from '../../../src/gpu-spatial-analysis/recipes/straight-line-catchments-recipe';
import {addFleetDwellRecipe} from '../../../src/gpu-spatial-analysis/recipes/fleet-dwell-recipe';
import {addFleetDwellZoneEventsRecipe} from '../../../src/gpu-spatial-analysis/recipes/fleet-dwell-recipe';
import {addInterpolationDiagnosticsRecipe} from '../../../src/gpu-spatial-analysis/recipes/interpolation-diagnostics-recipe';
import {addSurfacePreparationRecipe} from '../../../src/gpu-spatial-analysis/recipes/surface-preparation-recipe';
import {addLocalVariabilityRecipe} from '../../../src/gpu-spatial-analysis/recipes/local-variability-recipe';
import {addCellFlowTopologyRecipe} from '../../../src/gpu-spatial-analysis/recipes/cell-flow-topology-recipe';

it('RecipeBuilder.addGlue lists a named glue stage as a contributor and adds its nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const builder = new RecipeBuilder(graph);
    const values = createTransientView(graph, 'glue-values', 'uint32', 4);
    const node = createWGSLKernelNode(graph, {
      id: 'glue-fill',
      operation: 'Test',
      bindings: [{name: 'values', view: values, type: 'u32', access: 'read_write'}],
      invocationCount: 4,
      body: 'values[valuesOffset + index] = index;'
    });
    builder.addGlue('fill', node);
    expect(builder.glueStages).toEqual([{stage: 'fill', nodes: [node]}]);
    expect(builder.contributors).toHaveLength(1);
    expect(builder.contributors[0].getCommandNodes(graph)).toEqual([node]);
  } finally {
    device.destroy();
  }
});

it('addRecipeCountNodes adds a row-parallel kernel and a reduction per count', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const builder = new RecipeBuilder(graph);
    const validity = createTransientView(graph, 'count-validity', 'uint32', 1000);
    const missing = createTransientView(graph, 'count-missing', 'uint32', 1);
    addRecipeCountNodes(builder, {
      id: 'count-test',
      operation: 'Test',
      rowCount: 1000,
      bindings: [{name: 'validity', view: validity, type: 'u32', access: 'read'}],
      counts: [
        {name: 'missing', output: missing, value: 'u32(validity[validityOffset + index] == 0u)'}
      ]
    });
    expect(builder.glueStages.map(entry => entry.stage)).toEqual(['count-test']);
    // One glue producer for the kernel plus one published GPUReduction for the count.
    expect(builder.contributors.length).toBe(2);
  } finally {
    device.destroy();
  }
});

it('deferred recipes expand every stage before scheduling nodes in declaration order', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const builder = new RecipeBuilder(graph, true);
    const values = createTransientView(graph, 'deferred-values', 'uint32', 1);
    const createNode = (id: string) =>
      createWGSLKernelNode(graph, {
        id,
        operation: 'Test',
        bindings: [{name: 'values', view: values, type: 'u32', access: 'read_write'}],
        invocationCount: 1,
        body: 'values[valuesOffset + index] = 1u;'
      });
    const first = createNode('first');
    const second = createNode('second');
    const add = vi.spyOn(graph, 'add');
    builder.add({getCommandNodes: () => [first]});
    builder.addGlue('second', second);
    expect(add).not.toHaveBeenCalled();
    builder.addDeferredStages();
    expect(add.mock.calls[0][0]).toEqual([first, second]);

    const invalid = new RecipeBuilder(graph, true);
    invalid.addGlue('third', createNode('third'));
    invalid.add({
      getCommandNodes: () => {
        throw new Error('invalid later stage');
      }
    });
    add.mockClear();
    expect(() => invalid.addDeferredStages()).toThrow('invalid later stage');
    expect(add).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

type Format = 'float32' | 'uint32' | 'sint32' | 'float32x2' | 'float32x4' | 'uint32x2' | 'uint32x4';
function createDestinationContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters: GPUParameterBuffer[] = [];
  let bufferCount = 0;
  const view = <ValueFormat extends Format>(format: ValueFormat, length: number) =>
    createTransientView(graph, `input-${bufferCount++}`, format, length);
  function parameter(): GPUParameterBuffer<'float32'>;
  function parameter<ValueFormat extends 'float32' | 'uint32'>(
    format: ValueFormat
  ): GPUParameterBuffer<ValueFormat>;
  function parameter(format: 'float32' | 'uint32' = 'float32'): GPUParameterBuffer {
    const buffer = new GPUParameterBuffer(device, {
      id: `parameters-${parameters.length}`,
      format,
      length: 16
    });
    parameters.push(buffer);
    return buffer;
  }
  const grid = {
    width: 2,
    height: 2,
    affine: [1, 0, 0, 0, 1, 0],
    pixelInterpretation: 'area'
  } as const;
  const polygons = () =>
    ({
      kind: 'polygons',
      positions: view('float32x2', 8),
      featureOffsets: view('uint32', 5),
      polygonOffsets: view('uint32', 5),
      ringOffsets: view('uint32', 5),
      candidateCapacity: 8,
      crossingCapacity: 8
    }) as const;
  const weights = () => ({
    offsets: view('uint32', 5),
    neighbors: view('uint32', 8),
    weights: view('float32', 8)
  });
  const band = () =>
    ({
      id: 'band',
      format: 'float32',
      storage: {kind: 'buffer', values: view('float32', 4)}
    }) as const;
  const trajectory = () =>
    ({
      kind: 'trajectories',
      positions: view('float32x2', 4),
      timestamps: view('float32', 4),
      trackOffsets: view('uint32', 2)
    }) as const;
  return {
    device,
    graph,
    view,
    parameter,
    grid,
    polygons,
    weights,
    band,
    trajectory,
    parameters,
    bufferCount: () => bufferCount,
    destroy() {
      for (const buffer of parameters) buffer.destroy();
      device.destroy();
    }
  };
}
type Context = ReturnType<typeof createDestinationContext>;

for (const name of ['complete', 'medians'] as const)
  it(`B4 catchments validates inherited output ${name} before allocation and retries`, () => {
    const context = createDestinationContext();
    try {
      const destination = context.view('uint32', 1);
      const statistics =
        name === 'complete'
          ? {status: {complete: destination}}
          : (Object.create({medians: context.view('float32', 2)}) as {
              medians: ReturnType<typeof context.view<'float32'>>;
            });
      const props = Object.assign(Object.create({output: {statistics}}), {
        ...destinationFixtures.addStraightLineCatchmentsRecipe(context),
        id: `inherited-output-${name}`
      }) as Parameters<typeof addStraightLineCatchmentsRecipe>[1];
      const allocate = vi.spyOn(context.graph, 'createTransientBuffer');
      const add = vi.spyOn(context.graph, 'add');
      expect(() => addStraightLineCatchmentsRecipe(context.graph, props)).toThrow(
        name === 'complete' ? 'output.statistics.status.complete' : 'output.statistics.medians'
      );
      expect(allocate).not.toHaveBeenCalled();
      expect(add).not.toHaveBeenCalled();
      allocate.mockRestore();
      add.mockRestore();
      expect(() =>
        addStraightLineCatchmentsRecipe(context.graph, {
          ...props,
          output: {statistics: {means: context.view('float32', 2)}}
        })
      ).not.toThrow();
    } finally {
      vi.restoreAllMocks();
      context.destroy();
    }
  });

it.each([
  'areal',
  'catchments'
] as const)('B5 %s accepts foreign-realm maps and retains inherited caller ports', route => {
  const context = createDestinationContext();
  try {
    if (route === 'areal') {
      const count = context.view('uint32', 1);
      const status = runInNewContext('Object.create({})') as {count?: typeof count};
      Object.defineProperty(Object.getPrototypeOf(status), 'count', {value: count});
      const result = addChangeOfSupportRecipe(context.graph, {
        ...destinationFixtures.addChangeOfSupportRecipe(context),
        output: {extensiveWeights: {status}}
      });
      expect(result.outputs.extensiveWeights.status.count).toBe(count);
    } else {
      const means = context.view('float32', 2);
      const statistics = runInNewContext('Object.create({})') as {means?: typeof means};
      Object.defineProperty(Object.getPrototypeOf(statistics), 'means', {value: means});
      const result = addStraightLineCatchmentsRecipe(context.graph, {
        ...destinationFixtures.addStraightLineCatchmentsRecipe(context),
        output: {statistics}
      });
      expect(result.outputs.statistics.means).toBe(means);
    }
  } finally {
    context.destroy();
  }
});

it('ENT-02 preserves cross-format destinations and absent optional descriptors', () => {
  const context = createDestinationContext();
  const common = createRequire(import.meta.url)(
    '@luma.gl/graph/gpu-core'
  ) as typeof import('@luma.gl/graph/gpu-core');
  try {
    const original = context.view('float32', 4);
    const foreign = new common.GraphDataView(original.buffer, original);
    const vector = new common.GraphVectorView({
      id: 'foreign-vector',
      name: 'foreign-vector',
      format: 'float32',
      length: 4,
      valueLength: 4,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: [foreign]
    });
    expect(resolveRecipeDestinations('foreign', {output: foreign})).toBe(foreign);
    expect(resolveRecipeDestinations('foreign-vector', {output: vector})).toBe(vector);
    let statusReads = 0;
    const resolved = resolveRecipeDestinations('optional-overlay', {
      output: {
        get status() {
          statusReads++;
          return {overflow: undefined};
        },
        get format(): never {
          throw new Error('unconsumed format');
        }
      }
    });
    expect(statusReads).toBe(0);
    for (const name of [
      'complete',
      'ringCount',
      'requiredRingCount',
      'nodedSegmentRequiredCount'
    ]) {
      expect(Object.getOwnPropertyDescriptor(resolved?.status, name)).toBeUndefined();
      expect(Object.hasOwn(resolved?.status ?? {}, name)).toBe(false);
    }
    expect(statusReads).toBe(1);
    const events = context.view('float32', 4),
      populations = context.view('float32', 4);
    expect(
      addExcessRiskRecipe(context.graph, {
        id: 'foreign-excess',
        events,
        populations,
        output: foreign
      }).outputs.excessRisk
    ).toBe(foreign);
    const spatial = new common.GraphDataView(context.view('float32', 4).buffer, original);
    expect(
      addSpatialRateRecipe(context.graph, {
        id: 'foreign-rate',
        events,
        populations,
        weights: context.weights(),
        output: spatial
      }).outputs.spatialRates
    ).toBe(spatial);
    const surface = new common.GraphDataView(context.view('float32', 4).buffer, original);
    expect(
      addSurfacePreparationRecipe(context.graph, {
        id: 'foreign-surface',
        grid: context.grid,
        input: context.band(),
        maximumDistance: 1,
        output: {values: surface}
      }).outputs.values
    ).toBe(surface);
    const other = new GPUCommandGraph(context.device);
    const wrongOriginal = createTransientView(other, 'wrong-graph', 'float32', 4);
    const wrong = new common.GraphDataView(wrongOriginal.buffer, wrongOriginal);
    for (const build of [
      () =>
        addExcessRiskRecipe(context.graph, {
          id: 'wrong-excess',
          events,
          populations,
          output: wrong
        }),
      () =>
        addSpatialRateRecipe(context.graph, {
          id: 'wrong-rate',
          events,
          populations,
          weights: context.weights(),
          output: wrong
        }),
      () =>
        addSurfacePreparationRecipe(context.graph, {
          id: 'wrong-surface',
          grid: context.grid,
          input: context.band(),
          maximumDistance: 1,
          output: {values: wrong}
        })
    ])
      expect(build).toThrow(/graph/i);
  } finally {
    context.destroy();
  }
});

const destinationFixtures = {
  addHotSpotAnalysisRecipe: ({view, parameter}: Context) => ({
    parameters: parameter(),
    source: {
      kind: 'lattice',
      width: 2,
      height: 2,
      values: view('float32', 4),
      neighborCapacity: 16
    },
    minimumNeighbors: 0
  }),
  addRateClusterMapRecipe: ({view, parameter, polygons}: Context) => ({
    events: view('float32', 4),
    populations: view('float32', 4),
    polygons: polygons(),
    parameters: parameter(),
    neighborCapacity: 8,
    pairCapacity: 8,
    palette: view('uint32', 5)
  }),
  addSpatialRegressionRecipe: ({view, parameter, weights}: Context) => ({
    predictors: view('float32', 4),
    response: view('float32', 4),
    predictorCount: 1,
    parameters: parameter(),
    weights: weights()
  }),
  addPeriodComparisonRecipe: ({view}: Context) => ({
    family: 'quadbin',
    resolution: 4,
    tableCapacity: 4,
    unionCapacity: 8,
    before: {positions: view('float32x2', 4)},
    after: {positions: view('float32x2', 4)},
    classBreaksParameters: view('float32', 16),
    maximumClassCount: 2,
    methods: ['equal-interval'],
    palette: view('uint32', 4),
    colorScaleParameters: view('float32', 16),
    maximumPaletteCount: 4
  }),
  addClusterAndOutlineRecipe: ({view}: Context) => ({
    points: {kind: 'points', positions: view('float32x2', 4)},
    clusteringParameters: view('float32', 8),
    gridSize: [2, 2],
    geometryParameters: view('float32', 8),
    maximumClusterCount: 4,
    maximumVerticesPerHull: 4,
    hullCapacity: 16
  }),
  addChangeOfSupportRecipe: ({view, grid, polygons}: Context) => ({
    grid,
    parameters: view('float32', 4),
    source: polygons(),
    target: polygons(),
    pairCapacity: 8
  }),
  addStraightLineCatchmentsRecipe: ({view, grid}: Context) => ({
    grid,
    parameters: view('float32', 8),
    facilities: {kind: 'points', positions: view('float32x2', 2)},
    values: view('float32', 4)
  }),
  addFleetDwellRecipe: ({view, polygons, trajectory}: Context) => ({
    trajectories: trajectory(),
    parameters: view('float32', 4),
    stopCapacity: 4,
    zones: polygons()
  }),
  addFleetDwellZoneEventsRecipe: ({view, polygons, trajectory}: Context) => ({
    trajectories: trajectory(),
    zones: polygons(),
    candidateCapacity: 8,
    trackEventCapacity: 4,
    eventCapacity: 4
  }),
  addInterpolationDiagnosticsRecipe: ({view, grid}: Context) => ({
    kriging: {
      positions: view('float32x2', 4),
      values: view('float32', 4),
      parameters: view('float32', 12),
      grid,
      indexGridSize: [2, 2],
      indexBounds: [0, 0, 2, 2],
      maximumNeighborCount: 4
    }
  }),
  addSurfacePreparationRecipe: ({grid, band}: Context) => ({
    grid,
    input: band(),
    maximumDistance: 1,
    filter: {radius: 1}
  }),
  addLocalVariabilityRecipe: ({view, weights}: Context) => ({
    values: view('float32', 4),
    weights: weights(),
    parameters: view('float32', 8)
  }),
  addCellFlowTopologyRecipe: ({view}: Context) => ({
    cells: view('uint32x2', 2),
    origins: view('uint32', 2),
    destinations: view('uint32', 2),
    values: view('float32', 2)
  })
} satisfies {
  [Name in keyof typeof destinationBuilders]: (
    context: Context
  ) => Parameters<(typeof destinationBuilders)[Name]>[1];
};
const destinationBuilders = {
  addHotSpotAnalysisRecipe,
  addRateClusterMapRecipe,
  addSpatialRegressionRecipe,
  addPeriodComparisonRecipe,
  addClusterAndOutlineRecipe,
  addChangeOfSupportRecipe,
  addStraightLineCatchmentsRecipe,
  addFleetDwellRecipe,
  addFleetDwellZoneEventsRecipe,
  addInterpolationDiagnosticsRecipe,
  addSurfacePreparationRecipe,
  addLocalVariabilityRecipe,
  addCellFlowTopologyRecipe
};
const destinationFixtureCache = new WeakMap<Context, Map<string, object>>();
function getDestinationFixture<Name extends keyof typeof destinationFixtures>(
  name: Name,
  context: Context
): ReturnType<(typeof destinationFixtures)[Name]> {
  let fixtures = destinationFixtureCache.get(context);
  if (!fixtures) {
    fixtures = new Map();
    destinationFixtureCache.set(context, fixtures);
  }
  let props = fixtures.get(name);
  if (!props) {
    props = destinationFixtures[name](context);
    fixtures.set(name, props);
  }
  return props as ReturnType<(typeof destinationFixtures)[Name]>;
}

const destinationCases = [
  {
    name: 'HotSpotAnalysis',
    path: 'zScores',
    build: (
      context: Context,
      id: string,
      output?: import('@luma.gl/graph/gpu-core').GraphDataView,
      outputs?: import('@luma.gl/graph/gpu-core').GraphDataView
    ) =>
      addHotSpotAnalysisRecipe(context.graph, {
        ...getDestinationFixture('addHotSpotAnalysisRecipe', context),
        id,
        output: output
          ? {zScores: output as import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>}
          : undefined,
        ...(outputs === undefined ? {} : {outputs})
      })
  },
  {
    name: 'RateClusterMap',
    path: 'standardizedRates',
    build: (
      context: Context,
      id: string,
      output?: import('@luma.gl/graph/gpu-core').GraphDataView,
      outputs?: import('@luma.gl/graph/gpu-core').GraphDataView
    ) =>
      addRateClusterMapRecipe(context.graph, {
        ...getDestinationFixture('addRateClusterMapRecipe', context),
        id,
        output: output
          ? {
              standardizedRates:
                output as import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>
            }
          : undefined,
        ...(outputs === undefined ? {} : {outputs})
      })
  },
  {
    name: 'SpatialRegression',
    path: 'ols.coefficients',
    build: (
      context: Context,
      id: string,
      output?: import('@luma.gl/graph/gpu-core').GraphDataView,
      outputs?: import('@luma.gl/graph/gpu-core').GraphDataView
    ) =>
      addSpatialRegressionRecipe(context.graph, {
        ...getDestinationFixture('addSpatialRegressionRecipe', context),
        id,
        output: output
          ? {
              ols: {
                coefficients: output as import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>
              }
            }
          : undefined,
        ...(outputs === undefined ? {} : {outputs})
      })
  },
  {
    name: 'PeriodComparison',
    path: 'breaks',
    build: (
      context: Context,
      id: string,
      output?: import('@luma.gl/graph/gpu-core').GraphDataView,
      outputs?: import('@luma.gl/graph/gpu-core').GraphDataView
    ) =>
      addPeriodComparisonRecipe(context.graph, {
        ...getDestinationFixture('addPeriodComparisonRecipe', context),
        id,
        output: output
          ? {breaks: output as import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>}
          : undefined,
        ...(outputs === undefined ? {} : {outputs})
      })
  },
  {
    name: 'ClusterAndOutline',
    path: 'labels',
    build: (
      context: Context,
      id: string,
      output?: import('@luma.gl/graph/gpu-core').GraphDataView,
      outputs?: import('@luma.gl/graph/gpu-core').GraphDataView
    ) =>
      addClusterAndOutlineRecipe(context.graph, {
        ...getDestinationFixture('addClusterAndOutlineRecipe', context),
        id,
        output: output
          ? {labels: output as import('@luma.gl/graph/gpu-core').GraphDataView<'uint32'>}
          : undefined,
        ...(outputs === undefined ? {} : {outputs})
      })
  },
  {
    name: 'ChangeOfSupport',
    path: 'areas',
    build: (
      context: Context,
      id: string,
      output?: import('@luma.gl/graph/gpu-core').GraphDataView,
      outputs?: import('@luma.gl/graph/gpu-core').GraphDataView
    ) =>
      addChangeOfSupportRecipe(context.graph, {
        ...getDestinationFixture('addChangeOfSupportRecipe', context),
        id,
        output: output
          ? {areas: output as import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>}
          : undefined,
        ...(outputs === undefined ? {} : {outputs})
      })
  },
  {
    name: 'StraightLineCatchments',
    path: 'distances',
    build: (
      context: Context,
      id: string,
      output?: import('@luma.gl/graph/gpu-core').GraphDataView,
      outputs?: import('@luma.gl/graph/gpu-core').GraphDataView
    ) =>
      addStraightLineCatchmentsRecipe(context.graph, {
        ...getDestinationFixture('addStraightLineCatchmentsRecipe', context),
        id,
        output: output
          ? {distances: output as import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>}
          : undefined,
        ...(outputs === undefined ? {} : {outputs})
      })
  },
  {
    name: 'FleetDwell',
    path: 'stops.durations',
    build: (
      context: Context,
      id: string,
      output?: import('@luma.gl/graph/gpu-core').GraphDataView,
      outputs?: import('@luma.gl/graph/gpu-core').GraphDataView
    ) =>
      addFleetDwellRecipe(context.graph, {
        ...getDestinationFixture('addFleetDwellRecipe', context),
        id,
        output: output
          ? {
              stops: {
                durations: output as import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>
              }
            }
          : undefined,
        ...(outputs === undefined ? {} : {outputs})
      })
  },
  {
    name: 'FleetDwellZoneEvents',
    path: 'dwellTimes',
    build: (
      context: Context,
      id: string,
      output?: import('@luma.gl/graph/gpu-core').GraphDataView,
      outputs?: import('@luma.gl/graph/gpu-core').GraphDataView
    ) =>
      addFleetDwellZoneEventsRecipe(context.graph, {
        ...getDestinationFixture('addFleetDwellZoneEventsRecipe', context),
        id,
        output: output
          ? {dwellTimes: output as import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>}
          : undefined,
        ...(outputs === undefined ? {} : {outputs})
      })
  },
  {
    name: 'InterpolationDiagnostics',
    path: 'values',
    build: (
      context: Context,
      id: string,
      output?: import('@luma.gl/graph/gpu-core').GraphDataView,
      outputs?: import('@luma.gl/graph/gpu-core').GraphDataView
    ) =>
      addInterpolationDiagnosticsRecipe(context.graph, {
        ...getDestinationFixture('addInterpolationDiagnosticsRecipe', context),
        id,
        output: output
          ? {values: output as import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>}
          : undefined,
        ...(outputs === undefined ? {} : {outputs})
      })
  },
  {
    name: 'SurfacePreparation',
    path: 'values',
    build: (
      context: Context,
      id: string,
      output?: import('@luma.gl/graph/gpu-core').GraphDataView,
      outputs?: import('@luma.gl/graph/gpu-core').GraphDataView
    ) =>
      addSurfacePreparationRecipe(context.graph, {
        ...getDestinationFixture('addSurfacePreparationRecipe', context),
        id,
        output: output
          ? {values: output as import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>}
          : undefined,
        ...(outputs === undefined ? {} : {outputs})
      })
  },
  {
    name: 'LocalVariability',
    path: 'losh',
    build: (
      context: Context,
      id: string,
      output?: import('@luma.gl/graph/gpu-core').GraphDataView,
      outputs?: import('@luma.gl/graph/gpu-core').GraphDataView
    ) =>
      addLocalVariabilityRecipe(context.graph, {
        ...getDestinationFixture('addLocalVariabilityRecipe', context),
        id,
        output: output
          ? {losh: output as import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>}
          : undefined,
        ...(outputs === undefined ? {} : {outputs})
      })
  },
  {
    name: 'CellFlowTopology',
    path: 'outgoingTotals',
    build: (
      context: Context,
      id: string,
      output?: import('@luma.gl/graph/gpu-core').GraphDataView,
      outputs?: import('@luma.gl/graph/gpu-core').GraphDataView
    ) =>
      addCellFlowTopologyRecipe(context.graph, {
        ...getDestinationFixture('addCellFlowTopologyRecipe', context),
        id,
        output: output
          ? {outgoingTotals: output as import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>}
          : undefined,
        ...(outputs === undefined ? {} : {outputs})
      })
  }
];
for (const {name, path, build} of destinationCases) {
  it(`${name} preserves destination identity and rejects the removed prop before a same-ID retry`, () => {
    const context = createDestinationContext();
    try {
      // Each callback is checked against its actual public builder, including output spelling.
      const baseline = build(context, 'layout');
      const read = (record: object): unknown =>
        path
          .split('.')
          .reduce<unknown>((value, key) => (value as Record<string, unknown>)[key], record);
      const layout = read(baseline.outputs) as import('@luma.gl/graph/gpu-core').GraphDataView;
      const supplied = context.view(layout.format as Format, layout.length);
      const conflict = context.view(layout.format as Format, layout.length);
      const allocate = vi.spyOn(context.graph, 'createTransientBuffer');
      const importBuffer = vi.spyOn(context.graph, 'importBuffer');
      const add = vi.spyOn(context.graph, 'add');
      expect(() => build(context, 'retry', supplied, conflict)).toThrow(
        /outputs was removed; use output/
      );
      expect(() => build(context, 'retry', undefined, supplied)).toThrow(
        /outputs was removed; use output/
      );
      expect(add).not.toHaveBeenCalled();
      expect(allocate).not.toHaveBeenCalled();
      expect(importBuffer).not.toHaveBeenCalled();
      expect(read(build(context, 'retry', supplied).outputs)).toBe(supplied);
    } finally {
      context.destroy();
    }
  });
}

it('recipe destinations capture nested leaves and preserve bare views', () => {
  const context = createDestinationContext();
  try {
    const first = context.view('uint32', 1);
    const second = context.view('uint32', 1);
    const output = {status: {overflow: first, requiredCount: second}};
    const resolved = resolveRecipeDestinations('snapshot', {output});
    expect(resolved?.status.overflow).toBe(first);
    expect(resolved?.status.requiredCount).toBe(second);
    expect(resolved?.status).not.toBe(output.status);
    output.status.overflow = second;
    expect(resolved?.status.overflow).toBe(first);
    expect(resolveRecipeDestinations('bare', {output: first})).toBe(first);
    const retired = {output: first, outputs: second};
    expect(() => resolveRecipeDestinations('bare', retired)).toThrow(
      /bare.*outputs was removed; use output/
    );
    const inherited = Object.create({
      status: Object.defineProperty({}, 'overflow', {get: () => first})
    }) as typeof output;
    expect(resolveRecipeDestinations('inherited', {output: inherited})?.status.overflow).toBe(
      first
    );
  } finally {
    context.destroy();
  }
});

it('recipe spatial weights keep capacity validation through the defining-module import', () => {
  const context = createDestinationContext();
  try {
    expect(() => createTransientSpatialWeights(context.graph, 'invalid', 2, -1)).toThrow(
      /capacity/
    );
    const weights = createTransientSpatialWeights(context.graph, 'valid', 2, 3);
    expect(weights.offsets.length).toBe(3);
    expect(weights.neighbors.length).toBe(3);
    expect(weights.weights.length).toBe(3);
  } finally {
    context.destroy();
  }
});

it('change-of-support retains raster status tuples as arrays', () => {
  const context = createDestinationContext();
  try {
    const tuple: [
      ReturnType<typeof context.view<'uint32'>>,
      ReturnType<typeof context.view<'uint32'>>
    ] = [context.view('uint32', 1), context.view('uint32', 1)];
    const result = addChangeOfSupportRecipe(context.graph, {
      ...destinationFixtures.addChangeOfSupportRecipe(context),
      output: {
        status: {
          rasterRequiredCount: tuple,
          rasterOverflow: tuple.map(() => context.view('uint32', 1)) as typeof tuple
        }
      }
    });
    expect(Array.isArray(result.status.rasterRequiredCount)).toBe(true);
    expect(result.status.rasterRequiredCount).not.toBe(tuple);
    expect(result.status.rasterRequiredCount?.[0]).toBe(tuple[0]);
    expect(result.status.rasterRequiredCount?.[1]).toBe(tuple[1]);
  } finally {
    context.destroy();
  }
});

it('catchments reads proxy destinations and ignores unrelated getters', () => {
  const context = createDestinationContext();
  try {
    const distances = context.view('float32', 4);
    const proxy = new Proxy(
      {},
      {get: (_target, name) => (name === 'distances' ? distances : undefined)}
    );
    const first = addStraightLineCatchmentsRecipe(context.graph, {
      ...destinationFixtures.addStraightLineCatchmentsRecipe(context),
      id: 'proxy',
      output: proxy
    });
    expect(first.outputs.distances).toBe(distances);
    const outputs = {
      distances,
      get preview(): never {
        throw new Error('unconsumed preview');
      }
    };
    const second = addStraightLineCatchmentsRecipe(context.graph, {
      ...destinationFixtures.addStraightLineCatchmentsRecipe(context),
      id: 'getter',
      output: outputs
    });
    expect(second.outputs.distances).toBe(distances);
  } finally {
    context.destroy();
  }
});

it('WRT-02 change-of-support retains distinct supported nested status destinations and rejects unwritten ports', () => {
  const context = createDestinationContext();
  try {
    const props = destinationFixtures.addChangeOfSupportRecipe(context);
    const requiredCount = context.view('uint32', 1);
    const nestedRequiredCount = context.view('uint32', 1);
    const count = context.view('uint32', 1);
    const candidateOverflow = context.view('uint32', 1);
    const overflow = context.view('uint32', 1);
    let result!: ReturnType<typeof addChangeOfSupportRecipe>;
    const nodes = captureGraphCommandNodes(context.graph, () => {
      result = addChangeOfSupportRecipe(context.graph, {
        ...props,
        output: {
          status: {requiredCount},
          extensiveWeights: {
            status: {overflow, requiredCount: nestedRequiredCount, count, candidateOverflow}
          }
        }
      });
    });
    expect(result.outputs.extensiveWeights.status.requiredCount).toBe(nestedRequiredCount);
    expect(result.outputs.extensiveWeights.status.count).toBe(count);
    expect(result.outputs.extensiveWeights.status.candidateOverflow).toBe(candidateOverflow);
    expect(result.status.requiredCount).toBe(requiredCount);
    expect(
      nodes.some(node =>
        node.resources?.some(
          resource =>
            'buffer' in resource &&
            resource.buffer === requiredCount &&
            resource.usage === 'storage-read-write'
        )
      )
    ).toBe(true);
    for (const name of ['complete', 'unresolvedCount', 'invalidCount', 'unexpected']) {
      const destination = context.view('uint32', 1);
      for (const [index, status] of [
        {[name]: destination},
        Object.create({[name]: destination}) as Record<string, typeof destination>,
        Object.defineProperty({}, name, {value: destination})
      ].entries()) {
        expect(() =>
          addChangeOfSupportRecipe(context.graph, {
            ...props,
            id: `unsupported-${name}-${index}`,
            output: {extensiveWeights: {status}}
          })
        ).toThrow(`output.extensiveWeights.status.${name}`);
        expect(() =>
          addChangeOfSupportRecipe(context.graph, {...props, id: `unsupported-${name}-${index}`})
        ).not.toThrow();
      }
    }
    expect(() =>
      addChangeOfSupportRecipe(context.graph, {
        ...props,
        id: 'distances',
        output: {extensiveWeights: {distances: context.view('float32', 8)}}
      })
    ).toThrow(/distances.*not supported/);
  } finally {
    context.destroy();
  }
});

it('W5-15 catchments rejects unsupported moments/dictionaries before allocation and preserves partial destinations', () => {
  const context = createDestinationContext();
  try {
    const props = destinationFixtures.addStraightLineCatchmentsRecipe(context);
    for (const name of [
      'weightSums',
      'variances',
      'populationVariances',
      'standardDeviations',
      'sourceZoneIds',
      'denseZoneRows'
    ]) {
      const statistics = {sums: context.view('float32', 2), [name]: context.view('float32', 2)};
      const allocate = vi.spyOn(context.graph, 'createTransientBuffer');
      expect(() =>
        addStraightLineCatchmentsRecipe(context.graph, {
          ...props,
          id: `unsupported-${name}`,
          output: {statistics}
        })
      ).toThrow(`output.statistics.${name}`);
      expect(allocate).not.toHaveBeenCalled();
      allocate.mockRestore();
      expect(() =>
        addStraightLineCatchmentsRecipe(context.graph, {
          ...props,
          id: `unsupported-${name}`,
          output: {statistics: {sums: statistics.sums}}
        })
      ).not.toThrow();
    }
    const columns = {
      cellCounts: context.view('uint32', 2),
      valueCounts: context.view('uint32', 2),
      sums: context.view('float32', 2),
      means: context.view('float32', 2),
      minimums: context.view('float32', 2),
      maximums: context.view('float32', 2)
    };
    for (const name of [
      'cellCounts',
      'valueCounts',
      'sums',
      'means',
      'minimums',
      'maximums'
    ] as const) {
      const result = addStraightLineCatchmentsRecipe(context.graph, {
        ...props,
        id: `partial-${name}`,
        output: {statistics: {[name]: columns[name]}}
      });
      expect(result.outputs.statistics[name]).toBe(columns[name]);
    }
  } finally {
    context.destroy();
  }
});

it('catchments rejects conflicting overflow aliases and unsupported incomplete destinations', () => {
  const context = createDestinationContext();
  try {
    const props = destinationFixtures.addStraightLineCatchmentsRecipe(context);
    const overflow = context.view('uint32', 1);
    const statistics = {
      cellCounts: context.view('uint32', 2),
      valueCounts: context.view('uint32', 2),
      sums: context.view('float32', 2),
      means: context.view('float32', 2),
      minimums: context.view('float32', 2),
      maximums: context.view('float32', 2),
      status: {overflow: context.view('uint32', 1)}
    };
    expect(() =>
      addStraightLineCatchmentsRecipe(context.graph, {
        ...props,
        output: {status: {overflow}, statistics}
      })
    ).toThrow(/overflow.*same/);
    expect(() =>
      addStraightLineCatchmentsRecipe(context.graph, {
        ...props,
        output: {statistics: {...statistics, status: {...statistics.status, incomplete: overflow}}}
      })
    ).toThrow(/incomplete.*not supported/);
    const result = addStraightLineCatchmentsRecipe(context.graph, {
      ...props,
      output: {status: {overflow}, statistics: {...statistics, status: {overflow}}}
    });
    expect(result.outputs.statistics.status.overflow).toBe(overflow);
  } finally {
    context.destroy();
  }
});

it('change-of-support rejects disabled destinations and binds enabled category and value writers', () => {
  const context = createDestinationContext();
  try {
    const props = destinationFixtures.addChangeOfSupportRecipe(context);
    const categoryShares = context.view('float32', 8);
    const extensiveValues = context.view('float32', 4);
    const intensiveValues = context.view('float32', 4);
    for (const output of [{categoryShares}, {extensiveValues}, {intensiveValues}]) {
      expect(() => addChangeOfSupportRecipe(context.graph, {...props, output})).toThrow(
        /output.*(categories|sourceValues)/
      );
    }
    const nodes = captureGraphCommandNodes(context.graph, () => {
      addChangeOfSupportRecipe(context.graph, {
        ...props,
        categories: {categoryCount: 2, sourceCategories: context.view('uint32', 4)},
        sourceValues: context.view('float32', 4),
        output: {categoryShares, extensiveValues, intensiveValues}
      });
    });
    for (const view of [categoryShares, extensiveValues, intensiveValues]) {
      expect(
        nodes.some(node =>
          node.resources?.some(
            resource =>
              'buffer' in resource &&
              resource.buffer === view &&
              resource.usage === 'storage-read-write'
          )
        )
      ).toBe(true);
    }
  } finally {
    context.destroy();
  }
});

it('optional hot-spot, rate-cluster and local-fit destinations require their enabling stages', () => {
  const context = createDestinationContext();
  try {
    expect(() =>
      addHotSpotAnalysisRecipe(context.graph, {
        ...destinationFixtures.addHotSpotAnalysisRecipe(context),
        output: {color: {colors: context.view('uint32', 4)}}
      })
    ).toThrow(/output.color.*color/);
    expect(() =>
      addHotSpotAnalysisRecipe(context.graph, {
        ...destinationFixtures.addHotSpotAnalysisRecipe(context),
        output: {permutation: {pseudoPValues: context.view('float32', 4)}}
      })
    ).toThrow(/output.permutation.*permutation/);
    expect(() =>
      addRateClusterMapRecipe(context.graph, {
        ...destinationFixtures.addRateClusterMapRecipe(context),
        output: {permutation: {pseudoPValues: context.view('float32', 4)}}
      })
    ).toThrow(/output.permutation.*permutation/);
    expect(() =>
      addRateClusterMapRecipe(context.graph, {
        ...destinationFixtures.addRateClusterMapRecipe(context),
        permutation: {parameters: context.parameter('uint32'), maximumPermutations: 9},
        output: {analytic: {pValues: context.view('float32', 4)}}
      })
    ).toThrow(/output.analytic.*permutation/);
    expect(() =>
      addSpatialRegressionRecipe(context.graph, {
        ...destinationFixtures.addSpatialRegressionRecipe(context),
        output: {localFits: {coefficients: context.view('float32', 8)}}
      })
    ).toThrow(/output.localFits.*localFits/);
  } finally {
    context.destroy();
  }
});

it('filtered surface preparation binds terminal values to the filter writer', () => {
  const context = createDestinationContext();
  try {
    const values = context.view('float32', 4);
    const nodes = captureGraphCommandNodes(context.graph, () => {
      addSurfacePreparationRecipe(context.graph, {
        ...destinationFixtures.addSurfacePreparationRecipe(context),
        output: {values}
      });
    });
    expect(
      nodes.some(
        node =>
          node.id.includes('filter') &&
          node.resources?.some(
            resource =>
              'buffer' in resource &&
              resource.buffer === values &&
              resource.usage === 'storage-read-write'
          )
      )
    ).toBe(true);
  } finally {
    context.destroy();
  }
});

it('recipe capacity planning has no runtime edge to the spatial-weights barrel', async () => {
  const {readFileSync} = await import('node:fs');
  const source = readFileSync(
    new URL('../../../src/gpu-spatial-analysis/recipes/recipe-utils.ts', import.meta.url),
    'utf8'
  );
  const runtimeImport = source.match(
    /import\s*\{[^}]*getGPUSpatialWeightsCapacityPlan[^}]*\}\s*from\s*'([^']+)'/
  )?.[1];
  expect(runtimeImport).toBe('../spatial-weights/spatial-weights');
});

it('matching tuple aliases remain arrays and bind both raster required-count writers', () => {
  const context = createDestinationContext();
  try {
    const rasterRequiredCount: [
      ReturnType<typeof context.view<'uint32'>>,
      ReturnType<typeof context.view<'uint32'>>
    ] = [context.view('uint32', 1), context.view('uint32', 1)];
    let result!: ReturnType<typeof addChangeOfSupportRecipe>;
    const nodes = captureGraphCommandNodes(context.graph, () => {
      result = addChangeOfSupportRecipe(context.graph, {
        ...destinationFixtures.addChangeOfSupportRecipe(context),
        output: {status: {rasterRequiredCount}}
      });
    });
    expect(Array.isArray(result.status.rasterRequiredCount)).toBe(true);
    for (const [index, view] of rasterRequiredCount.entries()) {
      expect(result.status.rasterRequiredCount?.[index]).toBe(view);
      expect(
        nodes.some(node =>
          node.resources?.some(
            resource =>
              'buffer' in resource &&
              resource.buffer === view &&
              resource.usage === 'storage-read-write'
          )
        )
      ).toBe(true);
    }
  } finally {
    context.destroy();
  }
});

it('enabled optional hot-spot outputs retain their writers', () => {
  const context = createDestinationContext();
  try {
    const colors = context.view('uint32', 4);
    const pseudoPValues = context.view('float32', 4);
    const nodes = captureGraphCommandNodes(context.graph, () => {
      addHotSpotAnalysisRecipe(context.graph, {
        ...destinationFixtures.addHotSpotAnalysisRecipe(context),
        color: {
          palette: context.view('uint32', 4),
          classBreaksParameters: context.view('float32', 16),
          colorScaleParameters: context.view('float32', 16),
          maximumClassCount: 2,
          maximumPaletteCount: 4,
          methods: ['equal-interval']
        },
        permutation: {parameters: context.parameter('uint32'), maximumPermutations: 9},
        output: {color: {colors}, permutation: {pseudoPValues}}
      });
    });
    for (const view of [colors, pseudoPValues]) {
      expect(
        nodes.some(node =>
          node.resources?.some(
            resource =>
              'buffer' in resource &&
              resource.buffer === view &&
              resource.usage === 'storage-read-write'
          )
        )
      ).toBe(true);
    }
  } finally {
    context.destroy();
  }
});

for (const permutation of [false, true]) {
  it(`rate-cluster binds its enabled ${permutation ? 'permutation' : 'analytic'} destination`, () => {
    const context = createDestinationContext();
    try {
      const pValues = context.view('float32', 4);
      const nodes = captureGraphCommandNodes(context.graph, () => {
        addRateClusterMapRecipe(context.graph, {
          ...destinationFixtures.addRateClusterMapRecipe(context),
          ...(permutation
            ? {
                permutation: {parameters: context.parameter('uint32'), maximumPermutations: 9},
                output: {permutation: {pseudoPValues: pValues}}
              }
            : {output: {analytic: {pValues}}})
        });
      });
      expect(
        nodes.some(node =>
          node.resources?.some(
            resource =>
              'buffer' in resource &&
              resource.buffer === pValues &&
              resource.usage === 'storage-read-write'
          )
        )
      ).toBe(true);
    } finally {
      context.destroy();
    }
  });
}

it('spatial regression binds enabled local-fit coefficients', () => {
  const context = createDestinationContext();
  try {
    const coefficients = context.view('float32', 8);
    const nodes = captureGraphCommandNodes(context.graph, () => {
      addSpatialRegressionRecipe(context.graph, {
        ...destinationFixtures.addSpatialRegressionRecipe(context),
        localFits: {
          positions: context.view('float32x2', 4),
          parameters: context.view('float32', 16),
          maximumBandwidthCount: 2,
          maximumNeighborCount: 4
        },
        output: {localFits: {coefficients}}
      });
    });
    expect(
      nodes.some(node =>
        node.resources?.some(
          resource =>
            'buffer' in resource &&
            resource.buffer === coefficients &&
            resource.usage === 'storage-read-write'
        )
      )
    ).toBe(true);
  } finally {
    context.destroy();
  }
});

it('recipe kernels reject a storage format incompatible with the declared WGSL scalar', async () => {
  const {createRecipeKernelNode} = await import(
    '../../../src/gpu-spatial-analysis/recipes/recipe-utils'
  );
  const context = createDestinationContext();
  try {
    const wrong = context.view('uint32', 4);
    const bindings = [{name: 'values', view: wrong, type: 'f32', access: 'read'}] as const;
    expect(() =>
      createRecipeKernelNode(context.graph, {
        id: 'wrong-format',
        operation: 'Test',
        bindings,
        invocationCount: 4,
        body: 'let value = values[valuesOffset + index];'
      })
    ).toThrow(/float32/);
    expect(() =>
      addRecipeCountNodes(new RecipeBuilder(context.graph), {
        id: 'wrong-count-format',
        operation: 'Test',
        rowCount: 4,
        bindings,
        counts: []
      })
    ).toThrow(/float32/);
  } finally {
    context.destroy();
  }
});

it('hot-spot source-specific destinations require their source route', () => {
  const context = createDestinationContext();
  try {
    const props = destinationFixtures.addHotSpotAnalysisRecipe(context);
    const scalar = context.view('uint32', 1);
    expect(() =>
      addHotSpotAnalysisRecipe(context.graph, {
        ...props,
        output: {studyArea: {status: {requiredCount: scalar}}}
      })
    ).toThrow(/output.studyArea.*points/);
    expect(() =>
      addHotSpotAnalysisRecipe(context.graph, {
        ...props,
        output: {status: {pointsOutsideStudyArea: scalar}}
      })
    ).toThrow(/pointsOutsideStudyArea.*points/);
    expect(() =>
      addHotSpotAnalysisRecipe(context.graph, {
        ...props,
        output: {status: {weightsInvalidCount: scalar}}
      })
    ).toThrow(/weightsInvalidCount.*distanceBand/);
  } finally {
    context.destroy();
  }
});

it('cluster outlines reject destinations for the inactive hull route', () => {
  const context = createDestinationContext();
  try {
    const props = destinationFixtures.addClusterAndOutlineRecipe(context);
    let alpha!: ReturnType<typeof addClusterAndOutlineRecipe>;
    const nodes = captureGraphCommandNodes(context.graph, () => {
      alpha = addClusterAndOutlineRecipe(context.graph, {
        ...props,
        id: 'alpha',
        outline: {kind: 'alpha', maximumCircumradius: 1}
      });
    });
    expect(nodes.length).toBeGreaterThan(0);
    expect(() =>
      addClusterAndOutlineRecipe(context.graph, {
        ...props,
        output: {concaveHull: alpha.concaveHull}
      })
    ).toThrow(/concaveHull.*alpha/);
    expect(() =>
      addClusterAndOutlineRecipe(context.graph, {
        ...props,
        outline: {kind: 'alpha', maximumCircumradius: 1},
        output: {hullPositions: context.view('float32x2', 16)}
      })
    ).toThrow(/hullPositions.*convex/);
    const output = alpha.concaveHull;
    const result = addClusterAndOutlineRecipe(context.graph, {
      ...props,
      id: 'alpha-supplied',
      outline: {kind: 'alpha', maximumCircumradius: 1},
      output: {concaveHull: output}
    });
    expect(result.outputs.concaveHull?.triangles).toBe(output?.triangles);
  } finally {
    context.destroy();
  }
});
