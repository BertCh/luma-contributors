// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it, vi} from 'vitest';
import {readFileSync, readdirSync, writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {join} from 'node:path';
import ts from 'typescript';
import * as spatialRecipes from '../../../src/gpu-spatial-analysis/recipes/index';
import * as networkRecipes from '../../../src/gpu-network/recipes/index';
import * as publicSpatialRecipes from '../../../src/gpu-spatial-analysis/index';
import * as publicNetworkRecipes from '../../../src/gpu-network/index';
import * as publicTerrainRecipes from '../../../src/gpu-terrain/index';
import * as publicRasterRecipes from '../../../src/raster-analysis/index';
import * as publicDataframeRecipes from '../../../src/dataframe-analysis/index';
import * as publicPointcloudRecipes from '../../../src/pointcloud-analysis/index';
import * as publicGeospatialRecipes from '../../../src/geospatial-analysis/index';
import * as publicCoreRecipes from '../../../src/analysis-core/index';
import * as publicRecordRecipes from '../../../src/computation-record/index';
const recipes = {...spatialRecipes, ...networkRecipes};
import {RecipeBuilder} from '../../../src/gpu-spatial-analysis/recipes/recipe-utils';
import * as preparation from '../../../src/gpu-spatial-analysis/recipes/prepared-recipe';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {auditRecipeResult} from './recipe-result-audit';
import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';
import {createStopValueCubeFixture} from './stop-value-cube-fixture';

// Opt-in mutation run proves that bypassing preparation reproduces the original defect.
vi.mock('../../../src/gpu-spatial-analysis/recipes/prepared-recipe', async importOriginal => {
  const original =
    await importOriginal<
      typeof import('../../../src/gpu-spatial-analysis/recipes/prepared-recipe')
    >();
  return {
    ...original,
    addPreparedRecipe:
      process.env['RECIPE_ROLLBACK_BASELINE'] === '1'
        ? (((graph, props, declare) => declare(graph, props)) as typeof original.addPreparedRecipe)
        : original.addPreparedRecipe
  };
});

type Format = 'float32' | 'uint32' | 'sint32' | 'float32x2' | 'float32x4' | 'uint32x2' | 'uint32x4';
function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters: GPUParameterBuffer[] = [];
  let bufferCount = 0;
  const view = <ValueFormat extends Format>(format: ValueFormat, length: number) =>
    createTransientView(graph, `input-${bufferCount++}`, format, length);
  const memory = createCPUGraphMemory();
  const input = <ValueFormat extends Format>(
    format: ValueFormat,
    values: Uint32Array | Float32Array,
    length: number
  ) => {
    const result = view(format, length);
    memory.write(result, values);
    return result;
  };
  const parameter = (format: 'float32' | 'uint32' = 'float32') => {
    const buffer = new GPUParameterBuffer(device, {
      id: `parameters-${parameters.length}`,
      format,
      length: 16
    });
    parameters.push(buffer);
    return buffer;
  };
  const grid = {
    width: 2,
    height: 2,
    affine: [1, 0, 0, 0, 1, 0],
    pixelInterpretation: 'area'
  } as const;
  const polygons = () => ({
    kind: 'polygons',
    positions: view('float32x2', 8),
    featureOffsets: view('uint32', 5),
    polygonOffsets: view('uint32', 5),
    ringOffsets: view('uint32', 5),
    candidateCapacity: 8,
    crossingCapacity: 8
  });
  const weights = () => ({
    offsets: view('uint32', 5),
    neighbors: view('uint32', 8),
    weights: view('float32', 8)
  });
  const band = () => ({
    id: 'band',
    format: 'float32',
    storage: {kind: 'buffer', values: view('float32', 4)}
  });
  const trajectory = () => ({
    kind: 'trajectories',
    positions: view('float32x2', 4),
    timestamps: view('float32', 4),
    trackOffsets: view('uint32', 2)
  });
  return {
    device,
    graph,
    view,
    input,
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
type Context = ReturnType<typeof createContext>;

// Intentionally small fixed shapes. Shader execution is covered by the corresponding headless
// recipes; this matrix discriminates host declaration failures at every contributor/glue stage.
const fixtures = {
  addStopValueCubeRecipe: ({input}: Context) => createStopValueCubeFixture(input),
  addDriveTimeCatchmentRecipe: ({view}: Context) => ({
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
  }),
  addNetworkDisruptionRecipe: ({view}: Context) => ({
    search: {
      topology: {
        offsets: view('uint32', 4),
        neighbors: view('uint32', 4),
        weights: view('float32', 4)
      },
      startNodes: view('uint32', 2),
      maximumIterations: 2
    },
    rowCount: 2,
    edgeMask: view('uint32', 4),
    opportunityWeights: view('float32', 3),
    parameters: view('float32', 4)
  }),
  addFacilityScenarioRecipe: ({view}: Context) => ({
    costs: view('float32', 6),
    demand: view('float32', 3),
    supply: view('float32', 2),
    facilityMask: view('uint32', 2),
    parameters: view('float32', 4)
  }),
  addTrackQualityRecipe: ({view, trajectory}: Context) => ({
    trajectory: trajectory(),
    thresholds: view('float32', 2),
    similarity: {pairA: view('uint32', 2), pairB: view('uint32', 2)}
  }),
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
  addSpaceTimeHotSpotsRecipe: ({view, parameter}: Context) => ({
    timestamps: view('uint32x2', 4),
    calendarParameters: view('sint32', 2),
    slices: {step: 'month', origin: {year: 2023}, count: 6},
    cells: {
      kind: 'lattice',
      positions: view('float32x2', 4),
      width: 2,
      height: 2,
      bounds: [0, 0, 2, 2]
    },
    settings: {temporalWindow: 1},
    parameters: parameter()
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
  addPointsInPolygonsChoroplethRecipe: ({view, polygons}: Context) => ({
    points: {kind: 'points', positions: view('float32x2', 4)},
    polygons: polygons(),
    statistic: 'count'
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
  addZonalDistributionRecipe: ({view}: Context) => ({
    source: {kind: 'zones', zones: view('uint32', 4), zoneCount: 2},
    values: view('float32', 4)
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
  addRasterZonalDistributionRecipe: ({view, grid, band}: Context) => ({
    grid,
    input: band(),
    zones: view('uint32', 4),
    zoneCount: 2
  }),
  addExposureByZoneRecipe: ({view, grid, band}: Context) => ({
    grid,
    hazard: band(),
    population: band(),
    zones: view('uint32', 4),
    zoneCount: 2,
    weightMeaning: 'count',
    threshold: view('float32', 1)
  }),
  addRegionalRasterChangeRecipe: ({view, grid}: Context) => ({
    grid,
    change: {slices: view('float32', 8), sliceCount: 2, parameters: view('float32', 8)},
    zones: view('uint32', 4),
    zoneCount: 2,
    cellAreas: view('float32', 4),
    threshold: view('float32', 1)
  }),
  addSurfacePreparationRecipe: ({grid, band}: Context) => ({
    grid,
    input: band(),
    maximumDistance: 1,
    filter: {radius: 1}
  }),
  addDrainageNetworkRecipe: ({view, grid, band}: Context) => ({
    flow: {
      grid,
      elevation: band(),
      parameters: view('float32', 8),
      maximumFillIterations: 2,
      maximumFlatIterations: 2,
      maximumAccumulationIterations: 2
    },
    maximumSteps: 4
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
  }),
  addDasymetricRedistributionRecipe: ({view, grid, band}: Context) => ({
    grid,
    source: {
      kind: 'coverage',
      coverage: {
        zoneIds: view('uint32', 4),
        cellIds: view('uint32', 4),
        fractions: view('float32', 4),
        zoneValidity: view('uint32', 2),
        status: {
          count: view('uint32', 1),
          requiredCount: view('uint32', 1),
          overflow: view('uint32', 1)
        }
      }
    },
    sourceTotals: view('float32', 2),
    targetZones: view('uint32', 4),
    targetZoneCount: 2,
    ancillary: band(),
    pairCapacity: 8
  }),
  addExcessRiskRecipe: ({view}: Context) => ({
    events: view('float32', 4),
    populations: view('float32', 4)
  }),
  addSpatialRateRecipe: ({view, weights}: Context) => ({
    events: view('float32', 4),
    populations: view('float32', 4),
    weights: weights()
  }),
  addContiguityOrderCorrelogramRecipe: ({view, weights}: Context) => ({
    weights: weights(),
    values: view('float32', 4),
    parameters: view('float32', 8),
    maximumOrder: 2,
    capacity: 16
  })
};

function createTessellationFixture({input}: Context) {
  return {
    segments: input(
      'float32x4',
      Float32Array.of(0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0),
      4
    ),
    segmentBuildingIds: input('uint32', Uint32Array.of(0, 0, 0, 0), 4),
    enclosure: input('float32x2', Float32Array.of(-1, -1, 2, -1, 2, 2, -1, 2), 4),
    buildingCount: 1,
    sampleSpacing: 1,
    sampleCapacity: 4,
    vertexTolerance: 1 / 1024,
    capacity: {intersections: 16, nodedSegments: 32, vertices: 32, rings: 4}
  };
}

const domainFixtures = {
  addBuildingTessellationReferenceRecipe: (context: Context) => createTessellationFixture(context),
  addLocalBuildingTessellationRecipe: (context: Context) => ({
    ...createTessellationFixture(context),
    candidateRadius: 2
  }),
  addLandscapeMetricsRecipe: ({view}: Context) => ({
    patches: view('uint32', 4),
    patchClasses: view('uint32', 2),
    width: 2,
    height: 2,
    classCount: 2,
    cellSize: [1, 1],
    connectivity: 4
  }),
  addPatchProximityRecipe: ({view}: Context) => ({
    patches: view('uint32', 4),
    patchClasses: view('uint32', 2),
    width: 2,
    height: 2,
    cellSize: [1, 1],
    connectivity: 4,
    edgeDepth: 1,
    searchRadius: 2
  }),
  addResistanceConnectivityRecipe: ({view}: Context) => ({
    edges: view('uint32x2', 2),
    conductances: view('float32', 2),
    vertexCount: 3,
    source: 0,
    sink: 2,
    maximumIterations: 2
  }),
  addSparseResistanceConnectivityRecipe: ({view}: Context) => ({
    edges: view('uint32x2', 2),
    conductances: view('float32', 2),
    vertexCount: 3,
    source: 0,
    sink: 2,
    maximumIterations: 2,
    connectivityIterations: 2
  }),
  addRasterResistanceConnectivityRecipe: ({view, grid, band}: Context) => ({
    grid,
    resistance: band(),
    parameters: view('float32', 8),
    source: 0,
    sink: 3,
    maximumIterations: 2,
    connectivityIterations: 2,
    output: {
      potentials: view('float32', 4),
      vertexCurrents: view('float32', 4),
      diagnostics: view('float32x4', 1),
      solverStatus: view('uint32x4', 1)
    }
  }),
  addSegregationCounterfactualRecipe: ({view}: Context) => ({
    baseline: view('float32x2', 2),
    final: view('float32x2', 2),
    supportId: 'two-units',
    method: 'composition'
  }),
  addSegregationDecompositionRecipe: ({view}: Context) => ({
    populations: [0, 1, 2, 3].map(() => view('float32x2', 2)),
    supportId: 'two-units',
    counterfactualMethod: 'explicit'
  }),
  addPopulationSupportAlignmentRecipe: ({view}: Context) => ({
    populations: view('float32x2', 2),
    sourceIds: view('uint32', 2),
    targetIds: view('uint32', 2)
  }),
  addStreetProfilesRecipe: ({view}: Context) => ({
    positions: view('float32x2', 2),
    normals: view('float32x2', 2),
    segments: view('float32x4', 2),
    segmentBuildingIds: view('uint32', 2),
    buildingHeights: view('float32', 2),
    maximumDistance: 2,
    sampleSpacing: 1
  }),
  addIsovistsRecipe: ({view}: Context) => ({
    positions: view('float32x2', 2),
    segments: view('float32x4', 2),
    maximumDistance: 2,
    rayCount: 4
  }),
  addVisibilityGraphRecipe: ({view}: Context) => ({
    positions: view('float32x2', 3),
    segments: view('float32x4', 2),
    maximumDistance: 2,
    edgeCapacity: 3
  }),
  addBuildingTessellationRecipe: ({view}: Context) => ({
    segments: view('float32x4', 4),
    segmentBuildingIds: view('uint32', 4),
    enclosure: view('float32x2', 4),
    buildingCount: 1,
    sampleSpacing: 1,
    sampleCapacity: 4,
    candidateRadius: 2,
    vertexTolerance: 1 / 1024
  }),
  addBuildingAdjacencyRecipe: ({view}: Context) => ({
    sharedPaths: {
      runs: {
        leftSourceIds: view('uint32', 1),
        rightSourceIds: view('uint32', 1),
        offsets: view('uint32', 2),
        positions: view('float32x2', 2)
      },
      status: {
        count: view('uint32', 1),
        overflow: view('uint32', 1),
        uncertainCount: view('uint32', 1)
      }
    },
    lineBuildingIds: view('uint32', 2),
    orientations: view('float32', 2)
  })
};

const resultAuditFixtures = {
  ...fixtures,
  ...domainFixtures,
  'network:addVisibilityGraphRecipe': domainFixtures.addVisibilityGraphRecipe,
  'drive-time raster isochrones': (context: Context) => ({
    ...fixtures.addDriveTimeCatchmentRecipe(context),
    demandValues: context.view('float32', 4),
    isochrones: {
      costBreaks: context.view('float32', 4),
      parameters: context.view('float32', 12),
      raster: {
        width: 48,
        height: 40,
        output: {
          triangles: context.view('float32x2', 12000),
          triangleBands: context.view('uint32', 4000),
          status: {count: context.view('uint32', 1), overflow: context.view('uint32', 1)}
        }
      }
    }
  })
};

const resultAuditReport: {
  name: string;
  before: number;
  after: number;
  exposesContributors: boolean;
  entries: {id: string; class: string; nodes: readonly string[]}[];
}[] = [];
afterAll(() => {
  const reportPath = process.env['RECIPE_CONTRIBUTOR_AUDIT_OUTPUT'];
  if (reportPath) writeFileSync(reportPath, `${JSON.stringify(resultAuditReport, null, 2)}\n`);
});

it.each(
  Object.keys(resultAuditFixtures)
)('%s returns only unique caller-graph contributors and ports with single-declaration counts', name => {
  const addRecipe = (
    name === 'drive-time raster isochrones'
      ? recipes.addDriveTimeCatchmentRecipe
      : name === 'network:addVisibilityGraphRecipe'
        ? publicNetworkRecipes.addVisibilityGraphRecipe
        : Reflect.get({...publicSpatialRecipes, ...recipes}, name)
  ) as (graph: GPUCommandGraph, props: object) => object;
  function capture(prepare: boolean) {
    const context = createContext();
    const props = {...Reflect.get(resultAuditFixtures, name)(context), id: 'audit'};
    if (!prepare)
      vi.spyOn(preparation, 'addPreparedRecipe').mockImplementation((graph, props, declare) =>
        declare(graph, props)
      );
    try {
      return auditRecipeResult(context.graph, () => addRecipe(context.graph, props));
    } finally {
      vi.restoreAllMocks();
      context.destroy();
    }
  }
  const single = capture(false);
  const prepared = capture(true);
  resultAuditReport.push({
    name,
    before: single.entries.length,
    after: prepared.entries.length,
    exposesContributors: prepared.exposesContributors,
    entries: prepared.entries
  });
  expect(prepared.entries).toStrictEqual(single.entries);
  expect(prepared.nodeIds).toStrictEqual(single.nodeIds);
  expect(prepared.ports).toStrictEqual(single.ports);
  if (name === 'drive-time raster isochrones') {
    expect(prepared.entries.map(({id, class: className}) => [id, className])).toEqual([
      ['audit-edge-sources-kernel-clear', 'Object'],
      ['audit-edge-sources-kernel', 'Object'],
      ['audit-snap-facilities', 'GPUNetworkSnapping'],
      ['audit-facility-roads-kernel', 'Object'],
      ['audit-facility-starts-kernel', 'Object'],
      ['audit-facility-oriented-fractions-kernel', 'Object'],
      ['audit-service-areas', 'GPUNetworkServiceAreas'],
      ['audit-source-path-geometry', 'Object'],
      ['audit-source-path-topology', 'Object'],
      ['audit-isochrones', 'GPUNetworkIsochrones'],
      ['audit-snap-demand', 'GPUNetworkSnapping'],
      ['audit-demand-roads-kernel', 'Object'],
      ['audit-demand-times-kernel', 'Object'],
      ['audit-demand-bands-kernel', 'Object'],
      ['audit-band-statistics', 'GPUGroupStatistics']
    ]);
  }
});

it.each([
  'instance',
  'id',
  'contributor graph',
  'output graph'
] as const)('result audit rejects an injected duplicate or preparation leak: %s', defect => {
  const context = createContext();
  const foreign = new GPUCommandGraph(context.device);
  try {
    expect(() =>
      auditRecipeResult(context.graph, () => {
        const builder = new RecipeBuilder(defect === 'contributor graph' ? foreign : context.graph);
        const contributor = (nodeId: string) => ({
          id: 'stage',
          getCommandNodes: () => [
            {id: nodeId, type: 'copy' as const, resources: [], compile: () => ({encode: () => {}})}
          ]
        });
        const first = builder.add(contributor('first'));
        if (defect === 'id') builder.add(contributor('second'));
        return {
          contributors: defect === 'instance' ? [first, first] : builder.contributors,
          outputs: {
            value:
              defect === 'output graph'
                ? createTransientView(foreign, 'foreign-output', 'float32', 1)
                : context.view('float32', 1)
          }
        };
      })
    ).toThrow(
      defect === 'instance' || defect === 'id' ? /expected 1 to be 2/ : /expected GPUCommandGraph/
    );
  } finally {
    vi.restoreAllMocks();
    context.destroy();
  }
});

it('enumerates every add-recipe export from the public recipes index', () => {
  expect(
    Object.keys(recipes)
      .filter(name => /^add.*Recipe$/.test(name))
      .sort()
  ).toEqual(Object.keys(fixtures).sort());
});

it('X3 inventories add-recipe exports from the public family entry points', () => {
  // These family recipes are public but are not covered by the every-stage matrix below.
  // Keep the gap explicit rather than silently treating the two recipes barrels as complete.
  // Landscape's supplied-output preflight has a separate X3 regression in recipe-final-review-x1.
  const outsideStageMatrix = [
    'addBuildingAdjacencyRecipe',
    'addBuildingTessellationRecipe',
    'addBuildingTessellationReferenceRecipe',
    'addIsovistsRecipe',
    'addLandscapeMetricsRecipe',
    'addLocalBuildingTessellationRecipe',
    'addPatchProximityRecipe',
    'addPopulationSupportAlignmentRecipe',
    'addRasterResistanceConnectivityRecipe',
    'addResistanceConnectivityRecipe',
    'addSegregationCounterfactualRecipe',
    'addSegregationDecompositionRecipe',
    'addSparseResistanceConnectivityRecipe',
    'addStreetProfilesRecipe',
    'addVisibilityGraphRecipe'
  ];
  const families = [
    publicSpatialRecipes,
    publicNetworkRecipes,
    publicTerrainRecipes,
    publicRasterRecipes,
    publicDataframeRecipes,
    publicPointcloudRecipes,
    publicGeospatialRecipes,
    publicCoreRecipes,
    publicRecordRecipes
  ];
  const exported = new Set(
    families.flatMap(family => Object.keys(family)).filter(name => /^add.*Recipe$/.test(name))
  );
  expect([...exported].sort()).toEqual(Object.keys({...fixtures, ...domainFixtures}).sort());
  expect([...exported].sort()).toEqual([...Object.keys(fixtures), ...outsideStageMatrix].sort());
  for (const family of families) {
    for (const name of Object.keys(fixtures) as (keyof typeof fixtures)[]) {
      if (name in family) expect(Reflect.get(family, name), name).toBe(recipes[name]);
    }
  }
});

it.each([
  'addHotSpotAnalysisRecipe',
  'addRateClusterMapRecipe',
  'addSpatialRegressionRecipe',
  'addSpaceTimeHotSpotsRecipe'
] as const)('%s preserves single-declaration parameter bytes and uploads each caller buffer once', name => {
  function captureWrites(prepare: boolean) {
    const context = createContext();
    const initial = fixtures[name](context);
    const props = {
      ...initial,
      settings: {
        ...('settings' in initial ? initial.settings : {}),
        significanceLevel: 0.125,
        ...(name === 'addHotSpotAnalysisRecipe' || name === 'addRateClusterMapRecipe'
          ? {seed: 29, permutationCount: 3}
          : {})
      },
      ...(name === 'addHotSpotAnalysisRecipe' || name === 'addRateClusterMapRecipe'
        ? {
            permutation: {
              parameters: context.parameter('uint32'),
              maximumPermutations: 7,
              maximumNeighbors: 8
            }
          }
        : {})
    };
    const writes = context.parameters.map(parameter => vi.spyOn(parameter.buffer, 'write'));
    if (!prepare)
      vi.spyOn(preparation, 'addPreparedRecipe').mockImplementation((graph, props, declare) =>
        declare(graph, props)
      );
    try {
      recipes[name](context.graph, props as never);
      return writes.map(write => {
        expect(write).toHaveBeenCalledOnce();
        return write.mock.calls.map(([data, offset]) => ({
          offset: offset ?? 0,
          bytes: Array.from(
            ArrayBuffer.isView(data)
              ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
              : new Uint8Array(data)
          )
        }));
      });
    } finally {
      vi.restoreAllMocks();
      context.destroy();
    }
  }
  expect(captureWrites(true)).toStrictEqual(captureWrites(false));
});

it('audits every operation factory: one contributor or the prepared choropleth recipe adapter', () => {
  const root = fileURLToPath(new URL('../../../src/', import.meta.url));
  let factoryCount = 0;
  for (const path of readdirSync(root, {recursive: true})
    .map(path => path.toString())
    .filter(path => path.endsWith('.ts'))) {
    const source = ts.createSourceFile(
      path,
      readFileSync(`${root}/${path}`, 'utf8'),
      ts.ScriptTarget.Latest,
      true
    );
    const inspect = (node: ts.Node): void => {
      if (ts.isPropertyAssignment(node) && node.name.getText(source) === 'createContributor') {
        factoryCount++;
        const constructions: string[] = [];
        const declarations: string[] = [];
        const recipes: string[] = [];
        const visit = (child: ts.Node): void => {
          if (ts.isNewExpression(child)) constructions.push(child.expression.getText(source));
          if (ts.isCallExpression(child) && /^add\w+Recipe$/.test(child.expression.getText(source)))
            recipes.push(child.expression.getText(source));
          if (
            ts.isCallExpression(child) &&
            /\.(add|addToGraph|getCommandNodes|lower|createTransientBuffer|importBuffer|importFrameBuffer|createTransientTexture|importTexture|importFrameTexture)$/.test(
              child.expression.getText(source)
            )
          )
            declarations.push(child.expression.getText(source));
          ts.forEachChild(child, visit);
        };
        visit(node.initializer);
        // Multipart choropleths lower their complete prepared recipe as a primitive adapter.
        // Keep this exception exact: other factories still construct one contributor, and no
        // factory may bypass recipe preflight with direct graph declaration calls.
        if (
          path ===
          join('gpu-spatial-analysis', 'recipes', 'points-in-polygons-choropleth-operation.ts')
        ) {
          expect(constructions, path).toEqual([]);
          expect(recipes, path).toEqual(['addPointsInPolygonsChoroplethRecipe']);
        } else {
          expect(constructions, path).toHaveLength(1);
          expect(recipes, path).toEqual([]);
        }
        expect(declarations, path).toEqual([]);
      }
      ts.forEachChild(node, inspect);
    };
    inspect(source);
  }
  expect(factoryCount).toBeGreaterThan(0);
});

for (const name of Object.keys(fixtures) as (keyof typeof fixtures)[]) {
  it(`${name} leaves the caller graph untouched at every failed stage and permits same-ID retry`, () => {
    const baseline = createContext();
    const originalAdd = RecipeBuilder.prototype.add;
    const originalGlue = RecipeBuilder.prototype.addGlue;
    const originalSchedule = preparation.scheduleRecipeStage;
    let stageCount = 0;
    const addRecipe = recipes[name] as (
      graph: GPUCommandGraph,
      props: never
    ) => {contributors: readonly unknown[]};
    try {
      vi.spyOn(preparation, 'scheduleRecipeStage').mockImplementation((target, declare) => {
        if (target === baseline.graph) stageCount++;
        originalSchedule(target, declare);
      });
      vi.spyOn(RecipeBuilder.prototype, 'add').mockImplementation(function (
        this: RecipeBuilder<unknown>,
        contributor
      ) {
        if (this.graph === baseline.graph) stageCount++;
        return originalAdd.call(this, contributor);
      });
      vi.spyOn(RecipeBuilder.prototype, 'addGlue').mockImplementation(function (
        this: RecipeBuilder<unknown>,
        stage,
        nodes
      ) {
        if (this.graph === baseline.graph) stageCount++;
        return originalGlue.call(this, stage, nodes);
      });
      addRecipe(baseline.graph, {...fixtures[name](baseline), id: 'retry'} as never);
      expect(stageCount).toBeGreaterThan(0);
    } finally {
      vi.restoreAllMocks();
      baseline.destroy();
    }
    for (let failedStage = 1; failedStage <= stageCount; failedStage++) {
      for (const phase of ['construction', 'expansion'] as const) {
        for (const inspect of [true, false]) {
          const context = createContext();
          const props = {...fixtures[name](context), id: 'retry'};
          context.graph.addCopyPass({
            id: 'existing',
            resources: [],
            compile: () => ({encode: () => {}})
          });
          const allocate = vi.spyOn(context.graph, 'createTransientBuffer');
          const importBuffer = vi.spyOn(context.graph, 'importBuffer');
          const schedule = vi.spyOn(context.graph, 'add');
          const uploads = context.parameters.map(parameters =>
            vi.spyOn(parameters.buffer, 'write')
          );
          let stage = 0;
          vi.spyOn(preparation, 'scheduleRecipeStage').mockImplementation((target, declare) => {
            if (++stage !== failedStage) {
              originalSchedule(target, declare);
              return;
            }
            if (phase === 'construction') throw new Error(`invalid stage ${failedStage}`);
            originalSchedule(target, () => {
              declare();
              throw new Error(`invalid stage ${failedStage}`);
            });
          });
          vi.spyOn(RecipeBuilder.prototype, 'add').mockImplementation(function (
            this: RecipeBuilder<unknown>,
            contributor
          ) {
            if (++stage === failedStage) {
              if (phase === 'construction') throw new Error(`invalid stage ${failedStage}`);
              const expand = contributor.getCommandNodes.bind(contributor);
              contributor.getCommandNodes = target => {
                expand(target);
                throw new Error(`invalid stage ${failedStage}`);
              };
            }
            return originalAdd.call(this, contributor);
          });
          vi.spyOn(RecipeBuilder.prototype, 'addGlue').mockImplementation(function (
            this: RecipeBuilder<unknown>,
            label,
            nodes
          ) {
            if (++stage === failedStage) throw new Error(`invalid stage ${failedStage}`);
            return originalGlue.call(this, label, nodes);
          });
          try {
            expect(() => addRecipe(context.graph, props as never), `stage ${failedStage}`).toThrow(
              `invalid stage ${failedStage}`
            );
            expect(allocate, `stage ${failedStage} resources`).not.toHaveBeenCalled();
            expect(importBuffer, `stage ${failedStage} imports`).not.toHaveBeenCalled();
            expect(schedule, `stage ${failedStage} nodes`).not.toHaveBeenCalled();
            for (const upload of uploads)
              expect(upload, `stage ${failedStage} parameter upload`).not.toHaveBeenCalled();
            vi.restoreAllMocks();
            if (inspect) {
              const compiled = context.graph.compile();
              try {
                expect(compiled.stats.nodeOrder).toEqual(['existing']);
                expect(compiled.stats.logicalBufferCount).toBe(context.bufferCount());
                expect(compiled.stats.importedBufferCount).toBe(0);
              } finally {
                compiled.destroy();
              }
            } else {
              expect(
                () => addRecipe(context.graph, props as never),
                `stage ${failedStage} corrected retry`
              ).not.toThrow();
            }
          } finally {
            vi.restoreAllMocks();
            context.destroy();
          }
        }
      }
    }
  });
}
