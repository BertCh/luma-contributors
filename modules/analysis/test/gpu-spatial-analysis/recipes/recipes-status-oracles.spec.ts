// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from './recipe-harness';
import {addInterpolationDiagnosticsRecipe} from '../../../src/gpu-spatial-analysis/recipes/interpolation-diagnostics-recipe';
import {getGPUKrigingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/index';
import {
  addFleetDwellRecipe,
  addFleetDwellZoneEventsRecipe
} from '../../../src/gpu-spatial-analysis/recipes/fleet-dwell-recipe';
import {getGPUTrajectoryMetricsParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-analysis/index';
import {addPeriodComparisonRecipe} from '../../../src/gpu-spatial-analysis/recipes/period-comparison-recipe';
import {
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues
} from '../../../src/dataframe-analysis/column-classification/index';
import {addLocalVariabilityRecipe} from '../../../src/gpu-spatial-analysis/recipes/local-variability-recipe';
import {getGPUSpatialAutocorrelationParameterValues} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {addClusterAndOutlineRecipe} from '../../../src/gpu-spatial-analysis/recipes/cluster-and-outline-recipe';
import {getGPUDBSCANParameterValues} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {getGPUGeographicDistributionParameterValues} from '../../../src/gpu-spatial-analysis/geographic-distribution/index';
import {addSpatialRegressionRecipe} from '../../../src/gpu-spatial-analysis/recipes/spatial-regression-recipe';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {addRateClusterMapRecipe} from '../../../src/gpu-spatial-analysis/recipes/rate-cluster-map-recipe';
import {addChangeOfSupportRecipe} from '../../../src/gpu-spatial-analysis/recipes/change-of-support-recipe';
import {getGPUPolygonRasterizationParameterValues} from '../../../src/raster-analysis/polygon-rasterization/index';
import {addSpaceTimeHotSpotsRecipe} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {getGPUCalendarBucketsParameterValues} from '../../../src/dataframe-analysis/calendar-buckets/index';
import {addHotSpotAnalysisRecipe} from '../../../src/gpu-spatial-analysis/recipes/hot-spot-analysis-recipe';
import {addDrainageNetworkRecipe} from '../../../src/gpu-spatial-analysis/recipes/drainage-network-recipe';
import {getGPUTerrainFlowParameterValues} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';

function createSquare(fixture: RecipeTestFixture) {
  return {
    kind: 'polygons' as const,
    positions: fixture.input(
      'square',
      Float32Array.from([-1, -1, 1, -1, 1, 1, -1, 1]),
      'float32x2',
      4
    ),
    featureOffsets: fixture.input('features', Uint32Array.from([0, 1]), 'uint32', 2),
    polygonOffsets: fixture.input('polygons', Uint32Array.from([0, 1]), 'uint32', 2),
    ringOffsets: fixture.input('rings', Uint32Array.from([0, 4]), 'uint32', 2)
  };
}

it.each([
  'duplicate',
  'distinct'
] as const)('interpolation recipe standardized count for %s observations', async geometry => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'interpolation-status-oracle');
  try {
    const grid = {
      width: 4,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const count = fixture.output('standardized-count', 'uint32', 1);
    addInterpolationDiagnosticsRecipe(fixture.graph, {
      kriging: {
        positions: fixture.input(
          'positions',
          Float32Array.from(
            geometry === 'duplicate' ? [0, 0, 0, 0, 1, 0, 1, 0] : [0, 0, 1, 0, 1, 1, 0, 1]
          ),
          'float32x2',
          4
        ),
        values: fixture.input('values', Float32Array.from([2, 2, 2, 2]), 'float32', 4),
        grid,
        indexGridSize: [2, 1],
        indexBounds: [-1, -1, 4, 2],
        maximumNeighborCount: 4,
        parameters: fixture.parameters(
          'parameters',
          'float32',
          getGPUKrigingParameterValues(
            {
              searchRadius: Infinity,
              neighborCount: 4,
              minimumNeighborCount: 1,
              variogram: {
                model: 'spherical',
                range: geometry === 'duplicate' ? 10 : 0.5,
                sill: 1,
                nugget: 0
              }
            },
            grid
          )
        )
      },
      output: {standardizedCount: count.view}
    });
    // Each held-out point retains a coincident noiseless observation. Its conditional
    // variance is exactly zero, so none of the four residuals can be standardized.
    // Distinct unit-square corners exceed range .5: all three retained observations
    // are uncorrelated with each other and the held-out point. Ordinary kriging has
    // weights 1/3 and variance 1 + 3*(1/3)^2 = 4/3 > 0; all four scores are finite.
    for (const poison of [37, 0xdeadbeef]) {
      count.buffer.write(Uint32Array.from([poison]));
      fixture.run();
      expect(await fixture.readUint32(count, 1)).toEqual(geometry === 'duplicate' ? [0] : [4]);
    }
  } finally {
    fixture.destroy();
  }
});

it('fleet stop recipe overwrites both bounded and dense zero overflow words', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'fleet-stop-status-oracle');
  try {
    const stopOverflow = fixture.output('stop-overflow', 'uint32', 1);
    const tableOverflow = fixture.output('table-overflow', 'uint32', 1);
    const stopCount = fixture.output('stop-count', 'uint32', 1);
    const tableCount = fixture.output('table-count', 'uint32', 1);
    addFleetDwellRecipe(fixture.graph, {
      trajectories: {
        kind: 'trajectories',
        positions: fixture.input('track', Float32Array.from([0, 0, 0, 0]), 'float32x2', 2),
        timestamps: fixture.input('times', Float32Array.from([0, 2]), 'float32', 2),
        trackOffsets: fixture.input('tracks', Uint32Array.from([0, 2]), 'uint32', 2)
      },
      parameters: fixture.parameters(
        'parameters',
        'float32',
        getGPUTrajectoryMetricsParameterValues({stopSpeedThreshold: 1, stopMinimumDuration: 1})
      ),
      stopCapacity: 1,
      zones: {...createSquare(fixture), candidateCapacity: 1},
      output: {
        stops: {status: {count: stopCount.view, overflow: stopOverflow.view}},
        table: {status: {count: tableCount.view, overflow: tableOverflow.view}}
      }
    });
    // One stationary two-second track gives exactly one stop in the single square.
    // The one-slot stop list fits, and a one-zone dense table cannot overflow.
    for (const output of [stopCount, stopOverflow, tableCount, tableOverflow])
      output.buffer.write(Uint32Array.from([37]));
    fixture.run();
    expect(await fixture.readUint32(stopCount, 1)).toEqual([1]);
    expect(await fixture.readUint32(tableCount, 1)).toEqual([1]);
    expect(await fixture.readUint32(stopOverflow, 1)).toEqual([0]);
    expect(await fixture.readUint32(tableOverflow, 1)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('fleet event recipe publishes complete crossing counts and clears event overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'fleet-event-status-oracle');
  try {
    const count = fixture.output('event-count', 'uint32', 1);
    const overflow = fixture.output('event-overflow', 'uint32', 1);
    const visits = fixture.output('visits', 'uint32', 1);
    const tableCounts = fixture.output('table-counts', 'uint32', 1);
    const tableCount = fixture.output('table-count', 'uint32', 1);
    addFleetDwellZoneEventsRecipe(fixture.graph, {
      trajectories: {
        kind: 'trajectories',
        positions: fixture.input('track', Float32Array.from([-2, 0, 2, 0]), 'float32x2', 2),
        timestamps: fixture.input('times', Float32Array.from([0, 4]), 'float32', 2),
        trackOffsets: fixture.input('tracks', Uint32Array.from([0, 2]), 'uint32', 2)
      },
      zones: createSquare(fixture),
      candidateCapacity: 32,
      trackEventCapacity: 4,
      eventCapacity: 4,
      output: {
        events: {status: {count: count.view, overflow: overflow.view}},
        visitCounts: visits.view,
        table: {counts: tableCounts.view, status: {count: tableCount.view}}
      }
    });
    // x(t)=-2+t enters at t=1 and exits at t=3: two events, one visit,
    // one visiting track, and one dense zone row. Four event slots fit both events.
    for (const output of [count, overflow, visits, tableCounts, tableCount])
      output.buffer.write(Uint32Array.from([37]));
    fixture.run();
    expect(await fixture.readUint32(count, 1)).toEqual([2]);
    expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
    expect(await fixture.readUint32(visits, 1)).toEqual([1]);
    expect(await fixture.readUint32(tableCounts, 1)).toEqual([1]);
    expect(await fixture.readUint32(tableCount, 1)).toEqual([1]);
  } finally {
    fixture.destroy();
  }
});

it.each([
  {name: 'complete one-cell union', resolution: 0, before: [0, 0], after: [0, 0], overflow: [0]},
  {
    name: 'truncated two-cell union',
    resolution: 1,
    before: [-90, 45],
    after: [90, 45],
    overflow: [1]
  }
])('period recipe union status for $name', async scenario => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'period-status-oracle');
  try {
    const count = fixture.output('count', 'uint32', 1);
    const overflow = fixture.output('overflow', 'uint32', 1);
    const classCount = fixture.output('class-count', 'uint32', 1);
    addPeriodComparisonRecipe(fixture.graph, {
      family: 'quadbin',
      resolution: scenario.resolution,
      tableCapacity: 1,
      unionCapacity: 1,
      before: {
        positions: fixture.input('before', Float32Array.from(scenario.before), 'float32x2', 1)
      },
      after: {positions: fixture.input('after', Float32Array.from(scenario.after), 'float32x2', 1)},
      output: {
        comparison: {status: {count: count.view, overflow: overflow.view}},
        classCount: classCount.view
      },
      classBreaksParameters: fixture.parameters(
        'breaks',
        'float32',
        getGPUClassBreaksParameterValues({method: 'custom', customEdges: [-1, 0, 1]}, 2)
      ),
      maximumClassCount: 2,
      methods: ['custom'],
      palette: fixture.input('palette', Uint32Array.from([0xff000000, 0xffffffff]), 'uint32', 2),
      colorScaleParameters: fixture.parameters(
        'scale',
        'float32',
        getGPUColorScaleParameterValues({scale: 'threshold', domainCount: 3, paletteCount: 2})
      ),
      maximumPaletteCount: 2
    });
    // Resolution zero has one world cell. The union fits one slot; three custom
    // edges define exactly two classes, independently of the observed zero change.
    // At resolution one, longitude -90 and +90 occupy different tiles, each input
    // table fits one row but their two-cell union loses one row at unionCapacity=1.
    for (const output of [count, overflow, classCount]) output.buffer.write(Uint32Array.from([37]));
    fixture.run();
    expect(await fixture.readUint32(count, 1)).toEqual([1]);
    expect(await fixture.readUint32(overflow, 1)).toEqual(scenario.overflow);
    expect(await fixture.readUint32(classCount, 1)).toEqual([2]);
  } finally {
    fixture.destroy();
  }
});

it('local variability recipe clears every LOSH code after upstream loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'local-variability-status-oracle');
  try {
    const status = fixture.output('status', 'uint32', 4);
    const loss = fixture.output('loss', 'uint32', 1);
    addLocalVariabilityRecipe(fixture.graph, {
      weights: {
        offsets: fixture.input('offsets', Uint32Array.from([0, 1, 2, 3, 4]), 'uint32', 5),
        neighbors: fixture.input('neighbors', Uint32Array.from([1, 2, 3, 0]), 'uint32', 4),
        weights: fixture.input('weights', Float32Array.from([1, 1, 1, 1]), 'float32', 4),
        status: {overflow: loss.view}
      },
      values: fixture.input('values', Float32Array.from([1, 2, 4, 8]), 'float32', 4),
      parameters: fixture.parameters(
        'parameters',
        'float32',
        getGPUSpatialAutocorrelationParameterValues({})
      ),
      output: {loshStatus: status.view}
    });
    // Every row has one finite neighbor and the four unequal values have positive
    // dispersion. Complete CSR gives success 0; supplied global loss gives code 3.
    for (const flag of [1, 0, 1, 0]) {
      loss.buffer.write(Uint32Array.from([flag]));
      status.buffer.write(new Uint32Array(4).fill(37));
      fixture.run();
      expect(await fixture.readUint32(status, 4)).toEqual(flag ? [3, 3, 3, 3] : [0, 0, 0, 0]);
    }
  } finally {
    fixture.destroy();
  }
});

it.each([8, 3])('cluster outline recipe square hull status at capacity %i', async hullCapacity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'outline-status-oracle');
  try {
    const count = fixture.output('cluster-count', 'uint32', 1);
    const members = fixture.output('members', 'uint32', 2);
    const hullCounts = fixture.output('hull-counts', 'uint32', 2);
    const overflow = fixture.output('hull-overflow', 'uint32', 1);
    addClusterAndOutlineRecipe(fixture.graph, {
      points: {
        kind: 'points',
        positions: fixture.input(
          'positions',
          Float32Array.from([0, 0, 1, 0, 1, 1, 0, 1]),
          'float32x2',
          4
        )
      },
      clusteringParameters: fixture.parameters(
        'clustering',
        'float32',
        getGPUDBSCANParameterValues({
          bounds: [-1, -1, 2, 2],
          epsilon: 2,
          minimumSamples: 1
        })
      ),
      gridSize: [2, 2],
      geometryParameters: fixture.parameters(
        'geometry',
        'float32',
        getGPUGeographicDistributionParameterValues()
      ),
      maximumClusterCount: 2,
      maximumVerticesPerHull: 4,
      hullCapacity,
      output: {clusterCount: count.view, status: {hullOverflow: overflow.view}},
      scratch: {counts: members.view, hullCounts: hullCounts.view}
    });
    // All four corners are mutual neighbors (sqrt(2)<2): one four-member
    // cluster with four extreme vertices; the second group is empty. Capacity three
    // loses one of those four vertices; capacity eight fits the entire hull.
    count.buffer.write(Uint32Array.from([37]));
    overflow.buffer.write(Uint32Array.from([37]));
    members.buffer.write(Uint32Array.from([37, 37]));
    hullCounts.buffer.write(Uint32Array.from([37, 37]));
    fixture.run();
    expect(await fixture.readUint32(count, 1)).toEqual([1]);
    expect(await fixture.readUint32(members, 2)).toEqual([4, 0]);
    if (hullCapacity === 8) expect(await fixture.readUint32(hullCounts, 2)).toEqual([4, 0]);
    expect(await fixture.readUint32(overflow, 1)).toEqual(hullCapacity === 3 ? [1] : [0]);
  } finally {
    fixture.destroy();
  }
});

it.each([
  false,
  true
])('spatial regression recipe OLS status with constant predictor %s', async constant => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'regression-status-oracle');
  try {
    const status = fixture.output('ols-status', 'uint32', 1);
    addSpatialRegressionRecipe(fixture.graph, {
      predictors: fixture.input(
        'predictors',
        Float32Array.from(constant ? [0, 0, 0, 0, 0, 0, 0, 0] : [0, 1, 2, 3, 4, 5, 6, 7]),
        'float32',
        8
      ),
      response: fixture.input(
        'response',
        Float32Array.from([1, 3, 2, 5, 4, 8, 6, 9]),
        'float32',
        8
      ),
      predictorCount: 1,
      weights: {
        offsets: fixture.input(
          'offsets',
          Uint32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8]),
          'uint32',
          9
        ),
        neighbors: fixture.input(
          'neighbors',
          Uint32Array.from([1, 2, 3, 4, 5, 6, 7, 0]),
          'uint32',
          8
        ),
        weights: fixture.input('weights', new Float32Array(8).fill(1), 'float32', 8)
      },
      parameters: fixture.parameterBuffer(
        'parameters',
        'float32',
        GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
      ),
      output: {ols: {status: {code: status.view}}}
    });
    // With an intercept and x=0..7, X'X=[[8,28],[28,140]], determinant 336>0.
    // All eight responses are finite and n=8>2, so the complete OLS code is success 0.
    // A constant-zero predictor instead gives X'X=[[8,0],[0,0]], rank one,
    // requiring the singular-design code 1 independently of the fitted coefficients.
    status.buffer.write(Uint32Array.from([0xdeadbeef]));
    fixture.run();
    expect(await fixture.readUint32(status, 1)).toEqual(constant ? [1] : [0]);
  } finally {
    fixture.destroy();
  }
});

it.each([
  16, 3
])('rate cluster recipe three-square weights at neighbor capacity %i', async neighborCapacity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'rate-status-oracle');
  try {
    const overflow = fixture.output('overflow', 'uint32', 1);
    const candidate = fixture.output('candidate', 'uint32', 1);
    const required = fixture.output('required', 'uint32', 1);
    addRateClusterMapRecipe(fixture.graph, {
      polygons: {
        kind: 'polygons',
        positions: fixture.input(
          'positions',
          Float32Array.from([
            0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1, 2, 0, 3, 0, 3, 1, 2, 1
          ]),
          'float32x2',
          12
        ),
        featureOffsets: fixture.input('features', Uint32Array.from([0, 1, 2, 3]), 'uint32', 4),
        polygonOffsets: fixture.input('polygons', Uint32Array.from([0, 1, 2, 3]), 'uint32', 4),
        ringOffsets: fixture.input('rings', Uint32Array.from([0, 4, 8, 12]), 'uint32', 4)
      },
      events: fixture.input('events', Float32Array.from([1, 3, 2]), 'float32', 3),
      populations: fixture.input('populations', Float32Array.from([10, 10, 10]), 'float32', 3),
      palette: fixture.input('palette', new Uint32Array(5), 'uint32', 5),
      neighborCapacity,
      pairCapacity: 32,
      parameters: fixture.parameterBuffer(
        'parameters',
        'float32',
        GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
      ),
      output: {
        status: {
          weightsOverflow: overflow.view,
          weightsCandidateOverflow: candidate.view,
          weightsRequiredCount: required.view
        }
      }
    });
    // Only adjacent squares share a boundary: directed edges 0->1,1->0,1->2,2->1.
    // Four edges exceed a three-slot CSR and fit a 16-slot CSR. Both cases
    // fit 32 pair scratch slots, so final overflow is separate from candidate loss.
    for (const output of [overflow, candidate, required])
      output.buffer.write(Uint32Array.from([37]));
    fixture.run();
    expect(await fixture.readUint32(overflow, 1)).toEqual(neighborCapacity === 3 ? [1] : [0]);
    expect(await fixture.readUint32(candidate, 1)).toEqual([0]);
    expect(await fixture.readUint32(required, 1)).toEqual([4]);
  } finally {
    fixture.destroy();
  }
});

it('change of support recipe overwrites all complete one-cell diagnostics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'support-status-oracle');
  try {
    const grid = {
      width: 1,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const zones = {...createSquare(fixture), crossingCapacity: 16};
    const overflow = fixture.output('overflow', 'uint32', 1);
    const required = fixture.output('required', 'uint32', 1);
    const invalid = fixture.output('invalid', 'uint32', 1);
    addChangeOfSupportRecipe(fixture.graph, {
      grid,
      source: zones,
      target: zones,
      pairCapacity: 1,
      parameters: fixture.input(
        'parameters',
        getGPUPolygonRasterizationParameterValues(grid),
        'float32',
        4
      ),
      sourceValues: fixture.input('values', Float32Array.from([7]), 'float32', 1),
      output: {
        status: {overflow: overflow.view, requiredCount: required.view, invalidCount: invalid.view}
      }
    });
    // The sole finite cell center (.5,.5) belongs to both identical squares.
    // One source/target tuple fits one slot, with no missing or invalid cell.
    for (const output of [overflow, required, invalid]) output.buffer.write(Uint32Array.from([37]));
    fixture.run();
    expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
    expect(await fixture.readUint32(required, 1)).toEqual([1]);
    expect(await fixture.readUint32(invalid, 1)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('space time recipe counts all hot slices from closed form Gi star scores', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'space-time-status-oracle');
  try {
    const hot = fixture.output('hot-counts', 'uint32', 25);
    const words = [
      Date.UTC(2023, 0, 1),
      Date.UTC(2023, 1, 1),
      Date.UTC(2023, 2, 1),
      Date.UTC(2023, 3, 1),
      Date.UTC(2023, 4, 1),
      Date.UTC(2023, 5, 1)
    ].flatMap(time => [time % 2 ** 32, Math.floor(time / 2 ** 32)]);
    addSpaceTimeHotSpotsRecipe(fixture.graph, {
      timestamps: fixture.input('timestamps', Uint32Array.from(words), 'uint32x2', 6),
      calendarParameters: fixture.input(
        'calendar',
        getGPUCalendarBucketsParameterValues(0),
        'sint32',
        2
      ),
      slices: {step: 'month', origin: {year: 2023}, count: 6},
      cells: {
        kind: 'lattice',
        positions: fixture.input('positions', new Float32Array(12).fill(0.5), 'float32x2', 6),
        width: 5,
        height: 5,
        bounds: [0, 0, 5, 5]
      },
      settings: {
        radius: 0,
        temporalWindow: 0,
        persistentFraction: 0.75,
        minimumTrendSliceCount: 3,
        sporadicRequiresFinalSlice: false
      },
      parameters: fixture.parameterBuffer(
        'settings',
        'float32',
        GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
      ),
      output: {hotSliceCount: hot.view}
    });
    // Each slice is [1,0,...,0] over 25 cells. With only self weight, Gi* is
    // sqrt(24) at cell 0 and -1/sqrt(24) elsewhere. Even BH over all 150
    // bins keeps the six positive 4.899 scores significant; all other bins are nonsignificant.
    hot.buffer.write(new Uint32Array(25).fill(37));
    fixture.run();
    expect(await fixture.readUint32(hot, 25)).toEqual([
      6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
  } finally {
    fixture.destroy();
  }
});

it('hot spot recipe writes every guarded and unguarded lattice neighbor count', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'hot-neighbors-status-oracle');
  try {
    const counts = fixture.output('neighbors', 'uint32', 9);
    addHotSpotAnalysisRecipe(fixture.graph, {
      source: {
        kind: 'lattice',
        values: fixture.input(
          'values',
          Float32Array.from([1, 2, 3, 4, 9, 6, 7, 8, 5]),
          'float32',
          9
        ),
        width: 3,
        height: 3,
        neighborCapacity: 72
      },
      parameters: fixture.parameterBuffer(
        'parameters',
        'float32',
        GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
      ),
      output: {neighborCounts: counts.view}
    });
    // Queen adjacency excludes self: corners have three, edges five and center eight.
    // The default minimum-eight inference guard does not change available-neighbor counts.
    counts.buffer.write(new Uint32Array(9).fill(37));
    fixture.run();
    expect(await fixture.readUint32(counts, 9)).toEqual([3, 5, 3, 5, 8, 5, 3, 5, 3]);
  } finally {
    fixture.destroy();
  }
});

it('hot spot points recipe overwrites complete one-cell table overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'hot-table-status-oracle');
  try {
    const count = fixture.output('table-count', 'uint32', 1);
    const overflow = fixture.output('table-overflow', 'uint32', 1);
    const neighbors = fixture.output('neighbor-counts', 'uint32', 2);
    addHotSpotAnalysisRecipe(fixture.graph, {
      source: {
        kind: 'points',
        positions: fixture.input('points', Float32Array.from([0, 0]), 'float32x2', 1),
        family: 'quadbin',
        resolution: 0,
        studyArea: {...createSquare(fixture), candidateCapacity: 16},
        cellCapacity: 2,
        neighborCapacity: 1
      },
      parameters: fixture.parameterBuffer(
        'parameters',
        'float32',
        GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
      ),
      scratch: {table: {status: {count: count.view, overflow: overflow.view}}},
      output: {neighborCounts: neighbors.view}
    });
    // The resolution-zero world tile center (0,0) lies in the square study area.
    // The one-row deduplicated study table fits two slots. Its world cell has no
    // active neighbor; the unused second row must also overwrite stale neighbor counts.
    count.buffer.write(Uint32Array.from([37]));
    overflow.buffer.write(Uint32Array.from([37]));
    neighbors.buffer.write(Uint32Array.from([37, 37]));
    fixture.run();
    expect(await fixture.readUint32(count, 1)).toEqual([1]);
    expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
    expect(await fixture.readUint32(neighbors, 2)).toEqual([0, 0]);
  } finally {
    fixture.destroy();
  }
});

it('drainage recipe distinguishes a lost line vertex from a complete reach', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const vertexCapacity of [1, 6]) {
    const fixture = new RecipeTestFixture(device, 'drainage-status-oracle');
    try {
      const grid = {
        width: 3,
        height: 1,
        affine: [1, 0, 0, 0, 1, 0] as const,
        pixelInterpretation: 'area' as const
      };
      const result = addDrainageNetworkRecipe(fixture.graph, {
        flow: {
          grid,
          elevation: {
            id: 'elevation',
            format: 'float32',
            storage: {
              kind: 'buffer',
              values: fixture.input('elevation', Float32Array.from([3, 2, 1]), 'float32', 3)
            }
          },
          parameters: fixture.parameters(
            'parameters',
            'float32',
            getGPUTerrainFlowParameterValues({streamThreshold: 1}, grid)
          ),
          maximumAccumulationIterations: 8
        },
        maximumSteps: 3,
        vertexCapacity
      });
      const distanceStage = result.status.stages.find(
        stage => stage.stage === 'downstream-distance'
      )!.status;
      const networkStage = result.status.stages.find(
        stage => stage.stage === 'stream-network'
      )!.status;
      expect(distanceStage.unresolvedCount).toBeDefined();
      expect(distanceStage).not.toHaveProperty('uncertainCount');
      expect(networkStage.unresolvedCount).toBe(result.outputs.network.unresolvedCount);
      const overflow = fixture.output('published-overflow', 'uint32', 1);
      fixture.graph.add(
        createWGSLKernelNode(fixture.graph, {
          id: 'publish',
          operation: 'StatusOracle',
          invocationCount: 1,
          bindings: [
            {
              name: 'source',
              view: result.outputs.network.status.overflow,
              type: 'u32',
              access: 'read'
            },
            {name: 'destination', view: overflow.view, type: 'u32', access: 'read_write'}
          ],
          body: 'destination[destinationOffset] = source[sourceOffset];'
        })
      );
      // The descending three-cell chain has a nondegenerate stream reach. One
      // vertex cannot encode its two distinct endpoints; six slots fit all cells.
      fixture.run();
      expect(await fixture.readUint32(overflow, 1)).toEqual([vertexCapacity === 1 ? 1 : 0]);
    } finally {
      fixture.destroy();
    }
  }
});
