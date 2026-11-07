// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from './recipe-harness';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {addZonalDistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/zonal-distribution-recipe';
import {addSurfacePreparationRecipe} from '../../../src/gpu-spatial-analysis/recipes/surface-preparation-recipe';
import {
  addExposureByZoneRecipe,
  addRegionalRasterChangeRecipe,
  addRasterZonalDistributionRecipe
} from '../../../src/gpu-spatial-analysis/recipes/raster-summary-recipes';
import {
  addNetworkDisruptionRecipe,
  addFacilityScenarioRecipe
} from '../../../src/gpu-network/recipes/network-scenarios-recipe';
import {addDrainageNetworkRecipe} from '../../../src/gpu-spatial-analysis/recipes/drainage-network-recipe';
import {getGPUTerrainFlowParameterValues} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {addClusterAndOutlineRecipe} from '../../../src/gpu-spatial-analysis/recipes/cluster-and-outline-recipe';
import {getGPUSpatialClusteringParameterValues} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {getGPUGeographicDistributionParameterValues} from '../../../src/gpu-spatial-analysis/geographic-distribution/index';
import {addTrackQualityRecipe} from '../../../src/gpu-network/recipes/track-quality-recipe';
import {addInterpolationDiagnosticsRecipe} from '../../../src/gpu-spatial-analysis/recipes/interpolation-diagnostics-recipe';
import {getGPUKrigingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/index';
import {getGPUChangeDetectionParameterValues} from '../../../src/gpu-raster/change-detection/index';
import {getGPUNetworkAccessibilityParameterValues} from '../../../src/gpu-network/network-accessibility/index';
import type {GPURasterMetadata} from '../../../src/gpu-raster/types';
import reference from '../../oracles/fixtures/geospatial-expansion-recipes.json';

const grid: GPURasterMetadata = {
  width: 4,
  height: 1,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};
function publish<Format extends 'float32' | 'uint32'>(
  fixture: RecipeTestFixture,
  name: string,
  source: GraphDataView<Format>
) {
  const output = fixture.output(name, source.format, source.length);
  const type = source.format === 'float32' ? 'f32' : 'u32';
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: name,
      operation: 'test-copy',
      invocationCount: source.length,
      bindings: [
        {name: 'source', view: source, type, access: 'read'},
        {name: 'destination', view: output.view, type, access: 'read_write'}
      ],
      body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
    })
  );
  return output;
}
async function createFixture(name: string) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return new RecipeTestFixture(device, name);
}

it('zonal recipe matches NumPy quantiles, ties, finite filtering and empty zones', async () => {
  const fixture = await createFixture('expansion-zones');
  try {
    const result = addZonalDistributionRecipe(fixture.graph, {
      source: {
        kind: 'zones',
        zones: fixture.input('zones', Uint32Array.from(reference.zones), 'uint32', 8),
        zoneCount: 4
      },
      values: fixture.input(
        'values',
        Float32Array.from(reference.values.map(value => value ?? NaN)),
        'float32',
        8
      ),
      probabilities: fixture.input(
        'probabilities',
        Float32Array.from(reference.probabilities),
        'float32',
        5
      )
    });
    const median = publish(fixture, 'median-out', result.outputs.medians);
    const quantiles = publish(fixture, 'quantiles-out', result.outputs.quantiles!);
    const counts = publish(fixture, 'counts-out', result.outputs.validCounts);
    const modes = publish(fixture, 'modes-out', result.outputs.modes);
    fixture.run();
    expect(await fixture.readUint32(counts, 4)).toEqual(
      reference.distributions.map(row => row.count)
    );
    expect(await fixture.readFloat32(median, 4)).toEqual(
      reference.distributions.map(row => row.median ?? NaN)
    );
    expect(await fixture.readFloat32(modes, 4)).toEqual(
      reference.distributions.map(row => row.mode ?? NaN)
    );
    expect(await fixture.readFloat32(quantiles, 20)).toEqual(
      reference.distributions.flatMap(row => row.quantiles.map(value => value ?? NaN))
    );
  } finally {
    fixture.destroy();
  }
});

it('raster distribution calibrates after native nodata and preserves empty zones', async () => {
  const fixture = await createFixture('expansion-raster-zones');
  try {
    const result = addRasterZonalDistributionRecipe(fixture.graph, {
      grid,
      input: {
        id: 'raw',
        format: 'sint32',
        storage: {
          kind: 'buffer',
          values: fixture.input('raw', new Int32Array([1, 3, -9, 8]), 'sint32', 4)
        },
        noDataValue: -9,
        scale: 2,
        offset: 10
      },
      zones: fixture.input('zones', new Uint32Array([0, 0, 1, 1]), 'uint32', 4),
      zoneCount: 3
    });
    const median = publish(fixture, 'median-out', result.outputs.medians);
    fixture.run();
    expect(await fixture.readFloat32(median, 3)).toEqual([14, 26, NaN]);
  } finally {
    fixture.destroy();
  }
});

it('disruption preserves one-way costs and reports newly unreachable only after convergence', async () => {
  const fixture = await createFixture('expansion-disruption');
  try {
    const topology = {
      offsets: fixture.input('offsets', new Uint32Array([0, 1, 2, 3, 3, 3]), 'uint32', 6),
      neighbors: fixture.input('neighbors', new Uint32Array([1, 2, 3]), 'uint32', 3),
      weights: fixture.input('weights', new Float32Array([1, 2, 1]), 'float32', 3)
    };
    const result = addNetworkDisruptionRecipe(fixture.graph, {
      search: {
        topology,
        startNodes: fixture.input('origins', new Uint32Array([0, 2]), 'uint32', 2),
        maximumIterations: 8
      },
      rowCount: 2,
      edgeMask: fixture.input('closures', new Uint32Array([1, 0, 1]), 'uint32', 3),
      opportunityWeights: fixture.input(
        'opportunities',
        new Float32Array([1, 2, 3, 4, 5]),
        'float32',
        5
      ),
      parameters: fixture.parameters(
        'parameters',
        'float32',
        getGPUNetworkAccessibilityParameterValues({threshold: Infinity})
      )
    });
    const baseline = publish(fixture, 'baseline-out', result.outputs.baselineCosts);
    const scenario = publish(fixture, 'scenario-out', result.outputs.scenarioCosts);
    const unreachable = publish(fixture, 'unreachable-out', result.outputs.newlyUnreachable);
    fixture.run();
    expect(await fixture.readFloat32(baseline, 10)).toEqual(
      reference.baseline.flat().map(value => value ?? Infinity)
    );
    expect(await fixture.readFloat32(scenario, 10)).toEqual(
      reference.scenario.flat().map(value => value ?? Infinity)
    );
    expect(await fixture.readUint32(unreachable, 10)).toEqual([0, 0, 1, 1, 0, 0, 0, 0, 0, 0]);
  } finally {
    fixture.destroy();
  }
});

it('facility evaluation retains unreachable demand and stable facility indices', async () => {
  const fixture = await createFixture('expansion-facility');
  try {
    const result = addFacilityScenarioRecipe(fixture.graph, {
      costs: fixture.input('costs', new Float32Array([1, 5, Infinity, 3, 1, 2]), 'float32', 6),
      demand: fixture.input('demand', new Float32Array([10, 20, 30]), 'float32', 3),
      supply: fixture.input('supply', new Float32Array([1, 1]), 'float32', 2),
      facilityMask: fixture.input('mask', new Uint32Array([1, 0]), 'uint32', 2),
      parameters: fixture.parameters(
        'parameters',
        'float32',
        getGPUNetworkAccessibilityParameterValues({threshold: 2})
      )
    });
    const covered = publish(fixture, 'covered-out', result.outputs.coveredDemand);
    const uncovered = publish(fixture, 'uncovered-out', result.outputs.uncoveredDemand);
    const facilities = publish(fixture, 'facilities-out', result.outputs.nearestFacilities);
    fixture.run();
    expect(await fixture.readFloat32(covered, 1)).toEqual([10]);
    expect(await fixture.readFloat32(uncovered, 1)).toEqual([50]);
    expect(await fixture.readUint32(facilities, 3)).toEqual([0, 0, 0xffffffff]);
  } finally {
    fixture.destroy();
  }
});

it('exposure keeps unknown hazards out of observed denominator while retaining population coverage', async () => {
  const fixture = await createFixture('expansion-exposure');
  try {
    const result = addExposureByZoneRecipe(fixture.graph, {
      grid,
      zones: fixture.input('zones', new Uint32Array([0, 0, 0, 1]), 'uint32', 4),
      zoneCount: 2,
      hazard: {
        id: 'hazard',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: fixture.input('hazard', new Float32Array([5, NaN, 1, 8]), 'float32', 4)
        }
      },
      population: {
        id: 'population',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: fixture.input('population', new Float32Array([10, 20, 30, 0]), 'float32', 4)
        }
      },
      weightMeaning: 'count',
      threshold: fixture.input('threshold', new Float32Array([4]), 'float32', 1)
    });
    const total = publish(fixture, 'total-out', result.outputs.totalPopulation);
    const observed = publish(fixture, 'observed-out', result.outputs.observedPopulation);
    const share = publish(fixture, 'share-out', result.outputs.exposedShare);
    fixture.run();
    expect(await fixture.readFloat32(total, 2)).toEqual([60, 0]);
    expect(await fixture.readFloat32(observed, 2)).toEqual([40, 0]);
    expect(await fixture.readFloat32(share, 2)).toEqual([0.25, NaN]);
  } finally {
    fixture.destroy();
  }
});

it('regional changes count common support and area without interpreting missing observations as no change', async () => {
  const fixture = await createFixture('expansion-change');
  try {
    const result = addRegionalRasterChangeRecipe(fixture.graph, {
      grid,
      change: {
        slices: fixture.input('slices', new Float32Array([1, 3, NaN, 4, 5, 4, 2, 2]), 'float32', 8),
        sliceCount: 2,
        parameters: fixture.parameters(
          'parameters',
          'float32',
          getGPUChangeDetectionParameterValues({beforeSlice: 0, afterSlice: 1})
        )
      },
      zones: fixture.input('zones', new Uint32Array([0, 0, 1, 1]), 'uint32', 4),
      zoneCount: 2,
      cellAreas: fixture.input('areas', new Float32Array([2, 2, 3, 3]), 'float32', 4),
      threshold: fixture.input('threshold', new Float32Array([1]), 'float32', 1)
    });
    const area = publish(fixture, 'area-out', result.outputs.changedArea);
    const coverage = publish(fixture, 'coverage-out', result.outputs.coverage);
    const means = publish(fixture, 'means-out', result.outputs.meanChanges);
    fixture.run();
    expect(await fixture.readFloat32(area, 2)).toEqual([2, 3]);
    expect(await fixture.readFloat32(coverage, 2)).toEqual([0.5, 1]);
    expect(await fixture.readFloat32(means, 2)).toEqual([2, -0.5]);
  } finally {
    fixture.destroy();
  }
});

it('surface preparation preserves observed values and records unresolved gaps separately', async () => {
  const fixture = await createFixture('expansion-surface');
  try {
    const result = addSurfacePreparationRecipe(fixture.graph, {
      grid,
      input: {
        id: 'source',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: fixture.input('values', new Float32Array([2, NaN, NaN, NaN]), 'float32', 4)
        }
      },
      maximumDistance: 1,
      method: 'nearest'
    });
    const values = publish(fixture, 'values-out', result.outputs.values);
    const original = publish(fixture, 'original-out', result.outputs.originalValidity);
    const filled = publish(fixture, 'filled-out', result.outputs.filled);
    const unresolved = publish(fixture, 'unresolved-out', result.outputs.unresolvedCount);
    fixture.run();
    expect(await fixture.readFloat32(values, 4)).toEqual([2, 2, NaN, NaN]);
    expect(await fixture.readUint32(original, 4)).toEqual([1, 0, 0, 0]);
    expect(await fixture.readUint32(filled, 4)).toEqual([0, 1, 0, 0]);
    expect(await fixture.readUint32(unresolved, 1)).toEqual([2]);
  } finally {
    fixture.destroy();
  }
});

it('track quality flags duplicate timestamps, gaps and speed independently', async () => {
  const fixture = await createFixture('expansion-track');
  try {
    const result = addTrackQualityRecipe(fixture.graph, {
      trajectory: {
        positions: fixture.input(
          'positions',
          new Float32Array([0, 0, 1, 0, 2, 0, 20, 0]),
          'float32x2',
          4
        ),
        timestamps: fixture.input('times', new Float32Array([0, 0, 1, 10]), 'float32', 4),
        trackOffsets: fixture.input('offsets', new Uint32Array([0, 4]), 'uint32', 2)
      },
      thresholds: fixture.input('thresholds', new Float32Array([1.5, 5]), 'float32', 2)
    });
    const flags = publish(fixture, 'flags-out', result.outputs.flags);
    const counts = publish(fixture, 'counts-out', result.outputs.flaggedCounts);
    fixture.run();
    expect(await fixture.readUint32(flags, 4)).toEqual([0, 2, 0, 12]);
    expect(await fixture.readUint32(counts, 1)).toEqual([2]);
  } finally {
    fixture.destroy();
  }
});

it('interpolation diagnostics leave one out and omit zero-variance standardized scores', async () => {
  const fixture = await createFixture('expansion-interpolation');
  try {
    const result = addInterpolationDiagnosticsRecipe(fixture.graph, {
      kriging: {
        positions: fixture.input(
          'positions',
          new Float32Array([0, 0, 0, 0, 1, 0, 1, 0]),
          'float32x2',
          4
        ),
        values: fixture.input('values', new Float32Array([2, 2, 2, 2]), 'float32', 4),
        grid,
        indexGridSize: [2, 1],
        indexBounds: [-1, -1, 4, 1],
        maximumNeighborCount: 4,
        parameters: fixture.parameters(
          'parameters',
          'float32',
          getGPUKrigingParameterValues(
            {
              searchRadius: Infinity,
              neighborCount: 4,
              minimumNeighborCount: 1,
              variogram: {model: 'spherical', range: 10, sill: 1, nugget: 0}
            },
            grid
          )
        )
      }
    });
    const bias = publish(fixture, 'bias-out', result.outputs.bias);
    const error = publish(fixture, 'error-out', result.outputs.rootMeanSquaredError);
    const standardized = publish(fixture, 'standardized-out', result.outputs.standardizedCount);
    fixture.run();
    expect(await fixture.readFloat32(bias, 1)).toEqual([0]);
    expect(await fixture.readFloat32(error, 1)).toEqual([0]);
    expect(await fixture.readUint32(standardized, 1)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('drainage recipe carries D8 raster results into connected reaches and catchment summaries', async () => {
  const fixture = await createFixture('expansion-drainage');
  try {
    const drainageGrid: GPURasterMetadata = {...grid, width: 3, height: 3};
    const result = addDrainageNetworkRecipe(fixture.graph, {
      flow: {
        grid: drainageGrid,
        elevation: {
          id: 'dem',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: fixture.input(
              'dem',
              new Float32Array([9, 8, 7, 6, 5, 4, 3, 2, 1]),
              'float32',
              9
            )
          }
        },
        parameters: fixture.parameters(
          'parameters',
          'float32',
          getGPUTerrainFlowParameterValues({streamThreshold: 1}, drainageGrid)
        ),
        maximumAccumulationIterations: 32
      },
      maximumSteps: 9
    });
    const distance = publish(fixture, 'distance-out', result.outputs.distance.distance);
    const terminals = publish(fixture, 'terminals-out', result.outputs.distance.terminalCells);
    const reaches = publish(fixture, 'reaches-out', result.outputs.network.reachCount);
    const overflow = publish(fixture, 'overflow-out', result.outputs.network.status.overflow);
    fixture.run();
    // The southern slope is 3, greater than the diagonal slope 4 / sqrt(2).
    expect((await fixture.readFloat32(distance, 9))[4]).toBeCloseTo(2, 5);
    expect((await fixture.readUint32(terminals, 9))[4]).toBe(8);
    expect((await fixture.readUint32(reaches, 1))[0]).toBeGreaterThan(0);
    expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('cluster alpha outlines preserve disjoint memberships and return selected geometry measures', async () => {
  const fixture = await createFixture('expansion-cluster-alpha');
  try {
    const result = addClusterAndOutlineRecipe(fixture.graph, {
      points: {
        kind: 'points',
        positions: fixture.input(
          'points',
          new Float32Array([0, 0, 1, 0, 1, 1, 0, 1, 5, 0, 6, 0, 6, 1, 5, 1]),
          'float32x2',
          8
        )
      },
      clusteringParameters: fixture.parameters(
        'clustering',
        'float32',
        getGPUSpatialClusteringParameterValues({
          bounds: [-1, -1, 7, 2],
          epsilon: 1.5,
          minimumPoints: 1
        })
      ),
      geometryParameters: fixture.parameters(
        'geometry',
        'float32',
        getGPUGeographicDistributionParameterValues()
      ),
      gridSize: [8, 4],
      maximumClusterCount: 2,
      maximumVerticesPerHull: 8,
      hullCapacity: 16,
      outline: {kind: 'alpha', maximumCircumradius: 1}
    });
    expect(result.outlineType).toBe('alpha');
    expect(result.concaveHull).toBeDefined();
    const areas = publish(fixture, 'areas-out', result.areas);
    const perimeters = publish(fixture, 'perimeters-out', result.perimeters);
    const complete = publish(fixture, 'complete-out', result.concaveHull!.status.converged);
    fixture.run();
    expect(await fixture.readFloat32(areas, 2)).toEqual([1, 1]);
    expect(await fixture.readFloat32(perimeters, 2)).toEqual([4, 4]);
    expect(await fixture.readUint32(complete, 2)).toEqual([1, 1]);
  } finally {
    fixture.destroy();
  }
});
