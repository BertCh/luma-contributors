// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from './recipe-harness';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {addPointsInPolygonsChoroplethRecipe} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe';
import {addDriveTimeCatchmentRecipe} from '../../../src/gpu-network/recipes/drive-time-catchment-recipe';
import {addFacilityScenarioRecipe} from '../../../src/gpu-network/recipes/network-scenarios-recipe';
import {getGPUNetworkAccessibilityParameterValues} from '../../../src/gpu-network/network-accessibility/index';
import {addLandscapeMetricsRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/landscape-metrics';
import {addRasterZonalDistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/raster-summary-recipes';
import {addRateClusterMapRecipe} from '../../../src/gpu-spatial-analysis/recipes/rate-cluster-map-recipe';

function polygons(fixture: RecipeTestFixture, cells = 1) {
  const positions =
    cells === 1
      ? [0, 0, 1, 0, 1, 1, 0, 1]
      : [
          0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1, 0, 1, 1, 1, 1, 2, 0, 2, 1, 1, 2, 1, 2, 2,
          1, 2
        ];
  return {
    kind: 'polygons' as const,
    positions: fixture.input(
      'polygon-positions',
      new Float32Array(positions),
      'float32x2',
      cells * 4
    ),
    featureOffsets: fixture.input(
      'features',
      Uint32Array.from({length: cells + 1}, (_, index) => index),
      'uint32',
      cells + 1
    ),
    polygonOffsets: fixture.input(
      'polygons',
      Uint32Array.from({length: cells + 1}, (_, index) => index),
      'uint32',
      cells + 1
    ),
    ringOffsets: fixture.input(
      'rings',
      Uint32Array.from({length: cells + 1}, (_, index) => index * 4),
      'uint32',
      cells + 1
    ),
    candidateCapacity: 16,
    crossingCapacity: 16
  };
}

it.each([
  undefined,
  'group'
] as const)('X1 rejects two interior parts of one point feature for backend %s, then counts explicit rows', async backend => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'multipart-choropleth-x1');
  try {
    const positions = fixture.input(
      'points',
      new Float32Array([0.25, 0.25, 0.75, 0.75]),
      'float32x2',
      2
    );
    const featureOffsets = fixture.input('point-features', new Uint32Array([0, 2]), 'uint32', 2);
    const counts = fixture.output('counts', 'uint32', 1);
    const props = {
      id: 'retry',
      backend,
      points: {kind: 'points' as const, positions},
      polygons: polygons(fixture),
      outputs: {counts: counts.view}
    };
    const add = vi.spyOn(fixture.graph, 'add');
    expect(() =>
      Reflect.apply(addPointsInPolygonsChoroplethRecipe, undefined, [
        fixture.graph,
        {
          ...props,
          points: {...props.points, featureOffsets}
        }
      ])
    ).toThrow(/multipart/);
    expect(add).not.toHaveBeenCalled();
    add.mockRestore();
    addPointsInPolygonsChoroplethRecipe(fixture.graph, props);
    fixture.run();
    expect(await fixture.readUint32(counts, 1)).toEqual([2]);
  } finally {
    vi.restoreAllMocks();
    fixture.destroy();
  }
});

it('X2 X5 X6 executes four offset demand rows in one band without overwriting routing input', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'drive-time-x1');
  try {
    const times = fixture.output('times', 'float32', 8);
    const bands = fixture.output('bands', 'uint32', 8);
    const counts = fixture.output('counts', 'uint32', 1);
    const keys = fixture.output('keys', 'uint32', 1);
    const weights = fixture.input(
      'weights',
      new Float32Array([10, 10, 0, 0, 0, 0, 0, 0]),
      'float32',
      2
    );
    const demandTimes = fixture.graph.createDataView(times.view.buffer, {
      format: 'float32',
      length: 4,
      byteOffset: 4
    });
    const demandBands = fixture.graph.createDataView(bands.view.buffer, {
      format: 'uint32',
      length: 4,
      byteOffset: 4
    });
    const props = {
      id: 'retry',
      topology: {
        offsets: fixture.input('offsets', new Uint32Array([0, 1, 2]), 'uint32', 3),
        neighbors: fixture.input('neighbors', new Uint32Array([1, 0]), 'uint32', 2),
        weights,
        nodePositions: fixture.input('nodes', new Float32Array([0, 0, 10, 0]), 'float32x2', 2)
      },
      facilities: fixture.input('facilities', new Float32Array([0, 0]), 'float32x2', 1),
      demand: fixture.input('demand', new Float32Array([1, 0, 2, 0, 3, 0, 4, 0]), 'float32x2', 4),
      costBreaks: fixture.input('breaks', new Float32Array([10]), 'float32', 1),
      maximumIterations: 4,
      outputs: {demandTimes, demandBands, bands: {counts: counts.view, keys: keys.view}}
    };
    expect(() =>
      addDriveTimeCatchmentRecipe(fixture.graph, {
        ...props,
        outputs: {
          ...props.outputs,
          demandTimes: fixture.graph.createDataView(times.view.buffer, {
            format: 'float32',
            length: 1,
            byteOffset: 4
          })
        }
      })
    ).toThrow(/rows/);
    expect(() =>
      addDriveTimeCatchmentRecipe(fixture.graph, {
        ...props,
        outputs: {
          ...props.outputs,
          demandBands: fixture.graph.createDataView(bands.view.buffer, {
            format: 'uint32',
            length: 1,
            byteOffset: 4
          })
        }
      })
    ).toThrow(/rows/);
    expect(() =>
      addDriveTimeCatchmentRecipe(fixture.graph, {
        ...props,
        outputs: {
          ...props.outputs,
          demandTimes: fixture.graph.createDataView(weights.buffer, {format: 'float32', length: 4})
        }
      })
    ).toThrow(/share/);
    const result = addDriveTimeCatchmentRecipe(fixture.graph, props);
    expect(result.contributors.length).toBeGreaterThan(4);
    fixture.run();
    const actualTimes = await fixture.readFloat32(times, 8);
    for (let index = 0; index < 4; index++) {
      expect(Number.isFinite(actualTimes[index + 1])).toBe(true);
      expect(actualTimes[index + 1]).toBeCloseTo(index + 1, 6);
    }
    expect([actualTimes[0], ...actualTimes.slice(5)]).toEqual([0, 0, 0, 0]);
    expect(await fixture.readUint32(bands, 8)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
    expect(await fixture.readUint32(keys, 1)).toEqual([0]);
    expect(await fixture.readUint32(counts, 1)).toEqual([4]);
  } finally {
    fixture.destroy();
  }
});

it('X3 retries a short class output on a reconciled two-class landscape', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'landscape-x1');
  try {
    const classes = fixture.output('classes', 'float32x4', 2);
    const props = {
      id: 'retry',
      patches: fixture.input('patches', new Uint32Array([0, 1]), 'uint32', 2),
      patchClasses: fixture.input('patch-classes', new Uint32Array([0, 1]), 'uint32', 2),
      width: 2,
      height: 1,
      classCount: 2,
      cellSize: [1, 1] as const,
      connectivity: 4 as const,
      reconcilePatches: true
    };
    expect(() =>
      addLandscapeMetricsRecipe(fixture.graph, {
        ...props,
        output: {
          classMetrics: fixture.graph.createDataView(classes.view.buffer, {
            format: 'float32x4',
            length: 1
          })
        }
      })
    ).toThrow(/rows/);
    addLandscapeMetricsRecipe(fixture.graph, {...props, output: {classMetrics: classes.view}});
    fixture.run();
    const actual = await fixture.readFloat32(classes, 8);
    for (const offset of [0, 4]) {
      expect(actual[offset]).toBe(1);
      expect(actual[offset + 1]).toBe(4);
    }
  } finally {
    fixture.destroy();
  }
});

it('X5 keeps independent smoothed rates when spatial-lag scratch is supplied', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'rate-alias-x1');
  try {
    const rates = fixture.output('rates', 'float32', 4);
    const lag = fixture.output('lag', 'float32', 4);
    const props = {
      id: 'retry',
      events: fixture.input('events', new Float32Array([2, 2, 2, 2]), 'float32', 4),
      populations: fixture.input('populations', new Float32Array([10, 10, 10, 10]), 'float32', 4),
      polygons: polygons(fixture, 4),
      parameters: fixture.parameterBuffer('parameters', 'float32', 8),
      palette: fixture.input('palette', new Uint32Array(5), 'uint32', 5),
      neighborCapacity: 16,
      pairCapacity: 16,
      outputs: {smoothedRates: rates.view}
    };
    expect(() =>
      addRateClusterMapRecipe(fixture.graph, {...props, scratch: {spatialLag: rates.view}})
    ).toThrow(/share/);
    addRateClusterMapRecipe(fixture.graph, {...props, scratch: {spatialLag: lag.view}});
    fixture.run();
    for (const value of await fixture.readFloat32(rates, 4)) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeCloseTo(0.2, 6);
    }
  } finally {
    fixture.destroy();
  }
});

it('X5 rejects duplicate physical imports and retains nearest costs independently from accessibility', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'facility-alias-x1');
  try {
    const nearest = fixture.output('nearest', 'float32', 3);
    const accessibility = fixture.output('accessibility', 'float32', 3);
    const alias = importGraphBuffer(fixture.graph, 'alias', nearest.buffer, 'float32', 3);
    const props = {
      id: 'retry',
      costs: fixture.input('costs', new Float32Array([1, 2, 3, 1, 2, 3]), 'float32', 6),
      demand: fixture.input('demand', new Float32Array([1, 1, 1]), 'float32', 3),
      supply: fixture.input('supply', new Float32Array([0, 0]), 'float32', 2),
      facilityMask: fixture.input('mask', new Uint32Array([1, 1]), 'uint32', 2),
      parameters: fixture.parameters(
        'parameters',
        'float32',
        getGPUNetworkAccessibilityParameterValues({threshold: 3})
      )
    };
    expect(() =>
      addFacilityScenarioRecipe(fixture.graph, {
        ...props,
        outputs: {nearestCosts: nearest.view, accessibility: alias}
      })
    ).toThrow(/share/);
    addFacilityScenarioRecipe(fixture.graph, {
      ...props,
      outputs: {nearestCosts: nearest.view, accessibility: accessibility.view}
    });
    fixture.run();
    expect(await fixture.readFloat32(nearest, 3)).toEqual([1, 2, 3]);
    expect(await fixture.readFloat32(accessibility, 3)).toEqual([0, 0, 0]);
  } finally {
    fixture.destroy();
  }
});

it('X6 X8 calibrates raster values while ignoring auxiliary metadata from another graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'raster-preview-x1');
  try {
    const foreign = new GPUCommandGraph(device);
    const preview = createTransientView(foreign, 'preview', 'float32', 4);
    const medians = fixture.output('medians', 'float32', 2);
    const props = {
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
        storage: {
          kind: 'buffer' as const,
          values: fixture.input('values', new Float32Array([2, 4, 6, 8]), 'float32', 4)
        }
      },
      zones: fixture.input('zones', new Uint32Array([0, 0, 1, 1]), 'uint32', 4),
      zoneCount: 2,
      outputs: {medians: medians.view},
      preview
    };
    expect(() =>
      addRasterZonalDistributionRecipe(fixture.graph, {
        ...props,
        input: {...props.input, storage: {kind: 'buffer', values: preview}}
      })
    ).toThrow(/belong/);
    const result = addRasterZonalDistributionRecipe(fixture.graph, props);
    expect(result.contributors).toHaveLength(2);
    fixture.run();
    expect(await fixture.readFloat32(medians, 2)).toEqual([3, 7]);
  } finally {
    fixture.destroy();
  }
});
