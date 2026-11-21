// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
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
import {createPointsInPolygonsChoroplethOperation} from '../../../src/gpu-spatial-analysis/index';
import {createCPUPointsInPolygonsChoroplethOperation} from '../../../src/gpu-spatial-analysis/index';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {addChangeOfSupportRecipe} from '../../../src/gpu-spatial-analysis/recipes/change-of-support-recipe';
import {getGPUPolygonRasterizationParameterValues} from '../../../src/raster-analysis/polygon-rasterization/index';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {addZonalDistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/zonal-distribution-recipe';

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

it('WRT-02 change-of-support writes distinct requested nested statuses on every submission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'wrt-02');
  try {
    const grid = makeGPURasterMetadata({width: 2, height: 1, rowDirection: 'north'});
    const geometry = polygons(fixture, 2);
    const target = {
      kind: 'polygons' as const,
      positions: fixture.input(
        'target-positions',
        Float32Array.of(0, 0, 2, 0, 2, 1, 0, 1),
        'float32x2',
        4
      ),
      featureOffsets: fixture.input('target-features', Uint32Array.of(0, 1), 'uint32', 2),
      polygonOffsets: fixture.input('target-polygons', Uint32Array.of(0, 1), 'uint32', 2),
      ringOffsets: fixture.input('target-rings', Uint32Array.of(0, 4), 'uint32', 2),
      crossingCapacity: 16
    };
    const count = fixture.output('count', 'uint32', 1);
    const candidateOverflow = fixture.output('candidate-loss', 'uint32', 1);
    const requiredCount = fixture.output('required', 'uint32', 1);
    const nestedRequiredCount = fixture.output('nested-required', 'uint32', 1);
    const overflow = fixture.output('final-overflow', 'uint32', 1);
    const result = addChangeOfSupportRecipe(fixture.graph, {
      grid,
      source: geometry,
      target,
      pairCapacity: 1,
      parameters: fixture.input(
        'raster-parameters',
        getGPUPolygonRasterizationParameterValues(grid),
        'float32',
        4
      ),
      output: {
        status: {requiredCount: requiredCount.view, overflow: overflow.view},
        extensiveWeights: {
          status: {
            count: count.view,
            candidateOverflow: candidateOverflow.view,
            requiredCount: nestedRequiredCount.view
          }
        }
      }
    });
    expect(result.outputs.extensiveWeights.status.count).toBe(count.view);
    expect(result.outputs.extensiveWeights.status.candidateOverflow).toBe(candidateOverflow.view);
    expect(result.outputs.extensiveWeights.status.requiredCount).toBe(nestedRequiredCount.view);
    expect(result.status.requiredCount).toBe(requiredCount.view);
    // Each unit source square overlaps the one two-cell target: two required pairs, only one fits.
    for (let submission = 0; submission < 2; submission++) {
      for (const output of [count, candidateOverflow, requiredCount, nestedRequiredCount, overflow])
        output.buffer.write(Uint32Array.of(913));
      fixture.run();
      expect(await fixture.readUint32(count, 1)).toEqual([1]);
      expect(await fixture.readUint32(candidateOverflow, 1)).toEqual([0]);
      expect(await fixture.readUint32(requiredCount, 1)).toEqual([2]);
      expect(await fixture.readUint32(nestedRequiredCount, 1)).toEqual([2]);
      expect(await fixture.readUint32(overflow, 1)).toEqual([1]);
    }
  } finally {
    fixture.destroy();
  }
});

it.each([
  'zonal',
  'group'
] as const)('UGE-01 %s aggregates overlapping multipart memberships by polygon row', async backend => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `uge-01-${backend}`);
  try {
    const counts = fixture.output('counts', 'uint32', 2);
    const sums = fixture.output('sums', 'float32', 2);
    const result = addPointsInPolygonsChoroplethRecipe(fixture.graph, {
      backend,
      statistic: 'sum',
      values: fixture.input('values', Float32Array.of(7), 'float32', 1),
      points: {
        kind: 'points',
        positions: fixture.input('points', Float32Array.of(0.25, 0.25, 0.75, 0.75), 'float32x2', 2),
        featureOffsets: fixture.input('point-offsets', Uint32Array.of(0, 2), 'uint32', 2),
        sourceIds: fixture.input('point-ids', Uint32Array.of(701), 'uint32', 1)
      },
      polygons: {
        kind: 'polygons',
        positions: fixture.input(
          'polygon-positions',
          Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1),
          'float32x2',
          8
        ),
        featureOffsets: fixture.input('features', Uint32Array.of(0, 1, 2), 'uint32', 3),
        polygonOffsets: fixture.input('polygons', Uint32Array.of(0, 1, 2), 'uint32', 3),
        ringOffsets: fixture.input('rings', Uint32Array.of(0, 4, 8), 'uint32', 3),
        sourceIds: fixture.input('polygon-ids', Uint32Array.of(99, 5), 'uint32', 2),
        candidateCapacity: 8
      },
      output: {counts: counts.view, featureValues: sums.view}
    });
    fixture.run();
    // Each polygon contains both parts of the same feature: count it and its value once per row.
    expect(await fixture.readUint32(counts, 2)).toEqual([1, 1]);
    expect(await fixture.readFloat32(sums, 2)).toEqual([7, 7]);
    expect(result.outputs.counts).toBe(counts.view);
  } finally {
    fixture.destroy();
  }
});

for (const inherited of [false, true])
  for (const chunked of [false, true])
    it(`B1 N1 UGE-01 ordinary choropleth and zonal distributions inherited=${inherited} chunked=${chunked}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = new RecipeTestFixture(device, 'uge-01-ordinary');
      try {
        const points = {
          kind: 'points' as const,
          positions: chunked
            ? new GraphVectorView({
                id: 'point-chunks',
                name: 'point-chunks',
                format: 'float32x2',
                length: 3,
                valueLength: 6,
                stride: 2,
                byteStride: 8,
                rowByteLength: 8,
                data: [
                  fixture.input('points-first', Float32Array.of(0.5, 0.5), 'float32x2', 1),
                  fixture.input(
                    'points-rest',
                    Float32Array.of(1.25, 0.25, 1.75, 0.75),
                    'float32x2',
                    2
                  )
                ]
              })
            : fixture.input(
                'points',
                Float32Array.of(0.5, 0.5, 1.25, 0.25, 1.75, 0.75),
                'float32x2',
                3
              ),
          sourceIds: fixture.input('point-ids', Uint32Array.of(701, 222, 333), 'uint32', 3)
        };
        const geometry = {
          ...polygons(fixture, 2),
          sourceIds: fixture.input('polygon-ids', Uint32Array.of(99, 5), 'uint32', 2)
        };
        const values = fixture.input('values', Float32Array.of(7, 11, 13), 'float32', 3);
        const rowCounts = fixture.output('rows', 'uint32', 2);
        const medians = fixture.output('medians', 'float32', 2);
        const join = {points, polygons: geometry, candidateCapacity: 8};
        const source: Parameters<typeof addZonalDistributionRecipe>[1]['source'] = inherited
          ? (Object.assign(Object.create({kind: 'points'}), {join}) as {
              kind: 'points';
              join: typeof join;
            })
          : {kind: 'points', join};
        const distribution = addZonalDistributionRecipe(fixture.graph, {
          values,
          source,
          output: {rowCounts: rowCounts.view, medians: medians.view}
        });
        expect(distribution.status.stages.map(stage => stage.stage)).toEqual([
          'zone-assignment',
          'zone-distributions'
        ]);
        const sums = (['zonal', 'group'] as const).map(backend => {
          const output = fixture.output(`${backend}-sums`, 'float32', 2);
          addPointsInPolygonsChoroplethRecipe(fixture.graph, {
            id: `ordinary-${backend}`,
            backend,
            statistic: 'sum',
            points,
            polygons: geometry,
            values,
            output: {featureValues: output.view}
          });
          return output;
        });
        fixture.run();
        // First square has [7]; second has [11,13]: its median is 12 and its sum is 24.
        expect(await fixture.readUint32(rowCounts, 2)).toEqual([1, 2]);
        expect(await fixture.readFloat32(medians, 2)).toEqual([7, 12]);
        for (const output of sums) expect(await fixture.readFloat32(output, 2)).toEqual([7, 24]);
      } finally {
        fixture.destroy();
      }
    });

for (const getOnlyOffsets of [false, true])
  it.each([
    undefined,
    'group'
  ] as const)(`B2 X1 counts two interior parts once per feature for backend %s and twice as explicit rows (get-only offsets=${getOnlyOffsets})`, async backend => {
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
      const explicitCounts = fixture.output('explicit-counts', 'uint32', 1);
      const props = {
        id: 'retry',
        backend,
        points: {kind: 'points' as const, positions},
        polygons: polygons(fixture),
        output: {counts: counts.view}
      };
      addPointsInPolygonsChoroplethRecipe(fixture.graph, {
        ...props,
        points: getOnlyOffsets
          ? new Proxy(props.points, {
              get(target, key, receiver) {
                return key === 'featureOffsets'
                  ? featureOffsets
                  : Reflect.get(target, key, receiver);
              }
            })
          : {...props.points, featureOffsets}
      });
      addPointsInPolygonsChoroplethRecipe(fixture.graph, {
        ...props,
        id: 'explicit-rows',
        output: {counts: explicitCounts.view}
      });
      fixture.run();
      // Hand-counted oracle: one grouped feature and two independent point rows in the square.
      // Dropping featureOffsets incorrectly gives two for the first result as well.
      expect(await fixture.readUint32(counts, 1)).toEqual([1]);
      expect(await fixture.readUint32(explicitCounts, 1)).toEqual([2]);
    } finally {
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
      output: {demandTimes, demandBands, bands: {counts: counts.view, keys: keys.view}}
    };
    expect(() =>
      addDriveTimeCatchmentRecipe(fixture.graph, {
        ...props,
        output: {
          ...props.output,
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
        output: {
          ...props.output,
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
        output: {
          ...props.output,
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
      output: {smoothedRates: rates.view}
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
      output: {medians: medians.view},
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

it('R3 counts every assigned raster cell while omitting missing values from statistics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'raster-assigned-rows-review');
  try {
    const zoneIds = fixture.output('zone-ids', 'uint32', 1);
    const rowCounts = fixture.output('row-counts', 'uint32', 1);
    const validCounts = fixture.output('valid-counts', 'uint32', 1);
    const medians = fixture.output('medians', 'float32', 1);
    const modes = fixture.output('modes', 'float32', 1);
    const uniqueCounts = fixture.output('unique-counts', 'uint32', 1);
    const result = addRasterZonalDistributionRecipe(fixture.graph, {
      grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      input: {
        id: 'band',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: fixture.input('values', Float32Array.of(4, NaN), 'float32', 2)
        }
      },
      zones: fixture.input('zones', Uint32Array.of(0, 0), 'uint32', 2),
      zoneCount: 1,
      output: {
        zoneIds: zoneIds.view,
        rowCounts: rowCounts.view,
        validCounts: validCounts.view,
        medians: medians.view,
        modes: modes.view,
        uniqueCounts: uniqueCounts.view
      }
    });
    expect(result.contributors).toHaveLength(2);
    fixture.run();
    // Closed form: both cells are assigned; only 4 is finite, so all order statistics equal 4.
    expect(await fixture.readUint32(zoneIds, 1)).toEqual([0]);
    expect(await fixture.readUint32(rowCounts, 1)).toEqual([2]);
    expect(await fixture.readUint32(validCounts, 1)).toEqual([1]);
    expect(await fixture.readFloat32(medians, 1)).toEqual([4]);
    expect(await fixture.readFloat32(modes, 1)).toEqual([4]);
    expect(await fixture.readUint32(uniqueCounts, 1)).toEqual([1]);
  } finally {
    fixture.destroy();
  }
});

for (const {multipart, polygonCount, leafCapacity, expectedOverflow, chunks} of [
  {
    multipart: true,
    polygonCount: 1,
    leafCapacity: undefined,
    expectedOverflow: 0,
    chunks: undefined
  },
  {
    multipart: false,
    polygonCount: 1,
    leafCapacity: undefined,
    expectedOverflow: 0,
    chunks: undefined
  },
  {multipart: true, polygonCount: 2, leafCapacity: 1, expectedOverflow: 1, chunks: undefined},
  {multipart: false, polygonCount: 1, leafCapacity: undefined, expectedOverflow: 0, chunks: 0},
  {multipart: false, polygonCount: 1, leafCapacity: undefined, expectedOverflow: 0, chunks: 2}
])
  for (const backend of ['zonal', 'group'] as const)
    for (const statistic of ['count', 'sum', 'mean', 'minimum', 'maximum'] as const)
      it(`R5a R5b W5-13 empty multipart=${multipart} chunks=${chunks} polygons=${polygonCount} ${backend} ${statistic} agrees across recipe and both operations`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const fixture = new RecipeTestFixture(device, `empty-${backend}-${statistic}`);
        const expectedCounts = polygonCount === 1 ? [0] : [0, 0];
        // Closed-form empty reduction: zero counts/sums, undefined (NaN) mean and extrema.
        const assertRows = (rows: {
          counts: ArrayLike<number>;
          featureValues?: ArrayLike<number>;
          overflow: ArrayLike<number>;
          uncertainCount: ArrayLike<number>;
        }) => {
          expect(Array.from(rows.counts)).toEqual(expectedCounts);
          // No point queries imply no uncertainty; an undersized polygon BVH still reports loss.
          expect(Array.from(rows.overflow)).toEqual([expectedOverflow]);
          expect(Array.from(rows.uncertainCount)).toEqual([0]);
          if (statistic === 'sum') expect(Array.from(rows.featureValues!)).toEqual(expectedCounts);
          else if (statistic !== 'count') {
            expect(rows.featureValues).toHaveLength(polygonCount);
            for (const value of Array.from(rows.featureValues!))
              expect(Number.isNaN(value)).toBe(true);
          }
        };
        try {
          const counts = fixture.output('counts', 'uint32', polygonCount);
          const values = fixture.output('feature-values', 'float32', polygonCount);
          const overflow = fixture.output('overflow', 'uint32', 1);
          const uncertainCount = fixture.output('uncertainty', 'uint32', 1);
          // Poison every result: a missing empty-batch write must fail, including zero statuses.
          counts.buffer.write(new Uint32Array(polygonCount).fill(0xdeadbeef));
          values.buffer.write(new Float32Array(polygonCount).fill(123));
          overflow.buffer.write(Uint32Array.of(0xdeadbeef));
          uncertainCount.buffer.write(Uint32Array.of(0xdeadbeef));
          addPointsInPolygonsChoroplethRecipe(fixture.graph, {
            backend,
            statistic,
            points: {
              kind: 'points',
              positions:
                chunks === undefined
                  ? fixture.input('points', Float32Array.of(0, 0), 'float32x2', 0)
                  : new GraphVectorView({
                      id: 'empty-vector',
                      name: 'empty-vector',
                      format: 'float32x2',
                      length: 0,
                      valueLength: 0,
                      stride: 2,
                      byteStride: 8,
                      rowByteLength: 8,
                      data: Array.from({length: chunks}, (_, index) =>
                        fixture.input(`empty-${index}`, Float32Array.of(0, 0), 'float32x2', 0)
                      )
                    }),
              ...(multipart
                ? {featureOffsets: fixture.input('offsets', Uint32Array.of(0), 'uint32', 1)}
                : {})
            },
            polygons: {...polygons(fixture, polygonCount), leafCapacity},
            values: fixture.input('values', Float32Array.of(0), 'float32', 0),
            output: {
              counts: counts.view,
              ...(statistic === 'count' ? {} : {featureValues: values.view}),
              status: {overflow: overflow.view, uncertainCount: uncertainCount.view}
            }
          });
          fixture.run();
          assertRows({
            counts: await fixture.readUint32(counts, polygonCount),
            featureValues: await fixture.readFloat32(values, polygonCount),
            overflow: await fixture.readUint32(overflow, 1),
            uncertainCount: await fixture.readUint32(uncertainCount, 1)
          });
          for (const createOperation of [
            createPointsInPolygonsChoroplethOperation,
            createCPUPointsInPolygonsChoroplethOperation
          ]) {
            const outputs = {
              counts: 'counts',
              ...(statistic === 'count' ? {} : {featureValues: 'featureValues'}),
              overflow: 'overflow',
              uncertainCount: 'uncertainCount'
            };
            const rows = await runOperationGraph(device, {
              inputs: {
                points: {format: 'float32x2', values: new Float32Array(0)},
                offsets: {format: 'uint32', values: Uint32Array.of(0)},
                values: {format: 'float32', values: new Float32Array(0)},
                polygonPositions: {
                  format: 'float32x2',
                  values: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1).subarray(
                    0,
                    polygonCount * 8
                  )
                },
                features: {
                  format: 'uint32',
                  values: polygonCount === 1 ? Uint32Array.of(0, 1) : Uint32Array.of(0, 1, 2)
                },
                polygons: {
                  format: 'uint32',
                  values: polygonCount === 1 ? Uint32Array.of(0, 1) : Uint32Array.of(0, 1, 2)
                },
                rings: {
                  format: 'uint32',
                  values: polygonCount === 1 ? Uint32Array.of(0, 4) : Uint32Array.of(0, 4, 8)
                }
              },
              operations: [
                createOperation({
                  backend,
                  statistic,
                  candidateCapacity: 8,
                  leafCapacity,
                  points: {
                    kind: 'points',
                    positions: {value: 'points', length: 0},
                    ...(multipart ? {featureOffsets: {value: 'offsets', length: 1}} : {})
                  },
                  polygons: {
                    kind: 'polygons',
                    positions: {value: 'polygonPositions', length: 4 * polygonCount},
                    featureOffsets: {value: 'features', length: polygonCount + 1},
                    polygonOffsets: {value: 'polygons', length: polygonCount + 1},
                    ringOffsets: {value: 'rings', length: polygonCount + 1}
                  },
                  ...(statistic === 'count' ? {} : {values: {value: 'values', length: 0}}),
                  output: outputs
                })
              ],
              outputs: Object.values(outputs)
            });
            assertRows(rows as never);
          }
        } finally {
          fixture.destroy();
        }
      });

it.each([
  'count',
  'sum',
  'mean',
  'minimum',
  'maximum'
] as const)('RCP-108 public nonempty planar %s CPU and lowered GPU agree with hand tally', async statistic => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two points in the left square, one in the right, all strictly interior.
  // Values [2,4,8]: left sum 6 and mean 3; right sum/mean 8.
  const expected = {sum: [6, 8], mean: [3, 8], minimum: [2, 8], maximum: [4, 8]};
  for (const createOperation of [
    createCPUPointsInPolygonsChoroplethOperation,
    createPointsInPolygonsChoroplethOperation
  ]) {
    const destinations = {
      counts: 'counts',
      ...(statistic === 'count' ? {} : {featureValues: 'featureValues'}),
      overflow: 'overflow',
      uncertainCount: 'uncertainCount'
    };
    const rows = await runOperationGraph(device, {
      inputs: {
        pointPositions: {
          format: 'float32x2',
          values: Float32Array.of(0.25, 0.25, 0.75, 0.75, 1.5, 0.5)
        },
        values: {format: 'float32', values: Float32Array.of(2, 4, 8)},
        polygonPositions: {
          format: 'float32x2',
          values: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1)
        },
        features: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
        polygons: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
        rings: {format: 'uint32', values: Uint32Array.of(0, 4, 8)}
      },
      operations: [
        createOperation({
          statistic,
          candidateCapacity: 8,
          points: {kind: 'points', positions: {value: 'pointPositions', length: 3}},
          polygons: {
            kind: 'polygons',
            positions: {value: 'polygonPositions', length: 8},
            featureOffsets: {value: 'features', length: 3},
            polygonOffsets: {value: 'polygons', length: 3},
            ringOffsets: {value: 'rings', length: 3}
          },
          ...(statistic === 'count' ? {} : {values: {value: 'values', length: 3}}),
          output: destinations
        })
      ],
      outputs: Object.values(destinations)
    });
    expect(Array.from(rows['counts'])).toEqual([2, 1]);
    expect(Array.from(rows['overflow'])).toEqual([0]);
    expect(Array.from(rows['uncertainCount'])).toEqual([0]);
    if (statistic !== 'count') {
      expect(rows['featureValues']).toHaveLength(2);
      Array.from(rows['featureValues']).forEach((value, row) => {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBe(expected[statistic][row]);
      });
    }
  }
});
