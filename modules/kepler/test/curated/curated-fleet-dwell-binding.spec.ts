// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {
  computeTrajectoryOracle,
  createDoubleOracleTime,
  generateTrajectoryTracks
} from '../../../analysis/test/gpu-spatial-analysis/trajectory-analysis/trajectory-metrics-oracle';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import type {
  CatalogFeatureCollection,
  CatalogOutputTable
} from '../../src/analysis/catalog-geojson';
import {isCatalogPointInPolygon} from '../../src/analysis/catalog-gpu-pack';
import {CURATED_FLEET_DWELL_BINDING, prepareFleetDwell} from '../../src/curated/bindings/index';
import {createFleetDwellSample} from '../../src/curated/samples/index';
import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest
} from '../../src/curated/curated-analysis-mapping';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import {CURATED_FLEET_DWELL_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-fleet-dwell';

/** The original fleet-dwell recipe fixture: same tracks and fractional rectangles. */
function recipeFixture(): Record<string, CatalogFeatureCollection> {
  const generated = generateTrajectoryTracks(7, 40, 15, 30, 1);
  const tracks: CatalogFeatureCollection = {type: 'FeatureCollection', features: []};
  for (let track = 0; track + 1 < generated.trackOffsets.length; track++) {
    const first = generated.trackOffsets[track];
    const end = generated.trackOffsets[track + 1];
    // Empty tracks have no source feature, as in ordinary input preparation.
    for (let row = first; row < end; row++)
      tracks.features.push({
        geometry: {
          type: 'Point',
          coordinates: [...generated.positions.slice(row * 2, row * 2 + 2)]
        },
        properties: {vehicle_code: `fixture-${track}`, recorded_at: generated.timestamps[row]}
      });
  }
  const zones: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      [10, -300.37, -300.41, -0.37, -0.41],
      [20, -0.37, -300.41, 300.37, -0.41],
      [30, -300.37, -0.41, 300.37, 300.41]
    ].map(([id, minimumX, minimumY, maximumX, maximumY]) => ({
      id,
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [minimumX, minimumY],
            [maximumX, minimumY],
            [maximumX, maximumY],
            [minimumX, maximumY],
            [minimumX, minimumY]
          ]
        ]
      },
      properties: {}
    }))
  };
  tracks.features.reverse();
  return {tracks, zones};
}

function compareWithRecipeOracle(
  prepared: ReturnType<typeof prepareFleetDwell>,
  stops: CatalogOutputTable,
  dwell: CatalogOutputTable
) {
  const {trajectory, zones, values} = prepared;
  const oracle = computeTrajectoryOracle(
    {
      positions: trajectory.positions,
      timestamps: new Float32Array(trajectory.fixes.length),
      trackOffsets: [...trajectory.trackOffsets]
    },
    Number(values['stopSpeedThreshold']) / 1000,
    Number(values['stopMinimumDuration']) * 1000,
    createDoubleOracleTime(trajectory.fixes.map(fix => fix.timestamp))
  );
  const mismatches: string[] = [];
  const totals = Array(zones.featureCount).fill(0) as number[];
  const counts = Array(zones.featureCount).fill(0) as number[];
  const maximums = Array(zones.featureCount).fill(0) as number[];
  const near = (actual: unknown, expected: number, label: string, tolerance = 1e-3) => {
    if (
      typeof actual !== 'number' ||
      !Number.isFinite(actual) ||
      Math.abs(actual - expected) > tolerance
    )
      mismatches.push(`${label}: got ${String(actual)}, expected ${expected} ± ${tolerance}`);
  };
  if (stops.rowCount !== oracle.stops.length)
    throw new Error(
      `Fleet dwell recipe parity failed: stop count ${stops.rowCount} versus ${oracle.stops.length}`
    );
  for (const [index, stop] of oracle.stops.entries()) {
    const first = trajectory.fixes[stop.startRow];
    const last = trajectory.fixes[stop.endRow];
    const zone = zones.parts.findIndex(parts =>
      parts.some(rings => isCatalogPointInPolygon(stop.centroid, rings))
    );
    const seconds = stop.duration / 1000;
    if (stops.columns['track_id'][index] !== trajectory.trackIds[stop.track])
      mismatches.push(`stop ${index}: track identity`);
    for (const [column, value] of [
      ['source_index', first.sourceIndex],
      ['vertex_index', first.vertexIndex],
      ['end_source_index', last.sourceIndex],
      ['end_vertex_index', last.vertexIndex],
      ['start_time', first.timestamp],
      ['end_time', last.timestamp],
      ['duration_seconds', seconds]
    ] as const)
      near(stops.columns[column][index], value, `stop ${index}.${column}`);
    const sourceZone = zone < 0 ? null : zones.sourceIndices[zone];
    if (stops.columns['zone_index'][index] !== sourceZone)
      mismatches.push(
        `stop ${index}: zone ${stops.columns['zone_index'][index]} versus ${sourceZone}`
      );
    const expectedPosition = trajectory.frame.inverse(...stop.centroid);
    const geometry = JSON.parse(String(stops.columns['geometry'][index]));
    near(geometry.coordinates[0], expectedPosition[0], `stop ${index}.x`, 1e-3);
    near(geometry.coordinates[1], expectedPosition[1], `stop ${index}.y`, 1e-3);
    if (zone >= 0) {
      totals[zone] += seconds;
      counts[zone]++;
      maximums[zone] = Math.max(maximums[zone], seconds);
    }
  }
  for (let zone = 0; zone < zones.featureCount; zone++) {
    near(dwell.columns['stop_count'][zone], counts[zone], `zone ${zone}.count`);
    near(dwell.columns['total_dwell_seconds'][zone], totals[zone], `zone ${zone}.total`, 1e-2);
    if (counts[zone]) {
      near(
        dwell.columns['mean_dwell_seconds'][zone],
        totals[zone] / counts[zone],
        `zone ${zone}.mean`
      );
      near(dwell.columns['maximum_dwell_seconds'][zone], maximums[zone], `zone ${zone}.maximum`);
    } else if (
      dwell.columns['mean_dwell_seconds'][zone] !== null ||
      dwell.columns['maximum_dwell_seconds'][zone] !== null
    )
      mismatches.push(`zone ${zone}: empty mean/maximum must be null`);
    if (
      dwell.columns['geometry'][zone] !==
      JSON.stringify(prepared.collection.features[zones.sourceIndices[zone]].geometry)
    )
      mismatches.push(`zone ${zone}: publication geometry identity`);
  }
  if (mismatches.length)
    throw new Error(`Fleet dwell recipe parity failed:\n${mismatches.join('\n')}`);
}

for (const fixture of ['sample', 'independently-mapped-recipe'] as const) {
  it(`fleet dwell dispatches the recipe and publishes oracle-matching outputs on ${fixture}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const sample = fixture === 'sample' ? createFleetDwellSample() : recipeFixture();
    const independent = fixture !== 'sample';
    const analysis = getKeplerCuratedAnalysis('fleet-dwell')!;
    const profiles = Object.entries(sample).map(([datasetId, collection]) =>
      profileKeplerFeatureCollection(collection, {
        datasetId,
        label: datasetId,
        coordinateSystem: independent ? 'planar' : 'EPSG:4326'
      })
    );
    let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles);
    for (const [role, column] of [
      ['track', independent ? 'vehicle_code' : 'track_id'],
      ['time', independent ? 'recorded_at' : 'timestamp']
    ])
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'tracks', role, columns: [column]},
        analysis,
        profiles
      );
    const request = buildKeplerCuratedAnalysisRequest(analysis, mapping, {
      parameterOverrides: {
        coordinateSystem: independent ? 'planar' : 'EPSG:4326',
        timestampEncoding: independent ? 'unix-second' : 'unix-millisecond',
        stopMinimumDuration: independent ? 3 : 20
      }
    });
    const prepared = prepareFleetDwell(sample, request.parameters);
    const result = await runCatalogOperationAsync(
      CURATED_FLEET_DWELL_BINDING,
      request.parameters,
      sample,
      {device}
    );
    expect(result.engine).toBe('gpu');
    expect(result.dispatchedContributors).toEqual([
      'GPUTrajectoryMetrics',
      'GPUPointInPolygonJoin',
      'GPUGroupStatistics'
    ]);
    expect(result.summary['stopOverflow']).toBe(0);
    expect(result.summary['joinOverflow']).toBe(0);
    expect(result.summary['joinUncertainCount']).toBe(0);
    compareWithRecipeOracle(prepared, result.tables[0], result.tables[1]);
    for (const [index, table] of result.tables.entries())
      expect(
        planKeplerResultPresentation(table, CURATED_FLEET_DWELL_DESCRIPTOR.outputs[index])?.column
      ).toBe(index ? 'total_dwell_seconds' : 'duration_seconds');
    if (!independent) {
      expect(result.summary['stopCount']).toBe(128);
      expect(result.summary['unassignedStopCount']).toBe(64);
      expect(result.tables[1].columns['stop_count']).toEqual([32, 32, 0]);
    } else {
      const originalTracks = generateTrajectoryTracks(7, 40, 15, 30, 1);
      const originalFixture = computeTrajectoryOracle(originalTracks, 1, 3);
      expect(
        result.summary['stopCount'],
        'Original fleet-dwell-recipe.spec.ts fixture stop count'
      ).toBe(originalFixture.stops.length);
    }
  }, 120000);
}

it('fleet dwell refuses real stop/join capacity overflow without publishing truncated totals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capacity of [{stopCapacity: 1}, {candidateCapacity: 1}]) {
    await expect(
      runCatalogOperationAsync(CURATED_FLEET_DWELL_BINDING, capacity, createFleetDwellSample(), {
        device
      })
    ).rejects.toThrow(/curated-fleet-dwell: capacity overflow.*no partial result published/);
  }
}, 120000);

it('fleet dwell retains timed-line vertices and signed times, respects holes and overlap, and handles no stops', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const tracks: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        id: 'timed-line',
        geometry: {
          type: 'LineString',
          coordinates: [
            [0, 0],
            [0, 0],
            [0, 0],
            [10, 0],
            [10, 0],
            [10, 0]
          ]
        },
        properties: {observed: [-30, -20, -10, 0, 10, 20]}
      },
      {
        id: 'singleton',
        geometry: {type: 'Point', coordinates: [20, 20]},
        properties: {observed: -1}
      }
    ]
  };
  const square = (minimumX: number, minimumY: number, maximumX: number, maximumY: number) => [
    [minimumX, minimumY],
    [maximumX, minimumY],
    [maximumX, maximumY],
    [minimumX, maximumY],
    [minimumX, minimumY]
  ];
  const zones: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        id: 'hole',
        geometry: {type: 'Polygon', coordinates: [square(-5, -5, 5, 5), square(-1, -1, 1, 1)]}
      },
      {
        id: 'first-overlap',
        geometry: {
          type: 'MultiPolygon',
          coordinates: [[square(9, -1, 11, 1)], [square(30, 30, 40, 40)]]
        }
      },
      {id: 'second-overlap', geometry: {type: 'Polygon', coordinates: [square(9, -1, 11, 1)]}}
    ]
  };
  const parameters = {
    coordinateSystem: 'planar',
    timeProperty: 'observed',
    timestampEncoding: 'unix-second',
    stopSpeedThreshold: 0.5
  };
  const result = await runCatalogOperationAsync(
    CURATED_FLEET_DWELL_BINDING,
    parameters,
    {tracks, zones},
    {device}
  );
  const [stops, dwell] = result.tables;
  const report = JSON.stringify({
    summary: result.summary,
    stops: stops.columns,
    dwell: dwell.columns
  });
  expect(stops.columns['track_id'], report).toEqual(['timed-line', 'timed-line']);
  expect(stops.columns['source_index'], report).toEqual([0, 0]);
  expect(stops.columns['vertex_index'], report).toEqual([0, 3]);
  expect(stops.columns['end_vertex_index'], report).toEqual([2, 5]);
  expect(stops.columns['start_time'], report).toEqual([-30000, 0]);
  expect(stops.columns['duration_seconds'], report).toEqual([20, 20]);
  expect(stops.columns['zone_id'], report).toEqual([null, 'first-overlap']);
  expect(dwell.columns['total_dwell_seconds'], report).toEqual([0, 20, 0]);
  const empty = await runCatalogOperationAsync(
    CURATED_FLEET_DWELL_BINDING,
    {...parameters, stopMinimumDuration: 1000},
    {tracks, zones},
    {device}
  );
  expect(empty.tables[0].rowCount, JSON.stringify(empty.summary)).toBe(0);
  expect(empty.tables[1].columns['stop_count']).toEqual([0, 0, 0]);
  expect(empty.tables[1].columns['total_dwell_seconds']).toEqual([0, 0, 0]);
}, 120000);
