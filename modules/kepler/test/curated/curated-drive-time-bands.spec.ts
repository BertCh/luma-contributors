// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {getDriveTimeOracle} from '../../../analysis/test/gpu-network/recipes/drive-time-oracle';
import {
  snapOracle,
  createGridFixture
} from '../../../analysis/test/gpu-network/network-accessibility/network-accessibility-oracle';
import {createSeededRandom} from '../../../analysis/test/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-oracle';
import type {
  CatalogFeatureCollection,
  CatalogOutputTable
} from '../../src/analysis/catalog-geojson';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import {
  CURATED_DRIVE_TIME_BANDS_BINDING,
  prepareCuratedDriveTimeBands
} from '../../src/curated/bindings/curated-drive-time-bands-binding';
import {
  CURATED_DRIVE_TIME_BANDS_ANALYSIS,
  DRIVE_TIME_BANDS_SAMPLE,
  DRIVE_TIME_BANDS_EXPECTATIONS,
  DRIVE_TIME_BANDS_SECOND_SAMPLE,
  DRIVE_TIME_BANDS_SECOND_EXPECTATIONS,
  createDriveTimeBandsSample,
  createDriveTimeBandsSecondDataset
} from '../../src/curated/samples/drive-time-bands';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest
} from '../../src/curated/curated-analysis-mapping';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {CURATED_DRIVE_TIME_BANDS_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-drive-time-bands';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';

function compareColumn(
  table: CatalogOutputTable,
  column: string,
  expected: readonly (number | null)[],
  identifier: string,
  tolerance = 1e-5
): void {
  const actual = table.columns[column];
  const mismatches: string[] = [];
  if (actual.length !== expected.length)
    mismatches.push(`length ${actual.length} != ${expected.length}`);
  expected.forEach((value, row) => {
    const observed = actual[row];
    if (
      value === null
        ? observed !== null
        : typeof observed !== 'number' ||
          !Number.isFinite(observed) ||
          Math.abs(observed - value) > tolerance
    )
      mismatches.push(
        `row ${row}: ${String(observed)} != ${String(value)} (absolute tolerance ${tolerance})`
      );
  });
  if (mismatches.length)
    throw new Error(
      `drive-time-bands ${identifier} ${table.id}.${column}:\n${mismatches.join('\n')}`
    );
}

/** Test containment in published triangles, independently of their raster generation. */
function requireBandCoverage(
  table: CatalogOutputTable,
  band: number,
  point: readonly [number, number],
  identifier: string
): void {
  const counts: Record<number, number> = {};
  let covered = false;
  table.columns['band'].forEach((value, row) => {
    const observedBand = Number(value);
    counts[observedBand] = (counts[observedBand] ?? 0) + 1;
    if (observedBand !== band) return;
    const ring = JSON.parse(String(table.columns['geometry'][row])).coordinates[0] as number[][];
    const twiceArea =
      (ring[1][0] - ring[0][0]) * (ring[2][1] - ring[0][1]) -
      (ring[1][1] - ring[0][1]) * (ring[2][0] - ring[0][0]);
    if (Math.abs(twiceArea) < 1e-10) return;
    const signs = ring.slice(0, 3).map((start, vertex) => {
      const end = ring[(vertex + 1) % 3];
      return (
        (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])
      );
    });
    if (signs.every(value => value >= -1e-5) || signs.every(value => value <= 1e-5)) covered = true;
  });
  if (!covered)
    throw new Error(
      `drive-time ${identifier}: map band ${band} does not cover [${point}]; triangle counts ${JSON.stringify(counts)}`
    );
}

function mapInputs(
  inputs: Record<string, CatalogFeatureCollection>,
  columns: Readonly<Record<string, string>>,
  parameters: Readonly<Record<string, CatalogParameterValue>>
) {
  const analysis = CURATED_DRIVE_TIME_BANDS_ANALYSIS;
  const profiles = Object.entries(inputs).map(([slot, collection]) =>
    profileKeplerFeatureCollection(collection, {
      datasetId: slot,
      label: slot,
      coordinateSystem: 'planar',
      declarations: {coordinateModel: 'planar', rowSemantics: 'objects'}
    })
  );
  let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles, {parameters});
  for (const slot of analysis.slots)
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: slot.name, datasetId: slot.name},
      analysis,
      profiles
    );
  for (const [key, column] of Object.entries(columns)) {
    const [slot, role] = key.split('.');
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot, role, columns: [column]},
      analysis,
      profiles
    );
  }
  if (!mapping.ready)
    throw new Error(`drive-time-bands mapping failed: ${JSON.stringify(mapping.problems)}`);
  const request = buildKeplerCuratedAnalysisRequest(analysis, mapping);
  return {
    inputs: Object.fromEntries(
      Object.entries(request.inputDatasetIds).map(([port, datasetId]) => [port, inputs[datasetId]])
    ),
    parameters: request.parameters
  };
}

for (const fixture of [
  {
    inputs: createDriveTimeBandsSample,
    sample: DRIVE_TIME_BANDS_SAMPLE,
    expected: DRIVE_TIME_BANDS_EXPECTATIONS
  },
  {
    inputs: createDriveTimeBandsSecondDataset,
    sample: DRIVE_TIME_BANDS_SECOND_SAMPLE,
    expected: DRIVE_TIME_BANDS_SECOND_EXPECTATIONS
  }
]) {
  it(`drive-time bands dispatches the recipe and publishes the mapped ${fixture.sample.id}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const mapped = mapInputs(fixture.inputs(), fixture.sample.columns, fixture.sample.parameters);
    const result = await CURATED_DRIVE_TIME_BANDS_BINDING.runGpu!(
      {device},
      mapped.inputs,
      mapped.parameters
    );
    const [demand, bands, triangles] = result.tables;
    compareColumn(
      demand,
      'travel_time',
      fixture.expected.times.map(time => (Number.isFinite(time) ? time : null)),
      fixture.sample.id
    );
    compareColumn(
      demand,
      'band',
      fixture.expected.demandBands.map(band => (band === 0xffffffff ? null : band)),
      fixture.sample.id,
      0
    );
    compareColumn(bands, 'demand_count', fixture.expected.counts, fixture.sample.id, 0);
    compareColumn(bands, 'demand_sum', fixture.expected.sums, fixture.sample.id);
    compareColumn(bands, 'demand_mean', fixture.expected.means, fixture.sample.id);
    for (const contributor of [
      'GPUNetworkSnapping',
      'GPUNetworkServiceAreas',
      'GPUNetworkIsochrones',
      'GPUGroupStatistics'
    ])
      expect(
        result.dispatchedContributors,
        `actual recipe contributors: ${JSON.stringify(result.dispatchedContributors)}`
      ).toContain(contributor);
    expect(result.summary['converged']).toBe(1);
    for (let band = 0; band < fixture.sample.parameters.costBreaks.length; band++)
      if (!triangles.columns['band'].includes(band))
        throw new Error(
          `drive-time ${fixture.sample.id}: map band ${band} disappeared; present bands ${JSON.stringify([...new Set(triangles.columns['band'])])}`
        );
    if (fixture.sample.id === DRIVE_TIME_BANDS_SAMPLE.id) {
      requireBandCoverage(triangles, 0, [2, 0], fixture.sample.id);
      requireBandCoverage(triangles, 1, [10, 0], fixture.sample.id);
      requireBandCoverage(triangles, 2, [12, 0], fixture.sample.id);
    }

    expect(triangles.rowCount, 'actual raster isoband geometry must be nonempty').toBeGreaterThan(
      0
    );
    expect(demand.columns['source_id']).toEqual(
      mapped.inputs['demand'].features.map(feature => feature.id)
    );
    for (const table of [demand, triangles]) {
      const output = CURATED_DRIVE_TIME_BANDS_DESCRIPTOR.outputs.find(
        candidate => candidate.tableId === table.id
      )!;
      const presentation = planKeplerResultPresentation(table, output);
      expect(presentation, `${table.id} must publish categorical band metadata`).toBeDefined();
      for (const [row, text] of table.columns[table.geometryColumn!].entries()) {
        const geometry = JSON.parse(String(text));
        if (
          table.id === 'isobands' &&
          (geometry.type !== 'Polygon' ||
            geometry.coordinates[0].length !== 4 ||
            !geometry.coordinates[0].flat().every(Number.isFinite))
        )
          throw new Error(
            `drive-time-bands ${fixture.sample.id} malformed triangle ${row}: ${text}`
          );
      }
    }
  }, 120000);
}

it('drive-time bands matches the existing 6 by 5 recipe fixture and inserted-node oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Reproduce drive-time-catchment-recipe.spec.ts's first fixture using its fixture generators,
  // facility locations, seed 21 and demand/value generation. Exercise the binding preparation too.
  const {positions, edges} = createGridFixture(3, 6, 5);
  const random = createSeededRandom(21);
  const inputs: Record<string, CatalogFeatureCollection> = {
    network: {
      type: 'FeatureCollection',
      features: edges.map(([from, to, cost], index) => ({
        id: `road-${index}`,
        geometry: {
          type: 'LineString',
          coordinates: [
            [positions[2 * from], positions[2 * from + 1]],
            [positions[2 * to], positions[2 * to + 1]]
          ]
        },
        properties: {time: cost, direction: 'forward'}
      }))
    },
    facilities: {
      type: 'FeatureCollection',
      features: [
        [1.2, 1.3],
        [4.4, 3.1]
      ].map((coordinates, index) => ({
        id: `facility-${index}`,
        geometry: {type: 'Point', coordinates}
      }))
    },
    demand: {
      type: 'FeatureCollection',
      features: Array.from({length: 90}, (_, index) => ({
        id: `demand-${index}`,
        geometry: {type: 'Point', coordinates: [-0.4 + random() * 5.8, -0.4 + random() * 4.8]},
        properties: {weight: 1 + Math.floor(random() * 9)}
      }))
    }
  };
  const parameters = {
    coordinateSystem: 'planar',
    forwardCostProperty: 'time',
    onewayProperty: 'direction',
    weightProperty: 'weight',
    costUnit: 'minutes',
    maximumSnapDistance: 2,
    costBreaks: [3.3, 6.7, 11.1, 20.3],
    bufferRadius: 0.4,
    gridSize: 32
  };
  const prepared = prepareCuratedDriveTimeBands(inputs, parameters);
  const sources = new Uint32Array(prepared.topology.neighbors.length);
  for (let node = 0; node < prepared.graph.nodes.length; node++)
    sources.fill(node, prepared.topology.offsets[node], prepared.topology.offsets[node + 1]);
  const topology = {...prepared.topology, sources};
  const snapOptions = {
    edgeCosts: topology.weights,
    maximumSnapDistance: prepared.maximumSnapDistance
  };
  const facilities = snapOracle(
    prepared.facilityPositions,
    prepared.nodePositions,
    sources,
    topology.neighbors,
    snapOptions
  );
  const demand = snapOracle(
    prepared.demandPositions,
    prepared.nodePositions,
    sources,
    topology.neighbors,
    snapOptions
  );
  const expected = getDriveTimeOracle(topology, prepared.graph.nodes.length, facilities, demand);
  const result = await CURATED_DRIVE_TIME_BANDS_BINDING.runGpu!({device}, inputs, parameters);
  compareColumn(
    result.tables[0],
    'travel_time',
    expected.demandTimes,
    'existing-recipe-grid',
    0.003
  );
  const mismatches: string[] = [];
  expected.demandTimes.forEach((time, row) => {
    if (parameters.costBreaks.some(limit => Math.abs(time - limit) < 0.003)) return;
    const expectedBand = parameters.costBreaks.findIndex(limit => time <= limit);
    const actualBand = result.tables[0].columns['band'][row];
    if (actualBand !== (expectedBand < 0 ? null : expectedBand))
      mismatches.push(`row ${row}: ${String(actualBand)} != ${expectedBand}, oracle time ${time}`);
  });
  if (mismatches.length)
    throw new Error(`drive-time-bands existing recipe band mismatches:\n${mismatches.join('\n')}`);
  expect(result.tables[2].rowCount).toBeGreaterThan(0);
}, 120000);

it('drive-time grid size 64 retains the full requested corridor radius and first band', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await CURATED_DRIVE_TIME_BANDS_BINDING.runGpu!(
    {device},
    createDriveTimeBandsSample(),
    {...DRIVE_TIME_BANDS_SAMPLE.parameters, gridSize: 64}
  );
  const triangles = result.tables.find(table => table.id === 'isobands')!;
  // Radius 5 around the interior source must reach both sides at distance 4.4. The old
  // 16-pixel clipped window touched only about +/-2.5 at resolution 64, and endpoint-only
  // costs could not emit any band-zero triangles at all.
  requireBandCoverage(triangles, 0, [2, 0], '64-pixel source');
  requireBandCoverage(triangles, 0, [2, 4.4], '64-pixel north corridor');
  requireBandCoverage(triangles, 0, [2, -4.4], '64-pixel south corridor');
}, 120000);
