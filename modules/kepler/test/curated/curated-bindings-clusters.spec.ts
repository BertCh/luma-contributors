// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {expect, it} from 'vitest';
import type {
  CatalogParameterValue,
  CatalogRunResult
} from '../../src/analysis/catalog-binding-types';
import type {
  CatalogCoordinateSystem,
  CatalogFeatureCollection,
  CatalogOutputTable,
  CatalogPosition
} from '../../src/analysis/catalog-geojson';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import {CURATED_CLUSTERS_AND_FOOTPRINTS_BINDING} from '../../src/curated/bindings/curated-clusters-and-footprints-binding';
import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {createKeplerCuratedAnalysisSample} from '../../src/curated/curated-analysis-samples';
import {
  buildKeplerCuratedAnalysisRequest,
  createKeplerCuratedAnalysisMapping
} from '../../src/curated/curated-analysis-mapping';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {getKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {DBSCAN_GEOGRAPHIC, DBSCAN_PLANAR} from './curated-bindings-oracle-data';
import {
  expectCuratedClusterFootprintMeasures,
  expectCuratedClusterRingAdjacency
} from './curated-bindings-clusters-test-support';
import {
  clusterPointsOracle,
  createClusteredPoints,
  separateNearEpsilonPairs
} from '../../../analysis/test/gpu-spatial-analysis/spatial-clustering/spatial-clustering-oracle';
import {computeGroupConvexHullOracle} from '../../../analysis/test/gpu-spatial-analysis/group-geometry/group-geometry-oracle';

const BINDING = CURATED_CLUSTERS_AND_FOOTPRINTS_BINDING;

function makePoints(coordinates: readonly (readonly number[])[]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map((coordinates, index) => ({
      id: `independent-${index}`,
      properties: {observation: index},
      geometry: {type: 'Point', coordinates: [coordinates[0], coordinates[1]]}
    }))
  };
}

function getTable(result: CatalogRunResult, tableId: string): CatalogOutputTable {
  const table = result.tables.find(candidate => candidate.id === tableId);
  if (!table)
    throw new Error(`${BINDING.id}: missing ${tableId}; ${JSON.stringify(result.summary)}`);
  return table;
}

/** Includes fallback/status reports in thrown assertion text: the headless runner omits logs. */
function makeReport(datasetId: string, result: CatalogRunResult): string {
  return JSON.stringify({
    datasetId,
    engine: result.engine,
    dispatched: result.dispatchedContributors,
    fallbackReason: result.fallbackReason,
    parameters: result.parameters,
    diagnostics: result.diagnostics,
    summary: result.summary
  });
}

async function runMapped(
  datasetId: string,
  points: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem,
  overrides: Record<string, CatalogParameterValue> = {}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const analysis = getKeplerCuratedAnalysis('clusters-and-footprints')!;
  const profile = profileKeplerFeatureCollection(points, {
    datasetId,
    label: datasetId,
    coordinateSystem
  });
  const mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
  expect(mapping.complete, datasetId).toBe(true);
  const request = buildKeplerCuratedAnalysisRequest(analysis, mapping, {
    parameterOverrides: overrides
  });
  expect(request.inputDatasetIds['points'], datasetId).toBe(datasetId);
  const parameters = resolveKeplerOperationParameters(
    BINDING,
    getKeplerOperationDescriptor(BINDING.id)!,
    {points: profile},
    {overrides: request.parameters}
  ).parameters;
  const inputs = {[Object.keys(request.inputDatasetIds)[0]]: points};
  const reference = await runCatalogOperationAsync(BINDING, parameters, inputs, {engine: 'cpu'});
  const result = await runCatalogOperationAsync(BINDING, parameters, inputs, {device});
  const report = makeReport(datasetId, result);
  expect(result.engine, report).toBe('gpu');
  expect(result.fallbackReason, report).toBeUndefined();
  expect(result.dispatchedContributors, report).toEqual(BINDING.gpuContributorNames);
  return {result, reference, report};
}

function expectParity(result: CatalogRunResult, reference: CatalogRunResult, report: string) {
  expect(result.summary, report).toEqual(reference.summary);
  expect(result.diagnostics, report).toEqual(reference.diagnostics);
  expect(getTable(result, 'points'), report).toEqual(getTable(reference, 'points'));
  const footprints = getTable(result, 'footprints');
  const expected = getTable(reference, 'footprints');
  expect(footprints.rowCount, report).toBe(expected.rowCount);
  expect(footprints.geometryColumn, report).toBe('geometry');
  expect(footprints.columnTypes, report).toEqual(expected.columnTypes);
  for (const column of [
    'cluster',
    'point_count',
    'core_point_count',
    'has_footprint',
    'hull_vertex_count'
  ]) {
    expect(footprints.columns[column], `${report}; ${column}`).toEqual(expected.columns[column]);
  }
  const geographic = result.parameters['coordinateSystem'] !== 'planar';
  for (let row = 0; row < footprints.rowCount; row++) {
    for (const column of ['area', 'perimeter', 'centroid_x', 'centroid_y']) {
      const actual = footprints.columns[column][row];
      const wanted = expected.columns[column][row];
      const evidence = `${report}; row=${row}; ${column}=${actual}; expected=${wanted}`;
      if (wanted === null) expect(actual, evidence).toBeNull();
      else {
        expect(Number.isFinite(Number(actual)), evidence).toBe(true);
        // Spherical measures upload f32 lon/lat (sub-metre quanta in these fixtures). Allow 0.5%
        // for 100m–1km hulls; planar measures allow 1e-4 relative plus 1e-3 absolute scene units.
        const tolerance = column.startsWith('centroid')
          ? geographic
            ? 1e-7
            : 1e-3
          : Math.max(
              geographic ? 0.1 : 1e-3,
              Math.abs(Number(wanted)) * (geographic ? 5e-3 : 1e-4)
            );
        expect(Math.abs(Number(actual) - Number(wanted)), evidence).toBeLessThanOrEqual(tolerance);
      }
    }
    if (!footprints.columns['has_footprint'][row]) {
      expect(footprints.columns['geometry'][row], report).toBeNull();
      continue;
    }
    const ring = JSON.parse(String(footprints.columns['geometry'][row]))
      .coordinates[0] as CatalogPosition[];
    const expectedRing = JSON.parse(String(expected.columns['geometry'][row]))
      .coordinates[0] as CatalogPosition[];
    expectCuratedClusterRingAdjacency(ring, expectedRing, `${report}; hull ${row}`);
    expectCuratedClusterFootprintMeasures(
      ring,
      {
        area: Number(footprints.columns['area'][row]),
        perimeter: Number(footprints.columns['perimeter'][row])
      },
      result.parameters['coordinateSystem'] as CatalogCoordinateSystem,
      `${report}; hull ${row}`
    );
  }
}

it('runs the bundled city-incidents template on WebGPU through mapping, matching every CPU output', async () => {
  const {points} = createKeplerCuratedAnalysisSample('city-incidents');
  const {result, reference, report} = await runMapped('gallery-incidents', points, 'EPSG:4326');
  expectParity(result, reference, report);
  expect(result.summary['clusterCount'], report).toBeGreaterThanOrEqual(3);
  expect(result.summary['footprintCount'], report).toBeGreaterThanOrEqual(3);
}, 120000);

it('maps an independent planar survey and matches the committed scikit-learn/Shapely fixture', async () => {
  const points = makePoints(DBSCAN_PLANAR.points);
  const {result, reference, report} = await runMapped('projected-survey', points, 'planar', {
    epsilon: DBSCAN_PLANAR.epsilon,
    minimumSamples: DBSCAN_PLANAR.minimumSamples
  });
  expectParity(result, reference, report);
  const labels = getTable(result, 'points').columns['cluster'].map(Number);
  expect(getTable(result, 'points').columns['is_core'], report).toEqual(DBSCAN_PLANAR.core);
  for (const cluster of DBSCAN_PLANAR.clusters) {
    const label = labels[cluster.minimumMember];
    const footprints = getTable(result, 'footprints');
    const row = footprints.columns['cluster'].indexOf(label);
    expect(footprints.columns['point_count'][row], report).toBe(cluster.pointCount);
    expect(
      Math.abs(Number(footprints.columns['area'][row]) / cluster.area - 1),
      report
    ).toBeLessThan(1e-4);
    expect(
      Math.abs(Number(footprints.columns['perimeter'][row]) / cluster.perimeter - 1),
      report
    ).toBeLessThan(1e-4);
  }
}, 120000);

it('matches the geographic sphere oracle after a real spherical GPU measurement pass', async () => {
  const points = makePoints(
    DBSCAN_GEOGRAPHIC.longitudes.map((longitude, index) => [
      longitude,
      DBSCAN_GEOGRAPHIC.latitudes[index]
    ])
  );
  const {result, reference, report} = await runMapped('geographic-survey', points, 'EPSG:4326', {
    epsilon: DBSCAN_GEOGRAPHIC.epsilon,
    minimumSamples: DBSCAN_GEOGRAPHIC.minimumSamples
  });
  expectParity(result, reference, report);
  const labels = getTable(result, 'points').columns['cluster'].map(Number);
  const footprints = getTable(result, 'footprints');
  for (const cluster of DBSCAN_GEOGRAPHIC.clusters) {
    const row = footprints.columns['cluster'].indexOf(labels[cluster.minimumMember]);
    expect(footprints.columns['point_count'][row], report).toBe(cluster.pointCount);
    expect(
      Math.abs(Number(footprints.columns['area'][row]) / cluster.area - 1),
      report
    ).toBeLessThan(5e-3);
    expect(
      Math.abs(Number(footprints.columns['perimeter'][row]) / cluster.perimeter - 1),
      report
    ).toBeLessThan(5e-3);
  }
}, 120000);

it('matches the existing cluster-and-outline recipe 400-point fixture and independent hull oracle', async () => {
  // Same seed, generator, parameters and near-radius separation as cluster-and-outline-recipe.spec.ts.
  const positions = separateNearEpsilonPairs(
    createClusteredPoints(3, {
      pointCount: 400,
      blobCount: 3,
      extent: 100,
      blobSigma: 3,
      noiseFraction: 0.1
    }),
    [4]
  );
  const oracle = clusterPointsOracle(positions, {
    bounds: [0, 0, 100, 100],
    epsilon: 4,
    minimumSamples: 4
  });
  const hulls = computeGroupConvexHullOracle(positions, oracle.labels, 16);
  const points = makePoints(
    Array.from({length: 400}, (_, index) => [positions[2 * index], positions[2 * index + 1]])
  );
  const {result, reference, report} = await runMapped('recipe-fixture', points, 'planar', {
    epsilon: 4,
    minimumSamples: 4
  });
  expectParity(result, reference, report);
  expect(getTable(result, 'points').columns['cluster'], report).toEqual(
    oracle.labels.map(label => (label === 0xffffffff ? -1 : label))
  );
  expect(getTable(result, 'points').columns['is_core'], report).toEqual(
    oracle.coreFlags.map(Boolean)
  );
  expect(result.summary['clusterCount'], report).toBe(oracle.clusterCount);
  const footprints = getTable(result, 'footprints');
  for (let cluster = 0; cluster < oracle.clusterCount; cluster++) {
    expect(footprints.columns['hull_vertex_count'][cluster], report).toBe(
      hulls.hulls[cluster].length
    );
  }
}, 120000);

for (const fixture of [
  {
    id: 'noise',
    points: [
      [0, 0],
      [10, 0],
      [20, 0]
    ],
    epsilon: 1,
    minimumSamples: 4
  },
  {id: 'single', points: [[5, 5]], epsilon: 1, minimumSamples: 1},
  {
    id: 'duplicates',
    points: [
      [5, 5],
      [5, 5],
      [5, 5]
    ],
    epsilon: 1,
    minimumSamples: 3
  },
  {
    id: 'degenerate',
    points: [
      [0, 0],
      [1, 0],
      [2, 0],
      [10, 0],
      [11, 0],
      [20, 0],
      [21, 0],
      [20.5, 1],
      [50, 50]
    ],
    epsilon: 1.5,
    minimumSamples: 2
  },
  {
    id: 'inclusive-radius',
    points: [
      [0, 0],
      [1, 0],
      [0, 1]
    ],
    epsilon: 1,
    minimumSamples: 3
  }
]) {
  it(`preserves ${fixture.id} counts, core flags and null footprint semantics`, async () => {
    const {result, reference, report} = await runMapped(
      fixture.id,
      makePoints(fixture.points),
      'planar',
      {
        epsilon: fixture.epsilon,
        minimumSamples: fixture.minimumSamples
      }
    );
    expectParity(result, reference, report);
  }, 120000);
}

it('unwraps a GPU hull across the dateline and retains spherical units', async () => {
  const points = makePoints([
    [179.9, 10],
    [-179.9, 10],
    [-179.9, 10.1],
    [179.9, 10.1]
  ]);
  const {result, reference, report} = await runMapped('dateline', points, 'EPSG:4326', {
    epsilon: 30000,
    minimumSamples: 1
  });
  expectParity(result, reference, report);
  const footprints = getTable(result, 'footprints');
  const ring = JSON.parse(String(footprints.columns['geometry'][0])).coordinates[0] as number[][];
  for (let vertex = 1; vertex < ring.length; vertex++)
    expect(Math.abs(ring[vertex][0] - ring[vertex - 1][0]), report).toBeLessThan(1);
  expect(Number(footprints.columns['area'][0]), report).toBeGreaterThan(2e8);
  expect(Number(footprints.columns['perimeter'][0]), report).toBeGreaterThan(6e4);
}, 120000);

it('preserves source identities and the dropped-feature diagnostic after GPU preparation', async () => {
  const points = makePoints([
    [0, 0],
    [1, 0],
    [0, 1]
  ]);
  points.features.splice(1, 0, {id: 'missing-position', geometry: null, properties: {}});
  const {result, reference, report} = await runMapped('missing-position', points, 'planar', {
    epsilon: 2,
    minimumSamples: 1
  });
  expectParity(result, reference, report);
  expect(getTable(result, 'points').columns['source_index'], report).toEqual([0, 2, 3]);
  expect(getTable(result, 'points').columns['source_id'], report).toEqual([
    'independent-0',
    'independent-1',
    'independent-2'
  ]);
  expect(
    result.diagnostics.find(diagnostic => diagnostic.code === 'feature-dropped')?.count,
    report
  ).toBe(1);
}, 120000);
