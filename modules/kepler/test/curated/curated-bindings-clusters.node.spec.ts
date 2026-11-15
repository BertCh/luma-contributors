// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {DBSCAN_GEOGRAPHIC, DBSCAN_PLANAR} from './curated-bindings-oracle-data';
import {
  getNumbers,
  getResultTable,
  makePoints,
  recordBindingMargin,
  runCuratedBinding
} from './curated-bindings-test-support';

// Oracles: /usr/bin/python3 .typecheck/b1-oracles.py. DBSCAN_PLANAR: scikit-learn DBSCAN on 220
// planar points (three blobs and uniform noise; no border point is within reach of two clusters,
// so the partition is unique) and Shapely convex hulls of each cluster. DBSCAN_GEOGRAPHIC: 88
// longitude and latitude points clustered in the azimuthal equidistant frame, with hull area and
// perimeter on a sphere of the library's radius (pyproj laea and Geod).

const ID = 'curated-clusters-and-footprints';

function getPoints(): CatalogFeatureCollection {
  return makePoints(
    DBSCAN_PLANAR.points.map(point => point[0]),
    DBSCAN_PLANAR.points.map(point => point[1])
  );
}

function runPlanar(points: CatalogFeatureCollection, epsilon: number, minimumPoints: number) {
  return runCuratedBinding(ID, {coordinateSystem: 'planar', epsilon, minimumPoints}, {points});
}

/** Groups point indices by label, in order of the smallest member. */
function getMembers(labels: readonly number[]): Map<number, number[]> {
  const members = new Map<number, number[]>();
  labels.forEach((label, index) => {
    if (label < 0) return;
    members.set(label, [...(members.get(label) ?? []), index]);
  });
  return members;
}

describe('curated-clusters-and-footprints against scikit-learn and Shapely (planar)', () => {
  const result = runPlanar(getPoints(), DBSCAN_PLANAR.epsilon, DBSCAN_PLANAR.minimumPoints);
  const pointTable = getResultTable(result, 'points');
  const footprints = getResultTable(result, 'footprints');

  it('gives the same partition, noise and core flags as scikit-learn, up to cluster numbering', () => {
    const labels = getNumbers(pointTable, 'cluster');
    // Labels agree up to a relabelling exactly when the pairs of points that share a label agree.
    const mapping = new Map<number, number>();
    labels.forEach((label, index) => {
      const expected = DBSCAN_PLANAR.labels[index];
      expect(label < 0).toBe(expected < 0);
      if (expected < 0) return;
      const mapped = mapping.get(expected);
      if (mapped === undefined) mapping.set(expected, label);
      else expect(label).toBe(mapped);
    });
    expect(new Set(mapping.values()).size).toBe(mapping.size);
    expect(pointTable.columns['is_core']).toEqual(DBSCAN_PLANAR.core);
    expect(result.summary['clusterCount']).toBe(DBSCAN_PLANAR.clusters.length);
    expect(result.summary['noiseCount']).toBe(
      DBSCAN_PLANAR.labels.filter(label => label < 0).length
    );
  });

  it('draws the Shapely convex hull of every cluster: area and perimeter to 1e-9, same vertices, same count', () => {
    expect(footprints.rowCount).toBe(DBSCAN_PLANAR.clusters.length);
    const labels = getNumbers(pointTable, 'cluster');
    const members = getMembers(labels);
    const ours = [...members.entries()].map(([cluster, indices]) => ({
      cluster,
      minimum: Math.min(...indices)
    }));
    for (const expected of DBSCAN_PLANAR.clusters) {
      const cluster = ours.find(item => item.minimum === expected.minimumMember)!.cluster;
      const row = (footprints.columns['cluster'] as number[]).indexOf(cluster);
      expect(footprints.columns['point_count'][row]).toBe(expected.pointCount);
      const area = footprints.columns['area'][row] as number;
      const perimeter = footprints.columns['perimeter'][row] as number;
      const areaError = Math.abs(area - expected.area) / expected.area;
      const perimeterError = Math.abs(perimeter - expected.perimeter) / expected.perimeter;
      recordBindingMargin(`footprint area relative error cluster ${cluster}`, areaError);
      recordBindingMargin(`footprint perimeter relative error cluster ${cluster}`, perimeterError);
      expect(areaError).toBeLessThan(1e-9);
      expect(perimeterError).toBeLessThan(1e-9);
      expect(footprints.columns['hull_vertex_count'][row]).toBe(expected.vertices.length);
      const ring = (
        JSON.parse(String(footprints.columns['geometry'][row])).coordinates[0] as number[][]
      ).slice(0, -1);
      for (const vertex of expected.vertices)
        expect(
          ring.some(
            candidate => Math.hypot(candidate[0] - vertex[0], candidate[1] - vertex[1]) < 1e-9
          ),
          `vertex ${vertex}`
        ).toBe(true);
      expect(footprints.columns['has_footprint'][row]).toBe(true);
    }
  });

  it('reports noise in the points table and a diagnostic, and hulls nothing for it', () => {
    const noise = DBSCAN_PLANAR.labels.filter(label => label < 0).length;
    expect(noise).toBeGreaterThan(0);
    expect(result.diagnostics.find(item => item.code === 'noise-points')?.count).toBe(noise);
    const counted = (footprints.columns['point_count'] as number[]).reduce(
      (sum, count) => sum + count,
      0
    );
    expect(counted + noise).toBe(DBSCAN_PLANAR.points.length);
  });

  it('is deterministic', () => {
    expect(runPlanar(getPoints(), DBSCAN_PLANAR.epsilon, DBSCAN_PLANAR.minimumPoints)).toEqual(
      result
    );
  });
});

describe('curated-clusters-and-footprints on longitude and latitude', () => {
  const points = makePoints(DBSCAN_GEOGRAPHIC.longitudes, DBSCAN_GEOGRAPHIC.latitudes);
  const result = runCuratedBinding(
    ID,
    {
      coordinateSystem: 'EPSG:4326',
      epsilon: DBSCAN_GEOGRAPHIC.epsilon,
      minimumPoints: DBSCAN_GEOGRAPHIC.minimumPoints
    },
    {points}
  );

  it('finds the clusters of the sphere oracle and measures hulls in square metres and metres', () => {
    const footprints = getResultTable(result, 'footprints');
    const labels = getNumbers(getResultTable(result, 'points'), 'cluster');
    expect(footprints.rowCount).toBe(DBSCAN_GEOGRAPHIC.clusters.length);
    const members = getMembers(labels);
    for (const expected of DBSCAN_GEOGRAPHIC.clusters) {
      const cluster = [...members.entries()].find(
        ([, indices]) => Math.min(...indices) === expected.minimumMember
      )![0];
      const row = (footprints.columns['cluster'] as number[]).indexOf(cluster);
      expect(footprints.columns['point_count'][row]).toBe(expected.pointCount);
      const areaError =
        Math.abs((footprints.columns['area'][row] as number) - expected.area) / expected.area;
      const perimeterError =
        Math.abs((footprints.columns['perimeter'][row] as number) - expected.perimeter) /
        expected.perimeter;
      recordBindingMargin(`geographic footprint area relative error cluster ${cluster}`, areaError);
      recordBindingMargin(
        `geographic footprint perimeter relative error cluster ${cluster}`,
        perimeterError
      );
      // The hull vertices are chosen in the equidistant frame and the area measured on the sphere
      // (Chamberlain-Duquette); the oracle hulls in an equal-area frame. Over 1 km they agree to 1e-4.
      expect(areaError).toBeLessThan(1e-4);
      expect(perimeterError).toBeLessThan(1e-4);
    }
    expect(Math.abs(Number(footprints.columns['area'][0]))).toBeGreaterThan(1000);
  });
});

describe('curated-clusters-and-footprints declared behaviour', () => {
  it('lists a collinear cluster and a two-point cluster without a polygon, keeps the triangle, and reports noise', () => {
    const points = makePoints([0, 1, 2, 10, 11, 20, 21, 20.5, 50], [0, 0, 0, 0, 0, 0, 0, 1, 50]);
    const result = runPlanar(points, 1.5, 2);
    const footprints = getResultTable(result, 'footprints');
    expect(footprints.rowCount).toBe(3);
    expect(footprints.columns['point_count']).toEqual([3, 2, 3]);
    expect(footprints.columns['has_footprint']).toEqual([false, false, true]);
    expect(footprints.columns['geometry'].slice(0, 2)).toEqual([null, null]);
    expect(footprints.columns['area'].slice(0, 2)).toEqual([null, null]);
    expect(footprints.columns['perimeter'].slice(0, 2)).toEqual([null, null]);
    expect(Math.abs(Number(footprints.columns['area'][2]) - 0.5)).toBeLessThan(1e-12);
    expect(
      Math.abs(Number(footprints.columns['perimeter'][2]) - (1 + 2 * Math.sqrt(1.25)))
    ).toBeLessThan(1e-12);
    expect(result.diagnostics.find(item => item.code === 'degenerate-footprint')?.count).toBe(2);
    expect(result.diagnostics.find(item => item.code === 'noise-points')?.count).toBe(1);
    expect(result.summary['footprintCount']).toBe(1);
    expect(result.summary['degenerateClusterCount']).toBe(2);
    // The centroid of an unhulled cluster is still reported.
    expect(footprints.columns['centroid_x'][0]).toBeCloseTo(1, 12);
    expect(footprints.columns['centroid_x'][1]).toBeCloseTo(10.5, 12);
  });

  it('puts every point in noise, and lists no footprint, when no point has enough neighbours', () => {
    const result = runPlanar(makePoints([0, 10, 20, 30], [0, 0, 0, 0]), 1, 3);
    expect(getResultTable(result, 'footprints').rowCount).toBe(0);
    expect(new Set(getNumbers(getResultTable(result, 'points'), 'cluster'))).toEqual(new Set([-1]));
    expect(result.summary['noiseCount']).toBe(4);
  });

  it('skips features without a point position, counted, and throws a named error when none is left', () => {
    const points = makePoints([0, 0.5, 1, 100], [0, 0, 0, 100]);
    points.features.push({id: 'none', geometry: null, properties: {}});
    const result = runPlanar(points, 2, 3);
    expect(result.diagnostics.find(item => item.code === 'feature-dropped')?.count).toBe(1);
    expect(result.summary['pointCount']).toBe(4);
    expect(() =>
      runPlanar(
        {type: 'FeatureCollection', features: [{id: 'n', geometry: null, properties: {}}]},
        2,
        3
      )
    ).toThrow(/curated-clusters-and-footprints: no feature of the input has a point position/);
  });

  it('treats one repeated point as one cluster of a single location without a polygon', () => {
    const result = runPlanar(makePoints([5, 5, 5, 5], [5, 5, 5, 5]), 1, 3);
    const footprints = getResultTable(result, 'footprints');
    expect(footprints.rowCount).toBe(1);
    expect(footprints.columns['point_count'][0]).toBe(4);
    expect(footprints.columns['has_footprint'][0]).toBe(false);
    expect(footprints.columns['hull_vertex_count'][0]).toBe(1);
  });

  it('rejects polygons and parameters outside their range', () => {
    expect(() =>
      runCuratedBinding(
        ID,
        {coordinateSystem: 'planar'},
        {
          points: {
            type: 'FeatureCollection',
            features: [
              {
                id: 'a',
                geometry: {
                  type: 'Polygon',
                  coordinates: [
                    [
                      [0, 0],
                      [1, 0],
                      [1, 1],
                      [0, 0]
                    ]
                  ]
                },
                properties: {}
              }
            ]
          }
        }
      )
    ).toThrow(/geometry kind not accepted/);
    expect(() => runPlanar(makePoints([0], [0]), 0, 3)).toThrow(/out of range/);
    expect(() => runPlanar(makePoints([0], [0]), 1, 0)).toThrow(/out of range/);
  });
});

it('keeps local hull area and perimeter accurate at large planar offsets (Shapely)', () => {
  // /usr/bin/python3: MultiPoint(points + offset).convex_hull.area/.length, Shapely.
  const coordinates = [
    [-3, 1],
    [-2, -4],
    [1, -3],
    [5, 1],
    [2, 4],
    [-1, 3],
    [0, 0],
    [2, 1]
  ];
  for (const offset of [0, 1e6, 1e9, 1e12]) {
    const points = makePoints(
      coordinates.map(point => point[0] + offset),
      coordinates.map(point => point[1] + offset)
    );
    const footprints = getResultTable(runPlanar(points, 20, 1), 'footprints');
    expect(footprints.columns['area']).toEqual([38]);
    expect(Number(footprints.columns['perimeter'][0])).toBeCloseTo(24.151496895287398, 12);
    expect(footprints.columns['has_footprint']).toEqual([true]);
  }
});

it('measures and represents a small dateline hull without a globe-spanning edge (pyproj)', () => {
  // /usr/bin/python3: Geod(a=6371008.8,b=6371008.8).polygon_area_perimeter(
  // [179.9,-179.9,-179.9,179.9],[10,10,10.1,10.1]) -> (243492701.43635178,66034.55280921928).
  const points = makePoints([179.9, -179.9, -179.9, 179.9], [10, 10, 10.1, 10.1]);
  const footprints = getResultTable(
    runCuratedBinding(
      ID,
      {coordinateSystem: 'EPSG:4326', epsilon: 30000, minimumPoints: 1},
      {points}
    ),
    'footprints'
  );
  // Chamberlain-Duquette integrates latitude edges; pyproj uses geodesic edges. At this extent
  // the relative area difference is under 1e-6, while perimeter uses the same great-circle metric.
  expect(Math.abs(Number(footprints.columns['area'][0]) / 243492701.43635178 - 1)).toBeLessThan(
    1e-6
  );
  expect(Number(footprints.columns['perimeter'][0])).toBeCloseTo(66034.55280921928, 6);
  const ring = JSON.parse(String(footprints.columns['geometry'][0])).coordinates[0] as number[][];
  for (let index = 1; index < ring.length; index++)
    expect(Math.abs(ring[index][0] - ring[index - 1][0])).toBeLessThan(1);
  expect(ring.at(-1)).toEqual(ring[0]);
});
