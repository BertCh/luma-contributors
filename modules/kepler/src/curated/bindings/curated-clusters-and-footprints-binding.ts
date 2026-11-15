// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CatalogOperationBinding,
  CatalogParameterSpec,
  CatalogRunOutput
} from '../../analysis/catalog-binding-types';
import {COORDINATE_SYSTEM_PARAMETER} from '../../analysis/catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  makeCatalogGeometryText,
  makeCatalogTable,
  measurePathLength,
  measureRingSignedArea,
  type CatalogCoordinateSystem
} from '../../analysis/catalog-geojson';
import {
  PLATFORM3_CPU_CLEANUP,
  makeInputSpec,
  makeNumberParameter,
  makeOutputSpec,
  type Inputs,
  type Values
} from '../../analysis/catalog-platform3-b-common';
import {
  computeDbscanLabels,
  makeClusterOutput,
  readClusterSamples
} from '../../analysis/catalog-platform3-b-clusters';
import {getConvexHull, type PlanarPoint} from '../../analysis/catalog-platform3-a-shared';
import {makeCuratedError} from './curated-binding-utils';

const BINDING_ID = 'curated-clusters-and-footprints';

/**
 * Points the CPU reference accepts: DBSCAN compares every pair, and the catalog's pair-work bound
 * of 4e7 allows 6324 points, so 6000 leaves room.
 */
const MAXIMUM_POINTS = 6000;

/**
 * Relative slack added to the search distance. The local frame subtracts the minimum corner of the
 * data from every coordinate, which can turn a distance of exactly 1 into 1.0000000000000002, so a
 * pair exactly at the search distance would fall outside it. The error of that subtraction is about
 * 1e-16 times the coordinate magnitude, so 1e-9 of the search distance covers coordinates up to
 * about 1e7 times the search distance; beyond that a pair exactly at the distance can still fall
 * either side.
 */
const SEARCH_DISTANCE_RELATIVE_SLACK = 1e-9;

const PARAMETERS: readonly CatalogParameterSpec[] = [
  makeNumberParameter('epsilon', 'Search distance', 200, 1e-9, 1e9, 'crs'),
  makeNumberParameter(
    'minimumPoints',
    'Minimum points (including the point)',
    4,
    1,
    1000,
    undefined,
    'integer'
  ),
  COORDINATE_SYSTEM_PARAMETER
];

function runClustersAndFootprints(inputs: Inputs, values: Values): CatalogRunOutput {
  const points = inputs['points'];
  const coordinateSystem = values['coordinateSystem'] as CatalogCoordinateSystem;
  const epsilon = values['epsilon'] as number;
  const minimumPoints = values['minimumPoints'] as number;
  let context: ReturnType<typeof readClusterSamples>;
  try {
    context = readClusterSamples(points, coordinateSystem);
  } catch {
    throw makeCuratedError(BINDING_ID, 'no feature of the input has a point position');
  }
  const {frame, sample, diagnostics} = context;
  const {labels, coreFlags} = computeDbscanLabels(
    sample,
    epsilon * (1 + SEARCH_DISTANCE_RELATIVE_SLACK),
    minimumPoints
  );

  // Table 1: the labelled points, as the catalog's own DBSCAN publishes them (cluster, is_core).
  const clustered = makeClusterOutput(points, frame, sample, labels, coreFlags, diagnostics);
  const pointTable = clustered.tables[0];

  // Table 2: one row per cluster, with the convex hull of its members (core and border points) when
  // the members span an area. The hull is taken in the local frame, where distances are metres.
  const members = new Map<number, number[]>();
  let noiseCount = 0;
  for (let index = 0; index < sample.count; index++) {
    if (labels[index] < 0) {
      noiseCount++;
      continue;
    }
    const list = members.get(labels[index]) ?? [];
    list.push(index);
    members.set(labels[index], list);
  }
  const footprints = makeCatalogTable(
    'footprints',
    'Cluster footprints',
    {
      cluster: 'int32',
      point_count: 'int32',
      core_point_count: 'int32',
      has_footprint: 'boolean',
      hull_vertex_count: 'int32',
      area: 'float64',
      perimeter: 'float64',
      centroid_x: 'float64',
      centroid_y: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  let degenerateCount = 0;
  for (const [cluster, indices] of [...members.entries()].sort((a, b) => a[0] - b[0])) {
    let sumX = 0;
    let sumY = 0;
    let coreCount = 0;
    const framePoints: PlanarPoint[] = [];
    for (const index of indices) {
      sumX += sample.x[index];
      sumY += sample.y[index];
      if (coreFlags[index] === 1) coreCount++;
      framePoints.push([sample.x[index], sample.y[index]]);
    }
    const [centroidX, centroidY] = frame.inverse(sumX / indices.length, sumY / indices.length);
    const hull = getConvexHull(framePoints);
    // Fewer than three distinct points, or collinear points, enclose no area.
    if (hull.length < 3) {
      degenerateCount++;
      appendCatalogRow(footprints, {
        cluster,
        point_count: indices.length,
        core_point_count: coreCount,
        has_footprint: false,
        hull_vertex_count: hull.length,
        centroid_x: centroidX,
        centroid_y: centroidY
      });
      continue;
    }
    // Keep the shoelace products near zero even when the cluster is far from the frame origin.
    const localRing: PlanarPoint[] = hull.map(([x, y]) => [x - hull[0][0], y - hull[0][1]]);
    localRing.push(localRing[0]);
    const ring = hull.map(([x, y]) => frame.inverse(x, y));
    ring.push(ring[0]);
    // The inverse wraps longitudes; unwrap consecutive edges for the spherical surface integral
    // and output polygon so a dateline cluster keeps its short edges instead of spanning the globe.
    if (coordinateSystem !== 'planar') {
      for (let index = 1; index < ring.length; index++) {
        const difference = ring[index][0] - ring[index - 1][0];
        ring[index][0] -= 360 * Math.round(difference / 360);
      }
    }
    const measurementRing = coordinateSystem === 'planar' ? localRing : ring;
    appendCatalogRow(footprints, {
      cluster,
      point_count: indices.length,
      core_point_count: coreCount,
      has_footprint: true,
      hull_vertex_count: hull.length,
      area: Math.abs(measureRingSignedArea(measurementRing, coordinateSystem)),
      perimeter: measurePathLength(measurementRing, coordinateSystem),
      centroid_x: centroidX,
      centroid_y: centroidY,
      geometry: makeCatalogGeometryText('Polygon', [ring])
    });
  }
  if (degenerateCount)
    addCatalogDiagnostic(
      diagnostics,
      'degenerate-footprint',
      'warning',
      'Clusters with fewer than three distinct points, or only collinear points, enclose no area: they are listed with has_footprint false and no geometry, area or perimeter.',
      degenerateCount
    );
  if (noiseCount)
    addCatalogDiagnostic(
      diagnostics,
      'noise-points',
      'info',
      'Points that belong to no cluster are labelled -1 in the points table and are not part of any footprint.',
      noiseCount
    );
  return {
    tables: [pointTable, footprints],
    diagnostics,
    summary: {
      pointCount: sample.count,
      clusterCount: footprints.rowCount,
      footprintCount: footprints.rowCount - degenerateCount,
      degenerateClusterCount: degenerateCount,
      noiseCount,
      epsilon,
      minimumPoints
    }
  };
}

/**
 * Clusters and footprints: DBSCAN groups, and a convex hull polygon with the point count, area and
 * perimeter of each group.
 */
export const CURATED_CLUSTERS_AND_FOOTPRINTS_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Clusters and footprints',
  domain: 'clustering',
  summary:
    'Finds dense groups of points with DBSCAN, labels every point with its group or as noise, and draws the convex hull of each group with its point count, area and perimeter.',
  capabilityIds: ['group-and-regional-analysis'],
  contributorNames: [
    'GPUSpatialClustering',
    'GPUGroupGeometry',
    'GPUGroupConvexHull',
    'GPUGeometryMeasures'
  ],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('points', 'Points', ['point'], MAXIMUM_POINTS, MAXIMUM_POINTS)],
  parameters: PARAMETERS,
  outputs: [
    makeOutputSpec(
      'points',
      'Clustered points',
      'point',
      ['point_index', 'source_index', 'source_id'],
      MAXIMUM_POINTS
    ),
    makeOutputSpec('footprints', 'Cluster footprints', 'polygon', ['cluster'], MAXIMUM_POINTS)
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'Neighbourhoods include the point itself and use distance at most the search distance, in metres in a local frame around the data for EPSG:4326 and in coordinate units for planar. The frame subtracts the data minimum from every coordinate, which can make a pair exactly at the search distance 1e-16 times the coordinate magnitude longer, so the comparison allows a relative slack of 1e-9 of the search distance: a pair exactly at the distance counts as within it (as scikit-learn counts it on the raw coordinates) unless the coordinates exceed about 1e7 times the search distance, where it can still fall either side; for longitude and latitude data the frame distance equals the great-circle distance only from the frame origin and differs slightly elsewhere (more with distance from the data centre), so a pair very close to the search distance can fall either side of it.',
    'Clusters are numbered by their smallest core point in input order and noise is cluster -1. A border point (a non-core point within reach of core points) reachable from two clusters goes to the cluster with the smaller number, that is, the one whose first core point comes first in the input; reordering the input can therefore move such a point, and the cluster numbers, but never changes which core points group together or which points are noise.',
    'Each footprint is the convex hull of all members of a cluster, core and border points. Its perimeter is in metres (EPSG:4326, great-circle) or coordinate units (planar). Its area is in square metres on a sphere of radius 6,371,008.8 m (Chamberlain-Duquette) for EPSG:4326, which is smaller than the WGS84 ellipsoid area by about 0.12 percent at 40 degrees latitude and 0.56 percent at 60 degrees, and in squared coordinate units for planar. Planar measurements use translated local-frame hull coordinates to preserve precision at large offsets; geographic hull rings unwrap consecutive longitudes across the antimeridian, so output longitudes may extend beyond 180 degrees.',
    'Noise points are reported in the points table and counted in a diagnostic; they are not part of any hull.',
    'A cluster with fewer than three distinct points, or only collinear points, has no polygon: its row in the footprints table has has_footprint false, no geometry, area or perimeter, and a diagnostic counts such clusters.',
    'Features without a point are skipped and counted; no point at all throws; at most 6000 points.'
  ],
  limitation:
    'CPU reference engine only; the GPU path (GPUSpatialClustering, GPUGroupConvexHull) is future work. Footprints are convex hulls: the concave (alpha) outline of addClusterAndOutlineRecipe is not offered, so a crescent-shaped cluster is outlined with its gap filled. The hull vertices are chosen in the local equidistant frame, and the area is measured on the sphere.',
  run: (inputs, values) => runClustersAndFootprints(inputs, values)
};
