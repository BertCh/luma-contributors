// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CatalogGpuContext,
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
import {
  MAXIMUM_CURATED_CLUSTER_POINTS,
  runCuratedClustersGpu,
  type CuratedClustersGpuResult
} from './curated-clusters-and-footprints-gpu';

const BINDING_ID = 'curated-clusters-and-footprints';

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

function readClustersContext(inputs: Inputs, values: Values) {
  try {
    return readClusterSamples(
      inputs['points'],
      values['coordinateSystem'] as CatalogCoordinateSystem
    );
  } catch {
    throw makeCuratedError(BINDING_ID, 'no feature of the input has a point position');
  }
}

function publishClustersAndFootprints(
  inputs: Inputs,
  values: Values,
  context: ReturnType<typeof readClusterSamples>,
  gpu?: CuratedClustersGpuResult
): CatalogRunOutput {
  const points = inputs['points'];
  const coordinateSystem = values['coordinateSystem'] as CatalogCoordinateSystem;
  const epsilon = values['epsilon'] as number;
  const minimumPoints = values['minimumPoints'] as number;
  const {frame, sample, diagnostics} = context;
  // The CPU engine remains the binary64 reference. The GPU engine never calls the CPU DBSCAN.
  const {labels, coreFlags} = gpu ?? computeDbscanLabels(sample, epsilon, minimumPoints);

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
      if (!gpu) {
        sumX += sample.x[index];
        sumY += sample.y[index];
        framePoints.push([sample.x[index], sample.y[index]]);
      }
      if (coreFlags[index] === 1) coreCount++;
    }
    const [centroidX, centroidY] = frame.inverse(
      gpu ? gpu.meanCenters[2 * cluster] : sumX / indices.length,
      gpu ? gpu.meanCenters[2 * cluster + 1] : sumY / indices.length
    );
    // The GPU hull's source indices preserve original coordinates for publication. No CPU hull
    // runs on this path; the GPU also supplies member counts, mean centers, area and perimeter.
    const hull: PlanarPoint[] = gpu
      ? Array.from({length: gpu.hullCounts[cluster]}, (_, vertex) => {
          const sourceIndex = gpu.hullVertexIndices[gpu.hullOffsets[cluster] + vertex];
          return [sample.x[sourceIndex], sample.y[sourceIndex]];
        })
      : getConvexHull(framePoints);
    const pointCount = gpu ? gpu.counts[cluster] : indices.length;
    // Fewer than three distinct points, or collinear points, enclose no area.
    if (hull.length < 3) {
      degenerateCount++;
      appendCatalogRow(footprints, {
        cluster,
        point_count: pointCount,
        core_point_count: coreCount,
        has_footprint: false,
        hull_vertex_count: hull.length,
        centroid_x: centroidX,
        centroid_y: centroidY
      });
      continue;
    }
    // The shared area routine translates to the first vertex itself. Keep frame coordinates
    // for planar measurements to avoid another rounding step through frame.inverse.
    const frameRing = [...hull, hull[0]];
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
    const measurementRing = coordinateSystem === 'planar' ? frameRing : ring;
    appendCatalogRow(footprints, {
      cluster,
      point_count: pointCount,
      core_point_count: coreCount,
      has_footprint: true,
      hull_vertex_count: hull.length,
      area: gpu
        ? gpu.areas[cluster]
        : Math.abs(measureRingSignedArea(measurementRing, coordinateSystem)),
      perimeter: gpu
        ? gpu.perimeters[cluster]
        : measurePathLength(measurementRing, coordinateSystem),
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
    ...(gpu ? {dispatchedContributors: gpu.dispatchedContributors} : {}),
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

async function runClustersAndFootprintsGpu(
  gpuContext: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const context = readClustersContext(inputs, values);
  const gpu = await runCuratedClustersGpu(
    gpuContext,
    context.sample,
    context.frame,
    values['epsilon'] as number,
    values['minimumPoints'] as number
  );
  return publishClustersAndFootprints(inputs, values, context, gpu);
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
  gpuContributorNames: [
    'GPUSpatialClustering',
    'GPUGroupGeometry',
    'GPUGroupConvexHull',
    'GPUGeometryMeasures'
  ],
  engine: 'gpu-or-cpu',
  inputs: [
    makeInputSpec(
      'points',
      'Points',
      ['point'],
      MAXIMUM_CURATED_CLUSTER_POINTS,
      MAXIMUM_CURATED_CLUSTER_POINTS
    )
  ],
  parameters: PARAMETERS,
  outputs: [
    makeOutputSpec(
      'points',
      'Clustered points',
      'point',
      ['point_index', 'source_index', 'source_id'],
      MAXIMUM_CURATED_CLUSTER_POINTS
    ),
    makeOutputSpec(
      'footprints',
      'Cluster footprints',
      'polygon',
      ['cluster'],
      MAXIMUM_CURATED_CLUSTER_POINTS
    )
  ],
  cleanup:
    'Each GPU run owns its graph, compiled graphs and storage/readback buffers and releases them on success or failure. Results contain plain arrays; the caller owns the device and publication.',
  declaredBehavior: [
    'With a device, addClusterAndOutlineRecipe dispatches clustering, group summaries, convex hulls and planar measurements. Geographic hulls are re-measured by GPUGeometryMeasures on the same sphere as the CPU reference. Every overflow or invalid measurement status rejects the GPU result; the catalog reports the failure and falls back to the CPU reference.',
    'GPU clustering and planar measurements use float32 local-frame coordinates; geographic measurements use float32 longitude and latitude. The hull predicate snaps to a global lattice about 2e-9 of the extent. At that resolution memberships and hull vertices can differ from the binary64 CPU reference, especially near the search distance, with near-collinear points, or with tiny groups far from the frame origin. Published geometry retains original source coordinates, while GPU area, perimeter and mean centers reflect float32 precision.',
    'Neighbourhoods include the point itself and use distance at most the search distance, in metres in a local frame around the data for EPSG:4326 and in coordinate units for planar. The requested search distance is passed unchanged to the shared DBSCAN routine, which compensates measured binary64 frame-translation errors before an inclusive squared-distance comparison. No relative radius inflation is applied. For longitude and latitude data the frame distance equals the great-circle distance only from the frame origin and differs slightly elsewhere (more with distance from the data centre), so a pair very close to the search distance can fall either side of it.',
    'Clusters are numbered by their smallest core point in input order and noise is cluster -1. A border point (a non-core point within reach of core points) reachable from two clusters goes to the cluster with the smaller number, that is, the one whose first core point comes first in the input; reordering the input can therefore move such a point, and the cluster numbers, but never changes which core points group together or which points are noise.',
    'Each footprint is the convex hull of all members of a cluster, core and border points. Its perimeter is in metres (EPSG:4326, great-circle) or coordinate units (planar). Its area is in square metres on a sphere of radius 6,371,008.8 m (Chamberlain-Duquette) for EPSG:4326, which is smaller than the WGS84 ellipsoid area by about 0.12 percent at 40 degrees latitude and 0.56 percent at 60 degrees, and in squared coordinate units for planar. Planar measurements use local-frame hull coordinates and the shared area routine translates internally to preserve precision at large offsets; geographic hull rings unwrap consecutive longitudes across the antimeridian, so output longitudes may extend beyond 180 degrees.',
    'Noise points are reported in the points table and counted in a diagnostic; they are not part of any hull.',
    'A cluster with fewer than three distinct points, or only collinear points, has no polygon: its row in the footprints table has has_footprint false, no geometry, area or perimeter, and a diagnostic counts such clusters.',
    'Features without a point are skipped and counted; no point at all throws; at most 6000 points.'
  ],
  limitation:
    'GPU and CPU reference engines are available. Float32 rounding can affect near-radius membership and very small geographic measurements. Footprints are convex hulls: the concave (alpha) outline of addClusterAndOutlineRecipe is not offered, so a crescent-shaped cluster is outlined with its gap filled. The hull vertices are chosen in the local equidistant frame, and the area is measured on the sphere.',
  run: (inputs, values) =>
    publishClustersAndFootprints(inputs, values, readClustersContext(inputs, values)),
  runGpu: runClustersAndFootprintsGpu
};
