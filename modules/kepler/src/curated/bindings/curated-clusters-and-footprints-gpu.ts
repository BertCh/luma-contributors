// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '@luma-contributors/analysis/analysis-core';
import {
  addClusterAndOutlineRecipe,
  getGPUGeographicDistributionParameterValues,
  getGPUSpatialClusteringParameterValues,
  GPUGeometryMeasures,
  GPUGroupConvexHull,
  GPUGroupGeometry,
  GPUSpatialClustering,
  GPU_SPATIAL_CLUSTERING_NOISE
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogGpuContext} from '../../analysis/catalog-binding-types';
import {EARTH_RADIUS_METERS} from '../../analysis/catalog-geojson';
import {withCatalogGpuSession} from '../../analysis/catalog-gpu-session';
import type {LocalFrame} from '../../analysis/catalog-local-frame';
import type {Sample} from '../../analysis/catalog-platform3-b-common';

const BINDING_ID = 'curated-clusters-and-footprints';
/** Shared CPU/GPU materialization cap; bounds table sizes and worst-case one-cluster hull work. */
export const MAXIMUM_CURATED_CLUSTER_POINTS = 6000;

// Constructor identity survives bundler name mangling; glue producers are deliberately absent.
const CONTRIBUTOR_NAMES = new Map<object, string>([
  [GPUSpatialClustering, 'GPUSpatialClustering'],
  [GPUGroupGeometry, 'GPUGroupGeometry'],
  [GPUGroupConvexHull, 'GPUGroupConvexHull'],
  [GPUGeometryMeasures, 'GPUGeometryMeasures']
]);

export type CuratedClustersGpuResult = {
  labels: Int32Array;
  coreFlags: Uint32Array;
  clusterCount: number;
  counts: Uint32Array;
  meanCenters: Float32Array;
  hullVertexIndices: Uint32Array;
  hullOffsets: Uint32Array;
  hullCounts: Uint32Array;
  areas: Float32Array;
  perimeters: Float32Array;
  dispatchedContributors: string[];
};

/** Packs the existing catalog local frame once; no pair search or hull computation on the CPU. */
export function prepareCuratedClustersGpu(sample: Sample, epsilon: number, minimumPoints: number) {
  validateCapacity(BINDING_ID, 'pointCount', sample.count, 1, MAXIMUM_CURATED_CLUSTER_POINTS);
  validateCapacity(BINDING_ID, 'minimumPoints', minimumPoints, 1, 1000);
  validateFiniteNumber(BINDING_ID, 'epsilon', epsilon, Number.MIN_VALUE, 1e9);
  const positions = new Float32Array(sample.count * 2);
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let index = 0; index < sample.count; index++) {
    positions[2 * index] = sample.x[index];
    positions[2 * index + 1] = sample.y[index];
    minimumX = Math.min(minimumX, positions[2 * index]);
    minimumY = Math.min(minimumY, positions[2 * index + 1]);
    maximumX = Math.max(maximumX, positions[2 * index]);
    maximumY = Math.max(maximumY, positions[2 * index + 1]);
  }
  validateRequirement(
    BINDING_ID,
    [minimumX, minimumY, maximumX, maximumY].every(Number.isFinite),
    'positions must fit the finite float32 coordinate range'
  );
  // Derive bounds from the uploaded values, including a nonzero span for repeated points.
  const bounds = [
    minimumX,
    minimumY,
    Math.max(maximumX, minimumX + 1),
    Math.max(maximumY, minimumY + 1)
  ] as const;
  const side = Math.max(1, Math.min(256, Math.ceil(Math.sqrt(sample.count))));
  return {
    positions,
    clusteringParameters: getGPUSpatialClusteringParameterValues({bounds, epsilon, minimumPoints}),
    gridSize: [side, side] as const,
    maximumClusterCount: sample.count,
    maximumVerticesPerHull: Math.max(3, sample.count),
    hullCapacity: sample.count
  };
}

/** Dispatches the public convex recipe, reads bounded outputs and refuses incomplete results. */
export async function runCuratedClustersGpu(
  context: CatalogGpuContext,
  sample: Sample,
  frame: LocalFrame,
  epsilon: number,
  minimumPoints: number
): Promise<CuratedClustersGpuResult> {
  const prepared = prepareCuratedClustersGpu(sample, epsilon, minimumPoints);
  const result = await withCatalogGpuSession(context.device, BINDING_ID, async session => {
    const labels = session.output('labels', 'uint32', sample.count);
    const coreFlags = session.output('core-flags', 'uint32', sample.count);
    const clusterCount = session.output('cluster-count', 'uint32', 1);
    const counts = session.output('counts', 'uint32', sample.count);
    const meanCenters = session.output('mean-centers', 'float32x2', sample.count);
    const hullVertexIndices = session.output('hull-indices', 'uint32', sample.count);
    const hullPositions = session.output('hull-positions', 'float32x2', sample.count);
    const hullOffsets = session.output('hull-offsets', 'uint32', sample.count + 1);
    const hullCounts = session.output('hull-counts', 'uint32', sample.count);
    const areas = session.output('areas', 'float32', sample.count);
    const perimeters = session.output('perimeters', 'float32', sample.count);
    const clusterOverflow = session.output('cluster-overflow', 'uint32', 1);
    const hullOverflow = session.output('hull-overflow', 'uint32', 1);
    const invalidCount = session.output('invalid-count', 'uint32', 1);
    const recipe = addClusterAndOutlineRecipe(session.graph, {
      id: BINDING_ID,
      spatialContext: frame.gpuFrame?.spatialContext,
      points: {
        kind: 'points',
        positions: session.input('positions', prepared.positions, 'float32x2')
      },
      clusteringParameters: session.input(
        'clustering-parameters',
        prepared.clusteringParameters,
        'float32'
      ),
      geometryParameters: session.input(
        'geometry-parameters',
        getGPUGeographicDistributionParameterValues(),
        'float32'
      ),
      gridSize: prepared.gridSize,
      maximumClusterCount: prepared.maximumClusterCount,
      maximumVerticesPerHull: prepared.maximumVerticesPerHull,
      hullCapacity: prepared.hullCapacity,
      outputs: {
        labels: labels.view,
        coreFlags: coreFlags.view,
        clusterCount: clusterCount.view,
        hullPositions: hullPositions.view,
        hullOffsets: hullOffsets.view,
        areas: areas.view,
        perimeters: perimeters.view,
        status: {
          clusterOverflow: clusterOverflow.view,
          hullOverflow: hullOverflow.view,
          invalidCount: invalidCount.view
        }
      },
      scratch: {
        counts: counts.view,
        meanCenters: meanCenters.view,
        hullVertexIndices: hullVertexIndices.view,
        hullCounts: hullCounts.view
      }
    });
    session.run();
    const status = await Promise.all([
      session.readScalar(clusterOverflow),
      session.readScalar(hullOverflow),
      session.readScalar(invalidCount)
    ]);
    validateRequirement(
      BINDING_ID,
      status.every(value => value === 0),
      `recipe clusterOverflow, hullOverflow and invalidCount must be 0; received ${status.join(', ')}`
    );
    const [
      rawLabels,
      rawCoreFlags,
      rawClusterCount,
      rawCounts,
      rawCenters,
      rawIndices,
      rawOffsets,
      rawHullCounts,
      rawAreas,
      rawPerimeters
    ] = await Promise.all([
      session.readUint32(labels),
      session.readUint32(coreFlags),
      session.readScalar(clusterCount),
      session.readUint32(counts),
      session.readFloat32(meanCenters),
      session.readUint32(hullVertexIndices),
      session.readUint32(hullOffsets),
      session.readUint32(hullCounts),
      session.readFloat32(areas),
      session.readFloat32(perimeters)
    ]);
    // A finite input can still overflow the GPU's float32 coordinate sums without setting a
    // recipe status word. Empty group centers may be NaN; occupied groups must be publishable.
    rawCounts.forEach((count, cluster) => {
      if (count > 0) {
        validateRequirement(
          BINDING_ID,
          Number.isFinite(rawCenters[2 * cluster]) && Number.isFinite(rawCenters[2 * cluster + 1]),
          `meanCenters[${cluster}] must be finite for an occupied cluster (count ${count})`
        );
      }
    });
    return {
      labels: Int32Array.from(rawLabels, label =>
        label === GPU_SPATIAL_CLUSTERING_NOISE ? -1 : label
      ),
      coreFlags: rawCoreFlags,
      clusterCount: rawClusterCount,
      counts: rawCounts,
      meanCenters: rawCenters,
      hullVertexIndices: rawIndices,
      hullOffsets: rawOffsets,
      hullCounts: rawHullCounts,
      areas: rawAreas,
      perimeters: rawPerimeters,
      dispatchedContributors: recipe.contributors.flatMap(contributor => {
        const name = CONTRIBUTOR_NAMES.get(contributor.constructor);
        return name ? [name] : [];
      })
    };
  });
  if (frame.gpuFrame !== null && result.clusterCount > 0) {
    await measureGeographicHulls(context, sample, frame, result);
  }
  return result;
}

/** Re-measures the GPU-selected hulls on the binding's sphere rather than in the planar frame. */
async function measureGeographicHulls(
  context: CatalogGpuContext,
  sample: Sample,
  frame: LocalFrame,
  result: CuratedClustersGpuResult
): Promise<void> {
  const vertexCount = result.hullOffsets[result.clusterCount];
  const positions = new Float32Array(vertexCount * 2);
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const sourceIndex = result.hullVertexIndices[vertex];
    positions.set(frame.inverse(sample.x[sourceIndex], sample.y[sourceIndex]), 2 * vertex);
  }
  await withCatalogGpuSession(context.device, `${BINDING_ID}-spherical-measures`, async session => {
    const areas = session.output('areas', 'float32', result.clusterCount);
    const perimeters = session.output('perimeters', 'float32', result.clusterCount);
    const invalidCount = session.output('invalid-count', 'uint32', 1);
    const identity = session.input(
      'identity',
      Uint32Array.from({length: result.clusterCount + 1}, (_, index) => index),
      'uint32'
    );
    const measures = new GPUGeometryMeasures({
      id: `${BINDING_ID}-spherical-measures`,
      geometryType: 'polygons',
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters',
        edgeModel: 'linear',
        sphereRadius: EARTH_RADIUS_METERS
      },
      positions: session.input('positions', positions, 'float32x2'),
      featureOffsets: identity,
      polygonOffsets: identity,
      ringOffsets: session.input(
        'ring-offsets',
        result.hullOffsets.slice(0, result.clusterCount + 1),
        'uint32'
      ),
      output: {
        areas: areas.view,
        lengths: perimeters.view,
        status: {invalidCount: invalidCount.view}
      }
    });
    session.graph.add(measures);
    session.run();
    const invalid = await session.readScalar(invalidCount);
    validateRequirement(
      BINDING_ID,
      invalid === 0,
      `spherical hull invalidCount must be 0; received ${invalid}`
    );
    const [sphericalAreas, sphericalPerimeters] = await Promise.all([
      session.readFloat32(areas),
      session.readFloat32(perimeters)
    ]);
    result.areas.set(sphericalAreas);
    result.perimeters.set(sphericalPerimeters);
  });
}
