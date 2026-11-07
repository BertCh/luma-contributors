// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addClusterAndOutlineRecipe: DBSCAN clusters (point colors) with the convex hull vertices of every
// cluster (white rings) and the hull area per cluster as bars.

import {
  addClusterAndOutlineRecipe,
  getGPUGeographicDistributionParameterValues,
  getGPUSpatialClusteringParameterValues,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {parameters} from './cluster-and-outline-recipe.parameters';

const MAXIMUM_CLUSTERS = 16;
const HULL_CAPACITY = 512;

export default defineOperationRun({
  parameters,
  caption: 'Clusters and the hull of each: hull vertices and area per cluster.',
  createInputs: ({random, world}) => ({
    points: createClusteredPoints(random, world, 700, 4, 0.1),
    world
  }),
  build(builder, {points, world}) {
    const parameterBuffer = builder.allocate(
      'clustering-parameters',
      'float32',
      GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH
    );
    const geometryBuffer = builder.upload(
      'geometry-parameters',
      'float32',
      getGPUGeographicDistributionParameterValues()
    );
    const labels = builder.allocate('labels', 'uint32', points.count);
    const clusterCount = builder.allocate('cluster-count', 'uint32', 1);
    const hullOffsets = builder.allocate('hull-offsets', 'uint32', MAXIMUM_CLUSTERS + 1);
    const hullPositions = builder.allocate('hull-positions', 'float32x2', HULL_CAPACITY);
    const hullIndices = builder.allocate('hull-indices', 'uint32', HULL_CAPACITY);
    const hullOverflow = builder.allocate('hull-overflow', 'uint32', 1);
    const areas = builder.allocate('areas', 'float32', MAXIMUM_CLUSTERS);
    const perimeters = builder.allocate('perimeters', 'float32', MAXIMUM_CLUSTERS);
    const counts = builder.allocate('counts', 'uint32', MAXIMUM_CLUSTERS);
    const meanCenters = builder.allocate('mean-centers', 'float32x2', MAXIMUM_CLUSTERS);
    const hullCounts = builder.allocate('hull-counts', 'uint32', MAXIMUM_CLUSTERS);
    addClusterAndOutlineRecipe(builder.graph, {
      points: {
        kind: 'points',
        positions: builder.upload('positions', 'float32x2', points.positions).view
      },
      clusteringParameters: parameterBuffer.view,
      gridSize: [64, 36],
      geometryParameters: geometryBuffer.view,
      maximumClusterCount: MAXIMUM_CLUSTERS,
      maximumVerticesPerHull: 64,
      hullCapacity: HULL_CAPACITY,
      outputs: {
        labels: labels.view,
        clusterCount: clusterCount.view,
        hullOffsets: hullOffsets.view,
        hullPositions: hullPositions.view,
        areas: areas.view,
        perimeters: perimeters.view,
        status: {hullOverflow: hullOverflow.view}
      },
      scratch: {
        counts: counts.view,
        meanCenters: meanCenters.view,
        hullCounts: hullCounts.view,
        hullVertexIndices: hullIndices.view
      }
    });
    return {
      outputs: {
        labels,
        hullPositions,
        hullIndices,
        areas,
        positions: builder.upload('points', 'float32x2', points.positions)
      },
      update: current =>
        parameterBuffer.write(
          getGPUSpatialClusteringParameterValues({
            bounds: [0, 0, world.width, 1],
            epsilon: current.epsilon / 100,
            minimumPoints: current.minimumPoints
          })
        )
    };
  },
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.positions, values: outputs.labels, radius: 3.5},
    // Unused hull slots hold 0xffffffff indices, which draw as nothing.
    {
      layer: 'dots',
      positions: outputs.hullPositions,
      values: outputs.hullIndices,
      radius: 7,
      color: [1, 1, 1, 0.95]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.areas,
      ramp: 'sequential',
      rectangle: [0.74, 0.04, 0.97, 0.2]
    }
  ]
});
