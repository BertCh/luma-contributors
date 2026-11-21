// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addClusterAndOutlineRecipe: DBSCAN clusters (point colors) with the convex hull vertices of every
// cluster (white rings) and the hull area per cluster as bars.

import {
  addClusterAndOutlineRecipe,
  getGPUGeographicDistributionParameterValues,
  getGPUDBSCANParameterValues,
  GPU_DBSCAN_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {createClusteredPoints} from '../../inputs';
import {findBusiestPoint} from './lib/points';
import {parameters} from './cluster-and-outline-recipe.parameters';

const MAXIMUM_CLUSTERS = 16;
const HULL_CAPACITY = 512;
/** Pointer travel (world units) per world unit of epsilon. */
const POINTER_PER_EPSILON = 12;

export default defineOperationRun({
  parameters,
  caption: 'Clusters and the hull of each: hull vertices and area per cluster.',
  createInputs: ({random, world}) => {
    const points = createClusteredPoints(random, world, 700, 4, 0.1);
    return {points, world, anchor: findBusiestPoint(points.positions, 0.08)};
  },
  build(builder, {points, world}) {
    const parameterBuffer = builder.allocate(
      'clustering-parameters',
      'float32',
      GPU_DBSCAN_PARAMETER_LENGTH
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
    const hullEdges = builder.allocate('hull-edges', 'float32x4', HULL_CAPACITY);
    const hullEdgeCount = builder.allocate('hull-edge-count', 'uint32', 1);
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
      output: {
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
        clusterCount,
        hullOffsets,
        hullEdges,
        hullEdgeCount,
        hullPositions,
        hullIndices,
        areas,
        positions: builder.upload('points', 'float32x2', points.positions)
      },
      update: current =>
        parameterBuffer.write(
          getGPUDBSCANParameterValues({
            bounds: [0, 0, world.width, 1],
            epsilon: current.epsilon / 100,
            minimumSamples: current.minimumSamples
          })
        )
    };
  },
  // The distance from a busy point to the pointer is epsilon (the ring), and the hull edges are
  // drawn from the hull vertices the operation wrote.
  interaction: {
    kind: 'reach',
    hint: 'stretch epsilon',
    rest: ({toUnit, inputs}) =>
      toUnit([
        inputs.anchor[0] + (parameters[0].default / 100) * POINTER_PER_EPSILON,
        inputs.anchor[1]
      ]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const distance = Math.hypot(pointer.x - inputs.anchor[0], pointer.y - inputs.anchor[1]);
      const epsilon = snapToParameter(
        findParameter(parameters, 'epsilon'),
        (distance / POINTER_PER_EPSILON) * 100
      );
      const offsets = peek(outputs.hullOffsets);
      const positions = peek(outputs.hullPositions);
      const clusters = peek(outputs.clusterCount);
      if (offsets && positions && clusters) {
        const edges: number[] = [];
        for (let cluster = 0; cluster < Math.min(clusters[0], MAXIMUM_CLUSTERS); cluster++) {
          const start = offsets[cluster];
          const end = offsets[cluster + 1];
          for (let vertex = start; vertex < end; vertex++) {
            const next = vertex + 1 < end ? vertex + 1 : start;
            edges.push(
              positions[vertex * 2],
              positions[vertex * 2 + 1],
              positions[next * 2],
              positions[next * 2 + 1]
            );
          }
        }
        outputs.hullEdges.write(Float32Array.from(edges).subarray(0, HULL_CAPACITY * 4));
        outputs.hullEdgeCount.write(Uint32Array.of(Math.min(edges.length / 4, HULL_CAPACITY)));
      }
      return {
        values: {epsilon},
        glyphs: [
          {shape: 'ring', at: inputs.anchor, radius: epsilon / 100},
          {shape: 'line', from: inputs.anchor, to: [pointer.x, pointer.y], dashed: true},
          {shape: 'dot', at: inputs.anchor}
        ],
        readout: `epsilon ${epsilon} % · ${clusters ? clusters[0] : '...'} clusters · minimumSamples ${values.minimumSamples}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.positions, values: outputs.labels, radius: 3.5},
    {
      layer: 'lines',
      segments: outputs.hullEdges,
      count: outputs.hullEdgeCount,
      radius: 2,
      color: [1, 1, 1, 0.9]
    },
    // Unused hull slots hold 0xffffffff indices, which draw as nothing.
    {
      layer: 'dots',
      positions: outputs.hullPositions,
      values: outputs.hullIndices,
      radius: 4,
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
