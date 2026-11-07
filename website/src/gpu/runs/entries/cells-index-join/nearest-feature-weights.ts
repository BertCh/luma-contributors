// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNearestFeatureWeights turns the neighbor lists of GPUNearestFeatureJoin into a spatial weights
// matrix (k-nearest-neighbor weights). The points are their own features (a self-join with
// excludeSelf); lines are the CSR edges the operation wrote and dots are colored by each row's
// total weight, so inverse-distance weights make isolated points heavier than crowded ones.

import {
  GPUNearestFeatureJoin,
  GPUNearestFeatureWeights
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCsrEdgeSegments, addCsrRowSums} from './support/kernels';
import {createRandomPoints} from './support/join';
import {parameters} from './nearest-feature-weights.parameters';

const POINT_COUNT = 90;

export default defineOperationRun({
  parameters,
  caption: 'k-nearest-neighbor weights: edges and the total weight of every point.',
  createInputs: ({random, world}) => ({points: createRandomPoints(random, world, POINT_COUNT)}),
  build(builder, inputs, values) {
    const k = values.k;
    const slotCapacity = k + 1;
    const points = builder.upload('points', 'float32x2', inputs.points);
    const radius = builder.upload('radius', 'float32', Float32Array.of(1e6));
    const neighborIds = builder.allocate('neighbor-ids', 'uint32', POINT_COUNT * slotCapacity);
    const neighborCounts = builder.allocate('neighbor-counts', 'uint32', POINT_COUNT);
    const neighborDistances = builder.allocate(
      'neighbor-distances',
      'float32',
      POINT_COUNT * slotCapacity
    );
    const joinOverflow = builder.allocate('join-overflow', 'uint32', 1);
    builder.graph.add(
      new GPUNearestFeatureJoin({
        queries: {kind: 'points', positions: points.view},
        features: {kind: 'points', positions: points.view},
        radius: radius.view,
        k: slotCapacity,
        output: {
          neighborIds: neighborIds.view,
          neighborCounts: neighborCounts.view,
          neighborDistances: neighborDistances.view,
          status: {overflow: joinOverflow.view}
        }
      })
    );
    const offsets = builder.allocate('offsets', 'uint32', POINT_COUNT + 1);
    const neighbors = builder.allocate('neighbors', 'uint32', POINT_COUNT * k);
    const weights = builder.allocate('weights', 'float32', POINT_COUNT * k);
    const distances = builder.allocate('distances', 'float32', POINT_COUNT * k);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUNearestFeatureWeights({
        neighborIds: neighborIds.view,
        neighborCounts: neighborCounts.view,
        neighborDistances: neighborDistances.view,
        slotCapacity,
        k,
        excludeSelf: true,
        weightType: values.weightType === 0 ? 'binary' : 'inverse-distance',
        power: values.power,
        output: {
          offsets: offsets.view,
          neighbors: neighbors.view,
          weights: weights.view,
          distances: distances.view,
          status: {overflow: overflow.view}
        }
      })
    );
    return {
      outputs: {
        points,
        edges: addCsrEdgeSegments(builder, 'edges', {
          positions: points,
          offsets,
          neighbors,
          rowCount: POINT_COUNT
        }),
        rowSums: addCsrRowSums(builder, 'weights', {offsets, weights, rowCount: POINT_COUNT})
      }
    };
  },
  draw: ({outputs, values}) => [
    {
      layer: 'lines',
      segments: outputs.edges.segments,
      count: outputs.edges.count,
      radius: 1,
      color: [1, 1, 1, 0.45]
    },
    {
      layer: 'dots',
      positions: outputs.points,
      values: outputs.rowSums,
      ramp: 'sequential',
      // Binary rows sum to k; inverse-distance rows sum to about k / typical distance ^ power.
      range: [0, values.weightType === 0 ? values.k + 0.5 : values.k * 12 ** values.power],
      radius: 6
    }
  ]
});
