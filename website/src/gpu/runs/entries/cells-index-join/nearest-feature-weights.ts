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
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {addCsrEdgeSegments, addCsrRowSums} from './support/kernels';
import {getRainStations} from '../../../data/point-observations';
import {parameters} from './nearest-feature-weights.parameters';

const POINT_COUNT = 90;

export default defineOperationRun({
  parameters,
  caption: 'k-nearest-neighbor weights: edges and the total weight of every point.',
  createInputs: ({random, world}) => {
    const points = getRainStations(random, world, POINT_COUNT, true).positions;
    return {points, index: createPointIndex(points), held: -1};
  },
  build(builder, inputs, values) {
    const k = values['neighborCount'];
    const slotCapacity = k + 1;
    const points = builder.upload('points', 'float32x2', inputs.points);
    const radius = builder.upload('radius', 'float32', Float32Array.of(1e6));
    const neighborIds = builder.allocate('neighbor-ids', 'uint32', POINT_COUNT * slotCapacity);
    const neighborRows = builder.allocate('neighbor-rows', 'uint32', POINT_COUNT * slotCapacity);
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
        neighborCount: slotCapacity,
        output: {
          neighborIds: neighborIds.view,
          neighborRows: neighborRows.view,
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
    const candidateOverflow = builder.allocate('candidate-overflow', 'uint32', 1);
    builder.graph.add(
      new GPUNearestFeatureWeights({
        neighborRows: neighborRows.view,
        neighborStatus: {overflow: joinOverflow.view},
        neighborCounts: neighborCounts.view,
        neighborDistances: neighborDistances.view,
        slotCapacity,
        neighborCount: k,
        excludeSelf: true,
        weightType: values['weightType'] === 0 ? 'binary' : 'inverse-distance',
        power: values['power'],
        output: {
          offsets: offsets.view,
          neighbors: neighbors.view,
          weights: weights.view,
          distances: distances.view,
          status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
        }
      })
    );
    return {
      outputs: {
        points,
        offsets,
        neighbors,
        weights,
        overflow,
        candidateOverflow,
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
  // Render-only probe: the CSR row of the point under the pointer, read back from the GPU.
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.held);
      inputs.held = row;
      if (row < 0) return {};
      const x = inputs.points[row * 2];
      const y = inputs.points[row * 2 + 1];
      const glyphs: RunGlyph[] = [{shape: 'ring', at: [x, y], radius: 0.025}];
      const offsets = peek(outputs.offsets);
      const neighbors = peek(outputs.neighbors);
      const weights = peek(outputs.weights);
      let total = 0;
      if (offsets && neighbors && weights) {
        for (let entry = offsets[row]; entry < offsets[row + 1]; entry++) {
          const to = [
            inputs.points[neighbors[entry] * 2],
            inputs.points[neighbors[entry] * 2 + 1]
          ] as const;
          glyphs.push({shape: 'line', from: [x, y], to}, {shape: 'dot', at: to});
          total += weights[entry];
        }
      }
      return {
        focus: row,
        glyphs,
        readout: `row ${row} · k ${values['neighborCount']} · Σ weights ${total.toFixed(2)}`
      };
    }
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
      range: [
        0,
        values['weightType'] === 0
          ? values['neighborCount'] + 0.5
          : values['neighborCount'] * 12 ** values['power']
      ],
      radius: 6
    }
  ]
});
