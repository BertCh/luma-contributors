// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkSnapping: move each point to the nearest road edge. Muted dots are the raw points; the
// bright dots are where they snapped, colored by snap distance. Points farther than the maximum
// distance stay unsnapped.

import {GPUNetworkSnapping} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {createRoads, getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './network-snapping.parameters';

const POINT_COUNT = 500;

export default defineOperationRun({
  parameters,
  caption: 'Raw points (muted) and their snapped positions, colored by snap distance.',
  createInputs: ({random, world}) => ({
    network: createRoads(random, world, 14),
    raw: new Float32Array(POINT_COUNT * 2),
    offsets: Float32Array.from({length: POINT_COUNT * 2}, () => random() * 2 - 1),
    anchors: Float32Array.from({length: POINT_COUNT * 2}, (_, index) =>
      index % 2 === 0 ? random() * world.width : random()
    )
  }),
  build(builder, inputs) {
    const {network, raw, offsets, anchors} = inputs;
    const {topology} = uploadNetworkTopology(builder, network);
    const nodePositions = builder.upload('node-positions', 'float32x2', network.nodes);
    const points = builder.allocate('points', 'float32x2', POINT_COUNT);
    const maximumSnapDistance = builder.allocate('maximum-snap-distance', 'float32', 1);
    const snappedEdges = builder.allocate('snapped-edges', 'uint32', POINT_COUNT);
    const snapDistances = builder.allocate('snap-distances', 'float32', POINT_COUNT);
    const snappedPositions = builder.allocate('snapped-positions', 'float32x2', POINT_COUNT);
    builder.graph.add(
      new GPUNetworkSnapping({
        positions: points.view,
        topology: {...topology, nodePositions: nodePositions.view},
        maximumSnapDistance: maximumSnapDistance.view,
        output: {
          snappedEdges: snappedEdges.view,
          snapDistances: snapDistances.view,
          snappedPositions: snappedPositions.view
        }
      })
    );
    return {
      outputs: {...uploadRoads(builder, network), points, snapDistances, snappedPositions},
      update: values => {
        // Points scatter around fixed anchors; `spread` scales the scatter.
        for (let index = 0; index < raw.length; index++) {
          raw[index] = anchors[index] + offsets[index] * 0.03 * values.spread;
        }
        points.write(raw);
        maximumSnapDistance.write(Float32Array.of(values.maximumSnapDistance));
      }
    };
  },
  draw: ({outputs}) => [
    getRoadContext(outputs),
    {layer: 'dots', context: true, positions: outputs.points, radius: 3},
    {
      layer: 'dots',
      positions: outputs.snappedPositions,
      values: outputs.snapDistances,
      range: [0, 0.04],
      radius: 4.5
    }
  ]
});
