// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkSnapping: move each point to the nearest road edge. Muted dots are the raw points; the
// bright dots are where they snapped, colored by snap distance. Points farther than the maximum
// distance stay unsnapped.

import {GPUNetworkSnapping} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {findParameter, snapToParameter} from '../../interaction';
import {addPairSegments} from './shared/display';
import {createRoads, findReachPoint, getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './network-snapping.parameters';

const POINT_COUNT = 70;
const MAXIMUM_SNAP = findParameter(parameters, 'maximumSnapDistance');

export default defineOperationRun({
  parameters,
  caption: 'Raw points (muted) and their snapped positions, colored by snap distance.',
  createInputs: ({random, world}) => {
    const offsets = Float32Array.from({length: POINT_COUNT * 2}, () => random() * 2 - 1);
    const anchors = Float32Array.from({length: POINT_COUNT * 2}, (_, index) =>
      index % 2 === 0 ? random() * world.width : random()
    );
    return {
      network: createRoads(random, world, 14),
      raw: Float32Array.from(anchors, (anchor, index) => anchor + offsets[index] * 0.03),
      offsets,
      anchors
    };
  },
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
      outputs: {
        ...uploadRoads(builder, network),
        points,
        snappedEdges,
        snapDistances,
        snappedPositions,
        connectors: addPairSegments(builder, {id: 'snap', from: points, to: snappedPositions})
      },
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
  // Reach: the distance from the nearest raw point to the pointer is the maximum snap distance.
  interaction: {
    kind: 'reach',
    hint: 'reach for the road',
    rest: ({inputs, world, toUnit}) =>
      toUnit(findReachPoint(inputs.raw, MAXIMUM_SNAP.default, world)),
    resolve(pointer, {inputs, outputs, peek}) {
      let nearest = 0;
      let nearestDistance = Infinity;
      for (let point = 0; point < POINT_COUNT; point++) {
        const distance = Math.hypot(
          inputs.raw[point * 2] - pointer.x,
          inputs.raw[point * 2 + 1] - pointer.y
        );
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = point;
        }
      }
      const maximumSnapDistance = snapToParameter(MAXIMUM_SNAP, nearestDistance);
      const edges = peek(outputs.snappedEdges);
      let snapped = 0;
      if (edges) for (const edge of edges) if (edge !== 0xffffffff) snapped++;
      return {
        values: {maximumSnapDistance},
        glyphs: [
          {
            shape: 'ring',
            at: [inputs.raw[nearest * 2], inputs.raw[nearest * 2 + 1]],
            radius: maximumSnapDistance
          },
          {
            shape: 'line',
            from: [inputs.raw[nearest * 2], inputs.raw[nearest * 2 + 1]],
            to: [pointer.x, pointer.y],
            dashed: true
          }
        ],
        readout: `maximumSnapDistance ${maximumSnapDistance.toFixed(3)} · snapped ${Math.round((100 * snapped) / POINT_COUNT)}%`
      };
    }
  },
  draw: ({outputs, theme}) => [
    getRoadContext(outputs),
    // The snap links: raw point to its snapped position, bold, so the move is the first thing seen.
    {layer: 'lines', segments: outputs.connectors, radius: 2.2, color: [...theme.ink, 0.9]},
    // Raw points: orange; snapped positions in an ink ring on the roads, colored by snap distance.
    {
      layer: 'dots',
      context: true,
      positions: outputs.points,
      radius: 5,
      color: [...theme.categories[1], 0.95]
    },
    {layer: 'dots', positions: outputs.snappedPositions, radius: 8, color: [...theme.ink, 1]},
    {
      layer: 'dots',
      positions: outputs.snappedPositions,
      values: outputs.snapDistances,
      range: [-0.02, 0.1],
      radius: 6
    }
  ]
});
