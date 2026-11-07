// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUBufferSelection keeps the points within `distance` of a road (a chain of segments). Selected
// points are bright, the rest muted; drag the distance and the selection follows with no recompile.

import {GPUBufferSelection} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createRandomPoints} from './support/join';
import {parameters} from './buffer-selection.parameters';

const POINT_COUNT = 900;
const SEGMENT_COUNT = 40;

export default defineOperationRun({
  parameters,
  caption: 'Points within a distance of the road stay selected.',
  createInputs: ({random, world}) => ({
    points: createRandomPoints(random, world, POINT_COUNT),
    world
  }),
  build(builder, inputs) {
    const points = builder.upload('points', 'float32x2', inputs.points);
    const starts = builder.allocate('road-starts', 'float32x2', SEGMENT_COUNT);
    const ends = builder.allocate('road-ends', 'float32x2', SEGMENT_COUNT);
    const road = builder.allocate('road', 'float32x4', SEGMENT_COUNT);
    const distance = builder.allocate('distance', 'float32', 1);
    const outputMask = builder.allocate('output-mask', 'uint32', POINT_COUNT);
    const distances = builder.allocate('distances', 'float32', POINT_COUNT);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUBufferSelection({
        points: {kind: 'points', positions: points.view},
        features: {kind: 'segments', starts: starts.view, ends: ends.view},
        distance: distance.view,
        candidateCapacity: 1 << 16,
        output: {
          outputMask: outputMask.view,
          distances: distances.view,
          status: {overflow: overflow.view}
        }
      })
    );
    return {
      outputs: {points, road, outputMask},
      update: current => {
        const vertices = Array.from({length: SEGMENT_COUNT + 1}, (_, index) => {
          const fraction = index / SEGMENT_COUNT;
          return [
            fraction * inputs.world.width,
            0.5 + current.bend * Math.sin(fraction * Math.PI * 2.2) + (fraction - 0.5) * 0.2
          ];
        });
        starts.write(Float32Array.from(vertices.slice(0, -1).flat()));
        ends.write(Float32Array.from(vertices.slice(1).flat()));
        road.write(
          Float32Array.from(
            vertices.slice(0, -1).flatMap((vertex, index) => [...vertex, ...vertices[index + 1]])
          )
        );
        distance.write(Float32Array.of(current.distance));
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.points, values: outputs.outputMask, radius: 3},
    {layer: 'lines', context: true, segments: outputs.road, radius: 2, color: [1, 1, 1, 0.9]}
  ]
});
