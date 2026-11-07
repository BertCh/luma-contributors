// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSphericalConvexIntersection clips pairs of convex spherical polygons against each other. The
// thin outlines are the two inputs of each pair; the bright ring is the intersection polygon the
// operation wrote (unit vectors converted to longitude/latitude to draw them).

import {GPUSphericalConvexIntersection} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addGroupedRingSegments, addUnitVectorsToLonLat} from './support/kernels';
import {packUnitVectors} from './support/polygons';
import {parameters} from './spherical-convex-intersection.parameters';

const CAPACITY = 16;
const PAIR_COUNT = 3;

function createRing(center: [number, number], radius: number, sides: number, turn: number) {
  return Array.from({length: sides}, (_, index) => {
    const angle = (2 * Math.PI * index) / sides + turn;
    return [center[0] + radius * Math.cos(angle), center[1] + radius * Math.sin(angle)];
  });
}

export default defineOperationRun({
  parameters,
  caption: 'Intersection of each polygon pair on the sphere.',
  createInputs: ({world}) => ({
    centers: [
      [world.width * 0.2, 0.5],
      [world.width * 0.5, 0.5],
      [world.width * 0.8, 0.5]
    ] as [number, number][]
  }),
  build(builder, inputs, values) {
    const sides = values.sides;
    const first = builder.allocate('first-positions', 'float32x3', PAIR_COUNT * sides);
    const second = builder.allocate('second-positions', 'float32x3', PAIR_COUNT * sides);
    const offsets = Uint32Array.from({length: PAIR_COUNT + 1}, (_, index) => index * sides);
    const intersection = builder.allocate('intersection', 'float32x3', PAIR_COUNT * CAPACITY);
    const counts = builder.allocate('counts', 'uint32x2', PAIR_COUNT);
    const status = builder.allocate('status', 'uint32', PAIR_COUNT);
    const measures = builder.allocate('measures', 'float32x4', PAIR_COUNT);
    builder.graph.add(
      new GPUSphericalConvexIntersection({
        firstPositions: first.view,
        secondPositions: second.view,
        firstOffsets: builder.upload('first-offsets', 'uint32', offsets).view,
        secondOffsets: builder.upload('second-offsets', 'uint32', offsets).view,
        capacity: CAPACITY,
        output: {
          positions: intersection.view,
          counts: counts.view,
          status: status.view,
          measures: measures.view
        }
      })
    );
    const lonLat = addUnitVectorsToLonLat(builder, 'intersection', intersection);
    const outlines = [
      builder.allocate('first-outline', 'float32x4', PAIR_COUNT * sides),
      builder.allocate('second-outline', 'float32x4', PAIR_COUNT * sides)
    ];
    return {
      outputs: {
        outlines,
        ring: addGroupedRingSegments(builder, 'intersection', lonLat, counts, CAPACITY)
      },
      update: current => {
        const rings = [0, 1].map(side =>
          inputs.centers.map(([x, y]) =>
            createRing(
              [x + (side === 0 ? -1 : 1) * current.separation * 0.5, y],
              0.19,
              sides,
              side === 0 ? 0 : current.rotation
            )
          )
        );
        [first, second].forEach((buffer, side) => buffer.write(packUnitVectors(rings[side])));
        outlines.forEach((outline, side) =>
          outline.write(
            Float32Array.from(
              rings[side].flatMap(ring =>
                ring.flatMap(([x, y], index) => [x, y, ...ring[(index + 1) % sides]])
              )
            )
          )
        );
      }
    };
  },
  rebuildOn: ['sides'],
  draw: ({outputs}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.outlines[0],
      radius: 1.2,
      color: [0.45, 0.65, 1, 0.8]
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.outlines[1],
      radius: 1.2,
      color: [1, 0.6, 0.35, 0.8]
    },
    {layer: 'lines', segments: outputs.ring, radius: 3, color: [1, 1, 1, 1]}
  ]
});
