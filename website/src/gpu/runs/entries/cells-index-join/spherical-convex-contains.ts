// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSphericalConvexContains tests query points on the unit sphere against convex spherical
// polygons (great-circle edges). Each query belongs to the polygon of its half of the window; dots
// are colored by the `contains` flag the operation wrote.

import {GPUSphericalConvexContains} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {packUnitVectors} from './support/polygons';
import {parameters} from './spherical-convex-contains.parameters';

const QUERY_COUNT = 700;

export default defineOperationRun({
  parameters,
  caption: 'Query points inside each convex spherical polygon, tested on the GPU.',
  createInputs: ({random, world}) => {
    const queries = new Float32Array(QUERY_COUNT * 2);
    const polygonIndices = new Uint32Array(QUERY_COUNT);
    for (let index = 0; index < QUERY_COUNT; index++) {
      queries[index * 2] = random() * world.width;
      queries[index * 2 + 1] = random();
      polygonIndices[index] = queries[index * 2] < world.width / 2 ? 0 : 1;
    }
    return {
      queries,
      polygonIndices,
      centers: [
        [world.width * 0.25, 0.5],
        [world.width * 0.75, 0.5]
      ] as [number, number][]
    };
  },
  build(builder, inputs, values) {
    const sides = values.sides;
    const positions = builder.allocate('positions', 'float32x3', sides * 2);
    const outline = builder.allocate('outline', 'float32x4', sides * 2);
    const contains = builder.allocate('contains', 'uint32', QUERY_COUNT);
    const status = builder.allocate('status', 'uint32', QUERY_COUNT);
    builder.graph.add(
      new GPUSphericalConvexContains({
        positions: positions.view,
        offsets: builder.upload('offsets', 'uint32', Uint32Array.of(0, sides, sides * 2)).view,
        queries: builder.upload(
          'queries-xyz',
          'float32x3',
          packUnitVectors([[...toPairs(inputs.queries)]])
        ).view,
        polygonIndices: builder.upload('polygon-indices', 'uint32', inputs.polygonIndices).view,
        complement: values.complement === 1,
        output: {contains: contains.view, status: status.view}
      })
    );
    return {
      outputs: {contains, outline, queries: builder.upload('queries', 'float32x2', inputs.queries)},
      update: current => {
        const rings = inputs.centers.map(([x, y]) =>
          Array.from({length: sides}, (_, index) => {
            const angle = (2 * Math.PI * index) / sides + 0.3;
            return [x + current.size * Math.cos(angle), y + current.size * Math.sin(angle)];
          })
        );
        positions.write(packUnitVectors(rings));
        outline.write(
          Float32Array.from(
            rings.flatMap(ring =>
              ring.flatMap(([x, y], index) => [x, y, ...ring[(index + 1) % sides]])
            )
          )
        );
      }
    };
  },
  rebuildOn: ['complement', 'sides'],
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.queries, values: outputs.contains, radius: 3},
    {layer: 'lines', context: true, segments: outputs.outline, radius: 1.8, color: [1, 1, 1, 0.9]}
  ]
});

function toPairs(flat: Float32Array): number[][] {
  return Array.from({length: flat.length / 2}, (_, index) => [
    flat[index * 2],
    flat[index * 2 + 1]
  ]);
}
