// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSphericalPolygonContains tests query points against a spherical polygon with a hole: a
// non-convex star shell plus an inner ring. Dots are colored by the `contains` flag it wrote.

import {GPUSphericalPolygonContains} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {packUnitVectors} from './support/polygons';
import {parameters} from './spherical-polygon-contains.parameters';

const QUERY_COUNT = 900;
const SHELL_VERTEX_COUNT = 12;
const HOLE_VERTEX_COUNT = 8;

/** A star shell (alternating radii) and a counter-clockwise hole ring around `center`. */
function createRings(center: [number, number], size: number, holeSize: number) {
  const shell = Array.from({length: SHELL_VERTEX_COUNT}, (_, index) => {
    const angle = (2 * Math.PI * index) / SHELL_VERTEX_COUNT;
    const radius = size * (index % 2 === 0 ? 1 : 0.62);
    return [center[0] + radius * Math.cos(angle), center[1] + radius * Math.sin(angle)];
  });
  const hole = Array.from({length: HOLE_VERTEX_COUNT}, (_, index) => {
    const angle = (2 * Math.PI * index) / HOLE_VERTEX_COUNT;
    return [center[0] + holeSize * Math.cos(angle), center[1] + holeSize * Math.sin(angle)];
  });
  return [shell, hole];
}

export default defineOperationRun({
  parameters,
  caption: 'Points inside a spherical polygon with a hole, tested on the GPU.',
  createInputs: ({random, world}) => {
    const queries = new Float32Array(QUERY_COUNT * 2);
    for (let index = 0; index < QUERY_COUNT; index++) {
      queries[index * 2] = random() * world.width;
      queries[index * 2 + 1] = random();
    }
    return {queries, center: [world.width / 2, 0.5] as [number, number]};
  },
  build(builder, inputs, values) {
    const vertexCount = SHELL_VERTEX_COUNT + HOLE_VERTEX_COUNT;
    const positions = builder.allocate('positions', 'float32x3', vertexCount);
    const outline = builder.allocate('outline', 'float32x4', vertexCount);
    const contains = builder.allocate('contains', 'uint32', QUERY_COUNT);
    const status = builder.allocate('status', 'uint32', QUERY_COUNT);
    const queryPairs = Array.from({length: QUERY_COUNT}, (_, index) => [
      inputs.queries[index * 2],
      inputs.queries[index * 2 + 1]
    ]);
    builder.graph.add(
      new GPUSphericalPolygonContains({
        positions: positions.view,
        ringOffsets: builder.upload(
          'ring-offsets',
          'uint32',
          Uint32Array.of(0, SHELL_VERTEX_COUNT, vertexCount)
        ).view,
        polygonOffsets: builder.upload('polygon-offsets', 'uint32', Uint32Array.of(0, 2)).view,
        queries: builder.upload('queries-xyz', 'float32x3', packUnitVectors([queryPairs])).view,
        polygonIndices: builder.upload('polygon-indices', 'uint32', new Uint32Array(QUERY_COUNT))
          .view,
        complement: values.complement === 1,
        output: {contains: contains.view, status: status.view}
      })
    );
    return {
      outputs: {contains, outline, queries: builder.upload('queries', 'float32x2', inputs.queries)},
      update: current => {
        const rings = createRings(inputs.center, current.size, Math.max(current.holeSize, 0.02));
        positions.write(packUnitVectors(rings));
        outline.write(
          Float32Array.from(
            rings.flatMap(ring =>
              ring.flatMap(([x, y], index) => [x, y, ...ring[(index + 1) % ring.length]])
            )
          )
        );
      }
    };
  },
  rebuildOn: ['complement'],
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.queries, values: outputs.contains, radius: 2.6},
    {layer: 'lines', context: true, segments: outputs.outline, radius: 1.8, color: [1, 1, 1, 0.9]}
  ]
});
