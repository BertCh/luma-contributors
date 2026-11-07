// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellGridPath finds the H3 grid path between origin and destination cells. The cells of every
// path (origin first, destination last) are decoded with GPUCellGeometry and filled; the white dots
// are the origin and destination points that were keyed.

import {GPUCellGridPath} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCellFill, addPointKeys} from './support/cells';
import {parameters} from './cell-grid-path.parameters';

const RESOLUTION = 5;

export default defineOperationRun({
  parameters,
  caption: 'The cells along each grid path, and the pairs that were joined.',
  createInputs: ({world}) => ({
    origins: Float32Array.of(
      world.width * 0.2,
      0.25,
      world.width * 0.3,
      0.75,
      world.width * 0.55,
      0.3,
      world.width * 0.7,
      0.8
    ),
    world
  }),
  build(builder, inputs, values) {
    const pairCount = inputs.origins.length / 2;
    const origins = builder.upload('origins', 'float32x2', inputs.origins);
    const destinations = builder.allocate('destinations', 'float32x2', pairCount);
    const originCells = addPointKeys(builder, 'h3', RESOLUTION, origins, 'origin-keys');
    const destinationCells = addPointKeys(
      builder,
      'h3',
      RESOLUTION,
      destinations,
      'destination-keys'
    );
    const length = values.maximumPathLength;
    const pathCells = builder.allocate('path-cells', 'uint32x2', pairCount * length);
    const counts = builder.allocate('counts', 'uint32', pairCount);
    builder.graph.add(
      new GPUCellGridPath({
        family: 'h3',
        origins: originCells.view,
        destinations: destinationCells.view,
        maximumPathLength: length,
        output: {cells: pathCells.view, counts: counts.view}
      })
    );
    return {
      outputs: {
        origins,
        destinations,
        pairs: builder.upload(
          'pairs',
          'uint32',
          Uint32Array.from({length: pairCount * length}, (_, row) => Math.floor(row / length))
        ),
        fill: addCellFill(builder, 'h3', pathCells)
      },
      update: current => {
        // Destinations sit `spread` away from each origin, alternating direction.
        destinations.write(
          Float32Array.from(inputs.origins, (value, index) => {
            const pair = index >> 1;
            const direction = pair % 2 === 0 ? 1 : -1;
            return index % 2 === 0
              ? value + direction * current.spread * 0.9
              : value + direction * current.spread * 0.3;
          })
        );
      }
    };
  },
  rebuildOn: ['maximumPathLength'],
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.pairs,
      triangleCount: outputs.fill.triangleCount,
      alpha: 0.85
    },
    {layer: 'dots', positions: outputs.origins, radius: 4, color: [1, 1, 1, 1]},
    {layer: 'dots', positions: outputs.destinations, radius: 4, color: [1, 1, 1, 1]}
  ]
});
