// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUDistanceField from seed points (stations): Euclidean distance to the nearest seed (bands) and
// the nearest-seed allocation, which is the Voronoi zone. Cells farther than maximumDistance from
// every seed are left empty.

import {
  getGPUDistanceFieldParameterValues,
  GPUDistanceField,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './distance-field.parameters';

const SEED_CAPACITY = 12;
const ROWS = 60;

export default defineOperationRun({
  parameters,
  caption: 'Distance rings and Voronoi zones around seed points.',
  createInputs: ({random, world}) => {
    const columns = Math.round(ROWS * world.width);
    const seeds = new Float32Array(SEED_CAPACITY * 2);
    for (let index = 0; index < SEED_CAPACITY; index++) {
      seeds[index * 2] = world.width * (0.06 + 0.88 * random());
      seeds[index * 2 + 1] = 0.06 + 0.88 * random();
    }
    const cellSize = world.height / ROWS;
    return {
      columns,
      rows: ROWS,
      cellSize,
      seeds,
      grid: makeGPURasterMetadata({
        width: columns,
        height: ROWS,
        cellSize,
        origin: [0, ROWS * cellSize]
      })
    };
  },
  build(builder, inputs) {
    const cellCount = inputs.columns * inputs.rows;
    const seedPositions = builder.upload('seed-positions', 'float32x2', inputs.seeds);
    const seedCount = builder.allocate('seed-count', 'uint32', 1);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const distances = builder.allocate('distances', 'float32', cellCount);
    const allocation = builder.allocate('allocation', 'uint32', cellCount);
    builder.graph.add(
      new GPUDistanceField({
        grid: inputs.grid,
        parameters: parameterBuffer.view,
        seedPositions: seedPositions.view,
        seedCount: seedCount.view,
        output: {distances: distances.view, allocation: allocation.view}
      })
    );
    return {
      outputs: {seedPositions, distances, allocation},
      update: next => {
        seedCount.write(Uint32Array.of(Math.round(next.seedCount)));
        parameterBuffer.write(
          getGPUDistanceFieldParameterValues({maximumDistance: next.maximumDistance}, inputs.grid)
        );
      }
    };
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      values: outputs.allocation,
      columns: inputs.columns,
      rows: inputs.rows,
      alpha: 0.45
    },
    {
      layer: 'raster',
      values: outputs.distances,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: [0, values.maximumDistance],
      bands: 12,
      fade: true,
      alpha: 0.8
    },
    {
      layer: 'dots',
      context: true,
      positions: outputs.seedPositions,
      count: Math.round(values.seedCount),
      radius: 5,
      color: [1, 1, 1, 1]
    }
  ]
});
