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
import {getRainStations} from '../../../data/point-observations';
import {clamp} from '../../interaction';
import {parameters} from './distance-field.parameters';

const SEED_CAPACITY = 12;
const ROWS = 60;

export default defineOperationRun({
  parameters,
  caption: 'Distance to NOAA rain gauges across the Carolinas during Hurricane Helene.',
  createInputs: ({random, world}) => {
    const columns = Math.round(ROWS * world.width);
    const seeds = Float32Array.from(getRainStations(random, world, SEED_CAPACITY, true).positions);
    const cellSize = world.height / ROWS;
    return {
      columns,
      rows: ROWS,
      cellSize,
      seeds,
      defaultSeed: [seeds[0], seeds[1]] as const,
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
        // Seed 0 is the station the pointer drags.
        seedPositions.write(inputs.seeds);
        seedCount.write(Uint32Array.of(Math.round(next.seedCount)));
        parameterBuffer.write(
          getGPUDistanceFieldParameterValues({maximumDistance: next.maximumDistance}, inputs.grid)
        );
      }
    };
  },
  interaction: {
    kind: 'drag',
    hint: 'drag a station',
    rest: ({inputs, toUnit}) => toUnit(inputs.defaultSeed),
    resolve(pointer, {inputs, values, world}) {
      const x = Math.fround(clamp(pointer.x, 0, world.width));
      const y = Math.fround(clamp(pointer.y, 0, world.height));
      const moved = inputs.seeds[0] !== x || inputs.seeds[1] !== y;
      inputs.seeds[0] = x;
      inputs.seeds[1] = y;
      return {
        rerun: moved,
        glyphs: [
          {shape: 'ring', at: [x, y], radius: values.maximumDistance, dashed: true},
          {shape: 'crosshair', at: [x, y]}
        ],
        readout: `station 1 at (${x.toFixed(2)}, ${y.toFixed(2)}) · maximumDistance ${values.maximumDistance}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      values: outputs.allocation,
      columns: inputs.columns,
      rows: inputs.rows,
      alpha: 0.3
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
