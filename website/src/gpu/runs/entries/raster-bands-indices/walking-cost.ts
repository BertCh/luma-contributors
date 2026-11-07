// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterWalkingCost: travel time on foot across rugged terrain from one source cell. Slope
// slows the walker (Tobler's hiking function or GRASS-style coefficients) and a friction raster
// (a river band here) slows them further. The cost surface is published once the relaxation has
// converged; too few `maximumIterations` leave it blank. Model, friction weight and iterations are
// construction props; the source cell is rewritten per frame.

import {
  GPURasterWalkingCost,
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './walking-cost.parameters';
import {drawRaster} from './support/bands';

const CELL_SIZE = 30;

export default defineOperationRun({
  parameters,
  rebuildOn: ['model', 'frictionWeight', 'maximumIterations'],
  caption: 'Terrain, friction and a source in; the walking time to every cell out.',
  createInputs: ({random, world}) => {
    const raster = createFractalRaster(random, world, 44, 5);
    // Elevation in meters, and a river band (friction 4) crossing the map.
    const elevation = raster.values.map(value => value * 700);
    const friction = Float32Array.from(raster.values, (_, index) => {
      const column = (index % raster.columns) / raster.columns;
      const row = Math.floor(index / raster.columns) / raster.rows;
      return Math.abs(row - (0.5 + 0.18 * Math.sin(column * 7))) < 0.04 ? 4 : 1;
    });
    return {...raster, elevation, friction};
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const grid = makeGPURasterMetadata({
      width: inputs.columns,
      height: inputs.rows,
      cellSize: CELL_SIZE
    });
    const elevation = builder.upload('elevation', 'float32', inputs.elevation);
    const friction = builder.upload('friction', 'float32', inputs.friction);
    const sources = builder.allocate('sources', 'uint32', cellCount);
    const costs = builder.allocate('costs', 'float32', cellCount);
    const validity = builder.allocate('validity', 'uint32', cellCount);
    const converged = builder.allocate('converged', 'uint32', 1);
    builder.graph.add(
      new GPURasterWalkingCost({
        grid,
        elevation: makeGPURasterBand(elevation.view, grid, {id: 'elevation'}),
        friction: makeGPURasterBand(friction.view, grid, {id: 'friction'}),
        sources: sources.view,
        frictionWeight: values.frictionWeight,
        model:
          values.model === 1
            ? {kind: 'grass', coefficients: [0.8, 1.2, 0.5, 2], downhillSlopeThreshold: -0.1}
            : {kind: 'tobler'},
        maximumIterations: values.maximumIterations,
        output: {costs: costs.view, validity: validity.view, status: {converged: converged.view}}
      })
    );
    const terrain = builder.upload('terrain', 'float32', inputs.values);
    return {
      outputs: {terrain, costs},
      update: current => {
        const data = new Uint32Array(cellCount);
        const column = Math.round(current.sourceX * (inputs.columns - 1));
        data[Math.floor(inputs.rows / 2) * inputs.columns + column] = 1;
        sources.write(data);
      }
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.terrain, {context: true, range: [0, 1]}),
    drawRaster(inputs, outputs.costs, {ramp: 'sequential', bands: 12, alpha: 0.9})
  ]
});
