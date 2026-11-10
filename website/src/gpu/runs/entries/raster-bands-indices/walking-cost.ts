// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterWalkingCost: travel time on foot across rugged terrain from one source cell. Slope
// slows the walker (Tobler's hiking function or GRASS-style coefficients) and a friction raster
// (a river band here) slows them further. The cost surface is published once the relaxation has
// converged; too few `maximumIterations` leave it blank. Model, friction weight and iterations are
// construction props; the source cell is rewritten per frame and placed by the pointer.

import {
  GPURasterWalkingCost,
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {normalizeElevation, sampleElevation} from '../../../data/elevation';
import {clamp, findParameter} from '../../interaction';
import {parameters} from './walking-cost.parameters';
import {drawRaster} from './support/bands';
import {getCellAt, getCellBox, getCellCenter} from './support/gestures';

const SOURCE_X = findParameter(parameters, 'sourceX');

export default defineOperationRun({
  parameters,
  rebuildOn: ['model', 'frictionWeight', 'maximumIterations'],
  caption: 'Terrain, friction and a source in; the walking time to every cell out.',
  createInputs: ({world}) => {
    const rows = 44;
    const columns = Math.round((rows * world.width) / world.height);
    const {elevation, cellSize} = sampleElevation(columns, rows);
    // A valley-floor travel penalty illustrates the additional friction near the Colorado River.
    const friction = elevation.map(height => (height < 790 ? 4 : 1));
    return {
      columns,
      rows,
      cellSize,
      elevation,
      values: normalizeElevation(elevation),
      friction,
      // The source row is a pointer-owned input; the column is the sourceX parameter.
      sourceRow: Math.floor(rows / 2)
    };
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const grid = makeGPURasterMetadata({
      width: inputs.columns,
      height: inputs.rows,
      cellSize: inputs.cellSize
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
        data[inputs.sourceRow * inputs.columns + column] = 1;
        sources.write(data);
      }
    };
  },
  interaction: {
    kind: 'place',
    hint: 'place the walker',
    rest: ({inputs}) => [
      (Math.round(SOURCE_X.default * (inputs.columns - 1)) + 0.5) / inputs.columns,
      (Math.floor(inputs.rows / 2) + 0.5) / inputs.rows
    ],
    resolve: (pointer, {inputs, outputs, world, peek}) => {
      const {column, row} = getCellAt(inputs, world, pointer);
      inputs.sourceRow = row;
      const costs = peek(outputs.costs);
      let farthest = 0;
      if (costs)
        for (const cost of costs) if (Number.isFinite(cost)) farthest = Math.max(farthest, cost);
      return {
        values: {sourceX: clamp(column / (inputs.columns - 1), SOURCE_X.min, SOURCE_X.max)},
        rerun: true,
        glyphs: [
          getCellBox(inputs, world, column, row, 1),
          {shape: 'crosshair', at: getCellCenter(inputs, world, column, row)}
        ],
        readout: `source [${column}, ${row}] | farthest ${(farthest / 3600).toFixed(1)} h`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.terrain, {context: true, range: [0, 1]}),
    drawRaster(inputs, outputs.costs, {
      ramp: 'sequential',
      range: [0, 28800],
      bands: 12,
      alpha: 0.9
    })
  ]
});
