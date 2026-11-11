// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addDrainageNetworkRecipe: flow directions, accumulation and stream order from a terrain. The map
// shows flow accumulation (dim, logarithmic) and the stream network colored by Strahler order
// (brighter = higher order); lowering streamThreshold grows the network upstream.

import {addDrainageNetworkRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getGPUTerrainFlowParameterValues} from '@luma-contributors/analysis/gpu-terrain';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {getRasterCell} from '../../pick';
import {sampleElevation} from '../../../data/elevation';
import {addReadout} from './lib/readout';
import {parameters} from './drainage-network-recipe.parameters';

const ROWS = 80;
/** The pointer snaps to the strongest channel within this many cells. */
const REACH_CELLS = 3;

/** Largest accumulation within `REACH_CELLS` of a cell. */
function getStrongestNearby(
  accumulation: ArrayLike<number>,
  columns: number,
  column: number,
  row: number
): number {
  let strongest = 0;
  for (
    let nearRow = Math.max(0, row - REACH_CELLS);
    nearRow <= Math.min(ROWS - 1, row + REACH_CELLS);
    nearRow++
  ) {
    for (
      let nearColumn = Math.max(0, column - REACH_CELLS);
      nearColumn <= Math.min(columns - 1, column + REACH_CELLS);
      nearColumn++
    ) {
      strongest = Math.max(strongest, accumulation[nearRow * columns + nearColumn]);
    }
  }
  return strongest;
}

/** D8 flow accumulation on the CPU (no depression filling): the pointer's channel lookup. */
function getFlowAccumulation(elevation: ArrayLike<number>, columns: number): Float32Array {
  const accumulation = new Float32Array(columns * ROWS).fill(1);
  const order = Array.from({length: columns * ROWS}, (_, index) => index).sort(
    (first, second) => elevation[second] - elevation[first]
  );
  for (const cell of order) {
    const column = cell % columns;
    const row = Math.floor(cell / columns);
    let target = -1;
    let steepest = 0;
    for (let nearRow = Math.max(0, row - 1); nearRow <= Math.min(ROWS - 1, row + 1); nearRow++) {
      for (
        let nearColumn = Math.max(0, column - 1);
        nearColumn <= Math.min(columns - 1, column + 1);
        nearColumn++
      ) {
        const near = nearRow * columns + nearColumn;
        if (near === cell) continue;
        const drop =
          (elevation[cell] - elevation[near]) / Math.hypot(nearColumn - column, nearRow - row);
        if (drop > steepest) {
          steepest = drop;
          target = near;
        }
      }
    }
    if (target >= 0) accumulation[target] += accumulation[cell];
  }
  return accumulation;
}

export default defineOperationRun({
  parameters,
  caption: 'Drainage network from terrain: accumulation and stream order.',
  createInputs: ({world}) => {
    const columns = Math.round((ROWS * world.width) / world.height);
    const {elevation, cellSize} = sampleElevation(columns, ROWS);
    return {
      columns,
      source: elevation,
      accumulation: getFlowAccumulation(elevation, columns),
      grid: makeGPURasterMetadata({
        width: columns,
        height: ROWS,
        cellSize,
        origin: [0, ROWS * cellSize]
      })
    };
  },
  build(builder, {columns, grid, source}) {
    const cellCount = columns * ROWS;
    const elevation = builder.allocate('elevation', 'float32', cellCount);
    const parameterBuffer = builder.allocate('flow-parameters', 'float32', 8);
    const recipe = addDrainageNetworkRecipe(builder.graph, {
      flow: {
        grid,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {kind: 'buffer', values: elevation.view}
        },
        parameters: parameterBuffer.view,
        fillDepressions: true
      },
      maximumSteps: 512,
      reachCapacity: 4096,
      vertexCapacity: 16384
    });
    return {
      outputs: {
        accumulation: addReadout(builder, 'accumulation', recipe.outputs.accumulation),
        order: addReadout(builder, 'order', recipe.outputs.streamOrder)
      },
      update: current => {
        elevation.write(Float32Array.from(source));
        parameterBuffer.write(
          getGPUTerrainFlowParameterValues(
            {streamThreshold: current.streamThreshold, fillEpsilon: 0.02},
            grid
          )
        );
      }
    };
  },
  // The flow accumulation under the pointer is the stream threshold: the network grows until it
  // reaches the hovered channel.
  interaction: {
    kind: 'level',
    hint: 'pick a channel',
    rest: ({toUnit, inputs, world}) => {
      // The cell nearest the middle whose channel snaps to the default threshold (CPU inputs only).
      const parameter = findParameter(parameters, 'streamThreshold');
      let best: [number, number] = [world.width / 2, world.height / 2];
      let bestDistance = Infinity;
      for (let index = 0; index < inputs.accumulation.length; index++) {
        const column = index % inputs.columns;
        const row = Math.floor(index / inputs.columns);
        if (
          snapToParameter(
            parameter,
            getStrongestNearby(inputs.accumulation, inputs.columns, column, row)
          ) !== parameter.default
        ) {
          continue;
        }
        const x = (column + 0.5) * (world.width / inputs.columns);
        const y = world.height * (1 - (row + 0.5) / ROWS);
        const distance = Math.hypot(x - world.width / 2, y - world.height / 2);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = [x, y];
        }
      }
      return toUnit(best);
    },
    resolve(pointer, {inputs, world}) {
      const accumulation = inputs.accumulation;
      const cell = getRasterCell(
        inputs.columns,
        ROWS,
        [0, 0, world.width, world.height],
        pointer.x,
        pointer.y
      );
      const streamThreshold = snapToParameter(
        findParameter(parameters, 'streamThreshold'),
        getStrongestNearby(accumulation, inputs.columns, cell.column, cell.row)
      );
      return {
        values: {streamThreshold},
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: `streamThreshold ${streamThreshold} cells (channel of ${Math.round(getStrongestNearby(accumulation, inputs.columns, cell.column, cell.row))} cells)`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.accumulation,
      columns: inputs.columns,
      rows: ROWS,
      logarithmic: true,
      range: [1, 4000],
      alpha: 1
    },
    {
      layer: 'raster',
      values: outputs.order,
      columns: inputs.columns,
      rows: ROWS,
      ramp: 'sequential',
      range: [0, 4],
      fade: true
    }
  ]
});
