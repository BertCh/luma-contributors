// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addDrainageNetworkRecipe: flow directions, accumulation and stream order from a terrain. The map
// shows flow accumulation (dim, logarithmic) and the stream network colored by Strahler order
// (brighter = higher order); lowering streamThreshold grows the network upstream.

import {addDrainageNetworkRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getGPUTerrainFlowParameterValues} from '@luma-contributors/analysis/gpu-terrain';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createSmoothField} from '../../inputs';
import {addReadout} from './lib/readout';
import {parameters} from './drainage-network-recipe.parameters';

const ROWS = 80;
const CELL_SIZE = 60;

export default defineOperationRun({
  parameters,
  caption: 'Drainage network from terrain: accumulation and stream order.',
  createInputs: ({random, world}) => {
    const columns = Math.round((ROWS * world.width) / world.height);
    return {
      columns,
      grid: makeGPURasterMetadata({
        width: columns,
        height: ROWS,
        cellSize: CELL_SIZE,
        origin: [0, ROWS * CELL_SIZE]
      }),
      field: createSmoothField(random, 6, 3),
      tilt: random() * 2 - 1
    };
  },
  build(builder, {columns, grid, field, tilt}) {
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
        elevation.write(
          Float32Array.from({length: cellCount}, (_, index) => {
            const x = ((index % columns) + 0.5) / ROWS;
            const y = (Math.floor(index / columns) + 0.5) / ROWS;
            return (
              400 +
              900 * current.relief * field(x / (columns / ROWS), y) +
              250 * (1 - x / (columns / ROWS)) * (tilt + 1)
            );
          })
        );
        parameterBuffer.write(
          getGPUTerrainFlowParameterValues(
            {streamThreshold: current.streamThreshold, fillEpsilon: 0.02},
            grid
          )
        );
      }
    };
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
