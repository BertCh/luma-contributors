// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSimpleLocalRelief on a Grand Canyon DEM: elevation minus its local mean within a window radius,
// which flattens the regional trend and leaves hills and hollows. The radius and exaggeration are
// per-frame values; maximumRadius bounds them at construction.

import {
  getGPUSimpleLocalReliefParameterValues,
  GPUSimpleLocalRelief
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {findParameter, getFractionOf, getParameterAt} from '../../interaction';

import {
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  getCellOfWorldPosition,
  getWorldPerCell,
  uploadTerrain
} from './support/terrain';
import {parameters} from './simple-local-relief.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Elevation in; local relief (above or below the neighborhood mean) out.',
  createInputs: ({world}) => createTerrainInput(world),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const result = builder.allocate('result', 'float32', cellCount);
    builder.graph.add(
      new GPUSimpleLocalRelief({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        maximumRadius: 20,
        output: {relief: result.view}
      })
    );
    return {
      outputs: {elevation, result},
      update: current =>
        parameterBuffer.write(
          getGPUSimpleLocalReliefParameterValues({
            radius: current.radius,
            verticalExaggeration: current.verticalExaggeration
          })
        )
    };
  },
  // The pointer height is the window radius (low is small); the ring is the window whose mean is
  // subtracted from the cell under the cursor.
  interaction: {
    kind: 'reach',
    hint: 'size the window',
    rest: [0.5, 1 - getFractionOf(findParameter(parameters, 'radius'), 8)],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const radius = getParameterAt(findParameter(parameters, 'radius'), 1 - pointer.unitY);
      const [column, row] = getCellOfWorldPosition(inputs, world, pointer.x, pointer.y);
      const relief = peek(outputs.result);
      const meters = Math.round(radius * inputs.cellSize);
      return {
        values: {radius},
        glyphs: [
          {
            shape: 'ring',
            at: [pointer.x, pointer.y],
            radius: radius * getWorldPerCell(inputs, world)
          },
          {shape: 'dot', at: [pointer.x, pointer.y]}
        ],
        readout: `radius ${radius} cells (${meters} m)${
          relief ? ` · relief here ${Math.round(relief[row * inputs.columns + column])} m` : ''
        }`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.result, {ramp: 'diverging', range: [-250, 250], smoothing: true})
  ]
});
