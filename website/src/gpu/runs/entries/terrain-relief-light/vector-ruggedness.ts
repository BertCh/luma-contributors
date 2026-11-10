// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainVectorRuggedness on a Grand Canyon DEM: dispersion of surface normals in a moving window.
// The radius and z factor are per-frame values, so the sliders never recompile.

import {
  getGPUTerrainVectorRuggednessParameterValues,
  GPUTerrainVectorRuggedness
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {findParameter, getFractionOf, getParameterAt} from '../../interaction';
import {getCellWindow} from './support/pointer';
import {
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  getCellOfWorldPosition,
  getCellWorldPosition,
  getWorldPerCell,
  uploadTerrain
} from './support/terrain';
import {parameters} from './vector-ruggedness.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Elevation in; the vector ruggedness measure out: orientation scatter, not height.',
  createInputs: ({world}) => createTerrainInput(world),
  build(builder, terrain) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const ruggedness = builder.allocate('ruggedness', 'float32', cellCount);
    builder.graph.add(
      new GPUTerrainVectorRuggedness({
        grid,
        elevation: band,
        maximumRadius: 7,
        parameters: parameterBuffer.view,
        output: {vectorRuggedness: ruggedness.view}
      })
    );
    return {
      outputs: {elevation, ruggedness},
      update: current =>
        parameterBuffer.write(
          getGPUTerrainVectorRuggednessParameterValues(
            {radius: current.radius, zFactor: current.zFactor},
            grid
          )
        )
    };
  },
  // The pointer height is the window radius (low is small); the box is the window of the cell
  // under the cursor in which the normals are compared.
  interaction: {
    kind: 'reach',
    hint: 'size the window',
    rest: [0.5, 1 - getFractionOf(findParameter(parameters, 'radius'), 3)],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const radius = getParameterAt(findParameter(parameters, 'radius'), 1 - pointer.unitY);
      const [column, row] = getCellOfWorldPosition(inputs, world, pointer.x, pointer.y);
      const window = getCellWindow(
        getCellWorldPosition(inputs, world, column, row),
        radius,
        getWorldPerCell(inputs, world)
      );
      const ruggedness = peek(outputs.ruggedness);
      return {
        values: {radius},
        glyphs: [{shape: 'box', ...window}],
        readout: `radius ${radius} cells (${2 * radius + 1}x${2 * radius + 1})${
          ruggedness ? ` · VRM ${ruggedness[row * inputs.columns + column].toFixed(3)}` : ''
        }`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.ruggedness, {range: [0, 0.2], smoothing: true})
  ]
});
