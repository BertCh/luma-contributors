// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalDominance on a Grand Canyon DEM: how much an observer standing on each cell looks down on the
// ground within a ring of radii (archaeological relief visualization). The observer height is a
// per-frame value; the radii are construction props.

import {
  getGPULocalDominanceParameterValues,
  GPULocalDominance
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  getCellOfWorldPosition,
  getWorldPerCell,
  uploadTerrain
} from './support/terrain';
import {parameters} from './local-dominance.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['minimumRadius', 'maximumRadius'],
  caption: 'Elevation in; the local dominance of every cell over its surroundings out.',
  createInputs: ({world}) => createTerrainInput(world),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const result = builder.allocate('result', 'float32', cellCount);
    builder.graph.add(
      new GPULocalDominance({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        minimumRadius: values.minimumRadius,
        maximumRadius: values.maximumRadius,
        radiusIncrement: 2,
        angularResolution: 20,
        output: {dominance: result.view}
      })
    );
    return {
      outputs: {elevation, result},
      update: current =>
        parameterBuffer.write(
          getGPULocalDominanceParameterValues({observerHeight: current.observerHeight})
        )
    };
  },
  // The pointer is a cell: the annulus is the ring of ground it is compared against.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const [column, row] = getCellOfWorldPosition(inputs, world, pointer.x, pointer.y);
      const worldPerCell = getWorldPerCell(inputs, world);
      const dominance = peek(outputs.result);
      const at: [number, number] = [pointer.x, pointer.y];
      return {
        glyphs: [
          {shape: 'ring', at, radius: values.minimumRadius * worldPerCell, dashed: true},
          {shape: 'ring', at, radius: values.maximumRadius * worldPerCell},
          {shape: 'dot', at}
        ],
        readout: `${dominance ? `dominance ${dominance[row * inputs.columns + column].toFixed(2)} · ` : ''}ring ${values.minimumRadius}-${values.maximumRadius} cells`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.result, {range: [0, 30], smoothing: true})
  ]
});
