// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainRuggedness on a Grand Canyon DEM: the 3x3 ruggedness index, roughness (max minus min) or
// topographic position index per cell. All three props are construction props, so changes rebuild.

import {
  GPUTerrainRuggedness,
  type GPUTerrainRuggednessOutput
} from '@luma-contributors/analysis/gpu-terrain';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {defineOperationRun} from '../../define-run';
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
import {parameters} from './terrain-ruggedness.parameters';

// Meters over a 3x3 window: ruggedness index, roughness (max minus min), position index.
const RUGGEDNESS_RANGES = [
  [0, 300],
  [0, 300],
  [-40, 40]
] as const;

/** One selected ruggedness output; the class requires at least one named output. */
function getRuggednessOutput(
  key: keyof GPUTerrainRuggednessOutput,
  view: GraphDataView<'float32'>
): GPUTerrainRuggednessOutput {
  switch (key) {
    case 'terrainRuggednessIndex':
      return {terrainRuggednessIndex: view};
    case 'roughness':
      return {roughness: view};
    case 'topographicPositionIndex':
      return {topographicPositionIndex: view};
    default:
      throw new Error(`terrain-ruggedness run: unsupported output ${String(key)}`);
  }
}

export default defineOperationRun({
  parameters,
  caption: 'Elevation in; local ruggedness out: bright where the surface is broken up.',
  createInputs: ({world}) => createTerrainInput(world),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const result = builder.allocate('result', 'float32', cellCount);
    const outputKeys: (keyof GPUTerrainRuggednessOutput)[] = [
      'terrainRuggednessIndex',
      'roughness',
      'topographicPositionIndex'
    ];
    builder.graph.add(
      new GPUTerrainRuggedness({
        grid,
        elevation: band,
        terrainRuggednessAlgorithm: values.terrainRuggednessAlgorithm === 1 ? 'wilson' : 'riley',
        edgeMode: values.edgeMode === 1 ? 'extrapolate' : 'nodata',
        output: getRuggednessOutput(outputKeys[values.output], result.view)
      })
    );
    return {outputs: {elevation, result}};
  },
  // The pointer is a cell: the box is the 3x3 window the index is computed over.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const [column, row] = getCellOfWorldPosition(inputs, world, pointer.x, pointer.y);
      const window = getCellWindow(
        getCellWorldPosition(inputs, world, column, row),
        1,
        getWorldPerCell(inputs, world)
      );
      const result = peek(outputs.result);
      const name = ['TRI', 'roughness', 'TPI'][values.output];
      return {
        glyphs: [{shape: 'box', ...window}],
        readout: `${name} ${result ? Math.round(result[row * inputs.columns + column]) : '...'} m · ${
          ['riley', 'wilson'][values.terrainRuggednessAlgorithm]
        }`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.result, {
      ramp: values.output === 2 ? 'diverging' : 'sequential',
      range: RUGGEDNESS_RANGES[values.output],
      smoothing: true
    })
  ]
});
