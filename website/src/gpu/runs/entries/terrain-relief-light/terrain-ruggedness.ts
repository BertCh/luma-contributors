// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainRuggedness on a synthetic DEM: the 3x3 ruggedness index, roughness (max minus min) or
// topographic position index per cell. All three props are construction props, so changes rebuild.

import {
  GPUTerrainRuggedness,
  type GPUTerrainRuggednessOutput
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './terrain-ruggedness.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Elevation in; local ruggedness out: bright where the surface is broken up.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
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
        output: {[outputKeys[values.output]]: result.view}
      })
    );
    return {outputs: {elevation, result}};
  },
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.result, {
      ramp: values.output === 2 ? 'diverging' : 'sequential',
      smoothing: true
    })
  ]
});
