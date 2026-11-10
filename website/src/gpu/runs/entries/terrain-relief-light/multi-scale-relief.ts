// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMultiScaleRelief on a Grand Canyon DEM: local relief averaged over a geometric series of window radii
// between featureMinimum and featureMaximum. The feature range and scaling factor set the radii at
// construction; the exaggeration is a per-frame value.

import {
  getGPUMultiScaleReliefParameterValues,
  GPUMultiScaleRelief
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './multi-scale-relief.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['featureMaximum', 'scalingFactor'],
  caption: 'Elevation in; relief averaged over many window sizes out.',
  createInputs: ({world}) => createTerrainInput(world),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const result = builder.allocate('result', 'float32', cellCount);
    builder.graph.add(
      new GPUMultiScaleRelief({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        resolution: 1,
        featureMinimum: 2,
        featureMaximum: values.featureMaximum,
        scalingFactor: values.scalingFactor,
        output: {relief: result.view}
      })
    );
    return {
      outputs: {elevation, result},
      update: current =>
        parameterBuffer.write(
          getGPUMultiScaleReliefParameterValues({
            verticalExaggeration: current.verticalExaggeration
          })
        )
    };
  },
  // The pointer is a curtain: the hillshaded elevation on the left, the multi-scale relief on the right.
  interaction: {
    kind: 'swipe',
    hint: 'swipe elevation and relief',
    rest: [0.5, 0.5],
    resolve: (pointer, {values, outputs, peek}) => ({
      curtain: pointer.x,
      glyphs: [{shape: 'curtain', x: pointer.x}],
      readout: `elevation | relief over 2-${values.featureMaximum} cells`
    })
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = interaction.curtain ?? world.width / 2;
    return [
      drawGrid(inputs, outputs.elevation, {
        range: [650, 2300],
        hillshade: 0.9,
        elevation: 50,
        clip: {side: 'left', x: curtain}
      }),
      drawGrid(inputs, outputs.result, {
        ramp: 'diverging',
        range: [-80, 80],
        smoothing: true,
        clip: {side: 'right', x: curtain}
      })
    ];
  }
});
