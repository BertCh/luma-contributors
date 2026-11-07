// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUReliefBlend on a synthetic DEM: four terrain layers (hillshade, slope, positive openness and
// sky view factor), each stretched to [0, 1] and composited with its own blend mode and opacity
// into one greyscale relief image, the recipe of the Visualization for Archaeological Topography.
// The layer settings are a per-frame parameter buffer.

import {
  getGPUReliefBlendParameterValues,
  getGPUTerrainDerivativesParameterValues,
  GPUReliefBlend,
  GPUTerrainDerivatives,
  type GPUReliefBlendMode
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {addTerrainHorizon} from './support/horizon';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './relief-blend.parameters';

const BLEND_MODES: GPUReliefBlendMode[] = ['normal', 'multiply', 'luminosity', 'overlay'];

export default defineOperationRun({
  parameters,
  caption: 'Hillshade, slope, openness and sky view layers in; one blended relief image out.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
  build(builder, terrain) {
    const uploaded = uploadTerrain(builder, terrain);
    const {grid, band, elevation, cellCount} = uploaded;
    const derivativeParameters = builder.allocate('derivative-parameters', 'float32', 8);
    derivativeParameters.write(getGPUTerrainDerivativesParameterValues({zFactor: 2}, grid));
    const slope = builder.allocate('slope', 'float32', cellCount);
    const hillshade = builder.allocate('hillshade', 'float32', cellCount);
    builder.graph.add(
      new GPUTerrainDerivatives({
        grid,
        elevation: band,
        parameters: derivativeParameters.view,
        output: {slope: slope.view, hillshade: hillshade.view}
      })
    );
    const {skyViewFactor, positiveOpenness} = addTerrainHorizon(builder, uploaded, {
      directionCount: 16,
      maximumRadius: 30
    });
    const parameterBuffer = builder.allocate('parameters', 'float32', 20);
    const blend = builder.allocate('blend', 'float32', cellCount);
    builder.graph.add(
      new GPUReliefBlend({
        grid,
        layers: [hillshade.view, slope.view, positiveOpenness.view, skyViewFactor.view],
        parameters: parameterBuffer.view,
        output: {blend: blend.view}
      })
    );
    return {
      outputs: {elevation, blend},
      update: current =>
        parameterBuffer.write(
          getGPUReliefBlendParameterValues([
            {minimum: 0, maximum: 1, blendMode: 'normal', opacity: 1},
            {
              minimum: 0,
              maximum: 50,
              invert: true,
              blendMode: BLEND_MODES[current.slopeBlendMode],
              opacity: current.slopeOpacity
            },
            {minimum: 68, maximum: 93, blendMode: 'overlay', opacity: current.opennessOpacity},
            {minimum: 0.7, maximum: 1, blendMode: 'multiply', opacity: current.skyViewOpacity}
          ])
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.blend, {range: [0, 1], smoothing: true})
  ]
});
