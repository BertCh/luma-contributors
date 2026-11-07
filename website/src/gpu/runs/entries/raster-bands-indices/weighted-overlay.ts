// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUWeightedOverlay: suitability from three layers (greenness, elevation, near-infrared), each
// remapped linearly to [0, 1] and summed with the sliders' weights. Weights are per-frame
// parameters; negative weights penalize a layer.

import {
  getGPUWeightedOverlayParameterLength,
  getGPUWeightedOverlayParameterValues,
  GPUWeightedOverlay
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands} from '../../inputs';
import {parameters} from './weighted-overlay.parameters';
import {drawRaster} from './support/bands';

const LAYER_COUNT = 3;

export default defineOperationRun({
  parameters,
  caption: 'Three layers in; the weighted suitability score out.',
  createInputs: ({random, world}) => createImageryBands(random, world),
  build(builder, imagery) {
    const cellCount = imagery.values.length;
    const stackData = new Float32Array(cellCount * LAYER_COUNT);
    stackData.set(imagery.green, 0);
    stackData.set(imagery.values, cellCount);
    stackData.set(imagery.nearInfrared, cellCount * 2);
    const stack = builder.upload('stack', 'float32', stackData);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      getGPUWeightedOverlayParameterLength(LAYER_COUNT)
    );
    const score = builder.allocate('score', 'float32', cellCount);
    builder.graph.add(
      new GPUWeightedOverlay({
        stack: stack.view,
        layerCount: LAYER_COUNT,
        cellCount,
        parameters: parameterBuffer.view,
        output: {score: score.view}
      })
    );
    const elevation = builder.upload('elevation', 'float32', imagery.values);
    return {
      outputs: {elevation, score},
      update: current =>
        parameterBuffer.write(
          getGPUWeightedOverlayParameterValues({
            layers: [
              {weight: current.greenWeight, inputMinimum: 0, inputMaximum: 0.2},
              {weight: current.elevationWeight},
              {weight: current.nirWeight, inputMinimum: 0, inputMaximum: 0.7}
            ],
            normalizeWeights: current.normalizeWeights === 1
          })
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.elevation, {context: true}),
    drawRaster(inputs, outputs.score, {ramp: 'diverging', range: [-1, 1], smoothing: true})
  ]
});
