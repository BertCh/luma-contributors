// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUWeightedOverlay: suitability from three layers (green, shortwave infrared, near-infrared), each
// remapped linearly to [0, 1] between its own 5th and 95th percentile and summed with the sliders' weights. Weights are per-frame
// parameters; negative weights penalize a layer.

import {
  getGPUWeightedOverlayParameterLength,
  getGPUWeightedOverlayParameterValues,
  GPUWeightedOverlay
} from '@luma-contributors/analysis/raster-analysis';
import {bindAxes} from '../../interaction';
import {defineOperationRun} from '../../define-run';
import {createImageryBands} from '../../inputs';
import {parameters} from './weighted-overlay.parameters';
import {drawRaster} from './support/bands';

const LAYER_COUNT = 3;

/** The 5th and 95th percentile of a band, so each layer's remap spans its own spread. */
function getSpread(values: Float32Array): [number, number] {
  const sorted = Float32Array.from(values).sort();
  return [sorted[Math.floor(sorted.length * 0.05)], sorted[Math.floor(sorted.length * 0.95)]];
}

const PAD = bindAxes(parameters, {
  kind: 'pad',
  hint: 'weight NIR and SWIR',
  x: 'nirWeight',
  y: 'shortwaveWeight'
});

export default defineOperationRun({
  parameters,
  caption: 'Three layers in; the weighted suitability score out.',
  createInputs: ({world}) => {
    const imagery = createImageryBands(world);
    return {
      ...imagery,
      spreads: [
        getSpread(imagery.green),
        getSpread(imagery.shortwaveInfrared),
        getSpread(imagery.nearInfrared)
      ]
    };
  },
  build(builder, imagery) {
    const {spreads} = imagery;
    const cellCount = imagery.values.length;
    const stackData = new Float32Array(cellCount * LAYER_COUNT);
    stackData.set(imagery.green, 0);
    stackData.set(imagery.shortwaveInfrared, cellCount);
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
    const shortwave = builder.upload('shortwave', 'float32', imagery.shortwaveInfrared);
    return {
      outputs: {shortwave, score},
      update: current =>
        parameterBuffer.write(
          getGPUWeightedOverlayParameterValues({
            layers: [
              {
                weight: current.greenWeight,
                inputMinimum: spreads[0][0],
                inputMaximum: spreads[0][1]
              },
              {
                weight: current.shortwaveWeight,
                inputMinimum: spreads[1][0],
                inputMaximum: spreads[1][1]
              },
              {weight: current.nirWeight, inputMinimum: spreads[2][0], inputMaximum: spreads[2][1]}
            ],
            normalizeWeights: current.normalizeWeights === 1
          })
        )
    };
  },
  // The pointer is a pad over two of the three layer weights: NIR across, SWIR up.
  interaction: {
    ...PAD,
    resolve: (pointer, state) => {
      const base = PAD.resolve(pointer, state);
      const current = {...state.values, ...base.values};
      return {
        ...base,
        ...{
          glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
          readout: `NIR ${current.nirWeight.toFixed(2)} | SWIR ${current.shortwaveWeight.toFixed(2)}`
        }
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.shortwave, {context: true}),
    drawRaster(inputs, outputs.score, {ramp: 'sequential', range: [-0.5, 1], smoothing: true})
  ]
});
