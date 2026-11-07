// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineIntegralConvolution on a seeded swirl field: every output pixel averages white noise along
// its streamline for stepCount steps each way, so the noise smears into the flow. The raster it
// wrote (values) is drawn as light streaks over the field speed (speeds, also its output).

import {
  getGPULineIntegralConvolutionParameterValues,
  getGPULineIntegralConvolutionUint32ParameterValues,
  GPULineIntegralConvolution,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './line-integral-convolution.parameters';
import {createSwirlField} from './shared/vector-field';

const OUTPUT_SCALE = 2;

export default defineOperationRun({
  parameters,
  caption: 'White noise smeared along the flow reveals eddies and saddles.',
  createInputs: context => createSwirlField(context),
  build(builder, field, current) {
    const outputColumns = field.columns * OUTPUT_SCALE;
    const outputRows = field.rows * OUTPUT_SCALE;
    const velocities = builder.upload('velocities', 'float32x2', field.velocities);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const wordBuffer = builder.allocate('words', 'uint32', 4);
    const values = builder.allocate('values', 'float32', outputColumns * outputRows);
    const speeds = builder.allocate('speeds', 'float32', outputColumns * outputRows);
    builder.graph.add(
      new GPULineIntegralConvolution({
        velocities: velocities.view,
        fieldWidth: field.columns,
        fieldHeight: field.rows,
        grid: makeGPURasterMetadata({width: outputColumns, height: outputRows}),
        stepCount: Math.round(current.stepCount),
        parameters: parameterBuffer.view,
        uint32Parameters: wordBuffer.view,
        output: {values: values.view, speeds: speeds.view}
      })
    );
    return {
      outputs: {values, speeds},
      update: next => {
        const settings = {
          fieldExtent: [0, 0, field.cellSize, field.cellSize] as const,
          outputExtent: [
            0,
            0,
            field.cellSize / OUTPUT_SCALE,
            field.cellSize / OUTPUT_SCALE
          ] as const,
          stepLength: next.stepLength,
          seed: Math.round(next.seed)
        };
        parameterBuffer.write(getGPULineIntegralConvolutionParameterValues(settings));
        wordBuffer.write(getGPULineIntegralConvolutionUint32ParameterValues(settings));
      }
    };
  },
  rebuildOn: ['stepCount'],
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.speeds,
      columns: inputs.columns * OUTPUT_SCALE,
      rows: inputs.rows * OUTPUT_SCALE,
      smoothing: true
    },
    {
      layer: 'raster',
      values: outputs.values,
      columns: inputs.columns * OUTPUT_SCALE,
      rows: inputs.rows * OUTPUT_SCALE,
      ramp: 'sequential',
      range: [0.2, 0.8],
      alpha: 0.75
    }
  ]
});
