// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterStretch on a dull, low-contrast raster with a few hot pixels. Linear stretch is held
// back by the outliers, percentile stretch clips them, equalize flattens the histogram. The
// stretch curve (lut) is plotted as a line. All settings are per-frame parameters.

import {
  getGPURasterStretchParameterValues,
  GPURasterStretch,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './contrast-stretch.parameters';
import {drawRaster} from './support/bands';

const MODES = ['linear', 'percentile', 'equalize'] as const;
const LUT_SIZE = 128;

export default defineOperationRun({
  parameters,
  caption: 'A flat-looking raster in; the contrast-stretched raster and its curve out.',
  createInputs: ({random, world}) => {
    const raster = createFractalRaster(random, world, 64);
    const values = raster.values.map(value => 0.3 + 0.25 * value);
    for (let hot = 0; hot < 12; hot++) values[Math.floor(random() * values.length)] = 1;
    return {...raster, dull: values};
  },
  build(builder, inputs) {
    const source = builder.upload('source', 'float32', inputs.dull);
    const parameterBuffer = builder.allocate('parameters', 'float32', 16);
    const stretched = builder.allocate('stretched', 'float32', inputs.dull.length);
    const lut = builder.allocate('lut', 'float32', LUT_SIZE);
    builder.graph.add(
      new GPURasterStretch({
        values: source.view,
        grid: makeGPURasterMetadata({width: inputs.columns, height: inputs.rows}),
        parameters: parameterBuffer.view,
        binCount: 256,
        lutSize: LUT_SIZE,
        output: {stretched: stretched.view, lut: lut.view}
      })
    );
    return {
      outputs: {source, stretched, lut},
      update: current =>
        parameterBuffer.write(
          getGPURasterStretchParameterValues({
            mode: MODES[Math.round(current.mode)],
            percentiles: [
              current.percentileLow,
              Math.max(current.percentileHigh, current.percentileLow)
            ],
            gamma: current.gamma
          })
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.source, {context: true, range: [0, 1]}),
    drawRaster(inputs, outputs.stretched, {range: [0, 1], ramp: 'sequential'}),
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.lut,
      range: [0, 1],
      rectangle: [0.04, 0.04, 0.34, 0.3]
    }
  ]
});
