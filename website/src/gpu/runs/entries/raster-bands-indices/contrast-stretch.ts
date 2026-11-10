// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterStretch on a dull, low-contrast raster with a few hot pixels. Linear stretch is held
// back by the outliers, percentile stretch clips them, equalize flattens the histogram. The
// stretch curve (lut) is plotted against the identity line, and the dull input stays visible in
// the left strip. All settings are per-frame parameters.

import {
  getGPURasterStretchParameterValues,
  GPURasterStretch,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createElevationRaster} from '../../inputs';
import {findParameter, getFractionOf, getParameterAt, snapToParameter} from '../../interaction';
import {parameters} from './contrast-stretch.parameters';
import {drawRaster} from './support/bands';

const MODES = ['linear', 'percentile', 'equalize'] as const;
const LUT_SIZE = 128;
const STRIP = 0.16;

const lowParameter = findParameter(parameters, 'percentileLow');
const highParameter = findParameter(parameters, 'percentileHigh');
const gammaParameter = findParameter(parameters, 'gamma');

export default defineOperationRun({
  parameters,
  caption:
    'A flat-looking raster in (left strip); the contrast-stretched raster and its curve against the identity line out.',
  createInputs: ({random, world}) => {
    const raster = createElevationRaster(world, 64);
    const values = raster.values.map(value => 0.3 + 0.25 * value);
    for (let hot = 0; hot < 12; hot++) values[Math.floor(random() * values.length)] = 1;
    return {...raster, dull: values};
  },
  build(builder, inputs) {
    const source = builder.upload('source', 'float32', inputs.dull);
    const parameterBuffer = builder.allocate('parameters', 'float32', 16);
    const stretched = builder.allocate('stretched', 'float32', inputs.dull.length);
    const lut = builder.allocate('lut', 'float32', LUT_SIZE);
    const identity = builder.upload(
      'identity',
      'float32',
      Float32Array.from({length: LUT_SIZE}, (_, index) => index / (LUT_SIZE - 1))
    );
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
      outputs: {source, stretched, lut, identity},
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
  // The pointer sets the clip percentiles symmetrically (x: the tail cut from each end) and the
  // gamma (y: up brightens), the stretch's own props.
  interaction: {
    kind: 'level',
    hint: 'clip tails and gamma',
    rest: () => [
      getFractionOf(lowParameter, lowParameter.default),
      1 - getFractionOf(gammaParameter, gammaParameter.default, 'logarithmic')
    ],
    resolve: (pointer, {world, values}) => {
      const tail = getParameterAt(lowParameter, pointer.unitX);
      const gamma = getParameterAt(gammaParameter, 1 - pointer.unitY, 'logarithmic');
      const high = snapToParameter(highParameter, 100 - tail);
      return {
        values: {percentileLow: tail, percentileHigh: high, gamma},
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: `percentiles [${tail}, ${high}] | gamma ${gamma.toFixed(1)}${values.mode === 1 ? '' : ' (percentile mode only)'}`
      };
    }
  },
  draw: ({inputs, outputs, world}) => {
    const split = world.width * STRIP;
    return [
      drawRaster(inputs, outputs.source, {range: [0, 1], clip: {side: 'left', x: split}}),
      drawRaster(inputs, outputs.stretched, {
        range: [0, 1],
        ramp: 'sequential',
        clip: {side: 'right', x: split}
      }),
      {
        layer: 'chart',
        kind: 'line',
        context: true,
        values: outputs.identity,
        range: [0, 1],
        rectangle: [0.62, 0.05, 0.96, 0.4]
      },
      {
        layer: 'chart',
        kind: 'line',
        values: outputs.lut,
        range: [0, 1],
        rectangle: [0.62, 0.05, 0.96, 0.4]
      }
    ];
  }
});
