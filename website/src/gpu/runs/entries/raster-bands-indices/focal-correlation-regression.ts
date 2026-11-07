// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterFocalPairs: a moving-window regression of a response band on a predictor band. The
// response follows the predictor with a slope that changes sign across the map, so the local
// slope surface recovers that hidden relationship; correlation, R-squared and intercept are
// written too. Window radius and minimum pair count are construction props (the run rebuilds);
// noise only rewrites the response band.

import {GPURasterFocalPairs} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster, createSmoothField} from '../../inputs';
import {parameters} from './focal-correlation-regression.parameters';
import {drawRaster, makeGridOfRaster, uploadBand} from './support/bands';

const METRICS = ['slope', 'correlation', 'rSquared', 'intercept'] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['radius', 'minimumCount'],
  caption: 'Predictor and response bands in; the local regression of one on the other out.',
  createInputs: ({random, world}) => {
    const predictor = createFractalRaster(random, world, 56, 5);
    const relation = createSmoothField(random, 2, 2);
    const noise = Float32Array.from(predictor.values, () => random() - 0.5);
    const slope = Float32Array.from(predictor.values, (_, index) => {
      const row = Math.floor(index / predictor.columns);
      const column = index % predictor.columns;
      return 2 * relation((column + 0.5) / predictor.columns, (row + 0.5) / predictor.rows) - 1;
    });
    return {...predictor, noise, slope};
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const predictor = uploadBand(builder, inputs, 'predictor', inputs.values);
    const response = uploadBand(builder, inputs, 'response', inputs.values);
    const output = {
      pairedCount: builder.allocate('paired-count', 'uint32', cellCount),
      status: builder.allocate('status', 'uint32', cellCount),
      correlation: builder.allocate('correlation', 'float32', cellCount),
      slope: builder.allocate('slope', 'float32', cellCount),
      intercept: builder.allocate('intercept', 'float32', cellCount),
      rSquared: builder.allocate('r-squared', 'float32', cellCount)
    };
    builder.graph.add(
      new GPURasterFocalPairs({
        grid: makeGridOfRaster(inputs),
        predictor: predictor.band,
        response: response.band,
        radius: values.radius,
        minimumCount: values.minimumCount,
        output: {
          pairedCount: output.pairedCount.view,
          status: output.status.view,
          correlation: output.correlation.view,
          slope: output.slope.view,
          intercept: output.intercept.view,
          rSquared: output.rSquared.view
        }
      })
    );
    return {
      outputs: {...output, predictor: predictor.buffer},
      update: current =>
        response.buffer.write(
          Float32Array.from(
            inputs.values,
            (value, index) => value * inputs.slope[index] + inputs.noise[index] * current.noise
          )
        )
    };
  },
  draw: ({inputs, outputs, values}) => {
    const metric = METRICS[Math.round(values.metric)];
    return [
      drawRaster(inputs, outputs.predictor, {context: true}),
      drawRaster(inputs, outputs[metric], {
        ramp: metric === 'slope' || metric === 'correlation' ? 'diverging' : 'sequential',
        range: metric === 'slope' ? [-1.5, 1.5] : metric === 'correlation' ? [-1, 1] : 'auto',
        smoothing: true
      })
    ];
  }
});
