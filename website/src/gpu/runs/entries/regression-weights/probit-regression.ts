// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUProbitRegression of a binary outcome on two predictors. Left: the fitted probability of every
// tract. Right: the estimated coefficients (bars) against the latent-scale truth (dots), and the
// marginal effects below. A large latent noise flattens the probabilities and shrinks the slopes.

import {GPUProbitRegression} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';

import {mergeText} from '../../text';
import {parameters} from './probit-regression.parameters';
import {
  COEFFICIENT_RANGE,
  createCoefficientPanel,
  createRegressionInputs,
  drawCoefficientPanel,
  drawSignedBars,
  drawTractMap,
  drawZeroLine,
  MAP_SHARE,
  uploadRegressionMap
} from './shared/regression';

const LATENT_COEFFICIENTS = [-1.2, 2.4, -1.8] as const;
// Fisher scoring has settled after about three steps, so the early steps get most of the width:
// [start of the band across the canvas, iteration count], the last band being the converged fit.
const ITERATION_BANDS = [
  [0, 1],
  [0.3, 2],
  [0.55, 3],
  [0.8, 8]
] as const;
const MARGINAL_RECTANGLE = [MAP_SHARE + 0.08, 0.08, 0.97, 0.34] as const;

export default defineOperationRun({
  parameters,
  caption: 'Fitted probabilities (left) and probit coefficients against the truth (right).',
  createInputs: ({random, world}) => createRegressionInputs(random, world),
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const coefficients = builder.allocate('coefficients', 'float32', 3);
    const fitted = builder.allocate('fitted', 'float32', rowCount);
    const marginalEffects = builder.allocate('marginal-effects', 'float32', 2);
    builder.graph.add(
      new GPUProbitRegression({
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        maximumIterations: values.maximumIterations,
        output: {
          coefficients: coefficients.view,
          standardErrors: builder.allocate('standard-errors', 'float32', 3).view,
          fitted: fitted.view,
          marginalEffects: marginalEffects.view,
          summary: builder.allocate('summary', 'float32', 4).view,
          status: builder.allocate('status', 'uint32', 1).view
        }
      })
    );
    const panel = createCoefficientPanel(builder, coefficients, 1, LATENT_COEFFICIENTS, [
      'int',
      'x1',
      'x2'
    ]);
    return {
      outputs: {
        mesh: uploadRegressionMap(builder, inputs),
        fitted,
        response,
        marginalEffects,
        panel
      },
      update: current => {
        response.write(
          Float32Array.from({length: rowCount}, (_, tract) => {
            const latent =
              LATENT_COEFFICIENTS[0] +
              LATENT_COEFFICIENTS[1] * inputs.predictors[tract * 2] +
              LATENT_COEFFICIENTS[2] * inputs.predictors[tract * 2 + 1] +
              inputs.normals[tract] * current.noise;
            return latent > 0 ? 1 : 0;
          })
        );
        panel.truth.write(Float32Array.from(LATENT_COEFFICIENTS, value => value / current.noise));
      }
    };
  },
  rebuildOn: ['maximumIterations'],
  buildCache: 5,
  // The pointer along x steps through Fisher scoring: the estimates settle on one of four iteration counts and
  // the bars converge on the dots.
  interaction: {
    kind: 'time',
    hint: 'step the iterations',
    rest: [0.9, 0.5],
    resolve(pointer, {outputs, peek, values}) {
      const iterations = ITERATION_BANDS.reduce(
        (chosen, [start, count]) => (pointer.unitX >= start ? count : chosen),
        1
      );
      const estimates = peek(outputs.panel.estimates);
      const x = pointer.x;
      return {
        settle: {maximumIterations: iterations},
        glyphs: [{shape: 'curtain', x}],
        readout: `maximumIterations ${iterations} · x1 ${estimates ? estimates[1].toFixed(2) : '-'} (true ${(LATENT_COEFFICIENTS[1] / values.noise).toFixed(2)})`
      };
    }
  },
  draw: ({inputs, outputs, world, theme}) => {
    const titles = mergeText({
      labels: [
        {
          text: 'fitted probability of 1',
          at: [inputs.mapWidth / 2, world.height],
          offset: [0, -12],
          role: 'label',
          priority: 4
        },
        {
          text: 'coefficients (dot: latent truth)',
          at: [(MAP_SHARE + 0.08) * world.width, 0.9 * world.height],
          anchor: 'bottom-left',
          offset: [0, 3],
          role: 'label',
          priority: 4
        },
        {
          text: 'marginal effects',
          at: [(MAP_SHARE + 0.08) * world.width, MARGINAL_RECTANGLE[3] * world.height],
          anchor: 'bottom-left',
          offset: [0, 3],
          role: 'label',
          priority: 4
        }
      ]
    });
    return [
      ...drawTractMap(inputs, outputs.mesh, outputs.fitted, {ramp: 'sequential', range: [0, 1]}),
      // The observed 0/1 outcomes at the tract centers, on the fitted probability's scale.
      {
        layer: 'dots',
        positions: outputs.mesh.centers,
        values: outputs.response,
        ramp: 'sequential',
        range: [0, 1],
        radius: 3.2
      },
      ...drawCoefficientPanel(outputs.panel, {
        rectangle: [MAP_SHARE + 0.08, 0.42, 0.97, 0.9],
        world,
        ink: theme.ink
      }),
      ...drawSignedBars(outputs.marginalEffects, MARGINAL_RECTANGLE, COEFFICIENT_RANGE[1] / 4),
      drawZeroLine(outputs.panel.zeros, MARGINAL_RECTANGLE),
      {layer: 'text', ...titles}
    ];
  }
});
