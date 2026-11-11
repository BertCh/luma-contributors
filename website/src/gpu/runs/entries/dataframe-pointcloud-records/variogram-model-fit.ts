// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUVariogramModelFit on the bins of a GPUVariogram. The dots are the empirical semivariances;
// the three curves are the spherical (cyan), exponential (orange) and gaussian (white) models the
// GPU fitted, evaluated from the fit's parameter rows.

import {
  GPUVariogram,
  GPUVariogramModelFit,
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  getGPUVariogramParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {addFittedCurve} from './_lib/fitted-curve';
import {
  createPairProbe,
  DISTANCE_PLOT,
  getAxisFraction,
  getAxisPointer,
  getBinIndex,
  getChartWorldY
} from './_lib/distance-probe';
import {createFieldSample, getFieldValues} from './_lib/helpers';
import {parameters} from './variogram-model-fit.parameters';

const POINT_COUNT = 700;
const LAG_COUNT = 20;
const CURVE_COUNT = 48;
const PAIR_LIMIT = 40;
const PLOT = DISTANCE_PLOT;
const RANGE = [0, 2.2] as const;
const REST_FRACTION = 0.3;
const WEIGHTINGS = ['cressie', 'pairs', 'none'] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Three fitted models over the empirical semivariance of log flash energy; the thin rules mark the spherical fit’s nugget, sill and range.',
  createInputs: ({random, world}) => {
    const sample = createFieldSample(random, world.width * 0.5, POINT_COUNT);
    return {...sample, world, pairs: createPairProbe(sample.positions, PAIR_LIMIT)};
  },
  build(builder, sample, values) {
    const positions = builder.upload('positions', 'float32x2', sample.positions);
    const observed = builder.allocate('values', 'float32', POINT_COUNT);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_VARIOGRAM_PARAMETER_LENGTH
    );
    const semivariances = builder.allocate('semivariances', 'float32', LAG_COUNT);
    const pairCounts = builder.allocate('pair-counts', 'uint32', LAG_COUNT);
    const meanDistances = builder.allocate('mean-distances', 'float32', LAG_COUNT);
    builder.graph.add(
      new GPUVariogram({
        positions: positions.view,
        values: observed.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        lagCount: LAG_COUNT,
        output: {
          semivariances: semivariances.view,
          pairCounts: pairCounts.view,
          meanDistances: meanDistances.view
        }
      })
    );
    const fitted = builder.allocate('fitted', 'float32x4', 3);
    builder.graph.add(
      new GPUVariogramModelFit({
        distances: meanDistances.view,
        semivariances: semivariances.view,
        pairCounts: pairCounts.view,
        models: ['spherical', 'exponential', 'gaussian'],
        weighting: WEIGHTINGS[values.weighting],
        output: {
          parameters: fitted.view,
          diagnostics: builder.allocate('diagnostics', 'float32x4', 3).view,
          status: builder.allocate('status', 'uint32', 3).view
        }
      })
    );
    const curveDistances = builder.allocate('curve-distances', 'float32', CURVE_COUNT);
    // Nugget, sill and range of the spherical fit, drawn as rules by the pointer interaction.
    const marks = builder.allocate('marks', 'float32x4', 3);
    const pairs = builder.allocate('pairs', 'float32x4', PAIR_LIMIT);
    return {
      outputs: {
        fitted,
        pairs,
        pairCounts,
        marks,
        positions,
        observed,
        semivariances,
        meanDistances,
        spherical: addFittedCurve(builder, 'spherical', fitted, curveDistances, 0),
        exponential: addFittedCurve(builder, 'exponential', fitted, curveDistances, 1),
        gaussian: addFittedCurve(builder, 'gaussian', fitted, curveDistances, 2)
      },
      update: current => {
        observed.write(getFieldValues(sample, current.noise));
        curveDistances.write(
          Float32Array.from(
            {length: CURVE_COUNT},
            (_, index) => (index / (CURVE_COUNT - 1)) * current.maximumDistance
          )
        );
        parameterBuffer.write(
          getGPUVariogramParameterValues({
            bounds: [0, 0, sample.windowWidth, 1],
            maximumDistance: current.maximumDistance
          })
        );
      }
    };
  },
  rebuildOn: ['weighting'],
  // The pointer is a distance on the chart: the empirical semivariance and the three fitted models
  // are read at that distance, and the spherical fit's nugget, sill and range are marked.
  interaction: {
    kind: 'probe',
    hint: 'hover a distance',
    rest: [getAxisPointer(REST_FRACTION), 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const fraction = getAxisFraction(pointer.unitX);
      const distance = fraction * values.maximumDistance;
      const x = (PLOT[0] + fraction * (PLOT[2] - PLOT[0])) * inputs.world.width;
      const bottom = PLOT[1] * inputs.world.height;
      const top = PLOT[3] * inputs.world.height;
      const curveIndex = Math.round(fraction * (CURVE_COUNT - 1));
      const empirical = peek(outputs.semivariances)?.[getBinIndex(fraction, LAG_COUNT)];
      const models = [outputs.spherical, outputs.exponential, outputs.gaussian].map(
        buffer => peek(buffer)?.[curveIndex]
      );
      const fit = peek(outputs.fitted);
      if (fit) {
        const [nugget, partialSill, range] = fit;
        const left = PLOT[0] * inputs.world.width;
        const right = PLOT[2] * inputs.world.width;
        const rangeX = left + Math.min(1, range / values.maximumDistance) * (right - left);
        const nuggetY = getChartWorldY(PLOT, inputs.world, RANGE, nugget);
        const sillY = getChartWorldY(PLOT, inputs.world, RANGE, nugget + partialSill);
        outputs.marks.write(
          Float32Array.of(
            left,
            nuggetY,
            right,
            nuggetY,
            left,
            sillY,
            right,
            sillY,
            rangeX,
            bottom,
            rangeX,
            sillY
          )
        );
      }
      // The point pairs the hovered semivariance bin is made of, drawn on the map.
      const binWidth = values.maximumDistance / LAG_COUNT;
      const bin = getBinIndex(fraction, LAG_COUNT);
      const found = inputs.pairs.find(bin * binWidth, (bin + 1) * binWidth);
      outputs.pairs.write(found.segments);
      const pairCount = peek(outputs.pairCounts)?.[bin];
      const text = (value: number | undefined) => (value === undefined ? '…' : value.toFixed(1));
      return {
        spotlight: null,
        glyphs: [
          {shape: 'line', from: [x, bottom], to: [x, top]},
          ...models.flatMap((value, model) =>
            value === undefined
              ? []
              : [
                  {
                    shape: 'dot' as const,
                    at: [x, getChartWorldY(PLOT, inputs.world, RANGE, value)] as const
                  }
                ]
          )
        ],
        readout: `h ${distance.toFixed(2)} · ${pairCount ?? '…'} pairs (sample of ${found.shown}) · data ${text(empirical)} · sph ${text(models[0])} exp ${text(models[1])} gau ${text(models[2])}`
      };
    }
  },
  draw: ({outputs, values}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      values: outputs.observed,
      range: [1, 7],
      radius: 2.8
    },
    {layer: 'lines', segments: outputs.pairs, count: PAIR_LIMIT, color: [1, 0.6, 0.2, 0.55]},
    {layer: 'lines', context: true, segments: outputs.marks, count: 3, color: [0.3, 0.75, 1, 0.55]},
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.semivariances,
      xValues: outputs.meanDistances,
      xRange: [0, values.maximumDistance],
      range: RANGE,
      rectangle: PLOT,
      radius: 5,
      color: [0.85, 0.85, 0.9, 0.9]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.spherical,
      range: RANGE,
      rectangle: PLOT,
      radius: 2.4,
      color: [0.3, 0.75, 1, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.exponential,
      range: RANGE,
      rectangle: PLOT,
      radius: 2.4,
      color: [1, 0.6, 0.2, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.gaussian,
      range: RANGE,
      rectangle: PLOT,
      radius: 2.4,
      color: [0.7, 1, 0.7, 1]
    }
  ]
});
