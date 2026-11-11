// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMultiscaleGeographicallyTemporallyWeightedRegression gives every coefficient its own space-time
// bandwidth. Top row: the intercept in the four periods, which truly varies slowly, so a large
// bandwidth keeps it smooth (a small one makes it noisy). Bottom row: the slope in the four periods,
// which has a sharp front sweeping west to east, so it needs a small bandwidth. The pad sets the
// slope bandwidth (x) and the intercept bandwidth (y) of the same compiled backfitting.

import {
  getGPUMGTWRParameterValues,
  GPUMultiscaleGeographicallyTemporallyWeightedRegression
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes, formatNumber} from '../../interaction';
import {addKernelPass} from '../regression-weights/shared/kernel';
import {addTableColumn} from '../regression-weights/shared/weights';
import {getInk, getMutedInk} from '../raster-time-series/_lib/charts';
import {
  createSpaceTimeScene,
  drawSpaceTimeTiles,
  PERIOD_TEMPORAL_CONTEXT,
  type SpaceTimeScene,
  uploadSpaceTimeMesh,
  uploadSpaceTimeObservations
} from './_lib/gtwr-data';
import {parameters} from './multiscale-geographically-temporally-weighted-regression.parameters';

const ITERATION_COUNT = 5;
const INITIAL_BANDWIDTH = 0.3;
const INTERCEPT_RANGE = [0.5, 2.5] as const;
const BOTTOM_MARGIN = 0.24;
const SCORE_CHART = [0.06, 0.04, 0.4, 0.2] as const;
/** The chart shows log10 of the scores over this range. */
const LOG_SCORE_RANGE = [-7, 0] as const;
const SLOPE_RANGE = [0, 3.4] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Intercept (top) and slope (bottom) in four periods, each with its own bandwidth and its own color scale; bottom left: log score of each backfitting iteration; the response is simulated on real tract poverty shares with a slowly varying intercept, a sweeping slope and noise.',
  createInputs: ({random, world}) =>
    createSpaceTimeScene(random, world, {
      tileColumns: 4,
      tileRows: 2,
      periodsFillGrid: false,
      bottomMargin: BOTTOM_MARGIN
    }),
  build(builder, scene) {
    const observations = uploadSpaceTimeObservations(builder, scene);
    const parameterBuffer = builder.allocate('parameters', 'float32', 6);
    const coefficients = builder.allocate('coefficients', 'float32', scene.rowCount * 2);
    const scores = builder.allocate('scores', 'float32', ITERATION_COUNT);
    builder.graph.add(
      new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
        positions: observations.positions.view,
        times: observations.times.view,
        temporalContext: PERIOD_TEMPORAL_CONTEXT,
        predictors: observations.predictors.view,
        predictorCount: 1,
        response: observations.response.view,
        kernel: 'gaussian',
        iterationCount: ITERATION_COUNT,
        parameters: parameterBuffer.view,
        output: {
          coefficients: coefficients.view,
          fitted: builder.allocate('fitted', 'float32', scene.rowCount).view,
          residuals: builder.allocate('residuals', 'float32', scene.rowCount).view,
          localStatus: builder.allocate('local-status', 'uint32', scene.rowCount).view,
          scores: scores.view
        }
      })
    );
    const logScores = builder.allocate('log-scores', 'float32', ITERATION_COUNT);
    addKernelPass(builder, {
      id: 'log-scores',
      invocationCount: ITERATION_COUNT,
      bindings: [
        {name: 'scores', buffer: scores, access: 'read'},
        {name: 'values', buffer: logScores, access: 'read_write'}
      ],
      body: 'values[valuesOffset + index] = log(max(scores[scoresOffset + index], 1e-9)) / 2.302585;'
    });
    return {
      outputs: {
        logScores,
        backing: builder.upload('backing', 'float32', new Float32Array(ITERATION_COUNT).fill(1)),
        meshes: [uploadSpaceTimeMesh(builder, scene, 0), uploadSpaceTimeMesh(builder, scene, 1)],
        intercept: addTableColumn(builder, 'intercept', coefficients, 2, 0, scene.rowCount),
        slope: addTableColumn(builder, 'slope', coefficients, 2, 1, scene.rowCount),
        scores
      },
      update: current =>
        parameterBuffer.write(
          getGPUMGTWRParameterValues({
            initial: {bandwidth: INITIAL_BANDWIDTH, timeScale: current.timeScale},
            components: [
              {bandwidth: current.interceptBandwidth, timeScale: current.timeScale},
              {bandwidth: current.slopeBandwidth, timeScale: current.timeScale}
            ]
          })
        )
    };
  },
  interaction: bindAxes<SpaceTimeScene, {scores: RunBuffer<'float32'>}>(parameters, {
    kind: 'pad',
    hint: 'slope and intercept bandwidth',
    x: {parameter: 'slopeBandwidth', scale: 'logarithmic'},
    y: {parameter: 'interceptBandwidth', scale: 'logarithmic'},
    decorate: (values, pointer, {outputs, peek}) => {
      const scores = peek(outputs.scores);
      const score = scores
        ? ` · score ${scores[ITERATION_COUNT - 1].toExponential(0)} after ${ITERATION_COUNT}`
        : '';
      return {
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: `slope h ${formatNumber(values.slopeBandwidth)} · intercept h ${formatNumber(values.interceptBandwidth)}${score}`
      };
    }
  }),
  draw: ({inputs, outputs, theme}) => [
    ...drawSpaceTimeTiles(inputs, 0, outputs.meshes[0], outputs.intercept, INTERCEPT_RANGE),
    ...drawSpaceTimeTiles(inputs, 1, outputs.meshes[1], outputs.slope, SLOPE_RANGE),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.backing,
      range: [0, 1],
      color: getMutedInk(theme, 0.12),
      rectangle: SCORE_CHART
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.logScores,
      range: LOG_SCORE_RANGE,
      color: getInk(theme, 0.95),
      radius: 1.6,
      rectangle: SCORE_CHART
    }
  ]
});
