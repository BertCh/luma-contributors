// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialCrossValidatedOLS: fits ordinary least squares on each fold's training blocks and
// predicts everything. The map shows the residual (prediction minus observed) of one fold's
// held-out stations; the other stations are muted. Hovering a station picks its fold. The bars are
// the fitted coefficients of every fold (intercept first); when they agree across folds the model is
// stable across space. Stations the hovered fold trained on are drawn in blue, those buffered out
// in orange, and the held-out ones in the residual colors.

import {GPUSpatialCrossValidatedOLS} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getRainStations} from '../../../data/point-observations';
import {createChartAxes, getChartRectangle, mergeText} from '../../text';
import {chooseSticky, createPointIndex} from '../../pick';
import {drawSignedBars} from '../regression-weights/shared/regression';
import {CHART_BAND} from './shared/point-field';
import {parameters} from './spatial-cross-validated-ols.parameters';

const POINT_COUNT = 400;
const PREDICTOR_COUNT = 3;
const BLOCK_COLUMNS = 6;
const BLOCK_ROWS = 3;
const BARS = [0.2, 0.04, 0.8, 0.24] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Carolina rainfall predicted from the previous day and station location, with spatial holdouts.',
  createInputs: ({random, world}) => {
    const mapHeight = world.height - CHART_BAND;
    const points = getRainStations(
      random,
      {width: world.width, height: mapHeight},
      POINT_COUNT,
      true
    );
    const observedRain = Float32Array.from(
      points.rain,
      rain => Math.log1p(rain) / Math.log1p(343.4)
    );
    const positions = Float32Array.from(points.positions, (value, index) =>
      index % 2 === 1 ? value + CHART_BAND : value
    );
    const predictors = new Float32Array(POINT_COUNT * PREDICTOR_COUNT);
    const blocks = new Uint32Array(POINT_COUNT);
    const noise = Float32Array.from({length: POINT_COUNT}, () => random() - 0.5);
    for (let row = 0; row < POINT_COUNT; row++) {
      const x = points.positions[row * 2] / world.width;
      const y = points.positions[row * 2 + 1] / mapHeight;
      predictors[row * PREDICTOR_COUNT] = Math.log1p(points.previousRain[row]) / Math.log1p(343.4);
      predictors[row * PREDICTOR_COUNT + 1] = x;
      predictors[row * PREDICTOR_COUNT + 2] = y;
      blocks[row] =
        Math.min(Math.floor(y * BLOCK_ROWS), BLOCK_ROWS - 1) * BLOCK_COLUMNS +
        Math.min(Math.floor(x * BLOCK_COLUMNS), BLOCK_COLUMNS - 1);
    }
    return {
      positions,
      index: createPointIndex(positions),
      focus: {row: -1},
      predictors,
      observedRain,
      blocks,
      noise
    };
  },
  build(builder, inputs, values) {
    const foldCount = values.foldCount;
    const response = builder.allocate('response', 'float32', POINT_COUNT);
    const folds = builder.allocate('folds', 'uint32', POINT_COUNT);
    const training = builder.allocate('training', 'uint32', foldCount * POINT_COUNT);
    const testing = builder.allocate('testing', 'uint32', foldCount * POINT_COUNT);
    const counts = builder.allocate('counts', 'uint32', foldCount * 3);
    const foldStatus = builder.allocate('fold-status', 'uint32', foldCount);
    const coefficients = builder.allocate(
      'coefficients',
      'float32',
      foldCount * (PREDICTOR_COUNT + 1)
    );
    const predictions = builder.allocate('predictions', 'float32', foldCount * POINT_COUNT);
    const fitStatus = builder.allocate('fit-status', 'uint32', foldCount);
    const metrics = builder.allocate('metrics', 'float32', foldCount * 6);
    const metricStatus = builder.allocate('metric-status', 'uint32', foldCount);
    builder.graph.add(
      new GPUSpatialCrossValidatedOLS({
        folds: {
          positions: builder.upload('positions', 'float32x2', inputs.positions).view,
          blocks: builder.upload('blocks', 'uint32', inputs.blocks).view,
          foldCount,
          seed: 3,
          bufferDistance: values.bufferDistance,
          output: {
            folds: folds.view,
            training: training.view,
            testing: testing.view,
            counts: counts.view,
            status: foldStatus.view
          }
        },
        predictors: builder.upload('predictors', 'float32', inputs.predictors).view,
        predictorCount: PREDICTOR_COUNT,
        response: response.view,
        output: {
          coefficients: coefficients.view,
          predictions: predictions.view,
          fitStatus: fitStatus.view,
          metrics: metrics.view,
          metricStatus: metricStatus.view
        }
      })
    );
    return {
      outputs: {
        response,
        folds,
        testing,
        predictions,
        coefficients,
        counts,
        foldCount,
        // Render-only: the hovered fold's residuals at its held-out stations, NaN elsewhere.
        residuals: builder.allocate('residuals', 'float32', POINT_COUNT),
        // Render-only: the stations the buffer keeps out of the hovered fold's training set, and
        // the hovered fold's own coefficients, drawn over the bars.
        buffered: builder.allocate('buffered', 'float32x2', POINT_COUNT),
        trained: builder.allocate('trained', 'float32x2', POINT_COUNT),
        foldCoefficients: builder.allocate(
          'fold-coefficients',
          'float32',
          foldCount * (PREDICTOR_COUNT + 1)
        ),
        training,
        positions: builder.upload('map-positions', 'float32x2', inputs.positions)
      },
      update: current => {
        response.write(
          Float32Array.from(
            inputs.observedRain,
            (rain, row) => rain + current.noise * inputs.noise[row]
          )
        );
      }
    };
  },
  rebuildOn: ['foldCount', 'bufferDistance'],
  interaction: {
    kind: 'probe',
    hint: 'hover a station',
    rest: [0.5, 0.4],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      const folds = peek(outputs.folds);
      const testing = peek(outputs.testing);
      const predictions = peek(outputs.predictions);
      const response = peek(outputs.response);
      // Zero line through the coefficient bars (the bars span -1 to 1).
      const zero = (BARS[1] + (BARS[3] - BARS[1]) / 2) * world.height;
      const zeroLine = {
        shape: 'line',
        from: [BARS[0] * world.width, zero],
        to: [BARS[2] * world.width, zero]
      } as const;
      if (row < 0 || !folds || !testing || !predictions || !response) {
        return {glyphs: [zeroLine], focus: 0};
      }
      const fold = folds[row];
      const residuals = new Float32Array(POINT_COUNT).fill(NaN);
      for (let station = 0; station < POINT_COUNT; station++) {
        if (testing[fold * POINT_COUNT + station] === 1) {
          residuals[station] = predictions[fold * POINT_COUNT + station] - response[station];
        }
      }
      outputs.residuals.write(residuals);
      const training = peek(outputs.training);
      const coefficients = peek(outputs.coefficients);
      let trainedCount = 0;
      if (training) {
        const trained = new Float32Array(POINT_COUNT * 2);
        for (let station = 0; station < POINT_COUNT; station++) {
          if (training[fold * POINT_COUNT + station] !== 1) continue;
          trained.set(
            [inputs.positions[station * 2], inputs.positions[station * 2 + 1]],
            trainedCount++ * 2
          );
        }
        outputs.trained.write(trained);
        const buffered = new Float32Array(POINT_COUNT * 2).fill(-10);
        for (let station = 0; station < POINT_COUNT; station++) {
          const isUsed =
            training[fold * POINT_COUNT + station] === 1 ||
            testing[fold * POINT_COUNT + station] === 1;
          if (isUsed) continue;
          buffered.set(
            [inputs.positions[station * 2], inputs.positions[station * 2 + 1]],
            station * 2
          );
        }
        outputs.buffered.write(buffered);
      }
      if (coefficients) {
        const lit = new Float32Array(coefficients.length).fill(Number.NaN);
        const width = PREDICTOR_COUNT + 1;
        lit.set(coefficients.subarray(fold * width, (fold + 1) * width), fold * width);
        outputs.foldCoefficients.write(lit);
      }
      const counts = peek(outputs.counts);
      return {
        focus: trainedCount,
        glyphs: [
          zeroLine,
          {
            shape: 'crosshair',
            at: [inputs.positions[row * 2], inputs.positions[row * 2 + 1]],
            size: 12
          }
        ],
        readout: `fold ${fold + 1}/${outputs.foldCount} · ${counts ? counts[fold * 3 + 1] : '…'} test · ${
          counts ? counts[fold * 3 + 2] : '…'
        } buffered out (${values.bufferDistance})`
      };
    }
  },
  draw: ({outputs, theme, interaction, world}) => {
    const trainedCount = typeof interaction.focus === 'number' ? interaction.focus : 0;
    const axes = createChartAxes({
      rectangle: getChartRectangle(world, BARS),
      y: {range: [-1, 1], ticks: [-1, 0, 1], title: 'coefficients of each fold'}
    });
    return [
      // Buffered out: close enough to the held-out block that the fold may not train on them.
      {layer: 'dots', positions: outputs.buffered, radius: 8, color: [1, 0.72, 0.2, 0.85]},
      // Residual of the hovered fold's held-out stations; stations outside it are muted.
      {
        layer: 'dots',
        positions: outputs.positions,
        values: outputs.residuals,
        ramp: 'diverging',
        range: [-0.1, 0.1],
        radius: 4
      },
      ...(trainedCount > 0
        ? [
            {
              layer: 'dots' as const,
              positions: outputs.trained,
              count: trainedCount,
              color: [...theme.muted, 0.8] as const,
              radius: 3
            }
          ]
        : []),
      // Signed bars from the zero line: each fold's coefficients, then the hovered fold's own in ink.
      ...drawSignedBars(outputs.coefficients, BARS, 1),
      ...drawSignedBars(outputs.foldCoefficients, BARS, 1, [...theme.ink, 0.9]),
      {
        layer: 'text',
        ...mergeText(axes, {
          labels: [
            {
              text: 'trained on',
              at: [0.02 * world.width, 0.97 * world.height],
              anchor: 'top-left',
              swatch: [...theme.muted]
            },
            {
              text: 'buffered out',
              at: [0.02 * world.width, 0.97 * world.height],
              anchor: 'top-left',
              offset: [0, -16],
              swatch: [1, 0.72, 0.2]
            },
            {
              text: 'held out: residual above observed',
              at: [0.02 * world.width, 0.97 * world.height],
              anchor: 'top-left',
              offset: [0, -32],
              swatch: [...theme.diverging[5]]
            },
            {
              text: 'held out: residual below observed',
              at: [0.02 * world.width, 0.97 * world.height],
              anchor: 'top-left',
              offset: [0, -48],
              swatch: [...theme.diverging[0]]
            }
          ]
        })
      }
    ];
  }
});
