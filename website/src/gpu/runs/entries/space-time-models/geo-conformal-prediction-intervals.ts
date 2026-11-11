// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeoConformalPredictionIntervals: localized split-conformal intervals for a neighbour-mean
// forecast of Helene rainfall. Each test station weighs its nearest
// calibration stations with a Gaussian kernel. Left: the interval width at each test station;
// hovering one draws its weighted calibration neighbours (ring size is weight). Right: each test
// station's forecast against what fell, with its interval as a bar.

import {
  getGPUConformalPredictionParameterValues,
  GPUGeoConformalPredictionIntervals,
  GPU_CONFORMAL_PREDICTION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber, type RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {createConformalScene, getIntervalWidths} from './_lib/conformal';
import {getFrameGlyph} from './_lib/surface';
import {parameters} from './geo-conformal-prediction-intervals.parameters';

const MAXIMUM_NEIGHBORS = 64;
const WIDTH_RANGE: readonly [number, number] = [0, 3.5];

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const scene = createConformalScene(random, world);
  const {calibration, test} = scene;
  // The nearest calibration stations of every test station, nearest first, computed once.
  const nearestRows = new Uint32Array(test.count * MAXIMUM_NEIGHBORS);
  const nearestDistances = new Float32Array(test.count * MAXIMUM_NEIGHBORS);
  for (let row = 0; row < test.count; row++) {
    const x = test.positions[row * 2];
    const y = test.positions[row * 2 + 1];
    const order = Array.from({length: calibration.count}, (_, other) => ({
      other,
      distance: Math.hypot(
        calibration.positions[other * 2] - x,
        calibration.positions[other * 2 + 1] - y
      )
    }))
      .sort((first, second) => first.distance - second.distance)
      .slice(0, MAXIMUM_NEIGHBORS);
    order.forEach((entry, index) => {
      nearestRows[row * MAXIMUM_NEIGHBORS + index] = entry.other;
      nearestDistances[row * MAXIMUM_NEIGHBORS + index] = entry.distance;
    });
  }
  // Forecast against observation for the panel.
  const getRange = (array: Float32Array) =>
    [Math.floor(Math.min(...array) * 4) / 4, Math.ceil(Math.max(...array) * 4) / 4] as const;
  const index = createPointIndex(test.positions);
  const sortedScores = Float32Array.from(
    scene.calibration.observations,
    (value, row) =>
      Math.abs(value - scene.calibration.predictions[row]) / scene.calibration.scales[row]
  ).sort();
  const center = index.nearest(scene.mapWidth * 0.5, world.height * 0.5);
  return {
    ...scene,
    sortedScores,
    nearestRows,
    nearestDistances,
    index,
    restRow: center,
    focus: {row: -1, key: ''},
    predictionRange: getRange(test.predictions),
    valueRange: getRange(test.observations)
  };
}

type Inputs = ReturnType<typeof createInputs>;

function getWeight(distance: number, bandwidth: number): number {
  return Math.exp(-0.5 * (distance / bandwidth) ** 2);
}

function getPanelPoint(inputs: Inputs, prediction: number, value: number): [number, number] {
  const {panel, valueRange, predictionRange} = inputs;
  const clamped = Math.min(Math.max(value, valueRange[0]), valueRange[1]);
  return [
    panel.x0 +
      ((prediction - predictionRange[0]) / (predictionRange[1] - predictionRange[0])) *
        (panel.x1 - panel.x0),
    panel.y0 + ((clamped - valueRange[0]) / (valueRange[1] - valueRange[0])) * (panel.y1 - panel.y0)
  ];
}

/** The panel point of a (possibly unbounded) bound of a test row's interval. */
function getBoundPoint(inputs: Inputs, row: number, bound: number): [number, number] {
  return getPanelPoint(inputs, inputs.test.predictions[row], bound);
}

export default defineOperationRun({
  parameters,
  caption:
    'Localized conformal intervals of a Helene rainfall forecast; hover a station to see its weighted neighbours.',
  createInputs,
  build(builder, inputs) {
    const {calibration, test} = inputs;
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_CONFORMAL_PREDICTION_PARAMETER_LENGTH
    );
    const offsets = builder.allocate('offsets', 'uint32', test.count + 1);
    const indices = builder.allocate('indices', 'uint32', test.count * MAXIMUM_NEIGHBORS);
    const weights = builder.allocate('weights', 'float32', test.count * MAXIMUM_NEIGHBORS);
    const intervals = builder.allocate('intervals', 'float32', test.count * 2);
    builder.graph.add(
      new GPUGeoConformalPredictionIntervals({
        calibration: {
          observations: builder.upload(
            'calibration-observations',
            'float32',
            calibration.observations
          ).view,
          predictions: builder.upload('calibration-predictions', 'float32', calibration.predictions)
            .view,
          scales: builder.upload('calibration-scales', 'float32', calibration.scales).view
        },
        query: {
          predictions: builder.upload('test-predictions', 'float32', test.predictions).view,
          scales: builder.upload('test-scales', 'float32', test.scales).view
        },
        neighbors: {
          offsets: offsets.view,
          indices: indices.view,
          weights: weights.view
        },
        parameters: parameterBuffer.view,
        output: {
          intervals: intervals.view,
          status: builder.allocate('status', 'uint32', test.count).view
        }
      })
    );
    return {
      outputs: {
        intervals,
        calibrationPositions: builder.upload(
          'calibration-positions',
          'float32x2',
          calibration.positions
        ),
        testPositions: builder.upload('test-positions', 'float32x2', test.positions),
        focusPositions: builder.allocate('focus-positions', 'float32x2', MAXIMUM_NEIGHBORS),
        focusWeights: builder.allocate('focus-weights', 'float32', MAXIMUM_NEIGHBORS)
      },
      update: (current: Record<string, number>) => {
        const count = current.neighborCount;
        const offsetValues = Uint32Array.from({length: test.count + 1}, (_, row) => row * count);
        const indexValues = new Uint32Array(test.count * MAXIMUM_NEIGHBORS);
        const weightValues = new Float32Array(test.count * MAXIMUM_NEIGHBORS);
        for (let row = 0; row < test.count; row++) {
          for (let slot = 0; slot < count; slot++) {
            indexValues[row * count + slot] = inputs.nearestRows[row * MAXIMUM_NEIGHBORS + slot];
            weightValues[row * count + slot] = getWeight(
              inputs.nearestDistances[row * MAXIMUM_NEIGHBORS + slot],
              current.bandwidth
            );
          }
        }
        offsets.write(offsetValues);
        indices.write(indexValues);
        weights.write(weightValues);
        parameterBuffer.write(
          getGPUConformalPredictionParameterValues({
            miscoverage: current.miscoverage
          })
        );
      }
    };
  },
  async afterRun({inputs, outputs, upload}) {
    const {test} = inputs;
    const bytes = await outputs.intervals.buffer.readAsync();
    const intervals = new Float32Array(bytes.buffer, bytes.byteOffset, test.count * 2).slice();
    const bars: number[] = [];
    const observed: number[] = [];
    const missed = new Uint32Array(test.count);
    for (let row = 0; row < test.count; row++) {
      const lower = intervals[row * 2];
      const upper = intervals[row * 2 + 1];
      observed.push(...getPanelPoint(inputs, test.predictions[row], test.observations[row]));
      if (Number.isNaN(lower) || Number.isNaN(upper)) continue;
      missed[row] = test.observations[row] < lower || test.observations[row] > upper ? 1 : 0;
      // An unbounded side has no bar: the station is muted in the panel.
      if (Number.isFinite(lower) && Number.isFinite(upper)) {
        bars.push(...getBoundPoint(inputs, row, lower), ...getBoundPoint(inputs, row, upper));
      }
    }
    return {
      widths: upload('widths', 'float32', getIntervalWidths(intervals, test.count)),
      bars: upload('bars', 'float32x4', Float32Array.from(bars.length > 0 ? bars : [0, 0, 0, 0])),
      barCount: bars.length / 4,
      observed: upload('observed', 'float32x2', Float32Array.from(observed)),
      missed: upload('missed', 'uint32', missed)
    };
  },
  // The pointer is a test station; its weighted calibration neighbours are drawn bright.
  interaction: {
    kind: 'probe',
    hint: 'hover a station',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.test.positions[inputs.restRow * 2],
        inputs.test.positions[inputs.restRow * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, values, build, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      if (row < 0) return {};
      const {test, calibration} = inputs;
      const count = values.neighborCount;
      const x = test.positions[row * 2];
      const y = test.positions[row * 2 + 1];
      const key = `${row}|${count}|${values.bandwidth}|${build}`;
      const glyphs: RunGlyph[] = [{shape: 'ring', at: [x, y], radius: 0.02}];
      const neighborPositions = new Float32Array(MAXIMUM_NEIGHBORS * 2);
      const neighborWeights = new Float32Array(MAXIMUM_NEIGHBORS);
      let weightSum = 0;
      for (let slot = 0; slot < count; slot++) {
        const other = inputs.nearestRows[row * MAXIMUM_NEIGHBORS + slot];
        const weight = getWeight(
          inputs.nearestDistances[row * MAXIMUM_NEIGHBORS + slot],
          values.bandwidth
        );
        const at: [number, number] = [
          calibration.positions[other * 2],
          calibration.positions[other * 2 + 1]
        ];
        neighborPositions.set(at, slot * 2);
        neighborWeights[slot] = weight;
        weightSum += weight;
        if (weight > 0.02) {
          glyphs.push({shape: 'line', from: [x, y], to: at});
        }
      }
      if (inputs.focus.key !== key) {
        outputs.focusPositions.write(neighborPositions);
        outputs.focusWeights.write(neighborWeights);
        inputs.focus.key = key;
      }
      inputs.focus.row = row;
      glyphs.push(getFrameGlyph(inputs.panel));
      const intervals = peek(outputs.intervals);
      let readout = `station ${row} · k ${count} · weight ${formatNumber(weightSum, 1)}`;
      if (intervals) {
        const lower = intervals[row * 2];
        const upper = intervals[row * 2 + 1];
        glyphs.push({
          shape: 'ring',
          at: getPanelPoint(inputs, test.predictions[row], test.observations[row]),
          radius: 0.015
        });
        if (Number.isFinite(lower) && Number.isFinite(upper)) {
          const sorted = inputs.sortedScores;
          const rank = Math.ceil((1 - values.miscoverage) * (sorted.length + 1) * (1 - 2 ** -20));
          const globalHalfWidth =
            rank <= sorted.length ? test.scales[row] * sorted[rank - 1] : Number.NaN;
          readout = `±${formatNumber((upper - lower) / 2, 2)} here, ±${formatNumber(globalHalfWidth, 2)} global · station ${row}`;
          glyphs.push(
            {
              shape: 'line',
              from: getBoundPoint(inputs, row, lower),
              to: getBoundPoint(inputs, row, upper)
            },
            {shape: 'dot', at: getBoundPoint(inputs, row, lower)},
            {shape: 'dot', at: getBoundPoint(inputs, row, upper)}
          );
        } else {
          readout = `unbounded: too few neighbours · k ${count}`;
        }
      }
      return {focus: row, glyphs, readout};
    }
  },
  draw: ({outputs, extra, values, interaction}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.calibrationPositions,
      radius: 2.5,
      alpha: 0.5
    },
    ...(extra
      ? [
          {
            layer: 'dots' as const,
            positions: outputs.testPositions,
            values: extra.widths,
            ramp: 'sequential' as const,
            range: WIDTH_RANGE,
            radius: 5
          },
          {
            layer: 'lines' as const,
            context: true,
            segments: extra.bars,
            count: extra.barCount,
            radius: 1
          },
          {
            layer: 'dots' as const,
            positions: extra.observed,
            values: extra.missed,
            categories: [
              [0.55, 0.65, 0.8],
              [0.95, 0.5, 0.15]
            ] as const,
            radius: 2.5
          },
          ...(typeof interaction?.focus === 'number' && interaction.focus >= 0
            ? [
                {
                  layer: 'dots' as const,
                  positions: outputs.focusPositions,
                  values: outputs.focusWeights,
                  count: values.neighborCount,
                  ramp: 'sequential' as const,
                  range: [0, 1] as const,
                  radius: 5,
                  sizeByValue: true
                }
              ]
            : [])
        ]
      : [])
  ]
});
