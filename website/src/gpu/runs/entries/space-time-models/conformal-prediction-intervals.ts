// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUConformalPredictionIntervals: split-conformal intervals for a neighbour-mean forecast of Helene
// rainfall, calibrated on held-out stations. Right: the calibration scores |y - p| / scale sorted
// into bars; the rule stands at rank ceil((1 - alpha)(n + 1)) and the bar under it (dashed line) sets
// every interval's half-width. Left: each test station with its interval as a vertical segment and
// the rain that fell as a dot, amber where the observation is outside the interval. The pointer
// slides the rule, which sets `miscoverage`.

import {
  getGPUConformalPredictionParameterValues,
  GPUConformalPredictionIntervals,
  GPU_CONFORMAL_PREDICTION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuilder} from '../../define-run';
import {clamp, formatNumber, snapToParameter, type RunGlyph} from '../../interaction';
import {createConformalScene, summarizeIntervals} from './_lib/conformal';
import {getFrameGlyph} from './_lib/surface';
import {parameters} from './conformal-prediction-intervals.parameters';

/** World height of one log-rainfall unit in the interval segments of the map. */
const SEGMENT_SCALE = 0.045;
const UNBOUNDED_LIMIT = 3;
const AMBER: readonly [number, number, number, number] = [0.95, 0.6, 0.1, 1];
const MISCOVERAGE = parameters[0];
const MASK = parameters[1];

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const scene = createConformalScene(random, world);
  const {calibration} = scene;
  const plain = Float32Array.from(calibration.observations, (value, row) =>
    Math.abs(value - calibration.predictions[row])
  );
  const scaled = Float32Array.from(plain, (value, row) => value / calibration.scales[row]);
  // The score axis ends at the 97th percentile of the scores, so a few outliers do not flatten the bars.
  const getCap = (scores: Float32Array) =>
    1.1 * Float32Array.from(scores).sort()[Math.floor(0.97 * (scores.length - 1))];
  return {
    ...scene,
    world,
    scores: [plain, scaled],
    caps: [getCap(plain), getCap(scaled)],
    scoreCache: {key: '', sorted: new Float32Array(0)}
  };
}

type Inputs = ReturnType<typeof createInputs>;

function getKeptCount(inputs: Inputs, values: Readonly<Record<string, number>>): number {
  return Math.ceil((values.mask / 100) * inputs.calibration.count);
}

/** The kept calibration scores in ascending order (memoized on the parameters that change them). */
function getSortedScores(inputs: Inputs, values: Readonly<Record<string, number>>): Float32Array {
  const key = `${values.scales}|${values.mask}`;
  if (inputs.scoreCache.key !== key) {
    inputs.scoreCache = {
      key,
      sorted: inputs.scores[values.scales].slice(0, getKeptCount(inputs, values)).sort()
    };
  }
  return inputs.scoreCache.sorted;
}

/** The rank the operation reads, ceil((1 - alpha)(n + 1)), with its 2^-20 slack. */
function getRank(count: number, miscoverage: number): number {
  return Math.ceil((1 - miscoverage) * (count + 1) * (1 - 2 ** -20));
}

function getChartRectangle(inputs: Inputs) {
  const {panel, world} = inputs;
  return [panel.x0 / world.width, panel.y0, panel.x1 / world.width, panel.y1] as const;
}

function build(builder: RunBuilder, inputs: Inputs, values: Record<string, number>) {
  const {calibration, test} = inputs;
  const useScales = values.scales === 1;
  const mask = builder.allocate('mask', 'uint32', calibration.count);
  const scores = builder.allocate('scores', 'float32', calibration.count);
  const parameterBuffer = builder.allocate(
    'parameters',
    'float32',
    GPU_CONFORMAL_PREDICTION_PARAMETER_LENGTH
  );
  const intervals = builder.allocate('intervals', 'float32', test.count * 2);
  builder.graph.add(
    new GPUConformalPredictionIntervals({
      calibration: {
        observations: builder.upload(
          'calibration-observations',
          'float32',
          calibration.observations
        ).view,
        predictions: builder.upload('calibration-predictions', 'float32', calibration.predictions)
          .view,
        ...(useScales
          ? {scales: builder.upload('calibration-scales', 'float32', calibration.scales).view}
          : {}),
        mask: mask.view
      },
      query: {
        predictions: builder.upload('test-predictions', 'float32', test.predictions).view,
        ...(useScales ? {scales: builder.upload('test-scales', 'float32', test.scales).view} : {})
      },
      parameters: parameterBuffer.view,
      output: {
        intervals: intervals.view,
        status: builder.allocate('status', 'uint32', test.count).view,
        calibrationCount: builder.allocate('calibration-count', 'uint32', 1).view
      }
    })
  );
  return {
    outputs: {
      intervals,
      scores,
      testPositions: builder.upload('test-positions', 'float32x2', test.positions)
    },
    update: (current: Record<string, number>) => {
      parameterBuffer.write(
        getGPUConformalPredictionParameterValues({miscoverage: current.miscoverage})
      );
      const kept = getKeptCount(inputs, current);
      mask.write(Uint32Array.from({length: calibration.count}, (_, row) => (row < kept ? 1 : 0)));
      // Sorted scores for the bars; the rows the mask removes are NaN and draw as nothing.
      const sorted = getSortedScores(inputs, current);
      scores.write(
        Float32Array.from({length: calibration.count}, (_, row) =>
          row < sorted.length ? sorted[row] : Number.NaN
        )
      );
    }
  };
}

export default defineOperationRun({
  parameters,
  caption:
    'Split-conformal intervals of a Helene rainfall forecast from the nearest stations; the rule sets miscoverage.',
  createInputs,
  build,
  rebuildOn: ['scales'],
  // The intervals are read back to draw them on the map.
  async afterRun({inputs, outputs, upload}) {
    const {test} = inputs;
    const bytes = await outputs.intervals.buffer.readAsync();
    const intervals = new Float32Array(bytes.buffer, bytes.byteOffset, test.count * 2).slice();
    const covered: number[] = [];
    const missed: number[] = [];
    const observed = new Float32Array(test.count * 2);
    const flags = new Uint32Array(test.count);
    for (let row = 0; row < test.count; row++) {
      const x = test.positions[row * 2];
      const y = test.positions[row * 2 + 1];
      const prediction = test.predictions[row];
      const observation = test.observations[row];
      observed.set(
        [x, y + clamp(observation - prediction, -UNBOUNDED_LIMIT, UNBOUNDED_LIMIT) * SEGMENT_SCALE],
        row * 2
      );
      const lower = intervals[row * 2];
      const upper = intervals[row * 2 + 1];
      if (Number.isNaN(lower) || Number.isNaN(upper)) continue;
      const isMissed = observation < lower || observation > upper;
      flags[row] = isMissed ? 1 : 0;
      (isMissed ? missed : covered).push(
        x,
        y + clamp(lower - prediction, -UNBOUNDED_LIMIT, UNBOUNDED_LIMIT) * SEGMENT_SCALE,
        x,
        y + clamp(upper - prediction, -UNBOUNDED_LIMIT, UNBOUNDED_LIMIT) * SEGMENT_SCALE
      );
    }
    const pad = (segments: number[]) =>
      Float32Array.from(segments.length > 0 ? segments : [0, 0, 0, 0]);
    return {
      covered: upload('covered', 'float32x4', pad(covered)),
      coveredCount: covered.length / 4,
      missed: upload('missed', 'float32x4', pad(missed)),
      missedCount: missed.length / 4,
      observed: upload('observed', 'float32x2', observed),
      flags: upload('flags', 'uint32', flags)
    };
  },
  interaction: {
    kind: 'pad',
    hint: 'rank rule and calibration share',
    // x: the inner 90% of the canvas width spans the ranks the miscoverage range reaches.
    // y: the share of calibration rows kept (`calibration.mask`), all of them at the top.
    rest: () => [
      0.05 + 0.9 * ((1 - MISCOVERAGE.default - 0.5) / 0.48),
      0.05 + 0.9 * ((MASK.max - MASK.default) / (MASK.max - MASK.min))
    ],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const {panel, calibration} = inputs;
      const width = panel.x1 - panel.x0;
      const mask = snapToParameter(
        MASK,
        MASK.max - ((MASK.max - MASK.min) * clamp(pointer.unitY - 0.05, 0, 0.9)) / 0.9
      );
      values = {...values, mask};
      const sorted = getSortedScores(inputs, values);
      // Canvas x spans the rule's reachable ranks, from alpha 0.5 down to 0.02.
      const fraction = 0.5 + (0.48 * (pointer.unitX - 0.05)) / 0.9;
      const miscoverage = snapToParameter(MISCOVERAGE, 1 - clamp(fraction, 0, 1));
      const rank = getRank(sorted.length, miscoverage);
      const cap = inputs.caps[values.scales];
      const ruleX = panel.x0 + (Math.min(rank, sorted.length) - 0.5) * (width / calibration.count);
      const glyphs: RunGlyph[] = [
        getFrameGlyph(panel),
        {shape: 'line', from: [ruleX, panel.y0], to: [ruleX, panel.y1]}
      ];
      if (rank <= sorted.length) {
        const barTop = panel.y0 + Math.min(sorted[rank - 1] / cap, 1) * (panel.y1 - panel.y0);
        glyphs.push({shape: 'line', from: [panel.x0, barTop], to: [ruleX, barTop], dashed: true});
      }
      const intervals = peek(outputs.intervals);
      const summary = intervals ? summarizeIntervals(intervals, inputs.test.observations) : null;
      const alpha = formatNumber(miscoverage, 2);
      return {
        values: {miscoverage, mask},
        glyphs,
        spotlight: null,
        readout: summary
          ? `alpha ${alpha} · ±${formatNumber(summary.meanWidth / 2, 2)} · covered ${formatNumber(summary.coverage * 100, 0)}% · mask ${mask}%`
          : `alpha ${alpha} · mask ${mask}%`
      };
    }
  },
  draw: ({inputs, outputs, values, extra}) => [
    {layer: 'dots', context: true, positions: outputs.testPositions, radius: 2, alpha: 0.5},
    ...(extra
      ? [
          {
            layer: 'lines' as const,
            segments: extra.covered,
            count: extra.coveredCount,
            radius: 2.6,
            alpha: 0.75
          },
          {
            layer: 'lines' as const,
            segments: extra.missed,
            count: extra.missedCount,
            radius: 3.2,
            color: AMBER
          },
          {
            layer: 'dots' as const,
            positions: extra.observed,
            values: extra.flags,
            categories: [
              [0.6, 0.65, 0.75],
              [0.95, 0.6, 0.1]
            ] as const,
            radius: 2.5
          }
        ]
      : []),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.scores,
      rectangle: getChartRectangle(inputs),
      ramp: 'sequential',
      range: [0, inputs.caps[values.scales]]
    }
  ]
});
