// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpaceTimeHawkes fitted by EM to the 2023 Türkiye earthquake catalogue. Each event is colored
// by the operation's backgroundProbability on a diverging ramp centred at 0.5. The strip below is
// events per 4 days with the fitted expected count (background plus triggered, evaluated in the page
// from the four estimates) as a line. Hovering an event draws, from the fitted kernel, the later
// events it most likely triggered (arrows), its own most likely parent (dashed) and a ring of 2 sigma.

import {
  getGPUSpaceTimeHawkesParameterValues,
  GPUSpaceTimeHawkes,
  GPU_SPACE_TIME_HAWKES_ESTIMATE_LENGTH,
  GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createTurkiyeEvents, formatValue} from './_lib/events';
import {drawPanel, type Panel, uploadPanel} from './_lib/panel';
import {parameters} from './space-time-hawkes.parameters';

const WINDOW_DURATION = 352;
const BIN_DAYS = 4;
const BIN_COUNT = WINDOW_DURATION / BIN_DAYS;
const STRIP = [0.06, 0.04, 0.94, 0.18] as const;
const MAP_BOTTOM = 0.22;
/** Offspring with at least this responsibility get an arrow. */
const DRAWN_RESPONSIBILITY = 0.08;

type Inputs = ReturnType<typeof createTurkiyeEvents> & {
  worldWidth: number;
  binCounts: Float32Array;
  maximumBinCount: number;
  intensityCache: {key: string; values: Float64Array};
};
type Outputs = {
  panel: Panel;
  positions: RunBuffer;
  background: RunBuffer;
  estimates: RunBuffer;
  bins: RunBuffer;
};

/** Offspring density of `parent` at `child` under the fitted kernel; 0 outside the bounds. */
function getOffspringDensity(
  inputs: Inputs,
  parent: number,
  child: number,
  estimates: ArrayLike<number>,
  maximumLag: number,
  maximumDistance: number
): number {
  const lag = inputs.days[child] - inputs.days[parent];
  if (!(lag > 0) || lag > maximumLag) return 0;
  const distanceSquared =
    (inputs.positions[child * 2] - inputs.positions[parent * 2]) ** 2 +
    (inputs.positions[child * 2 + 1] - inputs.positions[parent * 2 + 1]) ** 2;
  if (distanceSquared > maximumDistance ** 2) return 0;
  const branchingRatio = estimates[1];
  const decayRate = estimates[2];
  const deviation = estimates[3];
  return (
    ((branchingRatio * decayRate * Math.exp(-decayRate * lag)) /
      (2 * Math.PI * deviation * deviation)) *
    Math.exp(-distanceSquared / (2 * deviation * deviation))
  );
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'USGS M4+ earthquakes, Türkiye 2023: color is P(background) (blue triggered, red background); strip is events per 4 days with the fitted rate. One constant background, no magnitude term (not ETAS).',
  createInputs: ({world}) => {
    const events = createTurkiyeEvents([0, MAP_BOTTOM, world.width, world.height]);
    const binCounts = new Float32Array(BIN_COUNT);
    for (const day of events.days) binCounts[Math.min(BIN_COUNT - 1, Math.floor(day / BIN_DAYS))]++;
    return {
      ...events,
      worldWidth: world.width,
      binCounts,
      maximumBinCount: Math.max(...binCounts),
      intensityCache: {key: '', values: new Float64Array(0)}
    };
  },
  build(builder, events, values) {
    const positions = builder.upload('positions', 'float32x2', events.positions);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH
    );
    const estimates = builder.allocate(
      'estimates',
      'float32',
      GPU_SPACE_TIME_HAWKES_ESTIMATE_LENGTH
    );
    const backgroundProbability = builder.allocate('background', 'float32', events.count);
    builder.graph.add(
      new GPUSpaceTimeHawkes({
        positions: positions.view,
        times: builder.upload('times', 'float32', events.days).view,
        parameters: parameterBuffer.view,
        maximumIterations: values.maximumIterations,
        maximumParents: 256,
        output: {estimates: estimates.view, backgroundProbability: backgroundProbability.view}
      })
    );
    return {
      outputs: {
        positions,
        background: backgroundProbability,
        estimates,
        panel: uploadPanel(builder, STRIP, {width: events.worldWidth, height: 1}),
        bins: builder.upload('bins', 'float32', events.binCounts)
      },
      update: current => {
        parameterBuffer.write(
          getGPUSpaceTimeHawkesParameterValues({
            windowArea: events.windowArea,
            windowDuration: WINDOW_DURATION,
            maximumLag: current.maximumLag,
            maximumDistance: current.maximumDistance / events.kilometersPerUnit,
            initialBackground: (0.5 * events.count) / (events.windowArea * WINDOW_DURATION),
            initialBranchingRatio: 0.5,
            initialDecayRate: 0.5,
            initialSpatialStandardDeviation: 20 / events.kilometersPerUnit
          })
        );
      }
    };
  },
  rebuildOn: ['maximumIterations'],
  buildCache: 5,
  interaction: {
    kind: 'probe',
    hint: 'hover an event',
    rest: [0.3, 0.45],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const estimates = peek(outputs.estimates);
      const fitted = estimates
        ? `branchingRatio ${formatValue(estimates[1])} · 1/decayRate ${formatValue(1 / estimates[2], 1)} d · σ ${formatValue(
            estimates[3] * inputs.kilometersPerUnit
          )} km`
        : '…';
      const strip =
        estimates && estimates.every(Number.isFinite)
          ? getFittedStrip(inputs, estimates, world.width)
          : [];
      const row = inputs.index.nearest(pointer.x, pointer.y);
      inputs.focus.row = row;
      if (row < 0 || !estimates || !estimates.every(Number.isFinite))
        return {glyphs: strip, readout: fitted};
      const maximumDistance = values.maximumDistance / inputs.kilometersPerUnit;
      const getDensity = (parent: number, child: number) =>
        getOffspringDensity(inputs, parent, child, estimates, values.maximumLag, maximumDistance);
      // Conditional intensity at every event, recomputed only when the fit or the bounds change.
      const key = `${Array.from(estimates).join()},${values.maximumLag},${maximumDistance}`;
      if (inputs.intensityCache.key !== key) {
        inputs.intensityCache.key = key;
        inputs.intensityCache.values = Float64Array.from({length: inputs.count}, (_, child) => {
          let intensity = estimates[0];
          for (let parent = 0; parent < inputs.count; parent++) {
            intensity += getDensity(parent, child);
          }
          return intensity;
        });
      }
      const intensities = inputs.intensityCache.values;
      const getIntensity = (child: number) => intensities[child];
      const from = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const glyphs: RunGlyph[] = [
        ...strip,
        {shape: 'crosshair', at: from, size: 12},
        {shape: 'ring', at: from, radius: 2 * estimates[3], dashed: true}
      ];
      let expectedOffspring = 0;
      for (let child = 0; child < inputs.count; child++) {
        const density = getDensity(row, child);
        if (density === 0) continue;
        const responsibility = density / getIntensity(child);
        expectedOffspring += responsibility;
        if (responsibility >= DRAWN_RESPONSIBILITY) {
          glyphs.push({
            shape: 'arrow',
            from,
            to: [inputs.positions[child * 2], inputs.positions[child * 2 + 1]]
          });
        }
      }
      // The most likely parent of the hovered event.
      const ownIntensity = getIntensity(row);
      let parentRow = -1;
      let parentShare = estimates[0] / ownIntensity;
      for (let parent = 0; parent < inputs.count; parent++) {
        const share = getDensity(parent, row) / ownIntensity;
        if (share > parentShare) {
          parentShare = share;
          parentRow = parent;
        }
      }
      const background = peek(outputs.background)?.[row];
      let parentText = 'no parent';
      if (parentRow >= 0) {
        parentText = `parent ${(inputs.days[row] - inputs.days[parentRow]).toFixed(1)} d earlier, ${(
          Math.hypot(
            inputs.positions[row * 2] - inputs.positions[parentRow * 2],
            inputs.positions[row * 2 + 1] - inputs.positions[parentRow * 2 + 1]
          ) * inputs.kilometersPerUnit
        ).toFixed(0)} km`;
        glyphs.push({
          shape: 'line',
          from: [inputs.positions[parentRow * 2], inputs.positions[parentRow * 2 + 1]],
          to: from,
          dashed: true
        });
      }
      return {
        focus: row,
        spotlight: {at: from, radius: 0.14},
        glyphs,
        readout: `P(background) ${formatValue(background)} · ${parentText} · M${inputs.magnitudes[row].toFixed(1)} · offspring ${expectedOffspring.toFixed(1)} · ${fitted}`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.background,
      ramp: 'diverging',
      range: [0, 1],
      radius: 4
    },
    drawPanel(outputs.panel, theme),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.bins,
      range: [0, inputs.maximumBinCount],
      rectangle: STRIP,
      color: [...theme.categories[0], 0.9]
    }
  ]
});

/** Expected events per bin from the fit: background plus the offspring of every earlier event. */
function getFittedStrip(inputs: Inputs, estimates: ArrayLike<number>, width: number): RunGlyph[] {
  const [left, bottom, right, top] = STRIP;
  const points: number[] = [];
  for (let bin = 0; bin < BIN_COUNT; bin++) {
    const start = bin * BIN_DAYS;
    const end = start + BIN_DAYS;
    let expected = estimates[0] * inputs.windowArea * BIN_DAYS;
    for (const day of inputs.days) {
      if (day >= end) continue;
      const from = Math.max(start, day);
      expected +=
        estimates[1] *
        (Math.exp(-estimates[2] * (from - day)) - Math.exp(-estimates[2] * (end - day)));
    }
    const x = (left + ((bin + 0.5) / BIN_COUNT) * (right - left)) * width;
    const y = bottom + Math.min(1, expected / inputs.maximumBinCount) * (top - bottom);
    points.push(x, y);
  }
  return [{shape: 'polyline', points}];
}
