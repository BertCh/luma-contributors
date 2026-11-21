// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUEnsembleScores: the sample CRPS of an ensemble at every station. Observation: Helene day-two
// rain (log residual). Ensemble: day-one rain plus spatially correlated perturbations (scenario
// modification). Left: stations coloured by CRPS (brighter is worse). Right: the probed station's
// members as ticks on a number line and as an empirical CDF staircase against the observation's
// step (dashed); CRPS is the area between them. Over the panel, height picks the station.

import {GPUEnsembleScores} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuilder} from '../../define-run';
import {formatNumber, type RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {createCorrelatedFields, createForecastScene} from './_lib/forecast';
import {getFrameGlyph} from './_lib/surface';
import {parameters} from './ensemble-scores.parameters';

const STATION_COUNT = 360;
const MAXIMUM_MEMBERS = 32;
/** Number line range, log-rain residual. */
const LINE_RANGE = 5;
const TICKS = [-4, -3, -2, -1, 0, 1, 2, 3, 4];

type Inputs = ReturnType<typeof createInputs>;

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const scene = createForecastScene(random, world, STATION_COUNT);
  const lineY = (scene.panel.y0 + scene.panel.y1) / 2;
  const toX = (value: number) =>
    scene.panel.x0 +
    ((Math.max(-LINE_RANGE, Math.min(LINE_RANGE, value)) + LINE_RANGE) / (2 * LINE_RANGE)) *
      (scene.panel.x1 - scene.panel.x0);
  const axis = new Float32Array((TICKS.length + 1) * 4);
  axis.set([toX(-LINE_RANGE), lineY, toX(LINE_RANGE), lineY]);
  TICKS.forEach((tick, index) =>
    axis.set([toX(tick), lineY - 0.012, toX(tick), lineY + 0.012], (index + 1) * 4)
  );
  return {
    ...scene,
    lineY,
    toX,
    axis,
    fields: createCorrelatedFields(random, scene.positions, MAXIMUM_MEMBERS, 0.22),
    index: createPointIndex(scene.positions),
    order: Uint32Array.from({length: scene.count}, (_, station) => station).sort(
      (first, second) => scene.positions[first * 2 + 1] - scene.positions[second * 2 + 1]
    ),
    probe: scene.defaultStation
  };
}

function getMember(inputs: Inputs, spread: number, member: number, station: number): number {
  return inputs.baseline[station] + spread * inputs.fields[member * inputs.count + station];
}

export default defineOperationRun({
  parameters,
  caption:
    'Sample CRPS of Helene day-two rain at Carolina stations; ensemble = day-one rain plus correlated perturbations (scenario).',
  createInputs,
  rebuildOn: ['memberCount', 'fair'],
  build(builder: RunBuilder, inputs: Inputs, values: Record<string, number>) {
    const memberCount = values.memberCount;
    const ensemble = builder.allocate('ensemble', 'float32', memberCount * inputs.count);
    const crps = builder.allocate('crps', 'float32', inputs.count);
    builder.graph.add(
      new GPUEnsembleScores({
        ensemble: ensemble.view,
        observations: builder.upload('observations', 'float32', inputs.residuals).view,
        memberCount,
        dimension: inputs.count,
        estimator: values.fair === 1 ? 'fair' : 'empirical',
        output: {crps: crps.view}
      })
    );
    return {
      outputs: {
        crps,
        positions: builder.upload('positions', 'float32x2', inputs.positions),
        axis: builder.upload('axis', 'float32x4', inputs.axis)
      },
      update: (current: Record<string, number>) => {
        const rows = new Float32Array(memberCount * inputs.count);
        for (let member = 0; member < memberCount; member++) {
          for (let station = 0; station < inputs.count; station++) {
            rows[member * inputs.count + station] = getMember(
              inputs,
              current.spread,
              member,
              station
            );
          }
        }
        ensemble.write(rows);
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a station',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.positions[inputs.defaultStation * 2],
        inputs.positions[inputs.defaultStation * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, values, pixelSize, peek}) {
      if (pointer.x < inputs.mapWidth) {
        inputs.probe = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.probe);
      } else {
        // Over the panel, the height picks the station: stations ordered from south to north.
        const fraction = (pointer.y - inputs.panel.y0) / (inputs.panel.y1 - inputs.panel.y0);
        const rank = Math.round(Math.min(1, Math.max(0, fraction)) * (inputs.order.length - 1));
        inputs.probe = inputs.order[rank];
      }
      const station = inputs.probe;
      const at: [number, number] = [
        inputs.positions[station * 2],
        inputs.positions[station * 2 + 1]
      ];
      const glyphs: RunGlyph[] = [
        getFrameGlyph(inputs.panel, true),
        {shape: 'ring', at, radius: 8 * pixelSize}
      ];
      const lineY = inputs.lineY;
      const memberValues = Array.from({length: values.memberCount}, (_, member) =>
        getMember(inputs, values.spread, member, station)
      ).sort((first, second) => first - second);
      const stairBottom = lineY + 0.1;
      const stairHeight = 0.28;
      const stairTop = stairBottom + stairHeight;
      const stairs: number[] = [inputs.toX(-LINE_RANGE), stairBottom];
      memberValues.forEach((value, member) => {
        const x = inputs.toX(value);
        glyphs.push({shape: 'line', from: [x, lineY], to: [x, lineY + 0.07]});
        stairs.push(
          x,
          stairBottom + (member / memberValues.length) * stairHeight,
          x,
          stairBottom + ((member + 1) / memberValues.length) * stairHeight
        );
      });
      stairs.push(inputs.toX(LINE_RANGE), stairTop);
      const observedX = inputs.toX(inputs.residuals[station]);
      glyphs.push(
        {shape: 'polyline', points: stairs},
        {
          shape: 'polyline',
          points: [
            inputs.toX(-LINE_RANGE),
            stairBottom,
            observedX,
            stairBottom,
            observedX,
            stairTop,
            inputs.toX(LINE_RANGE),
            stairTop
          ],
          dashed: true
        },
        {shape: 'line', from: [observedX, lineY - 0.06], to: [observedX, stairBottom]},
        {shape: 'ring', at: [observedX, lineY], radius: 6 * pixelSize}
      );
      const scores = peek(outputs.crps);
      let meanScore = 0;
      if (scores) {
        for (let row = 0; row < inputs.count; row++) meanScore += scores[row];
        meanScore /= inputs.count;
      }
      return {
        glyphs,
        spotlight: null,
        readout: scores
          ? `CRPS ${formatNumber(scores[station], 2)} here · mean ${formatNumber(meanScore, 2)}${
              values.fair === 1 ? ' · fair' : ''
            }`
          : `station ${station}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.axis, alpha: 0.8},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.crps,
      ramp: 'sequential',
      range: [0, 2.5],
      radius: 5
    }
  ]
});
