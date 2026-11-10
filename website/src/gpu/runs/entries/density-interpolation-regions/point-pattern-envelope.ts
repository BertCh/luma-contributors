// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointPatternEnvelope bounds 99 simulated complete-spatial-randomness curves of K(r) - pi r^2
// (grey) and tests the observed curve (gold) against them. The observed pattern leaves the
// envelope where it is clustered; `rank` widens or narrows the pointwise envelope.

import {GPUPointPatternEnvelope} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {makeRandom} from '../../../demos/kit';
import {createClusteredPoints} from '../../inputs';
import {getChartPosition, getChartWorldX} from '../../pick';
import {parameters} from './point-pattern-envelope.parameters';
import {computeKDeviation} from './support/ripley';

const RADII = Float32Array.from({length: 20}, (_, index) => 0.01 + index * 0.01);
const SIMULATION_COUNT = 99;
const POINT_COUNT = 60;
// The ranks the pointer can settle on; the default is the second.
const RANKS = [1, 3, 5, 7, 10] as const;
const RECTANGLE = [0.07, 0.14, 0.95, 0.88] as const;
const RANGE = [-0.03, 0.1] as const;
const STATISTICS = ['maximum-deviation', 'integrated-squared-deviation', 'extreme-rank'] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Japanese earthquake epicenters against 99 simulated patterns of complete spatial randomness.',
  createInputs: ({world}) => {
    // The 99 simulated curves do not depend on any slider: compute them once.
    const random = makeRandom(5);
    const simulations = new Float32Array(SIMULATION_COUNT * RADII.length);
    for (let replicate = 0; replicate < SIMULATION_COUNT; replicate++) {
      const points = Float32Array.from({length: POINT_COUNT * 2}, (_, index) =>
        index % 2 === 0 ? random() * world.width : random()
      );
      simulations.set(computeKDeviation(points, RADII, world.width), replicate * RADII.length);
    }
    return {world, simulations};
  },
  build(builder, {world, simulations}, values) {
    const random = makeRandom(6);
    const area = world.width;
    const uniform = () => [random() * world.width, random()];
    const observedPoints = createClusteredPoints(random, world, POINT_COUNT, 1, 0).positions;
    for (let point = 0; point < POINT_COUNT; point++) {
      if (random() > values.clustering) observedPoints.set(uniform(), point * 2);
    }
    const observed = builder.upload(
      'observed',
      'float32',
      computeKDeviation(observedPoints, RADII, area)
    );
    const lower = builder.allocate('lower', 'float32', RADII.length);
    const upper = builder.allocate('upper', 'float32', RADII.length);
    builder.graph.add(
      new GPUPointPatternEnvelope({
        observed: observed.view,
        simulations: builder.upload('simulations', 'float32', simulations).view,
        reference: builder.upload('reference', 'float32', new Float32Array(RADII.length)).view,
        integrationWeights: builder.upload(
          'weights',
          'float32',
          new Float32Array(RADII.length).fill(0.01)
        ).view,
        simulationCount: SIMULATION_COUNT,
        rank: values.rank,
        statistic: STATISTICS[values.statistic],
        nullModel: 'complete spatial randomness',
        fittedParameters: 'none',
        output: {
          lower: lower.view,
          upper: upper.view,
          statistics: builder.allocate('statistics', 'float32', 4).view,
          status: builder.allocate('status', 'uint32', 1).view,
          pValueInterval:
            values.statistic === 2 ? builder.allocate('p-interval', 'float32', 2).view : undefined
        }
      })
    );
    return {
      outputs: {
        observed,
        lower,
        upper,
        zero: builder.upload('zero', 'float32', new Float32Array(RADII.length))
      }
    };
  },
  buildCache: 5,
  rebuildOn: ['rank', 'statistic', 'clustering'],
  // x reads the curves at one radius r (the rule); y chooses the rank of the envelope, which is
  // rebuilt once the pointer rests on one of five ranks.
  interaction: {
    kind: 'level',
    hint: 'read r, pick the rank',
    rest: [0.5, 1 - 1.5 / RANKS.length],
    resolve(pointer, {outputs, values, world, peek}) {
      const index = Math.round(
        getChartPosition(RECTANGLE, world, pointer.x, pointer.y).u * (RADII.length - 1)
      );
      const rank =
        RANKS[Math.min(RANKS.length - 1, Math.floor((1 - pointer.unitY) * RANKS.length))];
      const observed = peek(outputs.observed)?.[index];
      const lower = peek(outputs.lower)?.[index];
      const upper = peek(outputs.upper)?.[index];
      const verdict =
        observed === undefined || lower === undefined || upper === undefined
          ? ''
          : observed > upper
            ? ' above envelope'
            : observed < lower
              ? ' below envelope'
              : ' inside envelope';
      return {
        settle: {rank},
        glyphs: [
          {shape: 'curtain', x: getChartWorldX(RECTANGLE, world, index / (RADII.length - 1))}
        ],
        readout: `r ${RADII[index].toFixed(2)} · observed ${observed?.toFixed(3) ?? '...'}${verdict} · rank ${rank === values.rank ? rank : `${values.rank} → ${rank}`}`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'chart',
      kind: 'line',
      context: true,
      values: outputs.zero,
      range: RANGE,
      rectangle: RECTANGLE,
      color: [0.5, 0.5, 0.5, 0.5]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.lower,
      range: RANGE,
      rectangle: RECTANGLE,
      color: [0.6, 0.75, 0.9, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.upper,
      range: RANGE,
      rectangle: RECTANGLE,
      color: [0.6, 0.75, 0.9, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.observed,
      range: RANGE,
      rectangle: RECTANGLE,
      color: [1, 0.85, 0.3, 1]
    }
  ]
});
