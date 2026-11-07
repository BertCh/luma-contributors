// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointPatternEnvelope bounds 99 simulated complete-spatial-randomness curves of K(r) - pi r^2
// (grey) and tests the observed curve (gold) against them. The observed pattern leaves the
// envelope where it is clustered; `rank` widens or narrows the pointwise envelope.

import {GPUPointPatternEnvelope} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {makeRandom} from '../../../demos/kit';
import {parameters} from './point-pattern-envelope.parameters';
import {computeKDeviation} from './support/ripley';

const RADII = Float32Array.from({length: 20}, (_, index) => 0.01 + index * 0.01);
const SIMULATION_COUNT = 99;
const POINT_COUNT = 60;
const STATISTICS = ['maximum-deviation', 'integrated-squared-deviation', 'extreme-rank'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Observed K deviation against the envelope of 99 random patterns.',
  createInputs: ({world}) => ({world}),
  build(builder, {world}, values) {
    const random = makeRandom(5);
    const area = world.width;
    const uniform = () => [random() * world.width, random()];
    const simulations = new Float32Array(SIMULATION_COUNT * RADII.length);
    for (let replicate = 0; replicate < SIMULATION_COUNT; replicate++) {
      const points = Float32Array.from(
        {length: POINT_COUNT * 2},
        (_, index) => uniform()[index % 2]
      );
      simulations.set(computeKDeviation(points, RADII, area), replicate * RADII.length);
    }
    const centers = [uniform(), uniform(), uniform()];
    const observedPoints = new Float32Array(POINT_COUNT * 2);
    for (let point = 0; point < POINT_COUNT; point++) {
      const [centerX, centerY] = centers[point % 3];
      const clustered = random() < values.clustering;
      observedPoints[point * 2] = clustered
        ? centerX + (random() - 0.5) * 0.12
        : random() * world.width;
      observedPoints[point * 2 + 1] = clustered ? centerY + (random() - 0.5) * 0.12 : random();
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
  draw: ({outputs}) => [
    {
      layer: 'chart',
      kind: 'line',
      context: true,
      values: outputs.zero,
      range: [-0.03, 0.1],
      color: [0.5, 0.5, 0.5, 0.5]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.lower,
      range: [-0.03, 0.1],
      color: [0.6, 0.75, 0.9, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.upper,
      range: [-0.03, 0.1],
      color: [0.6, 0.75, 0.9, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.observed,
      range: [-0.03, 0.1],
      color: [1, 0.85, 0.3, 1]
    }
  ]
});
