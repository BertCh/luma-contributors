// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURipleyEnvelope next to the observed GPURipley. The band between the two thin lines is the
// range of L(r) - r that random (CSR) patterns of the same size reach; the bright curve is the
// observed pattern. Where it leaves the band the pattern is clustered or regular at that radius.

import {
  GPURipley,
  GPURipleyEnvelope,
  GPU_RIPLEY_PARAMETER_LENGTH,
  getGPURipleyParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createPatternSource, getPatternPositions} from './_lib/helpers';
import {parameters} from './ripley-envelope.parameters';

const POINT_COUNT = 400;
const RADIUS_COUNT = 24;
const PLOT = [0.58, 0.12, 0.96, 0.88] as const;
const RANGE = [-0.02, 0.05] as const;

export default defineOperationRun({
  parameters,
  caption: 'Observed L(r) - r against the band from random patterns.',
  createInputs: ({random, world}) => ({
    source: createPatternSource(random, POINT_COUNT),
    windowWidth: world.width * 0.5
  }),
  build(builder, inputs, values) {
    const positions = builder.allocate('positions', 'float32x2', POINT_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', GPU_RIPLEY_PARAMETER_LENGTH);
    const k = builder.allocate('k', 'float32', RADIUS_COUNT);
    const observed = builder.allocate('observed', 'float32', RADIUS_COUNT);
    builder.graph.add(
      new GPURipley({
        positions: positions.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        radiusCount: RADIUS_COUNT,
        output: {k: k.view, lMinusR: observed.view}
      })
    );
    const lower = builder.allocate('lower', 'float32', RADIUS_COUNT);
    const upper = builder.allocate('upper', 'float32', RADIUS_COUNT);
    builder.graph.add(
      new GPURipleyEnvelope({
        positions: positions.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        radiusCount: RADIUS_COUNT,
        simulationCount: values.simulationCount,
        rank: values.rank,
        statistic: 'l-minus-r',
        observedK: k.view,
        output: {lower: lower.view, upper: upper.view}
      })
    );
    return {
      outputs: {positions, observed, lower, upper},
      update: current => {
        positions.write(
          getPatternPositions(inputs.source, inputs.windowWidth, 1, current.clustering)
        );
        parameterBuffer.write(
          getGPURipleyParameterValues({
            bounds: [0, 0, inputs.windowWidth, 1],
            maximumDistance: 0.2,
            edgeCorrection: 'isotropic'
          })
        );
      }
    };
  },
  rebuildOn: ['simulationCount', 'rank'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      color: [0.75, 0.8, 0.9, 0.8],
      radius: 2.2
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.upper,
      range: RANGE,
      rectangle: PLOT,
      color: [0.55, 0.6, 0.7, 0.45]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.lower,
      range: RANGE,
      rectangle: PLOT,
      color: [0.04, 0.06, 0.1, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.observed,
      range: RANGE,
      rectangle: PLOT,
      radius: 2.6,
      color: [0.3, 0.75, 1, 1]
    }
  ]
});
