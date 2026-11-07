// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURipleyDistanceFunctions: G(r), the share of events whose nearest neighbor is within r (cyan),
// and F(r), the share of empty locations within r of an event (orange). Clustered patterns have
// G above F; the J curve (white) falls below 1.

import {
  GPURipleyDistanceFunctions,
  GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH,
  getGPURipleyDistanceFunctionsParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createPatternSource, getPatternPositions} from './_lib/helpers';
import {parameters} from './ripley-distance-functions.parameters';

const POINT_COUNT = 500;
const PLOT = [0.58, 0.12, 0.96, 0.88] as const;
const CORRECTIONS = ['none', 'border', 'kaplan-meier', 'hanisch'] as const;

export default defineOperationRun({
  parameters,
  caption: 'G (cyan), F (orange) and J (white) against distance.',
  createInputs: ({random, world}) => ({
    source: createPatternSource(random, POINT_COUNT),
    windowWidth: world.width * 0.5
  }),
  build(builder, inputs, values) {
    const positions = builder.allocate('positions', 'float32x2', POINT_COUNT);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH
    );
    const g = builder.allocate('g', 'float32', values.radiusCount);
    const f = builder.allocate('f', 'float32', values.radiusCount);
    const j = builder.allocate('j', 'float32', values.radiusCount);
    builder.graph.add(
      new GPURipleyDistanceFunctions({
        positions: positions.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        referenceGrid: [32, 32],
        radiusCount: values.radiusCount,
        output: {g: g.view, f: f.view, j: j.view}
      })
    );
    return {
      outputs: {positions, g, f, j},
      update: current => {
        positions.write(
          getPatternPositions(inputs.source, inputs.windowWidth, 1, current.clustering)
        );
        parameterBuffer.write(
          getGPURipleyDistanceFunctionsParameterValues({
            bounds: [0, 0, inputs.windowWidth, 1],
            maximumDistance: current.maximumDistance,
            edgeCorrection: CORRECTIONS[current.edgeCorrection]
          })
        );
      }
    };
  },
  rebuildOn: ['radiusCount'],
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
      values: outputs.g,
      range: [0, 1],
      rectangle: PLOT,
      color: [0.3, 0.75, 1, 0.3]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.g,
      range: [0, 1],
      rectangle: PLOT,
      radius: 2.6,
      color: [0.3, 0.75, 1, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.f,
      range: [0, 1],
      rectangle: PLOT,
      radius: 2.6,
      color: [1, 0.6, 0.2, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.j,
      range: [0, 2],
      rectangle: PLOT,
      radius: 1.4,
      color: [1, 1, 1, 0.8]
    }
  ]
});
