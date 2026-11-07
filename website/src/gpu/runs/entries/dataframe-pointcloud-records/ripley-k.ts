// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURipley on a point pattern in the left panel. The curve is L(r) - r written by the GPU: it
// stays near zero for a random pattern and rises above it where points cluster at that radius.

import {
  GPURipley,
  GPU_RIPLEY_PARAMETER_LENGTH,
  getGPURipleyParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createPatternSource, getPatternPositions} from './_lib/helpers';
import {parameters} from './ripley-k.parameters';

const POINT_COUNT = 500;
const EDGE_CORRECTIONS = ['none', 'border', 'isotropic'] as const;

export default defineOperationRun({
  parameters,
  caption: 'L(r) - r from Ripley’s K; the window is the left panel.',
  createInputs: ({random, world}) => ({
    source: createPatternSource(random, POINT_COUNT),
    windowWidth: world.width * 0.5
  }),
  build(builder, inputs, values) {
    const positions = builder.allocate('positions', 'float32x2', POINT_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', GPU_RIPLEY_PARAMETER_LENGTH);
    const k = builder.allocate('k', 'float32', values.radiusCount);
    const lMinusR = builder.allocate('l-minus-r', 'float32', values.radiusCount);
    builder.graph.add(
      new GPURipley({
        positions: positions.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        radiusCount: values.radiusCount,
        output: {k: k.view, lMinusR: lMinusR.view}
      })
    );
    return {
      outputs: {positions, lMinusR},
      update: current => {
        positions.write(
          getPatternPositions(inputs.source, inputs.windowWidth, 1, current.clustering)
        );
        parameterBuffer.write(
          getGPURipleyParameterValues({
            bounds: [0, 0, inputs.windowWidth, 1],
            maximumDistance: current.maximumDistance,
            edgeCorrection: EDGE_CORRECTIONS[current.edgeCorrection]
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
      values: outputs.lMinusR,
      range: [-0.02, 0.08],
      rectangle: [0.58, 0.12, 0.96, 0.88],
      color: [0.3, 0.75, 1, 0.35]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.lMinusR,
      range: [-0.02, 0.08],
      rectangle: [0.58, 0.12, 0.96, 0.88],
      radius: 2.4,
      color: [0.3, 0.75, 1, 1]
    }
  ]
});
