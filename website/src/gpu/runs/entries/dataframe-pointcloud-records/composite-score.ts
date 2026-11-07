// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCompositeScore over four indicators per tract. Tracts are filled by the composite score; the
// dots show the first indicator alone.

import {
  GPUCompositeScore,
  GPU_COMPOSITE_SCORE_PARAMETER_LENGTH,
  getGPUCompositeScoreParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {uploadTractGeometry} from './_lib/helpers';
import {parameters} from './composite-score.parameters';

const SCALERS = ['min-max', 'z-score', 'rank'] as const;
const AGGREGATIONS = ['weighted-sum', 'weighted-geometric-mean', 'principal-component'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Tracts filled by the composite of four indicators.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world, 14, 0.1);
    const indicators = new Float32Array(tracts.count * 4);
    for (let row = 0; row < tracts.count; row++) {
      const x = tracts.centers[row * 2] / world.width;
      const y = tracts.centers[row * 2 + 1];
      indicators[row * 4] = tracts.values[row];
      indicators[row * 4 + 1] = 0.5 + 0.5 * Math.sin(x * 7) * Math.cos(y * 4);
      indicators[row * 4 + 2] = 1 - x * 0.9 + (tracts.values[row] - 0.5) * 0.2;
      indicators[row * 4 + 3] = 0.2 + 0.8 * y * y;
    }
    return {tracts, indicators};
  },
  build(builder, inputs) {
    const {tracts} = inputs;
    const indicators = builder.upload('indicators', 'float32', inputs.indicators);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_COMPOSITE_SCORE_PARAMETER_LENGTH
    );
    const score = builder.allocate('score', 'float32', tracts.count);
    builder.graph.add(
      new GPUCompositeScore({
        indicators: indicators.view,
        indicatorCount: 4,
        enableRank: true,
        enablePrincipalComponent: true,
        parameters: parameterBuffer.view,
        output: {score: score.view}
      })
    );
    return {
      outputs: {
        ...uploadTractGeometry(builder, tracts),
        score,
        centers: builder.upload('centers', 'float32x2', tracts.centers),
        firstIndicator: builder.upload('first', 'float32', Float32Array.from(tracts.values))
      },
      update: current =>
        parameterBuffer.write(
          getGPUCompositeScoreParameterValues({
            scaler: SCALERS[current.scaler],
            aggregation: AGGREGATIONS[current.aggregation],
            weights: [1, current.weight, 1, 1],
            directions: [1, 1, -1, 1]
          })
        )
    };
  },
  rebuildOn: [],
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.score,
      triangleCount: inputs.tracts.triangleCount
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount},
    {
      layer: 'dots',
      context: true,
      positions: outputs.centers,
      values: outputs.firstIndicator,
      range: [0, 1],
      radius: 3
    }
  ]
});
