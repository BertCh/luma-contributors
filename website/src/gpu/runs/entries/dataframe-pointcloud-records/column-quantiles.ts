// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUColumnQuantiles of a record column. The bars are the quantile values the GPU selected; the
// map shows which rows the percentile filter (outputMask) keeps.

import {
  GPUColumnQuantiles,
  getGPUColumnQuantilesParameterLength,
  getGPUColumnQuantilesParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {createRecordColumn} from './_lib/helpers';
import {parameters} from './column-quantiles.parameters';

const ROW_COUNT = 2000;

export default defineOperationRun({
  parameters,
  caption: 'Quantile bars, and the rows kept by the percentile filter.',
  createInputs: ({random, world}) => {
    const points = createUniformPoints(random, world, ROW_COUNT, 0.15);
    return {points, column: createRecordColumn(points.values)};
  },
  build(builder, inputs, values) {
    const quantileCount = values.quantileCount;
    const column = builder.upload('fare', 'float32', inputs.column);
    const positions = builder.upload('positions', 'float32x2', inputs.points.positions);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      getGPUColumnQuantilesParameterLength(quantileCount)
    );
    const quantiles = builder.allocate('quantiles', 'float32', quantileCount);
    const outputMask = builder.allocate('output-mask', 'uint32', ROW_COUNT);
    builder.graph.add(
      new GPUColumnQuantiles({
        values: column.view,
        quantileCount,
        parameters: parameterBuffer.view,
        output: {
          quantiles: quantiles.view,
          validCount: builder.allocate('valid-count', 'uint32', 1).view,
          outputMask: outputMask.view
        }
      })
    );
    return {
      outputs: {column, positions, quantiles, outputMask},
      update: current =>
        parameterBuffer.write(
          getGPUColumnQuantilesParameterValues({
            quantiles: Array.from(
              {length: quantileCount},
              (_, index) => (index + 1) / (quantileCount + 1)
            ),
            filterRange: [current.filterLower / 100, current.filterUpper / 100]
          })
        )
    };
  },
  rebuildOn: ['quantileCount'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      values: outputs.column,
      range: [0, 50],
      radius: 3
    },
    {layer: 'dots', positions: outputs.positions, values: outputs.outputMask, radius: 2.6},
    {
      layer: 'chart',
      kind: 'bars',
      color: [0.2, 0.55, 0.9, 1],
      values: outputs.quantiles,
      range: [0, 35],
      rectangle: [0.62, 0.06, 0.97, 0.4]
    }
  ]
});
