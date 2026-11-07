// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUVariableStandardization of a skewed record column. Dots on the map take the transformed
// value (diverging about zero).

import {GPUVariableStandardization} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {createRecordColumn} from './_lib/helpers';
import {parameters} from './variable-standardization.parameters';

const ROW_COUNT = 1500;
const METHODS = [
  'deviation-from-mean',
  'z-score',
  'mad',
  'range-adjust',
  'range-standardize'
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Each record after the transformation, on a diverging ramp about zero.',
  createInputs: ({random, world}) => createUniformPoints(random, world, ROW_COUNT, 0.2),
  build(builder, points, values) {
    const column = builder.allocate('values', 'float32', ROW_COUNT);
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const standardized = builder.allocate('standardized', 'float32', ROW_COUNT);
    builder.graph.add(
      new GPUVariableStandardization({
        values: column.view,
        method: METHODS[values.method],
        variance: 'sample',
        output: standardized.view
      })
    );
    return {
      outputs: {column, positions, standardized},
      update: current =>
        column.write(
          createRecordColumn(
            Float32Array.from(points.values, value => Math.min(1, value * current.spread))
          )
        )
    };
  },
  rebuildOn: ['method'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      values: outputs.column,
      range: [0, 50],
      radius: 4.4
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.standardized,
      ramp: 'diverging',
      radius: 4.4
    }
  ]
});
