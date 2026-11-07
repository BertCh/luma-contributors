// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUScalarScale mapping a record column to line widths. Each dot is one record: x is the input
// value, y the width the GPU scale wrote. The faint diagonal is the unscaled input.

import {
  GPUScalarScale,
  getGPUScalarScaleParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {createRecordColumn} from './_lib/helpers';
import {parameters} from './scalar-scale.parameters';

const ROW_COUNT = 1200;
const SCALES = ['linear', 'sqrt', 'log'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Input value (x) against the width the scale wrote (y).',
  createInputs: ({random, world}) =>
    createRecordColumn(createUniformPoints(random, world, ROW_COUNT, 0.3).values, 80),
  build(builder, column) {
    const input = builder.upload('values', 'float32', column);
    const domain = builder.allocate('domain', 'float32', 2);
    const range = builder.allocate('range', 'float32', 2);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const widths = builder.allocate('widths', 'float32', ROW_COUNT);
    builder.graph.add(
      new GPUScalarScale({
        values: input.view,
        domain: domain.view,
        range: range.view,
        parameters: parameterBuffer.view,
        maximumDomainCount: 2,
        output: {values: widths.view}
      })
    );
    return {
      outputs: {input, widths},
      update: current => {
        domain.write(Float32Array.of(1, current.domainMaximum));
        range.write(Float32Array.of(1, current.rangeMaximum));
        parameterBuffer.write(
          getGPUScalarScaleParameterValues({scale: SCALES[current.scale], domainCount: 2})
        );
      }
    };
  },
  rebuildOn: [],
  draw: ({outputs}) => [
    {
      layer: 'chart',
      kind: 'scatter',
      context: true,
      values: outputs.input,
      xValues: outputs.input,
      xRange: [0, 80],
      radius: 2,
      range: [0, 80],
      rectangle: [0.08, 0.1, 0.94, 0.9]
    },
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.widths,
      xValues: outputs.input,
      xRange: [0, 80],
      range: [0, 30],
      rectangle: [0.08, 0.1, 0.94, 0.9],
      radius: 7,
      color: [0.2, 0.7, 1, 1]
    }
  ]
});
