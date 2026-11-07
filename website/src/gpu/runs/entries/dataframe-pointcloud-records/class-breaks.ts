// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUClassBreaks on a heavy-tailed column. The bars are the row count of each class the chosen
// method produced (GPU-written classCounts); the sorted column above them is the input.

import {
  GPUClassBreaks,
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createRowAxis, createSkewedSource, getSkewedValues} from './_lib/helpers';
import {addIntegerToFloat} from './_lib/convert';
import {parameters} from './class-breaks.parameters';

const MAXIMUM_CLASS_COUNT = 9;
const ROW_COUNT = 1500;
const METHODS = [
  'equal-interval',
  'quantile',
  'standard-deviation',
  'natural-breaks',
  'head-tail'
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Class sizes per method; the sorted values are the input column.',
  createInputs: ({random}) => ({
    source: createSkewedSource(random, ROW_COUNT),
    rowAxis: createRowAxis(ROW_COUNT)
  }),
  build(builder, inputs, values) {
    const column = builder.allocate('values', 'float32', ROW_COUNT);
    const sorted = builder.allocate('sorted', 'float32', ROW_COUNT);
    const rowAxis = builder.upload('row-axis', 'float32', inputs.rowAxis);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      getGPUClassBreaksParameterLength(MAXIMUM_CLASS_COUNT)
    );
    const breaks = builder.allocate('breaks', 'float32', MAXIMUM_CLASS_COUNT + 1);
    const classCount = builder.allocate('class-count', 'uint32', 1);
    const classCounts = builder.allocate('class-counts', 'uint32', MAXIMUM_CLASS_COUNT);
    builder.graph.add(
      new GPUClassBreaks({
        values: column.view,
        parameters: parameterBuffer.view,
        maximumClassCount: MAXIMUM_CLASS_COUNT,
        methods: METHODS,
        output: {breaks: breaks.view, classCount: classCount.view, classCounts: classCounts.view}
      })
    );
    const classSizes = addIntegerToFloat(builder, 'class-sizes', classCounts);
    return {
      outputs: {column, sorted, rowAxis, classSizes},
      update: current => {
        const data = getSkewedValues(inputs.source, current.spread);
        column.write(data);
        sorted.write(Float32Array.from(data).sort());
        parameterBuffer.write(
          getGPUClassBreaksParameterValues(
            {method: METHODS[current.method], classCount: current.classCount},
            MAXIMUM_CLASS_COUNT
          )
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
      values: outputs.sorted,
      xValues: outputs.rowAxis,
      xRange: [0, ROW_COUNT],
      rectangle: [0.07, 0.56, 0.95, 0.92],
      radius: 1.5
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.classSizes,
      color: [0.2, 0.55, 0.9, 1],
      range: [0, ROW_COUNT / 2],
      rectangle: [0.07, 0.1, 0.95, 0.5]
    }
  ]
});
