// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMultipleTestingAdjustment: each p-value (x) against its adjusted value (y). The diagonal is
// "no adjustment". Bonferroni and Holm lift every value; the Benjamini procedures lift the
// large p-values most and keep the strong signals (bottom left) significant.

import {GPUMultipleTestingAdjustment} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './multiple-testing-adjustment.parameters';

const METHODS = [
  'none',
  'bonferroni',
  'holm',
  'benjamini-hochberg',
  'benjamini-yekutieli'
] as const;
const MAXIMUM_TESTS = 400;
const PLOT = [0.1, 0.1, 0.9, 0.92] as const;

export default defineOperationRun({
  parameters,
  caption: 'Raw against adjusted p-values for the chosen correction.',
  createInputs: ({random, world}) => ({
    uniform: Float32Array.from({length: MAXIMUM_TESTS}, () => random()),
    // The diagonal of the plot: adjusted equals raw.
    diagonal: Float32Array.of(
      PLOT[0] * world.width,
      PLOT[1] * world.height,
      PLOT[2] * world.width,
      PLOT[3] * world.height
    )
  }),
  build(builder, inputs, values) {
    const testCount = values.testCount;
    const pValues = builder.allocate('p-values', 'float32', testCount);
    const adjustedPValues = builder.allocate('adjusted-p-values', 'float32', testCount);
    const testedCount = builder.allocate('tested-count', 'uint32', 1);
    builder.graph.add(
      new GPUMultipleTestingAdjustment({
        pValues: pValues.view,
        method: METHODS[values.method],
        output: {adjustedPValues: adjustedPValues.view, testedCount: testedCount.view}
      })
    );
    return {
      outputs: {
        pValues,
        adjustedPValues,
        diagonal: builder.upload('diagonal', 'float32x4', inputs.diagonal)
      },
      update: current => {
        // The first signalShare of the tests are real effects: their p-values are pushed toward 0.
        const signalCount = Math.round(current.signalShare * testCount);
        pValues.write(
          Float32Array.from({length: testCount}, (_, index) =>
            index < signalCount ? inputs.uniform[index] ** 5 : inputs.uniform[index]
          )
        );
      }
    };
  },
  rebuildOn: ['method', 'testCount'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.diagonal, radius: 1},
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.adjustedPValues,
      xValues: outputs.pValues,
      xRange: [0, 1],
      range: [0, 1],
      color: [0.4, 0.68, 1, 0.9],
      radius: 5.5,
      rectangle: PLOT
    }
  ]
});
