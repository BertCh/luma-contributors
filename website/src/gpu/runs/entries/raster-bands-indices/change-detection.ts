// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUChangeDetection over twelve dates per cell. A region of the map steps up (or down) at date 6;
// the operation compares before and after (difference, percent change), tests every cell (t
// statistic, Sen slope, Mann-Kendall) and classes it as a significant increase, decrease or no
// change at `alpha`. The split date, alpha and dates are per-frame parameters.

import {
  getGPUChangeDetectionParameterValues,
  GPUChangeDetection
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './change-detection.parameters';
import {createStackInput} from './support/stacks';

const SLICE_COUNT = 12;
const STATISTICS = [
  'difference',
  'percentChange',
  'tStatistic',
  'senSlope',
  'mannKendallZ',
  'significance'
] as const;

export default defineOperationRun({
  parameters,
  caption: 'A time series per cell in; change statistics and a significance class out.',
  createInputs: ({random, aspect}) => createStackInput(random, aspect, 40, SLICE_COUNT),
  build(builder, inputs) {
    const {cellCount, sliceCount} = inputs;
    const slices = builder.allocate('slices', 'float32', cellCount * sliceCount);
    const parameterBuffer = builder.allocate('parameters', 'float32', 5);
    const output = {
      difference: builder.allocate('difference', 'float32', cellCount),
      percentChange: builder.allocate('percent-change', 'float32', cellCount),
      tStatistic: builder.allocate('t-statistic', 'float32', cellCount),
      senSlope: builder.allocate('sen-slope', 'float32', cellCount),
      mannKendallZ: builder.allocate('mann-kendall-z', 'float32', cellCount),
      significance: builder.allocate('significance', 'uint32', cellCount)
    };
    builder.graph.add(
      new GPUChangeDetection({
        slices: slices.view,
        parameters: parameterBuffer.view,
        cellCount,
        sliceCount,
        output: {
          difference: output.difference.view,
          percentChange: output.percentChange.view,
          tStatistic: output.tStatistic.view,
          senSlope: output.senSlope.view,
          mannKendallZ: output.mannKendallZ.view,
          significance: output.significance.view
        }
      })
    );
    return {
      outputs: output,
      update: current => {
        const data = new Float32Array(cellCount * sliceCount);
        for (let cell = 0; cell < cellCount; cell++) {
          // Cells where the region field is high step up at date 6; low ones step down.
          const direction = inputs.region[cell] > 0.65 ? 1 : inputs.region[cell] < 0.35 ? -1 : 0;
          for (let slice = 0; slice < sliceCount; slice++) {
            data[cell * sliceCount + slice] =
              0.4 +
              0.1 * inputs.level[cell] +
              (slice >= 6 ? direction * current.magnitude : 0) +
              0.08 * inputs.noise[cell * sliceCount + slice];
          }
        }
        slices.write(data);
        parameterBuffer.write(
          getGPUChangeDetectionParameterValues({
            beforeSlice: 2,
            afterSlice: 9,
            alpha: current.alpha,
            splitSlice: current.splitSlice
          })
        );
      }
    };
  },
  draw: ({inputs, outputs, values}) => {
    const statistic = STATISTICS[Math.round(values.statistic)];
    return [
      {
        layer: 'raster',
        values: outputs[statistic],
        columns: inputs.columns,
        rows: inputs.rows,
        ramp: statistic === 'significance' ? 'categorical' : 'diverging',
        smoothing: statistic !== 'significance'
      }
    ];
  }
});
