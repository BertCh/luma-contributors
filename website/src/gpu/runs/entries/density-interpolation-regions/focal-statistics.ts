// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUFocalStatistics summarizes a moving window around every cell. The operation writes mean,
// min, max, range and standard deviation in one pass; `output` picks which one is drawn.

import {
  getGPUFocalStatisticsParameterValues,
  GPUFocalStatistics
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './focal-statistics.parameters';
import {createRunGrid} from './support/grid';

export default defineOperationRun({
  parameters,
  caption: 'Window statistic of a terrain-like raster; the muted raster is the input.',
  createInputs: ({random, world}) => ({
    grid: createRunGrid(world, 72),
    raster: createFractalRaster(random, world, 72, 5)
  }),
  build(builder, {grid, raster}) {
    const cellCount = grid.columns * grid.rows;
    const input = builder.upload('input', 'float32', raster.values);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const statistics = {
      mean: builder.allocate('mean', 'float32', cellCount),
      min: builder.allocate('min', 'float32', cellCount),
      max: builder.allocate('max', 'float32', cellCount),
      range: builder.allocate('range', 'float32', cellCount),
      standardDeviation: builder.allocate('standard-deviation', 'float32', cellCount)
    };
    builder.graph.add(
      new GPUFocalStatistics({
        values: input.view,
        grid: grid.grid,
        maximumRadius: 8,
        parameters: parameterBuffer.view,
        output: {
          mean: statistics.mean.view,
          min: statistics.min.view,
          max: statistics.max.view,
          range: statistics.range.view,
          standardDeviation: statistics.standardDeviation.view
        }
      })
    );
    return {
      outputs: {input, ...statistics},
      update: values =>
        parameterBuffer.write(
          getGPUFocalStatisticsParameterValues({
            radius: values.radius,
            shape: values.shape === 0 ? 'square' : 'circle'
          })
        )
    };
  },
  draw: ({inputs, outputs, values}) => {
    const statistic = [
      outputs.mean,
      outputs.min,
      outputs.max,
      outputs.range,
      outputs.standardDeviation
    ][values.statistic];
    return [
      {
        layer: 'raster',
        context: true,
        values: outputs.input,
        columns: inputs.grid.columns,
        rows: inputs.grid.rows
      },
      {layer: 'raster', values: statistic, columns: inputs.grid.columns, rows: inputs.grid.rows}
    ];
  }
});
