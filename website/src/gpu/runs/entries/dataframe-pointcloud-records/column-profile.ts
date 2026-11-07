// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUColumnProfile of one numeric column. The bright bars are the GPU histogram of the masked rows
// over the per-frame domain; the faint bars are the same bins over all rows.

import {
  GPUColumnProfile,
  getGPUColumnProfileParameterValues,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {addIntegerToFloat} from './_lib/convert';
import {createHistogram, createRecordColumn} from './_lib/helpers';
import {parameters} from './column-profile.parameters';

const ROW_COUNT = 2000;

export default defineOperationRun({
  parameters,
  caption: 'Histogram of the masked rows (bright) over all rows (faint).',
  createInputs: ({random, world}) => {
    const points = createUniformPoints(random, world, ROW_COUNT, 0.15);
    return {points, column: createRecordColumn(points.values)};
  },
  build(builder, inputs, values) {
    const binCount = values.histogramBinCount;
    const column = builder.upload('fare', 'float32', inputs.column);
    const positions = builder.upload('positions', 'float32x2', inputs.points.positions);
    const mask = builder.allocate('mask', 'uint32', ROW_COUNT);
    const domain = builder.allocate('domain', 'float32', 2);
    const histograms = builder.allocate('histograms', 'uint32', binCount);
    const allRows = builder.allocate('all-rows', 'float32', binCount);
    builder.graph.add(
      new GPUColumnProfile({
        columns: [{values: column.view}],
        mask: mask.view,
        parameters: domain.view,
        histogramBinCount: binCount,
        output: {
          statistics: builder.allocate('statistics', 'float32', GPU_COLUMN_PROFILE_STATISTIC_COUNT)
            .view,
          counts: builder.allocate('counts', 'uint32', 2).view,
          histograms: histograms.view
        }
      })
    );
    const bars = addIntegerToFloat(builder, 'bars', histograms);
    return {
      outputs: {column, positions, bars, allRows},
      update: current => {
        const kept = Math.floor((ROW_COUNT * current.mask) / 100);
        mask.write(Uint32Array.from({length: ROW_COUNT}, (_, row) => (row < kept ? 1 : 0)));
        domain.write(getGPUColumnProfileParameterValues([[0, current.domainMaximum]]));
        allRows.write(createHistogram(inputs.column, binCount, 0, current.domainMaximum));
      }
    };
  },
  rebuildOn: ['histogramBinCount'],
  draw: ({inputs, outputs, values}) => {
    const tallest = Math.max(
      ...createHistogram(inputs.column, values.histogramBinCount, 0, values.domainMaximum)
    );
    const range = [0, tallest * 1.05] as const;
    return [
      {
        layer: 'dots',
        context: true,
        positions: outputs.positions,
        values: outputs.column,
        range: [0, 50],
        radius: 3
      },
      {
        layer: 'chart',
        kind: 'bars',
        context: true,
        values: outputs.allRows,
        range,
        rectangle: [0.07, 0.08, 0.95, 0.5]
      },
      {
        layer: 'chart',
        kind: 'bars',
        color: [0.2, 0.55, 0.9, 1],
        values: outputs.bars,
        range,
        rectangle: [0.07, 0.08, 0.95, 0.5]
      }
    ];
  }
});
