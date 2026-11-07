// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTemporalReduction of sensor readings into (cell, time bucket) slots. The grid is the dense
// per-slot count the GPU wrote: columns are time buckets, rows are the twelve map cells.

import {
  GPUTemporalReduction,
  getGPUTemporalReductionParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {addIntegerToFloat} from './_lib/convert';
import {parameters} from './temporal-reduction.parameters';

const ROW_COUNT = 4000;
const CELL_COUNT = 12;
const BUCKET_COUNT = 24;

export default defineOperationRun({
  parameters,
  caption: 'Readings per map cell (rows) and time bucket (columns).',
  createInputs: ({random}) => ({
    cells: Uint32Array.from({length: ROW_COUNT}, () => Math.floor(random() * CELL_COUNT)),
    uniform: Float32Array.from({length: ROW_COUNT}, () => random()),
    gaussian: Float32Array.from({length: ROW_COUNT}, () => random() + random() + random() - 1.5),
    values: Float32Array.from({length: ROW_COUNT}, () => random())
  }),
  build(builder, inputs) {
    const cellIds = builder.upload('cell-ids', 'uint32', inputs.cells);
    const timestamps = builder.allocate('timestamps', 'float32', ROW_COUNT);
    const values = builder.upload('values', 'float32', inputs.values);
    const parameterBuffer = builder.allocate('parameters', 'float32', 2);
    const slotCount = CELL_COUNT * BUCKET_COUNT;
    const counts = builder.allocate('counts', 'uint32', slotCount);
    builder.graph.add(
      new GPUTemporalReduction({
        cellIds: cellIds.view,
        timestamps: timestamps.view,
        values: values.view,
        cellCount: CELL_COUNT,
        bucketCount: BUCKET_COUNT,
        parameters: parameterBuffer.view,
        output: {
          counts: counts.view,
          minimums: builder.allocate('minimums', 'float32', slotCount).view,
          maximums: builder.allocate('maximums', 'float32', slotCount).view,
          first: builder.allocate('first', 'float32', slotCount).view,
          last: builder.allocate('last', 'float32', slotCount).view,
          occupiedSlots: {
            ids: builder.allocate('occupied', 'uint32', slotCount).view,
            status: {
              count: builder.allocate('occupied-count', 'uint32', 1).view,
              overflow: builder.allocate('occupied-overflow', 'uint32', 1).view
            }
          }
        }
      })
    );
    const slotCounts = addIntegerToFloat(builder, 'slot-counts', counts);
    return {
      outputs: {slotCounts},
      update: current => {
        // Each cell has its own rush hour; off-peak readings spread over the whole day.
        timestamps.write(
          Float32Array.from(inputs.uniform, (uniform, row) => {
            const peak = 4 + (inputs.cells[row] % 4) * 4.5;
            return uniform < current.noise
              ? uniform * (24 / current.noise)
              : Math.min(24, Math.max(0, peak + inputs.gaussian[row] * 3));
          })
        );
        parameterBuffer.write(
          getGPUTemporalReductionParameterValues(current.origin, current.width)
        );
      }
    };
  },
  rebuildOn: [],
  draw: ({outputs}) => [
    {
      layer: 'raster',
      values: outputs.slotCounts,
      columns: BUCKET_COUNT,
      rows: CELL_COUNT,
      range: [0, 80]
    }
  ]
});
