// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTemporalReduction of sensor readings into (cell, time bucket) slots. The grid is the dense
// per-slot count the GPU wrote: columns are time buckets, rows are the twelve map cells.

import {
  GPUTemporalReduction,
  getGPUTemporalReductionParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {getRasterCell} from '../../pick';
import {addIntegerToFloat} from './_lib/convert';
import {sampleFlashes} from '../../../data/lightning';
import {parameters} from './temporal-reduction.parameters';

const ROW_COUNT = 4000;
const CELL_COUNT = 12;
const BUCKET_COUNT = 18;

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {slotCounts: RunBuffer; ticks: RunBuffer};

function createInputs({world}: {world: {width: number; height: number}}) {
  const flashes = sampleFlashes({width: 1, height: 1}, ROW_COUNT);
  const cells = Uint32Array.from(flashes.hours, (_, row) =>
    Math.min(CELL_COUNT - 1, Math.floor(flashes.positions[row * 2] * CELL_COUNT))
  );
  // Counts are shown on the scale of the widest buckets, so widening them visibly raises the counts.
  let tallest = 1;
  for (let origin = 0; origin <= 12; origin += 0.5) {
    const counts = new Uint32Array(CELL_COUNT * BUCKET_COUNT);
    flashes.hours.forEach((hour, row) => {
      const bucket = Math.floor((hour - origin) / 2);
      if (bucket >= 0 && bucket < BUCKET_COUNT)
        tallest = Math.max(tallest, ++counts[cells[row] * BUCKET_COUNT + bucket]);
    });
  }
  return {world, cells, hours: flashes.hours, energy: flashes.energy, tallest};
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption: 'Midwest lightning flashes by longitude band and hours since May 21, 2024, 12:00 UTC.',
  createInputs,
  build(builder, inputs) {
    const cellIds = builder.upload('cell-ids', 'uint32', inputs.cells);
    const timestamps = builder.upload('timestamps', 'float32', inputs.hours);
    const values = builder.upload('energy', 'float32', inputs.energy);
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
    const ticks = builder.allocate('ticks', 'float32x4', 4);
    return {
      outputs: {slotCounts, ticks},
      update: current => {
        // Ticks at hours 0, 6, 12 and 18; those outside the bucketed span collapse to nothing.
        ticks.write(
          Float32Array.from({length: 16}, (_, index) => {
            const column = ((index >> 2) * 6 - current.origin) / current.width;
            const isVisible = column >= 0 && column <= BUCKET_COUNT;
            const x = isVisible ? (column / BUCKET_COUNT) * inputs.world.width : 0;
            return [x, 1, x, isVisible ? 0.96 : 1][index % 4];
          })
        );
        parameterBuffer.write(
          getGPUTemporalReductionParameterValues(current.origin, current.width)
        );
      }
    };
  },
  rebuildOn: [],
  // The pointer is a pad on the bucket grid: x slides its origin, y sets the bucket width.
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'pad',
    hint: 'slide origin, width up',
    x: 'origin',
    y: 'width',
    decorate: (values, pointer, {inputs, outputs, peek}) => {
      const cell = getRasterCell(
        BUCKET_COUNT,
        CELL_COUNT,
        [0, 0, inputs.world.width, 1],
        pointer.x,
        pointer.y
      );
      const count = peek(outputs.slotCounts)?.[cell.index];
      const left = (cell.column / BUCKET_COUNT) * inputs.world.width;
      const right = ((cell.column + 1) / BUCKET_COUNT) * inputs.world.width;
      return {
        glyphs: [
          {
            shape: 'box',
            minimum: [left, 1 - (cell.row + 1) / CELL_COUNT],
            maximum: [right, 1 - cell.row / CELL_COUNT]
          }
        ],
        readout: `origin ${values.origin} h · width ${values.width} h · ${count ?? '…'} flashes here`
      };
    }
  }),
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.slotCounts,
      columns: BUCKET_COUNT,
      rows: CELL_COUNT,
      range: [0, inputs.tallest]
    },
    {layer: 'lines', context: true, segments: outputs.ticks, count: 4, color: [1, 1, 1, 0.8]}
  ]
});
