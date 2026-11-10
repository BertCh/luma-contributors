// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterSeasonalBreak fits a seasonal curve (one harmonic) with a trend to every cell's
// 36-month series and looks for one level shift. Where a region of the map drops in the second
// half the break is found; the picture is the BIC improvement of the best break (zero means no
// break). Minimum gain is a construction prop; shift and noise rewrite the data. The pointer probes
// a cell: an inset draws its 36-month series with the detected break marked.

import {GPURasterSeasonalBreak} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './seasonal-break-detection.parameters';
import {getCellAt, getCellBox} from './support/gestures';
import {sampleNdviSeries} from '../../../data/ndvi';

const SLICE_COUNT = 36;
const INSET = [0.04, 0.05, 0.42, 0.3] as const;
const SERIES_RANGE = [-0.4, 1.3] as const;
const BREAK_MONTH = 20;

export default defineOperationRun({
  parameters,
  rebuildOn: ['minimumBICGain'],
  caption: 'A seasonal series per cell in; the strength of its level shift out.',
  createInputs: ({random, aspect}) => {
    const rows = 24;
    const columns = Math.round(rows * aspect);
    const observed = sampleNdviSeries(columns, rows, [7, 9]);
    const cellCount = columns * rows;
    const noise = Float32Array.from({length: cellCount * SLICE_COUNT}, () => random() - 0.5);
    return {columns, rows, cellCount, sliceCount: SLICE_COUNT, observed, noise};
  },
  build(builder, inputs, values) {
    const {cellCount, sliceCount} = inputs;
    const series = builder.allocate('series', 'float32', cellCount * sliceCount);
    const output = {
      coefficients: builder.allocate('coefficients', 'float32', cellCount * 5),
      components: builder.allocate('components', 'float32', cellCount * sliceCount * 3),
      breakIndex: builder.allocate('break-index', 'uint32', cellCount),
      improvement: builder.allocate('improvement', 'float32', cellCount),
      status: builder.allocate('status', 'uint32', cellCount)
    };
    builder.graph.add(
      new GPURasterSeasonalBreak({
        cellCount,
        sliceCount,
        values: series.view,
        period: 12,
        minimumSegmentLength: 8,
        minimumBICGain: values.minimumBICGain,
        output: {
          coefficients: output.coefficients.view,
          components: output.components.view,
          breakIndex: output.breakIndex.view,
          improvement: output.improvement.view,
          status: output.status.view
        }
      })
    );
    return {
      outputs: {
        improvement: output.improvement,
        breakIndex: output.breakIndex,
        inset: builder.allocate('inset', 'float32', sliceCount)
      },
      update: current => {
        const data = new Float32Array(cellCount * sliceCount);
        for (let cell = 0; cell < cellCount; cell++) {
          // Model a monthly season around two measured pre/post-fire NDVI levels.
          const before = inputs.observed[cell * 2];
          const after = inputs.observed[cell * 2 + 1];
          const change = Number.isFinite(after) ? after - before : 0;
          for (let slice = 0; slice < sliceCount; slice++) {
            data[cell * sliceCount + slice] =
              before +
              0.2 * Math.sin((slice / 12) * Math.PI * 2) +
              (slice >= BREAK_MONTH ? change * current.shift : 0) +
              current.noise * inputs.noise[cell * sliceCount + slice];
          }
        }
        series.write(data);
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a cell series',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, values, world, peek}) => {
      const {column, row, index} = getCellAt(inputs, world, pointer);
      const before = inputs.observed[index * 2];
      const after = inputs.observed[index * 2 + 1];
      const change = Number.isFinite(after) ? after - before : 0;
      outputs.inset.write(
        Float32Array.from(
          {length: SLICE_COUNT},
          (_, slice) =>
            before +
            0.2 * Math.sin((slice / 12) * Math.PI * 2) +
            (slice >= BREAK_MONTH ? change * values.shift : 0) +
            values.noise * inputs.noise[index * SLICE_COUNT + slice]
        )
      );
      const breakIndex = peek(outputs.breakIndex);
      const improvement = peek(outputs.improvement);
      const gain = improvement ? improvement[index] : 0;
      const found = breakIndex && gain > 0 ? breakIndex[index] : -1;
      const chartX = (month: number) =>
        world.width * (INSET[0] + (month / (SLICE_COUNT - 1)) * (INSET[2] - INSET[0]));
      return {
        glyphs: [
          getCellBox(inputs, world, column, row),
          {
            shape: 'box',
            minimum: [chartX(0), world.height * INSET[1]],
            maximum: [chartX(SLICE_COUNT - 1), world.height * INSET[3]]
          },
          ...(found >= 0
            ? [
                {
                  shape: 'line' as const,
                  from: [chartX(found), world.height * INSET[1]] as [number, number],
                  to: [chartX(found), world.height * INSET[3]] as [number, number]
                }
              ]
            : [])
        ],
        readout:
          found >= 0 ? `break at month ${found} | BIC gain ${gain.toFixed(0)}` : 'no break found'
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.improvement,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: [0, 200],
      smoothing: false
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.inset,
      range: SERIES_RANGE,
      rectangle: INSET,
      color: [1, 1, 1, 0.95]
    }
  ]
});
