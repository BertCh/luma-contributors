// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUChangeDetection over twelve dates per cell. Sentinel observations bracket the 2021 Dixie Fire;
// the operation compares before and after (difference, percent change), tests every cell (t
// statistic, Sen slope, Mann-Kendall) and classes it as a significant increase, decrease or no
// change at `alpha`. The split date, alpha and dates are per-frame parameters.

import {
  getGPUChangeDetectionParameterValues,
  GPUChangeDetection
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, getFractionOf, getParameterAt} from '../../interaction';
import {createLegend, mergeText, type RunLabel} from '../../text';
import {parameters} from './change-detection.parameters';
import {getCellAt, getCellBox} from './support/gestures';
import {createStackInput} from './support/stacks';

const STATISTICS = [
  'difference',
  'percentChange',
  'tStatistic',
  'senSlope',
  'mannKendallZ',
  'significance'
] as const;

const SPLIT = findParameter(parameters, 'splitSlice');
const STRIP = [0.05, 0.05, 0.95, 0.27] as const;
const SIGNIFICANCE_COLORS = [
  [0.36, 0.4, 0.5],
  [0.3, 0.6, 1],
  [1, 0.35, 0.3]
] as const;
const CLASS_NAMES = ['no significant change', 'significant increase', 'significant decrease'];

export default defineOperationRun({
  parameters,
  caption: 'A time series per cell in; change statistics and a significance class out.',
  createInputs: ({random, aspect}) => createStackInput(random, aspect, 40),
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
    const series = builder.allocate('series', 'float32', sliceCount);
    return {
      outputs: {...output, series},
      update: current => {
        const data = new Float32Array(cellCount * sliceCount);
        for (let cell = 0; cell < cellCount; cell++) {
          // Magnitude scales observed post-fire NDVI changes. Keep missing observations missing.
          const reference = inputs.observations[cell * sliceCount + 7];
          for (let slice = 0; slice < sliceCount; slice++) {
            const index = cell * sliceCount + slice;
            const measured = inputs.observations[index];
            data[index] =
              slice >= 8 && Number.isFinite(reference)
                ? reference + current.magnitude * (measured - reference)
                : measured;
          }
        }
        slices.write(data);
        parameterBuffer.write(
          getGPUChangeDetectionParameterValues({
            beforeSlice: 7,
            afterSlice: 8,
            alpha: current.alpha,
            splitSlice: current.splitSlice
          })
        );
      }
    };
  },
  // The pointer sets splitSlice along a timeline strip (x) and probes the cell under it: the strip
  // draws that cell's twelve NDVI dates with the split marked.
  interaction: {
    kind: 'time',
    hint: 'move the split date',
    rest: () => [getFractionOf(SPLIT, SPLIT.default), 0.5],
    resolve: (pointer, {inputs, outputs, values, world, peek}) => {
      const splitSlice = getParameterAt(SPLIT, pointer.unitX);
      const {column, row, index} = getCellAt(inputs, world, pointer);
      outputs.series.write(
        Float32Array.from({length: inputs.sliceCount}, (_, slice) => {
          const value = inputs.observations[index * inputs.sliceCount + slice];
          return Number.isFinite(value) ? value : 0;
        })
      );
      const classes = peek(outputs.significance);
      const name = classes ? CLASS_NAMES[classes[index]] : '';
      const x =
        world.width * (STRIP[0] + ((STRIP[2] - STRIP[0]) * splitSlice) / (inputs.sliceCount - 1));
      return {
        values: {splitSlice},
        glyphs: [
          getCellBox(inputs, world, column, row),
          {
            shape: 'line',
            from: [x, world.height * STRIP[1]],
            to: [x, world.height * STRIP[3]]
          }
        ],
        readout: `splitSlice ${splitSlice}${name ? ` | ${name}` : ''}`
      };
    }
  },
  draw: ({inputs, outputs, values, world}) => {
    const statistic = STATISTICS[Math.round(values.statistic)];
    const labels: RunLabel[] = [
      {
        text: 'NDVI of the pointed cell at the twelve dates',
        at: [STRIP[0] * world.width, STRIP[3] + 0.005],
        anchor: 'bottom-left',
        role: 'label'
      }
    ];
    const legend =
      statistic === 'significance'
        ? createLegend({
            corner: 'top-right',
            title: 'significance',
            entries: CLASS_NAMES.map((label, index) => ({
              label,
              swatch: SIGNIFICANCE_COLORS[index]
            }))
          })
        : {
            labels: [
              {
                text: `${statistic} (diverging colours)`,
                at: [world.width - 0.02, 0.97] as const,
                anchor: 'top-right' as const,
                role: 'title' as const
              }
            ]
          };
    return [
      {
        layer: 'raster',
        values: outputs[statistic],
        columns: inputs.columns,
        rows: inputs.rows,
        ramp: statistic === 'significance' ? 'categorical' : 'diverging',
        categories: statistic === 'significance' ? SIGNIFICANCE_COLORS : undefined,
        smoothing: statistic !== 'significance'
      },
      {
        layer: 'chart',
        kind: 'line',
        values: outputs.series,
        range: [0, 1],
        rectangle: STRIP
      },
      {layer: 'text', ...mergeText({labels}, legend)}
    ];
  }
});
