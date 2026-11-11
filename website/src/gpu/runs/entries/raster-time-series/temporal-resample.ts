// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterTemporalResample puts irregular acquisitions on a regular time grid by linear
// interpolation between the bracketing valid observations. It never extrapolates, and a grid point
// inside a gap wider than `maximumGap` gets no value. Here the grid is one point, so the map is the
// measured Dixie stack (16 Sentinel-2 scenes, June to September) resampled to the instant `gridStart`:
// move the pointer right to scrub through 2017 to 2024 (the fire appears as a linear cross-fade
// between 2021-07-13 and 2021-09-11, which is interpolation, not what happened) and up to widen
// `maximumGap`; below about 0.8 years the winters go blank, and outside the scenes nothing is
// extrapolated. The panel plots the regional mean of every scene on its real date.

import {
  getGPURasterTemporalResampleParameterValues,
  GPU_RASTER_SECONDS_PER_YEAR,
  GPURasterTemporalResample
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun, type ParameterValues, type RunBuilder} from '../../define-run';
import {bindAxes, formatNumber} from '../../interaction';
import {formatDate, uploadDateAxis} from './_lib/axis';
import {
  allocateDots,
  getChartRectangles,
  getDotsLayer,
  getFrameGlyph,
  getInk,
  getMapLayer,
  getMidpointSeconds,
  getVerticalRule,
  getYearTicks,
  writeDots
} from './_lib/charts';
import {createDixieStack, getSliceMap} from './_lib/stacks';
import {parameters} from './temporal-resample.parameters';

const [FRAME] = getChartRectangles(1);
const RANGE = [-0.2, 1] as const;

function getSceneMeans(observed: Float32Array, cellCount: number, sliceCount: number) {
  return Array.from({length: sliceCount}, (_, slice) => {
    let total = 0;
    let count = 0;
    for (let cell = 0; cell < cellCount; cell++) {
      const value = observed[cell * sliceCount + slice];
      if (Number.isFinite(value)) {
        total += value;
        count++;
      }
    }
    return count > 0 ? total / count : Number.NaN;
  });
}

function createInputs(aspect: number) {
  const stack = createDixieStack(aspect);
  return {...stack, means: getSceneMeans(stack.observed, stack.grid.cellCount, stack.sliceCount)};
}

type Inputs = ReturnType<typeof createInputs>;

function build(builder: RunBuilder, stack: Inputs) {
  const {grid, sliceCount} = stack;
  const series = builder.upload('series', 'float32', stack.observed);
  const parameterBuffer = builder.allocate('parameters', 'float32', 3);
  const output = {
    values: builder.allocate('resampled', 'float32', grid.cellCount),
    status: builder.allocate('status', 'uint32', grid.cellCount)
  };
  builder.graph.add(
    new GPURasterTemporalResample({
      cellCount: grid.cellCount,
      sliceCount,
      gridCount: 1,
      values: series.view,
      parameters: parameterBuffer.view,
      dateAxis: uploadDateAxis(builder, stack.times),
      output: {values: output.values.view, status: output.status.view}
    })
  );
  const means = allocateDots(builder, 'scene-means', sliceCount);
  writeDots(means, Array.from(stack.years), stack.means);
  return {
    outputs: {
      ...output,
      means,
      preFire: builder.upload('pre-fire', 'float32', getSliceMap(stack, 6))
    },
    update: (current: ParameterValues) =>
      parameterBuffer.write(
        getGPURasterTemporalResampleParameterValues({
          gridStart: current.gridStart,
          gridStep: 0.05,
          maximumGap: current.maximumGap
        })
      )
  };
}

type Outputs = ReturnType<typeof build>['outputs'];

export default defineOperationRun({
  parameters,
  caption:
    'Measured Dixie NDVI, 16 irregular Sentinel-2 scenes, resampled linearly between scenes onto a one-point grid.',
  createInputs: ({aspect}) => createInputs(aspect),
  build: (builder, stack) => build(builder, stack),
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'time',
    hint: 'scrub gridStart',
    x: 'gridStart',
    y: 'maximumGap',
    decorate: (values, _pointer, {inputs: stack, world}) => {
      const start = values.gridStart;
      const xRange = [stack.years[0], stack.years[stack.sliceCount - 1]] as const;
      const after = Array.from(stack.years).findIndex(year => year > start);
      const date = formatDate(
        BigInt(Math.round(getMidpointSeconds(stack.times) + start * GPU_RASTER_SECONDS_PER_YEAR))
      );
      const inside = after > 0;
      const span = inside ? stack.years[after] - stack.years[after - 1] : 0;
      return {
        glyphs: [
          getFrameGlyph(world, FRAME),
          ...getYearTicks(world, FRAME, xRange, getMidpointSeconds(stack.times)),
          getVerticalRule(world, FRAME, xRange, start)
        ],
        readout: !inside
          ? `gridStart ${date} · outside the scenes`
          : span > values.maximumGap
            ? `gridStart ${date} · gap ${formatNumber(span)} years over maximumGap`
            : `gridStart ${date} · between ${formatDate(stack.times[after - 1])} and ${formatDate(stack.times[after])}`
      };
    }
  }),
  draw: ({inputs: stack, outputs, theme}) => {
    const xRange = [stack.years[0], stack.years[stack.sliceCount - 1]] as const;
    return [
      getMapLayer(outputs.preFire, stack.grid, stack.grid.mapLeft, {
        context: true,
        ramp: 'sequential',
        range: [0, 1]
      }),
      getMapLayer(outputs.values, stack.grid, stack.grid.mapLeft, {
        ramp: 'sequential',
        range: [-0.5, 0.9]
      }),
      getDotsLayer(outputs.means, FRAME, RANGE, xRange, getInk(theme), 5)
    ];
  }
});
