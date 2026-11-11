// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterHarmonicRegression fits an intercept, a trend per year and `harmonicCount` cosine and
// sine pairs of `period` years to every pixel's series by least squares, on the real acquisition
// dates. The map is the residual standard deviation (`rmse`) of the fit. Point at a pixel: its
// observations (white) and the fitted curve (blue) are drawn, and below them the residuals, which
// show a seasonal wave when `period` is wrong or `harmonicCount` too small.

import {
  getGPURasterHarmonicRegressionParameterValues,
  GPURasterHarmonicRegression
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {uploadDateAxis} from './_lib/axis';
import {
  allocateDots,
  getAccent,
  getChartRectangles,
  getDotsLayer,
  getFrameGlyph,
  getHorizontalRule,
  getInk,
  getMapLayer,
  getMidpointSeconds,
  getYearTicks,
  writeDots
} from './_lib/charts';
import {formatCell, getCellOutline, getProbeCell} from './_lib/probe';
import {createSeasonalStack} from './_lib/stacks';
import {parameters} from './harmonic-regression.parameters';

const SLICE_COUNT = 64;
const GAP_FRACTION = 0.08;
const [SERIES_FRAME, RESIDUAL_FRAME] = getChartRectangles(2);
const SERIES_RANGE = [-0.2, 1] as const;
const RESIDUAL_RANGE = [-0.3, 0.3] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['harmonicCount'],
  buildCache: 3,
  caption:
    'Synthetic seasonal cycle on measured Dixie NDVI levels, 64 irregular dates with cloud dips and gaps.',
  createInputs: ({aspect, random}) => {
    const stack = createSeasonalStack(aspect, random, {sliceCount: SLICE_COUNT, yearCount: 3});
    return {...stack, observed: stack.observe(0.14, GAP_FRACTION)};
  },
  build(builder, stack, values) {
    const {grid, sliceCount} = stack;
    const harmonicCount = Math.round(values.harmonicCount);
    const series = builder.upload('series', 'float32', stack.observed);
    const parameterBuffer = builder.allocate('parameters', 'float32', 1);
    const output = {
      coefficients: builder.allocate(
        'coefficients',
        'float32',
        grid.cellCount * (2 + 2 * harmonicCount)
      ),
      summary: builder.allocate('summary', 'uint32', grid.cellCount * 2),
      fitted: builder.allocate('fitted', 'float32', grid.cellCount * sliceCount),
      rmse: builder.allocate('rmse', 'float32', grid.cellCount)
    };
    builder.graph.add(
      new GPURasterHarmonicRegression({
        cellCount: grid.cellCount,
        sliceCount,
        values: series.view,
        parameters: parameterBuffer.view,
        harmonicCount,
        dateAxis: uploadDateAxis(builder, stack.times),
        output: {
          coefficients: output.coefficients.view,
          summary: output.summary.view,
          fitted: output.fitted.view,
          rmse: output.rmse.view
        }
      })
    );
    return {
      outputs: {
        ...output,
        level: builder.upload('level', 'float32', stack.level),
        observations: allocateDots(builder, 'observations', sliceCount),
        curve: allocateDots(builder, 'curve', sliceCount),
        residuals: allocateDots(builder, 'residuals', sliceCount)
      },
      update: current => {
        stack.observed = stack.observe(current.cloudFraction, GAP_FRACTION);
        series.write(stack.observed);
        parameterBuffer.write(
          getGPURasterHarmonicRegressionParameterValues({period: current.period})
        );
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a pixel',
    rest: [0.72, 0.5],
    resolve: (pointer, {inputs: stack, outputs, values, world, peek}) => {
      const {grid, sliceCount} = stack;
      const cell = getProbeCell(grid, world, pointer);
      const series = stack.observed.subarray(
        cell.index * sliceCount,
        (cell.index + 1) * sliceCount
      );
      const fitted = peek(outputs.fitted);
      const rmse = peek(outputs.rmse);
      const summary = peek(outputs.summary);
      const curve = fitted
        ? fitted.subarray(cell.index * sliceCount, (cell.index + 1) * sliceCount)
        : [];
      const years = Array.from(stack.years);
      writeDots(outputs.observations, years, series);
      writeDots(outputs.curve, years, curve);
      writeDots(
        outputs.residuals,
        years,
        years.map((_, slice) => series[slice] - (curve[slice] ?? Number.NaN))
      );
      const xRange = [stack.years[0], stack.years[sliceCount - 1]] as const;
      const midpoint = getMidpointSeconds(stack.times);
      const fit = summary && summary[cell.index * 2] === 0;
      return {
        spotlight: null,
        glyphs: [
          getCellOutline(grid, world, cell),
          getFrameGlyph(world, SERIES_FRAME),
          getFrameGlyph(world, RESIDUAL_FRAME),
          getHorizontalRule(world, RESIDUAL_FRAME, RESIDUAL_RANGE, 0),
          ...getYearTicks(world, SERIES_FRAME, xRange, midpoint),
          ...getYearTicks(world, RESIDUAL_FRAME, xRange, midpoint)
        ],
        readout: !summary
          ? 'reading the fit'
          : fit
            ? `rmse ${formatNumber(rmse?.[cell.index] ?? Number.NaN, 3)} · ${summary[cell.index * 2 + 1]} valid · period ${formatNumber(values.period)} · ${formatCell(grid, cell)}`
            : 'too few valid observations'
      };
    }
  },
  draw: ({inputs: stack, outputs, theme}) => {
    const xRange = [stack.years[0], stack.years[stack.sliceCount - 1]] as const;
    return [
      getMapLayer(outputs.level, stack.grid, stack.grid.mapLeft, {
        context: true,
        ramp: 'sequential',
        range: [0, 1]
      }),
      getMapLayer(outputs.rmse, stack.grid, stack.grid.mapLeft, {
        ramp: 'sequential',
        range: [0, 0.12]
      }),
      getDotsLayer(outputs.observations, SERIES_FRAME, SERIES_RANGE, xRange, getInk(theme), 4),
      getDotsLayer(outputs.curve, SERIES_FRAME, SERIES_RANGE, xRange, getAccent(theme, 0), 3),
      getDotsLayer(
        outputs.residuals,
        RESIDUAL_FRAME,
        RESIDUAL_RANGE,
        xRange,
        getAccent(theme, 1),
        4
      )
    ];
  }
});
