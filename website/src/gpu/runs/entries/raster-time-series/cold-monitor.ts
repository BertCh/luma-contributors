// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterCOLDMonitor segments every pixel's series into stable seasonal-trend models: it fits a
// model to an initial window, tests later observations against it, and confirms a change after
// `consecutiveCount` observations in a row whose normalised residual exceeds `changeThreshold`
// (a chi-squared quantile); the pixel then starts a new segment. The map classes each pixel by its segments (grey: one, stable; orange: one break; red: more). Point at a pixel: observations (white), the model of each segment (blue,
// then orange) and the break (line). A synthetic six-year series with 96 irregular dates whose
// summer level and post-fire ratio are measured Dixie NDVI; the break is set at one date for every
// pixel, and only pixels the fire really darkened can show it. One band.

import {
  getGPURasterCOLDMonitorParameterValues,
  GPU_RASTER_COLD_MONITOR_NO_BREAK,
  GPURasterCOLDMonitor
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {formatDate, uploadDateAxis} from './_lib/axis';
import {
  allocateDots,
  getAccent,
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
import {formatCell, getCellOutline, getProbeCell} from './_lib/probe';
import {createBlankMap, readUint32} from './_lib/readback';
import {createSeasonalStack} from './_lib/stacks';
import {parameters} from './cold-monitor.parameters';

const SLICE_COUNT = 96;
const BREAK_SLICE = 62;
const SEGMENT_COUNT = 4;
const COEFFICIENT_COUNT = 6;
const COEFFICIENT_STRIDE = 8;
const CURVE_POINTS_PER_SEGMENT = 120;
const [FRAME] = getChartRectangles(1);
const RANGE = [-0.2, 1] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['consecutiveCount'],
  buildCache: 4,
  caption:
    'Synthetic series on measured Dixie NDVI levels and fire ratio, 96 irregular dates; one band.',
  createInputs: ({aspect, random}) => {
    const stack = createSeasonalStack(aspect, random, {
      sliceCount: SLICE_COUNT,
      yearCount: 6,
      breakSlice: BREAK_SLICE
    });
    return {...stack, observed: stack.observe(0.03, 0.05)};
  },
  build(builder, stack, values) {
    const {grid, sliceCount} = stack;
    const series = builder.upload('series', 'float32', stack.observed);
    const parameterBuffer = builder.allocate('parameters', 'float32', 7);
    const output = {
      segments: builder.allocate('segments', 'uint32', grid.cellCount * SEGMENT_COUNT * 4),
      coefficients: builder.allocate(
        'coefficients',
        'float32',
        grid.cellCount * SEGMENT_COUNT * COEFFICIENT_STRIDE
      ),
      segmentStatistics: builder.allocate(
        'segment-statistics',
        'float32',
        grid.cellCount * SEGMENT_COUNT * 2
      ),
      cellSummary: builder.allocate('cell-summary', 'uint32', grid.cellCount * 3)
    };
    builder.graph.add(
      new GPURasterCOLDMonitor({
        cellCount: grid.cellCount,
        sliceCount,
        bandCount: 1,
        values: series.view,
        parameters: parameterBuffer.view,
        dateAxis: uploadDateAxis(builder, stack.times),
        consecutiveCount: Math.round(values.consecutiveCount),
        maximumCoefficientCount: COEFFICIENT_COUNT,
        maximumSegmentCount: SEGMENT_COUNT,
        output: {
          segments: output.segments.view,
          coefficients: output.coefficients.view,
          segmentStatistics: output.segmentStatistics.view,
          cellSummary: output.cellSummary.view
        }
      })
    );
    return {
      outputs: {
        ...output,
        blank: createBlankMap(builder, 'blank', grid.cellCount),
        level: builder.upload('level', 'float32', stack.level),
        observations: allocateDots(builder, 'observations', sliceCount),
        firstModel: allocateDots(builder, 'first-model', CURVE_POINTS_PER_SEGMENT * 2),
        laterModel: allocateDots(builder, 'later-model', CURVE_POINTS_PER_SEGMENT * 2)
      },
      update: current => {
        stack.observed = stack.observe(current.cloudFraction, 0.05);
        series.write(stack.observed);
        parameterBuffer.write(
          getGPURasterCOLDMonitorParameterValues({
            changeThreshold: current.changeThreshold,
            outlierThreshold: 24,
            minimumRMSE: [0.03]
          })
        );
      }
    };
  },
  async afterRun({outputs, inputs, upload}) {
    const summary = await readUint32(outputs.cellSummary);
    // Segments per pixel as three classes: one (stable, muted), two (one break), more. Status bits
    // 1 (no segment initialized) and 2 (rank deficient) leave no usable segments; bits 8 (fit
    // limit) and 16 (segment overflow) still publish the segments found, so those cells are drawn.
    const map = Float32Array.from({length: inputs.grid.cellCount}, (_, cell) => {
      if (!(inputs.level[cell] > 0) || (summary[cell * 3 + 1] & 3) !== 0) return Number.NaN;
      const count = summary[cell * 3];
      return count <= 1 ? Number.NaN : count === 2 ? 1 : 2;
    });
    return {map: upload('segment-class-map', 'float32', map)};
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
      const years = Array.from(stack.years);
      writeDots(outputs.observations, years, series);
      const segments = peek(outputs.segments);
      const coefficients = peek(outputs.coefficients);
      const summary = peek(outputs.cellSummary);
      const xRange = [stack.years[0], stack.years[sliceCount - 1]] as const;
      const curves: [number[], number[]][] = [
        [[], []],
        [[], []]
      ];
      const glyphs = [
        getCellOutline(grid, world, cell),
        getFrameGlyph(world, FRAME),
        ...getYearTicks(world, FRAME, xRange, getMidpointSeconds(stack.times))
      ];
      let breakSlice = GPU_RASTER_COLD_MONITOR_NO_BREAK;
      const found = summary ? summary[cell.index * 3] : 0;
      if (segments && coefficients) {
        for (let segment = 0; segment < Math.min(found, SEGMENT_COUNT); segment++) {
          const word = (cell.index * SEGMENT_COUNT + segment) * 4;
          const start = segments[word];
          const end = segments[word + 1];
          const segmentBreak = segments[word + 2];
          if (segment === 0) breakSlice = segmentBreak;
          const offset = (cell.index * SEGMENT_COUNT + segment) * COEFFICIENT_STRIDE;
          const target = curves[segment % 2];
          for (let point = 0; point < CURVE_POINTS_PER_SEGMENT; point++) {
            const year =
              stack.years[start] +
              ((stack.years[Math.min(end, sliceCount - 1)] - stack.years[start]) * point) /
                (CURVE_POINTS_PER_SEGMENT - 1);
            let value = coefficients[offset] + coefficients[offset + 1] * year;
            for (let harmonic = 1; harmonic <= (COEFFICIENT_COUNT - 2) / 2; harmonic++) {
              const angle = 2 * Math.PI * harmonic * year;
              value +=
                coefficients[offset + 2 * harmonic] * Math.cos(angle) +
                coefficients[offset + 2 * harmonic + 1] * Math.sin(angle);
            }
            target[0].push(year);
            target[1].push(value);
          }
        }
      }
      writeDots(outputs.firstModel, curves[0][0], curves[0][1]);
      writeDots(outputs.laterModel, curves[1][0], curves[1][1]);
      const hasBreak = breakSlice !== GPU_RASTER_COLD_MONITOR_NO_BREAK && breakSlice < sliceCount;
      if (hasBreak) glyphs.push(getVerticalRule(world, FRAME, xRange, stack.years[breakSlice]));
      return {
        spotlight: null,
        glyphs,
        readout: !summary
          ? 'reading the segments'
          : hasBreak
            ? `break ${formatDate(stack.times[breakSlice])} · changeThreshold ${formatNumber(values.changeThreshold, 1)} · ${formatCell(grid, cell)}`
            : `no break · ${found} segment${found === 1 ? '' : 's'} · ${formatCell(grid, cell)}`
      };
    }
  },
  draw: ({inputs: stack, outputs, extra, theme}) => {
    const xRange = [stack.years[0], stack.years[stack.sliceCount - 1]] as const;
    return [
      getMapLayer(outputs.level, stack.grid, stack.grid.mapLeft, {
        context: true,
        ramp: 'sequential',
        range: [0, 1]
      }),
      getMapLayer(extra?.map ?? outputs.blank, stack.grid, stack.grid.mapLeft, {
        ramp: 'categorical',
        range: [0, 7]
      }),
      getDotsLayer(outputs.observations, FRAME, RANGE, xRange, getInk(theme), 3.5),
      getDotsLayer(outputs.firstModel, FRAME, RANGE, xRange, getAccent(theme, 0), 1.8),
      getDotsLayer(outputs.laterModel, FRAME, RANGE, xRange, getAccent(theme, 1), 1.8)
    ];
  }
});
