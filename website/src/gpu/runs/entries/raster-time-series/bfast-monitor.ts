// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterBFASTMonitor fits a seasonal model to a fixed stable history of every pixel, then
// watches the later observations: a moving sum of the residuals (the MOSUM process) crossing the
// boundary `criticalValue * sqrt(2 * log+(t / n))` is a break. The map colors each pixel by the
// magnitude of its break, the median residual of the monitoring period (grey: no break). Point at a pixel: its measured Dixie NDVI series is drawn
// with the history fit, and below it the MOSUM process against the boundary. `criticalValue` and
// `bandwidth` are per-frame; the history length is a construction prop.

import {
  getGPURasterBFASTMonitorParameterValues,
  GPU_RASTER_BFAST_MONITOR_NO_BREAK,
  GPURasterBFASTMonitor
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {formatDate, uploadDateAxis} from './_lib/axis';
import {
  getAccent,
  getChartRectangles,
  allocateDots,
  getDotLayer,
  getDotsLayer,
  getMutedInk,
  writeDots,
  getFrameGlyph,
  getInk,
  getMapLayer,
  getMidpointSeconds,
  getVerticalRule,
  getYearTicks,
  writeCompactPoints
} from './_lib/charts';
import {formatCell, getCellOutline, getProbeCell} from './_lib/probe';
import {createBlankMap, readFloat32, readUint32} from './_lib/readback';
import {createDixieStack, getSliceMap} from './_lib/stacks';
import {parameters} from './bfast-monitor.parameters';

const HARMONIC_COUNT = 1;
const COLUMN_COUNT = 1 + 2 * HARMONIC_COUNT;
const [SERIES_FRAME, PROCESS_FRAME] = getChartRectangles(2);
const SERIES_RANGE = [-0.2, 1] as const;
const PROCESS_RANGE = [-12, 12] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['historySliceCount'],
  buildCache: 4,
  caption:
    'Measured Dixie NDVI, 16 irregular Sentinel-2 scenes: break magnitude per pixel (grey: no break) from a history fit and a MOSUM test.',
  createInputs: ({aspect}) => createDixieStack(aspect),
  build(builder, stack, values) {
    const {grid, sliceCount} = stack;
    const history = Math.round(values.historySliceCount);
    const monitoring = sliceCount - history;
    const series = builder.upload('series', 'float32', stack.observed);
    const parameterBuffer = builder.allocate('parameters', 'float32', 3);
    const output = {
      coefficients: builder.allocate('coefficients', 'float32', grid.cellCount * COLUMN_COUNT),
      scores: builder.allocate('scores', 'float32', grid.cellCount * 2),
      cellSummary: builder.allocate('cell-summary', 'uint32', grid.cellCount * 3),
      process: builder.allocate('process', 'float32', grid.cellCount * monitoring)
    };
    builder.graph.add(
      new GPURasterBFASTMonitor({
        cellCount: grid.cellCount,
        sliceCount,
        values: series.view,
        parameters: parameterBuffer.view,
        dateAxis: uploadDateAxis(builder, stack.times),
        historySliceCount: history,
        harmonicCount: HARMONIC_COUNT,
        includeTrend: false,
        output: {
          coefficients: output.coefficients.view,
          scores: output.scores.view,
          cellSummary: output.cellSummary.view,
          process: output.process.view
        }
      })
    );
    return {
      outputs: {
        ...output,
        series,
        blank: createBlankMap(builder, 'blank', grid.cellCount),
        preFire: builder.upload('pre-fire', 'float32', getSliceMap(stack, 6)),
        historyDates: builder.allocate('history-dates', 'float32', sliceCount),
        historyValues: builder.allocate('history-values', 'float32', sliceCount),
        monitoringDates: builder.allocate('monitoring-dates', 'float32', sliceCount),
        monitoringValues: builder.allocate('monitoring-values', 'float32', sliceCount),
        curveDates: builder.allocate('curve-dates', 'float32', sliceCount),
        curveValues: builder.allocate('curve-values', 'float32', sliceCount),
        processDates: builder.allocate('process-dates', 'float32', sliceCount),
        processValues: builder.allocate('process-values', 'float32', sliceCount),
        upperBoundary: allocateDots(builder, 'upper-boundary', sliceCount),
        lowerBoundary: allocateDots(builder, 'lower-boundary', sliceCount)
      },
      update: current =>
        parameterBuffer.write(
          getGPURasterBFASTMonitorParameterValues({
            criticalValue: current.criticalValue,
            bandwidth: current.bandwidth
          })
        )
    };
  },
  async afterRun({outputs, inputs, upload}) {
    const [summary, scores] = await Promise.all([
      readUint32(outputs.cellSummary),
      readFloat32(outputs.scores)
    ]);
    const map = new Float32Array(inputs.grid.cellCount);
    for (let cell = 0; cell < map.length; cell++) {
      const breakSlice = summary[cell * 3];
      map[cell] =
        summary[cell * 3 + 1] === 0 && breakSlice !== GPU_RASTER_BFAST_MONITOR_NO_BREAK
          ? scores[cell * 2]
          : Number.NaN;
    }
    // Magnitude: the median residual of the monitoring period, where a break was found.
    return {map: upload('magnitude-map', 'float32', map)};
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a pixel',
    rest: [0.72, 0.5],
    resolve: (pointer, {inputs: stack, outputs, values, world, peek}) => {
      const {grid, sliceCount} = stack;
      const history = Math.round(values.historySliceCount);
      const cell = getProbeCell(grid, world, pointer);
      const summary = peek(outputs.cellSummary);
      const coefficients = peek(outputs.coefficients);
      const scores = peek(outputs.scores);
      const processAll = peek(outputs.process);
      const midpoint = getMidpointSeconds(stack.times);
      const xRange = [stack.years[0], stack.years[sliceCount - 1]] as const;
      const series = stack.observed.subarray(
        cell.index * sliceCount,
        (cell.index + 1) * sliceCount
      );
      const years = Array.from(stack.years);
      writeCompactPoints(
        outputs.historyDates,
        outputs.historyValues,
        years.slice(0, history),
        Array.from(series.subarray(0, history))
      );
      writeCompactPoints(
        outputs.monitoringDates,
        outputs.monitoringValues,
        years.slice(history),
        Array.from(series.subarray(history))
      );
      // The history model's prediction at every observation, from the coefficients the GPU wrote.
      const status = summary ? summary[cell.index * 3 + 1] : 1;
      const curveX = years;
      const curveY = years.map((year, slice) => {
        if (!coefficients || status !== 0 || !Number.isFinite(series[slice])) return Number.NaN;
        const offset = cell.index * COLUMN_COUNT;
        const angle = 2 * Math.PI * year;
        return (
          coefficients[offset] +
          coefficients[offset + 1] * Math.cos(angle) +
          coefficients[offset + 2] * Math.sin(angle)
        );
      });
      writeCompactPoints(outputs.curveDates, outputs.curveValues, curveX, curveY);
      // The MOSUM process of the monitoring observations.
      const monitoring = sliceCount - history;
      const processX: number[] = [];
      const processY: number[] = [];
      for (let slice = 0; slice < monitoring; slice++) {
        processX.push(years[history + slice]);
        processY.push(processAll ? processAll[cell.index * monitoring + slice] : Number.NaN);
      }
      writeCompactPoints(outputs.processDates, outputs.processValues, processX, processY);

      const historyValid = series.slice(0, history).filter(Number.isFinite).length;
      const totalValid = series.filter(Number.isFinite).length;
      // The boundary grows with t / n: t counts the valid observations so far, n those of the history.
      const getBoundary = (validCount: number) =>
        values.criticalValue *
        Math.sqrt(2 * Math.max(1, Math.log(validCount / Math.max(historyValid, 1))));
      const boundary = getBoundary(Math.max(totalValid, 1));
      const boundaryX: number[] = [];
      const boundaryY: number[] = [];
      let validSoFar = historyValid;
      for (let slice = history; slice < sliceCount; slice++) {
        if (!Number.isFinite(series[slice])) continue;
        validSoFar++;
        boundaryX.push(years[slice]);
        boundaryY.push(getBoundary(validSoFar));
      }
      writeDots(outputs.upperBoundary, boundaryX, boundaryY);
      writeDots(
        outputs.lowerBoundary,
        boundaryX,
        boundaryY.map(value => -value)
      );
      const breakSlice = summary ? summary[cell.index * 3] : GPU_RASTER_BFAST_MONITOR_NO_BREAK;
      const hasBreak = status === 0 && breakSlice !== GPU_RASTER_BFAST_MONITOR_NO_BREAK;
      const magnitude = scores ? scores[cell.index * 2] : Number.NaN;
      const historyEnd = (years[history - 1] + years[history]) / 2;
      return {
        spotlight: null,
        glyphs: [
          getCellOutline(grid, world, cell),
          getFrameGlyph(world, SERIES_FRAME),
          getFrameGlyph(world, PROCESS_FRAME),
          ...getYearTicks(world, SERIES_FRAME, xRange, midpoint),
          ...getYearTicks(world, PROCESS_FRAME, xRange, midpoint),
          getVerticalRule(world, SERIES_FRAME, xRange, historyEnd, true),
          getVerticalRule(world, PROCESS_FRAME, xRange, historyEnd, true),
          ...(hasBreak
            ? [
                getVerticalRule(world, SERIES_FRAME, xRange, years[breakSlice]),
                getVerticalRule(world, PROCESS_FRAME, xRange, years[breakSlice])
              ]
            : [])
        ],
        readout: !summary
          ? 'reading the fit'
          : status !== 0
            ? `no fit · ${historyValid} history observations · ${formatCell(grid, cell)}`
            : hasBreak
              ? `break ${formatDate(stack.times[breakSlice])} · magnitude ${formatNumber(magnitude)} · ${formatCell(grid, cell)}`
              : `no break · boundary ${formatNumber(boundary, 1)} · ${formatCell(grid, cell)}`
      };
    }
  },
  draw: ({inputs: stack, outputs, extra, theme}) => {
    const xRange = [stack.years[0], stack.years[stack.sliceCount - 1]] as const;
    return [
      getMapLayer(outputs.preFire, stack.grid, stack.grid.mapLeft, {
        context: true,
        ramp: 'sequential',
        range: [0, 1]
      }),
      getMapLayer(extra?.map ?? outputs.blank, stack.grid, stack.grid.mapLeft, {
        ramp: 'diverging',
        range: [-0.5, 0.5]
      }),
      getDotLayer(
        outputs.historyValues,
        outputs.historyDates,
        SERIES_FRAME,
        SERIES_RANGE,
        xRange,
        getInk(theme),
        4.5
      ),
      getDotLayer(
        outputs.monitoringValues,
        outputs.monitoringDates,
        SERIES_FRAME,
        SERIES_RANGE,
        xRange,
        getAccent(theme, 1),
        4.5
      ),
      getDotLayer(
        outputs.curveValues,
        outputs.curveDates,
        SERIES_FRAME,
        SERIES_RANGE,
        xRange,
        getAccent(theme, 0),
        3
      ),
      getDotsLayer(
        outputs.upperBoundary,
        PROCESS_FRAME,
        PROCESS_RANGE,
        xRange,
        getMutedInk(theme, 0.9),
        3
      ),
      getDotsLayer(
        outputs.lowerBoundary,
        PROCESS_FRAME,
        PROCESS_RANGE,
        xRange,
        getMutedInk(theme, 0.9),
        3
      ),
      getDotLayer(
        outputs.processValues,
        outputs.processDates,
        PROCESS_FRAME,
        PROCESS_RANGE,
        xRange,
        getAccent(theme, 2),
        4.5
      )
    ];
  }
});
