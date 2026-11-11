// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterPhenologyMetrics reads the season off a smoothed curve: a season starts where the curve
// rises past `startFraction` of its amplitude above the base and ends where it falls below
// `endFraction`; a second season needs a peak above `secondSeasonMinimumFraction`, and a pixel
// whose amplitude is under `minimumAmplitude` has none. Here GPURasterWhittakerSmoother feeds it.
// The map is the start of the first season (grey: no season). Point at a pixel: the curve, its start and end levels
// (dashed), and each season's start, peak and end.
// The series is a synthetic one-year cycle on measured Dixie NDVI levels, with two seasons per
// year in the right part of the map and the peak drifting later toward the east.

import {
  getGPURasterPhenologyMetricsParameterValues,
  getGPURasterWhittakerSmootherParameterValues,
  GPU_RASTER_PHENOLOGY_METRIC,
  GPURasterPhenologyMetrics,
  GPURasterWhittakerSmoother
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {formatDate, GPU_RASTER_SECONDS_PER_YEAR, uploadDateAxis} from './_lib/axis';
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
  getVerticalRule,
  writeDots
} from './_lib/charts';
import {formatCell, getCellOutline, getProbeCell} from './_lib/probe';
import {createBlankMap, readFloat32, readUint32} from './_lib/readback';
import {createSeasonalStack} from './_lib/stacks';
import {parameters} from './phenology-metrics.parameters';

const SLICE_COUNT = 52;
const SEASON_COUNT = 2;
const FIELD_COUNT = 10;
const [FRAME] = getChartRectangles(1);
const RANGE = [-0.2, 1] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Synthetic one-year cycle on measured Dixie NDVI levels, two seasons in the east; Whittaker-smoothed first.',
  createInputs: ({aspect, random}) => {
    const stack = createSeasonalStack(aspect, random, {
      sliceCount: SLICE_COUNT,
      yearCount: 1,
      phaseDrift: 0.1,
      doubleSeasonFrom: 0.62
    });
    return {...stack, observed: stack.observe(0.1, 0.05)};
  },
  build(builder, stack) {
    const {grid, sliceCount} = stack;
    const series = builder.upload('series', 'float32', stack.observed);
    const whittakerParameters = builder.allocate('whittaker-parameters', 'float32', 2);
    whittakerParameters.write(
      getGPURasterWhittakerSmootherParameterValues({smoothingParameter: 3})
    );
    const phenologyParameters = builder.allocate('phenology-parameters', 'float32', 4);
    const dateAxis = uploadDateAxis(builder, stack.times);
    const smoothed = builder.allocate('smoothed', 'float32', grid.cellCount * sliceCount);
    const output = {
      metrics: builder.allocate('metrics', 'float32', grid.cellCount * SEASON_COUNT * FIELD_COUNT),
      flags: builder.allocate('flags', 'uint32', grid.cellCount * SEASON_COUNT),
      seasonCount: builder.allocate('season-count', 'uint32', grid.cellCount),
      status: builder.allocate('status', 'uint32', grid.cellCount)
    };
    builder.graph.add(
      new GPURasterWhittakerSmoother({
        cellCount: grid.cellCount,
        sliceCount,
        values: series.view,
        parameters: whittakerParameters.view,
        reweightFunction: 'upper-envelope',
        dateAxis,
        output: {
          smoothed: smoothed.view,
          status: builder.allocate('whittaker-status', 'uint32', grid.cellCount).view
        }
      })
    );
    builder.graph.add(
      new GPURasterPhenologyMetrics({
        cellCount: grid.cellCount,
        sliceCount,
        maximumSeasonCount: SEASON_COUNT,
        smoothed: smoothed.view,
        parameters: phenologyParameters.view,
        dateAxis,
        output: {
          metrics: output.metrics.view,
          flags: output.flags.view,
          seasonCount: output.seasonCount.view,
          status: output.status.view
        }
      })
    );
    return {
      outputs: {
        ...output,
        smoothed,
        blank: createBlankMap(builder, 'blank', grid.cellCount),
        level: builder.upload('level', 'float32', stack.level),
        observations: allocateDots(builder, 'observations', sliceCount),
        curve: allocateDots(builder, 'curve', sliceCount),
        peaks: allocateDots(builder, 'peaks', SEASON_COUNT)
      },
      update: current =>
        phenologyParameters.write(
          getGPURasterPhenologyMetricsParameterValues({
            startFraction: current.startFraction,
            endFraction: current.endFraction,
            minimumAmplitude: current.minimumAmplitude,
            secondSeasonMinimumFraction: Math.max(
              current.secondSeasonMinimumFraction,
              current.startFraction,
              current.endFraction
            )
          })
        )
    };
  },
  async afterRun({outputs, inputs, upload}) {
    const [metrics, seasonCount] = await Promise.all([
      readFloat32(outputs.metrics),
      readUint32(outputs.seasonCount)
    ]);
    // Start of the first season, in centred years: the peak drifts later toward the east.
    const map = Float32Array.from({length: inputs.grid.cellCount}, (_, cell) =>
      seasonCount[cell] > 0
        ? metrics[cell * SEASON_COUNT * FIELD_COUNT + GPU_RASTER_PHENOLOGY_METRIC.startTime]
        : Number.NaN
    );
    return {map: upload('start-map', 'float32', map)};
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
      const smoothed = peek(outputs.smoothed);
      const metrics = peek(outputs.metrics);
      const seasonCount = peek(outputs.seasonCount);
      const curve = smoothed
        ? smoothed.subarray(cell.index * sliceCount, (cell.index + 1) * sliceCount)
        : [];
      const years = Array.from(stack.years);
      writeDots(outputs.observations, years, series);
      writeDots(outputs.curve, years, curve);
      const xRange = [stack.years[0], stack.years[sliceCount - 1]] as const;
      const midpoint = getMidpointSeconds(stack.times);
      const found = seasonCount ? seasonCount[cell.index] : 0;
      const field = (season: number, name: keyof typeof GPU_RASTER_PHENOLOGY_METRIC) =>
        metrics
          ? metrics[
              (cell.index * SEASON_COUNT + season) * FIELD_COUNT + GPU_RASTER_PHENOLOGY_METRIC[name]
            ]
          : Number.NaN;
      const peaksX: number[] = [];
      const peaksY: number[] = [];
      const glyphs = [getCellOutline(grid, world, cell), getFrameGlyph(world, FRAME)];
      for (let season = 0; season < found; season++) {
        peaksX.push(field(season, 'peakTime'));
        peaksY.push(field(season, 'peakValue'));
        glyphs.push(
          getVerticalRule(world, FRAME, xRange, field(season, 'startTime'), true),
          getVerticalRule(world, FRAME, xRange, field(season, 'endTime'), true)
        );
      }
      if (found > 0) {
        const base = field(0, 'baseValue');
        const amplitude = field(0, 'amplitude');
        glyphs.push(
          getHorizontalRule(world, FRAME, RANGE, base + values.startFraction * amplitude),
          getHorizontalRule(world, FRAME, RANGE, base + values.endFraction * amplitude)
        );
      }
      writeDots(outputs.peaks, peaksX, peaksY);
      const date = (year: number) =>
        formatDate(BigInt(Math.round(midpoint + year * GPU_RASTER_SECONDS_PER_YEAR))).slice(5);
      return {
        spotlight: null,
        glyphs,
        readout: !seasonCount
          ? 'reading the curve'
          : found === 0
            ? `no season · amplitude under ${formatNumber(values.minimumAmplitude)} · ${formatCell(grid, cell)}`
            : `${found} season${found > 1 ? 's' : ''} · ${date(field(0, 'startTime'))} to ${date(field(0, 'endTime'))} · ${formatCell(grid, cell)}`
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
        ramp: 'sequential',
        range: [-0.5, 0.3]
      }),
      getDotsLayer(outputs.observations, FRAME, RANGE, xRange, getInk(theme), 4),
      getDotsLayer(outputs.curve, FRAME, RANGE, xRange, getAccent(theme, 0), 3),
      getDotsLayer(outputs.peaks, FRAME, RANGE, xRange, getAccent(theme, 1), 7)
    ];
  }
});
