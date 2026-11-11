// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterHANTS fits harmonics to a series, drops the worst observation that lies further than
// `fitErrorTolerance` below the curve (`rejectionSide` 'low', for clouds), and refits until none is left, or
// `rejectionPassCount` passes are spent. Cloud
// dips pull a plain fit down; HANTS rejects them. The map counts the observations each pixel had
// rejected. Point at a pixel: kept observations are white, rejected ones orange, and the fitted
// curve (blue) is drawn with its tolerance band.

import {
  getGPURasterHANTSParameterValues,
  GPURasterHANTS
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
  getInk,
  getMapLayer,
  getMidpointSeconds,
  getMutedInk,
  getYearTicks,
  writeDots
} from './_lib/charts';
import {formatCell, getCellOutline, getProbeCell} from './_lib/probe';
import {createBlankMap, readUint32} from './_lib/readback';
import {createSeasonalStack} from './_lib/stacks';
import {parameters} from './hants.parameters';

const SLICE_COUNT = 64;
const GAP_FRACTION = 0.06;
const [FRAME] = getChartRectangles(1);
const RANGE = [-0.2, 1] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['harmonicCount', 'rejectionPassCount'],
  buildCache: 6,
  caption:
    'Synthetic seasonal cycle on measured Dixie NDVI levels with cloud dips and gaps; 64 irregular dates.',
  createInputs: ({aspect, random}) => {
    const stack = createSeasonalStack(aspect, random, {sliceCount: SLICE_COUNT, yearCount: 3});
    return {...stack, observed: stack.observe(0.12, GAP_FRACTION)};
  },
  build(builder, stack, values) {
    const {grid, sliceCount} = stack;
    const harmonicCount = Math.round(values.harmonicCount);
    const series = builder.upload('series', 'float32', stack.observed);
    const parameterBuffer = builder.allocate('parameters', 'float32', 3);
    const output = {
      smoothed: builder.allocate('smoothed', 'float32', grid.cellCount * sliceCount),
      coefficients: builder.allocate(
        'coefficients',
        'float32',
        grid.cellCount * (2 * harmonicCount + 1)
      ),
      status: builder.allocate('status', 'uint32', grid.cellCount),
      diagnostics: builder.allocate('diagnostics', 'uint32', grid.cellCount * 2)
    };
    builder.graph.add(
      new GPURasterHANTS({
        cellCount: grid.cellCount,
        sliceCount,
        values: series.view,
        parameters: parameterBuffer.view,
        harmonicCount,
        rejectionSide: 'low',
        rejectionPassCount: Math.round(values.rejectionPassCount),
        dateAxis: uploadDateAxis(builder, stack.times),
        output: {
          smoothed: output.smoothed.view,
          coefficients: output.coefficients.view,
          status: output.status.view,
          diagnostics: output.diagnostics.view
        }
      })
    );
    return {
      outputs: {
        ...output,
        blank: createBlankMap(builder, 'blank', grid.cellCount),
        level: builder.upload('level', 'float32', stack.level),
        kept: allocateDots(builder, 'kept', sliceCount),
        rejected: allocateDots(builder, 'rejected', sliceCount),
        curve: allocateDots(builder, 'curve', sliceCount),
        lowerBand: allocateDots(builder, 'lower-band', sliceCount),
        upperBand: allocateDots(builder, 'upper-band', sliceCount)
      },
      update: current => {
        stack.observed = stack.observe(current.cloudFraction, GAP_FRACTION);
        series.write(stack.observed);
        parameterBuffer.write(
          getGPURasterHANTSParameterValues({
            period: 1,
            fitErrorTolerance: current.fitErrorTolerance
          })
        );
      }
    };
  },
  async afterRun({outputs, inputs, upload}) {
    const diagnostics = await readUint32(outputs.diagnostics);
    const map = Float32Array.from({length: inputs.grid.cellCount}, (_, cell) =>
      inputs.level[cell] > 0 ? diagnostics[cell * 2] : Number.NaN
    );
    return {map: upload('rejected-map', 'float32', map)};
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
      const status = peek(outputs.status);
      const diagnostics = peek(outputs.diagnostics);
      const curve = smoothed
        ? Array.from(smoothed.subarray(cell.index * sliceCount, (cell.index + 1) * sliceCount))
        : [];
      const years = Array.from(stack.years);
      const tolerance = values.fitErrorTolerance;
      // The GPU reports how many observations it rejected; they are the ones furthest below the
      // final curve, which is the order the reject-and-refit loop removes them in.
      const rejectedCount = diagnostics ? diagnostics[cell.index * 2] : 0;
      const rejectedSlices = new Set(
        years
          .map((_, slice) => slice)
          .filter(slice => Number.isFinite(series[slice] - curve[slice]))
          .sort((first, second) => series[first] - curve[first] - (series[second] - curve[second]))
          .slice(0, rejectedCount)
      );
      const keptSlices = years.map((_, slice) => !rejectedSlices.has(slice));
      writeDots(
        outputs.kept,
        years,
        series.map((value, slice) => (keptSlices[slice] ? value : Number.NaN))
      );
      writeDots(
        outputs.rejected,
        years,
        series.map((value, slice) => (keptSlices[slice] ? Number.NaN : value))
      );
      writeDots(outputs.curve, years, curve);
      writeDots(
        outputs.lowerBand,
        years,
        curve.map(value => value - tolerance)
      );
      writeDots(
        outputs.upperBand,
        years,
        curve.map(value => value + tolerance)
      );
      const xRange = [stack.years[0], stack.years[sliceCount - 1]] as const;
      return {
        spotlight: null,
        glyphs: [
          getCellOutline(grid, world, cell),
          getFrameGlyph(world, FRAME),
          ...getYearTicks(world, FRAME, xRange, getMidpointSeconds(stack.times))
        ],
        readout:
          status && diagnostics
            ? status[cell.index] === 0 || status[cell.index] === 8
              ? `rejected ${diagnostics[cell.index * 2]} · passes ${diagnostics[cell.index * 2 + 1]} · tolerance ${formatNumber(tolerance)} · ${formatCell(grid, cell)}`
              : 'too few valid observations'
            : 'reading the fit'
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
        range: [0, 14]
      }),
      getDotsLayer(outputs.lowerBand, FRAME, RANGE, xRange, getMutedInk(theme, 0.8), 1.8),
      getDotsLayer(outputs.upperBand, FRAME, RANGE, xRange, getMutedInk(theme, 0.8), 1.8),
      getDotsLayer(outputs.kept, FRAME, RANGE, xRange, getInk(theme), 4.5),
      getDotsLayer(outputs.rejected, FRAME, RANGE, xRange, getAccent(theme, 1), 4.5),
      getDotsLayer(outputs.curve, FRAME, RANGE, xRange, getAccent(theme, 0), 3)
    ];
  }
});
