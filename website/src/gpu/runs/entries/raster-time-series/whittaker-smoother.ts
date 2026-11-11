// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterWhittakerSmoother balances fidelity against roughness: it minimises
// sum w (y - z)^2 + smoothingParameter * |D z|^2 for every pixel, with weight zero on missing
// dates (so gaps are filled) and robust reweighting passes that the 'upper-envelope' function
// aims at cloud-depressed vegetation series. The map is each pixel's smoothed seasonal peak. Point at a pixel: observations (white) and the smoothed curve (blue) over
// irregular dates; raise smoothingParameter to flatten the seasonal peaks.

import {
  getGPURasterWhittakerSmootherParameterValues,
  GPURasterWhittakerSmoother
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
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
  getYearTicks,
  writeDots
} from './_lib/charts';
import {formatCell, getCellOutline, getProbeCell} from './_lib/probe';
import {createBlankMap, readFloat32} from './_lib/readback';
import {createSeasonalStack} from './_lib/stacks';
import {parameters} from './whittaker-smoother.parameters';

const SLICE_COUNT = 64;
const GAP_FRACTION = 0.08;
const SMOOTHING_VALUES = [0.5, 2, 6, 20, 60, 200, 1000] as const;
const getSmoothing = (index: number) =>
  SMOOTHING_VALUES[Math.min(SMOOTHING_VALUES.length - 1, Math.max(0, Math.round(index)))];
const FUNCTIONS = ['bisquare', 'upper-envelope'] as const;
const [FRAME] = getChartRectangles(1);
const RANGE = [-0.2, 1] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['differenceOrder', 'reweightFunction'],
  buildCache: 6,
  caption:
    'Synthetic seasonal cycle on measured Dixie NDVI levels with cloud dips and gaps; 64 irregular dates.',
  createInputs: ({aspect, random}) => {
    const stack = createSeasonalStack(aspect, random, {sliceCount: SLICE_COUNT, yearCount: 3});
    return {...stack, observed: stack.observe(0.18, GAP_FRACTION)};
  },
  build(builder, stack, values) {
    const {grid, sliceCount} = stack;
    const series = builder.upload('series', 'float32', stack.observed);
    const parameterBuffer = builder.allocate('parameters', 'float32', 2);
    const output = {
      smoothed: builder.allocate('smoothed', 'float32', grid.cellCount * sliceCount),
      status: builder.allocate('status', 'uint32', grid.cellCount)
    };
    builder.graph.add(
      new GPURasterWhittakerSmoother({
        cellCount: grid.cellCount,
        sliceCount,
        values: series.view,
        parameters: parameterBuffer.view,
        differenceOrder: Math.round(values.differenceOrder),
        reweightPassCount: 4,
        reweightFunction: FUNCTIONS[Math.round(values.reweightFunction)],
        dateAxis: uploadDateAxis(builder, stack.times),
        output: {smoothed: output.smoothed.view, status: output.status.view}
      })
    );
    return {
      outputs: {
        ...output,
        blank: createBlankMap(builder, 'blank', grid.cellCount),
        level: builder.upload('level', 'float32', stack.level),
        observations: allocateDots(builder, 'observations', sliceCount),
        curve: allocateDots(builder, 'curve', sliceCount)
      },
      update: current => {
        stack.observed = stack.observe(current.cloudFraction, GAP_FRACTION);
        series.write(stack.observed);
        parameterBuffer.write(
          getGPURasterWhittakerSmootherParameterValues({
            smoothingParameter: getSmoothing(current.smoothingParameter)
          })
        );
      }
    };
  },
  async afterRun({outputs, inputs, upload}) {
    const smoothed = await readFloat32(outputs.smoothed);
    const {sliceCount} = inputs;
    // The map shows each pixel's smoothed seasonal peak: it follows the real NDVI structure and
    // drops as `smoothingParameter` flattens the peaks.
    const map = Float32Array.from({length: inputs.grid.cellCount}, (_, cell) => {
      let peak = Number.NaN;
      for (let slice = 0; slice < sliceCount; slice++) {
        const value = smoothed[cell * sliceCount + slice];
        if (Number.isFinite(value)) peak = Number.isFinite(peak) ? Math.max(peak, value) : value;
      }
      return peak;
    });
    return {map: upload('peak-map', 'float32', map)};
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
      const curve = smoothed
        ? smoothed.subarray(cell.index * sliceCount, (cell.index + 1) * sliceCount)
        : [];
      const years = Array.from(stack.years);
      writeDots(outputs.observations, years, series);
      writeDots(outputs.curve, years, curve);
      const xRange = [stack.years[0], stack.years[sliceCount - 1]] as const;
      const valid = series.filter(Number.isFinite).length;
      return {
        spotlight: null,
        glyphs: [
          getCellOutline(grid, world, cell),
          getFrameGlyph(world, FRAME),
          ...getYearTicks(world, FRAME, xRange, getMidpointSeconds(stack.times))
        ],
        readout: !status
          ? 'reading the fit'
          : status[cell.index] === 0
            ? `smoothingParameter ${getSmoothing(values.smoothingParameter)} · ${valid} of ${sliceCount} valid · ${formatCell(grid, cell)}`
            : 'too few valid observations'
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
        range: [0.2, 0.9]
      }),
      getDotsLayer(outputs.observations, FRAME, RANGE, xRange, getInk(theme), 4.5),
      getDotsLayer(outputs.curve, FRAME, RANGE, xRange, getAccent(theme, 0), 3.5)
    ];
  }
});
