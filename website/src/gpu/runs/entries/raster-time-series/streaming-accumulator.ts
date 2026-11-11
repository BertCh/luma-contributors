// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterStreamingAccumulator folds a stack into counts, exact fixed-point sums, squared
// deviations and extrema one chunk at a time, so a stream has no slice limit and the state is only
// per cell. The 16 measured Dixie NDVI scenes arrive as four chunks of four; `reset` is set on the
// first chunk and `activeSliceCount` lets the last chunk be short (`streamedSliceCount` sets how
// many slices have arrived). A sample above `maximumValueMagnitude` is treated as missing. The map is
// the running mean per pixel. The panel follows one burned pixel (the pointer sets `streamedSliceCount` across and `maximumValueMagnitude` up): scenes already streamed are white, the rest dim,
// scenes above the declared magnitude orange; the line is the mean, the thin lines its standard
// deviation, the dashes the extrema, all decoded from the accumulated state.

import {
  DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT,
  getGPURasterFixedPointUnitExponent,
  getGPURasterFixedPointValues,
  getGPURasterStreamingAccumulatorParameterValues,
  GPURasterStreamingAccumulator
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun, type ParameterValues, type RunBuilder} from '../../define-run';
import {bindAxes, formatNumber} from '../../interaction';
import {formatDate} from './_lib/axis';
import {
  allocateDots,
  getAccent,
  getChartRectangles,
  getDotsLayer,
  getFrameGlyph,
  getHorizontalRule,
  getInk,
  getLineLayer,
  getMapLayer,
  getMidpointSeconds,
  getMutedInk,
  getYearTicks,
  writeDots
} from './_lib/charts';
import {getCellOutline, type ProbeCell} from './_lib/probe';
import {createBlankMap, readUint32} from './_lib/readback';
import {createDixieStack, getSliceMap, type DixieStack} from './_lib/stacks';
import {parameters} from './streaming-accumulator.parameters';

const CHUNK_SLICE_COUNT = 4;
const SLICE_COUNT = 16;
const CHUNK_COUNT = SLICE_COUNT / CHUNK_SLICE_COUNT;
const LINE_POINT_COUNT = 40;
const [FRAME] = getChartRectangles(1);
const RANGE = [-0.2, 1] as const;

/** Chunk `chunk` of a cell-major stack as `cell * CHUNK_SLICE_COUNT + slice`. */
function getChunk(series: Float32Array, cellCount: number, chunk: number): Float32Array {
  const output = new Float32Array(cellCount * CHUNK_SLICE_COUNT);
  for (let cell = 0; cell < cellCount; cell++) {
    for (let slice = 0; slice < CHUNK_SLICE_COUNT; slice++) {
      output[cell * CHUNK_SLICE_COUNT + slice] =
        series[cell * SLICE_COUNT + chunk * CHUNK_SLICE_COUNT + slice];
    }
  }
  return output;
}

/** Fixed-point unit exponents of the sums and of the squared deviations. */
function getUnitExponents(maximumValueMagnitude: number) {
  const streamLength = DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT;
  return {
    sums: getGPURasterFixedPointUnitExponent(maximumValueMagnitude, streamLength),
    moments: getGPURasterFixedPointUnitExponent(4 * maximumValueMagnitude ** 2, streamLength)
  };
}

/** The burned pixel (NDVI fell by more than 0.3 across the fire) nearest the middle of the map. */
function getBurnedCell(stack: DixieStack): ProbeCell {
  const {grid} = stack;
  let best: ProbeCell | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let row = 0; row < grid.rows; row++) {
    for (let column = grid.chartColumns; column < grid.columns; column++) {
      const index = row * grid.columns + column;
      const drop =
        stack.observed[index * SLICE_COUNT + 7] - stack.observed[index * SLICE_COUNT + 8];
      const distance =
        (column - (grid.chartColumns + grid.mapColumns / 2)) ** 2 + (row - grid.rows / 2) ** 2;
      if (drop > 0.3 && distance < bestDistance) {
        best = {column, row, index};
        bestDistance = distance;
      }
    }
  }
  const column = grid.chartColumns + Math.floor(grid.mapColumns / 2);
  return best ?? {column, row: 10, index: 10 * grid.columns + column};
}

function createInputs(aspect: number) {
  const stack = createDixieStack(aspect);
  return {...stack, burned: getBurnedCell(stack)};
}

type Inputs = ReturnType<typeof createInputs>;

function build(builder: RunBuilder, stack: Inputs) {
  const {grid} = stack;
  const state = {
    counts: builder.allocate('counts', 'uint32', grid.cellCount),
    sums: builder.allocate('sums', 'uint32', grid.cellCount * 3),
    moments: builder.allocate('moments', 'uint32', grid.cellCount * 3),
    extrema: builder.allocate('extrema', 'float32', grid.cellCount * 2)
  };
  const chunks = Array.from({length: CHUNK_COUNT}, (_, chunk) => {
    const input = builder.upload(
      `chunk-${chunk}`,
      'float32',
      getChunk(stack.observed, grid.cellCount, chunk)
    );
    const parameterBuffer = builder.allocate(`parameters-${chunk}`, 'float32', 5);
    builder.graph.add(
      new GPURasterStreamingAccumulator({
        id: `streaming-accumulator-${chunk}`,
        cellCount: grid.cellCount,
        chunkSliceCount: CHUNK_SLICE_COUNT,
        values: input.view,
        parameters: parameterBuffer.view,
        state: {
          counts: state.counts.view,
          sums: state.sums.view,
          moments: state.moments.view,
          extrema: state.extrema.view
        }
      })
    );
    return parameterBuffer;
  });
  return {
    outputs: {
      ...state,
      blank: createBlankMap(builder, 'blank', grid.cellCount),
      preFire: builder.upload('pre-fire', 'float32', getSliceMap(stack, 6)),
      streamed: allocateDots(builder, 'streamed', SLICE_COUNT),
      waiting: allocateDots(builder, 'waiting', SLICE_COUNT),
      dropped: allocateDots(builder, 'dropped', SLICE_COUNT),
      mean: builder.upload('mean-line', 'float32', new Float32Array(LINE_POINT_COUNT)),
      upper: builder.upload('upper-line', 'float32', new Float32Array(LINE_POINT_COUNT)),
      lower: builder.upload('lower-line', 'float32', new Float32Array(LINE_POINT_COUNT))
    },
    update: (current: ParameterValues) =>
      chunks.forEach((parameterBuffer, chunk) =>
        parameterBuffer.write(
          getGPURasterStreamingAccumulatorParameterValues(
            {
              reset: chunk === 0,
              activeSliceCount: Math.min(
                CHUNK_SLICE_COUNT,
                Math.max(0, current.streamedSliceCount - chunk * CHUNK_SLICE_COUNT)
              ),
              maximumValueMagnitude: current.maximumValueMagnitude
            },
            CHUNK_SLICE_COUNT
          )
        )
      )
  };
}

type Outputs = ReturnType<typeof build>['outputs'];

export default defineOperationRun({
  parameters,
  caption: 'Measured Dixie NDVI, 16 irregular Sentinel-2 scenes streamed as four chunks of four.',
  createInputs: ({aspect}) => createInputs(aspect),
  build: (builder, stack) => build(builder, stack),
  async afterRun({outputs, inputs, values, upload}) {
    const [counts, sums] = await Promise.all([
      readUint32(outputs.counts),
      readUint32(outputs.sums)
    ]);
    const totals = getGPURasterFixedPointValues(
      sums,
      getUnitExponents(values.maximumValueMagnitude).sums
    );
    const map = Float32Array.from({length: inputs.grid.cellCount}, (_, cell) =>
      inputs.observed[cell * SLICE_COUNT + 6] > -2 && counts[cell] > 0
        ? totals[cell] / counts[cell]
        : Number.NaN
    );
    return {map: upload('mean-map', 'float32', map)};
  },
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'time',
    hint: 'feed the stream',
    x: 'streamedSliceCount',
    y: 'maximumValueMagnitude',
    decorate: (values, _pointer, {inputs: stack, outputs, world, peek}) => {
      const {grid} = stack;
      const cell = stack.burned;
      const series = stack.observed.subarray(
        cell.index * SLICE_COUNT,
        (cell.index + 1) * SLICE_COUNT
      );
      const years = Array.from(stack.years);
      const magnitude = values.maximumValueMagnitude;
      const arrived = (slice: number) => slice < values.streamedSliceCount;
      const kept = (slice: number) => Math.abs(series[slice]) <= magnitude;
      const pick = (include: (slice: number) => boolean) =>
        Array.from(series, (value, slice) => (include(slice) ? value : Number.NaN));
      writeDots(
        outputs.streamed,
        years,
        pick(slice => arrived(slice) && kept(slice))
      );
      writeDots(
        outputs.waiting,
        years,
        pick(slice => !arrived(slice))
      );
      writeDots(
        outputs.dropped,
        years,
        pick(slice => arrived(slice) && !kept(slice))
      );

      const counts = peek(outputs.counts);
      const sums = peek(outputs.sums);
      const moments = peek(outputs.moments);
      const extrema = peek(outputs.extrema);
      const xRange = [stack.years[0], stack.years[SLICE_COUNT - 1]] as const;
      let readout = 'reading the state';
      const glyphs = [
        getCellOutline(grid, world, cell),
        getFrameGlyph(world, FRAME),
        ...getYearTicks(world, FRAME, xRange, getMidpointSeconds(stack.times))
      ];
      let mean = Number.NaN;
      let deviation = Number.NaN;
      if (counts && sums && moments && extrema) {
        const count = counts[cell.index];
        const exponents = getUnitExponents(magnitude);
        mean =
          count > 0
            ? getGPURasterFixedPointValues(
                sums.subarray(cell.index * 3, cell.index * 3 + 3),
                exponents.sums
              )[0] / count
            : Number.NaN;
        deviation =
          count > 0
            ? Math.sqrt(
                getGPURasterFixedPointValues(
                  moments.subarray(cell.index * 3, cell.index * 3 + 3),
                  exponents.moments
                )[0] / count
              )
            : Number.NaN;
        if (count > 0) {
          glyphs.push(
            getHorizontalRule(world, FRAME, RANGE, extrema[cell.index * 2]),
            getHorizontalRule(world, FRAME, RANGE, extrema[cell.index * 2 + 1])
          );
        }
        readout =
          count > 0
            ? `count ${count} · mean ${formatNumber(mean)} · up to ${formatDate(stack.times[Math.min(values.streamedSliceCount, SLICE_COUNT) - 1])}`
            : 'no sample streamed yet';
      }
      const fill = (value: number) =>
        new Float32Array(LINE_POINT_COUNT).fill(Number.isFinite(value) ? value : -5);
      outputs.mean.write(fill(mean));
      outputs.upper.write(fill(mean + deviation));
      outputs.lower.write(fill(mean - deviation));
      return {glyphs, readout};
    }
  }),
  draw: ({inputs: stack, outputs, extra, theme}) => {
    const xRange = [stack.years[0], stack.years[SLICE_COUNT - 1]] as const;
    return [
      getMapLayer(outputs.preFire, stack.grid, stack.grid.mapLeft, {
        context: true,
        ramp: 'sequential',
        range: [0, 1]
      }),
      getMapLayer(extra?.map ?? outputs.blank, stack.grid, stack.grid.mapLeft, {
        ramp: 'sequential',
        range: [0, 0.9]
      }),
      getLineLayer(outputs.mean, FRAME, RANGE, getAccent(theme, 0), 1.6),
      getLineLayer(outputs.upper, FRAME, RANGE, getAccent(theme, 0), 0.9),
      getLineLayer(outputs.lower, FRAME, RANGE, getAccent(theme, 0), 0.9),
      getDotsLayer(outputs.waiting, FRAME, RANGE, xRange, getMutedInk(theme, 0.6), 4),
      getDotsLayer(outputs.streamed, FRAME, RANGE, xRange, getInk(theme), 4.5),
      getDotsLayer(outputs.dropped, FRAME, RANGE, xRange, getAccent(theme, 1), 4.5)
    ];
  }
});
