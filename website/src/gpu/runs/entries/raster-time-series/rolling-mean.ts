// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterRollingMean is the trailing mean over a chunked stream, the aggregation SPI and SPEI
// apply before fitting. The stack here arrives as eight chunks of eight slices; the window state
// (`ring`, `seenCount`) is carried from one chunk to the next, so the result does not depend on
// where the chunks fall. An output is missing until the window is full and whenever the window
// holds a missing sample (skipna false). The map is the share of slices that have a rolling
// mean. Point at a pixel: its series (white) and the rolling mean (blue), with holes where a gap
// sits inside the window. Synthetic seasonal series on measured Dixie NDVI levels; the slices are
// regular slots.

import {
  getGPURasterStreamingAccumulatorParameterValues,
  GPURasterRollingMean
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {
  allocateDots,
  getAccent,
  getChartRectangles,
  getDotsLayer,
  getFrameGlyph,
  getInk,
  getMapLayer,
  writeDots
} from './_lib/charts';
import {formatCell, getCellOutline, getProbeCell} from './_lib/probe';
import {createBlankMap, readFloat32} from './_lib/readback';
import {createSeasonalStack} from './_lib/stacks';
import {parameters} from './rolling-mean.parameters';

const CHUNK_SLICE_COUNT = 8;
const CHUNK_COUNT = 8;
const SLICE_COUNT = CHUNK_SLICE_COUNT * CHUNK_COUNT;
const [FRAME] = getChartRectangles(1);
const RANGE = [-0.2, 1] as const;
const X_RANGE = [0, SLICE_COUNT - 1] as const;

/** The chunk `chunk` of a cell-major stack, laid out `cell * CHUNK_SLICE_COUNT + slice`. */
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

export default defineOperationRun({
  parameters,
  rebuildOn: ['windowLength'],
  buildCache: 6,
  caption:
    'Synthetic seasonal series on measured Dixie NDVI levels, streamed as 8 chunks of 8 slices.',
  createInputs: ({aspect, random}) => {
    const stack = createSeasonalStack(aspect, random, {
      sliceCount: SLICE_COUNT,
      yearCount: 4,
      jitter: 0
    });
    return {...stack, observed: stack.observe(0.1, 0.03)};
  },
  build(builder, stack, values) {
    const {grid} = stack;
    const windowLength = Math.round(values.windowLength);
    const ring = builder.allocate('ring', 'float32', grid.cellCount * windowLength);
    const seenCount = builder.allocate('seen-count', 'uint32', grid.cellCount);
    const chunks = Array.from({length: CHUNK_COUNT}, (_, chunk) => {
      const input = builder.allocate(
        `chunk-${chunk}`,
        'float32',
        grid.cellCount * CHUNK_SLICE_COUNT
      );
      const parameterBuffer = builder.upload(
        `parameters-${chunk}`,
        'float32',
        getGPURasterStreamingAccumulatorParameterValues(
          {reset: chunk === 0, maximumValueMagnitude: 1},
          CHUNK_SLICE_COUNT
        )
      );
      const output = builder.allocate(
        `rolling-${chunk}`,
        'float32',
        grid.cellCount * CHUNK_SLICE_COUNT
      );
      builder.graph.add(
        new GPURasterRollingMean({
          id: `rolling-mean-${chunk}`,
          cellCount: grid.cellCount,
          chunkSliceCount: CHUNK_SLICE_COUNT,
          windowLength,
          values: input.view,
          parameters: parameterBuffer.view,
          ring: ring.view,
          seenCount: seenCount.view,
          output: output.view
        })
      );
      return {input, output};
    });
    return {
      outputs: {
        chunks,
        blank: createBlankMap(builder, 'blank', grid.cellCount),
        level: builder.upload('level', 'float32', stack.level),
        observations: allocateDots(builder, 'observations', SLICE_COUNT),
        rolling: allocateDots(builder, 'rolling', SLICE_COUNT)
      },
      update: current => {
        stack.observed = stack.observe(0.1, current.gapFraction);
        chunks.forEach(({input}, chunk) =>
          input.write(getChunk(stack.observed, grid.cellCount, chunk))
        );
      }
    };
  },
  async afterRun({outputs, inputs, upload}) {
    const rolled = await Promise.all(outputs.chunks.map(({output}) => readFloat32(output)));
    const map = Float32Array.from({length: inputs.grid.cellCount}, (_, cell) => {
      if (!(inputs.level[cell] > 0)) return Number.NaN;
      let defined = 0;
      for (const chunk of rolled) {
        for (let slice = 0; slice < CHUNK_SLICE_COUNT; slice++) {
          if (Number.isFinite(chunk[cell * CHUNK_SLICE_COUNT + slice])) defined++;
        }
      }
      return defined / SLICE_COUNT;
    });
    return {map: upload('defined-map', 'float32', map)};
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a pixel',
    rest: [0.72, 0.5],
    resolve: (pointer, {inputs: stack, outputs, values, world, peek}) => {
      const {grid} = stack;
      const cell = getProbeCell(grid, world, pointer);
      const series = stack.observed.subarray(
        cell.index * SLICE_COUNT,
        (cell.index + 1) * SLICE_COUNT
      );
      const rolled = outputs.chunks.map(({output}) => peek(output));
      const ready = rolled.every(chunk => chunk !== null);
      const rolling = Array.from({length: SLICE_COUNT}, (_, slice) => {
        const chunk = rolled[Math.floor(slice / CHUNK_SLICE_COUNT)];
        return chunk
          ? chunk[cell.index * CHUNK_SLICE_COUNT + (slice % CHUNK_SLICE_COUNT)]
          : Number.NaN;
      });
      const slices = Array.from({length: SLICE_COUNT}, (_, slice) => slice);
      writeDots(outputs.observations, slices, series);
      writeDots(outputs.rolling, slices, rolling);
      const defined = rolling.filter(Number.isFinite).length;
      return {
        spotlight: null,
        glyphs: [getCellOutline(grid, world, cell), getFrameGlyph(world, FRAME)],
        readout: ready
          ? `${defined} of ${SLICE_COUNT} slices · windowLength ${formatNumber(values.windowLength, 0)} · ${formatCell(grid, cell)}`
          : 'reading the stream'
      };
    }
  },
  draw: ({inputs: stack, outputs, extra, theme}) => [
    getMapLayer(outputs.level, stack.grid, stack.grid.mapLeft, {
      context: true,
      ramp: 'sequential',
      range: [0, 1]
    }),
    getMapLayer(extra?.map ?? outputs.blank, stack.grid, stack.grid.mapLeft, {
      ramp: 'sequential',
      range: [0, 1]
    }),
    getDotsLayer(outputs.observations, FRAME, RANGE, X_RANGE, getInk(theme), 4),
    getDotsLayer(outputs.rolling, FRAME, RANGE, X_RANGE, getAccent(theme, 0), 4)
  ]
});
