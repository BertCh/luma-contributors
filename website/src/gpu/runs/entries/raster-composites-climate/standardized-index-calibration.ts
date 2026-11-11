// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterStandardizedIndexCalibration streams a calibration period (here two 144-month chunks)
// into per-slot statistics and writes the fit of a distribution to every cell and calendar month:
// a gamma for SPI, a log-logistic (`fisk`) for SPEI. The record is synthetic monthly rainfall whose
// wetness follows the Hurricane Helene station rainfall (see support/climate-stack.ts). The map is
// the fitted shape of the chosen month. The pointer probes a cell: an inset draws a histogram of
// that cell's calibration months in the month shown, with the fitted density over it; shortening
// `activeSliceCount` leaves fewer samples and a rougher fit.

import {
  GPURasterStandardizedIndexCalibration,
  getGPURasterStandardizedIndexCalibrationParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {
  createPanelData,
  getPanelLayers,
  getPanelRectangle,
  getThemeColor,
  uploadPanel
} from './support/panel';
import {parameters} from './standardized-index-calibration.parameters';
import {getCellAt, getCellBox} from '../raster-bands-indices/support/gestures';
import {getCellSample, type IndexDistribution} from './support/distributions';
import {
  MAXIMUM_MONTHLY_PRECIPITATION,
  MONTH_COUNT,
  MONTH_NAMES,
  createClimateStack,
  getMonthChunk,
  getMonthSlots
} from './support/climate-stack';

const CHUNK = 144;
const SLOT_COUNT = 12;
const BIN_COUNT = 16;
const INSET = [0.03, 0.06, 0.4, 0.4] as const;
const DISTRIBUTIONS: IndexDistribution[] = ['gamma', 'fisk'];

export default defineOperationRun({
  parameters,
  rebuildOn: ['distribution'],
  buildCache: 2,
  caption:
    'Synthetic 24-year rainfall, Helene-interpolated wetness, in; fits per cell and month out.',
  createInputs: ({random, world}) => ({
    ...createClimateStack(random, world),
    panel: createPanelData(world, [getPanelRectangle(INSET)])
  }),
  build(builder, stack, values) {
    const {cellCount} = stack;
    const entries = cellCount * SLOT_COUNT;
    const counts = builder.allocate('state-counts', 'uint32', entries * 3);
    const moments = builder.allocate('state-moments', 'uint32', entries * 9);
    const fit = builder.allocate('fit', 'float32', entries * 3);
    const parameterBuffers = [0, 1].map(chunk =>
      builder.allocate(`parameters-${chunk}`, 'float32', 7)
    );
    for (const chunk of [0, 1]) {
      builder.graph.add(
        new GPURasterStandardizedIndexCalibration({
          id: `calibration-${chunk}`,
          cellCount,
          chunkSliceCount: CHUNK,
          slotCount: SLOT_COUNT,
          distribution: DISTRIBUTIONS[Math.round(values.distribution)],
          values: builder.upload(
            `values-${chunk}`,
            'float32',
            getMonthChunk(stack, chunk * CHUNK, CHUNK)
          ).view,
          slotIndices: builder.upload(
            `slots-${chunk}`,
            'uint32',
            getMonthSlots(chunk * CHUNK, CHUNK)
          ).view,
          parameters: parameterBuffers[chunk].view,
          state: {counts: counts.view, moments: moments.view},
          fit: fit.view
        })
      );
    }
    return {
      outputs: {
        fit,
        panel: uploadPanel(builder, stack.panel),
        bars: builder.allocate('bars', 'float32', BIN_COUNT),
        curve: builder.allocate('curve', 'float32', BIN_COUNT)
      },
      update: current =>
        [0, 1].forEach(chunk =>
          parameterBuffers[chunk].write(
            getGPURasterStandardizedIndexCalibrationParameterValues(
              {
                reset: chunk === 0,
                activeSliceCount: Math.max(
                  0,
                  Math.min(CHUNK, current.activeSliceCount - chunk * CHUNK)
                ),
                maximumValueMagnitude: MAXIMUM_MONTHLY_PRECIPITATION
              },
              CHUNK
            )
          )
        )
    };
  },
  // The fitted shape of the month shown, for the map.
  async afterRun({inputs, outputs, values, upload}) {
    const bytes = await outputs.fit.buffer.readAsync();
    const fit = new Float32Array(bytes.buffer, bytes.byteOffset, inputs.cellCount * SLOT_COUNT * 3);
    const month = Math.round(values.month);
    const shape = Float32Array.from(
      {length: inputs.cellCount},
      (_, cell) => fit[(cell * SLOT_COUNT + month) * 3]
    );
    return {shape: upload('shape', 'float32', shape)};
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a cell',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, values, world, peek}) => {
      const grid = {columns: inputs.columns, rows: inputs.rows};
      const {column, row, index} = getCellAt(grid, world, pointer);
      const fit = peek(outputs.fit);
      const month = Math.round(values.month);
      if (!fit) return {};
      const shape = fit[(index * SLOT_COUNT + month) * 3];
      const scale = fit[(index * SLOT_COUNT + month) * 3 + 1];
      const sample: number[] = [];
      for (let time = month; time < Math.min(MONTH_COUNT, values.activeSliceCount); time += 12) {
        sample.push(inputs.precipitation[index * MONTH_COUNT + time]);
      }
      const all = Array.from(
        {length: 24},
        (_, year) => inputs.precipitation[index * MONTH_COUNT + year * 12 + month]
      );
      const maximum = 2.6 * (all.reduce((total, value) => total + value, 0) / all.length);
      const result = getCellSample(
        sample,
        DISTRIBUTIONS[Math.round(values.distribution)],
        shape,
        scale,
        maximum,
        BIN_COUNT
      );
      outputs.bars.write(result.bars);
      outputs.curve.write(result.curve);
      return {
        glyphs: [
          getCellBox(grid, world, column, row),
          {
            shape: 'box',
            minimum: [world.width * INSET[0], world.height * INSET[1]],
            maximum: [world.width * INSET[2], world.height * INSET[3]]
          }
        ],
        readout: `${MONTH_NAMES[month]}: shape ${formatNumber(shape, 2)}, scale ${formatNumber(scale, 1)} | ${sample.length} years, 0-${formatNumber(maximum, 0)} mm`
      };
    }
  },
  draw: ({inputs, outputs, extra, theme}) => [
    {
      layer: 'raster',
      values: extra.shape,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: [1, 6]
    },
    ...getPanelLayers(outputs.panel, theme),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.bars,
      range: [0, 1],
      rectangle: INSET,
      color: getThemeColor(theme.ink, 0.7)
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.curve,
      range: [0, 1],
      rectangle: INSET,
      color: getThemeColor(theme.categories[0], 1)
    }
  ]
});
