// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterStandardizedIndex turns an observation into the number of standard deviations it sits
// from normal: it takes the probability of the value under the distribution fitted for that cell and
// calendar month (written by GPURasterStandardizedIndexCalibration) and maps it to a standard normal
// quantile, SPI for a gamma and SPEI for a log-logistic. The record is synthetic monthly rainfall
// whose wetness follows the Hurricane Helene station rainfall (see support/climate-stack.ts). The
// first twenty-five years calibrate the fit; the last five years (sixty months) are evaluated in one
// chunk. The pointer scrubs along those months: the map is that month's index, and the strip shows
// the area mean of every month with a cursor, so a drought or the Helene September stands out.

import {
  GPURasterStandardizedIndex,
  GPURasterStandardizedIndexCalibration,
  getGPURasterStandardizedIndexCalibrationParameterValues,
  getGPURasterStandardizedIndexParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes, formatNumber} from '../../interaction';
import {parameters} from './standardized-index.parameters';
import {
  MAXIMUM_MONTHLY_PRECIPITATION,
  MONTH_NAMES,
  createClimateStack,
  getMonthChunk,
  getMonthSlots,
  type ClimateStack
} from './support/climate-stack';
import {createPanelData, getPanelLayers, getThemeColor, uploadPanel} from './support/panel';

const CHUNK = 150;
const EVALUATION_START = 300;
const EVALUATION_COUNT = 60;
const SLOT_COUNT = 12;
const DISTRIBUTIONS = ['gamma', 'fisk'] as const;
const PANEL = [0.03, 0.03, 0.97, 0.22] as const;
const STRIP = [0.06, 0.06, 0.94, 0.2] as const;
const STRIP_RANGE = [-3, 3] as const;

type Inputs = ClimateStack & {panel: ReturnType<typeof createPanelData>};

export default defineOperationRun({
  parameters,
  rebuildOn: ['distribution'],
  buildCache: 2,
  caption: 'Synthetic 30-year rainfall, Helene-interpolated wetness, in; standard deviations out.',
  createInputs: ({random, world}): Inputs => ({
    ...createClimateStack(random, world),
    panel: createPanelData(world, [PANEL])
  }),
  build(builder, stack: Inputs, values) {
    const {cellCount} = stack;
    const entries = cellCount * SLOT_COUNT;
    const distribution = DISTRIBUTIONS[Math.round(values.distribution)];
    const counts = builder.allocate('state-counts', 'uint32', entries * 3);
    const moments = builder.allocate('state-moments', 'uint32', entries * 9);
    const fit = builder.allocate('fit', 'float32', entries * 3);
    for (const chunk of [0, 1]) {
      builder.graph.add(
        new GPURasterStandardizedIndexCalibration({
          id: `calibration-${chunk}`,
          cellCount,
          chunkSliceCount: CHUNK,
          slotCount: SLOT_COUNT,
          distribution,
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
          parameters: builder.upload(
            `calibration-parameters-${chunk}`,
            'float32',
            getGPURasterStandardizedIndexCalibrationParameterValues(
              {reset: chunk === 0, maximumValueMagnitude: MAXIMUM_MONTHLY_PRECIPITATION},
              CHUNK
            )
          ).view,
          state: {counts: counts.view, moments: moments.view},
          fit: fit.view
        })
      );
    }
    const index = builder.allocate('index', 'float32', cellCount * EVALUATION_COUNT);
    const status = builder.allocate('status', 'uint32', cellCount * EVALUATION_COUNT);
    builder.graph.add(
      new GPURasterStandardizedIndex({
        cellCount,
        chunkSliceCount: EVALUATION_COUNT,
        slotCount: SLOT_COUNT,
        distribution,
        values: builder.upload(
          'evaluated',
          'float32',
          getMonthChunk(stack, EVALUATION_START, EVALUATION_COUNT)
        ).view,
        slotIndices: builder.upload(
          'evaluated-slots',
          'uint32',
          getMonthSlots(EVALUATION_START, EVALUATION_COUNT)
        ).view,
        fit: fit.view,
        parameters: builder.upload(
          'index-parameters',
          'float32',
          getGPURasterStandardizedIndexParameterValues({}, EVALUATION_COUNT)
        ).view,
        output: {index: index.view, status: status.view}
      })
    );
    return {outputs: {index, panel: uploadPanel(builder, stack.panel)}, update: () => undefined};
  },
  // The month shown for the map and the area mean of every month for the strip.
  async afterRun({inputs, outputs, values, upload}) {
    const bytes = await outputs.index.buffer.readAsync();
    const index = new Float32Array(
      bytes.buffer,
      bytes.byteOffset,
      inputs.cellCount * EVALUATION_COUNT
    );
    const month = Math.round(values.month);
    const shown = Float32Array.from(
      {length: inputs.cellCount},
      (_, cell) => index[cell * EVALUATION_COUNT + month]
    );
    return {
      shown: upload('shown', 'float32', shown),
      strip: upload('strip', 'float32', getAreaMeans(index, inputs.cellCount))
    };
  },
  interaction: bindAxes<Inputs, {index: RunBuffer<'float32'>}>(parameters, {
    kind: 'time',
    hint: 'scrub the month',
    x: 'month',
    decorate: (values, _pointer, {inputs, outputs, peek, world}) => {
      const month = Math.round(values.month);
      const index = peek(outputs.index);
      let readout = '';
      if (index) {
        const means = getAreaMeans(index, inputs.cellCount);
        let low = 0;
        for (let cell = 0; cell < inputs.cellCount; cell++) {
          if (index[cell * EVALUATION_COUNT + month] < -1.5) low++;
        }
        readout = ` · area mean ${formatNumber(means[month], 2)} · ${formatNumber((100 * low) / inputs.cellCount, 0)}% below -1.5`;
      }
      const x =
        world.width * (STRIP[0] + ((month + 0.5) / EVALUATION_COUNT) * (STRIP[2] - STRIP[0]));
      return {
        glyphs: [
          {
            shape: 'line',
            from: [x, world.height * STRIP[1]],
            to: [x, world.height * STRIP[3]],
            dashed: false
          }
        ],
        readout: `${MONTH_NAMES[(EVALUATION_START + month) % 12]} of year ${
          Math.floor((EVALUATION_START + month) / 12) + 1
        }${readout}`
      };
    }
  }),
  draw: ({inputs, outputs, extra, theme}) => [
    {
      layer: 'raster',
      values: extra.shown,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'diverging',
      range: [-3, 3]
    },
    ...getPanelLayers(outputs.panel, theme),
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.strip,
      range: STRIP_RANGE,
      rectangle: STRIP,
      color: getThemeColor(theme.categories[0])
    }
  ]
});

/** Mean index over cells for each evaluated month (nonfinite cells skipped). */
function getAreaMeans(index: ArrayLike<number>, cellCount: number): Float32Array {
  const means = new Float32Array(EVALUATION_COUNT);
  for (let month = 0; month < EVALUATION_COUNT; month++) {
    let total = 0;
    let count = 0;
    for (let cell = 0; cell < cellCount; cell++) {
      const value = index[cell * EVALUATION_COUNT + month];
      if (Number.isFinite(value)) {
        total += value;
        count++;
      }
    }
    means[month] = count ? total / count : 0;
  }
  return means;
}
