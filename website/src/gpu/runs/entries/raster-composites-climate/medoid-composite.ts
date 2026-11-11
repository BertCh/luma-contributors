// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterMedoidComposite picks, per cell, the one observation whose six-band spectrum has the
// smallest total distance to the cell's other observations, so the composite is a real acquisition
// and the slice it came from is its provenance. The scenes are a synthetic stand-in (see
// support/hazy-scenes.ts): the real July 2021 reflectance bands, eight times, with seeded thick
// cloud (flagged by QA) and thin haze (not flagged). `sliceCount` is a construction prop. The map
// shows the near-infrared band of the medoid composite. The pointer probes a cell and an inset
// plots its observations across the six bands with the medoid and the plain mean of the valid ones.

import {GPURasterMedoidComposite} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {
  createPanelData,
  getPanelLayers,
  getPanelRectangle,
  getThemeColor,
  uploadPanel
} from './support/panel';
import {parameters} from './medoid-composite.parameters';
import {getCellAt, getCellBox, getCellCenter} from '../raster-bands-indices/support/gestures';
import {
  BAND_COUNT,
  SCENE_COUNT,
  composeHazyScenes,
  createHazyScenes,
  getSpectraPoints,
  findShowcaseCell,
  getValidMean,
  type ComposedScenes
} from './support/hazy-scenes';

const NO_SLICE = 0xffffffff;
const INSET = [0.03, 0.06, 0.4, 0.44] as const;
const REFLECTANCE_RANGE = [0, 0.55] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['sliceCount'],
  caption: 'Eight synthetic scenes from one July image in; one real observation per cell out.',
  createInputs: ({random, aspect, world}) => ({
    panel: createPanelData(world, [getPanelRectangle(INSET)]),
    scenes: createHazyScenes(random, aspect, 20),
    composed: null as ComposedScenes | null
  }),
  build(builder, inputs, values) {
    const {cellCount} = inputs.scenes;
    const sliceCount = Math.round(values.sliceCount);
    const observations = builder.allocate('values', 'float32', cellCount * sliceCount * BAND_COUNT);
    const validity = builder.allocate('validity', 'uint32', cellCount * sliceCount);
    const composite = builder.allocate('composite', 'float32', cellCount * BAND_COUNT);
    const sliceIndex = builder.allocate('slice-index', 'uint32', cellCount);
    const status = builder.allocate('status', 'uint32', cellCount);
    builder.graph.add(
      new GPURasterMedoidComposite({
        cellCount,
        sliceCount,
        bandCount: BAND_COUNT,
        values: observations.view,
        mask: validity.view,
        output: {composite: composite.view, sliceIndex: sliceIndex.view, status: status.view}
      })
    );
    const pointCount = sliceCount * BAND_COUNT;
    const bandNumbers = Float32Array.from({length: pointCount}, (_, index) => index % BAND_COUNT);
    return {
      outputs: {
        panel: uploadPanel(builder, inputs.panel),
        composite,
        sliceIndex,
        bandX: builder.upload('band-x', 'float32', bandNumbers),
        validY: builder.allocate('valid-y', 'float32', pointCount),
        flaggedY: builder.allocate('flagged-y', 'float32', pointCount),
        meanLine: builder.allocate('mean-line', 'float32', BAND_COUNT),
        medoidLine: builder.allocate('medoid-line', 'float32', BAND_COUNT)
      },
      update: current => {
        const composed = composeHazyScenes(inputs.scenes, sliceCount, current.haze);
        inputs.composed = composed;
        observations.write(composed.values);
        validity.write(composed.validity);
      }
    };
  },
  // The composite's near-infrared band, for the map.
  async afterRun({inputs, outputs, upload}) {
    const bytes = await outputs.composite.buffer.readAsync();
    const composite = new Float32Array(
      bytes.buffer,
      bytes.byteOffset,
      inputs.scenes.cellCount * BAND_COUNT
    );
    const nearInfrared = Float32Array.from(
      {length: inputs.scenes.cellCount},
      (_, cell) => composite[cell * BAND_COUNT + 3]
    );
    return {nearInfrared: upload('near-infrared', 'float32', nearInfrared)};
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a cell',
    rest: ({inputs, world, toUnit}) => {
      const {column, row} = findShowcaseCell(inputs.scenes, SCENE_COUNT);
      return toUnit(getCellCenter(inputs.scenes, world, column, row));
    },
    resolve: (pointer, {inputs, outputs, values, world, peek}) => {
      const {scenes, composed} = inputs;
      const grid = {columns: scenes.columns, rows: scenes.rows};
      const {column, row, index} = getCellAt(grid, world, pointer);
      if (!composed) return {};
      const sliceCount = Math.round(values.sliceCount);
      const spectra = getSpectraPoints(composed, index, sliceCount);
      outputs.validY.write(spectra.accepted);
      outputs.flaggedY.write(spectra.all);
      outputs.meanLine.write(getValidMean(composed, index, sliceCount));
      const medoid = peek(outputs.composite);
      const chosen = peek(outputs.sliceIndex);
      if (medoid) {
        outputs.medoidLine.write(
          medoid.slice(index * BAND_COUNT, (index + 1) * BAND_COUNT) as Float32Array
        );
      }
      let validCount = 0;
      for (let slice = 0; slice < sliceCount; slice++)
        validCount += composed.validity[index * sliceCount + slice];
      const scene = chosen ? chosen[index] : NO_SLICE;
      return {
        spotlight: null,
        glyphs: [
          getCellBox(grid, world, column, row),
          {
            shape: 'box',
            minimum: [world.width * INSET[0], world.height * INSET[1]],
            maximum: [world.width * INSET[2], world.height * INSET[3]]
          }
        ],
        readout:
          scene === NO_SLICE
            ? `no valid observation (${formatNumber(validCount, 0)} of ${sliceCount})`
            : `medoid: scene ${scene} | ${formatNumber(validCount, 0)} of ${sliceCount} valid`
      };
    }
  },
  draw: ({inputs, outputs, extra, theme}) => [
    {
      layer: 'raster',
      values: extra.nearInfrared,
      columns: inputs.scenes.columns,
      rows: inputs.scenes.rows,
      ramp: 'sequential',
      range: [0, 0.3]
    },
    ...getPanelLayers(outputs.panel, theme),
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.flaggedY,
      xValues: outputs.bandX,
      xRange: [-0.5, BAND_COUNT - 0.5],
      range: REFLECTANCE_RANGE,
      rectangle: INSET,
      radius: 3.5,
      color: [1, 0.72, 0.2, 0.95]
    },
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.validY,
      xValues: outputs.bandX,
      xRange: [-0.5, BAND_COUNT - 0.5],
      range: REFLECTANCE_RANGE,
      rectangle: INSET,
      radius: 3.5,
      color: getThemeColor(theme.ink, 0.8)
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.meanLine,
      range: REFLECTANCE_RANGE,
      rectangle: INSET,
      color: getThemeColor(theme.muted, 1)
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.medoidLine,
      range: REFLECTANCE_RANGE,
      rectangle: INSET,
      color: getThemeColor(theme.categories[0], 1)
    }
  ]
});
