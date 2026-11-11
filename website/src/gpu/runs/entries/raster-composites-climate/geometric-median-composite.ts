// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterGeometricMedianComposite finds, per cell, the six-band spectrum with the smallest total
// distance to the cell's valid observations (Weiszfeld's iteration, started at their mean). The
// scenes are a synthetic stand-in (see support/hazy-scenes.ts): the real July 2021 reflectance
// bands, eight times, with seeded cloud (flagged by QA) and thin haze (not flagged). The map is the
// composite's near-infrared band. The pointer probes a cell: an inset plots its observations in the
// red and near-infrared plane with the plain mean (orange) and the geometric median (cyan);
// `iterationCount` walks the median away from the mean towards the dense cluster, and a large
// `distanceEpsilon` flattens the weights and pulls it back to the mean.

import {
  GPURasterGeometricMedianComposite,
  getGPURasterGeometricMedianCompositeParameterValues
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
import {parameters} from './geometric-median-composite.parameters';
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

const ITERATION_COUNTS = [1, 2, 4, 8, 16, 32] as const;
const EPSILONS = [1e-5, 1e-3, 0.01, 0.05, 0.2, 0.5] as const;
const RED = 2;
const NEAR_INFRARED = 3;
const INSET = [0.03, 0.06, 0.4, 0.5] as const;
const RED_RANGE = [0, 0.35] as const;
const NEAR_INFRARED_RANGE = [0.05, 0.35] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['iterationCount'],
  buildCache: 6,
  caption: 'Eight synthetic scenes from one July image in; the geometric median per cell out.',
  createInputs: ({random, aspect, world}) => ({
    panel: createPanelData(world, [getPanelRectangle(INSET)]),
    scenes: createHazyScenes(random, aspect, 20),
    composed: null as ComposedScenes | null
  }),
  build(builder, inputs, values) {
    const {cellCount} = inputs.scenes;
    const observations = builder.allocate(
      'values',
      'float32',
      cellCount * SCENE_COUNT * BAND_COUNT
    );
    const validity = builder.allocate('validity', 'uint32', cellCount * SCENE_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 1);
    const composite = builder.allocate('composite', 'float32', cellCount * BAND_COUNT);
    const status = builder.allocate('status', 'uint32', cellCount);
    const lastStep = builder.allocate('last-step', 'float32', cellCount);
    builder.graph.add(
      new GPURasterGeometricMedianComposite({
        cellCount,
        sliceCount: SCENE_COUNT,
        bandCount: BAND_COUNT,
        iterationCount: ITERATION_COUNTS[Math.round(values.iterationCount)],
        values: observations.view,
        mask: validity.view,
        parameters: parameterBuffer.view,
        output: {composite: composite.view, status: status.view, lastStep: lastStep.view}
      })
    );
    const points = (id: string) => builder.allocate(id, 'float32', SCENE_COUNT);
    return {
      outputs: {
        panel: uploadPanel(builder, inputs.panel),
        composite,
        lastStep,
        allRed: points('all-red'),
        allNearInfrared: points('all-near-infrared'),
        acceptedRed: points('accepted-red'),
        acceptedNearInfrared: points('accepted-near-infrared'),
        meanRed: builder.allocate('mean-red', 'float32', 1),
        meanNearInfrared: builder.allocate('mean-near-infrared', 'float32', 1),
        medianRed: builder.allocate('median-red', 'float32', 1),
        medianNearInfrared: builder.allocate('median-near-infrared', 'float32', 1)
      },
      update: current => {
        const composed = composeHazyScenes(inputs.scenes, SCENE_COUNT, current.haze);
        inputs.composed = composed;
        observations.write(composed.values);
        validity.write(composed.validity);
        parameterBuffer.write(
          getGPURasterGeometricMedianCompositeParameterValues({
            distanceEpsilon: EPSILONS[Math.round(current.distanceEpsilon)]
          })
        );
      }
    };
  },
  // The composite's near-infrared band, for the map.
  async afterRun({inputs, outputs, upload}) {
    const bytes = await outputs.composite.buffer.readAsync();
    const {cellCount} = inputs.scenes;
    const composite = new Float32Array(bytes.buffer, bytes.byteOffset, cellCount * BAND_COUNT);
    const nearInfrared = Float32Array.from(
      {length: cellCount},
      (_, cell) => composite[cell * BAND_COUNT + NEAR_INFRARED]
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
    resolve: (pointer, {inputs, outputs, world, peek}) => {
      const {scenes, composed} = inputs;
      const grid = {columns: scenes.columns, rows: scenes.rows};
      const {column, row, index} = getCellAt(grid, world, pointer);
      if (!composed) return {};
      const spectra = getSpectraPoints(composed, index, SCENE_COUNT);
      const pick = (source: Float32Array, band: number) =>
        Float32Array.from({length: SCENE_COUNT}, (_, slice) => source[slice * BAND_COUNT + band]);
      outputs.allRed.write(pick(spectra.all, RED));
      outputs.allNearInfrared.write(pick(spectra.all, NEAR_INFRARED));
      outputs.acceptedRed.write(pick(spectra.accepted, RED));
      outputs.acceptedNearInfrared.write(pick(spectra.accepted, NEAR_INFRARED));
      const mean = getValidMean(composed, index, SCENE_COUNT);
      outputs.meanRed.write(Float32Array.of(mean[RED]));
      outputs.meanNearInfrared.write(Float32Array.of(mean[NEAR_INFRARED]));
      const median = peek(outputs.composite);
      const step = peek(outputs.lastStep);
      if (median) {
        outputs.medianRed.write(Float32Array.of(median[index * BAND_COUNT + RED]));
        outputs.medianNearInfrared.write(
          Float32Array.of(median[index * BAND_COUNT + NEAR_INFRARED])
        );
      }
      const shift = median
        ? Math.hypot(
            median[index * BAND_COUNT + RED] - mean[RED],
            median[index * BAND_COUNT + NEAR_INFRARED] - mean[NEAR_INFRARED]
          )
        : Number.NaN;
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
        readout: `median moved (red + NIR) ${formatNumber(shift, 3)} from the mean | last step ${
          step ? formatNumber(step[index], 4) : '-'
        }`
      };
    }
  },
  draw: ({inputs, outputs, extra, theme}) => {
    const plot = {
      layer: 'chart',
      kind: 'scatter',
      xRange: RED_RANGE,
      range: NEAR_INFRARED_RANGE,
      rectangle: INSET
    } as const;
    return [
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
        ...plot,
        values: outputs.allNearInfrared,
        xValues: outputs.allRed,
        radius: 4.5,
        color: [1, 0.72, 0.2, 0.95]
      },
      {
        ...plot,
        values: outputs.acceptedNearInfrared,
        xValues: outputs.acceptedRed,
        radius: 4.5,
        color: getThemeColor(theme.ink, 0.8)
      },
      {
        ...plot,
        values: outputs.meanNearInfrared,
        xValues: outputs.meanRed,
        radius: 8,
        color: getThemeColor(theme.muted, 1)
      },
      {
        ...plot,
        values: outputs.medianNearInfrared,
        xValues: outputs.medianRed,
        radius: 8,
        color: getThemeColor(theme.categories[0], 1)
      }
    ];
  }
});
