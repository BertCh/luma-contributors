// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterBestAvailablePixel scores every valid observation of a cell against a target date and
// keeps the best acquisition, so a composite is always one real scene. The sixteen Sentinel-2
// scenes carry their real day of year and year; the cloud distance is measured in cells from each
// scene's QA no-data mask. The map is the NDVI of the winning scene in every cell (the burn scar
// appears once the target year is after the 2021 fire); the bars count the cells each scene won,
// oldest on the left. The map is tinted by the winning scene (a patchwork of dates), or shows the
// winner's NDVI. The pointer sets the target year (x) and the weight of cloud distance (y).

import {
  GPURasterBestAvailablePixel,
  getGPURasterBestAvailablePixelParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes, formatNumber} from '../../interaction';
import {createPanelData, getPanelLayers, getThemeColor, uploadPanel} from './support/panel';
import {parameters} from './best-available-pixel.parameters';
import {
  ACQUISITION_DATES,
  createAcquisitionStack,
  getSliceAttributes
} from './support/acquisitions';

const NO_SLICE = 0xffffffff;
const BARS = [0.05, 0.07, 0.55, 0.27] as const;
const PANEL = [0.03, 0.03, 0.57, 0.3] as const;

export default defineOperationRun({
  parameters,
  caption: 'Sixteen dated scenes in (cloud distance on the CPU); the best scene per cell out.',
  createInputs: ({aspect, world}) => ({
    ...createAcquisitionStack(aspect, 24),
    panel: createPanelData(world, [PANEL])
  }),
  build(builder, inputs) {
    const {cellCount, sliceCount} = inputs;
    const observations = builder.upload('observations', 'float32', inputs.observations);
    const validity = builder.upload('validity', 'uint32', inputs.validity);
    const cloudDistance = builder.upload('cloud-distance', 'float32', inputs.cloudDistance);
    const attributes = builder.upload('slice-attributes', 'float32', getSliceAttributes());
    const parameterBuffer = builder.allocate('parameters', 'float32', 9);
    const bestSlice = builder.allocate('best-slice', 'uint32', cellCount);
    const bestScore = builder.allocate('best-score', 'float32', cellCount);
    const composite = builder.allocate('composite', 'float32', cellCount);
    builder.graph.add(
      new GPURasterBestAvailablePixel({
        cellCount,
        sliceCount,
        bandCount: 1,
        values: observations.view,
        mask: validity.view,
        cloudDistance: cloudDistance.view,
        sliceAttributes: attributes.view,
        parameters: parameterBuffer.view,
        output: {bestSlice: bestSlice.view, bestScore: bestScore.view, composite: composite.view}
      })
    );
    return {
      outputs: {bestSlice, bestScore, composite, panel: uploadPanel(builder, inputs.panel)},
      update: current =>
        parameterBuffer.write(
          getGPURasterBestAvailablePixelParameterValues({
            targetDayOfYear: current.targetDayOfYear,
            dayOfYearSigma: current.dayOfYearSigma,
            dayOfYearWeight: 0.5,
            targetYear: current.targetYear,
            maximumYearDistance: 8,
            yearWeight: 0.05,
            minimumCloudDistance: 1,
            maximumCloudDistance: 6,
            cloudWeight: current.cloudWeight
          })
        )
    };
  },
  // The number of cells each scene won, for the bars.
  async afterRun({inputs, outputs, upload}) {
    const bytes = await outputs.bestSlice.buffer.readAsync();
    const slices = new Uint32Array(bytes.buffer, bytes.byteOffset, inputs.cellCount);
    const counts = new Float32Array(inputs.sliceCount);
    for (const slice of slices) if (slice !== NO_SLICE) counts[slice]++;
    // The winning scene as a number the map can tint (NaN where no scene was valid).
    const scene = Float32Array.from(slices, slice => (slice === NO_SLICE ? Number.NaN : slice));
    return {counts: upload('counts', 'float32', counts), scene: upload('scene', 'float32', scene)};
  },
  interaction: bindAxes<unknown, {bestSlice: RunBuffer<'uint32'>}>(parameters, {
    kind: 'pad',
    hint: 'target year and cloud weight',
    x: 'targetYear',
    y: 'cloudWeight',
    decorate: (values, _pointer, {outputs, peek}) => {
      const slices = peek(outputs.bestSlice);
      const counts = new Map<number, number>();
      for (const slice of slices ?? []) counts.set(slice, (counts.get(slice) ?? 0) + 1);
      const [winner] = [...counts]
        .filter(([slice]) => slice !== NO_SLICE)
        .sort((a, b) => b[1] - a[1]);
      return {
        readout: `target ${formatNumber(values.targetYear, 1)} | cloudWeight ${formatNumber(values.cloudWeight, 2)} | ${counts.size - (counts.has(NO_SLICE) ? 1 : 0)} scenes used${
          winner ? ` | most: ${ACQUISITION_DATES[winner[0]]}` : ''
        }`
      };
    }
  }),
  draw: ({inputs, outputs, extra, theme, values}) => [
    values.shown === 1
      ? {
          layer: 'raster',
          values: outputs.composite,
          columns: inputs.columns,
          rows: inputs.rows,
          ramp: 'sequential',
          range: [-0.1, 0.9]
        }
      : {
          layer: 'raster',
          values: extra.scene,
          columns: inputs.columns,
          rows: inputs.rows,
          ramp: 'sequential',
          range: [0, inputs.sliceCount - 1]
        },
    ...getPanelLayers(outputs.panel, theme),
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.counts,
      range: [0, inputs.cellCount * 0.6],
      rectangle: BARS,
      color: getThemeColor(theme.categories[0])
    }
  ]
});
