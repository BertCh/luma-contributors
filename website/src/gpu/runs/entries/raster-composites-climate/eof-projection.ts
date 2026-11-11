// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterEOFProjection is the last step of EOF analysis by the method of snapshots: given the
// temporal modes and singular values from the CPU eigen step, it projects the field onto them and
// writes one spatial map per mode. Here the field is the Dixie Fire NDVI scenes (2017 to 2024); the
// run does the CPU step itself (a binary64 Gram matrix and getGPURasterSnapshotEOFDecomposition),
// uploads the modes, and draws the chosen EOF map with its temporal mode below: mode 1 is the burn
// scar, switching on in the September 2021 scene. `sliceCount` is a construction prop (the dates
// used). The pointer picks the mode (x) and the date (y) whose share of the field is drawn.

import {
  GPURasterEOFProjection,
  getGPURasterSnapshotCovarianceParameterValues,
  getGPURasterSnapshotEOFDecomposition
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {bindAxes, formatNumber} from '../../interaction';
import {
  createPanelData,
  getPanelLayers,
  getPanelRectangle,
  getThemeColor,
  uploadPanel
} from './support/panel';
import {parameters} from './eof-projection.parameters';
import {ACQUISITION_DATES} from './support/acquisitions';
import {createSnapshotField, getGramMatrix, packSnapshots} from './support/snapshots';

const MODE_COUNT = 4;
const MAXIMUM_VALUE_MAGNITUDE = 1;
const SCREE = [0.62, 0.05, 0.97, 0.3] as const;
const INSET = [0.03, 0.05, 0.4, 0.3] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['sliceCount'],
  buildCache: 6,
  caption: 'Dixie NDVI and CPU Gram-matrix modes in; one spatial map per mode out.',
  createInputs: ({aspect, world}) => ({
    ...createSnapshotField(aspect, 24),
    panel: createPanelData(world, [getPanelRectangle(INSET), getPanelRectangle(SCREE)])
  }),
  build(builder, inputs, values) {
    const sliceCount = Math.round(values.sliceCount);
    const packed = packSnapshots(inputs, sliceCount);
    const {matrix} = getGramMatrix(packed, inputs.cellCount, sliceCount, MAXIMUM_VALUE_MAGNITUDE);
    const decomposition = getGPURasterSnapshotEOFDecomposition(matrix, sliceCount, MODE_COUNT);
    const results = {
      explained: Float64Array.from(decomposition.explainedVarianceRatio),
      temporal: Float32Array.from(decomposition.temporalModes),
      singular: Float32Array.from(decomposition.singularValues)
    };
    const eofs = builder.allocate('eofs', 'float32', inputs.cellCount * MODE_COUNT);
    const status = builder.allocate('status', 'uint32', inputs.cellCount);
    builder.graph.add(
      new GPURasterEOFProjection({
        cellCount: inputs.cellCount,
        sliceCount,
        modeCount: MODE_COUNT,
        values: builder.upload('values', 'float32', packed).view,
        parameters: builder.upload(
          'parameters',
          'float32',
          getGPURasterSnapshotCovarianceParameterValues(
            {maximumValueMagnitude: MAXIMUM_VALUE_MAGNITUDE},
            inputs.cellCount
          )
        ).view,
        temporalModes: builder.upload('temporal-modes', 'float32', decomposition.temporalModes)
          .view,
        singularValues: builder.upload('singular-values', 'float32', decomposition.singularValues)
          .view,
        output: {eofs: eofs.view, status: status.view}
      })
    );
    const temporalLine = builder.allocate('temporal-line', 'float32', sliceCount);
    const scree = builder.allocate('scree', 'float32', MODE_COUNT);
    scree.write(Float32Array.from(results.explained));
    return {
      outputs: {eofs, temporalLine, scree, results, panel: uploadPanel(builder, inputs.panel)},
      update: current => {
        const mode = Math.round(current.mode);
        temporalLine.write(
          decomposition.temporalModes.slice(mode * sliceCount, (mode + 1) * sliceCount)
        );
      }
    };
  },
  // One mode of the interleaved EOF maps, for the raster.
  async afterRun({inputs, outputs, values, upload}) {
    const bytes = await outputs.eofs.buffer.readAsync();
    const eofs = new Float32Array(bytes.buffer, bytes.byteOffset, inputs.cellCount * MODE_COUNT);
    const mode = Math.round(values.mode);
    const sliceCount = Math.round(values.sliceCount);
    const date = Math.min(sliceCount - 1, Math.round(values.date));
    // The part of the field this mode carries on the chosen date: EOF x temporal mode x singular value.
    const scale =
      outputs.results.temporal[mode * sliceCount + date] * outputs.results.singular[mode];
    let peakScale = 0;
    for (let slice = 0; slice < sliceCount; slice++) {
      peakScale = Math.max(
        peakScale,
        Math.abs(
          outputs.results.temporal[mode * sliceCount + slice] * outputs.results.singular[mode]
        )
      );
    }
    const shown = Float32Array.from(
      {length: inputs.cellCount},
      (_, cell) => eofs[cell * MODE_COUNT + mode] * scale
    );
    // A few extreme cells would flatten the ramp: it spans the 97th percentile of the mode's
    // magnitude at its strongest date, so the map brightens and fades as the date moves.
    const magnitudes = Array.from({length: inputs.cellCount}, (_, cell) =>
      Math.abs(eofs[cell * MODE_COUNT + mode])
    )
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    const limit = (magnitudes[Math.floor(magnitudes.length * 0.97)] || 1) * peakScale || 1;
    return {shown: upload('shown', 'float32', shown), limit};
  },
  interaction: bindAxes<unknown, {results: {explained: Float64Array}}>(parameters, {
    kind: 'pad',
    hint: 'pick a mode and a date',
    x: 'mode',
    y: 'date',
    decorate: (values, _pointer, {outputs}) => {
      const mode = Math.round(values.mode);
      const sliceCount = Math.round(values.sliceCount);
      return {
        readout: `EOF ${mode + 1} of ${MODE_COUNT} on ${ACQUISITION_DATES[Math.min(sliceCount - 1, Math.round(values.date))]} | ${formatNumber(100 * outputs.results.explained[mode], 1)}% of variance | ${ACQUISITION_DATES[0]} to ${ACQUISITION_DATES[sliceCount - 1]}`
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
      range: [-extra.limit, extra.limit]
    },
    ...getPanelLayers(outputs.panel, theme),
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.temporalLine,
      range: [-1, 1],
      rectangle: INSET,
      color: getThemeColor(theme.ink, 0.95)
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.scree,
      range: [0, 1],
      rectangle: SCREE,
      color: getThemeColor(theme.categories[0], 0.95)
    }
  ]
});
