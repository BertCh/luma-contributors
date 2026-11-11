// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterSnapshotCovariance is the GPU half of EOF analysis by the method of snapshots: it removes
// each cell's temporal mean and accumulates, over cells, the products of every pair of snapshots
// into a sliceCount by sliceCount Gram matrix, as exact fixed-point words. The run reads the words
// back (the explicit readback the operation prescribes), decodes the matrix and draws it; the bars
// are the share of variance in its leading modes, from the CPU eigen step. The snapshots are the
// Dixie Fire NDVI scenes (2017 to 2024): a cell with a missing observation in the dates used is
// dropped entirely, and `maximumValueMagnitude` drops cells with a larger sample. The pointer
// probes one matrix entry.

import {
  GPURasterSnapshotCovariance,
  getGPURasterSnapshotCovarianceMatrix,
  getGPURasterSnapshotCovarianceParameterValues,
  getGPURasterSnapshotEOFDecomposition
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
import {parameters} from './snapshot-covariance.parameters';
import {getCellAt, getCellBox} from '../raster-bands-indices/support/gestures';
import {ACQUISITION_DATES} from './support/acquisitions';
import {createSnapshotField, packSnapshots} from './support/snapshots';

const MODE_COUNT = 6;
const BARS = [0.62, 0.05, 0.97, 0.3] as const;

function decodeMatrix(
  words: ArrayLike<number>,
  sliceCount: number,
  cellCount: number,
  magnitude: number
) {
  const unitExponent = getGPURasterSnapshotCovarianceParameterValues(
    {maximumValueMagnitude: magnitude},
    cellCount
  )[0];
  return getGPURasterSnapshotCovarianceMatrix(words, sliceCount, unitExponent);
}

export default defineOperationRun({
  parameters,
  rebuildOn: ['sliceCount'],
  buildCache: 4,
  caption: 'Dixie NDVI snapshots in; the Gram matrix of every date pair out, decoded on the CPU.',
  createInputs: ({aspect, world}) => ({
    ...createSnapshotField(aspect, 24),
    panel: createPanelData(world, [getPanelRectangle(BARS)])
  }),
  build(builder, inputs, values) {
    const sliceCount = Math.round(values.sliceCount);
    const entryCount = (sliceCount * (sliceCount + 1)) / 2;
    const parameterBuffer = builder.allocate('parameters', 'float32', 2);
    const covarianceWords = builder.allocate('covariance-words', 'uint32', entryCount * 3);
    const keptCellCount = builder.allocate('kept-cell-count', 'uint32', 1);
    builder.graph.add(
      new GPURasterSnapshotCovariance({
        cellCount: inputs.cellCount,
        sliceCount,
        values: builder.upload('values', 'float32', packSnapshots(inputs, sliceCount)).view,
        parameters: parameterBuffer.view,
        output: {covarianceWords: covarianceWords.view, keptCellCount: keptCellCount.view}
      })
    );
    return {
      outputs: {covarianceWords, keptCellCount, panel: uploadPanel(builder, inputs.panel)},
      update: current =>
        parameterBuffer.write(
          getGPURasterSnapshotCovarianceParameterValues(
            {maximumValueMagnitude: current.maximumValueMagnitude},
            inputs.cellCount
          )
        )
    };
  },
  // The readback and the CPU eigen step: the matrix to draw and the variance share of its modes.
  async afterRun({inputs, outputs, values, upload}) {
    const sliceCount = Math.round(values.sliceCount);
    const bytes = await outputs.covarianceWords.buffer.readAsync();
    const words = new Uint32Array(
      bytes.buffer,
      bytes.byteOffset,
      ((sliceCount * (sliceCount + 1)) / 2) * 3
    );
    const matrix = decodeMatrix(words, sliceCount, inputs.cellCount, values.maximumValueMagnitude);
    const {explainedVarianceRatio} = getGPURasterSnapshotEOFDecomposition(
      matrix,
      sliceCount,
      Math.min(MODE_COUNT, sliceCount)
    );
    const ratios = new Float32Array(MODE_COUNT);
    ratios.set(Float32Array.from(explainedVarianceRatio));
    return {
      matrix: upload('matrix', 'float32', Float32Array.from(matrix)),
      ratios: upload('ratios', 'float32', ratios)
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a pair of dates',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, values, world, peek}) => {
      const sliceCount = Math.round(values.sliceCount);
      const grid = {columns: sliceCount, rows: sliceCount};
      const {column, row} = getCellAt(grid, world, pointer);
      const words = peek(outputs.covarianceWords);
      const kept = peek(outputs.keptCellCount);
      const matrix = words
        ? decodeMatrix(words, sliceCount, inputs.cellCount, values.maximumValueMagnitude)
        : null;
      return {
        spotlight: null,
        glyphs: [getCellBox(grid, world, column, row), getCellBox(grid, world, row, column)],
        readout: `${ACQUISITION_DATES[row]} x ${ACQUISITION_DATES[column]}: ${
          matrix ? formatNumber(matrix[row * sliceCount + column], 1) : '-'
        } | ${kept ? kept[0] : '-'} of ${inputs.cellCount} cells kept`
      };
    }
  },
  draw: ({outputs, extra, values, theme}) => [
    {
      layer: 'raster',
      values: extra.matrix,
      columns: Math.round(values.sliceCount),
      rows: Math.round(values.sliceCount),
      ramp: 'diverging'
    },
    ...getPanelLayers(outputs.panel, theme),
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.ratios,
      range: [0, 1],
      rectangle: BARS,
      color: getThemeColor(theme.categories[0], 0.95)
    }
  ]
});
