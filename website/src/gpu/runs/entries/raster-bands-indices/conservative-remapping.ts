// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterConservativeRemapping averages a fine raster onto a coarser, offset grid by exact
// overlap area: every destination cell takes the area-weighted mean of the source cells it covers
// (or their sum, for extensive quantities). Cells that hang over the source edge are only
// partly covered, which the normalization handles differently. The destination grid is a
// construction prop, so the run rebuilds when it changes.

import {GPURasterConservativeRemapping} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './conservative-remapping.parameters';
import {drawRaster} from './support/bands';

const NORMALIZATIONS = ['destination-area', 'covered-area', 'extensive'] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['factor', 'normalization', 'shift'],
  caption: 'A fine raster in; the area-weighted values of a coarse grid out.',
  createInputs: ({random, world}) => createFractalRaster(random, world, 60, 6),
  build(builder, inputs, values) {
    const cellSize = inputs.cellSize;
    const destinationSpacing = cellSize * values.factor;
    // The destination starts half a destination cell before the source, so border cells overhang.
    const origin = -destinationSpacing * values.shift;
    const destinationWidth = Math.ceil((inputs.columns * cellSize - origin) / destinationSpacing);
    const destinationHeight = Math.ceil((inputs.rows * cellSize - origin) / destinationSpacing);
    const source = builder.upload('source', 'float32', inputs.values);
    const result = builder.allocate('values', 'float32', destinationWidth * destinationHeight);
    const coveredArea = builder.allocate(
      'covered-area',
      'float32',
      destinationWidth * destinationHeight
    );
    const status = builder.allocate('status', 'uint32', destinationWidth * destinationHeight);
    builder.graph.add(
      new GPURasterConservativeRemapping({
        source: {
          width: inputs.columns,
          height: inputs.rows,
          origin: [0, 0],
          spacing: [cellSize, cellSize]
        },
        destination: {
          width: destinationWidth,
          height: destinationHeight,
          origin: [origin, origin],
          spacing: [destinationSpacing, destinationSpacing]
        },
        values: source.view,
        normalization: NORMALIZATIONS[Math.round(values.normalization)],
        output: {values: result.view, coveredArea: coveredArea.view, status: status.view}
      })
    );
    return {
      outputs: {source, result, columns: destinationWidth, rows: destinationHeight}
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.source, {context: true, range: [0, 1]}),
    drawRaster(outputs, outputs.result, {ramp: 'sequential'})
  ]
});
