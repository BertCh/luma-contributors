// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterFillNoData: gaps in a raster (cloud shadows, sensor dropouts) are interpolated from the
// valid cells within `maximumDistance`. Cells farther than that from any donor stay missing.
// Method and distance are construction props (the run rebuilds); the gap slider rewrites the input.

import {GPURasterFillNoData} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster, createSmoothField} from '../../inputs';
import {parameters} from './fill-nodata.parameters';
import {drawRaster, makeGridOfRaster, uploadBand} from './support/bands';

const METHODS = ['nearest', 'inverse-distance', 'gdal-four-quadrant'] as const;
const NO_DATA = -9999;

export default defineOperationRun({
  parameters,
  rebuildOn: ['method', 'maximumDistance'],
  caption: 'A raster with gaps in; the gaps interpolated from nearby valid cells out.',
  createInputs: ({random, world}) => {
    const raster = createFractalRaster(random, world, 64, 4);
    const gapField = createSmoothField(random, 4, 5);
    const gap = Float32Array.from(raster.values, (_, index) =>
      gapField(
        ((index % raster.columns) + 0.5) / raster.columns,
        (Math.floor(index / raster.columns) + 0.5) / raster.rows
      )
    );
    return {...raster, gap};
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const source = builder.upload('source', 'float32', inputs.values);
    const band = uploadBand(builder, inputs, 'band', inputs.values);
    const result = builder.allocate('values', 'float32', cellCount);
    const validity = builder.allocate('validity', 'uint32', cellCount);
    const filled = builder.allocate('filled', 'uint32', cellCount);
    const unresolvedCount = builder.allocate('unresolved', 'uint32', 1);
    builder.graph.add(
      new GPURasterFillNoData({
        grid: makeGridOfRaster(inputs),
        input: {...band.band, noDataValue: NO_DATA},
        maximumDistance: values.maximumDistance,
        method: METHODS[Math.round(values.method)],
        output: {
          values: result.view,
          validity: validity.view,
          filled: filled.view,
          unresolvedCount: unresolvedCount.view
        }
      })
    );
    return {
      outputs: {source, result},
      update: current => {
        const holed = Float32Array.from(inputs.values, (value, index) =>
          inputs.gap[index] > current.gaps ? NO_DATA : value
        );
        band.buffer.write(holed);
        source.write(holed.map(value => (value === NO_DATA ? Number.NaN : value)));
      }
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.source, {context: true, range: [0, 1]}),
    drawRaster(inputs, outputs.result, {range: [0, 1], smoothing: true})
  ]
});
