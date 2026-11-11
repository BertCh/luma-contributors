// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterEmbeddingZoneMean: the mean vector of every zone (here square blocks given as a
// `zoneRaster`). `meanLengths` is the length of the mean before renormalization: near 1 where a
// zone's pixels agree, lower where it mixes water and forest. With `normalize` the means are unit
// length again (the default), ready as similarity queries. Vectors are six standardized Sentinel-2 bands.

import {GPURasterEmbeddingZoneMean} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {createEmbeddingRaster, DIMENSIONS, getPixelAt} from './_lib/embedding';
import {addZoneGather} from './_lib/kernels';
import {parameters} from './raster-embedding-zone-mean.parameters';

const BLOCK_SIZES = [2, 4, 6, 9];

export default defineOperationRun({
  parameters,
  rebuildOn: ['blockSize'],
  buildCache: 5,
  caption:
    'Mean vector per zone; dark (low meanLengths) marks a zone that mixes unlike pixels. Six Sentinel-2 bands stand in for a learned embedding.',
  createInputs: ({aspect}) => createEmbeddingRaster(aspect),
  build(builder, raster, values) {
    const blockSize = BLOCK_SIZES[Math.round(values.blockSize)];
    const zoneColumns = Math.ceil(raster.columns / blockSize);
    const zoneCount = zoneColumns * Math.ceil(raster.rows / blockSize);
    const zoneIds = Uint32Array.from(
      {length: raster.pixelCount},
      (_, pixel) =>
        Math.floor(Math.floor(pixel / raster.columns) / blockSize) * zoneColumns +
        Math.floor((pixel % raster.columns) / blockSize)
    );
    const zoneRaster = builder.upload('zones', 'uint32', zoneIds);
    const means = builder.allocate('means', 'float32', zoneCount * DIMENSIONS);
    const meanLengths = builder.allocate('mean-lengths', 'float32', zoneCount);
    const weights = builder.allocate('weights', 'float32', zoneCount);
    builder.graph.add(
      new GPURasterEmbeddingZoneMean({
        pixelCount: raster.pixelCount,
        dimensions: DIMENSIONS,
        zoneCount,
        vectors: builder.upload('vectors', 'float32', raster.vectors).view,
        zoneRaster: zoneRaster.view,
        normalize: true,
        output: {means: means.view, weights: weights.view, meanLengths: meanLengths.view}
      })
    );
    const zero = builder.upload('component-zero', 'uint32', Uint32Array.of(0));
    const nearInfrared = builder.upload('component-nir', 'uint32', Uint32Array.of(3));
    const lengthPicture = builder.allocate('length-picture', 'float32', raster.pixelCount);
    const meanPicture = builder.allocate('mean-picture', 'float32', raster.pixelCount);
    addZoneGather(builder, 'length-gather', meanLengths, 1, zoneRaster, zero, lengthPicture);
    addZoneGather(builder, 'mean-gather', means, DIMENSIONS, zoneRaster, nearInfrared, meanPicture);
    return {
      outputs: {lengthPicture, meanPicture, meanLengths, weights, means},
      update: () => {}
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a zone',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const blockSize = BLOCK_SIZES[Math.round(values.blockSize)];
      const {column, row} = getPixelAt(inputs, world, pointer.x, pointer.y);
      const zoneColumns = Math.ceil(inputs.columns / blockSize);
      const zone = Math.floor(row / blockSize) * zoneColumns + Math.floor(column / blockSize);
      const cellWidth = world.width / inputs.columns;
      const cellHeight = world.height / inputs.rows;
      const left = Math.floor(column / blockSize) * blockSize;
      const top = Math.floor(row / blockSize) * blockSize;
      const length = peek(outputs.meanLengths)?.[zone];
      const weight = peek(outputs.weights)?.[zone];
      return {
        focus: zone,
        spotlight: null,
        glyphs: [
          {
            shape: 'box',
            minimum: [
              left * cellWidth,
              world.height - Math.min(inputs.rows, top + blockSize) * cellHeight
            ],
            maximum: [
              Math.min(inputs.columns, left + blockSize) * cellWidth,
              world.height - top * cellHeight
            ]
          }
        ],
        readout: `meanLengths ${length === undefined ? '-' : formatNumber(length, 3)} | weights ${weight ?? '-'}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    values.shown === 0
      ? {
          layer: 'raster',
          values: outputs.lengthPicture,
          columns: inputs.columns,
          rows: inputs.rows,
          ramp: 'sequential',
          range: [0.3, 1]
        }
      : {
          layer: 'raster',
          values: outputs.meanPicture,
          columns: inputs.columns,
          rows: inputs.rows,
          ramp: 'diverging',
          range: [-1, 1]
        }
  ]
});
