// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterEmbeddingPyramidLevel: one overview level of an embedding raster. Each overview pixel
// is the renormalized mean of the valid source vectors in its `scale x scale` footprint, the rule
// for unit-length embeddings (averaging the int8 bytes would be wrong). `meanLength` falls where a
// footprint mixes unlike pixels. Vectors are six standardized Sentinel-2 bands, with a share of
// source pixels marked invalid.

import {GPURasterEmbeddingPyramidLevel} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {createEmbeddingRaster, DIMENSIONS, getPixelAt} from './_lib/embedding';
import {addComponentExtract} from './_lib/kernels';
import {parameters} from './raster-embedding-pyramid-level.parameters';

const SCALES = [1, 2, 4, 8];

export default defineOperationRun({
  parameters,
  rebuildOn: ['scale'],
  buildCache: 5,
  caption:
    'Overview pixels average the valid vectors under them, then renormalize; dark marks a mixed footprint. 10% of source pixels are set invalid by default. Six Sentinel-2 bands stand in for a learned embedding.',
  createInputs: ({aspect, random}) => {
    const raster = createEmbeddingRaster(aspect, 48, 8);
    return {...raster, order: Float32Array.from({length: raster.pixelCount}, () => random())};
  },
  build(builder, raster, values) {
    const {scale, width, height} = getOverview(raster, values);
    const overviewCount = width * height;
    const validity = builder.allocate('validity', 'uint32', raster.pixelCount);
    const vectors = builder.allocate('overview-vectors', 'float32', overviewCount * DIMENSIONS);
    const validCount = builder.allocate('valid-count', 'uint32', overviewCount);
    const meanLength = builder.allocate('mean-length', 'float32', overviewCount);
    builder.graph.add(
      new GPURasterEmbeddingPyramidLevel({
        sourceWidth: raster.columns,
        sourceHeight: raster.rows,
        dimensions: DIMENSIONS,
        scale,
        vectors: builder.upload('vectors', 'float32', raster.vectors).view,
        mask: validity.view,
        output: {vectors: vectors.view, validCount: validCount.view, meanLength: meanLength.view}
      })
    );
    const component = builder.upload('component', 'uint32', Uint32Array.of(3));
    const band = builder.allocate('band', 'float32', overviewCount);
    addComponentExtract(builder, 'band-extract', vectors, DIMENSIONS, component, band);
    return {
      outputs: {band, meanLength, validCount},
      update: current =>
        validity.write(Uint32Array.from(raster.order, value => (value < current.noData ? 0 : 1)))
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a footprint',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const {width, height, scale} = getOverview(inputs, values);
      const {index, column, row} = getPixelAt(
        {columns: width, rows: height},
        world,
        pointer.x,
        pointer.y
      );
      const cellWidth = world.width / width;
      const cellHeight = world.height / height;
      const count = peek(outputs.validCount)?.[index];
      const length = peek(outputs.meanLength)?.[index];
      return {
        spotlight: null,
        glyphs: [
          {
            shape: 'box',
            minimum: [column * cellWidth, world.height - (row + 1) * cellHeight],
            maximum: [(column + 1) * cellWidth, world.height - row * cellHeight]
          }
        ],
        readout: `validCount ${count ?? '-'} of ${scale * scale} | meanLength ${length === undefined ? '-' : formatNumber(length, 3)}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => {
    const {width, height} = getOverview(inputs, values);
    return [
      values.shown === 0
        ? {
            layer: 'raster',
            values: outputs.band,
            columns: width,
            rows: height,
            ramp: 'diverging',
            range: [-1, 1]
          }
        : {
            layer: 'raster',
            values: outputs.meanLength,
            columns: width,
            rows: height,
            ramp: 'sequential',
            range: [0.3, 1]
          }
    ];
  }
});

function getOverview(raster: {columns: number; rows: number}, values: Record<string, number>) {
  const scale = SCALES[Math.round(values.scale)];
  return {scale, width: Math.ceil(raster.columns / scale), height: Math.ceil(raster.rows / scale)};
}
