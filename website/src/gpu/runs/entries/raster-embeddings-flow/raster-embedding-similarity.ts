// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterEmbeddingSimilarity: the pointer picks a pixel and its vector becomes the query; every
// pixel gets its cosine similarity to it (`maximum`), and `matches` outlines the pixels at or above
// `matchThreshold`. Vectors are six standardized Sentinel-2 bands, not a learned embedding.

import {
  getGPURasterEmbeddingSimilarityParameterValues,
  GPURasterEmbeddingSimilarity
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {
  createEmbeddingRaster,
  DIMENSIONS,
  getPixelAt,
  getPixelBox,
  getVector
} from './_lib/embedding';
import {parameters} from './raster-embedding-similarity.parameters';

const MATCH_CATEGORIES = [
  [0, 0, 0, 0],
  [1, 1, 1, 0.6]
] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Pick a pixel: places like it light up (cosine similarity 0.5 to 1). Six Sentinel-2 bands stand in for a learned embedding.',
  createInputs: ({aspect}) => ({...createEmbeddingRaster(aspect), queryPixel: -1}),
  build(builder, raster) {
    const vectors = builder.upload('vectors', 'float32', raster.vectors);
    const nearInfrared = builder.upload('near-infrared', 'float32', raster.nearInfrared);
    const queries = builder.allocate('queries', 'float32', DIMENSIONS);
    const parameterBuffer = builder.allocate('parameters', 'float32', 1);
    const maximum = builder.allocate('maximum', 'float32', raster.pixelCount);
    const matches = builder.allocate('matches', 'uint32', raster.pixelCount);
    builder.graph.add(
      new GPURasterEmbeddingSimilarity({
        pixelCount: raster.pixelCount,
        dimensions: DIMENSIONS,
        vectors: vectors.view,
        queryCount: 1,
        queries: queries.view,
        parameters: parameterBuffer.view,
        output: {maximum: maximum.view, matches: matches.view}
      })
    );
    return {
      outputs: {maximum, matches, nearInfrared},
      update: values => {
        const pixel = raster.queryPixel < 0 ? getDefaultPixel(raster) : raster.queryPixel;
        queries.write(Float32Array.from(getVector(raster.vectors, pixel)));
        parameterBuffer.write(
          getGPURasterEmbeddingSimilarityParameterValues({matchThreshold: values.matchThreshold})
        );
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'pick a pixel',
    rest: [0.3, 0.4],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const {index} = getPixelAt(inputs, world, pointer.x, pointer.y);
      const changed = index !== inputs.queryPixel;
      inputs.queryPixel = index;
      const matches = peek(outputs.matches);
      let matchCount = 0;
      if (matches) for (const value of matches) matchCount += value;
      return {
        rerun: changed,
        spotlight: null,
        glyphs: [getPixelBox(inputs, world, index, 1)],
        readout: `${matchCount} px at or above ${values.matchThreshold.toFixed(2)} · query [${index % inputs.columns}, ${Math.floor(index / inputs.columns)}]`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    ...(values.shown === 0
      ? [
          {
            layer: 'raster' as const,
            context: true,
            values: outputs.nearInfrared,
            columns: inputs.columns,
            rows: inputs.rows,
            ramp: 'grey' as const,
            smoothing: true
          },
          {
            layer: 'raster' as const,
            values: outputs.maximum,
            columns: inputs.columns,
            rows: inputs.rows,
            ramp: 'sequential' as const,
            range: [0.5, 1] as const,
            fade: true
          }
        ]
      : []),
    {
      layer: 'raster',
      values: outputs.matches,
      columns: inputs.columns,
      rows: inputs.rows,
      categories: values.shown === 0 ? MATCH_CATEGORIES : undefined,
      ramp: values.shown === 0 ? undefined : 'sequential',
      range: values.shown === 0 ? undefined : [0, 1]
    }
  ]
});

/** The pixel under `rest`, so the poster and the first live frame agree. */
function getDefaultPixel(raster: {columns: number; rows: number}): number {
  return Math.floor(0.4 * raster.rows) * raster.columns + Math.floor(0.3 * raster.columns);
}
