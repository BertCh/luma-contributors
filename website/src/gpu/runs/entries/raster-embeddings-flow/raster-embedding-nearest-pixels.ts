// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterEmbeddingNearestPixels: the pointer picks a pixel, its vector is the query, and the
// operation returns the `resultCount` most similar pixels anywhere in the scene, ranked by cosine.
// Each result is drawn as a dot at its pixel, colored by its score. Vectors are six standardized
// Sentinel-2 bands, not a learned embedding.

import {GPURasterEmbeddingNearestPixels} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {
  createEmbeddingRaster,
  DIMENSIONS,
  getPixelAt,
  getPixelBox,
  getVector
} from './_lib/embedding';
import {addPixelPositions} from './_lib/kernels';
import {parameters} from './raster-embedding-nearest-pixels.parameters';

const RESULT_COUNTS = [8, 32, 128, 512];
const REST = [0.3, 0.4] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['resultCount'],
  caption:
    'Pick a pixel: the most similar pixels in the scene, ranked; ties order by pixel index. Six Sentinel-2 bands stand in for a learned embedding.',
  createInputs: ({aspect}) => {
    const raster = createEmbeddingRaster(aspect);
    const queryPixel =
      Math.floor(REST[1] * raster.rows) * raster.columns + Math.floor(REST[0] * raster.columns);
    return {...raster, queryPixel};
  },
  build(builder, raster, values) {
    const resultCount = RESULT_COUNTS[Math.round(values['resultCount'])];
    const vectors = builder.upload('vectors', 'float32', raster.vectors);
    const queries = builder.allocate('queries', 'float32', DIMENSIONS);
    const pixelRows = builder.allocate('pixel-rows', 'uint32', resultCount);
    const scores = builder.allocate('scores', 'float32', resultCount);
    const resultCounts = builder.allocate('result-counts', 'uint32', 1);
    builder.graph.add(
      new GPURasterEmbeddingNearestPixels({
        pixelCount: raster.pixelCount,
        dimensions: DIMENSIONS,
        vectors: vectors.view,
        queryCount: 1,
        queries: queries.view,
        resultCount,
        output: {pixelRows: pixelRows.view, scores: scores.view, resultCounts: resultCounts.view}
      })
    );
    const world = {width: raster.columns / raster.rows, height: 1};
    const positions = builder.allocate('positions', 'float32x2', resultCount);
    addPixelPositions(builder, 'result-positions', pixelRows, raster, world, positions);
    return {
      outputs: {
        positions,
        scores,
        nearInfrared: builder.upload('near-infrared', 'float32', raster.nearInfrared),
        vegetationIndex: builder.upload('vegetation-index', 'float32', raster.vegetationIndex)
      },
      update: () => queries.write(Float32Array.from(getVector(raster.vectors, raster.queryPixel)))
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'pick a pixel',
    rest: REST,
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const {index} = getPixelAt(inputs, world, pointer.x, pointer.y);
      const changed = index !== inputs.queryPixel;
      inputs.queryPixel = index;
      const scores = peek(outputs.scores);
      const last = scores ? scores[scores.length - 1] : Number.NaN;
      return {
        rerun: changed,
        spotlight: null,
        glyphs: [getPixelBox(inputs, world, index, 1)],
        readout: `resultCount ${RESULT_COUNTS[Math.round(values['resultCount'])]} | lowest score ${Number.isFinite(last) ? last.toFixed(3) : '-'}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      context: true,
      values: values['context'] === 0 ? outputs.nearInfrared : outputs.vegetationIndex,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'grey',
      smoothing: true
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.scores,
      ramp: 'sequential',
      range: 'auto',
      radius: 7
    }
  ]
});
