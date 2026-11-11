// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterEmbeddingChange: two years of vectors, compared pixel by pixel. "Before" is the scene's
// six standardized Sentinel-2 bands; "after" is a modelled later year: the same vectors with a
// little per-pixel noise, and a clearing inside a fixed disc where the vector of the scene's barest
// pixel is mixed in by `severity` times each pixel's own vegetation (dense forest changes most, bare
// ground hardly at all). The gesture is a level: the similarity under the pointer becomes
// `similarityThreshold`, so `changed` is everything that changed at least as much as there.

import {
  getGPURasterEmbeddingChangeParameterValues,
  GPURasterEmbeddingChange
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {
  createEmbeddingRaster,
  DIMENSIONS,
  getPixelAt,
  getPixelBox,
  getCosine,
  getPixelCenter,
  getVector,
  normalizeVector
} from './_lib/embedding';
import {parameters} from './raster-embedding-change.parameters';

const CLEARING_RADIUS = 0.28;
const CLEARING = [0.4, 0.45] as const;
const REST = [0.43, 0.47] as const;
const MAXIMUM_THRESHOLD = 0.995;
const CHANGED_CATEGORIES = [
  [0, 0, 0, 0],
  [1, 1, 1, 0.45]
] as const;

export default defineOperationRun({
  parameters,
  caption:
    "Point at a pixel: its similarity between years becomes the threshold, and everything that changed as much is marked. The later year is modelled: a clearing scaled by each pixel's vegetation. Six Sentinel-2 bands stand in for a learned embedding.",
  createInputs: ({aspect, random}) => {
    const raster = createEmbeddingRaster(aspect);
    const world = {width: raster.columns / raster.rows, height: 1};
    let barest = 0;
    let lowest = Number.POSITIVE_INFINITY;
    let highest = Number.NEGATIVE_INFINITY;
    for (let pixel = 0; pixel < raster.pixelCount; pixel++) {
      const value = raster.vegetationIndex[pixel];
      if (value < raster.vegetationIndex[barest]) barest = pixel;
      lowest = Math.min(lowest, value);
      highest = Math.max(highest, value);
    }
    const center = [CLEARING[0] * world.width, 1 - CLEARING[1]];
    // Share of the clearing each pixel takes: its vegetation, inside the disc, with a soft edge.
    const exposure = Float32Array.from({length: raster.pixelCount}, (_, pixel) => {
      const [x, y] = getPixelCenter(raster, world, pixel);
      const distance = Math.hypot(x - center[0], y - center[1]) / CLEARING_RADIUS;
      const edge = Math.min(1, Math.max(0, (1 - distance) * 4));
      return (edge * (raster.vegetationIndex[pixel] - lowest)) / (highest - lowest || 1);
    });
    const noise = Float32Array.from({length: raster.vectors.length}, () => random() * 2 - 1);
    const restPixel =
      Math.floor(REST[1] * raster.rows) * raster.columns + Math.floor(REST[0] * raster.columns);
    return {...raster, barest, noise, exposure, center, levelPixel: restPixel, threshold: 0.7};
  },
  build(builder, raster) {
    const before = builder.upload('before', 'float32', raster.vectors);
    const after = builder.allocate('after', 'float32', raster.vectors.length);
    const parameterBuffer = builder.allocate('parameters', 'float32', 1);
    const similarity = builder.allocate('similarity', 'float32', raster.pixelCount);
    const magnitude = builder.allocate('magnitude', 'float32', raster.pixelCount);
    const changed = builder.allocate('changed', 'uint32', raster.pixelCount);
    builder.graph.add(
      new GPURasterEmbeddingChange({
        pixelCount: raster.pixelCount,
        dimensions: DIMENSIONS,
        before: before.view,
        after: after.view,
        parameters: parameterBuffer.view,
        output: {similarity: similarity.view, magnitude: magnitude.view, changed: changed.view}
      })
    );
    return {
      outputs: {
        similarity,
        magnitude,
        changed,
        nearInfrared: builder.upload('near-infrared', 'float32', raster.nearInfrared)
      },
      update: values => {
        const data = Float32Array.from(raster.vectors);
        for (let pixel = 0; pixel < raster.pixelCount; pixel++) {
          const mix = values.severity * raster.exposure[pixel];
          for (let band = 0; band < DIMENSIONS; band++) {
            const element = pixel * DIMENSIONS + band;
            data[element] =
              (1 - mix) * data[element] +
              mix * raster.vectors[raster.barest * DIMENSIONS + band] +
              values.noise * 0.5 * raster.noise[element];
          }
          normalizeVector(data, pixel);
        }
        after.write(data);
        // The level: the similarity of the pointed pixel, just above it so that pixel counts.
        const here = getCosine(
          getVector(raster.vectors, raster.levelPixel),
          getVector(data, raster.levelPixel)
        );
        raster.threshold = Math.min(MAXIMUM_THRESHOLD, Math.ceil((here + 0.0005) * 1000) / 1000);
        parameterBuffer.write(
          getGPURasterEmbeddingChangeParameterValues({similarityThreshold: raster.threshold})
        );
      }
    };
  },
  interaction: {
    kind: 'level',
    hint: 'set the threshold',
    rest: REST,
    resolve(pointer, {inputs, outputs, world, peek}) {
      const {index} = getPixelAt(inputs, world, pointer.x, pointer.y);
      const moved = index !== inputs.levelPixel;
      inputs.levelPixel = index;
      const changed = peek(outputs.changed);
      let changedCount = 0;
      if (changed) for (const value of changed) changedCount += value;
      return {
        rerun: moved,
        spotlight: null,
        glyphs: [
          getPixelBox(inputs, world, index),
          {
            shape: 'ring',
            at: inputs.center as [number, number],
            radius: CLEARING_RADIUS,
            dashed: true
          }
        ],
        readout: `similarityThreshold ${inputs.threshold.toFixed(3)} · ${((100 * changedCount) / inputs.pixelCount).toFixed(1)}% changed`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.nearInfrared,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'grey',
      smoothing: true
    },
    {
      layer: 'raster',
      values: outputs.magnitude,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: [0, 2],
      fade: true
    },
    {
      layer: 'raster',
      values: outputs.changed,
      columns: inputs.columns,
      rows: inputs.rows,
      categories: CHANGED_CATEGORIES
    }
  ]
});
