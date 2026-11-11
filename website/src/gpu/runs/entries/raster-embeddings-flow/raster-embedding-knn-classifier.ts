// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterEmbeddingKNearestClassifier: a few labelled pixels per class (water, forest, open; picked by spectral
// rules, not field data) are the training set. Every pixel takes the majority class of its
// `neighborCount` nearest labelled vectors. Pointing at a pixel draws lines to those neighbours, so
// the vote is visible. Vectors are six standardized Sentinel-2 bands, not a learned embedding.

import {GPURasterEmbeddingKNearestClassifier} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {
  CLASS_NAMES,
  createEmbeddingRaster,
  createTrainingSet,
  DIMENSIONS,
  getPixelAt,
  getPixelBox,
  getPixelCenter,
  getVector
} from './_lib/embedding';
import {parameters} from './raster-embedding-knn-classifier.parameters';

const PER_CLASS = [3, 8, 20];

export default defineOperationRun({
  parameters,
  rebuildOn: ['neighborCount', 'perClass'],
  caption:
    'Each pixel votes among its nearest labelled pixels (dots); 24 labelled pixels by default (8 per class), labelled by spectral rules, not field data. Six Sentinel-2 bands stand in for a learned embedding.',
  createInputs: ({aspect, random}) => {
    const raster = createEmbeddingRaster(aspect);
    const world = {width: raster.columns / raster.rows, height: 1};
    const trainingSets = PER_CLASS.map(perClass => {
      const training = createTrainingSet(raster, random, perClass);
      const positions = Float32Array.from(
        training.pixels.flatMap(pixel => getPixelCenter(raster, world, pixel))
      );
      return {...training, positions};
    });
    return {...raster, trainingSets};
  },
  build(builder, raster, values) {
    const training = raster.trainingSets[Math.round(values.perClass)];
    const labels = builder.allocate('labels', 'uint32', raster.pixelCount);
    const confidence = builder.allocate('confidence', 'float32', raster.pixelCount);
    builder.graph.add(
      new GPURasterEmbeddingKNearestClassifier({
        pixelCount: raster.pixelCount,
        dimensions: DIMENSIONS,
        vectors: builder.upload('vectors', 'float32', raster.vectors).view,
        trainingVectors: builder.upload('training-vectors', 'float32', training.vectors).view,
        trainingLabels: builder.upload('training-labels', 'uint32', training.labels).view,
        trainingCount: training.count,
        classCount: CLASS_NAMES.length,
        neighborCount: Math.min(training.count, Math.round(values.neighborCount)),
        metric: 'euclidean',
        output: {labels: labels.view, confidence: confidence.view}
      })
    );
    return {
      outputs: {
        labels,
        confidence,
        samplePositions: builder.upload('sample-positions', 'float32x2', training.positions),
        sampleLabels: builder.upload('sample-labels', 'uint32', training.labels)
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a pixel',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const {index} = getPixelAt(inputs, world, pointer.x, pointer.y);
      const vector = getVector(inputs.vectors, index);
      const training = inputs.trainingSets[Math.round(values.perClass)];
      const distances = training.pixels.map((_, sample) => {
        const other = getVector(training.vectors, sample);
        return Math.hypot(...vector.map((value, band) => value - other[band]));
      });
      const neighbors = Array.from(distances.keys())
        .sort((left, right) => distances[left] - distances[right] || left - right)
        .slice(0, Math.round(values.neighborCount));
      const at = getPixelCenter(inputs, world, index);
      const label = peek(outputs.labels)?.[index];
      const confidence = peek(outputs.confidence)?.[index];
      return {
        spotlight: null,
        glyphs: [
          getPixelBox(inputs, world, index),
          ...neighbors.map(sample => ({
            shape: 'line' as const,
            from: at,
            to: [training.positions[sample * 2], training.positions[sample * 2 + 1]] as [
              number,
              number
            ]
          }))
        ],
        readout: `${label === undefined ? '-' : (CLASS_NAMES[label] ?? 'none')} | confidence ${confidence === undefined ? '-' : formatNumber(confidence)}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    values.shown === 0
      ? {layer: 'raster', values: outputs.labels, columns: inputs.columns, rows: inputs.rows}
      : {
          layer: 'raster',
          values: outputs.confidence,
          columns: inputs.columns,
          rows: inputs.rows,
          ramp: 'sequential',
          range: [0, 1]
        },
    {
      layer: 'dots',
      context: true,
      positions: outputs.samplePositions,
      values: outputs.sampleLabels,
      radius: 5
    }
  ]
});
