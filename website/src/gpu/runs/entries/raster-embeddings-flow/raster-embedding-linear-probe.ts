// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterEmbeddingLinearProbe: `softmax(W x + b)` per pixel with host-trained weights. The
// weights are a three-class logistic regression fitted here on 24 rule-picked pixels (water,
// forest, open). `Data: weights scale` multiplies `weights` and `bias`, which sharpens or flattens
// the probabilities without moving the boundaries much. The probe reads one pixel's probabilities
// into the bars. Vectors are six standardized Sentinel-2 bands, not a learned embedding.

import {GPURasterEmbeddingLinearProbe} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {
  CLASS_NAMES,
  createEmbeddingRaster,
  createTrainingSet,
  DIMENSIONS,
  getPixelAt,
  getPixelBox
} from './_lib/embedding';
import {createColoredBars, getColoredBarLayers, writeColoredBars} from './_lib/insets';
import {addComponentExtract} from './_lib/kernels';
import {parameters} from './raster-embedding-linear-probe.parameters';

const CLASS_COUNT = CLASS_NAMES.length;

export default defineOperationRun({
  parameters,
  caption:
    "A linear probe trained on the host on 24 pixels labelled by spectral rules, not field data, applied to every pixel; bars are one pixel's class probabilities. Six Sentinel-2 bands stand in for a learned embedding.",
  createInputs: ({aspect, random}) => {
    const raster = createEmbeddingRaster(aspect);
    const training = createTrainingSet(raster, random);
    return {
      ...raster,
      ...trainLogisticRegression(training.vectors, training.labels, training.count)
    };
  },
  build(builder, raster) {
    const weights = builder.allocate('weights', 'float32', CLASS_COUNT * DIMENSIONS);
    const bias = builder.allocate('bias', 'float32', CLASS_COUNT);
    const labels = builder.allocate('labels', 'uint32', raster.pixelCount);
    const probabilities = builder.allocate(
      'probabilities',
      'float32',
      raster.pixelCount * CLASS_COUNT
    );
    builder.graph.add(
      new GPURasterEmbeddingLinearProbe({
        pixelCount: raster.pixelCount,
        dimensions: DIMENSIONS,
        vectors: builder.upload('vectors', 'float32', raster.vectors).view,
        classCount: CLASS_COUNT,
        weights: weights.view,
        bias: bias.view,
        output: {labels: labels.view, probabilities: probabilities.view}
      })
    );
    const component = builder.allocate('component', 'uint32', 1);
    const probability = builder.allocate('probability', 'float32', raster.pixelCount);
    addComponentExtract(
      builder,
      'probability-extract',
      probabilities,
      CLASS_COUNT,
      component,
      probability
    );
    return {
      outputs: {
        labels,
        probabilities,
        probability,
        inset: createColoredBars(builder, 'inset', CLASS_COUNT, raster.columns / raster.rows)
      },
      update: values => {
        weights.write(raster.weights.map(value => value * values.weightScale));
        bias.write(raster.bias.map(value => value * values.weightScale));
        component.write(Uint32Array.of(Math.max(0, Math.round(values.shown) - 1)));
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a pixel',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const {index} = getPixelAt(inputs, world, pointer.x, pointer.y);
      const probabilities = peek(outputs.probabilities);
      const shares = Float32Array.from({length: CLASS_COUNT}, (_, label) =>
        probabilities ? probabilities[index * CLASS_COUNT + label] : 0
      );
      writeColoredBars(outputs.inset, shares);
      return {
        spotlight: null,
        glyphs: [getPixelBox(inputs, world, index)],
        readout: CLASS_NAMES.map((name, label) => `${name} ${formatNumber(shares[label])}`).join(
          ' · '
        )
      };
    }
  },
  draw: ({inputs, outputs, values, theme}) => [
    values.shown === 0
      ? {layer: 'raster', values: outputs.labels, columns: inputs.columns, rows: inputs.rows}
      : {
          layer: 'raster',
          values: outputs.probability,
          columns: inputs.columns,
          rows: inputs.rows,
          ramp: 'sequential',
          range: [0, 1]
        },
    ...getColoredBarLayers(outputs.inset, theme)
  ]
});

/** Multinomial logistic regression by gradient descent with a small L2 penalty: scikit-learn's `coef_`, `intercept_`. */
function trainLogisticRegression(vectors: Float32Array, labels: Uint32Array, count: number) {
  const weights = new Float32Array(CLASS_COUNT * DIMENSIONS);
  const bias = new Float32Array(CLASS_COUNT);
  const learningRate = 0.5;
  const penalty = 0.01;
  for (let iteration = 0; iteration < 400; iteration++) {
    const weightGradient = new Float32Array(weights.length);
    const biasGradient = new Float32Array(CLASS_COUNT);
    for (let sample = 0; sample < count; sample++) {
      const logits = Array.from({length: CLASS_COUNT}, (_, label) => {
        let sum = bias[label];
        for (let band = 0; band < DIMENSIONS; band++) {
          sum += weights[label * DIMENSIONS + band] * vectors[sample * DIMENSIONS + band];
        }
        return sum;
      });
      const largest = Math.max(...logits);
      const exponentials = logits.map(value => Math.exp(value - largest));
      const total = exponentials.reduce((sum, value) => sum + value, 0);
      for (let label = 0; label < CLASS_COUNT; label++) {
        const error = exponentials[label] / total - (labels[sample] === label ? 1 : 0);
        biasGradient[label] += error / count;
        for (let band = 0; band < DIMENSIONS; band++) {
          weightGradient[label * DIMENSIONS + band] +=
            (error * vectors[sample * DIMENSIONS + band]) / count;
        }
      }
    }
    for (let element = 0; element < weights.length; element++) {
      weights[element] -= learningRate * (weightGradient[element] + penalty * weights[element]);
    }
    for (let label = 0; label < CLASS_COUNT; label++)
      bias[label] -= learningRate * biasGradient[label];
  }
  return {weights, bias};
}
