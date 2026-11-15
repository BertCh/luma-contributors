// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUForestInferenceModel} from '../../../src/dataframe-analysis/forest-inference/index';

/** Test-only interpreter checks importer semantics against committed sklearn predictions, never used by GPU specs. */
export function evaluateImportedForest(
  model: GPUForestInferenceModel,
  features: (number | null)[][]
) {
  const floatingNodes = new Float32Array(model.nodes.buffer);
  const floatingTrees = new Float32Array(model.trees.buffer);
  return features.map(row => {
    const scores = Array.from(model.parameters.slice(1));
    const leaves: number[] = [];
    for (let tree = 0; tree < model.treeCount; tree++) {
      let node = model.trees[tree * 4];
      for (let depth = 0; depth <= model.maximumDepth; depth++) {
        const feature = model.nodes[node * 6];
        if (feature === 0xffffffff) {
          leaves.push(model.nodes[node * 6 + 5]);
          scores.forEach((_, channel) => {
            scores[channel] +=
              floatingTrees[tree * 4 + 2] * model.leafValues[node * model.outputCount + channel];
          });
          break;
        }
        const flags = model.nodes[node * 6 + 4];
        let value = row[feature] === null ? NaN : Math.fround(row[feature]!);
        if (Number.isNaN(value) && flags & 8) value = 0;
        const missing = Number.isNaN(value) || Boolean(flags & 4 && Math.abs(value) <= 1e-35);
        if (missing && !(flags & 2)) throw new Error('test model has no missing branch');
        const left = missing
          ? Boolean(flags & 1)
          : flags & 16
            ? value < floatingNodes[node * 6 + 3]
            : value <= floatingNodes[node * 6 + 3];
        node = model.nodes[node * 6 + (left ? 1 : 2)];
      }
    }
    if (model.transform === 'sigmoid') {
      scores[1] = 1 / (1 + Math.exp(-scores[1] * model.parameters[0]));
      scores[0] = 1 - scores[1];
    } else if (model.transform === 'softmax') {
      const maximum = Math.max(...scores);
      const exponentials = scores.map(score => Math.exp(score - maximum));
      const total = exponentials.reduce((sum, score) => sum + score, 0);
      exponentials.forEach((score, index) => {
        scores[index] = score / total;
      });
    }
    const prediction =
      model.task === 'regression' ? scores[0] : scores.indexOf(Math.max(...scores));
    return {scores, prediction, leaves};
  });
}
