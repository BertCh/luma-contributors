// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUForestInference,
  getGPUForestInferenceParameterValues,
  readScikitLearnForestModel
} from '../../../src/dataframe-analysis/forest-inference/index';
import {getZeroLeafReplayExpectation} from './forest-replay-expectations.test-utils';
import {evaluateImportedForest} from './model-reference';
import sklearnFixtures from './sklearn-fixtures.json';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {expectValidShaderIdentifiers} from '../calendar-buckets/chunk-shader-validation.test-utils';

it.each(
  sklearnFixtures.cases
)('zero-leaf replay oracle preserves the model link: $name', fixture => {
  const model = readScikitLearnForestModel(fixture.model);
  const expected = getZeroLeafReplayExpectation(model);
  const replayModel = {
    ...model,
    leafValues: new Float32Array(model.leafValues.length),
    parameters: getGPUForestInferenceParameterValues({baseScores: expected.baseScores})
  };
  const initialRows = evaluateImportedForest(model, fixture.features);
  const replayRows = evaluateImportedForest(replayModel, fixture.features);
  replayRows.forEach((row, index) => {
    expect(row.prediction).toBe(expected.prediction);
    row.scores.forEach((score, channel) => expect(score).toBeCloseTo(expected.scores[channel], 12));
    expect(row.leaves).toEqual(initialRows[index].leaves);
  });
  if (model.transform === 'sigmoid') {
    expect(expected.scores).toEqual([0.5, 0.5]);
    // The previous headless expectation used raw [1, 0], causing exactly the reported 0.5 error.
    expect(Math.abs(replayRows[0].scores[0] - expected.baseScores[0])).toBe(0.5);
  } else if (model.transform === 'softmax') {
    expect(expected.scores[0]).toBeCloseTo(1 / (1 + (model.outputCount - 1) / Math.E), 12);
    expect(expected.scores.reduce((sum, score) => sum + score, 0)).toBeCloseTo(1, 12);
    expect(expected.scores).not.toEqual(expected.baseScores);
  } else {
    expect(expected.scores).toEqual(expected.baseScores);
  }
});

it.each(
  sklearnFixtures.cases
)('keeps linked shader sources constant across chunks: $name', fixture => {
  const model = readScikitLearnForestModel(fixture.model);
  const sources = new Set<string>();
  const pipelines = new Set<string>();
  for (const lengths of [[771], [513, 1, 257]]) {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const packed = <Format extends 'float32' | 'uint32'>(
        name: string,
        format: Format,
        length: number
      ) => createTransientView(graph, name, format, length);
      const column = <Format extends 'float32' | 'uint32'>(
        name: string,
        format: Format,
        width = 1
      ) =>
        createVectorView(
          name,
          format,
          lengths.map((length, index) => {
            const backing = packed(`${name}-${index}`, format, length * width + 64);
            return graph.createDataView(backing.buffer, {
              format,
              byteOffset: (index + 1) * 4,
              length: length * width
            });
          })
        );
      const contributor = new GPUForestInference({
        model,
        rowCount: 771,
        features: column('features', 'float32', model.featureCount),
        nodes: packed('nodes', 'uint32', model.nodes.length),
        trees: packed('trees', 'uint32', model.trees.length),
        leafValues: packed('model-leaves', 'float32', model.leafValues.length),
        parameters: packed('parameters', 'float32', model.parameters.length),
        output: {
          predictions: column('predictions', 'float32'),
          validity: column('validity', 'uint32'),
          scores: column('scores', 'float32', model.outputCount),
          leafIndices: column('leaf-indices', 'uint32', model.treeCount)
        }
      });
      const nodes = contributor.getCommandNodes(graph);
      getParsedKernelSources(nodes);
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) sources.add(kernel.source);
      for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
        pipelines.add(key);
    } finally {
      device.destroy();
    }
  }
  expect(sources.size).toBe(2);
  expect(pipelines.size).toBe(2);
  expectValidShaderIdentifiers(sources);
});
