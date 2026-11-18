// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUForestInference,
  readScikitLearnForestModel,
  readXGBoostForestModel,
  readLightGBMForestModel,
  getGPUForestInferenceParameterValues,
  type GPUForestInferenceProps
} from '../../../src/dataframe-analysis/forest-inference/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../../src/raster-analysis/raster-grid/index';
import sklearnFixtures from './sklearn-fixtures.json';
import handwrittenFixtures from './handwritten-fixtures.json';
const {xgboostBinary, lightgbmBinary, tiedSklearn} = handwrittenFixtures;
import {evaluateImportedForest} from './model-reference';
import {getNativeForestCases, getOneClassXGBoostModelJSON} from './native-model-cases';

it.each(sklearnFixtures.cases)('imports sklearn oracle: $name', fixture => {
  const model = readScikitLearnForestModel(fixture.model);
  const results = evaluateImportedForest(model, fixture.features);
  results.forEach((result, row) => {
    expect(result.prediction).toBeCloseTo(fixture.predictions[row], 5);
    result.scores.forEach((score, channel) =>
      expect(score).toBeCloseTo(fixture.scores[row][channel], 5)
    );
    expect(result.leaves).toEqual(fixture.leafIndices[row]);
  });
  expect(model.maximumDepth).toBeLessThanOrEqual(4);
  expect(model.nodes.length).toBe(model.nodeCount * 6);
});

it('imports native XGBoost strict splits, default direction and logistic base score', () => {
  const model = readXGBoostForestModel(xgboostBinary);
  const results = evaluateImportedForest(model, [[0], [1], [2], [null]]);
  expect(results.map(result => result.leaves)).toEqual([[1], [2], [2], [1]]);
  [
    1 / (1 + 3 * Math.exp(2)),
    1 / (1 + 3 * Math.exp(-2)),
    1 / (1 + 3 * Math.exp(-2)),
    1 / (1 + 3 * Math.exp(2))
  ].forEach((probability, row) => expect(results[row].scores[1]).toBeCloseTo(probability, 7));
});

it('imports LightGBM shrinkage once, float64 thresholds and original leaf indices', () => {
  const model = readLightGBMForestModel(lightgbmBinary);
  const results = evaluateImportedForest(model, [[1], [1.0000001192092896], [null]]);
  expect(results.map(result => result.leaves)).toEqual([[0], [1], [1]]);
  expect(results[0].scores[1]).toBeCloseTo(1 / (1 + Math.exp(2)), 7);
  expect(results[1].scores[1]).toBeCloseTo(1 / (1 + Math.exp(-2)), 7);
  const zeroMissing = structuredClone(lightgbmBinary);
  zeroMissing.tree_info[0].tree_structure.missing_type = 'Zero';
  const zeroResults = evaluateImportedForest(readLightGBMForestModel(zeroMissing), [
    [0],
    [null],
    [1e-36],
    [-1]
  ]);
  expect(zeroResults.map(result => result.leaves)).toEqual([[1], [1], [1], [0]]);
  zeroMissing.tree_info[0].tree_structure.missing_type = 'None';
  expect(evaluateImportedForest(readLightGBMForestModel(zeroMissing), [[null]])[0].leaves).toEqual([
    0
  ]);
});

it('averages leaf probabilities and chooses the first class on a tie', () => {
  const results = evaluateImportedForest(readScikitLearnForestModel(tiedSklearn), [[0]]);
  expect(results[0]).toEqual({scores: [0.5, 0.5], prediction: 0, leaves: [0, 0]});
});

it('preserves sklearn <= routing when a double threshold rounds upward to a float32 feature', () => {
  const json = {
    ...tiedSklearn,
    estimator: 'RandomForestRegressor',
    trees: [
      {
        children_left: [1, -1, -1],
        children_right: [2, -1, -1],
        feature: [0, -2, -2],
        threshold: [0.99999999, -2, -2],
        value: [[0], [-3], [7]]
      }
    ]
  };
  const model = readScikitLearnForestModel(json);
  expect(
    evaluateImportedForest(model, [[1], [0.9999999403953552]]).map(result => result.prediction)
  ).toEqual([7, -3]);
});

it('rejects unsupported formats and malformed model topology by name', () => {
  const wrong = structuredClone(xgboostBinary);
  wrong.learner.gradient_booster.name = 'dart';
  expect(() => readXGBoostForestModel(wrong)).toThrow(/GPUForestInference.*gbtree/);
  wrong.learner.gradient_booster.name = 'gbtree';
  wrong.learner.gradient_booster.model.trees[0].split_type[0] = 1;
  expect(() => readXGBoostForestModel(wrong)).toThrow(/GPUForestInference.*categorical/);
  const categorical = structuredClone(lightgbmBinary);
  categorical.tree_info[0].tree_structure.decision_type = '==';
  expect(() => readLightGBMForestModel(categorical)).toThrow(/GPUForestInference.*decision_type/);
  const invalidTree = {
    children_left: [0],
    children_right: [0],
    feature: [0],
    threshold: [0],
    value: [[0]]
  };
  expect(() => readScikitLearnForestModel({...tiedSklearn, trees: [invalidTree]})).toThrow(
    /GPUForestInference.*cycles/
  );
  expect(() => readScikitLearnForestModel({...tiedSklearn, trees: []})).toThrow(
    /GPUForestInference.*treeCount/
  );
  expect(() => readScikitLearnForestModel({...tiedSklearn, n_features_in: 257})).toThrow(
    /GPUForestInference.*featureCount/
  );
  expect(() => readScikitLearnForestModel({...tiedSklearn, version: 2})).toThrow(
    /GPUForestInference.*version 1/
  );
  expect(() =>
    readScikitLearnForestModel({...tiedSklearn, classes: ['duplicate', 'duplicate']})
  ).toThrow(/GPUForestInference.*classes/);
  expect(() => getGPUForestInferenceParameterValues({baseScores: [0], sigmoid: 0})).toThrow(
    /GPUForestInference.*sigmoid/
  );
  expect(() => getGPUForestInferenceParameterValues({baseScores: [Infinity]})).toThrow(
    /GPUForestInference.*baseScores/
  );
});

function fixtureProps() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const model = readScikitLearnForestModel(tiedSklearn);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const props: GPUForestInferenceProps = {
    model,
    rowCount: 3,
    features: view('float32', 3),
    nodes: view('uint32', model.nodes.length),
    trees: view('uint32', model.trees.length),
    leafValues: view('float32', model.leafValues.length),
    parameters: view('float32', model.parameters.length),
    output: {
      predictions: view('float32', 3),
      validity: view('uint32', 3),
      scores: view('float32', 6),
      leafIndices: view('uint32', 6)
    }
  };
  return {device, graph, props, view};
}

describe('construction and binding limits', () => {
  it('rejects JavaScript featureLayout on raster inputs and accepts a raster-only control', () => {
    const {device, props, view} = fixtureProps();
    try {
      const {features: _features, featureLayout: _featureLayout, ...shared} = props;
      const grid = makeGPURasterMetadata({width: 1, height: 1});
      const rasterProps = {
        ...shared,
        rowCount: 1,
        rasterStack: {bands: [makeGPURasterBand(view('float32', 1), grid)], grid},
        output: {predictions: view('float32', 1), validity: view('uint32', 1)}
      };
      expect(() => new GPUForestInference(rasterProps)).not.toThrow();
      expect(() =>
        Reflect.construct(GPUForestInference, [{...rasterProps, featureLayout: 'row-major'}])
      ).toThrow(/featureLayout.*features/);
    } finally {
      device.destroy();
    }
  });

  it('validates shape, mask, layouts, output lengths and output aliasing once', () => {
    const {device, props, view} = fixtureProps();
    try {
      const build = (updates: Partial<GPUForestInferenceProps>) => () =>
        // Deliberately invalid JavaScript inputs exercise runtime validation; the public union
        // is checked independently by core-api-exceptions.node.spec.ts.
        new GPUForestInference({...props, ...updates} as GPUForestInferenceProps);
      expect(build({model: {...props.model, maximumDepth: 65}})).toThrow(
        /GPUForestInference.*maximumDepth.*64/
      );
      expect(build({model: {...props.model, treeCount: 1025}})).toThrow(
        /GPUForestInference.*treeCount.*1024/
      );
      expect(build({rowCount: -1})).toThrow(/GPUForestInference.*rowCount/);
      expect(build({mask: view('uint32', 2)})).toThrow(/GPUForestInference.*mask.*3/);
      expect(build({features: undefined})).toThrow(/GPUForestInference.*exactly one/);
      expect(build({output: {...props.output, predictions: props.features!}})).toThrow(
        /GPUForestInference/
      );
      expect(build({output: {...props.output, predictions: view('float32', 2)}})).toThrow(
        /GPUForestInference.*predictions.*3/
      );
      expect(
        build({rowCount: 16777216, model: {...props.model, treeCount: 1024, maximumDepth: 64}})
      ).toThrow(/GPUForestInference.*work/);
    } finally {
      device.destroy();
    }
  });

  it('rejects a storage binding before creating any nodes', () => {
    const {device, graph, props} = fixtureProps();
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 16}
    });
    try {
      expect(() => new GPUForestInference(props).getCommandNodes(graph)).toThrow(
        /GPUForestInference.*binding.*limit 16/
      );
      expect((graph as unknown as {nodes: unknown[]}).nodes).toHaveLength(0);
    } finally {
      device.destroy();
    }
  });

  it('supports empty rows without changing the command-node estimate', () => {
    const {device, graph, props, view} = fixtureProps();
    try {
      const contributor = new GPUForestInference({
        ...props,
        rowCount: 0,
        features: view('float32', 0),
        output: {predictions: view('float32', 0), validity: view('uint32', 0)}
      });
      expect(contributor.getCommandNodes(graph).length).toBe(contributor.estimatedCommandNodeCount);
    } finally {
      device.destroy();
    }
  });
});

it.each(getNativeForestCases())('imports hand-derived native JSON: $name', fixture => {
  const results = evaluateImportedForest(fixture.model, fixture.features);
  results.forEach((result, row) => {
    expect(result.prediction).toBe(fixture.predictions[row]);
    result.scores.forEach((score, channel) =>
      expect(score).toBeCloseTo(fixture.scores[row][channel], 6)
    );
    expect(result.leaves).toEqual(fixture.leaves[row]);
  });
});

it('supports a one-class forest and probability averaging that differs from majority voting', () => {
  const single = readScikitLearnForestModel({
    ...tiedSklearn,
    classes: ['only'],
    trees: [{...tiedSklearn.trees[0], value: [[4]]}]
  });
  expect(evaluateImportedForest(single, [[0]])[0].scores).toEqual([1]);
  const votingDifference = readScikitLearnForestModel({
    ...tiedSklearn,
    trees: [
      {...tiedSklearn.trees[0], value: [[49, 51]]},
      {...tiedSklearn.trees[0], value: [[49, 51]]},
      {...tiedSklearn.trees[0], value: [[100, 0]]}
    ]
  });
  expect(evaluateImportedForest(votingDifference, [[0]])[0].prediction).toBe(0);
});

it('rejects unsupported links, linear leaves and inconsistent model arrays', () => {
  expect(() => readLightGBMForestModel({...lightgbmBinary, objective: 'regression sqrt'})).toThrow(
    /GPUForestInference.*sqrt/
  );
  expect(() => readLightGBMForestModel({...lightgbmBinary, average_output: true})).toThrow(
    /GPUForestInference.*average_output/
  );
  expect(() =>
    readLightGBMForestModel({
      ...lightgbmBinary,
      tree_info: [{...lightgbmBinary.tree_info[0], is_linear: true}]
    })
  ).toThrow(/GPUForestInference.*is_linear/);
  const invalid = structuredClone(xgboostBinary);
  invalid.learner.gradient_booster.model.trees[0].left_children.pop();
  expect(() => readXGBoostForestModel(invalid)).toThrow(/GPUForestInference.*array lengths/);
  invalid.learner.learner_model_param.base_score = '[1]';
  expect(() => readXGBoostForestModel(invalid)).toThrow(/GPUForestInference.*base_score/);
});

it('rejects malformed XGBoost base-score JSON with a named diagnostic', () => {
  const invalid = structuredClone(xgboostBinary);
  invalid.learner.learner_model_param.base_score = '[broken';
  expect(() => readXGBoostForestModel(invalid)).toThrow(/GPUForestInference.*base_score/);
});

it.each([
  'multi:softprob',
  'multi:softmax'
] as const)('keeps one-class XGBoost %s classification metadata and softmax link', objective => {
  const model = readXGBoostForestModel(getOneClassXGBoostModelJSON(objective));
  expect(model.task).toBe('classification');
  expect(model.transform).toBe('softmax');
  expect(model.classes).toEqual([0]);
  expect(model.outputCount).toBe(1);
  expect(evaluateImportedForest(model, [[0], [1]])).toEqual([
    {prediction: 0, scores: [1], leaves: [1]},
    {prediction: 0, scores: [1], leaves: [2]}
  ]);
});

it.each(['Infinity', Infinity])('imports sklearn missing-only threshold %s', threshold => {
  const model = readScikitLearnForestModel({
    format: 'sklearn-forest',
    version: 1,
    estimator: 'RandomForestRegressor',
    n_features_in: 1,
    trees: [
      {
        children_left: [1, -1, -1],
        children_right: [2, -1, -1],
        feature: [0, -2, -2],
        threshold: [threshold, -2, -2],
        missing_go_to_left: [0, 0, 0],
        value: [[1 / 3], [0], [1]]
      }
    ]
  });
  expect(new Float32Array(model.nodes.buffer)[3]).toBe(Infinity);
  expect(evaluateImportedForest(model, [[0], [null]])).toEqual([
    {prediction: 0, scores: [0], leaves: [1]},
    {prediction: 1, scores: [1], leaves: [2]}
  ]);
});

it.each([
  '1000000000',
  '-1',
  '1.5',
  '4294967296'
])('rejects XGBoost outputCount %s before any dependent allocation', classCount => {
  const json = getOneClassXGBoostModelJSON('multi:softprob');
  json.learner.learner_model_param.num_class = classCount;
  const outputCount = Number(classCount);
  const attemptedLengths: number[] = [];
  const originalArray = globalThis.Array;
  let caught: unknown;
  // Intercept the dangerous constructor: this regression never allocates a billion elements.
  globalThis.Array = new Proxy(originalArray, {
    construct(target, argumentsList, newTarget) {
      if (argumentsList.length === 1 && argumentsList[0] === outputCount) {
        attemptedLengths.push(outputCount);
        throw new Error('model-dependent allocation reached before outputCount validation');
      }
      return Reflect.construct(target, argumentsList, newTarget);
    }
  });
  try {
    readXGBoostForestModel(json);
  } catch (error) {
    caught = error;
  } finally {
    globalThis.Array = originalArray;
  }
  expect(attemptedLengths).toEqual([]);
  expect(caught).toBeInstanceOf(Error);
  expect((caught as Error).message).toMatch(/GPUForestInference outputCount.*\[1, 32\]/);
});

it.each([
  NaN,
  null,
  'NaN'
])('rejects malformed sklearn threshold %s instead of changing the split', threshold => {
  const json = {...tiedSklearn, trees: [{...tiedSklearn.trees[0], threshold: [threshold]}]};
  expect(() => readScikitLearnForestModel(json)).toThrow(/GPUForestInference.*threshold/);
});

it.each([
  '-Infinity',
  -Infinity
])('preserves sklearn threshold %s and its missing branch', threshold => {
  const model = readScikitLearnForestModel({
    format: 'sklearn-forest',
    version: 1,
    estimator: 'RandomForestRegressor',
    n_features_in: 1,
    trees: [
      {
        children_left: [1, -1, -1],
        children_right: [2, -1, -1],
        feature: [0, -2, -2],
        threshold: [threshold, -2, -2],
        missing_go_to_left: [1, 0, 0],
        value: [[0], [5], [9]]
      }
    ]
  });
  expect(new Float32Array(model.nodes.buffer)[3]).toBe(-Infinity);
  expect(evaluateImportedForest(model, [[0], [null]])).toEqual([
    {prediction: 9, scores: [9], leaves: [2]},
    {prediction: 5, scores: [5], leaves: [1]}
  ]);
});
