// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUForestInference,
  readScikitLearnForestModel,
  readXGBoostForestModel,
  readLightGBMForestModel,
  getGPUForestInferenceParameterValues,
  type GPUForestInferenceModel
} from '../../../src/dataframe-analysis/forest-inference/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createVectorView,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import sklearnFixtures from './sklearn-fixtures.json';
import handwrittenFixtures from './handwritten-fixtures.json';
import {getNativeForestCases} from './native-model-cases';
import {getZeroLeafReplayExpectation} from './forest-replay-expectations.test-utils';
const {xgboostBinary, lightgbmBinary, tiedSklearn} = handwrittenFixtures;

type ExpectedRow = {prediction: number; scores: number[]; leaves: number[]; valid?: boolean};

it('preserves every output guard when empty forest input emits no commands', async () => {
  const result = await runForest(readScikitLearnForestModel(tiedSklearn), [], []);
  expect(result.predictions).toEqual([-777, -777]);
  expect(result.validity).toEqual([777, 777]);
  expect(result.scores).toEqual([-777, -777]);
  expect(result.leaves).toEqual([777, 777]);
});

/** Builds with nonzero offsets and sentinel guards around EVERY input and output. */
async function runForest(
  model: GPUForestInferenceModel,
  features: (number | null)[][],
  expected: ExpectedRow[],
  options: {
    raster?: boolean;
    featureMajor?: boolean;
    replay?: boolean;
    minimal?: boolean;
    chunkLengths?: readonly number[];
  } = {}
) {
  const testDevice = await getWebGPUTestDevice();
  requireTestDevice(testDevice);
  const device = testDevice;
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  function upload<Format extends 'float32' | 'uint32'>(
    values: Float32Array | Uint32Array,
    format: Format
  ): {buffer: Buffer; view: GraphDataView<Format>} {
    const guarded =
      format === 'float32'
        ? new Float32Array(values.length + 2).fill(-777)
        : new Uint32Array(values.length + 2).fill(777);
    guarded.set(values, 1);
    const buffer = device.createBuffer({
      data: guarded,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const imported = importGraphBuffer(graph, `buffer-${serial++}`, buffer, format);
    const view = graph.createDataView(imported.buffer, {
      format,
      byteOffset: 4,
      length: values.length
    }) as GraphDataView<Format>;
    return {buffer, view};
  }
  const rowCount = features.length;
  const parameters = upload(model.parameters, 'float32');
  const modelNodes = upload(model.nodes, 'uint32');
  const modelTrees = upload(model.trees, 'uint32');
  const modelLeaves = upload(model.leafValues, 'float32');
  const mask = upload(new Uint32Array(rowCount).fill(1), 'uint32');
  const predictions = upload(new Float32Array(rowCount).fill(-777), 'float32');
  const validity = upload(new Uint32Array(rowCount).fill(777), 'uint32');
  const scores = options.minimal
    ? undefined
    : upload(new Float32Array(rowCount * model.outputCount).fill(-777), 'float32');
  const leaves = options.minimal
    ? undefined
    : upload(new Uint32Array(rowCount * model.treeCount).fill(777), 'uint32');
  let rowBase = 0;
  const packed = (options.chunkLengths ?? [rowCount]).flatMap(length => {
    const batch = features.slice(rowBase, rowBase + length);
    rowBase += length;
    return options.featureMajor
      ? Array.from({length: model.featureCount}, (_, channel) =>
          batch.map(row => row[channel] ?? NaN)
        ).flat()
      : batch.flatMap(row => row.map(value => value ?? NaN));
  });
  const source = upload(Float32Array.from(packed), 'float32');
  const bands = Array.from({length: model.featureCount}, (_, channel) => ({
    id: `band-${channel}`,
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: upload(Float32Array.from(features.map(row => row[channel] ?? 12345)), 'float32').view
    },
    validity: upload(Uint32Array.from(features.map(row => Number(row[channel] !== null))), 'uint32')
      .view
  }));
  const split = <Format extends 'float32' | 'uint32'>(
    view: GraphDataView<Format>,
    width: number
  ) => {
    if (!options.chunkLengths) return view;
    let firstRow = 0;
    return createVectorView(
      view.buffer.id,
      view.format,
      options.chunkLengths.map(rows => {
        const chunk = graph.createDataView(view.buffer, {
          format: view.format,
          byteOffset: view.byteOffset + firstRow * width * 4,
          length: rows * width
        });
        firstRow += rows;
        return chunk;
      })
    );
  };
  const contributor = new GPUForestInference({
    model,
    rowCount,
    nodes: modelNodes.view,
    trees: modelTrees.view,
    leafValues: modelLeaves.view,
    parameters: parameters.view,
    ...(options.raster
      ? {
          rasterStack: {
            bands,
            grid: {
              width: rowCount,
              height: 1,
              affine: [2, 0, 37, 0, -2, 19],
              pixelInterpretation: 'area'
            }
          }
        }
      : {
          features: split(source.view, model.featureCount),
          featureLayout: options.featureMajor ? 'feature-major' : 'row-major'
        }),
    mask: split(mask.view, 1),
    output: {
      predictions: split(predictions.view, 1),
      validity: split(validity.view, 1),
      scores: scores ? split(scores.view, model.outputCount) : undefined,
      leafIndices: leaves ? split(leaves.view, model.treeCount) : undefined
    }
  });
  graph.add(contributor);
  const pipelineCreation = vi.spyOn(device, 'createComputePipeline');
  const compiled = graph.compile();
  const pipelineCount = pipelineCreation.mock.calls.length;
  const sourceReads = [source.buffer, modelNodes.buffer, modelTrees.buffer, modelLeaves.buffer].map(
    buffer => vi.spyOn(buffer, 'readAsync')
  );
  async function check(rows: ExpectedRow[], phase: string) {
    submitGraph(device, compiled, undefined);
    const actualPredictions = await readFloat32(predictions.buffer, rowCount + 2);
    const actualValidity = await readUint32(validity.buffer, rowCount + 2);
    const actualScores = scores
      ? await readFloat32(scores.buffer, rowCount * model.outputCount + 2)
      : undefined;
    const actualLeaves = leaves
      ? await readUint32(leaves.buffer, rowCount * model.treeCount + 2)
      : undefined;
    for (const values of [actualPredictions, actualScores]) {
      if (values) {
        expect(values[0]).toBe(-777);
        expect(values[values.length - 1]).toBe(-777);
      }
    }
    for (const values of [actualValidity, actualLeaves]) {
      if (values) {
        expect(values[0]).toBe(777);
        expect(values[values.length - 1]).toBe(777);
      }
    }
    rows.forEach((row, index) => {
      const valid = row.valid !== false;
      expect(actualValidity[index + 1], `validity row ${index}`).toBe(Number(valid));
      if (valid) expect(actualPredictions[index + 1]).toBeCloseTo(row.prediction, 4);
      else expect(Number.isNaN(actualPredictions[index + 1])).toBe(true);
      if (actualScores)
        row.scores.forEach((score, channel) => {
          const actual = actualScores[1 + index * model.outputCount + channel];
          if (valid)
            expect(
              Math.abs(actual - score),
              `${phase}, ${model.transform}, row ${index}, channel ${channel}: ${actual} vs ${score}`
            ).toBeLessThanOrEqual(2e-5 * Math.max(1, Math.abs(score)));
          else expect(Number.isNaN(actual)).toBe(true);
        });
      if (actualLeaves)
        expect(
          actualLeaves.slice(1 + index * model.treeCount, 1 + (index + 1) * model.treeCount)
        ).toEqual(valid ? row.leaves : new Array(model.treeCount).fill(0xffffffff));
    });
    return {
      predictions: actualPredictions,
      validity: actualValidity,
      scores: actualScores,
      leaves: actualLeaves
    };
  }
  try {
    const result = await check(expected, 'initial inference');
    if (options.replay) {
      // Rewrite controls and the mask in place; a masked padding row cannot retain old leaves/scores.
      const controls = getGPUForestInferenceParameterValues({
        baseScores: Array.from(model.parameters.slice(1)),
        sigmoid: model.parameters[0]
      });
      const guardedControls = new Float32Array(controls.length + 2).fill(-777);
      guardedControls.set(controls, 1);
      parameters.buffer.write(guardedControls);
      const changedMask = new Uint32Array(rowCount + 2).fill(1);
      changedMask[rowCount] = 0;
      mask.buffer.write(changedMask);
      await check(
        expected.map((row, index) => (index === rowCount - 1 ? {...row, valid: false} : row)),
        'changed mask replay'
      );
      // The same compiled traversal can consume changed caller-owned leaf weights and parameters.
      const changedLeaves = new Float32Array(model.leafValues.length + 2).fill(-777);
      changedLeaves.set(new Float32Array(model.leafValues.length), 1);
      modelLeaves.buffer.write(changedLeaves);
      const replayExpected = getZeroLeafReplayExpectation(model);
      guardedControls.set(
        getGPUForestInferenceParameterValues({baseScores: replayExpected.baseScores}),
        1
      );
      parameters.buffer.write(guardedControls);
      await check(
        expected.map((row, index) => ({
          ...row,
          valid: index === rowCount - 1 ? false : row.valid,
          prediction: replayExpected.prediction,
          scores: replayExpected.scores
        })),
        'zero leaves and changed base scores replay'
      );
    }
    expect(pipelineCreation.mock.calls.length).toBe(pipelineCount);
    for (const read of sourceReads) expect(read).not.toHaveBeenCalled();
    return result;
  } finally {
    for (const read of sourceReads) read.mockRestore();
    pipelineCreation.mockRestore();
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

// Includes independently trained regressor/classifier missing-only splits with Infinity thresholds.
it.each(
  sklearnFixtures.cases
)('GPUForestInference matches independent sklearn predictions and leaves: $name', async fixture => {
  await runForest(
    readScikitLearnForestModel(fixture.model),
    fixture.features,
    fixture.predictions.map((prediction, row) => ({
      prediction,
      scores: fixture.scores[row],
      leaves: fixture.leafIndices[row]
    }))
  );
});

it('matches a calibrated raster stack with band validity and feature-major rows', async () => {
  const fixture = sklearnFixtures.cases[0];
  const expected = fixture.predictions.map((prediction, row) => ({
    prediction,
    scores: fixture.scores[row],
    leaves: fixture.leafIndices[row]
  }));
  await runForest(readScikitLearnForestModel(fixture.model), fixture.features, expected, {
    raster: true
  });
  await runForest(readScikitLearnForestModel(fixture.model), fixture.features, expected, {
    featureMajor: true
  });
});

it('honours native XGBoost strict splits, missing routes and base margin', async () => {
  const low = 1 / (1 + 3 * Math.exp(2));
  const high = 1 / (1 + 3 * Math.exp(-2));
  await runForest(
    readXGBoostForestModel(xgboostBinary),
    [[0], [1], [2], [null]],
    [low, high, high, low].map((probability, index) => ({
      prediction: probability > 0.5 ? 1 : 0,
      scores: [1 - probability, probability],
      leaves: [index === 0 || index === 3 ? 1 : 2]
    }))
  );
});

it('honours LightGBM double thresholds, preweighted leaves, link multiplier and original leaf ids', async () => {
  const low = 1 / (1 + Math.exp(2));
  const high = 1 - low;
  await runForest(
    readLightGBMForestModel(lightgbmBinary),
    [[1], [1.0000001192092896], [null]],
    [low, high, high].map((probability, index) => ({
      prediction: index === 0 ? 0 : 1,
      scores: [1 - probability, probability],
      leaves: [index === 0 ? 0 : 1]
    })),
    {raster: true}
  );
});

it('rewrites padding rows and replays masks, parameters and weights without pipeline creation', async () => {
  await runForest(
    readScikitLearnForestModel(tiedSklearn),
    [[0], [1], [3]],
    Array.from({length: 3}, () => ({prediction: 0, scores: [0.5, 0.5], leaves: [0, 0]})),
    {replay: true}
  );
});

it('invalidates unsupported missing values and clears all leaf slots after a failed tree', async () => {
  const fixture = sklearnFixtures.cases[2];
  const model = readScikitLearnForestModel(fixture.model);
  await runForest(
    model,
    [[null, null]],
    [{prediction: 0, scores: [0], leaves: new Array(model.treeCount).fill(0), valid: false}]
  );
  // A caller overwriting a root with a cycle must finish at the bound and invalidate every leaf slot.
  const secondRoot = model.trees[4];
  model.nodes[secondRoot * 6 + 1] = secondRoot;
  model.nodes[secondRoot * 6 + 2] = secondRoot;
  await runForest(
    model,
    [[0, 0]],
    [{prediction: 0, scores: [0], leaves: new Array(model.treeCount).fill(0), valid: false}]
  );
});

it('supports empty row input and minimal optional outputs without touching guards', async () => {
  await runForest(readScikitLearnForestModel(tiedSklearn), [], [], {minimal: true});
  await runForest(
    readScikitLearnForestModel(tiedSklearn),
    [[0]],
    [{prediction: 0, scores: [0.5, 0.5], leaves: [0, 0]}],
    {minimal: true}
  );
});

// Includes one-class multi:softprob and multi:softmax: both must publish class 0, score 1 and exact leaves.
it.each(
  getNativeForestCases()
)('matches native JSON hand-derived predictions: $name', async fixture => {
  await runForest(
    fixture.model,
    fixture.features,
    fixture.predictions.map((prediction, row) => ({
      prediction,
      scores: fixture.scores[row],
      leaves: fixture.leaves[row]
    }))
  );
});

it('supports a one-class random forest', async () => {
  const model = readScikitLearnForestModel({
    ...tiedSklearn,
    classes: ['only'],
    trees: [{...tiedSklearn.trees[0], value: [[4]]}]
  });
  await runForest(model, [[0]], [{prediction: 0, scores: [1], leaves: [0]}]);
});

it('invalidates a visited infinity and a caller-shortened traversal bound', async () => {
  const model = readXGBoostForestModel(xgboostBinary);
  const invalid = {prediction: 0, scores: [0, 0], leaves: [0], valid: false};
  await runForest(model, [[Infinity], [-Infinity]], [invalid, invalid]);
  model.maximumDepth = 0;
  await runForest(model, [[0]], [invalid]);
});

for (const mode of ['row-major', 'feature-major', 'raster'] as const) {
  for (const fixture of sklearnFixtures.cases.filter(item => item.features.length >= 7)) {
    it(`GPUForestInference ${mode} ${fixture.name} matches packed, uneven chunks and linked replay expectations`, async () => {
      const model = readScikitLearnForestModel(fixture.model);
      const expected = fixture.predictions.map((prediction, row) => ({
        prediction,
        scores: fixture.scores[row],
        leaves: fixture.leafIndices[row]
      }));
      const options = {
        raster: mode === 'raster',
        featureMajor: mode === 'feature-major',
        replay: true
      };
      const packed = await runForest(model, fixture.features, expected, options);
      const chunked = await runForest(model, fixture.features, expected, {
        ...options,
        chunkLengths: [1, 3, fixture.features.length - 4]
      });
      expect(chunked).toEqual(packed);
    });
  }
}
