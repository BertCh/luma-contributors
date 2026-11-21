// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  type GPUParameterValues,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUGeographicallyWeightedRegression,
  type GPUGeographicallyWeightedRegressionProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {GPUGeographicallyWeightedRegressionNonstationarityTest} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression-nonstationarity-test';
import {getGWRLargeSampleInferenceWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-large-sample-inference';
import {createGWRVarianceReductionNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-global-reductions';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {
  getGPUGeographicallyWeightedRegressionParameterValues,
  type GPUGeographicallyWeightedRegressionSettings
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {computeGeographicallyWeightedRegressionOnCPU} from './geographically-weighted-regression-oracle';
import {
  getNonstationarityAssignment,
  getLocalEstimateStandardDeviations,
  type NonstationarityScene
} from './geographically-weighted-regression-nonstationarity-oracle';
import {
  createBoundaryReference,
  LARGE_SAMPLE_QUANTILE_CASES,
  REVIEW_BANDWIDTHS,
  REVIEW_FOCUS_ROWS
} from './w2-gwr-review-reference';
import {
  simulateGoldenSectionSearch,
  getGPUGoldenSectionTolerance
} from './gwr-golden-section-reference';

const SEED = 473;
type Readback = Record<
  | 'coefficients'
  | 'standardErrors'
  | 'tValues'
  | 'significant'
  | 'localR2'
  | 'fitted'
  | 'residuals'
  | 'hatDiagonal'
  | 'localStatus'
  | 'localConditionNumber'
  | 'selectedBandwidth'
  | 'summary'
  | 'bandwidthScores'
  | 'bandwidthSingularRowCounts'
  | 'candidateTable'
  | 'approximation'
  | 'table'
  | 'standardDeviations',
  Float32Array | Uint32Array
>;
type Options = {
  settings: GPUGeographicallyWeightedRegressionSettings;
  indexGridSize?: readonly [number, number] | false;
  maximumPermutations?: number;
  permutations?: readonly number[];
  mask?: Uint32Array;
  excludeLastEncoding?: boolean;
  refinement?: number | false;
  bandwidthSearch?: 'ladder' | 'golden-section';
  maximumSearchIterations?: number;
  gaussianCutoff?: number;
  gaussianMode?: 'exact' | 'truncated';
  nonstationarity?: boolean;
  settingsByEncoding?: readonly GPUGeographicallyWeightedRegressionSettings[];
};

/** One compilation, optionally several encodings, with every public numeric distribution output. */
async function runScene(device: Device, scene: NonstationarityScene, options: Options) {
  const rowCount = scene.response.length;
  const coefficientCount = scene.predictorCount + 1;
  const maximumPermutations = options.maximumPermutations ?? 1;
  const maximumBandwidthCount = Math.max(2, options.settings.bandwidths.length);
  const searchRows =
    options.bandwidthSearch === 'golden-section'
      ? (options.maximumSearchIterations ?? 8) + 1
      : (1 + (options.refinement || 0)) * maximumBandwidthCount;
  const buffers: Buffer[] = [];
  const parameters: {destroy: () => void}[] = [];
  const graph = new GPUCommandGraph(device);
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format,
    length: number
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const output = (name: string, length: number, format: 'float32' | 'uint32' = 'float32') => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length), length, format};
  };
  const parameter = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    values: GPUParameterValues<Format>
  ) => {
    const buffer = new GPUParameterBuffer(device, {id: name, format, length: values.length});
    buffer.write(values);
    parameters.push(buffer);
    return buffer;
  };
  const bandwidthParameters = parameter(
    'bandwidth-parameters',
    'float32',
    getGPUGeographicallyWeightedRegressionParameterValues(options.settings, maximumBandwidthCount)
  );
  const permutationParameters = parameter(
    'permutation-parameters',
    'uint32',
    getGPUPermutationParameterValues({seed: SEED, permutationCount: maximumPermutations})
  );
  const rowMask = parameter(
    'row-mask',
    'uint32',
    options.mask ?? new Uint32Array(rowCount).fill(1)
  );
  const float = (name: string, length: number) => {
    const result = output(name, length);
    return {...result, view: graph.createDataView(result.view.buffer, {format: 'float32', length})};
  };
  const word = (name: string, length: number) => {
    const result = output(name, length, 'uint32');
    return {...result, view: graph.createDataView(result.view.buffer, {format: 'uint32', length})};
  };
  const outputs = {
    coefficients: float('coefficients', rowCount * coefficientCount),
    standardErrors: float('standardErrors', rowCount * coefficientCount),
    tValues: float('tValues', rowCount * coefficientCount),
    significant: word('significant', rowCount * coefficientCount),
    localR2: float('localR2', rowCount),
    fitted: float('fitted', rowCount),
    residuals: float('residuals', rowCount),
    hatDiagonal: float('hatDiagonal', rowCount),
    localStatus: word('localStatus', rowCount),
    localConditionNumber: float('localConditionNumber', rowCount),
    selectedBandwidth: float('selectedBandwidth', 2),
    summary: float('summary', 12),
    bandwidthScores: float('bandwidthScores', maximumBandwidthCount),
    bandwidthSingularRowCounts: word('bandwidthSingularRowCounts', maximumBandwidthCount),
    candidateTable: float('candidateTable', searchRows * 7),
    approximation: word('approximation', 1)
  };
  const testOutputs = {
    table: float('test-table', coefficientCount * 6),
    summary: float('test-summary', 4),
    standardDeviations: float('standardDeviations', (maximumPermutations + 1) * coefficientCount),
    approximation: word('test-approximation', 1)
  };
  const bandwidthParameterView = bandwidthParameters.importToGraph(graph);
  const shared = {
    positions: input('positions', scene.positions, 'float32x2', rowCount),
    predictors: input('predictors', scene.predictors, 'float32', rowCount * scene.predictorCount),
    response: input('response', scene.response, 'float32', rowCount),
    predictorCount: scene.predictorCount,
    mask: rowMask.importToGraph(graph),
    indexGridSize: options.indexGridSize,
    maximumBandwidthCount,
    gaussianCutoff: options.gaussianCutoff,
    gaussianMode: options.gaussianMode
  };
  const regression = new GPUGeographicallyWeightedRegression({
    ...shared,
    parameters: bandwidthParameterView,
    refinement: options.refinement ?? false,
    bandwidthSearch: options.bandwidthSearch,
    maximumSearchIterations: options.maximumSearchIterations,
    output: {
      ...Object.fromEntries(
        Object.entries(outputs)
          .filter(([name]) => name !== 'approximation')
          .map(([name, result]) => [name, result.view])
      ),
      status: {approximation: outputs.approximation.view}
    } as GPUGeographicallyWeightedRegressionProps['output']
  });
  graph.add(regression);
  const test = new GPUGeographicallyWeightedRegressionNonstationarityTest({
    ...shared,
    bandwidthParameters: bandwidthParameterView,
    selectedBandwidth: outputs.selectedBandwidth.view,
    coefficients: outputs.coefficients.view,
    parameters: permutationParameters.importToGraph(graph),
    maximumPermutations,
    tileRowCount: 256,
    output: {
      table: testOutputs.table.view,
      summary: testOutputs.summary.view,
      standardDeviations: testOutputs.standardDeviations.view,
      status: {approximation: testOutputs.approximation.view}
    }
  });
  if (options.nonstationarity !== false) graph.add(test);
  const compiled = graph.compile();
  try {
    const results = [];
    for (const [encoding, permutationCount] of (
      options.permutations ?? [maximumPermutations]
    ).entries()) {
      const nextSettings = options.settingsByEncoding?.[encoding];
      if (nextSettings)
        bandwidthParameters.write(
          getGPUGeographicallyWeightedRegressionParameterValues(nextSettings, maximumBandwidthCount)
        );
      permutationParameters.write(getGPUPermutationParameterValues({seed: SEED, permutationCount}));
      if (options.excludeLastEncoding && encoding === (options.permutations?.length ?? 1) - 1)
        rowMask.write(new Uint32Array(rowCount));
      submitGraph(device, compiled, undefined);
      const read = async (values: typeof outputs | typeof testOutputs) =>
        Object.fromEntries(
          await Promise.all(
            Object.entries(values).map(async ([name, result]) => [
              name,
              result.format === 'uint32'
                ? new Uint32Array(await readUint32(result.buffer, result.length))
                : new Float32Array(await readFloat32(result.buffer, result.length))
            ])
          )
        ) as Readback;
      results.push({
        output: await read(outputs),
        test: options.nonstationarity === false ? ({} as Readback) : await read(testOutputs)
      });
    }
    return {
      results,
      indexGridSize: regression.indexGridSize,
      testIndexGridSize: test.indexGridSize
    };
  } finally {
    compiled.destroy();
    for (const buffer of parameters) buffer.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

function expectNumber(
  actual: number,
  expected: number,
  absolute = 1e-4,
  relative = 1e-5,
  message?: string
) {
  if (Number.isNaN(expected)) expect(Number.isNaN(actual)).toBe(true);
  else {
    expect(Number.isFinite(actual)).toBe(true);
    expect(Math.abs(actual - expected), message).toBeLessThanOrEqual(
      absolute + relative * Math.abs(expected)
    );
  }
}
function expectArray(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  absolute = 1e-4,
  relative = 1e-5
) {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++)
    expectNumber(actual[index], expected[index], absolute, relative);
}

function createSmallScene() {
  const scene = createBoundaryReference(60).scene;
  // A singular observed site remains part of the permuted training population.
  for (let row = 55; row < 60; row++) scene.predictors[row] = 1;
  const mask = new Uint32Array(60).fill(1);
  mask[1] = 0;
  mask[19] = 0;
  return {scene, mask};
}

/** Physically compact/permutate observations and run the independent float64 CPU fit oracle. */
function getDistributionReference(
  scene: NonstationarityScene,
  mask: Uint32Array,
  permutations: number,
  bandwidth: number
) {
  const included = Array.from({length: mask.length}, (_, row) => row).filter(
    row => mask[row] !== 0
  );
  const positions = new Float32Array(included.length * 2);
  const predictors = new Float32Array(included.length * scene.predictorCount);
  const response = new Float32Array(included.length);
  included.forEach((row, index) => {
    positions.set(scene.positions.subarray(row * 2, row * 2 + 2), index * 2);
    response[index] = scene.response[row];
    predictors.set(
      scene.predictors.subarray(row * scene.predictorCount, (row + 1) * scene.predictorCount),
      index * scene.predictorCount
    );
  });
  const spreads: number[][] = [];
  const counts: number[] = [];
  for (let run = 0; run <= permutations; run++) {
    const assignment = run
      ? getNonstationarityAssignment(included.length, SEED, run)
      : included.map((_, index) => index);
    const permutedPredictors = new Float32Array(predictors.length);
    const permutedResponse = new Float32Array(response.length);
    assignment.forEach((source, row) => {
      permutedResponse[row] = response[source];
      permutedPredictors.set(
        predictors.subarray(source * scene.predictorCount, (source + 1) * scene.predictorCount),
        row * scene.predictorCount
      );
    });
    const fit = computeGeographicallyWeightedRegressionOnCPU({
      positions,
      predictors: permutedPredictors,
      response: permutedResponse,
      predictorCount: scene.predictorCount,
      settings: {kernel: 'bisquare', bandwidths: [bandwidth]}
    });
    spreads.push(getLocalEstimateStandardDeviations(fit.coefficients, scene.predictorCount + 1));
    counts.push(fit.localStatus.filter(status => status === 0).length);
  }
  const table: number[] = [];
  for (let coefficient = 0; coefficient <= scene.predictorCount; coefficient++) {
    const observed = spreads[0][coefficient];
    const values = spreads
      .slice(1)
      .map(row => row[coefficient])
      .filter(Number.isFinite);
    const greater = values.filter(value => value >= observed).length;
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const deviation = Math.sqrt(
      values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length
    );
    table.push(
      observed,
      (greater + 1) / (permutations + 1),
      greater,
      mean,
      deviation,
      deviation > 0 ? (observed - mean) / deviation : NaN
    );
  }
  return {
    spreads: spreads.flat(),
    table,
    summary: [
      permutations,
      included.length,
      counts[0],
      counts.slice(1).reduce((sum, value) => sum + included.length - value, 0)
    ]
  };
}

it('forces grid versus scan for 65 masked permutations, singular fits and every distribution output', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {scene, mask} = createSmallScene();
  const settings = {kernel: 'bisquare' as const, bandwidths: [0.25]};
  const indexed = await runScene(device, scene, {
    settings,
    mask,
    indexGridSize: [8, 4],
    maximumPermutations: 65
  });
  const scanned = await runScene(device, scene, {
    settings,
    mask,
    indexGridSize: false,
    maximumPermutations: 65
  });
  expect(indexed.indexGridSize).toEqual([8, 4]);
  expect(indexed.testIndexGridSize).toEqual([8, 4]);
  expect(scanned.indexGridSize).toBeNull();
  expect(scanned.testIndexGridSize).toBeNull();
  const regression = computeGeographicallyWeightedRegressionOnCPU({...scene, mask, settings});
  expect(regression.singularRowCount).toBeGreaterThan(0);
  for (const key of [
    'coefficients',
    'standardErrors',
    'tValues',
    'fitted',
    'residuals',
    'hatDiagonal',
    'localR2',
    'significant',
    'localStatus'
  ] as const) {
    expectArray(indexed.results[0].output[key], regression[key], 5e-4, 2e-4);
    expectArray(scanned.results[0].output[key], regression[key], 5e-4, 2e-4);
  }
  const expected = getDistributionReference(scene, mask, 65, 0.25);
  for (const result of [indexed, scanned]) {
    expectArray(result.results[0].test.standardDeviations, expected.spreads, 1e-3, 2e-4);
    expectArray(result.results[0].test.table, expected.table, 1e-3, 2e-4);
    expectArray(result.results[0].test.summary, expected.summary, 0, 0);
    expect(result.results[0].output.approximation[0]).toBe(0);
    expect(result.results[0].test.approximation[0]).toBe(0);
  }
}, 90000);

function assertBoundaryResult(
  result: Readback,
  reference: ReturnType<typeof createBoundaryReference>,
  rowCount: number
) {
  const candidateCount = result.candidateTable.length / 7;
  for (const candidate of [0, 1]) {
    expect(Number.isFinite(result.candidateTable[candidate * 7])).toBe(true);
    expect(Number.isFinite(result.candidateTable[candidate * 7 + 1])).toBe(true);
  }
  const evaluated: {value: number; score: ReturnType<typeof reference.score>; index: number}[] = [];
  for (let candidate = 0; candidate < candidateCount; candidate++) {
    const value = result.candidateTable[candidate * 7];
    if (Number.isNaN(value)) continue; // Only unused, GPU-gated search slots may be NaN.
    const score = reference.score(value);
    evaluated.push({value, score, index: candidate});
    expect(Number.isFinite(result.candidateTable[candidate * 7 + 1])).toBe(true);
    // Counts are exact: losing even a single tail row must fail regardless of score tolerance.
    expect(result.candidateTable[candidate * 7 + 5]).toBe(score.count);
    expect(result.candidateTable[candidate * 7 + 6]).toBe(rowCount);
    expect(result.candidateTable[candidate * 7 + 2]).toBe(score.singular);
    const candidateRow = result.candidateTable.subarray(candidate * 7, candidate * 7 + 7);
    // AICc derivatives at fixed n are n/RSS and n*(2*n-2)/(n-trace-2)^2. For the
    // million-row candidates at bandwidth .25 and .61287 the RSS/trace/output half-ULP
    // budget is about .26 and .25, respectively. The 1.5 bound leaves over 1.2 units
    // for local-fit, reduction and score arithmetic rather than requiring exact float32 sums.
    expectNumber(
      candidateRow[1],
      score.aicc,
      1.5,
      0,
      `candidate ${candidate}: [${candidateRow}], reference RSS=${score.residualSum}, trace=${score.trace}, AICc=${score.aicc}`
    );
    expectNumber(result.candidateTable[candidate * 7 + 3], score.residualSum, 0.5, 2e-6);
    expectNumber(result.candidateTable[candidate * 7 + 4], score.trace, 0.5, 2e-6);
    if (candidate < 2) {
      expectNumber(result.bandwidthScores[candidate], score.aicc, 1.5, 0);
      expect(result.bandwidthSingularRowCounts[candidate]).toBe(score.singular);
    }
  }
  expect(evaluated.length).toBeGreaterThanOrEqual(2);
  expect(result.approximation[0]).toBe(0);
  const selected = reference.score(result.selectedBandwidth[1]);
  const expectedSummary = [
    selected.residualSum,
    selected.trace,
    selected.aicc,
    selected.rSquared,
    selected.count,
    1,
    selected.singular,
    selected.sigmaSquared,
    selected.adjustedAlpha,
    selected.criticalT,
    selected.logLikelihood,
    rowCount
  ];
  expectedSummary.forEach((value, index) =>
    expectNumber(
      result.summary[index],
      value,
      index === 9 ? 2e-5 : [0, 1, 2, 10].includes(index) ? 1.5 : 5e-4,
      index === 9 ? 0 : 2e-5
    )
  );
  const sampleRows = new Set([0, rowCount - 1]);
  for (let boundary = REVIEW_FOCUS_ROWS; boundary < rowCount; boundary += REVIEW_FOCUS_ROWS)
    for (const delta of [-1, 0, 1])
      if (boundary + delta < rowCount) sampleRows.add(boundary + delta);
  for (const row of sampleRows) {
    const expected = reference.local(row, result.selectedBandwidth[1]);
    for (const key of ['coefficients', 'standardErrors', 'tValues', 'significant'] as const)
      for (let column = 0; column < 2; column++)
        expectNumber(result[key][row * 2 + column], expected[key][column], 7e-4, 2e-4);
    expectNumber(result.fitted[row], expected.fitted, 2e-4, 1e-4);
    expectNumber(result.residuals[row], expected.residual, 2e-4, 1e-4);
    expectNumber(result.hatDiagonal[row], expected.hat, 2e-4, 1e-4);
    expectNumber(result.localR2[row], expected.localR2, 2e-4, 1e-4);
    expect(result.localStatus[row]).toBe(0);
    expectNumber(result.localConditionNumber[row], expected.conditionNumber, 2e-4, 1e-4);
  }
  return evaluated;
}

it('evaluates large-df inference against SciPy and compiles parallel summary NaN branches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const input = createInputBuffer(
    device,
    Float32Array.from(LARGE_SAMPLE_QUANTILE_CASES.flatMap(([df, tail]) => [df, tail]))
  );
  const output = createOutputBuffer(device, LARGE_SAMPLE_QUANTILE_CASES.length);
  const statistics = createInputBuffer(device, Float32Array.of(2, 1, 0.5, 2, 3, 0.5));
  const summary = createOutputBuffer(device, 12);
  const inputView = importGraphBuffer(
    graph,
    'quantile-input',
    input,
    'float32',
    LARGE_SAMPLE_QUANTILE_CASES.length * 2
  );
  const outputView = importGraphBuffer(
    graph,
    'quantile-output',
    output,
    'float32',
    LARGE_SAMPLE_QUANTILE_CASES.length
  );
  const statisticsView = importGraphBuffer(graph, 'summary-statistics', statistics, 'float32', 6);
  const summaryView = importGraphBuffer(graph, 'summary-output', summary, 'float32', 12);
  graph.add({
    getCommandNodes: () => [
      createWGSLKernelNode(graph, {
        id: 'large-sample-inference',
        operation: 'test',
        invocationCount: LARGE_SAMPLE_QUANTILE_CASES.length,
        bindings: [
          {name: 'inputs', view: inputView, type: 'f32', access: 'read'},
          {name: 'outputs', view: outputView, type: 'f32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${getGWRLargeSampleInferenceWGSL()}`,
        body: 'outputs[outputsOffset + index] = gwrLargeSampleCriticalValue(inputs[inputsOffset + index * 2u + 1u], inputs[inputsOffset + index * 2u]);'
      }),
      ...createGWRVarianceReductionNodes(graph, {
        id: 'parallel-summary',
        tileCount: 2,
        statistics: statisticsView,
        summary: summaryView
      })
    ]
  });
  const compiled = graph.compile();
  try {
    // The same compiled graph must publish finite R², invalid-fit NaN, and zero-variance NaN.
    for (const [valid, constant] of [
      [1, false],
      [0, false],
      [1, true]
    ] as const) {
      const initial = new Float32Array(12);
      initial[0] = 1;
      initial[5] = valid;
      summary.write(initial);
      if (constant) statistics.write(Float32Array.of(2, 1, 0, 2, 1, 0));
      submitGraph(device, compiled, undefined);
      const quantiles = await readFloat32(output, LARGE_SAMPLE_QUANTILE_CASES.length);
      quantiles.forEach((value, index) =>
        expectNumber(value, LARGE_SAMPLE_QUANTILE_CASES[index][2], 2e-5, 0)
      );
      const result = await readFloat32(summary, 12);
      if (valid && !constant) expectNumber(result[3], 0.8, 1e-6, 0);
      else expect(result[3]).toBeNaN();
    }
  } finally {
    compiled.destroy();
    for (const buffer of [input, output, statistics, summary]) buffer.destroy();
  }
});

for (const rowCount of [262143, 262144, 262145, 1048584])
  it(`crosses focus boundaries and checks tail, inference, every candidate and summary at ${rowCount} rows`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const reference = createBoundaryReference(rowCount);
    const result = await runScene(device, reference.scene, {
      settings: {kernel: 'bisquare', bandwidths: REVIEW_BANDWIDTHS},
      nonstationarity: false
    });
    const evaluated = assertBoundaryResult(result.results[0].output, reference, rowCount);
    const best = [...evaluated].sort(
      (first, second) =>
        first.score.singular - second.score.singular || first.score.aicc - second.score.aicc
    )[0];
    expect(result.results[0].output.selectedBandwidth[0]).toBe(best.index);
    expectNumber(result.results[0].output.selectedBandwidth[1], best.value, 1e-6, 0);
  }, 180000);

for (const search of ['ladder', 'golden-section'] as const)
  it(`selects one global bandwidth across heterogeneous focus batches using ${search}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const reference = createBoundaryReference(1048584);
    expect(reference.score(1.2, 0).aicc).toBeLessThan(reference.score(0.25, 0).aicc);
    expect(reference.score(0.25).aicc).toBeLessThan(reference.score(1.2).aicc);
    const result = await runScene(device, reference.scene, {
      settings: {kernel: 'bisquare', bandwidths: REVIEW_BANDWIDTHS},
      bandwidthSearch: search,
      refinement: search === 'ladder' ? 1 : false,
      maximumSearchIterations: 8,
      nonstationarity: false
    });
    const evaluated = assertBoundaryResult(result.results[0].output, reference, 1048584);
    if (search === 'ladder') {
      expect(evaluated).toHaveLength(4);
      const best = [...evaluated].sort(
        (first, second) =>
          first.score.singular - second.score.singular || first.score.aicc - second.score.aicc
      )[0];
      expect(result.results[0].output.selectedBandwidth[0]).toBe(best.index);
    } else {
      const expected = simulateGoldenSectionSearch({
        lower: 0.25,
        upper: 1.2,
        isAdaptive: false,
        maximumIterations: 8,
        precision: 'float32',
        getTolerance: getGPUGoldenSectionTolerance,
        evaluate: value => ({
          score: reference.score(value).aicc,
          singularRowCount: reference.score(value).singular
        })
      });
      expect(evaluated.length).toBe(expected.evaluations.length);
      expectNumber(result.results[0].output.selectedBandwidth[1], expected.optimumValue, 2e-4, 0);
    }
  }, 180000);

it('re-encodes 65 → 1 → 33 permutations and then excludes every row without stale distributions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {scene, mask} = createSmallScene();
  const result = await runScene(device, scene, {
    settings: {kernel: 'bisquare', bandwidths: [0.25]},
    mask,
    indexGridSize: [8, 4],
    maximumPermutations: 65,
    permutations: [65, 1, 33, 33],
    excludeLastEncoding: true
  });
  for (const [encoding, permutations] of [65, 1, 33].entries()) {
    const expected = getDistributionReference(scene, mask, permutations, 0.25);
    expectArray(result.results[encoding].test.summary, expected.summary, 0, 0);
    expectArray(result.results[encoding].test.table, expected.table, 1e-3, 2e-4);
    expectArray(
      result.results[encoding].test.standardDeviations.subarray(0, expected.spreads.length),
      expected.spreads,
      1e-3,
      2e-4
    );
    expect(
      Array.from(
        result.results[encoding].test.standardDeviations.subarray(expected.spreads.length)
      ).every(Number.isNaN)
    ).toBe(true);
  }
  const excluded = result.results[3];
  expectArray(excluded.test.summary, [33, 0, 0, 0], 0, 0);
  expect(Array.from(excluded.test.table).every(Number.isNaN)).toBe(true);
  expect(Array.from(excluded.test.standardDeviations).every(Number.isNaN)).toBe(true);
  expect(excluded.output.summary[5]).toBe(0);
}, 90000);

it('compares exact and cutoff-eight Gaussian support with distant large observations and a poorly conditioned design', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 31;
  const positions = new Float32Array(rows * 2);
  const predictors = new Float32Array(rows * 2);
  const response = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    const first = (row % 5) - 2;
    const second = first + 0.01 * Math.sin(row * 2.1);
    positions[row * 2] = row === rows - 1 ? 8.5 : (row % 6) * 0.01;
    predictors[row * 2] = first;
    predictors[row * 2 + 1] = second;
    response[row] = row === rows - 1 ? 1e8 : 1 + 2 * first - second + 0.005 * Math.cos(row);
  }
  const scene = {positions, predictors, response, predictorCount: 2};
  const settings = {kernel: 'gaussian' as const, bandwidths: [1]};
  for (const gaussianCutoff of [undefined, 8]) {
    const actual = await runScene(device, scene, {
      settings,
      indexGridSize: [8, 4],
      gaussianMode: gaussianCutoff ? 'truncated' : 'exact',
      gaussianCutoff,
      maximumPermutations: 1,
      permutations: [1, 1],
      settingsByEncoding: [settings, {kernel: 'bisquare', bandwidths: [1]}]
    });
    const expected = computeGeographicallyWeightedRegressionOnCPU({
      ...scene,
      settings,
      gaussianCutoff
    });
    expectArray(actual.results[0].output.coefficients, expected.coefficients, 0.05, 0.04);
    expectArray(actual.results[0].output.localStatus, expected.localStatus, 0, 0);
    expect(actual.results[0].output.approximation[0]).toBe(gaussianCutoff ? 1 : 0);
    expect(actual.results[0].test.approximation[0]).toBe(gaussianCutoff ? 1 : 0);
    expect(Number.isFinite(actual.results[0].output.coefficients[0])).toBe(true);
    expect(actual.results[1].output.approximation[0]).toBe(0);
    expect(actual.results[1].test.approximation[0]).toBe(0);
  }
  const bisquare = await runScene(device, scene, {
    settings: {kernel: 'bisquare', bandwidths: [1]},
    gaussianCutoff: 8,
    indexGridSize: [8, 4]
  });
  expect(bisquare.results[0].output.approximation[0]).toBe(0);
  expect(bisquare.results[0].test.approximation[0]).toBe(0);
}, 90000);

it('compiles a large finite Gaussian cutoff and publishes canonical status with closed-form coefficients', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 32;
  const predictors = Float32Array.from({length: rowCount}, (_, row) => (row % 5) - 2);
  const response = Float32Array.from(predictors, value => 1 + 2 * value);
  const positions = new Float32Array(rowCount * 2);
  for (let row = 0; row < rowCount; row++) positions[2 * row] = row / rowCount;
  const result = await runScene(
    device,
    {positions, predictors, response, predictorCount: 1},
    {
      settings: {kernel: 'gaussian', bandwidths: [1]},
      gaussianCutoff: 1e20,
      indexGridSize: [8, 4]
    }
  );
  const output = result.results[0].output;
  for (let row = 0; row < rowCount; row++) {
    expectNumber(output.coefficients[2 * row], 1, 2e-4, 0);
    expectNumber(output.coefficients[2 * row + 1], 2, 2e-4, 0);
    expect(output.localStatus[row]).toBe(0);
  }
  expect(output.summary[4]).toBe(rowCount);
  expect(output.approximation[0]).toBe(1);
  expect(result.results[0].test.approximation[0]).toBe(1);
});

it('publishes adjacent coefficient and inference views with closed-form inference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const rowCount = 16;
  const predictors = Float32Array.from({length: rowCount}, (_, row) => (row % 2 === 0 ? -1 : 1));
  const parameters = getGPUGeographicallyWeightedRegressionParameterValues(
    {kernel: 'bisquare', bandwidths: [1]},
    1
  );
  const buffers = [
    createInputBuffer(device, new Float32Array(rowCount * 2)),
    createInputBuffer(device, predictors),
    createInputBuffer(
      device,
      Float32Array.from(predictors, (value, row) => 1 + 2 * value + (row % 4 < 2 ? 0.25 : -0.25))
    ),
    createInputBuffer(device, parameters),
    createOutputBuffer(device, rowCount * 4)
  ];
  const coefficients = importGraphBuffer(
    graph,
    'adjacent-output',
    buffers[4],
    'float32',
    rowCount * 2
  );
  const standardErrors = graph.createDataView(coefficients.buffer, {
    format: 'float32',
    length: rowCount * 2,
    byteOffset: rowCount * 8
  });
  graph.add(
    new GPUGeographicallyWeightedRegression({
      positions: importGraphBuffer(graph, 'positions', buffers[0], 'float32x2', rowCount),
      predictors: importGraphBuffer(graph, 'predictors', buffers[1], 'float32', rowCount),
      response: importGraphBuffer(graph, 'response', buffers[2], 'float32', rowCount),
      parameters: importGraphBuffer(graph, 'parameters', buffers[3], 'float32', parameters.length),
      predictorCount: 1,
      maximumBandwidthCount: 1,
      refinement: false,
      indexGridSize: false,
      output: {coefficients, standardErrors}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const values = await readFloat32(buffers[4], rowCount * 4);
    // Coincident sites give uniform weights. Noise is orthogonal to [1, x], RSS=1,
    // tr(S)=2, and X'X=diag(16,16), hence both standard errors equal 1/sqrt(224).
    const standardError = 1 / Math.sqrt(rowCount * (rowCount - 2));
    for (let row = 0; row < rowCount; row++) {
      expectNumber(values[2 * row], 1, 2e-5, 0);
      expectNumber(values[2 * row + 1], 2, 2e-5, 0);
      expectNumber(values[rowCount * 2 + 2 * row], standardError, 1e-5, 0);
      expectNumber(values[rowCount * 2 + 2 * row + 1], standardError, 1e-5, 0);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
