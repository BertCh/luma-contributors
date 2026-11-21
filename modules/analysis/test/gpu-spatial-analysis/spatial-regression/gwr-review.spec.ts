// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import type {GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {getStudentTWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-student-t';
import {GPUGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {GPUGeographicallyWeightedRegressionNonstationarityTest} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression-nonstationarity-test';
import {getGPUGeographicallyWeightedRegressionParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createGWRReviewHarness} from './gwr-review-test-utils';
import {
  checkTranslatedSingularity,
  checkCountCompleteness,
  checkNonzeroTies,
  checkZeroBudget,
  checkConstantStatistics,
  checkEmptyFirstConstantReduction,
  GWR_REVIEW_TAIL_CASES
} from './gwr-review-cases';
import {createGWRAuditScene} from './gwr-mathematical-audit-fixtures';

type Harness = ReturnType<typeof createGWRReviewHarness>;
it('R3 parallel variance with an empty first tile and thirteen identical means', async () =>
  run((harness, execute) =>
    checkEmptyFirstConstantReduction(harness, execute, harness.read, true)
  ));
async function run(
  check: (
    harness: Harness,
    execute: (nodes: readonly GPUCommandNode[]) => Promise<void>
  ) => Promise<void>
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createGWRReviewHarness(device);
  let compiled: ReturnType<Harness['submit']> | undefined;
  try {
    await check(harness, async nodes => {
      if (compiled) submitGraph(device, compiled);
      else compiled = harness.submit(nodes);
    });
  } finally {
    compiled?.destroy();
    harness.destroy();
  }
}

for (const translation of [0, 1000])
  it(`R1 rank-one translated design ${translation}`, async () =>
    run((harness, execute) =>
      checkTranslatedSingularity(harness, execute, translation, harness.read)
    ));
for (const chunked of [false, true, 'chunk'] as const)
  for (const [capacity, includeCount] of [
    [40, true],
    [32, false]
  ] as const)
    it(`R2 count-only loss and recovery chunked=${chunked}, capacity=${capacity}, count=${includeCount}`, async () =>
      run((harness, execute) =>
        checkCountCompleteness(harness, execute, chunked, harness.read, capacity, includeCount)
      ));
it('R4 ten nonzero identical simulated spreads', async () =>
  run((harness, execute) => checkNonzeroTies(harness, execute, harness.read)));
it('R7 valid submission followed by raw zero clears the distribution', async () =>
  run((harness, execute) => checkZeroBudget(harness, execute, harness.read)));
it('R3 ten float32(0.1) responses and zero RSS have undefined global R2', async () =>
  run((harness, execute) => checkConstantStatistics(harness, execute, harness.read)));

for (const indexed of [false, true]) {
  for (const value of [3, 0.1])
    it(`R3 nonzero constant local and global R2: value=${value}, grid=${indexed}`, async () =>
      run(async (harness, execute) => {
        const scene = createGWRAuditScene();
        scene.response.fill(value);
        const {port, graph} = harness;
        const output = {
          coefficients: port('float32', Array(27).fill(-999)),
          localR2: port('float32', Array(9).fill(-999)),
          localStatus: port('uint32', Array(9).fill(99)),
          summary: port('float32', Array(12).fill(-999))
        };
        const nodes = new GPUGeographicallyWeightedRegression({
          positions: port('float32x2', scene.positions),
          predictors: port('float32', scene.predictors),
          predictorCount: 2,
          response: port('float32', scene.response),
          mask: port('uint32', scene.mask),
          parameters: port('float32', [0, 0, 1, 0, 3]),
          maximumBandwidthCount: 1,
          indexGridSize: indexed ? [2, 2] : false,
          gaussianCutoff: indexed ? 8 : undefined,
          refinement: false,
          output
        }).getCommandNodes(graph);
        await execute(nodes);
        expect(await harness.read(output.localStatus)).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 2]);
        expect((await harness.read(output.localR2)).every(Number.isNaN)).toBe(true);
        expect((await harness.read(output.summary))[3]).toBeNaN();
      }));
  for (const kind of ['default', 'invalid'] as const)
    it(`R5/R8 adaptive ${kind} candidate grid=${indexed}`, async () =>
      run(async (harness, execute) => {
        const scene = createGWRAuditScene();
        const {port, graph} = harness;
        const capacity = kind === 'default' ? 2 : 1;
        const output = {
          coefficients: port('float32', Array(27).fill(-999)),
          candidateTable: port('float32', Array(7 * capacity).fill(-999)),
          selectedBandwidth: port('float32', [-999, -999]),
          summary: port('float32', Array(12).fill(-999))
        };
        const nodes = new GPUGeographicallyWeightedRegression({
          positions: port('float32x2', scene.positions),
          predictors: port('float32', scene.predictors),
          predictorCount: 2,
          response: port('float32', scene.response),
          mask: port('uint32', scene.mask),
          parameters: port(
            'float32',
            getGPUGeographicallyWeightedRegressionParameterValues(
              {bandwidthMode: 'adaptive', bandwidths: kind === 'default' ? [0, 0] : [9]},
              capacity
            )
          ),
          maximumBandwidthCount: capacity,
          bandwidthSearch: kind === 'default' ? 'golden-section' : 'ladder',
          maximumSearchIterations: 1,
          indexGridSize: indexed ? [2, 2] : false,
          refinement: false,
          output
        }).getCommandNodes(graph);
        await execute(nodes);
        const table = await harness.read(output.candidateTable);
        if (kind === 'invalid') {
          expect(table[1]).toBeNaN();
          expect(table.slice(2, 7)).toEqual([0, 0, 0, 0, 8]);
        } else {
          expect((await harness.read(output.selectedBandwidth))[1]).toBe(8);
          expect((await harness.read(output.summary))[5]).toBe(1);
          expect(Number.isFinite(table[1])).toBe(true);
          expect(Math.abs(table[1] - 339.5339004011)).toBeLessThan(0.01);
        }
      }));
}

it('R11 direct Student-t df-cap tail and NaN', async () =>
  run(async (harness, execute) => {
    const output = harness.port('float32', Array(GWR_REVIEW_TAIL_CASES.length + 1).fill(-999));
    const body = GWR_REVIEW_TAIL_CASES.map(
      ([value, degrees], index) =>
        `result[resultOffset+${index}u]=gwrStudentTwoSidedTail(${value.toFixed(2)},${degrees}.0);`
    ).join('\n');
    await execute([
      createWGSLKernelNode(harness.graph, {
        id: 'review-direct-t',
        operation: 'GWRReview',
        invocationCount: 1,
        bindings: [{name: 'result', view: output, type: 'f32', access: 'read_write'}],
        declarations: `${GET_NAN_WGSL}\n${getStudentTWGSL()}`,
        body: `${body}\nresult[resultOffset+${GWR_REVIEW_TAIL_CASES.length}u]=gwrStudentTwoSidedTail(getNaN(),7.0);`
      })
    ]);
    const values = await harness.read(output);
    for (const [index, [, , expected]] of GWR_REVIEW_TAIL_CASES.entries()) {
      expect(Number.isFinite(values[index])).toBe(true);
      expect(Math.abs(values[index] / expected - 1)).toBeLessThan(3e-5);
    }
    expect(values[GWR_REVIEW_TAIL_CASES.length]).toBeNaN();
  }));

it('R6 conditional refits retain the observed winner from a two-value ladder', async () =>
  run(async (harness, execute) => {
    const {port, graph} = harness;
    const positions = port('float32x2', Array.from({length: 16}, (_, row) => [row, 0]).flat());
    const predictors = port(
      'float32',
      Array.from({length: 16}, (_, row) => row % 2)
    );
    const response = port(
      'float32',
      Array.from({length: 16}, (_, row) => Math.floor(row / 4) ** 2 + (row % 2))
    );
    const parameters = port(
      'float32',
      getGPUGeographicallyWeightedRegressionParameterValues(
        {kernel: 'gaussian', bandwidths: [2, 8]},
        2
      )
    );
    const coefficients = port('float32', Array(32).fill(-999));
    const selectedBandwidth = port('float32', [-999, -999]);
    const observed = new GPUGeographicallyWeightedRegression({
      positions,
      predictors,
      response,
      predictorCount: 1,
      parameters,
      maximumBandwidthCount: 2,
      refinement: false,
      indexGridSize: false,
      output: {coefficients, selectedBandwidth}
    }).getCommandNodes(graph);
    const output = {
      table: port('float32', Array(12).fill(-999)),
      summary: port('float32', Array(4).fill(-999)),
      standardDeviations: port('float32', Array(12).fill(-999))
    };
    const permutation = new GPUGeographicallyWeightedRegressionNonstationarityTest({
      positions,
      predictors,
      response,
      predictorCount: 1,
      coefficients,
      selectedBandwidth,
      bandwidthParameters: parameters,
      parameters: port(
        'uint32',
        getGPUPermutationParameterValues({seed: 12345, permutationCount: 5})
      ),
      maximumBandwidthCount: 2,
      maximumPermutations: 5,
      indexGridSize: false,
      output
    }).getCommandNodes(graph);
    await execute([...observed, ...permutation]);
    expect((await harness.read(selectedBandwidth))[1]).toBe(2);
    // Independent NumPy SVD, gwr-review.oracle.py; float32 local solves allow 0.02%.
    const expected = [
      3.1470446556653178,
      1 / 6,
      0,
      1.5568862058165682,
      0.5235275694903528,
      3.0373919971334993,
      0.34080305790515947,
      1,
      5,
      1.822684149383678,
      0.4624661922227455,
      -3.204301452515203
    ];
    const actual = await harness.read(output.table);
    for (const [index, value] of expected.entries()) {
      expect(Number.isFinite(actual[index])).toBe(true);
      expect(Math.abs(actual[index] - value)).toBeLessThan(2e-4 * Math.max(1, Math.abs(value)));
    }
    expect(await harness.read(output.summary)).toEqual([5, 16, 16, 0]);
    const spreads = [
      3.1470446556653178, 0.34080305790515947, 1.523568732110783, 1.7754390033496723,
      0.716827029150525, 1.0760594125475909, 1.6086275324574435, 1.9592146811315811,
      2.3698087285352947, 1.7805397233389555, 1.5655990068287946, 2.522167926550591
    ];
    const distribution = await harness.read(output.standardDeviations);
    for (const [index, value] of spreads.entries()) {
      expect(Number.isFinite(distribution[index])).toBe(true);
      expect(Math.abs(distribution[index] - value)).toBeLessThan(2e-4 * Math.max(1, value));
    }
  }));
