// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A contributor inside a published luma.gl GPUGraph beside GPUCPUOperation stages, and its CPU twin
// in the same graph: the pilot of dev-docs/contributor-contract.md section 14.

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {getCPUOperationFloat32Rows} from '../../../src/analysis-core/operations/cpu-operation-rows';
import {CPUVariogramModelFitOperation} from '../../../src/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation';
import {
  GPUVariogramModelFitOperation,
  type VariogramModelFitOperationProps
} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const MODELS = ['gaussian', 'exponential', 'spherical'] as const;
const BIN_COUNT = 20;
// Noise-free bins of an exponential model: nugget 0.2, partial sill 1.5, practical range 2.2.
const TRUTH = {nugget: 0.2, sill: 1.5, range: 2.2};
const DISTANCES = Float32Array.from({length: BIN_COUNT}, (_, index) => (index + 1) * 0.2);
const SEMIVARIANCES = Float32Array.from(
  DISTANCES,
  distance => TRUTH.nugget + TRUTH.sill * (1 - Math.exp((-3 * distance) / TRUTH.range))
);
const PAIR_COUNTS = Uint32Array.from({length: BIN_COUNT}, (_, index) => 10 + index);
const EXPONENTIAL_ROW = MODELS.indexOf('exponential');

function getInputs(semivariances: Float32Array = SEMIVARIANCES) {
  return {
    distances: {format: 'float32' as const, values: DISTANCES},
    semivariances: {format: 'float32' as const, values: semivariances},
    pairCounts: {format: 'uint32' as const, values: PAIR_COUNTS}
  };
}

function getFitProps(semivariances: string): VariogramModelFitOperationProps {
  return {
    id: 'fit',
    distances: 'distances',
    semivariances,
    pairCounts: 'pairCounts',
    models: MODELS,
    weighting: 'pairs',
    output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
  };
}

for (const location of ['gpu', 'cpu'] as const) {
  it(`the ${location} fit operation recovers the model the bins were drawn from`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props = getFitProps('semivariances');
    const rows = await runOperationGraph(device, {
      inputs: getInputs(),
      operations: [
        location === 'gpu'
          ? new GPUVariogramModelFitOperation(props)
          : new CPUVariogramModelFitOperation(props)
      ],
      outputs: ['parameters', 'diagnostics', 'status']
    });
    const [nugget, sill, range, residual] = rows['parameters'].subarray(
      EXPONENTIAL_ROW * 4,
      EXPONENTIAL_ROW * 4 + 4
    );
    expect(rows['status'][EXPONENTIAL_ROW]).toBe(0);
    expect(nugget).toBeCloseTo(TRUTH.nugget, 2);
    expect(sill).toBeCloseTo(TRUTH.sill, 2);
    expect(range).toBeCloseTo(TRUTH.range, 2);
    // The bins are exact, so the weighted residual of the right family is rounding only.
    expect(residual).toBeLessThan(1e-4);
    expect(rows['diagnostics'][EXPONENTIAL_ROW * 4]).toBe(BIN_COUNT);
    // The other families cannot reproduce exponential bins.
    for (const model of ['gaussian', 'spherical'] as const) {
      expect(rows['parameters'][MODELS.indexOf(model) * 4 + 3]).toBeGreaterThan(residual * 10);
    }
  });
}

it('CPU stages before and after the GPU fit see its values, and the CPU twin agrees', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scale = 1.5;
  const results: Record<string, Awaited<ReturnType<typeof runOperationGraph>>> = {};
  for (const location of ['gpu', 'cpu'] as const) {
    const props = getFitProps('scaled');
    results[location] = await runOperationGraph(device, {
      id: `mixed-${location}`,
      inputs: getInputs(),
      operations: [
        new GPUCPUOperation({
          id: 'scale-semivariances',
          inputs: {semivariances: 'semivariances'},
          outputs: {scaled: gpuGraph.vector('float32', {length: BIN_COUNT})},
          execute({inputs}) {
            const values = getCPUOperationFloat32Rows(
              'scale-semivariances',
              inputs,
              'semivariances'
            );
            return {scaled: Float32Array.from(values, value => value * scale)};
          }
        }),
        location === 'gpu'
          ? new GPUVariogramModelFitOperation(props)
          : new CPUVariogramModelFitOperation(props),
        new GPUCPUOperation({
          id: 'select-model',
          inputs: {parameters: 'parameters'},
          outputs: {best: gpuGraph.vector('float32x4', {length: 1})},
          execute({inputs}) {
            const parameters = getCPUOperationFloat32Rows(
              'select-model',
              inputs,
              'parameters',
              'float32x4'
            );
            let bestRow = 0;
            for (let row = 1; row < MODELS.length; row++) {
              if (parameters[row * 4 + 3] < parameters[bestRow * 4 + 3]) {
                bestRow = row;
              }
            }
            return {best: Float32Array.from(parameters.subarray(bestRow * 4, bestRow * 4 + 4))};
          }
        })
      ],
      outputs: ['best', 'parameters', 'diagnostics', 'status']
    });
  }
  for (const location of ['gpu', 'cpu'] as const) {
    const [nugget, sill, range] = results[location]['best'];
    // Scaling the semivariances scales the nugget and the sill and leaves the range alone.
    expect(nugget).toBeCloseTo(TRUTH.nugget * scale, 2);
    expect(sill).toBeCloseTo(TRUTH.sill * scale, 2);
    expect(range).toBeCloseTo(TRUTH.range, 2);
  }
  expect(Array.from(results['cpu']['status'])).toEqual(Array.from(results['gpu']['status']));
  // Every output, relative to max(1, |cpu|). Not float32 level: both sides stop a golden-section
  // range search at the same bracket, and the GPU reduces in float32, so the fitted values differ
  // by the fit's convergence tolerance. Measured on the test GPU: at most 1.4e-4 on parameters
  // and best (the nugget, sill and range of the gaussian and spherical rows), 2.5e-5 on the
  // unweighted residual, 5e-5 on the AIC of an inexact family, 0 on bin counts and bracket widths.
  const TOLERANCE = 3e-4;
  // The exponential bins are exact, so that row's unweighted residual is rounding noise and its
  // AIC, n log(residual / n) + 6, is the log of noise (measured -504 on the GPU, -644 on the
  // CPU). Both must show a residual below 1e-10 per bin rather than agree with each other.
  const EXACT_FIT_AIC_INDEX = EXPONENTIAL_ROW * 4 + 2;
  const exactFitAICBound = BIN_COUNT * Math.log(1e-10) + 6;
  const differences: string[] = [];
  for (const output of ['best', 'parameters', 'diagnostics'] as const) {
    const expected = results['cpu'][output];
    const actual = results['gpu'][output];
    expect(actual.length).toBe(expected.length);
    for (let index = 0; index < expected.length; index++) {
      if (output === 'diagnostics' && index === EXACT_FIT_AIC_INDEX) {
        expect(actual[index]).toBeLessThan(exactFitAICBound);
        expect(expected[index]).toBeLessThan(exactFitAICBound);
        continue;
      }
      const difference =
        Math.abs(actual[index] - expected[index]) / Math.max(1, Math.abs(expected[index]));
      if (!(difference <= TOLERANCE)) {
        differences.push(`${output}[${index}] gpu=${actual[index]} cpu=${expected[index]}`);
      }
    }
  }
  expect(differences).toEqual([]);
});

it('both fit operations report insufficient bins with status 1 and NaN rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const location of ['gpu', 'cpu'] as const) {
    const props = getFitProps('semivariances');
    const rows = await runOperationGraph(device, {
      inputs: {...getInputs(), pairCounts: {format: 'uint32', values: new Uint32Array(BIN_COUNT)}},
      operations: [
        location === 'gpu'
          ? new GPUVariogramModelFitOperation(props)
          : new CPUVariogramModelFitOperation(props)
      ],
      outputs: ['parameters', 'status']
    });
    expect(Array.from(rows['status'])).toEqual([1, 1, 1]);
    expect(rows['parameters'].every(Number.isNaN)).toBe(true);
  }
});
