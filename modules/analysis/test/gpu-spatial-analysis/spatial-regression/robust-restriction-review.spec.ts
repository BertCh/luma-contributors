// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPURestrictedRegimeRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression-operation';
import {CPURestrictedRegimeRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-restricted-regime-regression-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GPURestrictedRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import type {
  CPURestrictedRegimeRegressionInput,
  CPURestrictedRegimeKernel
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-restricted-regime-regression-operation';
import {ROBUST_RESTRICTION_FIXTURE as fixture} from './robust-restriction-fixture';

const regimes = Uint32Array.of(0, 1, 0, 1, 0, 1, 0, 1);
const response = Float32Array.of(0, 1, 4, 9, 3, 12, 10, 10);

it('R3 uses restricted residual degrees without an unrestricted comparison', async () => {
  for (const result of await runScene(
    {
      predictors: Float32Array.of(0, 0, 1, 1),
      regimes: Uint32Array.of(0, 1, 0, 1),
      response: Float32Array.of(0, 1, 4, 9)
    },
    [false, true]
  )) {
    expectRows(result['status'], [1]);
    expectRows(result['robustStatus'], [0]);
    expectRows(result['robustCovariance'], [0.75, 0.25, -0.5, 0.25, 0.75, -0.5, -0.5, -0.5, 1]);
    expectRows(result['robustChow'], [9, 1, 0.0026997960632601883]);
    expect(Array.from(result['coefficients']).every(Number.isNaN)).toBe(true);
  }
});

async function runScene(
  input: CPURestrictedRegimeRegressionInput,
  common: boolean[],
  kernel?: CPURestrictedRegimeKernel,
  predictorCount = 1,
  regimeCount = 2,
  operationTypes = [GPURestrictedRegimeRegressionOperation, CPURestrictedRegimeRegressionOperation],
  robustEnabled = true
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const results = [];
  for (const Operation of operationTypes) {
    const inputs: Record<
      string,
      {format: 'float32' | 'uint32'; values: Float32Array | Uint32Array}
    > = {
      predictors: {format: 'float32', values: input.predictors},
      response: {format: 'float32', values: input.response},
      regimes: {format: 'uint32', values: input.regimes},
      ...(input.mask ? {mask: {format: 'uint32' as const, values: input.mask}} : {}),
      ...(kernel
        ? {
            kernelOffsets: {format: 'uint32' as const, values: kernel.offsets},
            kernelNeighbors: {format: 'uint32' as const, values: kernel.neighbors},
            kernelWeights: {format: 'float32' as const, values: kernel.weights},
            ...Object.fromEntries(
              Object.entries(kernel.status ?? {})
                .filter(([, values]) => values)
                .map(([name, values]) => [`kernel_${name}`, {format: 'uint32', values}])
            )
          }
        : {})
    };
    results.push(
      await runOperationGraph(device, {
        inputs,
        operations: [
          new Operation({
            rowCount: input.response.length,
            predictorCount,
            regimeCount,
            common,
            predictors: 'predictors',
            response: 'response',
            regimes: 'regimes',
            mask: input.mask ? 'mask' : undefined,
            robust: !robustEnabled
              ? undefined
              : kernel
                ? {
                    type: 'hac',
                    kernelWeights: {
                      offsets: 'kernelOffsets',
                      neighbors: 'kernelNeighbors',
                      weights: 'kernelWeights',
                      status: Object.fromEntries(
                        Object.entries(kernel.status ?? {})
                          .filter(([, values]) => values)
                          .map(([name]) => [name, `kernel_${name}`])
                      )
                    }
                  }
                : {type: 'white'},
            output: {
              coefficients: 'coefficients',
              covariance: 'covariance',
              fitted: 'fitted',
              chow: 'chow',
              stability: 'stability',
              status: 'status',
              robust: robustEnabled
                ? {
                    covariance: 'robustCovariance',
                    chow: 'robustChow',
                    stability: 'robustStability',
                    status: 'robustStatus'
                  }
                : undefined
            }
          })
        ],
        outputs: [
          'coefficients',
          'covariance',
          'fitted',
          'chow',
          'stability',
          'status',
          ...(robustEnabled
            ? ['robustCovariance', 'robustChow', 'robustStability', 'robustStatus']
            : [])
        ]
      })
    );
  }
  return results;
}

function expectRows(actual: ArrayLike<number>, expected: readonly number[], tolerance = 5e-4) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, slot) => {
    if (Number.isNaN(value)) expect(actual[slot]).toBeNaN();
    else {
      expect(Number.isFinite(actual[slot])).toBe(true);
      expect(Math.abs(actual[slot] - value)).toBeLessThanOrEqual(
        tolerance * Math.max(1, Math.abs(value))
      );
    }
  });
}

it('R2 suppresses the exactly zero all-ones HAC sandwich on both forms', async () => {
  const results = await runScene(
    {predictors: Float32Array.of(0, 0, 1, 1, 2, 2, 3, 3), regimes, response},
    [false, true],
    {
      offsets: Uint32Array.from({length: 9}, (_, row) => row * 8),
      neighbors: Uint32Array.from({length: 64}, (_, slot) => slot % 8),
      weights: new Float32Array(64).fill(1)
    }
  );
  // Exact normal equations: K=11' makes the meat (X'e)(X'e)'=0.
  for (const result of results) {
    expectRows(result['status'], [0]);
    expectRows(result['robustStatus'], [3]);
    expectNaNCovariance(result['robustCovariance'], 3);
    expectRows(result['robustChow'], [NaN, 0, NaN]);
    expectRows(result['robustStability'], [NaN, 0, NaN, NaN, 0, NaN, NaN, 0, NaN]);
  }
});

it('R9 merges both fixed row tiles, including the complete observation in the partial final tile', async () => {
  const input = {
    predictors: Float32Array.from([...fixture.predictors, ...fixture.predictors]),
    response: Float32Array.from([...fixture.response, ...fixture.response]),
    regimes: Uint32Array.from([...fixture.regimes, ...fixture.regimes]),
    mask: Uint32Array.from([...fixture.mask, ...fixture.mask])
  };
  for (const result of await runScene(input, [false, true, false], undefined, 2, 3)) {
    expectRows(result['status'], [0]);
    expectRows(result['robustStatus'], [0]);
    expectRows(
      result['robustCovariance'],
      fixture.expected.white.covariance.map(value => value / 2),
      1e-5
    );
    expectRows(result['robustChow'], [28.17795671174808, 4, 0.000011478741611841643]);
    // Oracle with only the first tile's meat gives 28.1804010073 and loses 0.0002131745
    // from covariance[8]. These bounds accommodate float32 but reject that mutation.
    expect(Math.abs(result['robustChow'][0] - 28.17795671174808)).toBeLessThan(5e-4);
    expect(Math.abs(result['robustCovariance'][0] - 0.00296013184)).toBeLessThan(2e-6);
    // For df 2, survival=exp(-x/2); for df 4, survival=exp(-x/2)*(1+x/2).
    const expectedStability = fixture.expected.white.stability.map(value =>
      value === null ? NaN : value
    );
    for (let test = 0; test < 4; test++) {
      if (expectedStability[test * 3 + 1] === 0) continue;
      const statistic = 2 * expectedStability[test * 3];
      const degrees = expectedStability[test * 3 + 1];
      expectedStability[test * 3] = statistic;
      expectedStability[test * 3 + 2] =
        Math.exp(-statistic / 2) * (degrees === 4 ? 1 + statistic / 2 : 1);
    }
    expectRows(result['robustStability'], expectedStability);
    expectRows(result['coefficients'], fixture.expected.coefficients);
    expectRows(
      result['fitted'],
      [...fixture.expected.fitted, ...fixture.expected.fitted].map(value =>
        value === null ? NaN : value
      )
    );
  }
});

it('R7 rejects two-word optional kernel status ports through each executing operation', async () => {
  for (const Operation of [
    GPURestrictedRegimeRegressionOperation,
    CPURestrictedRegimeRegressionOperation
  ]) {
    await expect(
      runScene(
        {predictors: Float32Array.of(0, 0, 1, 1, 2, 2, 3, 3), regimes, response},
        [false, true],
        {
          offsets: Uint32Array.from({length: 9}, (_, row) => row),
          neighbors: Uint32Array.from({length: 8}, (_, row) => row),
          weights: new Float32Array(8).fill(1),
          status: {overflow: new Uint32Array(2)}
        },
        1,
        2,
        [Operation]
      )
    ).rejects.toThrow(/status.*(length|one)/);
  }
});

it('R5 preserves joint HC0 inference across an exactly representable varying-predictor shift', async () => {
  for (const shift of [0, 1e6])
    for (const result of await runScene(
      {
        predictors: Float32Array.from([0, 0, 1, 1, 0, 0, 1, 1], value => shift + value / 16),
        regimes,
        response
      },
      [false, false]
    )) {
      expectRows(result['status'], [0]);
      expectRows(result['robustStatus'], [0]);
      expectRows(result['robustChow'], [2.889812889812891, 2, 0.2357681328338333]);
      expectRows(result['robustStability'].subarray(6), [2.889812889812891, 2, 0.2357681328338333]);
      if (shift === 0)
        expectRows(
          result['robustCovariance'],
          [1.125, 0, -18, 0, 0, 15.125, 0, -242, -18, 0, 1440, 0, 0, -242, 0, 3904]
        );
    }
});

it('R4 rejects excess requiredCount without an overflow port and accepts equality', async () => {
  for (const excess of [0, 2])
    for (const result of await runScene(
      {
        predictors: Float32Array.from(fixture.predictors),
        regimes: Uint32Array.from(fixture.regimes),
        response: Float32Array.from(fixture.response),
        mask: Uint32Array.from(fixture.mask)
      },
      [false, true, false],
      {
        offsets: Uint32Array.from(fixture.offsets),
        neighbors: Uint32Array.from(fixture.neighbors),
        weights: Float32Array.from(fixture.weights),
        status: {requiredCount: Uint32Array.of(fixture.neighbors.length + excess)}
      },
      2,
      3
    )) {
      expectRows(result['robustStatus'], [excess ? 2 : 0]);
      if (excess) {
        expectNaNCovariance(result['robustCovariance'], 7);
        expectRows(result['robustChow'], [NaN, 0, NaN]);
      } else {
        expectRows(result['robustCovariance'], fixture.expected.hac.covariance);
        expectRows(result['robustChow'], fixture.expected.hac.chow);
        expectRows(
          result['robustStability'],
          fixture.expected.hac.stability.map(value => (value === null ? NaN : value))
        );
      }
    }
});

it('R3 preserves valid restricted inference beside classical status three', async () => {
  for (const result of await runScene(
    {predictors: Float32Array.of(0, 0, 0, 1, 0, 2, 0, 3), regimes, response},
    [false, true]
  )) {
    // NumPy dense SVD/HC0 and SciPy tails: robust-restriction-review.oracle.py.
    expectRows(result['status'], [3]);
    expectRows(result['robustStatus'], [0]);
    expectRows(result['coefficients'], [4.25, 3.5, 3]);
    expectRows(result['robustCovariance'], [3.296875, 0, 0, 0, 4.375, -1.875, 0, -1.875, 1.25]);
    expectRows(result['robustChow'], [0.07331975560081463, 1, 0.7865631586537749]);
    expectRows(result['robustStability'], [
      0.07331975560081463,
      1,
      0.7865631586537749,
      NaN,
      0,
      NaN,
      0.07331975560081463,
      1,
      0.7865631586537749
    ]);
    expect(Array.from(result['chow']).every(Number.isNaN)).toBe(true);
  }
});

function identityKernel(capacity = 8): CPURestrictedRegimeKernel {
  return {
    offsets: Uint32Array.from({length: 9}, (_, row) => row),
    neighbors: Uint32Array.from({length: capacity}, (_, row) => row),
    weights: new Float32Array(capacity).fill(1),
    status: {requiredCount: Uint32Array.of(8), complete: Uint32Array.of(1)}
  };
}

it('W5-41 rejects retained-edge loss without count and recovers on persistent GPU storage', async () => {
  for (const withCount of [false, true]) {
    const context = await createRegressionTestContext();
    const {graph, device, input, output} = context;
    const required = output('uint32', 1);
    const count = output('uint32', 1);
    const complete = output('uint32', 1);
    const unresolved = output('uint32', 1);
    const covariance = output('float32', 9);
    const chow = output('float32', 3);
    const status = output('uint32', 1);
    const kernel = identityKernel(10);
    graph.add(
      new GPURestrictedRegimeRegression({
        predictors: input('float32', [0, 0, 1, 1, 2, 2, 3, 3]),
        response: input('float32', response),
        regimes: input('uint32', regimes),
        predictorCount: 1,
        regimeCount: 2,
        common: [false, true],
        robust: {
          type: 'hac',
          kernelWeights: {
            offsets: input('uint32', kernel.offsets),
            neighbors: input('uint32', kernel.neighbors),
            weights: input('float32', kernel.weights),
            status: {
              requiredCount: required.view,
              ...(withCount ? {count: count.view} : {}),
              complete: complete.view,
              unresolvedCount: unresolved.view,
              uncertainCount: input('uint32', [1])
            }
          }
        },
        output: {
          coefficients: output('float32', 3).view,
          covariance: output('float32', 9).view,
          fitted: output('float32', 8).view,
          chow: output('float32', 3).view,
          status: output('uint32', 1).view,
          robust: {
            covariance: covariance.view,
            chow: chow.view,
            stability: output('float32', 9).view,
            status: status.view
          }
        }
      })
    );
    const compiled = graph.compile();
    try {
      // These are persistent Buffer identities, compiled once, with no mixed-executor uploads.
      // In the first route there is deliberately NO count port: capacity 10, retained offset 8.
      const frames = withCount
        ? [
            [8, 8, 1, 0, 0],
            [8, 7, 1, 0, 2],
            [8, 9, 1, 0, 2],
            [8, 17, 1, 0, 2],
            [8, 8, 1, 0, 0]
          ]
        : [
            [8, 8, 1, 0, 0],
            [10, 8, 1, 0, 2],
            [8, 8, 1, 0, 0],
            [8, 8, 0, 0, 2],
            [8, 8, 1, 1, 2],
            [8, 8, 1, 0, 0]
          ];
      for (const [requirement, retained, certificate, loss, expected] of frames) {
        required.buffer.write(Uint32Array.of(requirement));
        count.buffer.write(Uint32Array.of(retained));
        complete.buffer.write(Uint32Array.of(certificate));
        unresolved.buffer.write(Uint32Array.of(loss));
        submitGraph(device, compiled, undefined);
        expectRows(await readUint32(status.buffer), [expected]);
        if (!expected) expectRows(await readFloat32(chow.buffer), [6.2981105, 1, 0.01208667]);
        else {
          expectNaNCovariance(await readFloat32(covariance.buffer), 3);
          expectRows(await readFloat32(chow.buffer), [NaN, 0, NaN]);
        }
      }
    } finally {
      compiled.destroy();
      context.destroy();
    }
  }
});

it('W5-42 suppresses rank-one two-block HAC joint inference on both forms', async () => {
  const groups = [1, 0, 0, 0, 0, 1, 0, 0];
  const neighbors = groups.flatMap(group =>
    groups.flatMap((other, row) => (other === group ? [row] : []))
  );
  let retained = 0;
  const offsets = Uint32Array.from([
    ...groups.map(group => {
      const begin = retained;
      retained += groups.filter(other => other === group).length;
      return begin;
    }),
    retained
  ]);
  // Two opposite block score sums give a covariance of rank one, despite positive marginals.
  for (const result of await runScene(
    {
      predictors: Float32Array.of(6, 3, 6, 2, 1, 5, 1, 1),
      regimes,
      response: Float32Array.of(0, 8, 6, 4, 8, 8, 7, 1)
    },
    [false, false],
    {
      offsets,
      neighbors: Uint32Array.from(neighbors),
      weights: new Float32Array(neighbors.length).fill(1)
    }
  )) {
    expectRows(result['robustStatus'], [3]);
    expectNaNCovariance(result['robustCovariance'], 4);
    expectRows(result['robustChow'], [NaN, 0, NaN]);
  }
});

it('W5-43 rejects exact-fit residual roundoff and retains a representable nonzero residual', async () => {
  const predictors = Float32Array.of(3, 2, 6, 5, 1, 2, 6, 6);
  for (const perturbation of [0, 2 ** -10]) {
    const observed = Float32Array.from(
      predictors,
      (value, row) => 3 * value + 4 * regimes[row] + (row === 0 ? perturbation : 0)
    );
    for (const kernel of [undefined, identityKernel()]) {
      for (const result of await runScene(
        {predictors, regimes, response: observed},
        [false, false],
        kernel
      )) {
        expectRows(result['robustStatus'], [perturbation ? 0 : 3]);
        if (!perturbation) {
          expectNaNCovariance(result['robustCovariance'], 4);
          expectRows(result['robustChow'], [NaN, 0, NaN]);
        } else {
          expect(Number.isFinite(result['robustChow'][0])).toBe(true);
          expect(result['robustChow'].length).toBe(3);
          expectRows(result['robustChow'].subarray(1), [2, 0]);
          expect(result['robustChow'][2]).toBe(0);
          // Exact rational normal equations and HC0: Wald=22031625132/25.
          expect(Math.abs(result['robustChow'][0] / 881265005.28 - 1)).toBeLessThan(0.03);
        }
      }
    }
  }
});

it('W5-44 accepts the identified 1024-row half-identity HAC contrast', async () => {
  const rowCount = 1024;
  const input = {
    predictors: Float32Array.from({length: rowCount}, (_, row) => (row % 8 < 4 ? -1 : 1)),
    regimes: Uint32Array.from({length: rowCount}, (_, row) => row % 2),
    response: Float32Array.from(
      {length: rowCount},
      (_, row) => (row % 4 < 2 ? 1 : -1) + (row % 2) / 16
    )
  };
  // OLS normal equations annihilate 11': variance=half HC0=1/512, Wald=(1/16)^2/(1/512)=2.
  for (const result of await runScene(input, [false, true], {
    offsets: Uint32Array.from({length: rowCount + 1}, (_, row) => row * rowCount),
    neighbors: Uint32Array.from({length: rowCount ** 2}, (_, slot) => slot % rowCount),
    weights: Float32Array.from({length: rowCount ** 2}, (_, slot) =>
      Math.floor(slot / rowCount) === slot % rowCount ? 1 : 0.5
    )
  })) {
    expectRows(result['robustStatus'], [0]);
    const covariance = result['robustCovariance'];
    const variance = covariance[0] + covariance[4] - 2 * covariance[1];
    expect(Number.isFinite(variance)).toBe(true);
    expect(Math.abs(variance - 0.001953125)).toBeLessThan(1e-7);
    expectRows(result['robustChow'], [2, 1, 0.15729921]);
  }
});

it('W5-45 keeps classical insufficient-row status with and without optional robust inference', async () => {
  for (const robustEnabled of [false, true]) {
    for (const result of await runScene(
      {
        predictors: new Float32Array(4),
        regimes: Uint32Array.of(0, 1, 0, 1),
        response: Float32Array.of(0, 1, 4, 9)
      },
      [false, true],
      undefined,
      1,
      2,
      [GPURestrictedRegimeRegressionOperation, CPURestrictedRegimeRegressionOperation],
      robustEnabled
    )) {
      expectRows(result['status'], [1]);
      if (robustEnabled) {
        expectRows(result['robustStatus'], [1]);
        expectNaNCovariance(result['robustCovariance'], 3);
      }
    }
  }
});

it('W5-46 preserves the identified two-contrast oracle across high-leverage row permutations', async () => {
  const predictors = [20000000, 0, 0, 1, 1, 2, 2, 3];
  for (const order of [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [2, 1, 0, 3, 4, 5, 6, 7]
  ]) {
    for (const result of await runScene(
      {
        predictors: Float32Array.from(order, row => predictors[row]),
        regimes: Uint32Array.from(order, row => regimes[row]),
        response: Float32Array.from(order, row => response[row])
      },
      [false, false]
    )) {
      // Review NumPy SVD oracle, independent of either implementation.
      expectRows(result['robustStatus'], [0]);
      expectRows(result['robustChow'], [8.34675964, 2, 0.01540012]);
    }
  }
});

it('B3 retains the 256-row strongly cancelling identified HAC contrast', async () => {
  const rowCount = 256;
  for (const result of await runScene(
    {
      predictors: Float32Array.from({length: rowCount}, (_, row) => (row % 2 ? 1 : -1)),
      regimes: Uint32Array.from({length: rowCount}, (_, row) => Math.floor(row / 2) % 2),
      response: Float32Array.from(
        {length: rowCount},
        (_, row) => (row % 8 < 4 ? -0.5 : 0.5) + (Math.floor(row / 2) % 2) / 32
      )
    },
    [false, true],
    {
      offsets: Uint32Array.from({length: rowCount + 1}, (_, row) => row * rowCount),
      neighbors: Uint32Array.from({length: rowCount ** 2}, (_, slot) => slot % rowCount),
      weights: Float32Array.from({length: rowCount ** 2}, (_, slot) =>
        Math.floor(slot / rowCount) === slot % rowCount ? 1 : 63 / 64
      )
    }
  )) {
    // X'e=0 leaves HC0/64; contrast variance=1/16384, Wald=(1/32)^2/(1/16384)=16.
    expectRows(result['robustStatus'], [0]);
    const covariance = result['robustCovariance'];
    const variance = covariance[0] + covariance[4] - 2 * covariance[1];
    expect(Number.isFinite(variance)).toBe(true);
    expect(Math.abs(variance - 1 / 16384)).toBeLessThan(1e-8);
    expectRows(result['robustChow'], [16, 1, 0.00006334249]);
  }
});

function expectNaNCovariance(values: ArrayLike<number>, coefficientCount: number) {
  expect(values.length).toBe(coefficientCount * coefficientCount);
  expect(Array.from(values).every(Number.isNaN)).toBe(true);
}
