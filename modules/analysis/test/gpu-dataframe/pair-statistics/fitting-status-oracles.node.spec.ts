// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeVariogramModelFitOnCPU,
  CPUVariogramModelFitOperation
} from '../../../src/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation';
import {
  computeCoregionalizationFitOnCPU,
  CPUCoregionalizationFitOperation
} from '../../../src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation';
import {getGPUCoregionalizationFitParameterValues} from '../../../src/dataframe-analysis/pair-statistics/coregionalization-parameters';

function createRows(values: Float32Array | Uint32Array) {
  return {
    format: values instanceof Uint32Array ? ('uint32' as const) : ('float32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  };
}

it('CPU variogram status follows the independent bracket budget and missing-bin oracles', async () => {
  const distances = Float32Array.from({length: 32}, (_, index) => (index + 1) / 8);
  const semivariances = Float32Array.from(
    distances,
    lag => 0.2 + 1.5 * (1 - Math.exp((-3 * lag) / 2.2))
  );
  // W=2*log(960)/199, contracted by phi per step: budgets 0 and 1
  // leave width >1e-5; 32 steps leave width <1.5e-8. Known optimum 2.2
  // is interior to [1/80,12]. These literals do not come from the CPU fit.
  for (const maximumIterations of [0, 1, 32]) {
    const options = {
      models: ['exponential'] as const,
      weighting: 'pairs' as const,
      rangeSteps: 200,
      maximumIterations
    };
    const operation = new CPUVariogramModelFitOperation({
      ...options,
      distances: 'distances',
      semivariances: 'semivariances',
      pairCounts: 'pairCounts',
      output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
    });
    for (const available of [true, false, true]) {
      const pairCounts = new Uint32Array(32).fill(available ? 10 : 0);
      const result = computeVariogramModelFitOnCPU({distances, semivariances, pairCounts}, options);
      expect(Array.from(result.status)).toEqual([available ? (maximumIterations < 32 ? 3 : 0) : 1]);
      const rows = (await operation.execute({
        inputs: {
          distances: createRows(distances),
          semivariances: createRows(semivariances),
          pairCounts: createRows(pairCounts)
        }
      })) as Record<string, Uint32Array>;
      expect(Array.from(rows['status'])).toEqual([
        available ? (maximumIterations < 32 ? 3 : 0) : 1
      ]);
      if (available) {
        expect(Number.isFinite(result.diagnostics[3])).toBe(true);
        if (maximumIterations < 32) expect(result.diagnostics[3]).toBeGreaterThan(1e-5);
        else expect(result.diagnostics[3]).toBeLessThanOrEqual(1e-5);
      }
    }
  }
});

it('CPU coregionalization statuses follow the independent two-sweep PSD fixed-point and invalid-bin oracles', async () => {
  // At unit spherical lag shape=1. B=[[2,-1],[-1,3]] is PSD with
  // eigenvalues (5±sqrt(5))/2: sweep 1 reaches B, sweep 2 measures zero
  // change. One sweep has measured change 3>tolerance*3; zero has no fit.
  for (const maximumIterations of [0, 1, 4]) {
    const expected =
      maximumIterations === 0
        ? [1, 0, 0, 1, 0]
        : maximumIterations === 1
          ? [1, 1, 0, 1, 0]
          : [1, 2, 1, 0, 0];
    const operation = new CPUCoregionalizationFitOperation({
      variableCount: 2,
      structureCount: 1,
      maximumIterations,
      distances: 'distances',
      semivariances: 'semivariances',
      pairCounts: 'pairCounts',
      parameters: 'parameters',
      output: {
        matrices: 'matrices',
        loadings: 'loadings',
        diagnostics: 'diagnostics',
        status: {
          validity: 'validity',
          iterationCount: 'iterationCount',
          converged: 'converged',
          iterationLimitReached: 'iterationLimitReached',
          invalidCount: 'invalidCount'
        }
      }
    });
    for (const available of [true, false, true]) {
      const input = {
        distances: new Float32Array([1]),
        semivariances: new Float32Array(available ? [2, -1, 3] : [2, NaN, 3]),
        pairCounts: new Uint32Array([10]),
        parameters: getGPUCoregionalizationFitParameterValues([
          {model: 'spherical', range: 1, sill: 1}
        ])
      };
      const result = computeCoregionalizationFitOnCPU(input, {
        variableCount: 2,
        structureCount: 1,
        maximumIterations
      });
      expect([
        result.validity[0],
        result.iterationCount[0],
        result.converged[0],
        result.iterationLimitReached[0],
        result.invalidCount[0]
      ]).toEqual(available ? expected : [0, 0, 0, 0, 1]);
      const rows = (await operation.execute({
        inputs: {
          distances: createRows(input.distances),
          semivariances: createRows(input.semivariances),
          pairCounts: createRows(input.pairCounts),
          parameters: createRows(input.parameters)
        }
      })) as Record<string, Uint32Array>;
      expect(
        ['validity', 'iterationCount', 'converged', 'iterationLimitReached', 'invalidCount'].map(
          name => rows[name][0]
        )
      ).toEqual(available ? expected : [0, 0, 0, 0, 1]);
    }
  }
});

it('R6 coregionalization accepts the inclusive relative-change tolerance boundary one', () => {
  // First sweep changes zero to B; max absolute change / largest coefficient
  // is exactly 3/3=1. The inclusive test 1<=tolerance converges at tolerance 1.
  const result = computeCoregionalizationFitOnCPU(
    {
      distances: new Float32Array([1]),
      semivariances: new Float32Array([2, -1, 3]),
      pairCounts: new Uint32Array([10]),
      parameters: getGPUCoregionalizationFitParameterValues(
        [{model: 'spherical', range: 1, sill: 1}],
        1
      )
    },
    {variableCount: 2, structureCount: 1, maximumIterations: 1}
  );
  expect([result.iterationCount[0], result.converged[0], result.iterationLimitReached[0]]).toEqual([
    1, 1, 0
  ]);
});
