// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import {
  computeEnsembleScoresOnCPU,
  CPUEnsembleScoresOperation
} from '../../../src/gpu-spatial-analysis/scoring-rules/cpu-ensemble-scores-operation';
import {
  GPUEnsembleScoresOperation,
  type EnsembleScoresOperationProps
} from '../../../src/gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores-operation';
import {SCORING_FIXTURE} from './scoring-rules-harness';

// Integral of the squared empirical CDF error; an independent CRPS algorithm.
function integrateCdfError(sample: number[], observed: number): number {
  const edges = [...new Set([...sample, observed])].sort((first, second) => first - second);
  let integral = 0;
  for (let interval = 1; interval < edges.length; interval++) {
    const middle = (edges[interval - 1] + edges[interval]) / 2;
    const error =
      sample.filter(value => value <= middle).length / sample.length - Number(observed <= middle);
    integral += (edges[interval] - edges[interval - 1]) * error * error;
  }
  return integral;
}

const INPUT = {
  ensemble: new Float32Array([0, 0, 3, 0, 0, 4]),
  observations: new Float32Array([0, 0]),
  variogramPairs: new Uint32Array([0, 1, 1, 0, 1, 1]),
  variogramWeights: new Float32Array([2, 0.5, 9])
};
const PROPS: EnsembleScoresOperationProps = {
  id: 'scores-check',
  ensemble: 'ensemble',
  observations: 'observations',
  memberCount: 3,
  dimension: 2,
  variogramPairs: 'pairs',
  variogramWeights: 'weights',
  output: {crps: 'crps', energyScore: 'energy', variogramScore: 'variogram'}
};

it('agrees with CDF integration and the hand-computed 3-4-5 triangle', () => {
  for (const fair of [false, true]) {
    const result = computeEnsembleScoresOnCPU(INPUT, {
      memberCount: 3,
      dimension: 2,
      estimator: fair ? 'fair' : 'empirical',
      variogramOrder: 1
    });
    expect(result.energyScore![0]).toBeCloseTo(fair ? 1 / 3 : 1, 6);
    expect(result.variogramScore![0]).toBeCloseTo(245 / 18, 5);
    expect(result.crps![0]).toBeCloseTo(fair ? 0 : 1 / 3, 6);
    expect(result.crps![1]).toBeCloseTo(fair ? 0 : 4 / 9, 6);
  }
  const ensemble = new Float32Array([-5, 7, 2, -3, 2, 1, 8, -9]);
  const observations = new Float32Array([1, -4]);
  const result = computeEnsembleScoresOnCPU(
    {ensemble, observations},
    {memberCount: 4, dimension: 2, scores: ['crps']}
  );
  for (let site = 0; site < 2; site++)
    expect(result.crps![site]).toBeCloseTo(
      integrateCdfError(
        Array.from({length: 4}, (_, member) => ensemble[member * 2 + site]),
        observations[site]
      ),
      6
    );
  expect(result).not.toHaveProperty('energyScore');
});

it('every CPU score matches the independent Python fixtures for empirical and fair ensembles', () => {
  const differences = {crps: 0, energyScore: 0, variogramScore: 0};
  for (const entry of SCORING_FIXTURE.ensembles) {
    for (const fair of [false, true]) {
      const actual = computeEnsembleScoresOnCPU(
        {
          ensemble: Float32Array.from(entry.ensemble),
          observations: Float32Array.from(entry.observations),
          variogramPairs: Uint32Array.from(entry.variogramPairsAll.flat())
        },
        {...entry, estimator: fair ? 'fair' : 'empirical'}
      );
      const expected = {
        crps: fair ? entry.crpsFair : entry.crps,
        energyScore: [fair ? entry.energyScoreFair : entry.energyScore],
        variogramScore: [entry.variogramScoreAll]
      };
      for (const name of ['crps', 'energyScore', 'variogramScore'] as const) {
        expect(actual[name]!.length, `${name} output length`).toBe(expected[name].length);
        expected[name].forEach((value, index) => {
          expect(Number.isFinite(actual[name]![index])).toBe(true);
          const difference = Math.abs(actual[name]![index] - value);
          expect(difference).toBeLessThanOrEqual(2e-6 * Math.max(1, Math.abs(value)));
          differences[name] = Math.max(differences[name], difference);
        });
      }
    }
  }
  console.info('Scoring audit maximum absolute differences', differences);
});

it('delegates GPU limits to contributor construction and drops that GPU-only cap on CPU', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      ensemble: {format: 'float32', length: 6},
      observations: {format: 'float32', length: 2},
      variogramPairs: {format: 'uint32x2', length: 3},
      variogramWeights: {format: 'float32', length: 3},
      crps: {format: 'float32', length: 2},
      energyScore: {format: 'float32', length: 1},
      variogramScore: {format: 'float32', length: 1}
    });
    const props = {...PROPS, memberCount: 65536};
    expect(() => new GPUEnsembleScoresOperation(props).createContributor(getView)).toThrow(
      /memberCount/
    );
    expect(() => new CPUEnsembleScoresOperation(props)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('handles one member, invalid pair indices, and NaN members', () => {
  const single = computeEnsembleScoresOnCPU(
    {ensemble: new Float32Array([3, 4]), observations: new Float32Array([0, 0])},
    {memberCount: 1, dimension: 2}
  );
  expect(Array.from(single.crps!)).toEqual([3, 4]);
  expect(single.energyScore![0]).toBe(5);
  expect(
    computeEnsembleScoresOnCPU(
      {...INPUT, variogramPairs: new Uint32Array([0, 2])},
      {memberCount: 3, dimension: 2}
    ).variogramScore![0]
  ).toBeNaN();
  const missing = computeEnsembleScoresOnCPU(
    {...INPUT, ensemble: new Float32Array([NaN, 0, 3, 0, 0, 4])},
    {memberCount: 3, dimension: 2}
  );
  expect(missing.crps![0]).toBeNaN();
  expect(missing.crps![1]).toBeCloseTo(4 / 9, 6);
  expect(missing.energyScore![0]).toBeNaN();
  expect(missing.variogramScore![0]).toBeNaN();
});

it('checks arrays and operation construction', () => {
  const options = {memberCount: 3, dimension: 2};
  expect(() =>
    computeEnsembleScoresOnCPU({...INPUT, ensemble: new Float32Array(5)}, options)
  ).toThrow('ensemble');
  expect(() =>
    computeEnsembleScoresOnCPU({...INPUT, observations: new Float32Array(1)}, options)
  ).toThrow('observations');
  expect(() =>
    computeEnsembleScoresOnCPU({...INPUT, variogramPairs: new Uint32Array(3)}, options)
  ).toThrow('variogramPairs');
  expect(() =>
    computeEnsembleScoresOnCPU({...INPUT, variogramWeights: new Float32Array(1)}, options)
  ).toThrow('variogramWeights');
  expect(() => new GPUEnsembleScoresOperation({...PROPS, dimension: 0})).toThrow(/^scores-check/);
  expect(() => new GPUEnsembleScoresOperation({...PROPS, output: {crps: 'ensemble'}})).toThrow(
    /^scores-check/
  );
  const invalidProps = [
    {memberCount: 0},
    {memberCount: 1, estimator: 'fair'},
    {dimension: 0},
    {variogramOrder: 0},
    {variogramOrder: Infinity},
    {output: {}}
  ] satisfies Partial<EnsembleScoresOperationProps>[];
  for (const replacement of invalidProps) {
    expect(() => new CPUEnsembleScoresOperation({...PROPS, ...replacement})).toThrow(
      /^scores-check/
    );
  }
  expect(() => new CPUEnsembleScoresOperation({...PROPS, variogramPairs: undefined})).toThrow(
    /^scores-check.*variogram/
  );
  expect(new GPUEnsembleScoresOperation(PROPS).id).toBe('scores-check');
  expect(new CPUEnsembleScoresOperation(PROPS).id).toBe('scores-check');
});
