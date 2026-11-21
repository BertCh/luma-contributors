// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUEnsembleScores,
  GPUGaussianPredictiveScores
} from '../../../src/gpu-spatial-analysis/scoring-rules/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {createRandom} from '../trajectory-interpolation/trajectory-interpolation-fixtures';
import {SCORING_FIXTURE} from './scoring-rules-harness';

/**
 * Sample CRPS from the sorted-member identity, a third formulation next to the Python double sum
 * and the Python integral: with x sorted, sum_ij |x_i - x_j| = 2 sum_k (2k - m + 1) x_k (k from 0).
 */
function computeSortedCrps(members: readonly number[], observation: number, fair: boolean) {
  const sorted = [...members].sort((first, second) => first - second);
  const count = sorted.length;
  const pairSum = 2 * sorted.reduce((sum, value, k) => sum + (2 * k - count + 1) * value, 0);
  const deviation = sorted.reduce((sum, value) => sum + Math.abs(value - observation), 0) / count;
  return deviation - pairSum / (2 * count * (fair ? count - 1 : count));
}

it('the fixture matches an independent sorted-member CRPS and the hand values', () => {
  for (const entry of SCORING_FIXTURE.ensembles) {
    for (let site = 0; site < entry.dimension; site++) {
      const members = Array.from(
        {length: entry.memberCount},
        (_, member) => entry.ensemble[member * entry.dimension + site]
      );
      expect(computeSortedCrps(members, entry.observations[site], false), entry.name).toBeCloseTo(
        entry.crps[site],
        9
      );
      expect(computeSortedCrps(members, entry.observations[site], true), entry.name).toBeCloseTo(
        entry.crpsFair[site],
        9
      );
    }
  }
  const {hand} = SCORING_FIXTURE;
  expect(hand.crps).toBe(5 / 9);
  expect(hand.crpsFair).toBe(1 / 3);
  const centre = SCORING_FIXTURE.gaussian[0];
  expect(centre.crps).toBeCloseTo((Math.SQRT2 - 1) / Math.sqrt(Math.PI), 14);
  // A point forecast scores the absolute error.
  const point = SCORING_FIXTURE.gaussian.find(entry => entry.variance === 0)!;
  expect(point.crps).toBe(Math.abs(point.observation - point.mean));
});

it('the CRPS is proper: the true forecast beats a shifted and a too-wide one', () => {
  const random = createRandom(11);
  const normal = () => {
    const [first, second] = [random() || 1e-12, random()];
    return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
  };
  const meanScore = (shift: number, scale: number) => {
    let total = 0;
    const trials = 600;
    for (let trial = 0; trial < trials; trial++) {
      const observation = normal();
      const members = Array.from({length: 30}, () => shift + scale * normal());
      total += computeSortedCrps(members, observation, true);
    }
    return total / trials;
  };
  const truth = meanScore(0, 1);
  expect(meanScore(0.7, 1)).toBeGreaterThan(truth + 0.02);
  expect(meanScore(0, 2)).toBeGreaterThan(truth + 0.02);
  // The expected CRPS of a calibrated N(0, 1) forecast is 1 / sqrt(pi) = 0.5642 (half the mean absolute
  // difference of two draws); the fair score is unbiased for it at any member count.
  expect(truth).toBeGreaterThan(0.52);
  expect(truth).toBeLessThan(0.61);
});

function createViews(graph: GPUCommandGraph) {
  const view = <Format extends 'float32' | 'uint32x2' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${Math.random()}`, format, length);
  return view;
}

it('GPUEnsembleScores kernels fit the core limits at large sizes and use no atomics', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = createViews(graph);
  const nodes = new GPUEnsembleScores({
    ensemble: view('float32', 512 * 8192),
    observations: view('float32', 8192),
    memberCount: 512,
    dimension: 8192,
    estimator: 'fair',
    variogramPairs: view('uint32x2', 100000),
    variogramWeights: view('float32', 100000),
    output: {
      crps: view('float32', 8192),
      energyScore: view('float32', 1),
      variogramScore: view('float32', 1)
    }
  }).getCommandNodes(graph);
  const sources = getParsedKernelSources(nodes);
  for (const id of [
    'ensemble-scores-crps',
    'ensemble-scores-energy-terms',
    'ensemble-scores-variogram-terms'
  ]) {
    for (const kernel of sources.getConfigurations(id)) {
      expect(getStorageBindingCount(kernel.source), id).toBeLessThanOrEqual(8);
      expect(getWorkgroupMemoryBytes(kernel.source, kernel.constants), id).toBeLessThanOrEqual(
        16384
      );
    }
  }
  for (const [id, source] of sources) {
    expect(source, id).not.toMatch(/atomic/);
  }
  const gaussian = new GPUGaussianPredictiveScores({
    means: view('float32', 1000),
    variances: view('float32', 1000),
    observations: view('float32', 1000),
    output: {crps: view('float32', 1000)}
  }).getCommandNodes(graph);
  for (const [id, source] of getParsedKernelSources(gaussian)) {
    expect(source, id).not.toMatch(/atomic/);
  }
});

it('GPUEnsembleScores validates its props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = createViews(graph);
  const props = {
    ensemble: view('float32', 12),
    observations: view('float32', 3),
    memberCount: 4,
    dimension: 3,
    output: {crps: view('float32', 3)}
  };
  expect(() => new GPUEnsembleScores(props)).not.toThrow();
  expect(() => new GPUEnsembleScores({...props, ensemble: view('float32', 11)})).toThrow(
    /ensemble/
  );
  expect(() => new GPUEnsembleScores({...props, memberCount: 1, estimator: 'fair'})).toThrow(
    /memberCount/
  );
  expect(() => new GPUEnsembleScores({...props, output: {}})).toThrow(/needs crps/);
  expect(
    () => new GPUEnsembleScores({...props, output: {variogramScore: view('float32', 1)}})
  ).toThrow(/go together/);
  expect(() => new GPUEnsembleScores({...props, variogramOrder: 0})).toThrow(/variogramOrder/);
  expect(
    () => new GPUEnsembleScores({...props, output: {energyScore: view('float32', 2)}})
  ).toThrow(/energyScore/);
  expect(
    () =>
      new GPUGaussianPredictiveScores({
        means: view('float32', 4),
        variances: view('float32', 3),
        observations: view('float32', 4),
        output: {crps: view('float32', 4)}
      })
  ).toThrow(/one row per mean/);
});
