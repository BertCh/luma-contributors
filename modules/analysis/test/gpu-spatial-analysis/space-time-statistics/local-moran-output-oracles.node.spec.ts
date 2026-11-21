// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUSpaceTimeLocalMoran} from '../../../src/gpu-spatial-analysis/space-time-statistics/index';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {computeLocalPermutationOracle} from '../permutation-inference/permutation-oracle';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {CYCLE_CSR, TIED_CSR, expectMoranOracle} from './local-moran-output-oracles-fixture';

it('local Moran CPU reference agrees with the tied exhaustive null and independent integer draw oracle', () => {
  for (const cycle of [false, true]) {
    const values = new Float32Array(cycle ? [1, 2, 4, NaN] : [1, 3, 9, NaN]);
    const weights = cycle ? CYCLE_CSR : TIED_CSR;
    const result = computeLocalPermutationOracle({
      weights: {
        offsets: new Uint32Array(weights.offsets),
        neighbors: new Uint32Array(weights.neighbors),
        weights: new Float32Array(weights.weights)
      },
      values,
      secondValues: values,
      statistic: 'localBivariateMoran',
      seed: 11,
      permutationCount: 7,
      significanceLevel: cycle ? 0.75 : 0.05,
      maximumNeighbors: 1,
      alternative: 'greater',
      islands: 'exclude',
      multipleTesting: 'benjamini-yekutieli',
      ...(cycle ? {} : {stopAfterExceedances: 3})
    });
    const {classes, permutationCounts, overflow, islandCount, ...outputs} = result;
    expect(classes).toEqual([]);
    expectMoranOracle(
      {
        ...outputs,
        overflow: [overflow],
        islandCount: [islandCount],
        ...(cycle ? {} : {permutationCounts})
      },
      cycle
    );
  }
});

it('space-time Moran refuses incompatible outputs before graph mutation and permits corrected retries', () => {
  const device = createNullWebGPUDevice();
  const rig = new WeightsRig(device);
  try {
    for (const {variant, port} of [
      {variant: 'lagged', port: 'classes'},
      {variant: 'differential', port: 'classes'},
      {variant: 'differential', port: 'quadrants'}
    ] as const) {
      const props = {
        id: `moran-${variant}-${port}`,
        weights: rig.uploadWeights(TIED_CSR),
        cube: rig.input(new Float32Array([1, 3, 9, NaN, 1, 3, 9, NaN]), 'float32', 8),
        layout: 'period-major' as const,
        periodCount: 2,
        period: 1,
        variant,
        parameters: rig.input(
          getGPUPermutationParameterValues({seed: 11, permutationCount: 7}),
          'uint32',
          4
        ),
        maximumPermutations: 7,
        output: {
          exceedances: rig.output('uint32', 4).view,
          pseudoPValues: rig.output('float32', 4).view,
          [port]: rig.output('uint32', 4).view,
          status: {overflow: rig.output('uint32', 1).view}
        }
      };
      expect(() => new GPUSpaceTimeLocalMoran(props)).toThrow(
        new RegExp(`${props.id}.*output.${port}.*use`)
      );
      // A refused construction must not register scratch resources: same graph and ID can recover.
      const corrected = new GPUSpaceTimeLocalMoran({
        ...props,
        output: {...props.output, [port]: undefined}
      });
      expect(() => corrected.getCommandNodes(rig.graph)).not.toThrow();
      if (port !== 'classes') {
        const supported = new GPUSpaceTimeLocalMoran({
          ...props,
          id: `${props.id}-supported`,
          variant: 'lagged'
        });
        expect(() => supported.getCommandNodes(rig.graph)).not.toThrow();
      }
    }
  } finally {
    rig.destroy();
    device.destroy();
  }
});

it('space-time Moran admits actual permutation counts for fixed and stopped inference', () => {
  const device = createNullWebGPUDevice();
  const rig = new WeightsRig(device);
  try {
    for (const stopping of [false, true]) {
      const counts = rig.output('uint32', 4).view;
      const contributor = new GPUSpaceTimeLocalMoran({
        id: `counts-${stopping}`,
        weights: rig.uploadWeights(TIED_CSR),
        cube: rig.input(new Float32Array([1, 3, 9, NaN, 1, 3, 9, NaN]), 'float32', 8),
        layout: 'period-major',
        periodCount: 2,
        period: 1,
        variant: 'lagged',
        parameters: rig.input(
          getGPUPermutationParameterValues({seed: 11, permutationCount: 7}),
          'uint32',
          4
        ),
        maximumPermutations: 7,
        ...(stopping ? {stopAfterExceedances: 3} : {}),
        output: {
          exceedances: rig.output('uint32', 4).view,
          pseudoPValues: rig.output('float32', 4).view,
          permutationCounts: counts,
          status: {overflow: rig.output('uint32', 1).view}
        }
      });
      expect(() => contributor.getCommandNodes(rig.graph)).not.toThrow();
    }
  } finally {
    rig.destroy();
    device.destroy();
  }
});

it('Moran BY significance at level one-half differs from raw-p significance', () => {
  const values = new Float32Array([1, 2, 4, NaN]);
  const result = computeLocalPermutationOracle({
    weights: {
      offsets: new Uint32Array(CYCLE_CSR.offsets),
      neighbors: new Uint32Array(CYCLE_CSR.neighbors),
      weights: new Float32Array(CYCLE_CSR.weights)
    },
    values,
    secondValues: values,
    statistic: 'localBivariateMoran',
    seed: 11,
    permutationCount: 7,
    significanceLevel: 0.5,
    maximumNeighbors: 1,
    alternative: 'greater',
    islands: 'exclude',
    multipleTesting: 'benjamini-yekutieli'
  });
  // Independent oracle gives raw .125 < .5 < BY .6875: adjustment must control significance.
  expect(result.pseudoPValues[0]).toBe(0.125);
  expect(result.adjustedPValues[0]).toBe(0.6875);
  expectMoranOracle({significant: result.significant}, true, 0.5);
  expect(() => expectMoranOracle({significant: [1, 0, 0, 0]}, true, 0.5)).toThrow();
});
