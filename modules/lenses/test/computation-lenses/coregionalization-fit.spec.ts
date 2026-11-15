// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {expect, it} from 'vitest';
import {run} from '../../src/computation-lenses/scenarios/gpu-dataframe/coregionalization-fit.run';

it('captures the hand-derived signed PSD fit, factors and excluded-bin diagnostics from the GPU', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await run(device);
  expect(result.graphs).toHaveLength(1);
  expect(result.graphs[0].nodes.length).toBeGreaterThan(0);
  expect(result.captures?.runId).toBe(result.runId);
  const snapshots = result.captures!.snapshots;
  const getValues = (name: string) => {
    const snapshot = snapshots.find(capture => capture.name === name)!;
    expect(snapshot.provenance).toBe('gpu-readback');
    return Array.from(snapshot.values!);
  };
  expect(getValues('distances')).toEqual([0.5, 1, 2, 3]);
  expect(getValues('pairCounts')).toEqual([10, 20, 30, 0]);
  expect(getValues('semivariances')).toEqual([
    47 / 64,
    -47 / 128,
    141 / 128,
    11 / 8,
    -11 / 16,
    33 / 16,
    2,
    -1,
    3,
    100,
    100,
    100
  ]);
  const expected = [2, -1, -1, 3];
  const matrices = getValues('matrices');
  expect(matrices).toHaveLength(4);
  matrices.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 4));
  const loadings = getValues('loadings');
  expect(loadings).toHaveLength(4);
  // Eigenvector signs/bases are arbitrary. Check L L^T against the independent known B.
  for (let row = 0; row < 2; row++) {
    for (let column = 0; column < 2; column++) {
      const reconstructed =
        loadings[row * 2] * loadings[column * 2] + loadings[row * 2 + 1] * loadings[column * 2 + 1];
      expect(reconstructed).toBeCloseTo(expected[row * 2 + column], 4);
    }
  }
  const diagnostics = getValues('diagnostics');
  expect(diagnostics).toHaveLength(4);
  expect(diagnostics[0]).toBeCloseTo(0, 5);
  expect(diagnostics[1]).toBeLessThanOrEqual(1e-5);
  expect(diagnostics.slice(2)).toEqual([3, 60]);
  expect(getValues('validity')).toEqual([1]);
  expect(getValues('iterationCount')).toEqual([2]);
  expect(getValues('converged')).toEqual([1]);
  expect(getValues('iterationLimitReached')).toEqual([0]);
  expect(getValues('invalidCount')).toEqual([1]);
  expect(
    result.refs.formulas.some(formula => formula.id === 'gpu-dataframe:GPUCoregionalizationFit')
  ).toBe(true);
});
