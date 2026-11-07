// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {runNestedInterpolationValidationLens as run} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/nested-interpolation-validation.run';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
it('nested validation lens executes public examples and captures fitted results', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await run(device);
  expect(result.graphs[0].nodes.length).toBeGreaterThan(10);
  const snapshots = result.captures!.snapshots;
  expect(Array.from(snapshots.find(snapshot => snapshot.name === 'status')!.values!)).toEqual([
    0, 0
  ]);
  expect(
    Array.from(snapshots.find(snapshot => snapshot.name === 'selected')!.values!).every(
      value => value < 2
    )
  ).toBe(true);
});
