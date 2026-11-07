// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {runInterpolationLens} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/interpolation.run';
import {run as runVariogramFitLens} from '../../src/computation-lenses/scenarios/gpu-dataframe/variogram-model-fit.run';
import {requireTestDevice} from '@luma-contributors/graph/testing';

for (const kind of [
  'local-kriging',
  'local-rbf',
  'natural-neighbor',
  'conditional-gaussian-simulation'
] as const) {
  it(`${kind} executes the public example and captures graph plus real GPU results`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runInterpolationLens(device, kind);
    expect(result.graphs.length).toBe(1);
    expect(result.graphs[0].nodes.length).toBeGreaterThanOrEqual(2);
    expect(result.captures?.runId).toBe(result.runId);
    const snapshots = result.captures!.snapshots;
    const statuses = snapshots.find(snapshot => snapshot.name === 'statuses')!;
    expect(Array.from(statuses.values!).every(value => value === 0)).toBe(true);
    const predictions = Array.from(
      snapshots.find(snapshot => snapshot.name === 'predictions')!.values!
    );
    expect(predictions).toHaveLength(4);
    expect(predictions.every(Number.isFinite)).toBe(true);
    if (kind !== 'conditional-gaussian-simulation') {
      const queries = [0.8, 1.1, 1.1, 0.4, 0.6, 0.7, 1.7, 1.6];
      for (let row = 0; row < 4; row++) {
        expect(predictions[row]).toBeCloseTo(
          2 + 0.5 * queries[row * 2] - 0.25 * queries[row * 2 + 1],
          4
        );
      }
    } else {
      expect(snapshots.find(snapshot => snapshot.name === 'values')!.values!.length).toBe(64);
      expect(snapshots.find(snapshot => snapshot.name === 'moments')!.values!.length).toBe(8);
    }
  });
}

it('variogram fitting lens captures GPU fits and convergence diagnostics for all model candidates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runVariogramFitLens(device);
  expect(result.graphs[0].nodes.length).toBeGreaterThan(0);
  const snapshots = result.captures!.snapshots;
  const parameters = snapshots.find(snapshot => snapshot.name === 'parameters')!.values!;
  expect(parameters.length).toBe(12);
  expect(parameters[4]).toBeCloseTo(0.2, 3);
  expect(parameters[5]).toBeCloseTo(1.5, 3);
  expect(parameters[6]).toBeCloseTo(2.2, 3);
  expect(
    Array.from(snapshots.find(snapshot => snapshot.name === 'statuses')!.values!).every(
      value => value === 0 || value === 3
    )
  ).toBe(true);
  expect(snapshots.find(snapshot => snapshot.name === 'diagnostics')!.values!.length).toBe(12);
});
