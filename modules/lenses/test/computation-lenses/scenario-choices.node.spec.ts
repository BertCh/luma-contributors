// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {getLensScenario} from '../../src/computation-lenses/registry-implementation';
import {runS2Lens} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s2-lens-run';
import {declareLensScenario} from './scenario-declaration.test-utils';
import '../../src/computation-lenses/scenarios/index';

test('missing S2 choice fails before building or submitting the graph', async () => {
  const device = createNullWebGPUDevice();
  const build = vi.fn(() => ({}));
  try {
    await expect(
      runS2Lens(device, {
        scenarioId: 'gpu-spatial-analysis.point-density',
        element: 'GPUPointDensity',
        datasets: ['fixture'],
        choices: {gridSize: '1 x 1', binning: 'grid', statistic: 'count'},
        build
      })
    ).rejects.toThrow(/gpu-spatial-analysis.point-density.*choices.sumAccumulation/);
    expect(build).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

test.each([
  ['spatial-rate-recipe', {islandPolicy: 'undefined', scale: 1000}],
  ['quantile-lisa', {quantileParameters: 'quantile 4 of 4', permutationCount: 199}],
  [
    'point-pattern-envelope',
    {statistic: 'integrated-squared-deviation', simulationCount: 3, fittedParameters: 'none'}
  ],
  [
    'multiscale-geographically-weighted-regression',
    {'bandwidthSelection.criterion': 'disabled', 'bandwidthSelection.searchMethod': 'disabled'}
  ]
] as const)('%s records the actual example settings', async (name, expected) => {
  const scenario = getLensScenario(`gpu-spatial-analysis.${name}`)!;
  const declared = await declareLensScenario(scenario);
  const choices = Object.fromEntries(
    declared.record.choices.map(choice => [choice.name, choice.value])
  );
  expect(choices).toMatchObject(expected);
});
