// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getPointProcessLensChoices} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/point-process.lens';
import {getSpatialOperatorEntry} from '../../src/gpu-spatial-analysis/computation-pack/pack';
import {createSpatialChoice} from '../../src/gpu-spatial-analysis/computation-pack/choices';
import {getLensScenario} from '../../src/computation-lenses/registry-implementation';
import {declareLensScenario} from './scenario-declaration.test-utils';

test.each([
  'dbscan',
  'xi'
] as const)('X6 %s receipts distinguish inactive extraction settings', extractionMethod => {
  const values = getPointProcessLensChoices('GPUOPTICS', {
    extractionMethod,
    extractionDistance: 1.5,
    xi: 0.05,
    minimumClusterSize: 2
  });
  const choices = getSpatialOperatorEntry('GPUOPTICS')!.choices!.map(template =>
    createSpatialChoice(template, values[template.name])
  );
  for (const name of ['extractionDistance', 'xi', 'minimumClusterSize']) {
    const choice = choices.find(choice => choice.name === name)!;
    const active =
      extractionMethod === 'dbscan' ? name === 'extractionDistance' : name !== 'extractionDistance';
    expect(choice.sweepable).toBe(active);
    expect(typeof choice.value === 'string' && choice.value.startsWith('disabled:')).toBe(!active);
    if (active)
      expect(choice.value).toBe(name === 'extractionDistance' ? 1.5 : name === 'xi' ? 0.05 : 2);
  }
});

test.each([
  'integrated-squared-deviation',
  'extreme-rank'
] as const)('X6 %s receipts expose only effective significance settings', statistic => {
  const values = getPointProcessLensChoices('GPUPointPatternEnvelope', {
    statistic,
    rank: 1,
    significanceLevel: 0.05
  });
  const template = getSpatialOperatorEntry('GPUPointPatternEnvelope')!.choices!.find(
    choice => choice.name === 'significanceLevel'
  )!;
  const choice = createSpatialChoice(template, values['significanceLevel']);
  expect(choice.sweepable).toBe(statistic === 'extreme-rank');
  expect(choice.value).toBe(
    statistic === 'extreme-rank' ? 0.05 : 'disabled: integrated-squared-deviation'
  );
  const rank = getSpatialOperatorEntry('GPUPointPatternEnvelope')!.choices!.find(
    choice => choice.name === 'rank'
  )!;
  expect(createSpatialChoice(rank, values['rank'])).toMatchObject({
    value: statistic === 'extreme-rank' ? 'disabled: extreme-rank envelope' : 1,
    sweepable: statistic !== 'extreme-rank'
  });
});

test.each([
  ['inhomogeneous-point-pattern', ['bandwidth']],
  ['point-pattern-envelope', ['significanceLevel']],
  ['optics', ['xi', 'minimumClusterSize']]
] as const)('X6 live %s receipts mark branch-inapplicable settings non-sweepable', async (name, disabledNames) => {
  const declared = await declareLensScenario(getLensScenario(`gpu-spatial-analysis.${name}`)!);
  for (const name of disabledNames)
    expect(declared.record.choices.find(choice => choice.name === name)).toMatchObject({
      value: expect.stringMatching(/^disabled:/),
      sweepable: false
    });
});

test('X6 OPTICS receipt matches the executed default frontier batch size', async () => {
  const declared = await declareLensScenario(getLensScenario('gpu-spatial-analysis.optics')!);
  expect(declared.record.choices.find(choice => choice.name === 'frontierBatchSize')?.value).toBe(
    64
  );
});
