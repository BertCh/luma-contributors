// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUPointPatternEnvelopeOperation} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope-operation';
import {requireTestDevice} from '../../utils/require-test-device';
import {runPoisonedOperation} from './output-oracle-harness';
import {
  ENVELOPE_INPUTS,
  getHierarchicalRankEnvelopeCase,
  expectHierarchicalRankEnvelopeOracle,
  expectEnvelopeOracle,
  getEnvelopeProps,
  getTiedEnvelopeProps,
  TIED_ENVELOPE_INPUTS,
  expectTiedEnvelopeOracle,
  getHalfRankEnvelopeCase,
  expectHalfRankEnvelopeOracle
} from './envelope-output-oracles-fixture';

it('GPU rank envelope reduces 300 hand-ranked curves and propagates invalidation past row256', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const invalid of [false, true]) {
    const {props, inputs} = getHierarchicalRankEnvelopeCase(invalid);
    expectHierarchicalRankEnvelopeOracle(
      await runPoisonedOperation(device, new GPUPointPatternEnvelopeOperation(props), inputs, [
        'lower',
        'upper',
        'statistics',
        'status',
        'interval'
      ]),
      invalid
    );
  }
});

it('GPU extreme-rank cutoff separates attainable half-integer tie groups', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const defaultLevel of [false, true]) {
    const {props, inputs} = getHalfRankEnvelopeCase(defaultLevel);
    expectHalfRankEnvelopeOracle(
      await runPoisonedOperation(device, new GPUPointPatternEnvelopeOperation(props), inputs, [
        'lower',
        'upper',
        'statistics',
        'status',
        'interval'
      ]),
      defaultLevel
    );
  }
});

it('GPU extreme-rank bounds exclude the entire rejected tie group', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const withInterval of [true, false])
    expectTiedEnvelopeOracle(
      await runPoisonedOperation(
        device,
        new GPUPointPatternEnvelopeOperation(getTiedEnvelopeProps(withInterval)),
        TIED_ENVELOPE_INPUTS,
        Object.values(getTiedEnvelopeProps(withInterval).output) as string[]
      ),
      withInterval
    );
});

it('GPU envelope preserves maximum deviation after caller props reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = getEnvelopeProps('maximum-deviation');
  const requested = Object.values(props.output) as string[];
  const operation = new GPUPointPatternEnvelopeOperation(props);
  props.statistic = 'integrated-squared-deviation';
  props.output.statistics = 'replacement';
  props.output.pValueInterval = 'late-interval';
  expectEnvelopeOracle(
    await runPoisonedOperation(device, operation, ENVELOPE_INPUTS, requested),
    'maximum-deviation'
  );
});

for (const statistic of [
  'maximum-deviation',
  'integrated-squared-deviation',
  'extreme-rank'
] as const)
  for (const withInterval of statistic === 'extreme-rank' ? [true, false] : [false])
    it(`GPU ${statistic} interval=${withInterval} matches hand-sorted envelopes and inference`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const invalid of [false, true]) {
        const inputs = {
          ...ENVELOPE_INPUTS,
          observed: {
            ...ENVELOPE_INPUTS.observed,
            values: new Float32Array(invalid ? [NaN, 1, -1] : [2, 1, -1])
          }
        };
        expectEnvelopeOracle(
          await runPoisonedOperation(
            device,
            new GPUPointPatternEnvelopeOperation(getEnvelopeProps(statistic, withInterval)),
            inputs,
            Object.values(getEnvelopeProps(statistic, withInterval).output) as string[]
          ),
          statistic,
          invalid,
          withInterval
        );
      }
    });
