// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUSpaceTimeHawkesOperation} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes-operation';
import {requireTestDevice} from '../../utils/require-test-device';
import {runPoisonedOperation} from '../point-process/output-oracle-harness';
import {
  HAWKES_INPUTS,
  HAWKES_STATUS_CASES,
  expectHawkesOracle,
  getHawkesProps,
  SEPARATED_HAWKES_INPUTS
} from './hawkes-output-oracles-fixture';

it('GPU Hawkes writes a getter-backed overflow diagnostic against an explicit request', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const status = Object.create({
    get candidateOverflow() {
      return 'candidateOverflow';
    }
  });
  expectHawkesOracle(
    await runPoisonedOperation(
      device,
      new GPUSpaceTimeHawkesOperation(getHawkesProps(status)),
      HAWKES_INPUTS,
      ['estimates', 'candidateOverflow']
    ),
    true,
    ['candidateOverflow']
  );
});

it('GPU Hawkes preserves the requested diagnostic after caller deletion', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const status: {candidateOverflow?: string} = {candidateOverflow: 'candidateOverflow'};
  const operation = new GPUSpaceTimeHawkesOperation(getHawkesProps(status));
  delete status.candidateOverflow;
  expectHawkesOracle(
    await runPoisonedOperation(device, operation, HAWKES_INPUTS, [
      'estimates',
      'candidateOverflow'
    ]),
    true,
    ['candidateOverflow']
  );
});

it('GPU Hawkes publishes independently counted parent cuts and excluded events with each status alone and omitted', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const status of HAWKES_STATUS_CASES)
    expectHawkesOracle(
      await runPoisonedOperation(
        device,
        new GPUSpaceTimeHawkesOperation(getHawkesProps(status)),
        HAWKES_INPUTS,
        ['estimates', ...Object.values(status)]
      ),
      true,
      Object.values(status)
    );
});

it('GPU Hawkes distinguishes candidate overflow, invalid count and approximation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const status of HAWKES_STATUS_CASES)
    expectHawkesOracle(
      await runPoisonedOperation(
        device,
        new GPUSpaceTimeHawkesOperation(getHawkesProps(status)),
        SEPARATED_HAWKES_INPUTS,
        ['estimates', ...Object.values(status)]
      ),
      true,
      Object.values(status),
      true
    );
});
