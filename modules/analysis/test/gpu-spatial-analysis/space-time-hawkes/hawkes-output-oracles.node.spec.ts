// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {CPUSpaceTimeHawkesOperation} from '../../../src/gpu-spatial-analysis/space-time-hawkes/cpu-space-time-hawkes-operation';
import {GPUSpaceTimeHawkesOperation} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes-operation';
import type {GPUSpaceTimeHawkes} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes';
import {constructSamplingContributor} from '../point-process/validation-sampling-operation-test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  HAWKES_INPUTS,
  HAWKES_STATUS_CASES,
  expectHawkesOracle,
  getHawkesProps,
  SEPARATED_HAWKES_INPUTS
} from './hawkes-output-oracles-fixture';

it('CPU Hawkes publishes independently counted excluded events and cutoff diagnostics', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const status of HAWKES_STATUS_CASES) {
      const operation = new CPUSpaceTimeHawkesOperation(getHawkesProps(status));
      expectHawkesOracle(
        await runOperationGraph(device, {
          inputs: HAWKES_INPUTS,
          operations: [operation],
          outputs: ['estimates', ...Object.values(status)]
        }),
        false,
        Object.values(status)
      );
    }
  } finally {
    device.destroy();
  }
});

it('Hawkes retains a requested status binding after caller deletion', () => {
  const status: {candidateOverflow?: string} = {candidateOverflow: 'candidateOverflow'};
  const operation = new GPUSpaceTimeHawkesOperation(getHawkesProps(status));
  delete status.candidateOverflow;
  const factory = operation.createContributor.bind(operation);
  let retained = false;
  operation.createContributor = getView => {
    const contributor = factory(getView) as GPUSpaceTimeHawkes;
    retained = Boolean(contributor.props.output.status?.candidateOverflow);
    return contributor;
  };
  constructSamplingContributor(operation, {
    positions: {format: 'float32x2', length: 4},
    times: {format: 'float32', length: 4},
    parameters: {format: 'float32', length: 9}
  });
  expect(retained).toBe(true);
});

it('Hawkes captures a prototype getter diagnostic declaration', async () => {
  const status = Object.create({
    get candidateOverflow() {
      return 'candidateOverflow';
    }
  });
  const operation = new GPUSpaceTimeHawkesOperation(getHawkesProps(status));
  expect(
    Object.values(operation.outputs)
      .map(output => output.value)
      .sort()
  ).toEqual(['candidateOverflow', 'estimates']);
  const device = createNullWebGPUDevice();
  try {
    expectHawkesOracle(
      await runOperationGraph(device, {
        inputs: HAWKES_INPUTS,
        operations: [new CPUSpaceTimeHawkesOperation(getHawkesProps(status))],
        outputs: ['estimates', 'candidateOverflow']
      }),
      false,
      ['candidateOverflow']
    );
  } finally {
    device.destroy();
  }
});

it('CPU Hawkes distinguishes candidate overflow, invalid count and approximation', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const status of HAWKES_STATUS_CASES) {
      const result = await runOperationGraph(device, {
        inputs: SEPARATED_HAWKES_INPUTS,
        operations: [new CPUSpaceTimeHawkesOperation(getHawkesProps(status))],
        outputs: ['estimates', ...Object.values(status)]
      });
      expectHawkesOracle(result, false, Object.values(status), true);
      if (result['candidateOverflow'] && result['invalidCount'])
        expect(() =>
          expectHawkesOracle(
            {
              ...result,
              candidateOverflow: result['invalidCount'],
              invalidCount: result['candidateOverflow']
            },
            false,
            Object.values(status),
            true
          )
        ).toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('Hawkes oracle refuses missing requested diagnostic outputs', () => {
  expect(() =>
    expectHawkesOracle({estimates: new Float32Array([0.75, 1e-30, 1, 1])}, true, [
      'candidateOverflow',
      'invalidCount',
      'approximation'
    ])
  ).toThrow();
});
