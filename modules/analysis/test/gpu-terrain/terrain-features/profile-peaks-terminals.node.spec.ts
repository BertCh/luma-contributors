// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, expectTypeOf, it} from 'vitest';
import {CPUProfilePeaksOperation} from '../../../src/gpu-terrain/terrain-features/cpu-profile-peaks-operation';
import {
  GPUProfilePeaksOperation,
  type ProfilePeaksOperationProps
} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {expectProfileSingletonStatus} from './terrain-terminal-assertions';

const INPUTS = {
  id: 'profile-terminals',
  values: 'values',
  offsets: 'offsets',
  parameters: 'parameters',
  sampleCount: 3,
  window: 1,
  minimumSide: 1
};

it('TSC-02 excludes empty and nested-empty selections from the named output type', () => {
  expectTypeOf<{}>().not.toMatchTypeOf<ProfilePeaksOperationProps['output']>();
  expectTypeOf<{status: {}}>().not.toMatchTypeOf<ProfilePeaksOperationProps['output']>();
});

for (const Operation of [CPUProfilePeaksOperation, GPUProfilePeaksOperation]) {
  it(`${Operation.name} rejects empty and nested-empty JavaScript output selections`, () => {
    for (const output of [{}, {status: {}}]) {
      // Reflect models untyped JavaScript callers without weakening the public props.
      expect(() => Reflect.construct(Operation, [{...INPUTS, output}])).toThrow(
        /profile-terminals.*output/
      );
    }
  });

  it(`${Operation.name} accepts either status leaf alone`, () => {
    const selections: ProfilePeaksOperationProps['output'][] = [
      {status: {converged: 'converged'}},
      {status: {iterationLimitReached: 'iterationLimitReached'}}
    ];
    for (const output of selections) {
      const operation = new Operation({...INPUTS, output});
      const declaredOutputs = operation.contract.outputs;
      if (!declaredOutputs || typeof declaredOutputs === 'function')
        throw new Error('Expected explicit operation contract output declarations');
      expect(Object.keys(declaredOutputs)).toEqual(Object.values(output.status!));
    }
  });
}

for (const terminal of ['converged', 'iterationLimitReached'] as const) {
  for (const maximumSuppressionIterations of [0, 16] as const) {
    it(`CPU profile operation publishes ${terminal} alone at ${maximumSuppressionIterations} rounds`, async () => {
      const output =
        terminal === 'converged'
          ? {status: {converged: 'result'}}
          : {status: {iterationLimitReached: 'result'}};
      const operation = new CPUProfilePeaksOperation({
        ...INPUTS,
        maximumSuppressionIterations,
        output
      });
      const device = createNullWebGPUDevice();
      try {
        const rows = await runOperationGraph(device, {
          inputs: {
            values: {format: 'float32', values: Float32Array.of(0, 1, 0)},
            offsets: {format: 'uint32', values: Uint32Array.of(0, 3)},
            parameters: {format: 'float32', values: Float32Array.of(1)}
          },
          operations: [operation],
          outputs: ['result']
        });
        expectProfileSingletonStatus(rows['result'], terminal, maximumSuppressionIterations);
      } finally {
        device.destroy();
      }
    });
  }
}
