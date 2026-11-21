// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeProfilePeaksOnCPU,
  CPUProfilePeaksOperation
} from '../../../src/gpu-terrain/terrain-features/cpu-profile-peaks-operation';
import {
  PROFILE_STATUS_PORTS,
  PROFILE_STATUS_VALUES,
  PROFILE_STATUS_CASES,
  PROFILE_EMPTY_STATUS,
  createProfileStatusProps
} from './profile-peaks-status-oracles-fixtures';

for (const testCase of PROFILE_STATUS_CASES) {
  it(`CPU profile status has an independent two-peak oracle: ${testCase.name}`, async () => {
    for (const hasCandidates of [true, false]) {
      const validity = new Uint32Array(10).fill(Number(hasCandidates));
      const inputs = {
        values: new Float32Array(PROFILE_STATUS_VALUES),
        validity,
        offsets: new Uint32Array([0, 10]),
        parameters: new Float32Array([1])
      };
      const expected = hasCandidates ? testCase.expected : PROFILE_EMPTY_STATUS;
      const result = computeProfilePeaksOnCPU(
        inputs.values,
        inputs.offsets,
        inputs.parameters,
        {
          window: 2,
          minimumSide: 1,
          suppressionRadius: 0,
          capacity: testCase.capacity,
          maximumSuppressionIterations: testCase.maximumSuppressionIterations
        },
        validity
      );
      expect(PROFILE_STATUS_PORTS.map(port => Array.from(result[port]))).toEqual(
        expected.map(value => [value])
      );
      const operation = new CPUProfilePeaksOperation(createProfileStatusProps(testCase));
      const output = await operation.execute({
        inputs: Object.fromEntries(
          Object.entries(inputs).map(([port, values]) => [
            port,
            {
              format:
                port === 'values' || port === 'parameters'
                  ? ('float32' as const)
                  : ('uint32' as const),
              length: values.length,
              chunks: [{values, length: values.length}],
              pack: () => values
            }
          ])
        )
      });
      expect(PROFILE_STATUS_PORTS.map(port => Array.from(output[port] as Uint32Array))).toEqual(
        expected.map(value => [value])
      );
    }
  });
}
