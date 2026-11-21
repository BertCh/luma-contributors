// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPURadiationFlowModelOperation} from '../../../src/gpu-network/flow-aggregation/cpu-radiation-flow-model-operation';
import {
  GPURadiationFlowModelOperation,
  type RadiationFlowModelOperationProps
} from '../../../src/gpu-network/flow-aggregation/gpu-radiation-flow-model-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

it('radiation twins preserve tiny distinct great-circle separations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: RadiationFlowModelOperationProps = {
    zoneCount: 3,
    positions: 'positions',
    populations: 'populations',
    outflows: 'outflows',
    normalization: 'none',
    spatialContext: {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters'
    },
    output: {flows: 'flows'}
  };
  const inputs = {
    positions: {
      format: 'float32x2' as const,
      values: new Float32Array([0, 0, 2 ** -48, 0, 2 ** -47, 0])
    },
    populations: {format: 'float32' as const, values: new Float32Array([1, 1, 1])},
    outflows: {format: 'float32' as const, values: new Float32Array([1, 1, 1])}
  };
  // End origins have one intervening population before the far destination; the middle
  // origin's destinations tie. This checks the GPU result as well as twin agreement.
  const expected = new Float32Array([0, 1 / 2, 1 / 6, 1 / 2, 0, 1 / 2, 1 / 6, 1 / 2, 0]);
  const results = [];
  for (const Operation of [GPURadiationFlowModelOperation, CPURadiationFlowModelOperation]) {
    const rows = await runOperationGraph(device, {
      inputs,
      operations: [new Operation(props)],
      outputs: ['flows']
    });
    results.push(rows['flows']);
    // Absolute tolerance 1e-5 covers float32 shader evaluation; every flow is at most 0.5.
    for (let index = 0; index < expected.length; index++)
      expect(Math.abs(rows['flows'][index] - expected[index])).toBeLessThanOrEqual(1e-5);
  }
  for (let index = 0; index < expected.length; index++)
    expect(Math.abs(results[0][index] - results[1][index])).toBeLessThanOrEqual(1e-5);
});

for (const normalization of ['none', 'finite-size'] as const) {
  for (const fixture of ['ordinary', 'tied', 'zero-population', 'great-circle'] as const) {
    it(`radiation twins agree: ${normalization}, ${fixture}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const props: RadiationFlowModelOperationProps = {
        zoneCount: 3,
        positions: 'positions',
        populations: 'populations',
        outflows: 'outflows',
        normalization,
        spatialContext:
          fixture === 'great-circle'
            ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'}
            : undefined,
        output: {flows: 'flows'}
      };
      const inputs = {
        positions: {
          format: 'float32x2' as const,
          values: new Float32Array(
            fixture === 'great-circle'
              ? [179, 0, -179, 0, 170, 0]
              : fixture === 'tied'
                ? [0, 0, 1, 0, -1, 0]
                : [0, 0, 1, 0, 4, 0]
          )
        },
        populations: {
          format: 'float32' as const,
          values: new Float32Array(fixture === 'zero-population' ? [0, 0, 0] : [10, 20, 30])
        },
        outflows: {format: 'float32' as const, values: new Float32Array([3, 7, 11])}
      };
      const results = [];
      for (const Operation of [GPURadiationFlowModelOperation, CPURadiationFlowModelOperation])
        results.push(
          await runOperationGraph(device, {
            inputs,
            operations: [new Operation(props)],
            outputs: ['flows']
          })
        );
      // Float64 versus shader float32: relative/absolute tolerance 1e-5, every output row.
      for (let index = 0; index < 9; index++)
        expect(
          Math.abs(results[0]['flows'][index] - results[1]['flows'][index])
        ).toBeLessThanOrEqual(1e-5 * Math.max(1, Math.abs(results[1]['flows'][index])));
    });
  }
}

it('DGN-05 radiation operation publishes invalid centroids and preserves the zero diagonal', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: RadiationFlowModelOperationProps = {
    zoneCount: 2,
    positions: 'positions',
    populations: 'populations',
    outflows: 'outflows',
    output: {flows: 'flows', invalidInput: 'invalid'}
  };
  const rows = await runOperationGraph(device, {
    inputs: {
      positions: {format: 'float32x2', values: Float32Array.of(0, 0, NaN, 0)},
      populations: {format: 'float32', values: Float32Array.of(1, 1)},
      outflows: {format: 'float32', values: Float32Array.of(8, 6)}
    },
    operations: [new GPURadiationFlowModelOperation(props)],
    outputs: ['flows', 'invalid']
  });
  // Self flow is zero independently of centroids; unknown distances suppress both other flows.
  expect(Array.from(rows['flows'])).toEqual([0, NaN, NaN, 0]);
  expect(Array.from(rows['invalid'])).toEqual([1, 1]);
});
