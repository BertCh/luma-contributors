// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation';
import {GPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('the CPU operation accepts zero neighbor capacity while the GPU contributor rejects it', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs = {
    offsets: {format: 'uint32' as const, values: Uint32Array.of(0, 0)},
    neighbors: {format: 'uint32' as const, values: new Uint32Array()},
    weights: {format: 'float32' as const, values: new Float32Array()}
  };
  const props = {
    id: 'empty-validation',
    rowCount: 1,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    output: {flags: 'flags', status: {invalidCount: 'invalid'}}
  };
  await expect(
    runOperationGraph(device, {
      inputs,
      operations: [new GPUSpatialWeightsValidationOperation(props)],
      outputs: ['flags', 'invalid']
    })
  ).rejects.toThrow(/empty-validation weights.neighbors length/);
  const result = await runOperationGraph(device, {
    inputs,
    operations: [new CPUSpatialWeightsValidationOperation(props)],
    outputs: ['flags', 'invalid']
  });
  expect(Array.from(result['flags'])).toEqual([0]);
  expect(Array.from(result['invalid'])).toEqual([0]);
});

it('compares every status and flags word on valid, invalid and empty-neighborhood CSR', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scenes = [
    {
      offsets: Uint32Array.of(0, 2, 3, 4),
      neighbors: Uint32Array.of(1, 2, 0, 0, 0xffffffff),
      weights: Float32Array.of(0.5, 0.5, 1, 1, NaN)
    },
    {
      offsets: Uint32Array.of(0, 3, 4, 5, 6),
      neighbors: Uint32Array.of(0, 1, 1, 2, 9, 3),
      weights: Float32Array.of(1, 1, NaN, 0, -1, 1)
    },
    {
      offsets: Uint32Array.of(1, 0, 9),
      neighbors: Uint32Array.of(0, 0),
      weights: Float32Array.of(Infinity, -1)
    },
    {
      offsets: Uint32Array.of(0, 0, 0),
      neighbors: Uint32Array.of(0xffffffff),
      weights: Float32Array.of(NaN)
    }
  ];
  for (const scene of scenes) {
    for (const flags of ['flags', undefined]) {
      const props = {
        rowCount: scene.offsets.length - 1,
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        output: {flags, status: {invalidCount: 'invalid'}}
      };
      const outputs = flags ? ['flags', 'invalid'] : ['invalid'];
      const results = [];
      for (const Operation of [
        GPUSpatialWeightsValidationOperation,
        CPUSpatialWeightsValidationOperation
      ])
        results.push(
          await runOperationGraph(device, {
            inputs: {
              offsets: {format: 'uint32', values: scene.offsets},
              neighbors: {format: 'uint32', values: scene.neighbors},
              weights: {format: 'float32', values: scene.weights}
            },
            operations: [new Operation(props)],
            outputs
          })
        );
      // Zero tolerance: flags and invalid counts are exact uint32 outputs.
      for (const name of outputs)
        expect(Array.from(results[0][name])).toEqual(Array.from(results[1][name]));
    }
  }
});
