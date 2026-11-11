// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUContinuousSpatialSampleOperation} from '../../../src/gpu-spatial-analysis/point-process/cpu-continuous-spatial-sample-operation';
import {
  GPUContinuousSpatialSampleOperation,
  type ContinuousSpatialSampleOperationProps
} from '../../../src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './continuous-spatial-sample-operation-fixtures.json';

for (const frameType of ['line', 'area'] as const)
  for (const scenario of ['ordinary', 'overflow', 'invalid'] as const) {
    it(`compares every continuous ${frameType} output on ${scenario} input`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = fixtures[frameType === 'line' ? 0 : 1];
      const positions = new Float32Array(fixture.positions);
      if (scenario === 'invalid') positions.fill(0);
      const props: ContinuousSpatialSampleOperationProps = {
        id: 'continuous-comparison',
        frameRowCount: positions.length / 2,
        legacySiteCount: fixture.legacySites.length / (frameType === 'line' ? 1 : 2),
        frame:
          frameType === 'line'
            ? {type: frameType, positions: 'positions', legacySites: 'legacySites'}
            : {
                type: frameType,
                positions: 'positions',
                legacySites: 'legacySites',
                bounds: fixture.bounds as [number, number, number, number]
              },
        spacing: fixture.spacing,
        seed: 0,
        waves: fixture.waves,
        capacity: scenario === 'overflow' ? 2 : fixture.capacity,
        parameters: 'parameters',
        output: {
          positions: 'selected',
          designWeights: 'weights',
          provenance: 'provenance',
          status: 'status'
        }
      };
      const inputs = {
        positions: {format: 'float32x2' as const, values: positions},
        legacySites: {
          format: frameType === 'line' ? ('float32' as const) : ('float32x2' as const),
          values: new Float32Array(fixture.legacySites)
        },
        parameters: {
          format: 'uint32' as const,
          values: new Uint32Array([fixture.seed, fixture.replicate])
        }
      };
      const outputs = Object.values(props.output);
      const gpu = await runOperationGraph(device, {
        inputs,
        operations: [new GPUContinuousSpatialSampleOperation(props)],
        outputs
      });
      const cpu = await runOperationGraph(device, {
        inputs,
        operations: [new CPUContinuousSpatialSampleOperation(props)],
        outputs
      });
      expect(Array.from(gpu['status'])).toEqual(Array.from(cpu['status']));
      expect(Array.from(gpu['provenance'])).toEqual(Array.from(cpu['provenance']));
      if (scenario === 'ordinary')
        expect(Array.from(cpu['status'])).toEqual(fixture.expectedStatus);
      if (scenario === 'invalid')
        expect(Array.from(cpu['status'])).toEqual([0, 0, 0, 1, 0, 0, 0, 1]);
      // Float64 versus float32 geometric arithmetic on these small planar frames: 2e-5 absolute.
      for (const [index, value] of cpu['selected'].entries())
        expect(Math.abs(gpu['selected'][index] - value)).toBeLessThanOrEqual(2e-5);
      // Weights are copied float32 literals, including legacy/unused zeros.
      expect(Array.from(gpu['weights'])).toEqual(Array.from(cpu['weights']));
    });
  }

it('copies exactly four half-unit line design weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: ContinuousSpatialSampleOperationProps = {
    id: 'exact-line-weights',
    frameRowCount: 2,
    frame: {type: 'line', positions: 'positions'},
    spacing: 0.5,
    seed: 1,
    capacity: 4,
    output: {
      positions: 'selected',
      designWeights: 'weights',
      provenance: 'provenance',
      status: 'status'
    }
  };
  const inputs = {
    positions: {format: 'float32x2' as const, values: new Float32Array([0, 0, 2, 0])}
  };
  const outputs = Object.values(props.output);
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUContinuousSpatialSampleOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUContinuousSpatialSampleOperation(props)],
    outputs
  });
  expect(Array.from(cpu['weights'])).toEqual([0.5, 0.5, 0.5, 0.5]);
  for (const output of ['weights', 'provenance', 'status'])
    expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
  expect(Array.from(cpu['status'])).toEqual([4, 4, 4, 0]);
  for (const [index, expected] of cpu['selected'].entries())
    expect(Math.abs(gpu['selected'][index] - expected)).toBeLessThanOrEqual(2e-5);
});
