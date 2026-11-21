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
import {requireTestDevice} from '../../../src/testing/require-test-device';
import fixtures from './continuous-spatial-sample-operation-fixtures.json';

it('translated projected line retains membership spacing and two complete sampling waves (R19)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Binary32 spacing at 2^20 is 1/8. A 4-unit design step is exactly representable,
  // exceeds the contributor's coordinate-resolution gate, and divides this straight line
  // into four cells. Any phase in [0,4) produces exactly four sites per wave.
  const origin = 1048576;
  const props: ContinuousSpatialSampleOperationProps = {
    id: 'projected-line',
    frameRowCount: 2,
    frame: {type: 'line', positions: 'positions'},
    spacing: 4,
    seed: 17,
    waves: 2,
    capacity: 4,
    output: {
      positions: 'selected',
      designWeights: 'weights',
      provenance: 'provenance',
      status: 'status'
    }
  };
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {
        format: 'float32x2',
        values: Float32Array.from([origin, origin, origin + 16, origin])
      }
    },
    operations: [new GPUContinuousSpatialSampleOperation(props)],
    outputs: Object.values(props.output)
  });
  expect(Array.from(result['status'])).toEqual([4, 4, 4, 0, 4, 4, 4, 0]);
  expect(Array.from(result['weights'])).toEqual(Array<number>(8).fill(4));
  expect(result['selected']).toHaveLength(16);
  expect(result['provenance']).toHaveLength(32);
  // Independent Python integer Philox4x32-10, key=(17,0x43535350), counter=(wave,0,0,0):
  // phases are 3.721618175506592 and 2.192657470703125. Rounded to coordinate ulps=1/8.
  for (const [wave, expected] of [1048579.75, 1048578.25].entries()) {
    expect(Number.isFinite(result['selected'][wave * 8])).toBe(true);
    expect(Math.abs(result['selected'][wave * 8] - expected)).toBeLessThanOrEqual(0.125);
  }
  for (let wave = 0; wave < 2; wave++)
    for (let site = 0; site < 4; site++) {
      const row = wave * 4 + site;
      const positionX = result['selected'][row * 2];
      const positionY = result['selected'][row * 2 + 1];
      expect(Number.isFinite(positionX) && Number.isFinite(positionY)).toBe(true);
      expect(positionY).toBe(origin);
      expect(positionX).toBeGreaterThanOrEqual(origin + site * 4);
      expect(positionX).toBeLessThan(origin + (site + 1) * 4);
      if (site)
        expect(Math.abs(positionX - result['selected'][(row - 1) * 2] - 4)).toBeLessThanOrEqual(
          0.25
        );
      expect(Array.from(result['provenance'].slice(row * 4, row * 4 + 4))).toEqual([
        0,
        wave,
        site,
        wave === 0 ? 0 : 1
      ]);
    }
});

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
