// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSpatialBalancedSampleOperation} from '../../../src/gpu-spatial-analysis/point-process/cpu-spatial-balanced-sample-operation';
import {
  GPUSpatialBalancedSampleOperation,
  type SpatialBalancedSampleOperationProps
} from '../../../src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './spatial-balanced-sample-operation-fixtures.json';

for (const weighted of [false, true])
  for (const invalid of [false, true]) {
    it(`compares every balanced output, weighted=${weighted}, invalid=${invalid}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = fixtures[weighted ? 1 : 0];
      const strata = new Uint32Array(fixture.strata);
      if (invalid) strata[0] = 123;
      const props: SpatialBalancedSampleOperationProps = {
        id: 'balanced-comparison',
        rowCount: fixture.frameIds.length,
        positions: 'positions',
        frameIds: 'frameIds',
        strata: 'strata',
        inclusionTickets: weighted ? 'tickets' : undefined,
        parameters: 'parameters',
        bounds: [0, 0, 1, 1],
        seed: 0,
        allocations: fixture.allocations,
        output: {
          frameIds: 'selected',
          inclusionProbabilities: 'probabilities',
          designWeights: 'weights',
          status: 'status'
        }
      };
      const inputs = {
        positions: {format: 'float32x2' as const, values: new Float32Array(fixture.positions)},
        frameIds: {format: 'uint32' as const, values: new Uint32Array(fixture.frameIds)},
        strata: {format: 'uint32' as const, values: strata},
        parameters: {
          format: 'uint32' as const,
          values: new Uint32Array([fixture.seed, fixture.replicate])
        },
        ...(weighted
          ? {tickets: {format: 'uint32' as const, values: new Uint32Array(fixture.tickets!)}}
          : {})
      };
      const outputs = Object.values(props.output);
      const gpu = await runOperationGraph(device, {
        inputs,
        operations: [new GPUSpatialBalancedSampleOperation(props)],
        outputs
      });
      const cpu = await runOperationGraph(device, {
        inputs,
        operations: [new CPUSpatialBalancedSampleOperation(props)],
        outputs
      });
      expect(Array.from(gpu['selected'])).toEqual(Array.from(cpu['selected']));
      expect(Array.from(gpu['status'])).toEqual(Array.from(cpu['status']));
      expect(Array.from(cpu['status'])).toEqual(invalid ? [0, 1, 0, 1] : fixture.expectedStatus);
      if (!invalid) expect(Array.from(cpu['selected'])).toEqual(fixture.expectedIds);
      // Probabilities and weights are strictly positive: a relative tolerance cannot accept zero.
      for (const output of ['probabilities', 'weights'])
        for (const [index, expected] of cpu[output].entries()) {
          if (Number.isNaN(expected)) expect(Number.isNaN(gpu[output][index])).toBe(true);
          else
            expect(Math.abs(gpu[output][index] - expected)).toBeLessThanOrEqual(
              2e-6 * Math.abs(expected)
            );
        }
    });
  }
