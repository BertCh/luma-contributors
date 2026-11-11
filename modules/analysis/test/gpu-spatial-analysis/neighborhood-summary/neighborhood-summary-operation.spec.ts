// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNeighborhoodSummaryOperation} from '../../../src/gpu-spatial-analysis/neighborhood-summary/cpu-neighborhood-summary-operation';
import {
  GPUNeighborhoodSummaryOperation,
  type NeighborhoodSummaryOperationProps
} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const standardized of [false, true]) {
  for (const truncated of [false, true]) {
    it(`summary operations agree, row standardized=${standardized}, truncated=${truncated}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const props: NeighborhoodSummaryOperationProps = {
        rowCount: 5,
        weights: {
          offsets: 'offsets',
          neighbors: 'neighbors',
          weights: 'weights',
          status: {overflow: 'weightsOverflow'},
          information: {rowCount: 5, transform: standardized ? 'row' : 'raw', symmetric: 'unknown'}
        },
        values: 'values',
        categories: 'categories',
        mask: 'mask',
        includeFocal: true,
        focalWeight: 2,
        statistics: [
          'count',
          'weightSum',
          'sum',
          'mean',
          'min',
          'max',
          'standardDeviation',
          'median'
        ],
        output: {table: 'table', modes: 'modes', entropy: 'entropy', status: {overflow: 'overflow'}}
      };
      const inputs = {
        offsets: {format: 'uint32' as const, values: new Uint32Array([0, 5, 7, 9, 9, 10])},
        neighbors: {
          format: 'uint32' as const,
          values: new Uint32Array([0, 1, 2, 3, 10, 0, 2, 0, 1, 1])
        },
        weights: {
          format: 'float32' as const,
          values: new Float32Array([9, 1, 3, 0, 2, 2, 1, 0.5, 2, 1])
        },
        weightsOverflow: {format: 'uint32' as const, values: Uint32Array.of(Number(truncated))},
        values: {format: 'float32' as const, values: new Float32Array([0, 2, 6, NaN, 8])},
        categories: {format: 'uint32' as const, values: new Uint32Array([5, 5, 3, 8, 9])},
        mask: {format: 'uint32' as const, values: new Uint32Array([1, 1, 1, 1, 0])}
      };
      const outputs = ['table', 'modes', 'entropy', 'overflow'];
      const cpu = await runOperationGraph(device, {
        inputs,
        outputs,
        operations: [new CPUNeighborhoodSummaryOperation(props)]
      });
      const gpu = await runOperationGraph(device, {
        inputs,
        outputs,
        operations: [new GPUNeighborhoodSummaryOperation(props)]
      });
      // Exact counts, keys, copies, medians, flags and zeros. Relative 3e-5 for arithmetic.
      for (const output of outputs) {
        expect(gpu[output].length).toBe(cpu[output].length);
        for (let index = 0; index < cpu[output].length; index++) {
          const expected = cpu[output][index];
          const exact =
            output === 'modes' ||
            output === 'overflow' ||
            (output === 'table' && [0, 4, 5, 7].includes(index % 8)) ||
            expected === 0;
          if (Number.isNaN(expected)) expect(gpu[output][index]).toBeNaN();
          else if (exact) expect(gpu[output][index] === expected).toBe(true);
          else
            expect(Math.abs(gpu[output][index] - expected)).toBeLessThan(3e-5 * Math.abs(expected));
        }
      }
    });
  }
}
