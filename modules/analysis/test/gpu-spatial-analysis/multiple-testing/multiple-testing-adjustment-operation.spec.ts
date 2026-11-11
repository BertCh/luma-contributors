// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUMultipleTestingAdjustmentOperation} from '../../../src/gpu-spatial-analysis/multiple-testing/cpu-multiple-testing-adjustment-operation';
import {GPUMultipleTestingAdjustmentOperation} from '../../../src/gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const method of [
  'none',
  'bonferroni',
  'holm',
  'benjamini-hochberg',
  'benjamini-yekutieli'
] as const) {
  it(`${method}: both operations agree on ties, missing rows, and an entirely untested family`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const pValues of [
      Float32Array.of(0, 0.0001, 0.0002, 0.0002, 0.7, 1, -0, NaN, -1, Infinity),
      // Normal float32 powers of two, far above the subnormal range; H_80 stays unsaturated.
      Float32Array.of(0, 2 ** -80, 2 ** -79, 2 ** -78),
      new Float32Array(3).fill(NaN),
      Float32Array.from({length: 300}, (_, row) => (row % 97) / 97)
    ]) {
      const props = {
        pValues: 'pValues',
        rowCount: pValues.length,
        method,
        minimumFamilySize: 80,
        output: {adjustedPValues: 'adjusted', testedCount: 'tested'}
      };
      const results: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
      for (const Operation of [
        GPUMultipleTestingAdjustmentOperation,
        CPUMultipleTestingAdjustmentOperation
      ]) {
        results.push(
          await runOperationGraph(device, {
            inputs: {pValues: {format: 'float32', values: pValues}},
            operations: [new Operation(props)],
            outputs: ['adjusted', 'tested']
          })
        );
      }
      expect(Array.from(results[0]['tested'])).toEqual(Array.from(results[1]['tested']));
      results[1]['adjusted'].forEach((expected, row) => {
        if (Number.isNaN(expected)) {
          expect(results[0]['adjusted'][row]).toBeNaN();
        } else if (method === 'none') {
          // A copy: exact value equality. The sign of a zero is not compared (the GPU and the
          // twin differ in it on the test GPU), since either zero is the same p-value.
          expect(results[0]['adjusted'][row] === expected).toBe(true);
        } else if (expected === 0) {
          // Either sign of zero is the same exact p-value; no nonzero error is accepted.
          expect(results[0]['adjusted'][row] === 0).toBe(true);
        } else {
          // Relative tolerance only: small positive results must not pass as zero.
          expect(Math.abs(results[0]['adjusted'][row] - expected)).toBeLessThanOrEqual(
            2e-6 * Math.abs(expected)
          );
        }
      });
    }
  });
}
