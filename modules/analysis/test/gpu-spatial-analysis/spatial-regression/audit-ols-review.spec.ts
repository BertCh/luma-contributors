// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUOrdinaryLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-ordinary-least-squares-operation';
import {GPUOrdinaryLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const mode of ['varying', 'constant', 'exact'] as const) {
  it(`R11 dimensionless BP survives raw-square underflow and rejects exact fits: ${mode}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scale = 2 ** -76;
    const predictors = Float32Array.from(Array.from({length: 4}, () => [-1, -1, 1, 1]).flat());
    const response =
      mode === 'exact'
        ? predictors
        : Float32Array.from(
            Array.from({length: 4}, () =>
              mode === 'varying'
                ? [-scale, scale, -2 * scale, 2 * scale]
                : [-scale, scale, -scale, scale]
            ).flat()
          );
    for (const Operation of [GPUOrdinaryLeastSquaresOperation, CPUOrdinaryLeastSquaresOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          predictors: {format: 'float32', values: predictors},
          response: {format: 'float32', values: response}
        },
        operations: [
          new Operation({
            rowCount: 16,
            predictorCount: 1,
            tileRowCount: 4,
            predictors: 'predictors',
            response: 'response',
            output: {
              status: {code: 'status'},
              coefficients: 'coefficients',
              standardErrors: 'standardErrors',
              tStatistics: 'tStatistics',
              summary: 'summary'
            }
          })
        ],
        outputs: ['status', 'summary']
      });
      expect(Array.from(result['status'])).toEqual([0]);
      const summary = result['summary'];
      if (mode === 'exact') {
        expect(summary[16]).toBeNaN();
        expect(summary[17]).toBeNaN();
      } else {
        // Independent explained SS/(2 meanSquare²), and SciPy sf: review2 oracle.
        const expected = mode === 'varying' ? [2.88, 0.08968602177036485] : [0, 1];
        expected.forEach((value, index) => {
          expect(Number.isFinite(summary[16 + index])).toBe(true);
          expect(Math.abs(summary[16 + index] - value)).toBeLessThan(2e-5);
        });
      }
    }
  });
}

it('R10 nearly constant squared magnitudes retain the independently nonsignificant KB result', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const difference = 2 ** -23;
  const first = 1.375 + 3 * difference;
  const predictors = Float32Array.from(Array.from({length: 64}, () => [1, 1, -2, -2, 1, 1]).flat());
  const response = Float32Array.from(
    Array.from({length: 64}, () => [
      -first,
      first,
      -(first + difference),
      first + difference,
      -(first + 2 * difference),
      first + 2 * difference
    ]).flat()
  );
  for (const Operation of [GPUOrdinaryLeastSquaresOperation, CPUOrdinaryLeastSquaresOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        predictors: {format: 'float32', values: predictors},
        response: {format: 'float32', values: response}
      },
      operations: [
        new Operation({
          rowCount: 384,
          predictorCount: 1,
          tileRowCount: 6,
          predictors: 'predictors',
          response: 'response',
          output: {
            status: {code: 'status'},
            coefficients: 'coefficients',
            standardErrors: 'standardErrors',
            tStatistics: 'tStatistics',
            summary: 'summary'
          }
        })
      ],
      outputs: ['status', 'coefficients', 'summary']
    });
    expect(Array.from(result['status'])).toEqual([0]);
    // Exact pairwise symmetry gives beta=[0,0] (review2 oracle). The GPU's origin-shifted
    // f32 means/correction need not cancel bitwise at zero: bound coefficients by one
    // input step at this response magnitude (2^-23), and reject NaN explicitly.
    // Keep the independent KB/p-value checks below: they distinguish the numerical defect.
    for (const coefficient of result['coefficients']) {
      expect(Number.isFinite(coefficient)).toBe(true);
      expect(Math.abs(coefficient)).toBeLessThanOrEqual(2 ** -23);
    }
    const summary = result['summary'];
    expect(Number.isFinite(summary[9])).toBe(true);
    expect(Number.isFinite(summary[10])).toBe(true);
    // Exact rational moments + SciPy chi-square sf: audit-ols-review2.oracle.py.
    // f32 cannot resolve the tiny second-order curvature relatively; absolute bounds cover
    // that rounding while decisively rejecting the old KB=13.7143, p=0.00021283 decision.
    expect(Math.abs(summary[9] - 2.4052735761050423e-13)).toBeLessThan(1e-8);
    expect(Math.abs(summary[10] - 0.999999608688779)).toBeLessThan(1e-5);
  }
});

for (const scale of [1, 2 ** 32, 2 ** -40]) {
  it(`R8 GPU and CPU OLS diagnostics are invariant at response scale ${scale}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [GPUOrdinaryLeastSquaresOperation, CPUOrdinaryLeastSquaresOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          predictors: {format: 'float32', values: Float32Array.of(-1, -1, 1, 1)},
          response: {
            format: 'float32',
            values: Float32Array.from([-1, 1, -2, 2], value => scale * value)
          }
        },
        operations: [
          new Operation({
            rowCount: 4,
            predictorCount: 1,
            predictors: 'predictors',
            response: 'response',
            output: {
              status: {code: 'status'},
              coefficients: 'coefficients',
              standardErrors: 'standardErrors',
              tStatistics: 'tStatistics',
              summary: 'summary'
            }
          })
        ],
        outputs: ['status', 'summary']
      });
      expect(Array.from(result['status'])).toEqual([0]);
      // Independent SciPy values printed by audit-ols-review.oracle.py. All are scale free.
      for (const [slot, expected] of [
        [14, 1.36],
        [7, 0.4482666666666666],
        [9, 4],
        [16, 0.72]
      ]) {
        expect(Number.isFinite(result['summary'][slot])).toBe(true);
        // f32 centered residuals, normalized fourth powers and auxiliary Cholesky rounding.
        expect(Math.abs(result['summary'][slot] - expected)).toBeLessThan(2e-5);
      }
    }
  });
}

it('R1 GPU and CPU operations distinguish classic BP from undefined studentized KB', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUOrdinaryLeastSquaresOperation, CPUOrdinaryLeastSquaresOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        predictors: {format: 'float32', values: Float32Array.of(-1, -1, 1, 1)},
        response: {format: 'float32', values: Float32Array.of(0, 2, 4, 6)}
      },
      operations: [
        new Operation({
          rowCount: 4,
          predictorCount: 1,
          predictors: 'predictors',
          response: 'response',
          output: {
            status: {code: 'status'},
            coefficients: 'coefficients',
            standardErrors: 'standardErrors',
            tStatistics: 'tStatistics',
            summary: 'summary',
            residuals: 'residuals'
          }
        })
      ],
      outputs: ['status', 'coefficients', 'summary', 'residuals']
    });
    expect(Array.from(result['status'])).toEqual([0]);
    expect(Array.from(result['coefficients'])).toEqual([3, 2]);
    expect(Array.from(result['residuals'])).toEqual([-1, 1, -1, 1]);
    // Hand derivation is in the matching Node case: e²/s²-1=0, BP=0, sf=1; KB has zero divisor.
    expect(Array.from(result['summary'].slice(9, 11))).toEqual([NaN, NaN]);
    expect(Array.from(result['summary'].slice(16, 18))).toEqual([0, 1]);
  }
});
