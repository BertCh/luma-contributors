// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUColumnQuantilesOperation} from '../../../src/dataframe-analysis/column-classification/cpu-column-quantiles-operation';
import {
  GPUColumnQuantilesOperation,
  type ColumnQuantilesOperationProps
} from '../../../src/dataframe-analysis/column-classification/gpu-column-quantiles-operation';
import {
  getGPUColumnQuantilesParameterValues,
  getGPUColumnQuantilesRationalProbabilityValues,
  type GPUColumnQuantileInterpolation
} from '../../../src/dataframe-analysis/column-classification/column-quantiles-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

// Four surviving values [-4, 0, 2, 8] give ranks 0, .75, 1.5, 2.25, 3.
// The median is a half-even tie; always selecting lower now fails every other method.
const EXPECTED_QUANTILES: Record<GPUColumnQuantileInterpolation, number[]> = {
  lower: [-4, -4, 0, 2, 8, NaN],
  higher: [-4, 0, 2, 8, 8, NaN],
  nearest: [-4, 0, 2, 2, 8, NaN],
  linear: [-4, -1, 1, 3.5, 8, NaN],
  midpoint: [-4, -2, 1, 5, 8, NaN]
};

for (const interpolation of [
  'lower',
  'higher',
  'nearest',
  'linear',
  'midpoint'
] as GPUColumnQuantileInterpolation[]) {
  for (const rational of [false, true]) {
    for (const empty of [false, true]) {
      it(`quantile operations agree: ${interpolation}, rational=${rational}, empty=${empty}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const values = new Float32Array([9, -4, 0, 2, 8, 100, NaN, Infinity]);
        const props: ColumnQuantilesOperationProps = {
          values: 'values',
          parameters: 'parameters',
          mask: 'mask',
          rowCount: values.length,
          quantileCount: 6,
          rationalProbabilities: rational ? 'ratios' : undefined,
          output: {
            quantiles: 'quantiles',
            validCount: 'validCount',
            outputMask: 'outputMask',
            filterBounds: 'filterBounds'
          }
        };
        const inputs = {
          values: {format: 'float32' as const, values},
          parameters: {
            format: 'float32' as const,
            values: getGPUColumnQuantilesParameterValues({
              quantiles: [0, 0.25, 0.5, 0.75, 1, NaN],
              interpolation,
              filterRange: [0.25, 0.75]
            })
          },
          mask: {
            format: 'uint32' as const,
            values: empty
              ? new Uint32Array(values.length)
              : new Uint32Array([0, 1, 1, 1, 1, 0, 1, 1])
          },
          ...(rational
            ? {
                ratios: {
                  format: 'uint32' as const,
                  values: getGPUColumnQuantilesRationalProbabilityValues(
                    [
                      [0, 1],
                      [1, 4],
                      [1, 2],
                      [3, 4],
                      [1, 1]
                    ],
                    6
                  )
                }
              }
            : {})
        };
        const outputs = Object.values(props.output);
        const gpu = await runOperationGraph(device, {
          inputs,
          outputs,
          operations: [new GPUColumnQuantilesOperation(props)]
        });
        const cpu = await runOperationGraph(device, {
          inputs,
          outputs,
          operations: [new CPUColumnQuantilesOperation(props)]
        });
        if (!empty) {
          for (const rows of [gpu, cpu]) {
            expect(rows['validCount'][0]).toBe(4);
            EXPECTED_QUANTILES[interpolation].forEach((expected, index) => {
              if (Number.isNaN(expected)) expect(rows['quantiles'][index]).toBeNaN();
              else expect(rows['quantiles'][index] === expected).toBe(true);
            });
          }
        }
        // Binary-exact probabilities and quarter-ranks make every output exact, including copies,
        // masks and count. Numeric equality deliberately ignores the sign of zero.
        for (const output of outputs) {
          expect(cpu[output].length).toBe(gpu[output].length);
          for (let index = 0; index < cpu[output].length; index++) {
            const expected = cpu[output][index];
            if (Number.isNaN(expected)) expect(gpu[output][index]).toBeNaN();
            else expect(gpu[output][index] === expected).toBe(true);
          }
        }
      });
    }
  }
}

it('TWN-41 pins packed 0.7 row selection in both operation forms', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // f32(10 * f32(.7)) = 7; quantile rank uses n-1, filter rank uses n.
  for (const rowCount of [11, 10]) {
    const props: ColumnQuantilesOperationProps = {
      values: 'values',
      parameters: 'parameters',
      rowCount,
      quantileCount: 1,
      output: {
        quantiles: 'quantiles',
        validCount: 'validCount',
        filterBounds: 'filterBounds',
        outputMask: 'outputMask'
      }
    };
    for (const Operation of [CPUColumnQuantilesOperation, GPUColumnQuantilesOperation]) {
      const rows = await runOperationGraph(device, {
        inputs: {
          values: {
            format: 'float32',
            values: Float32Array.from({length: rowCount}, (_, row) => row)
          },
          parameters: {
            format: 'float32',
            values: getGPUColumnQuantilesParameterValues({
              quantiles: [0.7],
              interpolation: 'lower',
              filterRange: [0.7, 1]
            })
          }
        },
        outputs: Object.values(props.output),
        operations: [new Operation(props)]
      });
      if (rowCount === 11) expect(Array.from(rows['quantiles'])).toEqual([7]);
      else {
        expect(Array.from(rows['filterBounds'])).toEqual([7, 9]);
        expect(Array.from(rows['outputMask'])).toEqual([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]);
      }
    }
  }
});

it('UDF-08 preserves invalid host probabilities in both operation forms', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: ColumnQuantilesOperationProps = {
    values: 'values',
    parameters: 'parameters',
    rowCount: 3,
    quantileCount: 2,
    output: {quantiles: 'quantiles', validCount: 'validCount'}
  };
  for (const Operation of [CPUColumnQuantilesOperation, GPUColumnQuantilesOperation]) {
    const rows = await runOperationGraph(device, {
      inputs: {
        values: {format: 'float32', values: Float32Array.of(0, 10, 20)},
        parameters: {
          format: 'float32',
          values: getGPUColumnQuantilesParameterValues({quantiles: [-1e-50, 1 + Number.EPSILON]})
        }
      },
      outputs: Object.values(props.output),
      operations: [new Operation(props)]
    });
    expect(Array.from(rows['quantiles'])).toEqual([NaN, NaN]);
    expect(Array.from(rows['validCount'])).toEqual([3]);
  }
});
