// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUClassBreaksOperation} from '../../../src/dataframe-analysis/column-classification/cpu-class-breaks-operation';
import {
  GPUClassBreaksOperation,
  type ClassBreaksOperationProps
} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks-operation';
import {
  getGPUClassBreaksParameterValues,
  GPU_CLASS_BREAKS_METHODS
} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const method of GPU_CLASS_BREAKS_METHODS) {
  for (const empty of [false, true]) {
    it(`class-break operations agree for ${method}, all masked=${empty}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const values = new Float32Array([
        0,
        1,
        2,
        9,
        10,
        11,
        30,
        31,
        32,
        NaN,
        Infinity,
        -Infinity,
        1000
      ]);
      const props: ClassBreaksOperationProps = {
        values: 'values',
        parameters: 'parameters',
        mask: 'mask',
        rowCount: values.length,
        maximumClassCount: 6,
        methods: [method],
        naturalBreaksMode: 'exact',
        output: {breaks: 'breaks', classCount: 'classCount', classCounts: 'classCounts'}
      };
      const inputs = {
        values: {format: 'float32' as const, values},
        parameters: {
          format: 'float32' as const,
          values: getGPUClassBreaksParameterValues(
            {method, classCount: 3, customEdges: [-Infinity, 2, 11, Infinity]},
            6
          )
        },
        mask: {
          format: 'uint32' as const,
          values: empty
            ? new Uint32Array(values.length)
            : new Uint32Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        }
      };
      const outputs = ['breaks', 'classCount', 'classCounts'];
      const cpu = await runOperationGraph(device, {
        inputs,
        outputs,
        operations: [new CPUClassBreaksOperation(props)]
      });
      const gpu = await runOperationGraph(device, {
        inputs,
        outputs,
        operations: [new GPUClassBreaksOperation(props)]
      });
      // Counts and copied/natural/max-gap edges are exact. Arithmetic edges use relative 3e-5.
      expect(Array.from(gpu['classCount'])).toEqual(Array.from(cpu['classCount']));
      expect(Array.from(gpu['classCounts'])).toEqual(Array.from(cpu['classCounts']));
      if (!empty) expect(cpu['classCount'][0]).toBeGreaterThan(1);
      for (let edge = 0; edge < cpu['breaks'].length; edge++) {
        const expected = cpu['breaks'][edge];
        if (Number.isNaN(expected)) expect(gpu['breaks'][edge]).toBeNaN();
        else if (
          !Number.isFinite(expected) ||
          expected === 0 ||
          ['natural-breaks', 'custom', 'maximum-breaks'].includes(method)
        )
          expect(gpu['breaks'][edge] === expected).toBe(true);
        else
          expect(Math.abs(gpu['breaks'][edge] - expected)).toBeLessThan(3e-5 * Math.abs(expected));
      }
    });
  }
}

for (const method of ['custom', 'maximum-breaks'] as const) {
  it(`class-break operations preserve signed-zero key distinctions for ${method}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const custom = method === 'custom';
    const values = custom ? new Float32Array([-0, +0]) : new Float32Array([-1, -0, +0, 1]);
    const maximumClassCount = custom ? 2 : 4;
    const props: ClassBreaksOperationProps = {
      values: 'values',
      parameters: 'parameters',
      rowCount: values.length,
      maximumClassCount,
      methods: [method],
      output: {breaks: 'breaks', classCount: 'classCount', classCounts: 'classCounts'}
    };
    const inputs = {
      values: {format: 'float32' as const, values},
      parameters: {
        format: 'float32' as const,
        values: getGPUClassBreaksParameterValues(
          {method, classCount: maximumClassCount, customEdges: [-1, +0, 1]},
          maximumClassCount
        )
      }
    };
    const outputs = ['breaks', 'classCount', 'classCounts'];
    const cpu = await runOperationGraph(device, {
      inputs,
      outputs,
      operations: [new CPUClassBreaksOperation(props)]
    });
    const gpu = await runOperationGraph(device, {
      inputs,
      outputs,
      operations: [new GPUClassBreaksOperation(props)]
    });
    const expectedEdges = custom ? [-1, 0, 1] : [-1, -0.5, 0, 0.5, 1];
    // Integer counts differ under numeric comparisons: this tests key semantics, not zero signs.
    for (const rows of [cpu, gpu]) {
      expect(Array.from(rows['classCount'])).toEqual([maximumClassCount]);
      expect(Array.from(rows['classCounts'])).toEqual(new Array<number>(maximumClassCount).fill(1));
      expect(rows['breaks'].length).toBe(expectedEdges.length);
      expectedEdges.forEach((expected, edge) =>
        expect(rows['breaks'][edge] === expected).toBe(true)
      );
    }
  });
}
