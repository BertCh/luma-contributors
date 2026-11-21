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

for (const inherited of [false, true]) {
  it(`both class-break operation forms retain getter mask and output names, inherited=${inherited}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [CPUClassBreaksOperation, GPUClassBreaksOperation]) {
      const mask = {get: () => 'mask', enumerable: false};
      const outputFields = Object.fromEntries(
        ['breaks', 'classCount', 'classCounts'].map(name => [
          name,
          {get: () => name, enumerable: false}
        ])
      );
      const output = inherited
        ? Object.create(Object.defineProperties({}, outputFields))
        : Object.defineProperties({}, outputFields);
      const props: ClassBreaksOperationProps = Object.assign(
        inherited
          ? Object.create(Object.defineProperty({}, 'mask', mask))
          : Object.defineProperty({}, 'mask', mask),
        {
          values: 'values',
          parameters: 'parameters',
          maximumClassCount: 2,
          methods: ['equal-interval'],
          output
        }
      );
      const result = await runOperationGraph(device, {
        operations: [new Operation(props)],
        outputs: ['breaks', 'classCount', 'classCounts'],
        inputs: {
          values: {format: 'float32', values: Float32Array.of(1, 100)},
          mask: {format: 'uint32', values: Uint32Array.of(1, 0)},
          parameters: {
            format: 'float32',
            values: getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2)
          }
        }
      });
      expect(Array.from(result['breaks'])).toEqual([1, 1, NaN]);
      expect(Array.from(result['classCount'])).toEqual([1]);
      expect(Array.from(result['classCounts'])).toEqual([1, 0]);
    }
  });
}

it('both class-break operation forms require exact mode for exact storage even with only equal intervals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: ClassBreaksOperationProps = {
    values: 'values',
    parameters: 'parameters',
    maximumClassCount: 2,
    methods: ['equal-interval'],
    naturalBreaksExactStorage: 'recompute',
    output: {breaks: 'breaks', classCount: 'classCount'}
  };
  for (const Operation of [CPUClassBreaksOperation, GPUClassBreaksOperation]) {
    expect(() => new Operation(props)).toThrow(/naturalBreaksExactStorage requires exact mode/);
    const result = await runOperationGraph(device, {
      operations: [new Operation({...props, naturalBreaksMode: 'exact'})],
      outputs: ['breaks'],
      inputs: {
        values: {format: 'float32', values: Float32Array.of(1, 100)},
        parameters: {
          format: 'float32',
          values: getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2)
        }
      }
    });
    expect(Array.from(result['breaks'])).toEqual([1, 50.5, 100]);
  }
});

it('reuses each class-break operation after caller props, methods and output ports mutate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [CPUClassBreaksOperation, GPUClassBreaksOperation]) {
    const methods: NonNullable<ClassBreaksOperationProps['methods']>[number][] = ['equal-interval'];
    const props: ClassBreaksOperationProps = {
      values: 'values',
      parameters: 'parameters',
      rowCount: 2,
      methods,
      maximumClassCount: 2,
      output: {breaks: 'breaks', classCount: 'classCount', classCounts: 'classCounts'}
    };
    const operation = new Operation(props);
    for (const run of [0, 1]) {
      if (run === 1) {
        props.rowCount = 3;
        props.valuesFormat = 'float32';
        props.maximumClassCount = 6;
        props.values = 'other-values';
        props.mask = 'other-mask';
        props.output.breaks = 'other-breaks';
        props.output.classCounts = undefined;
        methods.splice(0, methods.length, 'quantile');
      }
      const result = await runOperationGraph(device, {
        inputs: {
          values: {format: 'uint32', values: new Uint32Array([0, 2])},
          parameters: {
            format: 'float32',
            values: getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2)
          }
        },
        operations: [operation],
        outputs: ['breaks', 'classCount', 'classCounts']
      });
      expect(Array.from(result['breaks'])).toEqual([0, 1, 2]);
      expect(Array.from(result['classCount'])).toEqual([2]);
      expect(Array.from(result['classCounts'])).toEqual([1, 1]);
    }
  }
});

for (const format of ['uint32', 'sint32'] as const) {
  it(`class-break operations infer ${format} and rowCount from the same input`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const values = format === 'uint32' ? new Uint32Array([0, 2]) : new Int32Array([-2, 0]);
    const props: ClassBreaksOperationProps = {
      values: 'values',
      parameters: 'parameters',
      maximumClassCount: 2,
      methods: ['equal-interval'],
      output: {breaks: 'breaks', classCount: 'classCount', classCounts: 'classCounts'}
    };
    for (const Operation of [GPUClassBreaksOperation, CPUClassBreaksOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          values: {format, values},
          parameters: {
            format: 'float32',
            values: getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2)
          }
        },
        operations: [new Operation(props)],
        outputs: ['breaks', 'classCount', 'classCounts']
      });
      expect(Array.from(result['breaks'])).toEqual(format === 'uint32' ? [0, 1, 2] : [-2, -1, 0]);
      expect(Array.from(result['classCount'])).toEqual([2]);
      expect(Array.from(result['classCounts'])).toEqual([1, 1]);
    }
  });
}

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
  it(`class-break operations merge signed zeros numerically for ${method}`, async () => {
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
    const expectedEdges = custom ? [-1, 0, 1] : [-1, -0.5, 0.5, 1, NaN];
    // Numeric membership puts both signed zeros in the same class.
    for (const rows of [cpu, gpu]) {
      expect(Array.from(rows['classCount'])).toEqual([custom ? 2 : 3]);
      expect(Array.from(rows['classCounts'])).toEqual(custom ? [0, 2] : [1, 2, 1, 0]);
      expect(rows['breaks'].length).toBe(expectedEdges.length);
      expectedEdges.forEach((expected, edge) => {
        if (Number.isNaN(expected)) expect(Number.isNaN(rows['breaks'][edge])).toBe(true);
        else expect(rows['breaks'][edge] === expected).toBe(true);
      });
    }
  });
}
