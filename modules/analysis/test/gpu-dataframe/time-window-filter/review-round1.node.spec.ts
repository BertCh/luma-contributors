// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {createTransientView, GPUCommandGraph, type GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {ScalarData, WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {getTimeWindowClassifyNodes} from '../../../src/dataframe-analysis/time-window-filter/time-window-classify-node';
import {
  splitTimestamps,
  getGPUTimeWindowFilterParameterValues
} from '../../../src/dataframe-analysis/time-window-filter/time-window-parameters';
import {createSortedFloatTimeWindowRangeNodes} from '../../../src/dataframe-analysis/time-window-filter/time-window-float-range-nodes';
import {GPUTimeWindowFilter} from '../../../src/dataframe-analysis/time-window-filter/gpu-time-window-filter';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

const foreign = createRequire(import.meta.url)(
  '@luma.gl/graph/gpu-core'
) as typeof import('@luma.gl/graph/gpu-core');

for (const port of ['selection', 'outputMask', 'sourceIds', 'trackIds'] as const) {
  it(`normalizes foreign ${port} vector wrappers at upstream boundaries (B1)`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const packed = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `port-${serial++}`, format, length);
      const vector = <Format extends 'float32' | 'uint32'>(format: Format) =>
        createVectorView(`vector-${serial++}`, format, [packed(format, 2), packed(format, 3)]);
      const local = vector('uint32');
      const foreignPort = new foreign.GraphVectorView({...local}) as GraphVectorView<'uint32'>;
      const props = {
        timestamps: vector('float32'),
        window: packed('float32', 8),
        additionalPredicates:
          port === 'selection' ? [{kind: 'selection' as const, mask: foreignPort}] : undefined,
        sourceIds: port === 'sourceIds' ? foreignPort : undefined,
        trackIds: port === 'trackIds' ? foreignPort : undefined,
        output: {
          rows: packed('uint32', 5),
          ids: port === 'sourceIds' ? packed('uint32', 5) : undefined,
          outputMask: port === 'outputMask' ? foreignPort : undefined,
          trackVisibleCounts: port === 'trackIds' ? packed('uint32', 2) : undefined,
          status: {count: packed('uint32', 1), overflow: packed('uint32', 1)}
        }
      };
      expect(() => new GPUTimeWindowFilter(props).getCommandNodes(graph)).not.toThrow();
      // Normalization must not replace the graph's underlying chunk handles.
      expect(foreignPort.data).toEqual(local.data);
    } finally {
      device.destroy();
    }
  });
}

it('keeps strict subnormal low lanes distinct under float comparison flush-to-zero (earlier-fix note 1)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const high = createTransientView(graph, 'high', 'float32', 1);
  const low = createTransientView(graph, 'low', 'float32', 1);
  const parameters = createTransientView(graph, 'parameters', 'float32', 8);
  const mask = createTransientView(graph, 'mask', 'uint32', 1);
  const endpoint = 2 ** -103;
  const row = endpoint + 2 ** -149;
  const split = splitTimestamps([row], {requireExact: true});
  const values: Record<string, Uint32Array> = {
    params: new Uint32Array(
      getGPUTimeWindowFilterParameterValues({start: endpoint, end: endpoint}).buffer
    ),
    startHigh: new Uint32Array(split.high.buffer),
    startLow: new Uint32Array(split.low.buffer),
    outputMask: new Uint32Array(1)
  };
  expect(row > endpoint).toBe(true); // One exact binary64 increment of 2^-149 beyond the endpoint.
  const [kernel] = getGPUCommandNodeKernelSources(
    getTimeWindowClassifyNodes(graph, {
      id: 'subnormal-classify',
      timestamps: high,
      timestampsLow: low,
      window: parameters,
      mask
    })
  );
  try {
    executeWithFloatComparisonFlush(kernel, values);
    expect(Array.from(values['outputMask'])).toEqual([0]);
  } finally {
    device.destroy();
  }
});

function executeWithFloatComparisonFlush(
  kernel: ReturnType<typeof getGPUCommandNodeKernelSources>[number],
  values: Record<string, Uint32Array>
): void {
  const execution = new WgslExec(new WgslParser().parse(kernel.source));
  const binary = execution._evalBinaryOp.bind(execution);
  execution._evalBinaryOp = (expression, context) => {
    if (['==', '!=', '>=', '<=', '>', '<'].includes(expression.operator)) {
      const left = execution.evalExpression(expression.left, context);
      const right = execution.evalExpression(expression.right, context);
      if (
        left instanceof ScalarData &&
        right instanceof ScalarData &&
        left.typeInfo.name === 'f32' &&
        right.typeInfo.name === 'f32'
      ) {
        const flush = (value: number) => (Math.abs(value) < 2 ** -126 ? 0 : value);
        const first = flush(left.value),
          second = flush(right.value);
        const comparisons: Record<string, boolean> = {
          '==': first === second,
          '!=': first !== second,
          '>=': first >= second,
          '<=': first <= second,
          '>': first > second,
          '<': first < second
        };
        return new ScalarData(
          comparisons[expression.operator] ? 1 : 0,
          execution.getTypeInfo('bool')!
        );
      }
    }
    return binary(expression, context);
  };
  const errors = vi.spyOn(console, 'error').mockImplementation(message => {
    if (!String(message).startsWith('Variable _ not found.')) throw new Error(String(message));
  });
  try {
    const bindings = Object.fromEntries(
      new WgslReflect(kernel.source).storage.map(binding => [binding.binding, values[binding.name]])
    );
    execution.dispatchWorkgroups('main', 1, {0: bindings}, {constants: kernel.constants});
  } finally {
    errors.mockRestore();
  }
}

it('sorted bit searches preserve subnormal boundaries in both signs (earlier-fix note 1)', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const sign of [1, -1]) {
      const graph = new GPUCommandGraph(device);
      const endpoint = sign * 2 ** -103;
      const rows = sign === 1 ? [endpoint, endpoint + 2 ** -149] : [endpoint - 2 ** -149, endpoint];
      const split = splitTimestamps(rows, {requireExact: true});
      const view = <Format extends 'float32' | 'uint32'>(
        name: string,
        format: Format,
        length: number
      ) => createTransientView(graph, name, format, length);
      const kernels = getGPUCommandNodeKernelSources(
        createSortedFloatTimeWindowRangeNodes(graph, {
          id: 'subnormal-search',
          timestamps: view('high', 'float32', 2),
          timestampsLow: view('low', 'float32', 2),
          window: view('parameters', 'float32', 8),
          range: view('range', 'uint32', 2),
          total: view('total', 'uint32', 1),
          firstInstance: view('first', 'uint32', 1)
        })
      );
      const values: Record<string, Uint32Array> = {
        parameters: new Uint32Array(
          getGPUTimeWindowFilterParameterValues({start: endpoint, end: endpoint}).buffer
        ),
        rowHigh: new Uint32Array(split.high.buffer),
        rowLow: new Uint32Array(split.low.buffer),
        rangeWords: new Uint32Array([99, 99]),
        total: new Uint32Array([99]),
        firstInstance: new Uint32Array([99])
      };
      for (const kernel of kernels) executeWithFloatComparisonFlush(kernel, values);
      expect(Array.from(values['rangeWords'])).toEqual([sign === 1 ? 0 : 1, 1]);
      expect(Array.from(values['total'])).toEqual([1]);
      expect(Array.from(values['firstInstance'])).toEqual([sign === 1 ? 0 : 1]);
    }
  } finally {
    device.destroy();
  }
});
