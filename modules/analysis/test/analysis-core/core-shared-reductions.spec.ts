// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  createSegmentSumNode,
  getSortedSegmentSumNodes
} from '../../src/analysis-core/kernels/sorted-segment-sums';
import {COMPENSATED_SUM_WGSL} from '../../src/analysis-core/kernels/compensated-sum-wgsl';
import {createWGSLKernelNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createFittingTestGraph} from './fitting/fitting-test-graph';
import {
  createSegmentParityFixture,
  validateCompensatedCancellationReadback
} from './kernel-device-probe-fixtures';

it.each([
  'nonempty',
  'empty'
] as const)('review B3–B5: named reductions preserve packed outputs for %s input', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const test = createFittingTestGraph(device);
  try {
    const segmentKeys = test.uints('keys', mode === 'empty' ? [] : [0, 1, 0]);
    // Nonempty logical contributions [2,3,5], with irrelevant interleaved words [99,99].
    // Hand-worked expected sums: segment 0 = 2+5 = 7, segment 1 = 3.
    const source = test.floats('source', [2, 99, 3, 99, 5, 99, 99, 99]);
    const contributions =
      mode === 'empty'
        ? source
        : test.graph.createDataView(source.buffer, {
            format: 'float32',
            length: 3,
            byteStride: 8
          });
    const keysResult = test.floats('keys-result', [99, 99], 1);
    const indicesResult = test.floats('indices-result', [99, 99], 1);
    test.graph.add({
      getCommandNodes: graph =>
        getSortedSegmentSumNodes(graph, {
          id: 'named',
          operation: 'SharedReductionReview',
          segmentCount: 2,
          segmentKeys,
          reductions: [
            {name: 'keys', contributions, output: keysResult},
            {name: 'indices', contributions, output: indicesResult}
          ]
        })
    });
    test.run();
    const expected = mode === 'empty' ? [0, 0] : [7, 3];
    for (const result of [keysResult, indicesResult]) {
      const values = await test.readFloats(result);
      expect(values).toHaveLength(2);
      expect(values.every(Number.isFinite)).toBe(true);
      expect(values).toEqual(expected);
      expect(await test.readFloatStorage(result)).toEqual([12345, ...expected, 12345]);
    }
  } finally {
    test.destroy();
  }
});

it('GEO-131 compensated scalar accumulator retains a literal nonzero cancellation residual', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const test = createFittingTestGraph(device);
  try {
    const input = test.floats('terms', [16777216, 1, -16777216]);
    const output = test.floats('result', [99, 99, 99, 99], 1);
    test.graph.add({
      getCommandNodes: graph => [
        createWGSLKernelNode(graph, {
          id: 'compensated',
          operation: 'CompensatedCancellation',
          invocationCount: 1,
          bindings: [
            {name: 'terms', view: input, type: 'f32', access: 'read'},
            {name: 'result', view: output, type: 'f32', access: 'read_write'}
          ],
          declarations: COMPENSATED_SUM_WGSL,
          body: `var accumulator = vec2<f32>(0.0);
  addCompensated(&accumulator, terms[termsOffset]);
  accumulator = addCompensatedValue(accumulator, terms[termsOffset + 1u]);
  addCompensated(&accumulator, terms[termsOffset + 2u]);
  result[resultOffset] = accumulator.x;
  result[resultOffset + 1u] = accumulator.y;
  result[resultOffset + 2u] = getCompensated(accumulator);
  result[resultOffset + 3u] = (terms[termsOffset] + terms[termsOffset + 1u]) + terms[termsOffset + 2u];`
        })
      ]
    });
    test.run();
    const actual = await test.readFloatStorage(output);
    expect(actual).toHaveLength(6);
    validateCompensatedCancellationReadback(actual);
    expect(actual.every(Number.isFinite)).toBe(true);
    // Exact arithmetic: 2^24 + 1 - 2^24 = 1. Binary32 first loses the 1, recovered as correction.
    expect(actual.slice(0, 4)).toEqual([12345, 0, 1, 1]);
    // The plain expression is diagnostic only: a reassociating compiler may retain its 1.
    expect(actual[5]).toBe(12345);
  } finally {
    test.destroy();
  }
});

it('LUMA-115 public and local trees preserve empty groups, prefixes, tails and gather order', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const test = createFittingTestGraph(device);
  try {
    const fixture = createSegmentParityFixture();
    const values = [
      ...fixture.values.filter((_, index) => fixture.keys[index] === 1),
      ...fixture.values.filter((_, index) => fixture.keys[index] === 3)
    ];
    // Independent binary32 tree derivation: lane 0 first adds 2 -> 16777218;
    // stride 128 adds -16777216 -> 2; lane 1 contributes 1 in the last stride -> 3.
    // A serial left fold gives 2, so the literal expectation pins the reduction order.
    const input = test.floats('input', values, 1);
    const gatherInput = test.floats('gather-input', fixture.values, 1);
    const offsets = test.uints('offsets', [0, 0, 257, 257, 259, 0xffffffff], 1);
    const outputs = ['public', 'local', 'gather'].map(name =>
      test.floats(name, [99, 99, 99, 99, 888, 889], 1)
    );
    test.graph.add({
      getCommandNodes: graph => [
        createSegmentSumNode(graph, {
          id: 'public',
          operation: 'ParitySum',
          segmentCount: 4,
          input,
          segmentOffsets: offsets,
          output: outputs[0]
        }),
        createSegmentSumNode(graph, {
          id: 'local',
          operation: 'ParitySum',
          segmentCount: 4,
          input,
          segmentOffsets: offsets,
          output: outputs[1],
          extraResources: []
        }),
        ...getSortedSegmentSumNodes(graph, {
          id: 'gather',
          operation: 'ParitySum',
          segmentCount: 4,
          segmentKeys: test.uints('keys', fixture.keys, 1),
          reductions: [{name: 'total', contributions: gatherInput, output: outputs[2]}]
        })
      ]
    });
    test.run();
    for (const output of outputs) {
      const actual = await test.readFloatStorage(output);
      expect(actual.every(Number.isFinite)).toBe(true);
      expect(actual).toEqual([12345, ...fixture.expected, 888, 889, 12345]);
    }
    expect(await test.readFloatStorage(input)).toEqual([12345, ...values, 12345]);
    expect(await test.readUints(offsets)).toEqual([0, 0, 257, 257, 259, 0xffffffff]);
  } finally {
    test.destroy();
  }
});
