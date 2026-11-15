// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {createStableLexicographicIndexSortNodes} from '../../src/analysis-core/kernels/stable-lexicographic-index-sort';
import {createWGSLKernelNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from './gpu-contributor-test-utils';
import {requireTestDevice} from './require-test-device';

const SORT_CASES = [
  {
    name: 'eight rows preserve lower-order tuple components',
    primary: [1, 0, 1, 0, 1, 0, 2, 2],
    secondary: [0, 1, 0, 1, 1, 0, 0, 0],
    tertiary: [2, 3, 1, 1, 0, 9, 0, 0],
    indices: [0, 1, 2, 3, 4, 5, 6, 7],
    sharedInputs: false
  },
  {
    name: 'eleven rows use offset views in one shared input buffer',
    primary: [1, 0, 1, 0, 1, 0, 2, 2, 0, 2, 1],
    secondary: [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    tertiary: [2, 3, 1, 1, 0, 9, 0, 0, 1, 0, 2],
    // Equal tuples must retain this incoming order, including rows 7, 6 and 9.
    indices: [7, 6, 9, 0, 10, 3, 5, 1, 2, 8, 4],
    sharedInputs: true
  },
  {
    name: 'full-width words distinguish all-ones low words from unused slots',
    primary: [0xffffffff, 0, 1, 0, 0xffffffff, 1, 0xffffffff],
    secondary: [0xffffffff, 0xffffffff, 0, 0, 0, 0, 0xffffffff],
    tertiary: [0xffffffff, 7, 9, 8, 4, 9, 0xffffffff],
    indices: [6, 0, 5, 2, 1, 4, 3],
    sharedInputs: true,
    keyBits: [32, 32, 32]
  },
  {
    name: 'exactly 32 packed bits restore raw primary words and preserve ties',
    primary: [3, 0, 3, 0xffffffff, 1, 3, 0xffffffff],
    secondary: [0x3fffffff, 4, 0, 0x3fffffff, 0, 0, 0x3fffffff],
    tertiary: [8, 3, 5, 2, 6, 5, 2],
    indices: [6, 5, 3, 2, 4, 1, 0],
    sharedInputs: true,
    keyBits: [2, 30, 32]
  },
  {
    name: 'one row returns its original index and primary key',
    primary: [3],
    secondary: [1],
    tertiary: [7],
    indices: [0],
    sharedInputs: false
  },
  {
    name: 'zero rows return empty columns without writing output storage',
    primary: [],
    secondary: [],
    tertiary: [],
    indices: [],
    sharedInputs: false
  }
];

describe.each([
  {name: 'unpacked', packed: false, identity: false},
  {name: 'packed', packed: true, identity: false},
  {name: 'identity unpacked', packed: false, identity: true},
  {name: 'identity packed', packed: true, identity: true}
])('stable lexicographic index sort with $name keys', ({name, packed, identity}) => {
  it.each(SORT_CASES)('$name', async testCase => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const length = testCase.indices.length;
    const indices = identity ? Array.from({length}, (_, row) => row) : testCase.indices;
    const columns = [testCase.primary, testCase.secondary, testCase.tertiary, indices];
    const outputBuffers = [createOutputBuffer(device, length), createOutputBuffer(device, length)];
    // Empty logical columns still need physical storage for guarded kernels and readback.
    const sentinel = 0xdeadbeef;
    for (const buffer of outputBuffers)
      buffer.write(new Uint32Array(Math.max(length, 1)).fill(sentinel));
    const inputBuffers = testCase.sharedInputs
      ? [
          createInputBuffer(
            device,
            new Uint32Array(columns.flatMap(values => [sentinel, ...values, sentinel]))
          )
        ]
      : columns.map(values =>
          createInputBuffer(device, new Uint32Array(values.length ? values : [sentinel]))
        );
    try {
      const sharedStorage = testCase.sharedInputs
        ? importGraphBuffer(graph, 'shared-input', inputBuffers[0], 'uint32')
        : undefined;
      const views = columns.map((_, columnIndex) =>
        sharedStorage
          ? graph.createDataView(sharedStorage.buffer, {
              format: 'uint32',
              length,
              byteOffset: (columnIndex * (length + 2) + 1) * Uint32Array.BYTES_PER_ELEMENT
            })
          : importGraphBuffer(
              graph,
              `input-${columnIndex}`,
              inputBuffers[columnIndex],
              'uint32',
              length
            )
      );
      if (testCase.sharedInputs) {
        expect(new Set(views.map(view => view.buffer)).size).toBe(1);
        for (const view of views) expect(view.byteOffset).toBeGreaterThan(0);
      }
      const output = importGraphBuffer(graph, 'output', outputBuffers[0], 'uint32', length);
      const primaryOutput = importGraphBuffer(
        graph,
        'primary-output',
        outputBuffers[1],
        'uint32',
        length
      );
      const keyBits = testCase.keyBits ?? [2, 1, 4];
      const sorted = createStableLexicographicIndexSortNodes(graph, {
        id: name,
        operation: 'TupleTest',
        indices: views[3],
        indicesAreIdentity: identity,
        keys: views.slice(0, 3).map((view, keyIndex) => ({
          view,
          ...(packed ? {keyBits: keyBits[keyIndex]} : {})
        }))
      });
      const nodeIds = sorted.nodes.map(node => node.id);
      const passCount = !packed || keyBits[0] === 32 ? 3 : keyBits[2] === 32 ? 2 : 1;
      const primaryPacked = packed && keyBits[0] !== 32;
      const firstPacked = packed && passCount === 1;
      expect(nodeIds.filter(nodeId => /-pass-\d+-gather$/.test(nodeId))).toHaveLength(
        length
          ? passCount - (identity && !firstPacked ? 1 : 0)
          : firstPacked || primaryPacked
            ? 1
            : 0
      );
      expect(nodeIds.includes(`${name}-primary-gather`)).toBe(primaryPacked && length > 0);
      if (length) {
        for (let pass = 0; pass < passCount; pass++) {
          expect(nodeIds.some(nodeId => nodeId.startsWith(`${name}-pass-${pass}-sort`))).toBe(true);
        }
        expect(nodeIds.some(nodeId => nodeId.startsWith(`${name}-pass-${passCount}-sort`))).toBe(
          false
        );
      }
      expect(sorted.sortedIndices.length).toBe(length);
      expect(sorted.sortedPrimaryKeys.length).toBe(length);
      graph.add([
        ...sorted.nodes,
        createWGSLKernelNode(graph, {
          id: 'copy-result',
          operation: 'TupleTest',
          variant: 'copy',
          bindings: [
            {name: 'input', view: sorted.sortedIndices, type: 'u32', access: 'read'},
            {name: 'primary', view: sorted.sortedPrimaryKeys, type: 'u32', access: 'read'},
            {name: 'output', view: output, type: 'u32', access: 'read_write'},
            {name: 'primaryOutput', view: primaryOutput, type: 'u32', access: 'read_write'}
          ],
          invocationCount: length,
          body: `output[outputOffset + index] = input[inputOffset + index];
  primaryOutput[primaryOutputOffset + index] = primary[primaryOffset + index];`
        })
      ]);
      const expectedIndices = indices
        .map((row, position) => ({row, position}))
        .sort((left, right) => {
          for (const [columnIndex, column] of columns.slice(0, 3).entries()) {
            const mask = packed ? 2 ** keyBits[columnIndex] - 1 : 0xffffffff;
            const difference =
              ((column[left.row] & mask) >>> 0) - ((column[right.row] & mask) >>> 0);
            if (difference) return difference;
          }
          return left.position - right.position;
        })
        .map(({row}) => row);
      const expectedPrimaryKeys = expectedIndices.map(row => testCase.primary[row]);
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(outputBuffers[0], length)).toEqual(expectedIndices);
        expect(await readUint32(outputBuffers[1], length)).toEqual(expectedPrimaryKeys);
        if (length === 0) {
          for (const buffer of outputBuffers) expect(await readUint32(buffer)).toEqual([sentinel]);
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      for (const buffer of [...inputBuffers, ...outputBuffers]) buffer.destroy();
    }
  });
});
