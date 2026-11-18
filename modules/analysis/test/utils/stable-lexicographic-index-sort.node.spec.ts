// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {useCPUExecutorTestHooks} from '../gpu-spatial-analysis/segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {createStableLexicographicIndexSortNodes} from '../../src/analysis-core/kernels/stable-lexicographic-index-sort';
import {
  getKeyPairSortNodes,
  KEY_PAIR_SORT_WORDS_PER_ITEM
} from '../../src/gpu-spatial-analysis/spatial-weights/key-pair-grouping';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from './gpu-contributor-test-utils';

function setup(length: number) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const indices = createTransientView(graph, 'indices', 'uint32', length);
  const primary = createTransientView(graph, 'primary', 'uint32', length);
  const secondary = createTransientView(graph, 'secondary', 'uint32', length);
  return {device, graph, indices, primary, secondary};
}

it('identity order skips only the first unpacked gather and keeps the two stable passes', () => {
  const {device, graph, indices, primary, secondary} = setup(11);
  try {
    const sorted = createStableLexicographicIndexSortNodes(graph, {
      id: 'identity',
      operation: 'TupleTest',
      indices,
      indicesAreIdentity: true,
      keys: [{view: primary}, {view: secondary}]
    });
    const identifiers = sorted.nodes.map(node => node.id);
    expect(identifiers).not.toContain('identity-pass-0-gather');
    expect(identifiers).toContain('identity-pass-1-gather');
    expect(identifiers.some(identifier => identifier.startsWith('identity-pass-0-sort'))).toBe(
      true
    );
    expect(identifiers.some(identifier => identifier.startsWith('identity-pass-1-sort'))).toBe(
      true
    );
    expect(sorted.sortedIndices.length).toBe(11);
    expect(sorted.sortedPrimaryKeys.length).toBe(11);
  } finally {
    device.destroy();
  }
});

it('an existing row permutation gathers the least significant key before sorting', () => {
  const {device, graph, indices, primary, secondary} = setup(11);
  try {
    const sorted = createStableLexicographicIndexSortNodes(graph, {
      id: 'permuted',
      operation: 'TupleTest',
      indices,
      keys: [{view: primary}, {view: secondary}]
    });
    expect(sorted.nodes[0].id).toBe('permuted-pass-0-gather');
    expect(sorted.nodes.some(node => node.id === 'permuted-pass-1-gather')).toBe(true);
  } finally {
    device.destroy();
  }
});

it('packs exactly 32 bits into one pass and restores the raw primary key', () => {
  const {device, graph, indices, primary, secondary} = setup(11);
  try {
    const sorted = createStableLexicographicIndexSortNodes(graph, {
      id: 'packed',
      operation: 'TupleTest',
      indices,
      indicesAreIdentity: true,
      keys: [
        {view: primary, keyBits: 2},
        {view: secondary, keyBits: 30}
      ]
    });
    const identifiers = sorted.nodes.map(node => node.id);
    expect(identifiers[0]).toBe('packed-pass-0-gather');
    expect(identifiers.some(identifier => identifier.startsWith('packed-pass-1-sort'))).toBe(false);
    expect(identifiers.at(-1)).toBe('packed-primary-gather');
    for (const {id: identifier, source} of getParsedKernelSources(sorted.nodes).configurations) {
      expect(getStorageBindingCount(source), identifier).toBeLessThanOrEqual(8);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  0, 1, 11, 1025
])('key-pair sorting retains its exact scratch plan for %i items', length => {
  const {device, graph, primary, secondary} = setup(length);
  const createBuffer = graph.createTransientBuffer.bind(graph);
  let words = 0;
  graph.createTransientBuffer = descriptor => {
    if (!/-pass-\d+-sort-/.test(descriptor.id)) words += descriptor.byteLength / 4;
    return createBuffer(descriptor);
  };
  try {
    const sorted = getKeyPairSortNodes(graph, 'pair', 'TupleTest', length, primary, secondary, {
      high: 3,
      low: 4
    });
    // Even narrow keys retain two passes here: capacity plans promise six declared words/item.
    expect(words).toBe(KEY_PAIR_SORT_WORDS_PER_ITEM * Math.max(length, 1));
    expect(sorted.sortedItems.length).toBe(length);
    if (length) {
      expect(sorted.nodes.some(node => node.id.startsWith('pair-sort-pass-1-sort'))).toBe(true);
    }
    expect(new Set(sorted.nodes.map(node => node.id)).size).toBe(sorted.nodes.length);
  } finally {
    graph.createTransientBuffer = createBuffer;
    device.destroy();
  }
});

describe('lexicographic sort validates key shape and bit budgets before declaring scratch', () => {
  it.each([0, 33, 1.5, NaN])('rejects keyBits %s', keyBits => {
    const {device, graph, indices, primary} = setup(11);
    try {
      expect(() =>
        createStableLexicographicIndexSortNodes(graph, {
          id: 'limits',
          operation: 'TupleTest',
          indices,
          keys: [{view: primary, keyBits}]
        })
      ).toThrow(/^limits keys\[0\].keyBits must be an integer in \[1, 32\]/);
    } finally {
      device.destroy();
    }
  });

  it('rejects missing keys and mismatched column lengths', () => {
    const {device, graph, indices} = setup(11);
    try {
      expect(() =>
        createStableLexicographicIndexSortNodes(graph, {
          id: 'limits',
          operation: 'TupleTest',
          indices,
          keys: []
        })
      ).toThrow(/^limits keys.length/);
      const shorter = createTransientView(graph, 'shorter', 'uint32', 10);
      expect(() =>
        createStableLexicographicIndexSortNodes(graph, {
          id: 'limits',
          operation: 'TupleTest',
          indices,
          keys: [{view: shorter}]
        })
      ).toThrow(/^limits keys\[0\].view.length must equal indices.length \(11\)/);
    } finally {
      device.destroy();
    }
  });
});

// Execute the real gather/pack WGSL around the trusted published sort's CPU stand-in. The
// comparison oracle sorts whole tuples directly, independently of least-to-most stable passes.
describe('lexicographic gather and pack results on the CPU interpreter', () => {
  useCPUExecutorTestHooks();

  for (const packed of [false, true]) {
    for (const identity of [false, true]) {
      it(`matches tuple ordering with packed=${packed} and identity=${identity}`, () => {
        const {device, graph} = setup(0);
        try {
          const primary = [3, 0, 3, 0xffffffff, 1, 3, 0xffffffff];
          const secondary = [0x3fffffff, 4, 0, 0x3fffffff, 0, 0, 0x3fffffff];
          const incoming = identity ? [0, 1, 2, 3, 4, 5, 6] : [6, 5, 3, 2, 4, 1, 0];
          const length = incoming.length;
          const sentinel = 0xdeadbeef;
          const storage = createTransientView(graph, 'shared', 'uint32', 3 * (length + 2));
          const columns = [primary, secondary, incoming];
          const views = columns.map((_, column) =>
            graph.createDataView(storage.buffer, {
              format: 'uint32',
              length,
              byteOffset: (column * (length + 2) + 1) * 4
            })
          );
          const memory = createCPUGraphMemory();
          memory.write(
            storage,
            new Uint32Array(columns.flatMap(values => [sentinel, ...values, sentinel]))
          );
          const sorted = createStableLexicographicIndexSortNodes(graph, {
            id: 'interpreted',
            operation: 'TupleTest',
            indices: views[2],
            indicesAreIdentity: identity,
            keys: packed
              ? [
                  {view: views[0], keyBits: 2},
                  {view: views[1], keyBits: 30}
                ]
              : [{view: views[0]}, {view: views[1]}]
          });
          const expected = incoming.slice().sort((left, right) => {
            const primaryLeft = packed ? primary[left] % 4 : primary[left];
            const primaryRight = packed ? primary[right] % 4 : primary[right];
            return primaryLeft - primaryRight || secondary[left] - secondary[right];
          });
          executeCommandNodesOnCPU(sorted.nodes, memory);
          expect(Array.from(memory.readUint32(sorted.sortedIndices))).toEqual(expected);
          expect(Array.from(memory.readUint32(sorted.sortedPrimaryKeys))).toEqual(
            expected.map(row => primary[row])
          );
          expect(Array.from(memory.readUint32(storage))).toEqual(
            columns.flatMap(values => [sentinel, ...values, sentinel])
          );
        } finally {
          device.destroy();
        }
      });
    }
  }
});
