// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect, WgslExec, WgslParser} from 'wgsl_reflect';
import {getSumOverflowNodes} from '../../../src/dataframe-analysis/key-join/sum-overflow-nodes';
import {GPUGroupStatistics} from '../../../src/dataframe-analysis/group-statistics/gpu-group-statistics';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it.each([
  undefined,
  4
])('DGGS-102 declares group numeric-loss publication independently of capacity (tile=%s)', scratchTileSize => {
  const device = createNullWebGPUDevice();
  try {
    for (const capacity of [4, 257]) {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `sum-${serial++}`, format, length);
      const nodes = new GPUGroupStatistics({
        keys: view('uint32', 12),
        scratchTileSize,
        columns: [
          {
            values: view('float32', 12),
            statistics: ['sum'],
            output: {sums: view('uint32x2', capacity), status: {sumOverflow: view('uint32', 1)}}
          }
        ],
        output: {
          keys: view('uint32', capacity),
          counts: view('uint32', capacity),
          status: {count: view('uint32', 1), overflow: view('uint32', 1)}
        }
      }).getCommandNodes(graph);
      const kernels = getGPUCommandNodeKernelSources(nodes).filter(
        kernel =>
          kernel.source.includes('MAXIMUM_SCALED_VALUE') || kernel.source.includes('sumOverflow')
      );
      expect(kernels.some(kernel => kernel.source.includes('guardCarry'))).toBe(true);
      expect(kernels.some(kernel => kernel.source.includes('sumOverflow[sumOverflowOffset]'))).toBe(
        true
      );
      for (const kernel of kernels) expect(() => new WgslReflect(kernel.source)).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('DGGS-102 key join validates and publishes caller-scale sum loss', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `join-${serial++}`, format, length);
    const aggregate = {
      operation: 'sum' as const,
      column: view('float32', 4),
      output: view('float32', 3),
      sums: view('uint32x2', 3),
      status: {sumOverflow: view('uint32', 1)}
    };
    const props = {
      leftKeys: view('uint32', 3),
      rightKeys: view('uint32', 4),
      aggregates: [aggregate],
      output: {}
    };
    const nodes = new GPUKeyLookup(props).getCommandNodes(graph);
    expect(
      getGPUCommandNodeKernelSources(nodes).some(kernel =>
        kernel.source.includes('sumOverflow[sumOverflowOffset]')
      )
    ).toBe(true);
    expect(
      () =>
        new GPUKeyLookup({
          ...props,
          aggregates: [{...aggregate, status: {sumOverflow: view('uint32', 2)}}]
        })
    ).toThrow(/sumOverflow.*one uint32 row/);
  } finally {
    device.destroy();
  }
});

it.each([
  {value: 1e14, count: 1, scale: 65536, expected: 1},
  {value: 7e13, count: 3, scale: 65536, expected: 1},
  {value: -1e14, count: 1, scale: 65536, expected: 1},
  {value: -7e13, count: 3, scale: 65536, expected: 1},
  {value: 1e14, count: 1, scale: 1, expected: 0},
  {value: 7e13, count: 3, scale: 1, expected: 0},
  {value: -7e13, count: 3, scale: 1, expected: 0}
])('DGGS-102 host WGSL replay flags exact integer loss: $value x $count at $scale', ({
  value,
  count,
  scale,
  expected
}) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const values = createTransientView(graph, 'values', 'float32', count);
    const groups = createTransientView(graph, 'groups', 'uint32', count);
    const output = createTransientView(graph, 'output', 'uint32', 1);
    const kernels = getGPUCommandNodeKernelSources(
      getSumOverflowNodes(graph, {
        id: 'replay',
        capacity: 1,
        sumScale: scale,
        chunks: [{values, groups}],
        output
      })
    );
    // Literal expectation: at 65536 one uploaded 1e14 exceeds the per-row clamp, and
    // three uploaded 7e13 total 13762560381715218432 > 2^63-1. Scale 1 is in range.
    // This replays WGSL on the host; it is not a device qualification.
    const floating = new Float32Array(count).fill(value);
    const buffers: Record<string, Uint32Array> = {
      values: new Uint32Array(floating.buffer),
      groups: new Uint32Array(count),
      words: new Uint32Array(2),
      guards: new Uint32Array(2),
      sumOverflow: new Uint32Array(1).fill(9)
    };
    for (const kernel of kernels) {
      const bindings: Record<number, Uint32Array> = {};
      for (const binding of new WgslReflect(kernel.source).storage)
        bindings[binding.binding] = buffers[binding.name];
      new WgslExec(
        new WgslParser().parse(kernel.source.replace(/^\s*_ = [^;]+;\s*$/gm, ''))
      ).dispatchWorkgroups('main', 1, {0: bindings}, {constants: kernel.constants});
    }
    expect(Array.from(buffers['sumOverflow'])).toEqual([expected]);
  } finally {
    device.destroy();
  }
});

it.each([
  {left: [4], mask: [1], expected: 0},
  {left: [4, 9], mask: [1, 1], expected: 1},
  {left: [4, 9], mask: [1, 0], expected: 0}
])('B3 join loss follows retained matches: $left mask $mask', ({left, mask, expected}) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `retained-${serial++}`, format, length);
  try {
    const kernels = getGPUCommandNodeKernelSources(
      new GPUKeyLookup({
        leftKeys: view('uint32', left.length),
        leftMask: view('uint32', left.length),
        rightKeys: view('uint32', 2),
        sumScale: 65536,
        aggregates: [
          {
            operation: 'sum',
            column: view('float32', 2),
            output: view('float32', left.length),
            sums: view('uint32x2', left.length),
            status: {sumOverflow: view('uint32', 1)}
          }
        ],
        output: {}
      }).getCommandNodes(graph)
    ).filter(kernel => kernel.id.endsWith('-probe') || kernel.id.includes('-sum-overflow-'));
    // Only key 4 is retained in the first/third fixture: its exact sum is 2*65536=131072.
    // Key 9's 2^46 * 65536 = 2^62 row clamps, but must affect status only if retained.
    const values = Float32Array.of(2, 2 ** 46);
    const retained = new Uint32Array(2);
    const buffers: Record<string, Uint32Array> = {
      leftKeys: Uint32Array.from(left),
      leftMask: Uint32Array.from(mask),
      uniqueLow: Uint32Array.of(4, 9),
      uniqueTotal: Uint32Array.of(2),
      leftUnique: new Uint32Array(left.length),
      uniqueMatched: retained,
      retainedGroups: retained,
      values: new Uint32Array(values.buffer),
      groups: Uint32Array.of(0, 1),
      permutation: Uint32Array.of(0, 1),
      words: new Uint32Array(4),
      guards: new Uint32Array(3),
      sumOverflow: new Uint32Array(1).fill(9)
    };
    for (const kernel of kernels) {
      const bindings: Record<number, Uint32Array> = {};
      for (const binding of new WgslReflect(kernel.source).storage)
        bindings[binding.binding] = buffers[binding.name];
      new WgslExec(
        new WgslParser().parse(kernel.source.replace(/^\s*_ = [^;]+;\s*$/gm, ''))
      ).dispatchWorkgroups('main', 1, {0: bindings}, {constants: kernel.constants});
    }
    expect(Array.from(buffers['sumOverflow'])).toEqual([expected]);
  } finally {
    device.destroy();
  }
});

it.each([
  {values: [2 ** 61, 2 ** 61, 2 ** 61, 2 ** 61], expected: 1},
  {values: [2 ** 61, 2 ** 61, 2 ** 61, 2 ** 61, -(2 ** 61)], expected: 0},
  {values: [1e19], expected: 1}
])('B5 separate loss fixtures replay all chunk guards before publication: $expected', ({
  values,
  expected
}) => {
  const device = createNullWebGPUDevice();
  try {
    for (const capacity of [4, 257]) {
      const graph = new GPUCommandGraph(device);
      const groups = createTransientView(graph, 'groups', 'uint32', values.length);
      const chunks = Array.from({length: Math.ceil(values.length / 2)}, (_, index) => {
        const payload = Float32Array.from(values.slice(2 * index, 2 * index + 2));
        return {
          payload,
          values: createTransientView(graph, `chunk-${index}`, 'float32', payload.length),
          groups,
          rowOffset: 2 * index
        };
      });
      const kernels = getGPUCommandNodeKernelSources(
        getSumOverflowNodes(graph, {
          id: 'cross-tile',
          capacity,
          sumScale: 1,
          chunks,
          useGroupDispatchUniforms: true,
          output: createTransientView(graph, 'loss', 'uint32', 1)
        })
      );
      const replay = (dropFinalCheck = false) => {
        const words = new Uint32Array(2 * capacity);
        const guards = new Uint32Array(capacity + 1);
        const loss = Uint32Array.of(9);
        for (const kernel of kernels) {
          if (dropFinalCheck && kernel.id.endsWith('-check')) continue;
          const chunkIndex = Number(/accumulate-(\d+)$/.exec(kernel.id)?.[1] ?? 0);
          const chunk = chunks[chunkIndex];
          const parameters = new Uint32Array(16);
          parameters.set([chunk.payload.length, 1, 1, chunk.rowOffset]);
          const buffers: Record<string, Uint32Array> = {
            values: new Uint32Array(chunk.payload.buffer),
            groups: new Uint32Array(values.length),
            words,
            guards,
            sumOverflow: loss,
            groupParameters: parameters
          };
          const reflection = new WgslReflect(kernel.source);
          const bindings: Record<number, Uint32Array> = {};
          for (const binding of [...reflection.storage, ...reflection.uniforms])
            bindings[binding.binding] = buffers[binding.name];
          // Pointer-use discard statements are semantically inert; the interpreter logs them.
          const source = kernel.source
            .replace(/^\s*_ = [^;]+;\s*$/gm, '')
            // The interpreter cannot index a vector inside a uniform array. Substitute only
            // these transport reads with the exact invocation/offset record for this chunk.
            .replace(
              /groupParameters\[(\d+)\]\[(\d+)\]/g,
              (_match, vector, lane) => `${parameters[4 * Number(vector) + Number(lane)]}u`
            )
            .replace(
              /groupParameters\[(\d+)\]\.([xyzw])/g,
              (_match, vector, lane) => `${parameters[4 * Number(vector) + 'xyzw'.indexOf(lane)]}u`
            );
          new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
            'main',
            1,
            {0: bindings},
            {constants: kernel.constants}
          );
        }
        expect(Array.from(words.slice(2))).toEqual(new Array<number>(2 * capacity - 2).fill(0));
        return loss[0];
      };
      expect(replay()).toBe(expected);
      if (values.length === 4) {
        // Mutation check: unlike the earlier mixed fixture, omitting the cross-chunk total
        // check must fail the literal loss=1 expectation; there is no clamped row to hide it.
        expect(replay(true)).toBe(0);
      }
    }
  } finally {
    device.destroy();
  }
});
