// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {GPUTrajectorySplit} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {expectSplitRowPrefix} from './movement-output-assertions.test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';

useCPUExecutorTestHooks();

for (const offsets of [[0], [0, 0], [0, 2]]) {
  it(`t1 split defines offsets and the readable row prefix for offsets ${offsets}`, () => {
    const context = createKernelAuditContext('t1-empty-split');
    try {
      const rowCount = offsets[offsets.length - 1];
      const rowCapacity = Math.max(1, rowCount);
      const positions = context.view('float32x2', rowCount);
      const timestamps = context.view('float32', rowCount);
      const trackOffsets = context.view('uint32', offsets.length);
      const parameters = context.view('float32', 4);
      const segmentOffsets = context.view('uint32', 2);
      const segmentTracks = context.view('uint32', 1);
      const rowIds = context.view('uint32', rowCapacity);
      const count = context.view('uint32', 1);
      const overflow = context.view('uint32', 1);
      const nodes = new GPUTrajectorySplit({
        mode: 'gap',
        positions,
        timestamps,
        trackOffsets,
        parameters,
        output: {segmentOffsets, segmentTracks, rowIds, status: {count, overflow}}
      }).getCommandNodes(context.graph);
      const memory = createCPUGraphMemory();
      memory.write(positions, new Float32Array(rowCount * 2));
      memory.write(
        timestamps,
        Float32Array.from({length: rowCount}, (_, row) => row)
      );
      memory.write(trackOffsets, Uint32Array.from(offsets));
      memory.write(parameters, Float32Array.of(10, 0, 0, 0));
      for (const view of [segmentOffsets, segmentTracks, rowIds, count, overflow])
        memory.write(view, new Uint32Array(view.length).fill(93));
      executeCommandNodesOnCPU(nodes, memory);
      const empty = rowCount === 0;
      expect(Array.from(memory.readUint32(count))).toEqual([empty ? 0 : 1]);
      expect(Array.from(memory.readUint32(overflow))).toEqual([0]);
      expect(Array.from(memory.readUint32(segmentOffsets))).toEqual(
        empty ? [0, 0xffffffff] : [0, 2]
      );
      expect(Array.from(memory.readUint32(segmentTracks))).toEqual([empty ? 0xffffffff : 0]);
      // Two stationary fixes one tick apart form one retained two-row segment.
      // With no fixes there is no readable row, regardless of spare capacity bytes.
      expectSplitRowPrefix(
        memory.readUint32(rowIds),
        memory.readUint32(segmentOffsets),
        memory.readUint32(count)[0],
        rowCapacity,
        empty ? [] : [0, 1]
      );
    } finally {
      context.device.destroy();
    }
  });
}

it('t1 split prefix assertions reject a corrupt terminal or live row without inspecting tails', () => {
  expect(() =>
    expectSplitRowPrefix(Uint32Array.of(0), Uint32Array.of(1, 0xffffffff), 0, 1, [])
  ).toThrow();
  expect(() =>
    expectSplitRowPrefix(Uint32Array.of(1, 0), Uint32Array.of(0, 2), 1, 2, [0, 1])
  ).toThrow();
  expect(() =>
    expectSplitRowPrefix(Uint32Array.of(0), Uint32Array.of(0, 0xffffffff), 0, 1, [])
  ).not.toThrow();
  expect(() =>
    expectSplitRowPrefix(Uint32Array.of(93), Uint32Array.of(0, 0xffffffff), 0, 1, [])
  ).not.toThrow();
});
