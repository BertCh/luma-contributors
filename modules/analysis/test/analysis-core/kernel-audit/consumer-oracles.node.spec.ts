// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {createKernelAuditContext} from './kernel-audit-context';
import {expectJoinedFloatAggregates, expectNearestTieResult} from './consumer-audit-expectations';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';

it('review B1 device oracle accepts unmatched floating sums and means from the publication kernels', () => {
  const context = createKernelAuditContext('join-unmatched-oracle');
  const {view, graph} = context;
  try {
    const sums = view('float32', 1);
    const means = view('float32', 1);
    const words = view('uint32x2', 1);
    const values = view('float32', 1);
    const join = new GPUKeyLookup({
      leftKeys: view('uint32', 1),
      rightKeys: view('uint32', 1),
      aggregates: [
        {operation: 'sum', column: values, output: sums, sums: words},
        {operation: 'mean', column: values, output: means}
      ],
      output: {rightRows: view('uint32', 1)}
    });
    const nodes = join
      .getCommandNodes(graph)
      .filter(node => /aggregate-[01]-finish$/.test(node.id));
    expect(nodes).toHaveLength(2);
    const leftUnique = nodes[0].resources?.find(
      resource =>
        'buffer' in resource &&
        resource.buffer instanceof GraphDataView &&
        resource.buffer.buffer.id.endsWith('-left-unique')
    );
    if (!leftUnique || !('buffer' in leftUnique) || !(leftUnique.buffer instanceof GraphDataView))
      throw new Error('missing leftUnique publication input');
    const memory = createCPUGraphMemory();
    memory.write(leftUnique.buffer, Uint32Array.of(0xffffffff));
    memory.write(words, Uint32Array.of(99, 99));
    executeCommandNodesOnCPU(nodes, memory);
    // The absent unique key publishes missing floating values, but zero fixed-point words.
    // Matched row literals come independently from sum([1,2])=3 and mean([1,2])=3/2.
    expectJoinedFloatAggregates(
      [3, memory.readFloat32(sums)[0]],
      [1.5, memory.readFloat32(means)[0]]
    );
    expect(Array.from(memory.readUint32(words))).toEqual([0, 0]);
  } finally {
    context.device.destroy();
  }
});

for (const ties of ['lowest-row'] as const) {
  it(`review S1 rejects an all-ties loss flag under ${ties}`, () => {
    expect(() =>
      expectNearestTieResult({
        ties,
        capacity: 1,
        count: 1,
        rows: [0],
        ids: [99],
        distances: [1],
        overflow: 1
      })
    ).toThrow();
  });
}
it('review S1 requires overflow when all ties cannot fit one slot', () => {
  expect(() =>
    expectNearestTieResult({
      ties: 'all',
      capacity: 1,
      count: 1,
      rows: [0],
      ids: [99],
      distances: [1],
      overflow: 0
    })
  ).toThrow();
  expectNearestTieResult({
    ties: 'all',
    capacity: 1,
    count: 1,
    rows: [0],
    ids: [99],
    distances: [1],
    overflow: 1
  });
});
