// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUTrajectorySplit} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {recordWGSLKernelNodes} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {computeTrajectorySplitOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-split-operation';

// Same hand-derived literals as the headless spec: three two-row runs require
// three segments and six rows; isolated singleton runs produce nothing.
for (const scenario of [
  {name: 'exact fit', segmentCapacity: 3, rowCapacity: 6, empty: false, status: [3, 3, 0]},
  {name: 'segment loss', segmentCapacity: 2, rowCapacity: 6, empty: false, status: [2, 3, 1]},
  {name: 'row loss', segmentCapacity: 3, rowCapacity: 5, empty: false, status: [3, 3, 1]},
  {name: 'singletons dropped', segmentCapacity: 3, rowCapacity: 6, empty: true, status: [0, 0, 0]}
]) {
  it(`CPU split publishes independently derived status: ${scenario.name}`, () => {
    const result = computeTrajectorySplitOnCPU(
      {
        positions: new Float32Array(12),
        timestamps: new Float32Array(scenario.empty ? [0, 4, 8, 12, 16, 20] : [0, 1, 4, 5, 9, 10]),
        trackOffsets: new Uint32Array([0, 6]),
        parameters: new Float32Array([2, 0, 0, 0])
      },
      {mode: 'gap', segmentCapacity: scenario.segmentCapacity, rowCapacity: scenario.rowCapacity}
    );
    expect([...result.count, ...result.requiredCount, ...result.overflow]).toEqual(scenario.status);
  });
}

it('captures inherited stop loss and executes its dispatch from the original view', () => {
  const context = createKernelAuditContext('split-inherited-loss');
  try {
    const {graph, view} = context;
    let reads = 0;
    const overflow = view('uint32', 1);
    const unused = view('uint32', 1);
    const staypointStatus = new (class {
      get overflow() {
        reads++;
        return reads === 1 ? overflow : unused;
      }
    })();
    const candidateOverflow = view('uint32', 1);
    recordWGSLKernelNodes(graph);
    const nodes = new GPUTrajectorySplit({
      mode: 'stop',
      positions: view('float32x2', 2),
      timestamps: view('float32', 2),
      trackOffsets: view('uint32', 2),
      staypointIds: view('uint32', 2),
      staypointStatus,
      output: {
        segmentOffsets: view('uint32', 2),
        segmentTracks: view('uint32', 1),
        rowIds: view('uint32', 2),
        status: {count: view('uint32', 1), overflow: view('uint32', 1), candidateOverflow}
      }
    }).getCommandNodes(graph);
    const status = nodes.find(node => node.id === 'trajectory-split-staypoint-loss-status')!;
    const memory = createCPUGraphMemory();
    memory.write(overflow, Uint32Array.of(1));
    memory.write(unused, Uint32Array.of(0));
    executeCommandNodesOnCPU([status], memory);
    expect(memory.readUint32(candidateOverflow)).toEqual(Uint32Array.of(1));
    expect(reads).toBe(1);
  } finally {
    context.device.destroy();
  }
});

it('accepts adjacent split status words and rejects overlap across their dispatches', () => {
  const context = createKernelAuditContext('split-packed-status');
  try {
    const {graph, view} = context;
    const packed = view('uint32', 2);
    const count = graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
    const overflow = view('uint32', 1);
    const props = {
      mode: 'stop',
      positions: view('float32x2', 2),
      timestamps: view('float32', 2),
      trackOffsets: view('uint32', 2),
      staypointIds: view('uint32', 2)
    } as const;
    const output = {
      segmentOffsets: view('uint32', 2),
      segmentTracks: view('uint32', 1),
      rowIds: view('uint32', 2)
    };
    const declare = (byteOffset: number) =>
      new GPUTrajectorySplit({
        ...props,
        output: {
          ...output,
          status: {
            count,
            overflow,
            candidateOverflow: graph.createDataView(packed.buffer, {
              format: 'uint32',
              length: 1,
              byteOffset
            })
          }
        }
      }).getCommandNodes(graph);
    expect(() => declare(0)).toThrow(/overlap/);
    expect(() => declare(4)).not.toThrow();
  } finally {
    context.device.destroy();
  }
});
