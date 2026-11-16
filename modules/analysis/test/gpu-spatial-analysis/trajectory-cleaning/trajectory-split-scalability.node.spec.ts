// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTrajectorySplit,
  getGPUTrajectorySplitParameterValues,
  type GPUTrajectorySplitProps
} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  expectPackedTrajectorySplitReadback,
  expectSaturatedTrajectorySplitReadback
} from './trajectory-split-scalability-assertions';

// Use the real public readback helper, with only its GPU byte transfer replaced.
function readTransferredRows(values: readonly number[]): Promise<number[]> {
  const bytes = new Uint8Array(Uint32Array.from(values).buffer);
  return readUint32({readAsync: async () => bytes} as Buffer, values.length);
}

it('checks packed headless expectations against the public array readback contract', async () => {
  expectPackedTrajectorySplitReadback({
    packed: await readTransferredRows([0, 1]),
    offsets: await readTransferredRows([0, 2]),
    rows: await readTransferredRows([0, 1]),
    overflow: await readTransferredRows([0])
  });
});

it.each([
  false,
  true
])('checks saturated headless expectations against array readback (segments=%s)', async segmentOverflow => {
  expectSaturatedTrajectorySplitReadback(
    {
      written: await readTransferredRows(segmentOverflow ? [99, 99] : [0, 1, 99, 99]),
      prefixes: await readTransferredRows([0, 0xffffffff]),
      count: await readTransferredRows([2]),
      required: await readTransferredRows([0xffffffff]),
      overflow: await readTransferredRows([1])
    },
    segmentOverflow
  );
});

function createFixture(trackCount = 2) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props: GPUTrajectorySplitProps = {
    id: 'split',
    mode: 'gap',
    positions: createTransientView(graph, 'positions', 'float32x2', trackCount * 2),
    timestamps: createTransientView(graph, 'timestamps', 'float32', trackCount * 2),
    trackOffsets: createTransientView(graph, 'tracks', 'uint32', trackCount + 1),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {
      segmentOffsets: createTransientView(graph, 'offsets', 'uint32', trackCount + 1),
      segmentTracks: createTransientView(graph, 'segment-tracks', 'uint32', trackCount),
      rowIds: createTransientView(graph, 'rows', 'uint32', trackCount * 2),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }
    }
  };
  return {device, graph, props};
}

it('uses bounded parallel saturated scans and stays within its node estimate for many tracks', () => {
  for (const trackCount of [2, 257, 65537, 1048577]) {
    const {device, graph, props} = createFixture(trackCount);
    try {
      const contributor = new GPUTrajectorySplit(props);
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      expect(kernels.some(kernel => kernel.source.includes('for (var track ='))).toBe(false);
      expect(kernels.filter(kernel => kernel.source.includes('0xffffffffu, high['))).toHaveLength(
        2
      );
      for (const kernel of kernels.filter(kernel => kernel.operation === 'GPUTrajectorySplit')) {
        const source = kernel.source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
        const identifiers = source.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
      }
    } finally {
      device.destroy();
    }
  }
});

it('rejects aligned writable overlap and malformed scalar status at construction', () => {
  const {device, graph, props} = createFixture();
  try {
    const shared = createTransientView(graph, 'shared', 'uint32', 128);
    const count = graph.createDataView(shared.buffer, {format: 'uint32', byteOffset: 0, length: 1});
    const overflow = graph.createDataView(shared.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 1
    });
    expect(
      () => new GPUTrajectorySplit({...props, output: {...props.output, status: {count, overflow}}})
    ).toThrow(/split .*overlap/);
    expect(
      () =>
        new GPUTrajectorySplit({
          ...props,
          output: {...props.output, status: {...props.output.status, count: shared}}
        })
    ).toThrow('split output.status.count must have 1 rows');
  } finally {
    device.destroy();
  }
});

it('preflights the actual aligned range of an output before declaring scratch or nodes', () => {
  const {device, graph, props} = createFixture();
  try {
    const shared = createTransientView(graph, 'shared', 'uint32', 128);
    const rowIds = graph.createDataView(shared.buffer, {
      format: 'uint32',
      byteOffset: 252,
      length: 4
    });
    const contributor = new GPUTrajectorySplit({...props, output: {...props.output, rowIds}});
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 256}
    });
    const declarations = graph as unknown as {buffers: Map<string, unknown>; nodes: unknown[]};
    const before = [declarations.buffers.size, declarations.nodes.length];
    expect(() => contributor.getCommandNodes(graph)).toThrow(/split output.rowIds needs 268 bytes/);
    expect([declarations.buffers.size, declarations.nodes.length]).toEqual(before);
  } finally {
    device.destroy();
  }
});

it('rejects split settings that overflow float32 and short parameter targets', () => {
  expect(() => getGPUTrajectorySplitParameterValues({maximumGap: 1e40})).toThrow(
    /GPUTrajectorySplit maximumGap/
  );
  expect(() => getGPUTrajectorySplitParameterValues({maximumGap: 1}, new Float32Array(3))).toThrow(
    /parameter target/
  );
});

it('reserves a uint32 row for the terminal segment offset', () => {
  const {device, graph, props} = createFixture();
  try {
    expect(
      () =>
        new GPUTrajectorySplit({
          ...props,
          output: {
            ...props.output,
            segmentTracks: createTransientView(graph, 'maximum-tracks', 'uint32', 0xffffffff),
            segmentOffsets: createTransientView(graph, 'maximum-offsets', 'uint32', 0x100000000)
          }
        })
    ).toThrow('split output.segmentTracks must be an integer in [1, 4294967294]');
  } finally {
    device.destroy();
  }
});

it('accepts adjacent logical writes from separate dispatches', () => {
  const {device, graph, props} = createFixture(1);
  try {
    const shared = createTransientView(graph, 'packed', 'uint32', 2);
    const segmentTracks = graph.createDataView(shared.buffer, {format: 'uint32', length: 1});
    const count = graph.createDataView(shared.buffer, {format: 'uint32', length: 1, byteOffset: 4});
    expect(
      () =>
        new GPUTrajectorySplit({
          ...props,
          output: {...props.output, segmentTracks, status: {...props.output.status, count}}
        })
    ).not.toThrow();
    expect(
      () =>
        new GPUTrajectorySplit({
          ...props,
          output: {
            ...props.output,
            segmentTracks,
            status: {...props.output.status, count: segmentTracks}
          }
        })
    ).toThrow(/logical.*overlap/);
  } finally {
    device.destroy();
  }
});

it('rejects non-storage parameters before declaring resources', () => {
  const {device, graph, props} = createFixture();
  const buffer = device.createBuffer({byteLength: 16, usage: Buffer.COPY_DST});
  try {
    const parameters = importGraphBuffer(graph, 'non-storage', buffer, 'float32', 4);
    const contributor = new GPUTrajectorySplit({...props, parameters});
    const declarations = graph as unknown as {buffers: Map<string, unknown>; nodes: unknown[]};
    const before = [declarations.buffers.size, declarations.nodes.length];
    expect(() => contributor.getCommandNodes(graph)).toThrow(/parameters.*Buffer.STORAGE/);
    expect([declarations.buffers.size, declarations.nodes.length]).toEqual(before);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('preserves the oversized positions rejection fixture hint', () => {
  const {device, graph, props} = createFixture();
  try {
    const positions = createTransientView(graph, 'oversized-positions', 'float32x2', 16777217);
    const timestamps = createTransientView(graph, 'oversized-times', 'float32', positions.length);
    const contributor = new GPUTrajectorySplit({...props, positions, timestamps});
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /split the tracks over several graphs/
    );
  } finally {
    device.destroy();
  }
});

it('executes split consumers with seeded prefixes and prevents uint32 wrap into earlier slots', () => {
  const {device, graph, props} = createFixture(2);
  try {
    recordWGSLKernelNodes(graph);
    const nodes = new GPUTrajectorySplit({
      ...props,
      mode: 'value-change',
      values: createTransientView(graph, 'values', 'uint32', 4)
    }).getCommandNodes(graph);
    const writeNode = nodes.find(node => node.id === 'split-write')!;
    const statusNode = nodes.find(node => node.id === 'split-status')!;
    const write = getWGSLKernelNodeRecord(graph, writeNode.id)!;
    const status = getWGSLKernelNodeRecord(graph, statusNode.id)!;
    const memory = createCPUGraphMemory();
    const binding = (name: string) => write.bindings.find(binding => binding.name === name)!.view;
    memory.write(props.trackOffsets, new Uint32Array([0, 2, 4]));
    memory.write(binding('values'), new Uint32Array([1, 1, 2, 2]));
    memory.write(binding('trackSegmentOffsets'), new Uint32Array([0, 0xfffffffe]));
    memory.write(binding('trackRowOffsets'), new Uint32Array([0, 0xffffffff]));
    memory.write(props.output.rowIds, new Uint32Array([99, 99, 99, 99]));
    memory.write(
      status.bindings.find(binding => binding.name === 'totalSegments')!.view,
      new Uint32Array([0xffffffff])
    );
    memory.write(
      status.bindings.find(binding => binding.name === 'totalRows')!.view,
      new Uint32Array([0xffffffff])
    );
    executeCommandNodesOnCPU([writeNode, statusNode], memory);
    expect(Array.from(memory.readUint32(props.output.rowIds))).toEqual([0, 1, 99, 99]);
    expect(Array.from(memory.readUint32(props.output.segmentOffsets))).toEqual([0, 0, 4]);
    expect(memory.readUint32(props.output.status.count)[0]).toBe(2);
    expect(memory.readUint32(props.output.status.overflow)[0]).toBe(1);
    // Also cover segment-base addition within a track: the second overlapping segment must
    // saturate rather than wrap to output segment zero.
    memory.write(props.trackOffsets, new Uint32Array([0, 4, 4]));
    memory.write(binding('values'), new Uint32Array([1, 2, 3, 4]));
    memory.write(binding('trackSegmentOffsets'), new Uint32Array([0xffffffff, 0xffffffff]));
    memory.write(props.output.segmentTracks, new Uint32Array([99, 99]));
    executeCommandNodesOnCPU([writeNode], memory);
    expect(Array.from(memory.readUint32(props.output.segmentTracks))).toEqual([99, 99]);
  } finally {
    device.destroy();
  }
});
