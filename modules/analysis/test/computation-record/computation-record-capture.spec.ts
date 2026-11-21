// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {GPUCommandGraph, createTransientView, getViewBinding} from '@luma.gl/graph/gpu-core';
import {
  GPUComputationRecorder,
  GPUSnapshotArena,
  GPUSnapshotTap,
  GPUTransferMask,
  createGPUComputationCaptures,
  describeGPUCommandGraph,
  createGPUComputationRecord,
  validateGPUComputationRecord,
  validateGPUComputationSidecars
} from '@luma-contributors/analysis/computation-record';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createSegmentSumNode} from '../../src/analysis-core/kernels/sorted-segment-sums';

/** Three iterations of a counter, each followed by a tap that snapshots the state. */
function createChain(device: Device, recorder: GPUComputationRecorder, arena: GPUSnapshotArena) {
  const graph = new GPUCommandGraph(device, {id: 'capture-chain'});
  // The nonzero view offset also exercises snapshot source range resolution.
  const storage = createTransientView(
    graph,
    'state',
    'uint32',
    2,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const state = graph.createDataView(storage.buffer, {format: 'uint32', length: 1, byteOffset: 4});
  for (let iteration = 1; iteration <= 3; iteration++) {
    const node = {
      id: `iteration-${iteration}`,
      publication: {
        id: `result-${iteration}`,
        completeness: iteration === 3 ? ('complete' as const) : ('partial' as const)
      },
      resources: [{buffer: state, usage: 'storage-read-write' as const}],
      compile: () => {
        const kernel = new Kernel(device, {
          source: `@group(0) @binding(0) var<storage, read_write> state: array<u32>;
@compute @workgroup_size(1) fn main() { state[1] = state[1] + 1u; }`,
          shaderLayout: {bindings: [{name: 'state', type: 'storage', group: 0, location: 0}]}
        });
        return {
          encode: ({computePass, getBuffer}: any) =>
            kernel.dispatch(computePass, {
              bindings: {state: getViewBinding(state, getBuffer)},
              x: 1
            }),
          destroy: () => kernel.destroy()
        };
      }
    };
    graph.addComputePass(node);
    // The tap follows its node in authoring order, so later iterations cannot overwrite the value.
    graph.add(
      new GPUSnapshotTap({
        id: `observe-${iteration}`,
        arena,
        views: {state},
        afterNode: node.id,
        recorder
      })
    );
  }
  return {graph, state};
}

test('taps snapshot an iterative transient while later iterations overwrite it', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const recorder = new GPUComputationRecorder();
  const arena = new GPUSnapshotArena(device, {
    id: 'chain-snapshots',
    slots: [{name: 'state', byteLength: 4, elementType: 'u32'}],
    capacity: 3
  });
  const {graph} = createChain(device, recorder, arena);
  const compiled = graph.compile();
  try {
    const {graph: described, sources} = describeGPUCommandGraph(compiled, {recorder});
    const record = createGPUComputationRecord({
      id: 'chain',
      adapter: {backend: 'webgpu'},
      domains: ['gpu-core'],
      graphs: [{graph: described, sources}]
    });
    expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
    expect(described.nodes.filter(node => node.attrs['observationTap'])).toHaveLength(3);
    const commandEncoder = device.createCommandEncoder();
    compiled.encode(commandEncoder, {parameters: undefined});
    device.submit(commandEncoder.finish());
    const snapshots = await arena.read({
      graph: described.id,
      resources: {state: 'state'},
      runId: record.runId
    });
    expect(snapshots.map(capture => Array.from(capture.values))).toEqual([[1], [2], [3]]);
    expect(snapshots.map(capture => capture.afterNode)).toEqual([
      'iteration-1',
      'iteration-2',
      'iteration-3'
    ]);
    expect(
      snapshots.every(capture => capture.elementType === 'u32' && capture.resource === 'state')
    ).toBe(true);
    const captures = createGPUComputationCaptures({
      runId: record.runId,
      executionId: record.executionId,
      snapshots
    });
    expect(validateGPUComputationSidecars(record, {captures})).toMatchObject({valid: true});
  } finally {
    compiled.destroy();
    arena.destroy();
  }
});

test('taps refuse oversized and strided logical views before graph copies are added', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const storage = createTransientView(
    graph,
    'storage',
    'uint32',
    4,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const short = graph.createDataView(storage.buffer, {format: 'uint32', length: 1, byteOffset: 4});
  const strided = graph.createDataView(storage.buffer, {
    format: 'uint32',
    length: 2,
    byteStride: 8
  });
  const arena = new GPUSnapshotArena(device, {
    slots: [{name: 'value', byteLength: 8, elementType: 'u32'}],
    capacity: 1
  });
  try {
    expect(() =>
      graph.add(new GPUSnapshotTap({id: 'oversized', arena, views: {value: short}}))
    ).toThrow('views.value byteLength');
    expect(() =>
      graph.add(new GPUSnapshotTap({id: 'strided', arena, views: {value: strided}}))
    ).toThrow('views.value byteStride');
    expect(arena.count).toBe(0);
  } finally {
    arena.destroy();
  }
});

test('R3 repeated encodings copy the original range after the caller mutates its view object', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({
    data: new Uint32Array([11, 22, 33, 44]),
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const handle = graph.importBuffer(
    {id: 'source', byteLength: buffer.byteLength, usage: buffer.usage},
    buffer
  );
  const view = graph.createDataView(handle, {format: 'uint32', length: 2});
  const arena = new GPUSnapshotArena(device, {
    slots: [{name: 'value', byteLength: 8, elementType: 'u32'}],
    capacity: 2
  });
  graph.add(new GPUSnapshotTap({id: 'review-copy', arena, views: {value: view}}));
  const compiled = graph.compile();
  try {
    for (let frame = 0; frame < 2; frame++) {
      if (frame === 1) Object.assign(view, {byteOffset: 8, length: 1});
      const encoder = device.createCommandEncoder();
      compiled.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish());
    }
    expect((await arena.read()).map(capture => Array.from(capture.values))).toEqual([
      [11, 22],
      [11, 22]
    ]);
  } finally {
    compiled.destroy();
    arena.destroy();
    buffer.destroy();
  }
});

test('LIF-13: discarded tap encodings disappear before submitted capture', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const recorder = new GPUComputationRecorder();
  const arena = new GPUSnapshotArena(device, {
    id: 'discarded-chain',
    slots: [{name: 'state', byteLength: 4, elementType: 'u32'}],
    capacity: 6
  });
  const {graph} = createChain(device, recorder, arena);
  const compiled = graph.compile();
  try {
    const first = device.createCommandEncoder();
    compiled.encode(first, {parameters: undefined});
    const boundary = arena.checkpoint();
    const abandoned = device.createCommandEncoder();
    compiled.encode(abandoned, {parameters: undefined});
    expect(arena.count).toBe(6);
    arena.markSubmitted(boundary);
    device.submit(first.finish());
    arena.discard(boundary); // This encoder is never submitted.
    expect(arena.count).toBe(3);
    const submitted = device.createCommandEncoder();
    compiled.encode(submitted, {parameters: undefined});
    arena.markSubmitted();
    device.submit(submitted.finish());
    const captures = await arena.read();
    expect(
      captures.map(capture => [capture.iteration, capture.afterNode, Array.from(capture.values)])
    ).toEqual([
      [0, 'iteration-1', [1]],
      [1, 'iteration-2', [2]],
      [2, 'iteration-3', [3]],
      [3, 'iteration-1', [4]],
      [4, 'iteration-2', [5]],
      [5, 'iteration-3', [6]]
    ]);
  } finally {
    compiled.destroy();
    arena.destroy();
  }
});

test('ENT-03/B3: capture qualifies the recorded reduction with literal segment sums', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const recorder = new GPUComputationRecorder({sourceText: true});
  const graph = recorder.recordKernels(new GPUCommandGraph(device, {id: 'recorded-reduction'}));
  const inputBuffer = device.createBuffer({
    data: new Float32Array([2, 3, 7]),
    usage: Buffer.STORAGE
  });
  const offsetsBuffer = device.createBuffer({
    data: new Uint32Array([0, 2, 3, 3, 3, 3]),
    usage: Buffer.STORAGE
  });
  const input = graph.createDataView(
    graph.importBuffer(
      {id: 'input', byteLength: inputBuffer.byteLength, usage: inputBuffer.usage},
      inputBuffer
    ),
    {format: 'float32', length: 3}
  );
  const offsets = graph.createDataView(
    graph.importBuffer(
      {id: 'offsets', byteLength: offsetsBuffer.byteLength, usage: offsetsBuffer.usage},
      offsetsBuffer
    ),
    {format: 'uint32', length: 6}
  );
  const output = createTransientView(
    graph,
    'output',
    'float32',
    5,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const reduction = createSegmentSumNode(graph, {
    id: 'sum',
    operation: 'RecordedSum',
    segmentCount: 2,
    input,
    segmentOffsets: offsets,
    output
  });
  graph.add(reduction);
  const arena = new GPUSnapshotArena(device, {
    slots: [{name: 'sums', byteLength: 8, elementType: 'f32'}],
    capacity: 1
  });
  graph.add(
    new GPUSnapshotTap({id: 'observe-sum', arena, views: {sums: output}, afterNode: reduction.id})
  );
  const compiled = graph.compile();
  try {
    const description = describeGPUCommandGraph(compiled, {recorder});
    const serialized = description.graph.nodes.find(node => node.id === reduction.id)!;
    expect(serialized.op).toBe('RecordedSum');
    expect(serialized.bindings?.map(binding => binding.length)).toEqual([3, 6, 5]);
    expect(serialized.dispatch?.workgroups).toEqual([2, 1, 1]);
    expect(description.sources.find(source => source.id === serialized.source)?.text).toContain(
      'partialSums'
    );
    const encoder = device.createCommandEncoder();
    compiled.encode(encoder, {parameters: undefined});
    arena.markSubmitted();
    device.submit(encoder.finish());
    const captures = await arena.read();
    expect(Array.from(captures[0].values).every(Number.isFinite)).toBe(true);
    // Closed-form sums: rows [0,2) contain 2+3=5; rows [2,3) contain 7.
    expect(Array.from(captures[0].values)).toEqual([5, 7]);
  } finally {
    compiled.destroy();
    arena.destroy();
    inputBuffer.destroy();
    offsetsBuffer.destroy();
  }
});

for (const mode of ['any', 'count'] as const) {
  test(`VAL-04: ${mode} aggregates two selected sources into one target`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const sourceBuffer = device.createBuffer({
      data: new Uint32Array([1, 1]),
      usage: Buffer.STORAGE
    });
    const targetsBuffer = device.createBuffer({
      data: new Uint32Array([0, 0]),
      usage: Buffer.STORAGE
    });
    const source = graph.createDataView(
      graph.importBuffer({id: 'source', byteLength: 8, usage: sourceBuffer.usage}, sourceBuffer),
      {format: 'uint32', length: 2}
    );
    const targets = graph.createDataView(
      graph.importBuffer({id: 'targets', byteLength: 8, usage: targetsBuffer.usage}, targetsBuffer),
      {format: 'uint32', length: 2}
    );
    const target = createTransientView(
      graph,
      'target',
      'uint32',
      1,
      Buffer.STORAGE | Buffer.COPY_SRC
    );
    graph.add(
      new GPUTransferMask({
        id: 'transfer',
        source,
        relation: {layout: 'index', targets},
        target,
        mode
      })
    );
    const arena = new GPUSnapshotArena(device, {
      slots: [{name: 'mask', byteLength: 4, elementType: 'u32'}],
      capacity: 1
    });
    graph.add(
      new GPUSnapshotTap({
        id: 'observe-mask',
        arena,
        views: {mask: target},
        afterNode: 'transfer-scatter'
      })
    );
    const compiled = graph.compile();
    try {
      const encoder = device.createCommandEncoder();
      compiled.encode(encoder, {parameters: undefined});
      arena.markSubmitted();
      device.submit(encoder.finish());
      // Two selected incoming entries: logical OR is 1, incoming entry count is 2.
      expect(Array.from((await arena.read())[0].values)).toEqual([mode === 'count' ? 2 : 1]);
    } finally {
      compiled.destroy();
      arena.destroy();
      sourceBuffer.destroy();
      targetsBuffer.destroy();
    }
  });
}
