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
  createGPUComputationCaptures,
  describeGPUCommandGraph,
  createGPUComputationRecord,
  validateGPUComputationRecord,
  validateGPUComputationSidecars
} from '@luma-contributors/analysis/computation-record';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';

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
  if (!device) return;
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
    const captures = createGPUComputationCaptures({runId: record.runId, snapshots});
    expect(validateGPUComputationSidecars(record, {captures})).toMatchObject({valid: true});
  } finally {
    compiled.destroy();
    arena.destroy();
  }
});
