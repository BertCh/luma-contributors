// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {DynamicBuffer} from '@luma.gl/core/dynamic';
import {GPUData} from '@luma.gl/graph/gpu-data';
import {
  createGPUCopyCommandNode,
  createTransientView,
  GPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createSpatialPreparationRetryViews} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-preparation-retry';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {GPUSpatialJoinCandidates} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-candidates';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const dynamic of [false, true])
  it(`B2 explicit retry view encodes a writing producer dynamic=${dynamic}`, async () => {
    const device = createNullWebGPUDevice();
    const original = new GPUCommandGraph(device);
    const fresh = new GPUCommandGraph(device);
    const buffer = device.createBuffer({
      byteLength: 512,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    const wrapper = new DynamicBuffer(device, {buffer, ownsBuffer: false});
    const source = original.importGPUData(
      'original',
      new GPUData({
        buffer: dynamic ? wrapper : buffer,
        format: 'float32x2',
        length: 1,
        byteOffset: 256,
        ownsBuffer: false
      })
    );
    // The producer handle promises only STORAGE. Its backing allocation has extra copy usages;
    // that physical allocation does not imply importGPUData must reuse this logical handle.
    const canonicalHandle = fresh.importBuffer(
      {id: 'canonical', byteLength: 512, usage: dynamic ? buffer.usage : Buffer.STORAGE},
      buffer
    );
    const canonical = fresh.createDataView(canonicalHandle, {
      format: 'float32x2',
      length: 1,
      byteOffset: 256
    });
    const retryViews = {positions: canonical};
    const retried = createSpatialPreparationRetryViews(
      fresh,
      'idx',
      {positions: source},
      retryViews
    ).positions;
    const outputView = createTransientView(fresh, 'result', 'float32x2', 1);
    const writer = createWGSLKernelNode(fresh, {
      id: 'writer',
      operation: 'PreparedRetryReview',
      invocationCount: 1,
      bindings: [{name: 'positions', view: canonical, type: 'f32', access: 'read_write'}],
      body: 'positions[positionsOffset] = 5.0; positions[positionsOffset + 1u] = 7.0;'
    });
    const reader = createWGSLKernelNode(fresh, {
      id: 'reader',
      operation: 'PreparedRetryReview',
      invocationCount: 1,
      bindings: [
        {name: 'positions', view: retried, type: 'f32', access: 'read'},
        {name: 'result', view: outputView, type: 'f32', access: 'read_write'}
      ],
      body: 'result[resultOffset] = positions[positionsOffset]; result[resultOffset + 1u] = positions[positionsOffset + 1u];'
    });
    // Host encoding shells preserve the original storage resource uses/hazards. NullDevice has
    // no compute pass; execute the actual emitted writer/reader separately with the interpreter.
    for (const node of [writer, reader])
      fresh.add(
        createGPUCopyCommandNode({
          id: node.id,
          resources: node.resources,
          compile: () => ({encode: () => {}})
        })
      );
    const compiled = fresh.compile();
    const encoder = device.createCommandEncoder();
    try {
      expect(() => compiled.encode(encoder, {parameters: undefined})).not.toThrow();
      const commands = encoder.finish();
      try {
        device.submit(commands);
      } finally {
        commands.destroy();
      }
      const memory = createCPUGraphMemory();
      executeCommandNodesOnCPU([writer, reader], memory);
      expect(Array.from(memory.readFloat32(outputView))).toEqual([5, 7]);
      expect(retried).toBe(canonical);
      retryViews.positions = fresh.createDataView(canonicalHandle, {...canonical, byteOffset: 264});
      expect(() =>
        createSpatialPreparationRetryViews(fresh, 'idx', {positions: source}, retryViews)
      ).toThrow(/positions.*same physical range/);
      expect(() =>
        createSpatialPreparationRetryViews(fresh, 'idx', {positions: source}, {positions: source})
      ).toThrow(/positions.*target graph/);
    } finally {
      encoder.destroy();
      compiled.destroy();
      wrapper.destroy();
      expect(buffer.destroyed).toBe(false);
      buffer.destroy();
      device.destroy();
    }
  });

it('B3 mixed-graph source IDs cannot be published by prepared.getGeometry', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const other = new GPUCommandGraph(device);
  const prepared = new GPUSpatialJoinPrepared({
    id: 'mixed-ids',
    geometry: {
      kind: 'points',
      positions: createTransientView(graph, 'positions', 'float32x2', 1),
      sourceIds: createTransientView(other, 'ids', 'uint32', 1)
    }
  });
  try {
    expect(() => graph.add(prepared)).toThrow(/mixed-ids.*(sourceIds|views).*target graph/);
    expect(prepared.isDeclaredIn(graph)).toBe(false);
    expect(() => prepared.getGeometry(graph)).toThrow(/graph.add/);
  } finally {
    prepared.destroy();
    device.destroy();
  }
});

it('S1 candidates validate supplied right views before substituting the prepared snapshot', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const other = new GPUCommandGraph(device);
  const buffer = device.createBuffer({
    byteLength: 8,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const data = () =>
    new GPUData({buffer, format: 'float32x2' as const, length: 1, ownsBuffer: false});
  const positions = graph.importGPUData('positions', data());
  const foreign = other.importGPUData('foreign', data());
  const prepared = new GPUSpatialJoinPrepared({geometry: {kind: 'points', positions}});
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  try {
    graph.add(prepared);
    const candidates = new GPUSpatialJoinCandidates({
      id: 'foreign-right',
      left: {kind: 'points', positions},
      right: {kind: 'points', positions: foreign},
      prepared,
      output: {
        leftRows: scalar('left'),
        rightRows: scalar('right'),
        status: {count: scalar('count'), overflow: scalar('overflow')}
      }
    });
    expect(() => graph.add(candidates)).toThrow(/foreign-right.*target graph/);
  } finally {
    prepared.destroy();
    buffer.destroy();
    device.destroy();
  }
});

it('B2 failed prepared declaration accepts captured explicit fresh geometry views', () => {
  const device = createNullWebGPUDevice();
  const original = new GPUCommandGraph(device);
  const buffer = device.createBuffer({
    byteLength: 512,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const positions = original.importGPUData(
    'old',
    new GPUData({buffer, format: 'float32x2', length: 1, byteOffset: 256, ownsBuffer: false})
  );
  const prepared = new GPUSpatialJoinPrepared({
    id: 'mapped-retry',
    geometry: {kind: 'points', positions}
  });
  const allocate = device.createBuffer.bind(device);
  const spy = vi.spyOn(device, 'createBuffer').mockImplementation(settings => {
    if (
      typeof settings === 'object' &&
      settings !== null &&
      'id' in settings &&
      settings.id === 'mapped-retry-node-minima'
    )
      throw new Error('injected failure');
    return allocate(settings);
  });
  try {
    try {
      expect(() => original.add(prepared)).toThrow('injected failure');
    } finally {
      spy.mockRestore();
    }
    expect(() => prepared.getGeometry(original)).toThrow(/graph.add/);
    const fresh = new GPUCommandGraph(device);
    const handle = fresh.importBuffer(
      {id: 'producer', byteLength: 512, usage: Buffer.STORAGE},
      buffer
    );
    const canonical = fresh.createDataView(handle, {
      format: 'float32x2',
      length: 1,
      byteOffset: 256
    });
    const geometry = {kind: 'points' as const, positions: canonical};
    prepared.setRetryViews(fresh, {geometry});
    geometry.positions = fresh.createDataView(handle, {...canonical, byteOffset: 264});
    const nodes = prepared.getCommandNodes(fresh);
    const resource = nodes.find(node => node.id === 'mapped-retry-snapshot-positions')!
      .resources![0];
    expect('buffer' in resource && 'buffer' in resource.buffer && resource.buffer.buffer).toBe(
      handle
    );
    expect(prepared.getGeometry(fresh).positions.buffer).not.toBe(handle);
    expect(() => prepared.setRetryViews(fresh, {geometry})).toThrow(/failed declaration/);
  } finally {
    spy.mockRestore();
    prepared.destroy();
    expect(buffer.destroyed).toBe(false);
    buffer.destroy();
    device.destroy();
  }
});

it('S2 predicate class documentation describes retained exact-refinement coordinates', async () => {
  const {readFile} = await import('node:fs/promises');
  const source = await readFile(
    new URL(
      '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join.ts',
      import.meta.url
    ),
    'utf8'
  );
  expect(source).not.toContain('refinement reads current right positions/offsets');
});
