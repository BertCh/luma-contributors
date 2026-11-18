// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, Texture, type CommandEncoder} from '@luma.gl/core';
import {
  GPUCommandGraph,
  createGPURenderCommandNode,
  createTransientView
} from '@luma.gl/graph/gpu-core';
import {expect, test, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../src/testing/index';
import {
  createGPUComputationRecord,
  getGPUComputationRunId,
  validateGPUComputationRecord,
  GPUComputationRecorder,
  describeGPUCommandGraph,
  GPUSnapshotArena,
  GPUSnapshotTap,
  printGPUComputationGraph
} from '../../src/computation-record/index';

const identity = {id: 'review', graphs: [], domains: [], adapter: {backend: 'webgpu'}};

test('R3 both tap encodings retain the validated handle and offset after in-place view mutation', async () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const storage = createTransientView(
    graph,
    'storage',
    'uint32',
    4,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const view = graph.createDataView(storage.buffer, {format: 'uint32', length: 2});
  const arena = new GPUSnapshotArena(device, {
    slots: [{name: 'value', byteLength: 8, elementType: 'u32'}],
    capacity: 2
  });
  try {
    const node = new GPUSnapshotTap({
      id: 'review-tap',
      arena,
      views: {value: view}
    }).getCommandNodes(graph)[0];
    const executable = await node.compile({device} as never);
    const copyBufferToBuffer = vi.fn();
    const getBuffer = vi.fn((_handle: unknown) => ({byteLength: 16}));
    const context = {commandEncoder: {copyBufferToBuffer} as unknown as CommandEncoder, getBuffer};
    executable.encode(context as never);
    Object.assign(view, {byteOffset: 8, length: 1});
    executable.encode(context as never);
    expect(copyBufferToBuffer.mock.calls.map(([copy]) => [copy.sourceOffset, copy.size])).toEqual([
      [0, 8],
      [0, 8]
    ]);
    expect(getBuffer.mock.calls.every(([handle]) => handle === storage.buffer)).toBe(true);
  } finally {
    arena.destroy();
    device.destroy();
  }
});

test('R4 texture estimates use descriptors and recorded external textures are imported with unavailable size', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const recorder = new GPUComputationRecorder();
  const texture = graph.createTransientTexture({
    id: 'image',
    format: 'rgba8unorm',
    width: 4,
    height: 4,
    usage: Texture.SAMPLE
  });
  const externalTexture = graph.importExternalTexture({id: 'external', width: 8, height: 8});
  recorder.addNode(
    graph,
    createGPURenderCommandNode({
      id: 'sample',
      resources: [
        {texture, usage: 'sampled'},
        {externalTexture, usage: 'sampled'}
      ],
      compile: () => ({encode: () => {}})
    })
  );
  const compiled = graph.compile();
  try {
    const recorded = describeGPUCommandGraph(compiled, {recorder}).graph.resources!;
    const plain = describeGPUCommandGraph(compiled).graph.resources!;
    expect(recorded.find(resource => resource.id === 'image')).toMatchObject({
      bytes: 64,
      bytesKind: 'estimated',
      ownership: 'transient'
    });
    expect(plain.find(resource => resource.id === 'image')).toMatchObject({
      bytes: 64,
      bytesKind: 'lower-bound',
      ownership: 'unknown'
    });
    expect(recorded.find(resource => resource.id === 'external')).toMatchObject({
      bytesKind: 'unavailable',
      ownership: 'imported'
    });
  } finally {
    compiled.destroy();
    device.destroy();
  }
});

test.each(
  [NaN, Infinity, -Infinity, {}, [], null].map(value => [value])
)('R7 rejects non-scalar or non-finite identity value %j before hashing and on load', value => {
  const parameters = {x: value} as never;
  expect(() => getGPUComputationRunId({...identity, parameters})).toThrow('parameters.x');
  expect(() => createGPUComputationRecord({...identity, runId: 'explicit', parameters})).toThrow(
    'parameters.x'
  );
  expect(
    validateGPUComputationRecord({...createGPUComputationRecord(identity), parameters}).valid
  ).toBe(false);
});

test.each([
  NaN,
  Infinity,
  -Infinity
])('R7 numeric seed and choice values must be finite: %j', value => {
  expect(() => getGPUComputationRunId({...identity, seed: value})).toThrow('seed');
  expect(() =>
    getGPUComputationRunId({
      ...identity,
      choices: [{stage: 'fit', name: 'threshold', value, sweepable: true}]
    })
  ).toThrow('choices');
  expect(
    validateGPUComputationRecord({
      ...createGPUComputationRecord(identity),
      choices: [{stage: 'fit', name: 'threshold', value, sweepable: true}]
    }).valid
  ).toBe(false);
  expect(() =>
    getGPUComputationRunId({...identity, adapter: {backend: 'webgpu', description: value as never}})
  ).toThrow('adapter.description');
});

test('R8 records snapshot caller choices and adapter before the next execution edits them', () => {
  const adapter = {backend: 'webgpu', description: 'first'};
  const choices = [
    {stage: 'fit', name: 'threshold', value: 1, sweepable: true, alternatives: [1, 2]}
  ];
  const first = createGPUComputationRecord({...identity, adapter, choices});
  const originalId = first.runId;
  choices[0].value = 2;
  choices[0].alternatives[0] = 9;
  adapter.description = 'second';
  const second = createGPUComputationRecord({...identity, adapter, choices});
  expect(first.choices[0]).toMatchObject({value: 1, alternatives: [1, 2]});
  expect(first.adapter.description).toBe('first');
  expect(first.runId).toBe(originalId);
  expect(second.runId).not.toBe(originalId);
});

test('R10 legacy and explicitly logical imported resources print different certainty', () => {
  const text = printGPUComputationGraph({
    id: 'compiled',
    form: 'compiled',
    domain: 'test',
    nodes: [],
    edges: [],
    resources: [
      {id: 'legacy', kind: 'buffer', bytes: 8, ownership: 'imported'},
      {id: 'known', kind: 'buffer', bytes: 8, bytesKind: 'logical', ownership: 'imported'}
    ]
  });
  expect(text).toContain('legacy unknown 8B (precision unknown)');
  expect(text).toContain('known imported 8B');
});
