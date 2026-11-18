// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {NullDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, test} from 'vitest';
import {GPUSnapshotArena, GPUSnapshotTap} from '../../src/computation-record/index';

test('final1 core:F12 preserves getter-backed tap fields and dependency', () => {
  const device = new NullDevice({id: 'tap-getters'});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  const graph = new GPUCommandGraph(device);
  const view = createTransientView(graph, 'value', 'uint32', 1, Buffer.STORAGE | Buffer.COPY_SRC);
  const arena = new GPUSnapshotArena(device, {
    slots: [{name: 'value', byteLength: 4, elementType: 'u32'}],
    capacity: 1
  });
  let afterNode = 'producer';
  class TapProps {
    get id() {
      return 'getter-tap';
    }
    get arena() {
      return arena;
    }
    get views() {
      return {value: view};
    }
    get afterNode() {
      return afterNode;
    }
  }
  try {
    const tap = new GPUSnapshotTap(new TapProps());
    afterNode = 'changed';
    const node = tap.getCommandNodes(graph)[0];
    expect(node.id).toBe('getter-tap');
    expect(tap.props.views['value']).toBe(view);
    expect(node.dependsOn).toEqual(['producer']);
  } finally {
    arena.destroy();
    device.destroy();
  }
});

test('snapshot taps reject oversized offset views, strided views and mismatched slot names', () => {
  const device = new NullDevice({id: 'tap-bounds'});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  const graph = new GPUCommandGraph(device);
  const storage = createTransientView(
    graph,
    'storage',
    'uint32',
    4,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const arena = new GPUSnapshotArena(device, {
    slots: [{name: 'value', byteLength: 8, elementType: 'u32'}],
    capacity: 1
  });
  const packed = graph.createDataView(storage.buffer, {format: 'uint32', length: 2, byteOffset: 4});
  const short = graph.createDataView(storage.buffer, {format: 'uint32', length: 1, byteOffset: 4});
  const strided = graph.createDataView(storage.buffer, {
    format: 'uint32',
    length: 2,
    byteStride: 8
  });
  try {
    expect(() => new GPUSnapshotTap({id: 'oversized', arena, views: {value: short}})).toThrow(
      'oversized views.value byteLength'
    );
    expect(() => new GPUSnapshotTap({id: 'strided', arena, views: {value: strided}})).toThrow(
      'strided views.value byteStride'
    );
    expect(() => new GPUSnapshotTap({id: 'missing', arena, views: {}})).toThrow('missing views');
    expect(
      () => new GPUSnapshotTap({id: 'extra', arena, views: {value: packed, extra: packed}})
    ).toThrow('extra views');
    expect(() => new GPUSnapshotTap({id: 'packed', arena, views: {value: packed}})).not.toThrow();
    expect(
      () => new GPUSnapshotTap({id: 'whole', arena, views: {value: storage.buffer}})
    ).not.toThrow();
    const props = {id: 'snapshot', arena, views: {value: packed}};
    const tap = new GPUSnapshotTap(props);
    props.id = 'changed';
    props.views.value = short;
    const node = tap.getCommandNodes(graph)[0];
    expect(node.id).toBe('snapshot');
    const resource = node.resources?.[0];
    expect(resource && 'buffer' in resource && resource.buffer).toBe(packed.buffer);
    expect(arena.count).toBe(0);
  } finally {
    arena.destroy();
    device.destroy();
  }
});
