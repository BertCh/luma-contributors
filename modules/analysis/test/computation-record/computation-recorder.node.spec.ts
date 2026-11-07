// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {NullDevice} from '@luma.gl/core/test-utils';
import {
  GPUCommandGraph,
  createGPUComputeCommandNode,
  createTransientView
} from '@luma.gl/graph/gpu-core';
import {describe, expect, test} from 'vitest';
import {
  GPUComputationRecorder,
  createGPUComputationRecord,
  createGPUKernelAnnotation,
  describeGPUCommandGraph,
  describeGPUCommandGraphEncoding,
  getGPUCommandGraphPassIndices,
  validateGPUComputationRecord
} from '../../src/computation-record/index';

function createDevice(): NullDevice {
  const device = new NullDevice({id: 'recorder-device'});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  return device;
}

function createChain(recorder?: GPUComputationRecorder) {
  const device = createDevice();
  const graph = new GPUCommandGraph(device, {id: 'chain'});
  const input = createTransientView(graph, 'input', 'uint32', 4);
  const middle = createTransientView(graph, 'middle', 'uint32', 4);
  const output = createTransientView(graph, 'output', 'uint32', 4);
  const addPass = (
    id: string,
    read: typeof input,
    write: typeof input,
    annotation?: Parameters<GPUComputationRecorder['recordNode']>[1]
  ) => {
    const node = createGPUComputeCommandNode({
      id,
      resources: [
        {buffer: read, usage: 'storage-read'},
        {buffer: write, usage: 'storage-write'}
      ],
      workload: {operation: 'Copy', commandCount: 1},
      compile: () => ({encode: () => {}})
    });
    if (recorder) recorder.addNode(graph, node, annotation);
    else graph.add(node);
  };
  const run = () => {
    addPass('first', input, middle, {label: 'First'});
    addPass('second', middle, output, {dedupName: 'pass'});
  };
  if (recorder) recorder.withScope({namespace: 'chain', op: 'Copy'}, run);
  else run();
  return {device, graph};
}

describe('GPUComputationRecorder', () => {
  test('describes a compiled graph from published facts plus recorded annotations', () => {
    const recorder = new GPUComputationRecorder();
    const {device, graph} = createChain(recorder);
    const compiled = graph.compile();
    try {
      const {graph: described, sources} = describeGPUCommandGraph(compiled, {recorder});
      expect(sources).toEqual([]);
      expect(described.id).toBe('chain-compiled');
      expect(described.form).toBe('compiled');
      expect(described.nodes.map(node => node.id)).toEqual(['first', 'second']);
      expect(described.nodes[0]).toMatchObject({
        label: 'First',
        namespace: 'chain',
        kind: 'compute',
        op: 'Copy',
        reads: ['input'],
        writes: ['middle'],
        workload: {operation: 'Copy', commandCount: 1}
      });
      expect(described.nodes[1].label).toBe('pass');
      expect(described.edges).toEqual([
        {from: 'first', to: 'second', kind: 'raw', resource: 'middle'}
      ]);
      expect(described.resources?.map(resource => [resource.id, resource.ownership])).toEqual([
        ['input', 'transient'],
        ['middle', 'transient'],
        ['output', 'transient']
      ]);
      const record = createGPUComputationRecord({
        id: 'chain',
        adapter: {backend: 'webgpu'},
        domains: ['gpu-core'],
        graphs: [{graph: described, sources}]
      });
      expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
    } finally {
      compiled.destroy();
      device.destroy();
    }
  });

  test('describes without a recorder using only published facts', () => {
    const {device, graph} = createChain();
    const compiled = graph.compile();
    try {
      const {graph: described} = describeGPUCommandGraph(compiled);
      expect(described.nodes.map(node => [node.id, node.label, node.namespace])).toEqual([
        ['first', 'first', ''],
        ['second', 'second', '']
      ]);
      expect(described.resources?.every(resource => resource.ownership === 'imported')).toBe(true);
    } finally {
      compiled.destroy();
      device.destroy();
    }
  });

  test('merges nested scopes: namespaces join, innermost wins, node values win', () => {
    const recorder = new GPUComputationRecorder();
    const device = createDevice();
    const graph = new GPUCommandGraph(device);
    const buffer = createTransientView(graph, 'data', 'uint32', 1, Buffer.STORAGE);
    const node = createGPUComputeCommandNode({
      id: 'inner',
      resources: [{buffer, usage: 'storage-read-write'}],
      compile: () => ({encode: () => {}})
    });
    recorder.withScope({namespace: 'outer', op: 'Outer', derivedFrom: ['intent:a']}, () =>
      recorder.withScope(
        {namespace: 'level-0', derivedFrom: ['intent:b'], loop: {id: 'loop', iteration: 1}},
        () => recorder.recordNode(node, {namespace: 'leaf'})
      )
    );
    expect(recorder.getNodeRecord('inner')).toMatchObject({
      namespace: 'outer/level-0/leaf',
      op: 'Outer',
      derivedFrom: ['intent:b'],
      loop: {id: 'loop', iteration: 1}
    });
    recorder.withScope({namespace: 'popped'}, () => {});
    expect(recorder.getNodeRecord('inner')?.namespace).toBe('outer/level-0/leaf');
    device.destroy();
  });

  test('deduplicates sources and strips text unless sourceText', () => {
    for (const sourceText of [false, true]) {
      const recorder = new GPUComputationRecorder({sourceText});
      const device = createDevice();
      const graph = new GPUCommandGraph(device, {id: 'sources'});
      const annotation = createGPUKernelAnnotation({
        op: 'Copy',
        moduleId: 'Copy',
        source: 'fn main() {}',
        workgroupSize: [64, 1, 1],
        workgroups: [1, 1, 1]
      });
      const nodes = ['a', 'b'].map(id =>
        recorder.recordNode(
          createGPUComputeCommandNode({
            id,
            workload: {operation: 'Copy'},
            compile: () => ({encode: () => {}})
          }),
          annotation
        )
      );
      for (const node of nodes) graph.add(node);
      const compiled = graph.compile();
      const {sources, graph: described} = describeGPUCommandGraph(compiled, {recorder});
      expect(sources).toHaveLength(1);
      expect(sources[0]).toMatchObject({language: 'wgsl', entryPoint: 'main'});
      expect(sources[0].text !== undefined).toBe(sourceText);
      expect(described.nodes.every(node => node.source === sources[0].id)).toBe(true);
      compiled.destroy();
      device.destroy();
    }
  });

  test('infers pass indices only when they agree with the encoding pass count', () => {
    const node = (id: string, type: 'compute' | 'copy') => ({
      id,
      type,
      cpuEncodeTimeMilliseconds: 0,
      hasGPUTimestamps: false
    });
    const stats = {
      cpuEncodeTimeMilliseconds: 0,
      nodeCount: 4,
      skippedNodeCount: 0,
      computePassCount: 2,
      coalescedComputeNodeCount: 1,
      timestampedNodeCount: 0,
      nodes: [node('a', 'compute'), node('b', 'compute'), node('c', 'copy'), node('d', 'compute')]
    };
    expect(getGPUCommandGraphPassIndices(stats)).toEqual({a: 0, b: 0, d: 1});
    const encoded = describeGPUCommandGraphEncoding(stats, {compiledGraphId: 'chain-compiled'});
    expect(encoded.nodes.map(entry => entry.id)).toEqual(['pass-0', 'copy-c', 'pass-1']);
    expect(encoded.nodes[0].derivedFrom).toEqual(['chain-compiled:a', 'chain-compiled:b']);
    const disagreeing = {...stats, computePassCount: 5};
    expect(getGPUCommandGraphPassIndices(disagreeing)).toBeUndefined();
    expect(
      describeGPUCommandGraphEncoding(disagreeing, {compiledGraphId: 'chain-compiled'}).attrs
    ).toMatchObject({passGrouping: 'unknown'});
    expect(() =>
      describeGPUCommandGraph({id: 'x', stats: {nodeOrder: []}, preflight: {nodes: []}} as never, {
        graphId: 'a:b'
      })
    ).toThrow('graph id');
  });
});
