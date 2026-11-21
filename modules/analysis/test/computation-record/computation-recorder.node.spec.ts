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
  if (recorder) recorder.withSynchronousScope({namespace: 'chain', op: 'Copy'}, run);
  else run();
  return {device, graph};
}

describe('GPUComputationRecorder', () => {
  test('TYP-06: synchronous scopes nest and unwind through exceptions', () => {
    const recorder = new GPUComputationRecorder();
    const node = (id: string) =>
      createGPUComputeCommandNode({id, compile: () => ({encode: () => {}})});
    expect(() =>
      recorder.withSynchronousScope({namespace: 'outer'}, () => {
        recorder.withSynchronousScope({namespace: 'inner'}, () =>
          recorder.recordNode(node('nested'))
        );
        throw new Error('author failed');
      })
    ).toThrow('author failed');
    recorder.recordNode(node('after'));
    expect(recorder.getNodeRecord('nested')?.namespace).toBe('outer/inner');
    expect(recorder.getNodeRecord('after')?.namespace).toBe('');
    expect(recorder.withSynchronousScope({}, () => 7)).toBe(7);
  });

  test('TYP-06: rejects async callbacks at the synchronous boundary', () => {
    const recorder = new GPUComputationRecorder();
    let started = false;
    const author = async () => {
      started = true;
      await Promise.resolve();
    };
    // @ts-expect-error PromiseLike results require withAsyncScope.
    expect(() => recorder.withSynchronousScope({}, author)).toThrow('withAsyncScope');
    expect(started).toBe(false);
    // JavaScript callers can return a thenable from an ordinary function too.
    // @ts-expect-error A thenable is not a synchronous callback result.
    expect(() => recorder.withSynchronousScope({}, () => Promise.resolve(1))).toThrow(
      'withAsyncScope'
    );
  });

  test('TYP-06: two interleaved async scopes use independent namespace stacks and shared records', async () => {
    const recorder = new GPUComputationRecorder();
    const node = (id: string) =>
      createGPUComputeCommandNode({id, compile: () => ({encode: () => {}})});
    let resumeFirst!: () => void;
    const firstWait = new Promise<void>(resolve => {
      resumeFirst = resolve;
    });
    const firstScope = {namespace: 'first', derivedFrom: ['intent:first']};
    let first!: Promise<void>;
    recorder.withSynchronousScope({namespace: 'root'}, () => {
      first = recorder.withAsyncScope(firstScope, async scoped => {
        await firstWait;
        scoped.withSynchronousScope({namespace: 'nested'}, () => scoped.recordNode(node('first')));
      });
    });
    const second = recorder.withAsyncScope({namespace: 'second'}, async scoped => {
      scoped.recordNode(node('second-before'));
      firstScope.namespace = 'mutated';
      firstScope.derivedFrom[0] = 'intent:mutated';
      resumeFirst();
      await first;
      scoped.recordNode(node('second-after'));
      throw new Error('async failed');
    });
    await expect(second).rejects.toThrow('async failed');
    recorder.recordNode(node('outside'));
    expect(recorder.getNodeRecord('first')).toMatchObject({
      namespace: 'root/first/nested',
      derivedFrom: ['intent:first']
    });
    expect(recorder.getNodeRecord('second-before')?.namespace).toBe('second');
    expect(recorder.getNodeRecord('second-after')?.namespace).toBe('second');
    expect(recorder.getNodeRecord('outside')?.namespace).toBe('');
  });

  test('review B4: scope snapshots retain declared getter and non-enumerable metadata fields', async () => {
    const recorder = new GPUComputationRecorder();
    const node = (id: string) =>
      createGPUComputeCommandNode({id, compile: () => ({encode: () => {}})});
    let loopId = 'loop';
    let keySpace = 'rows';
    class Loop {
      get id() {
        return loopId;
      }
      get iteration() {
        return 2;
      }
      get phase() {
        return 'gated' as const;
      }
      get maximumIterations() {
        return 7;
      }
    }
    class Datum {
      get keySpace() {
        return keySpace;
      }
      get relation() {
        return 'invocation-to-row';
      }
    }
    const scope = {loop: new Loop(), datum: new Datum()};
    recorder.withSynchronousScope(scope, () => recorder.recordNode(node('sync-getters')));
    const pending = recorder.withAsyncScope(scope, async scoped => {
      await Promise.resolve();
      scoped.recordNode(node('async-getters'));
    });
    loopId = 'mutated';
    keySpace = 'mutated';
    await pending;
    const expected = {
      loop: {id: 'loop', iteration: 2, phase: 'gated', maximumIterations: 7},
      datum: {keySpace: 'rows', relation: 'invocation-to-row'}
    };
    expect(recorder.getNodeRecord('sync-getters')).toMatchObject(expected);
    expect(recorder.getNodeRecord('async-getters')).toMatchObject(expected);
  });

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
      expect(described.resources?.every(resource => resource.ownership === 'unknown')).toBe(true);
      expect(described.resources?.every(resource => resource.bytesKind === 'lower-bound')).toBe(
        true
      );
    } finally {
      compiled.destroy();
      device.destroy();
    }
  });

  test('unobserved resource size includes the offset of disjoint views', () => {
    const device = createDevice();
    const graph = new GPUCommandGraph(device, {id: 'offset-extent'});
    const storage = createTransientView(graph, 'storage', 'uint32', 4);
    for (const byteOffset of [0, 4]) {
      const view = graph.createDataView(storage.buffer, {format: 'uint32', length: 1, byteOffset});
      graph.add(
        createGPUComputeCommandNode({
          id: `read-${byteOffset}`,
          resources: [{buffer: view, usage: 'storage-read'}],
          compile: () => ({encode: () => {}})
        })
      );
    }
    const compiled = graph.compile();
    try {
      expect(describeGPUCommandGraph(compiled).graph.resources).toEqual([
        expect.objectContaining({
          id: 'storage',
          bytes: 8,
          bytesKind: 'lower-bound',
          ownership: 'unknown'
        })
      ]);
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
    recorder.withSynchronousScope(
      {namespace: 'outer', op: 'Outer', derivedFrom: ['intent:a']},
      () =>
        recorder.withSynchronousScope(
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
    recorder.withSynchronousScope({namespace: 'popped'}, () => {});
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
      if (sourceText) {
        expect(
          describeGPUCommandGraph(compiled, {recorder, sourceText: false}).sources[0].text
        ).toBeUndefined();
        expect(describeGPUCommandGraph(compiled, {recorder}).sources[0].text).toBe('fn main() {}');
      }
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

  test('DPC-102: nodes sharing WGSL text keep their own entry point and constants', () => {
    const recorder = new GPUComputationRecorder({sourceText: true});
    const device = createDevice();
    const graph = new GPUCommandGraph(device, {id: 'shared-text'});
    const text =
      'override FACTOR: u32 = 1u;\n@group(0) @binding(0) var<storage, read_write> result: array<u32>;\n@compute @workgroup_size(1) fn a() { result[0] = FACTOR; }\n@compute @workgroup_size(1) fn b() { result[1] = FACTOR; }';
    for (const [id, factor] of [
      ['a', 2],
      ['b', 3]
    ] as const) {
      graph.add(
        recorder.recordNode(
          createGPUComputeCommandNode({
            id,
            workload: {operation: 'Scale'},
            compile: () => ({encode: () => {}})
          }),
          createGPUKernelAnnotation({
            op: 'Scale',
            moduleId: 'Scale',
            source: text,
            entryPoint: id,
            constants: {FACTOR: factor},
            workgroupSize: [1, 1, 1],
            workgroups: [1, 1, 1]
          })
        )
      );
    }
    const compiled = graph.compile();
    try {
      const {sources, graph: described} = describeGPUCommandGraph(compiled, {recorder});
      // The text is stored once; every node states its own pipeline overrides.
      expect(sources).toHaveLength(1);
      const invocations = described.nodes.map(node => ({
        id: node.id,
        source: node.source,
        entryPoint: node.entryPoint ?? sources[0].entryPoint,
        constants: node.constants
      }));
      expect(invocations).toEqual([
        {id: 'a', source: sources[0].id, entryPoint: 'a', constants: {FACTOR: 2}},
        {id: 'b', source: sources[0].id, entryPoint: 'b', constants: {FACTOR: 3}}
      ]);
      // A node whose entry point matches the shared source ref does not repeat it.
      expect(described.nodes[0].entryPoint).toBeUndefined();
      const record = createGPUComputationRecord({
        id: 'shared-text',
        adapter: {backend: 'webgpu'},
        domains: ['gpu-core'],
        graphs: [{graph: described, sources}]
      });
      expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
      // A second description whose first node used entry point b has a source ref saying b and a
      // node c that states a. Merged into one record, its nodes are restated against the merged ref.
      const {entryPoint: _omitted, ...nodeB} = described.nodes[1];
      const merged = createGPUComputationRecord({
        id: 'shared-text-merged',
        adapter: {backend: 'webgpu'},
        domains: ['gpu-core'],
        graphs: [
          {graph: described, sources},
          {
            graph: {
              ...described,
              id: 'reversed',
              nodes: [nodeB, {...described.nodes[0], id: 'c', entryPoint: 'a'}]
            },
            sources: [{...sources[0], entryPoint: 'b'}]
          }
        ]
      });
      expect(merged.refs.sources.map(source => source.entryPoint)).toEqual(['a']);
      expect(
        merged.graphs.map(graph =>
          graph.nodes.map(node => [node.id, node.entryPoint ?? merged.refs.sources[0].entryPoint])
        )
      ).toEqual([
        [
          ['a', 'a'],
          ['b', 'b']
        ],
        [
          ['b', 'b'],
          ['c', 'a']
        ]
      ]);
    } finally {
      compiled.destroy();
      device.destroy();
    }
  });

  test('DPC-102: an overridden pipeline followed by shader defaults keeps an empty override set', () => {
    const recorder = new GPUComputationRecorder({sourceText: true});
    const device = createDevice();
    const graph = new GPUCommandGraph(device, {id: 'default-overrides'});
    const text =
      'override FACTOR: u32 = 7u;\n@group(0) @binding(0) var<storage, read_write> result: array<u32>;\n@compute @workgroup_size(1) fn a() { result[0] = FACTOR; }\n@compute @workgroup_size(1) fn b() { result[1] = FACTOR; }';
    for (const entryPoint of ['a', 'b'] as const)
      graph.add(
        recorder.recordNode(
          createGPUComputeCommandNode({id: entryPoint, compile: () => ({encode: () => {}})}),
          createGPUKernelAnnotation({
            op: 'Scale',
            moduleId: 'Scale',
            source: text,
            entryPoint,
            ...(entryPoint === 'a' && {constants: {FACTOR: 2}}),
            workgroupSize: [1, 1, 1],
            workgroups: [1, 1, 1]
          })
        )
      );
    const compiled = graph.compile();
    try {
      const description = describeGPUCommandGraph(compiled, {recorder});
      expect(description.sources).toHaveLength(1);
      expect(description.graph.nodes.map(node => node.constants)).toEqual([{FACTOR: 2}, {}]);
      expect(
        description.graph.nodes.map(node => node.entryPoint ?? description.sources[0].entryPoint)
      ).toEqual(['a', 'b']);
      const reversed = {
        graph: {
          ...description.graph,
          id: 'reversed-defaults',
          nodes: [
            {...description.graph.nodes[1], entryPoint: undefined},
            {...description.graph.nodes[0], entryPoint: 'a'}
          ]
        },
        sources: [{...description.sources[0], entryPoint: 'b'}]
      };
      const merged = createGPUComputationRecord({
        id: 'defaults',
        adapter: {backend: 'webgpu'},
        domains: ['gpu-core'],
        graphs: [description, reversed]
      });
      expect(merged.refs.sources).toHaveLength(1);
      expect(
        merged.graphs.map(graph =>
          graph.nodes.map(node => [
            node.entryPoint ?? merged.refs.sources[0].entryPoint,
            node.constants
          ])
        )
      ).toEqual([
        [
          ['a', {FACTOR: 2}],
          ['b', {}]
        ],
        [
          ['b', {}],
          ['a', {FACTOR: 2}]
        ]
      ]);
      expect(validateGPUComputationRecord(merged).valid).toBe(true);
    } finally {
      compiled.destroy();
      device.destroy();
    }
  });

  test('DPC-105: a render node in a mixed encoding gets its own pass', () => {
    const node = (id: string, type: 'compute' | 'render') => ({
      id,
      type,
      cpuEncodeTimeMilliseconds: 0,
      hasGPUTimestamps: false
    });
    const stats = {
      cpuEncodeTimeMilliseconds: 0,
      nodeCount: 3,
      skippedNodeCount: 0,
      computePassCount: 2,
      coalescedComputeNodeCount: 0,
      timestampedNodeCount: 0,
      nodes: [node('a', 'compute'), node('r', 'render'), node('b', 'compute')]
    };
    const summarize = (computePassCount: number) =>
      describeGPUCommandGraphEncoding(
        {...stats, computePassCount},
        {compiledGraphId: 'g'}
      ).nodes.map(entry => [entry.id, entry.attrs['passType'], entry.derivedFrom]);
    const expected = [
      ['pass-0', 'compute', ['g:a']],
      ['pass-1', 'render', ['g:r']],
      ['pass-2', 'compute', ['g:b']]
    ];
    // Inferred grouping (agrees with two compute passes) and the fallback (disagrees) both separate r.
    expect(summarize(2)).toEqual(expected);
    expect(summarize(7)).toEqual(expected);
    // Coalesced compute nodes on either side of a render pass still share only their own passes.
    const coalesced = describeGPUCommandGraphEncoding(
      {
        ...stats,
        nodeCount: 5,
        computePassCount: 2,
        coalescedComputeNodeCount: 2,
        nodes: [
          node('a', 'compute'),
          node('a2', 'compute'),
          node('r', 'render'),
          node('b', 'compute'),
          node('b2', 'compute')
        ]
      },
      {compiledGraphId: 'g'}
    );
    expect(coalesced.nodes.map(entry => [entry.id, entry.derivedFrom])).toEqual([
      ['pass-0', ['g:a', 'g:a2']],
      ['pass-1', ['g:r']],
      ['pass-2', ['g:b', 'g:b2']]
    ]);
  });
});
