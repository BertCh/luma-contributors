// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {
  GPUCommandGraph,
  GPUScan,
  GPUSort,
  createGPUComputeCommandNode,
  createTransientView,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPUComputationRecorder,
  hashGPUComputationText,
  describeGPUCommandGraph,
  createGPUComputationRecord,
  validateGPUComputationRecord
} from '../../src/computation-record/index';

// Compile the selected nodes on their graph's device; no command buffer is submitted.
function createDevice() {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy: () => {}})
  });
  return device;
}

function addNodes(graph: GPUCommandGraph, nodes: readonly GPUCommandNode[]) {
  for (const node of nodes) graph.add(node);
}

it('S1: selected callbacks compile once on the graph device during graph compilation', () => {
  const device = createDevice();
  const graph = new GPUCommandGraph(device);
  const compilationDevices: (typeof device)[] = [];
  addNodes(graph, [
    createGPUComputeCommandNode({
      id: 'selected',
      compile: context => {
        compilationDevices.push(context.device);
        return {encode: () => {}};
      }
    })
  ]);
  try {
    expect(compilationDevices).toEqual([]);
    const compiled = graph.compile();
    try {
      expect(compilationDevices).toEqual([device]);
    } finally {
      compiled.destroy();
    }
  } finally {
    device.destroy();
  }
});

it.each([
  undefined,
  'layout',
  'source'
] as const)('LUMA-107: preserves scan metadata with collection %s and independent source opt-in', collection => {
  for (const sourceText of [false, true]) {
    const device = createDevice();
    const graph = new GPUCommandGraph(device, {id: 'scan'});
    const input = createTransientView(graph, 'input', 'uint32', 4);
    const output = createTransientView(graph, 'output', 'uint32', 4);
    addNodes(graph, new GPUScan({id: 'scan', input, output}).getCommandNodes(graph));
    const compiled = graph.compile(collection ? {kernelMetadata: collection} : undefined);
    try {
      const recorder = new GPUComputationRecorder({sourceText});
      const description = describeGPUCommandGraph(compiled, {recorder});
      const node = description.graph.nodes[0];
      expect(node.op).toBe('GPUScan');
      expect(node.bindings).toBeUndefined(); // layout slots do not identify graph resources
      expect(node.constants).toBeUndefined();
      if (collection) {
        expect(node.workgroupSize).toEqual([256, 1, 1]);
        expect(node.entryPoint ?? description.sources[0]?.entryPoint).toBe('main');
        expect(node.declaredBindings).toEqual([
          {name: 'inputValues', type: 'storage', group: 0, location: 0},
          {name: 'outputValues', type: 'storage', group: 0, location: 1}
        ]);
      } else {
        expect(node.workgroupSize).toBeUndefined();
        expect(node.declaredBindings).toBeUndefined();
      }
      expect(description.sources).toHaveLength(collection === 'source' ? 1 : 0);
      if (collection === 'source') {
        expect(description.sources[0].moduleId).toBe('GPUScan/portable');
        expect(description.sources[0].text !== undefined).toBe(sourceText);
      }
      expect(description.graph.resources).toEqual([
        {
          id: 'input',
          kind: 'buffer',
          bytes: 16,
          bytesKind: 'lower-bound',
          ownership: 'unknown',
          firstNode: node.id,
          lastNode: node.id
        },
        {
          id: 'output',
          kind: 'buffer',
          bytes: 16,
          bytesKind: 'lower-bound',
          ownership: 'unknown',
          firstNode: node.id,
          lastNode: node.id
        }
      ]);
      const record = createGPUComputationRecord({
        id: 'scan',
        adapter: {backend: 'webgpu'},
        domains: ['gpu-core'],
        graphs: [description]
      });
      expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
    } finally {
      compiled.destroy();
      device.destroy();
    }
  }
});

it('LUMA-107: explicit annotation wins one field while factory facts retain precedence over preflight', () => {
  const device = createDevice();
  const graph = new GPUCommandGraph(device, {id: 'factory'});
  const recorder = new GPUComputationRecorder();
  recorder.recordKernels(graph);
  const output = createTransientView(graph, 'output', 'uint32', 4);
  const node = createWGSLKernelNode(graph, {
    id: 'factory',
    operation: 'Fill',
    invocationCount: 4,
    workgroupSize: 64,
    bindings: [{name: 'output', view: output, type: 'u32', access: 'read_write'}],
    body: 'output[index] = 1u;'
  });
  const conflicting = {
    ...node,
    kernel: {
      entryPoint: 'publicEntry',
      workgroupSize: [128, 1, 1] as const,
      bindings: [{name: 'publicOutput', type: 'storage' as const, group: 0, location: 0}],
      source: '@compute @workgroup_size(128) fn publicEntry() {}'
    }
  };
  addNodes(graph, [recorder.recordNode(conflicting, {workgroupSize: [32, 1, 1]})]);
  const compiled = graph.compile({kernelMetadata: 'source'});
  try {
    const description = describeGPUCommandGraph(compiled, {recorder});
    expect(description.graph.nodes[0]).toMatchObject({
      workgroupSize: [32, 1, 1],
      op: 'Fill',
      constants: {},
      bindings: [
        {name: 'output', resource: 'output', elementType: 'u32', access: 'read_write', length: 4}
      ]
    });
    expect(description.graph.nodes[0].declaredBindings).toEqual([
      {name: 'publicOutput', type: 'storage', group: 0, location: 0}
    ]);
    expect(description.sources[0]).toMatchObject({entryPoint: 'main', moduleId: 'Fill'});
    expect(description.graph.nodes[0].entryPoint ?? description.sources[0].entryPoint).toBe('main');
    expect(description.sources[0].text).toBeUndefined();
  } finally {
    compiled.destroy();
    device.destroy();
  }
});

it('LUMA-107: a 513-row radix sort preserves nested scan facts without fabricating sort kernels', () => {
  const device = createDevice();
  const graph = new GPUCommandGraph(device, {id: 'radix'});
  const view = (id: string) => createTransientView(graph, id, 'uint32', 513);
  addNodes(
    graph,
    new GPUSort({
      id: 'radix',
      keys: view('keys'),
      values: view('values'),
      outputKeys: view('outputKeys'),
      outputValues: view('outputValues'),
      algorithm: 'radix'
    }).getCommandNodes(graph)
  );
  const compiled = graph.compile({kernelMetadata: 'layout'});
  try {
    const description = describeGPUCommandGraph(compiled);
    const scans = description.graph.nodes.filter(node => node.op === 'GPUScan');
    expect(scans.map(node => node.id)).toEqual(
      [0, 4, 8, 12, 16, 20, 24, 28].map(bit => `radix-radix-digit-${bit}-scan-level-0-scan`)
    );
    for (const node of scans) {
      expect(node.entryPoint).toBe('main');
      expect(node.workgroupSize).toEqual([256, 1, 1]);
      expect(node.declaredBindings).toEqual([
        {name: 'inputValues', type: 'storage', group: 0, location: 0},
        {name: 'outputValues', type: 'storage', group: 0, location: 1}
      ]);
      expect(node.bindings).toBeUndefined();
      expect(node.constants).toBeUndefined();
    }
    const others = description.graph.nodes.filter(node => node.op !== 'GPUScan');
    expect(others).toHaveLength(16);
    for (const node of others) {
      expect(node.declaredBindings).toBeUndefined();
      expect(node.workgroupSize).toBeUndefined();
    }
    expect(description.sources).toEqual([]);
    expect(
      description.graph.resources?.every(
        resource => resource.ownership === 'unknown' && resource.bytesKind === 'lower-bound'
      )
    ).toBe(true);
  } finally {
    compiled.destroy();
    device.destroy();
  }
});

it.each([
  false,
  true
])('B1: hash-only annotations retain public source in either node order (reverse=%s)', reverse => {
  const device = createDevice();
  const graph = new GPUCommandGraph(device);
  const recorder = new GPUComputationRecorder();
  const view = (id: string) => createTransientView(graph, id, 'uint32', 14);
  const nodes = ['first', 'second'].flatMap(id =>
    new GPUScan({id, input: view(`${id}-input`), output: view(`${id}-output`)}).getCommandNodes(
      graph
    )
  );
  const first = nodes[0];
  if (first.type !== 'compute' || first.kernel?.source === undefined)
    throw new Error('expected public scan kernel');
  const text = first.kernel.source;
  const hash = hashGPUComputationText(text);
  recorder.recordNode(first, {
    source: {
      id: `wgsl:${hash}`,
      language: 'wgsl',
      moduleId: 'annotated-scan',
      hash,
      entryPoint: 'main',
      text
    }
  });
  addNodes(graph, reverse ? [...nodes].reverse() : nodes);
  const compiled = graph.compile({kernelMetadata: 'source'});
  try {
    const description = describeGPUCommandGraph(compiled, {recorder, sourceText: true});
    expect(description.sources).toHaveLength(1);
    expect(description.sources[0].text).toBe(text);
    expect(
      description.graph.nodes.map(node => node.entryPoint ?? description.sources[0].entryPoint)
    ).toEqual(['main', 'main']);
  } finally {
    compiled.destroy();
    device.destroy();
  }
});

it('B2: manual source-ref constants override factory constants, including a known empty set', () => {
  const device = createDevice();
  const graph = new GPUCommandGraph(device);
  const recorder = new GPUComputationRecorder();
  recorder.recordKernels(graph);
  const constantSets: Readonly<Record<string, number>>[] = [{SCALE: 9}, {}];
  for (const [index, constants] of constantSets.entries()) {
    const output = createTransientView(graph, `manual-output-${index}`, 'uint32', 1);
    const node = createWGSLKernelNode(graph, {
      id: `manual-${index}`,
      operation: 'Scale',
      invocationCount: 1,
      declarations: 'override SCALE: u32 = 1u;',
      constants: {SCALE: 2},
      bindings: [{name: 'output', type: 'u32', access: 'read_write', view: output}],
      body: 'output[index] = SCALE;'
    });
    const local = getWGSLKernelNodeRecord(graph, node.id)!;
    const hash = hashGPUComputationText(local.source);
    addNodes(graph, [
      recorder.recordNode(node, {
        source: {
          id: `wgsl:${hash}`,
          language: 'wgsl',
          moduleId: 'ManualScale',
          hash,
          entryPoint: 'main',
          constants
        }
      })
    ]);
  }
  const compiled = graph.compile();
  try {
    const description = describeGPUCommandGraph(compiled, {recorder});
    expect(description.graph.nodes.map(node => node.constants)).toEqual([{SCALE: 9}, {}]);
  } finally {
    compiled.destroy();
    device.destroy();
  }
});

it('B1: two conflicting retained texts still fail after source-ref merging', () => {
  const device = createDevice();
  const graph = new GPUCommandGraph(device);
  const recorder = new GPUComputationRecorder({sourceText: true});
  const input = createTransientView(graph, 'collision-input', 'uint32', 14);
  const output = createTransientView(graph, 'collision-output', 'uint32', 14);
  const nodes = new GPUScan({id: 'collision', input, output}).getCommandNodes(graph);
  const node = nodes[0];
  if (node.type !== 'compute' || node.kernel?.source === undefined)
    throw new Error('expected scan source');
  const hash = hashGPUComputationText(node.kernel.source);
  recorder.recordNode(node, {
    source: {
      id: `wgsl:${hash}`,
      hash,
      language: 'wgsl',
      moduleId: 'collision',
      text: `${node.kernel.source}\n// different retained bytes`
    }
  });
  addNodes(graph, nodes);
  const compiled = graph.compile({kernelMetadata: 'source'});
  try {
    expect(() => describeGPUCommandGraph(compiled, {recorder})).toThrow('Conflicting source text');
  } finally {
    compiled.destroy();
    device.destroy();
  }
});

it('LENS-104: serialized bindings preserve aligned input, output and chunk parameter slots', () => {
  const device = createDevice();
  const graph = new GPUCommandGraph(device, {id: 'binding-ranges'});
  const recorder = new GPUComputationRecorder();
  recorder.recordKernels(graph);
  const input = createTransientView(graph, 'input', 'uint32', 128);
  const output = createTransientView(graph, 'output', 'uint32', 128);
  const parameters = createTransientView(graph, 'chunk-parameters', 'uint32', 128);
  for (const byteOffset of [8, 264]) {
    const view = (buffer: typeof input.buffer) =>
      graph.createDataView(buffer, {format: 'uint32', byteOffset, length: 3});
    addNodes(graph, [
      createWGSLKernelNode(graph, {
        id: `chunk-${byteOffset}`,
        operation: 'CopyChunk',
        invocationCount: 3,
        bindings: [
          {name: 'input', view: view(input.buffer), type: 'u32', access: 'read'},
          {name: 'output', view: view(output.buffer), type: 'u32', access: 'read_write'},
          {name: 'parameters', view: view(parameters.buffer), type: 'u32', access: 'read'}
        ],
        body: 'output[index] = input[index] + parameters[index];'
      })
    ]);
  }
  const compiled = graph.compile();
  try {
    const description = describeGPUCommandGraph(compiled, {recorder});
    expect(description.graph.nodes.map(node => node.id)).toEqual(['chunk-8', 'chunk-264']);
    for (const [index, node] of description.graph.nodes.entries()) {
      expect(node.bindings).toEqual(
        ['input', 'output', 'parameters'].map(name => ({
          name,
          resource: name === 'parameters' ? 'chunk-parameters' : name,
          access: name === 'output' ? 'read_write' : 'read',
          elementType: 'u32',
          elementOffset: 2,
          length: 3,
          bindingByteOffset: index * 256,
          bindingByteLength: 20
        }))
      );
    }
    const record = createGPUComputationRecord({
      id: 'ranges',
      adapter: {backend: 'webgpu'},
      domains: [],
      graphs: [description]
    });
    expect(validateGPUComputationRecord(record).valid).toBe(true);
    Object.assign(record.graphs[0].nodes[0].bindings![0], {bindingByteOffset: -1});
    expect(validateGPUComputationRecord(record).valid).toBe(false);
  } finally {
    compiled.destroy();
    device.destroy();
  }
});
