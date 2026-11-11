// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {describe, expect, test} from 'vitest';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  isRecordingWGSLKernelNodes,
  recordWGSLKernelNodes
} from '../../src/analysis-core/index';
import {createFillNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPUComputationRecorder,
  createGPUComputationRecord,
  describeGPUCommandGraph,
  validateGPUComputationRecord,
  type GPUComputationNode
} from '../../src/computation-record/index';
import {createNullWebGPUDevice} from '../../src/testing/index';

/** Null WebGPU device whose compute pipelines are stubs, so `graph.compile()` runs on the CPU. */
function createDevice(): Device {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy: () => {}})
  });
  return device;
}

/** Two helper kernels over one imported input and one transient: fill, then scale. */
function buildGraph(graph: GPUCommandGraph) {
  const device = graph.device;
  const inputBuffer = device.createBuffer({
    id: 'input',
    byteLength: 16,
    usage: Buffer.STORAGE | Buffer.COPY_DST
  });
  const input = graph.createDataView(
    graph.importBuffer({id: 'input', byteLength: 16, usage: inputBuffer.usage}, inputBuffer),
    {format: 'float32', length: 4}
  );
  const scratch = createTransientView(graph, 'scratch', 'float32', 4);
  graph.add(
    createFillNode(graph, {
      id: 'fill',
      operation: 'GPUExample',
      view: scratch,
      type: 'f32',
      value: '0.0'
    })
  );
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'scale',
      operation: 'GPUExample',
      variant: 'scale',
      bindings: [
        {name: 'values', view: input, type: 'f32', access: 'read'},
        {name: 'scaled', view: scratch, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 4,
      declarations: 'override FACTOR: f32;',
      constants: {FACTOR: 2},
      body: 'scaled[scaledOffset + index] = values[valuesOffset + index] * FACTOR;'
    })
  );
  return {destroy: () => inputBuffer.destroy()};
}

describe('createWGSLKernelNode kernel records', () => {
  test('a graph that records nothing keeps nothing and describes as before', () => {
    const device = createDevice();
    const graph = new GPUCommandGraph(device, {id: 'plain'});
    const inputs = buildGraph(graph);
    expect(isRecordingWGSLKernelNodes(graph)).toBe(false);
    expect(getWGSLKernelNodeRecord(graph, 'scale')).toBeUndefined();
    const compiled = graph.compile();
    try {
      const {graph: described, sources} = describeGPUCommandGraph(compiled);
      expect(sources).toEqual([]);
      for (const node of described.nodes) {
        expect(node.source).toBeUndefined();
        expect(node.bindings).toBeUndefined();
        expect(node.workgroupSize).toBeUndefined();
      }
      expect(described.attrs).toBeUndefined();
      // Passing the graph does not invent facts for a graph that recorded nothing.
      const withGraph = describeGPUCommandGraph(compiled, {graph});
      expect(withGraph.sources).toEqual([]);
      expect(withGraph.graph.nodes.every(node => node.source === undefined)).toBe(true);
    } finally {
      compiled.destroy();
      inputs.destroy();
      device.destroy();
    }
  });

  test('a recording graph describes source, bindings, dispatch and ownership of every kernel', () => {
    const device = createDevice();
    const graph = new GPUCommandGraph(device, {id: 'recorded'});
    recordWGSLKernelNodes(graph);
    const inputs = buildGraph(graph);
    const kernel = getWGSLKernelNodeRecord(graph, 'scale');
    expect(kernel).toMatchObject({
      operation: 'GPUExample',
      variant: 'scale',
      entryPoint: 'main',
      workgroupSize: [32, 1, 1],
      workgroups: [1, 1, 1],
      invocationCount: 4,
      constants: {FACTOR: 2, INVOCATION_COUNT: 4},
      indirect: false
    });
    expect(kernel?.source).toContain('fn main(');
    expect(kernel?.source).toContain(kernel!.body);
    const compiled = graph.compile();
    try {
      const {graph: described, sources} = describeGPUCommandGraph(compiled, {
        graph,
        sourceText: true
      });
      const scale = described.nodes.find(node => node.id === 'scale')!;
      const fill = described.nodes.find(node => node.id === 'fill')!;
      expect(scale).toMatchObject({
        op: 'GPUExample',
        workgroupSize: [32, 1, 1],
        dispatch: {workgroups: [1, 1, 1], invocationCount: 4, indirect: false},
        constants: {FACTOR: 2, INVOCATION_COUNT: 4},
        bindings: [
          {
            name: 'values',
            resource: 'input',
            access: 'read',
            elementType: 'f32',
            elementOffset: 0,
            length: 4
          },
          {name: 'scaled', resource: 'scratch', access: 'read_write', elementType: 'f32'}
        ]
      });
      expect(fill.bindings?.map(binding => binding.name)).toEqual(['fillTarget']);
      expect(sources.map(source => source.moduleId).sort()).toEqual([
        'GPUExample/fill',
        'GPUExample/scale'
      ]);
      const scaleSource = sources.find(source => source.id === scale.source)!;
      expect(scaleSource).toMatchObject({language: 'wgsl', entryPoint: 'main'});
      expect(scaleSource.text).toBe(kernel!.source);
      expect(scaleSource.bodyHash).toMatch(/\S/);
      expect(Object.fromEntries(described.resources!.map(r => [r.id, r.ownership]))).toEqual({
        input: 'imported',
        scratch: 'transient'
      });
      expect(described.attrs).toMatchObject({
        logicalTransientBufferCount: compiled.stats.logicalTransientBufferCount,
        physicalTransientBufferCount: compiled.stats.physicalTransientBufferCount
      });
      const record = createGPUComputationRecord({
        id: 'recorded',
        adapter: {backend: 'webgpu'},
        domains: ['gpu-core'],
        graphs: [{graph: described, sources}]
      });
      expect(validateGPUComputationRecord(record).issues).toEqual([]);
      // Without sourceText the source keeps its hash and module id but drops the text.
      const hashed = describeGPUCommandGraph(compiled, {graph});
      expect(hashed.sources.every(source => source.text === undefined)).toBe(true);
      expect(hashed.sources.map(source => source.id).sort()).toEqual(
        sources.map(source => source.id).sort()
      );
    } finally {
      compiled.destroy();
      inputs.destroy();
      device.destroy();
    }
  });

  test('recorder.recordKernels fills kernel facts and recorder annotations win', () => {
    const device = createDevice();
    const recorder = new GPUComputationRecorder({sourceText: true});
    const graph = recorder.recordKernels(new GPUCommandGraph(device, {id: 'scoped'}));
    expect(recorder.graph).toBe(graph);
    const inputs = buildGraph(graph);
    // A pack-level annotation for one node: its own fields win, kernel facts fill the rest.
    recorder.recordNode(
      {id: 'scale', type: 'compute', compile: () => ({encode: () => {}})} as never,
      {op: 'GPUScaled', formula: 'example.scale', loop: {id: 'scale', iteration: 0}}
    );
    const compiled = graph.compile();
    try {
      const {graph: described, sources} = describeGPUCommandGraph(compiled, {recorder});
      const scale = described.nodes.find(node => node.id === 'scale')!;
      expect(scale).toMatchObject({
        op: 'GPUScaled',
        formula: 'example.scale',
        loop: {id: 'scale', iteration: 0},
        workgroupSize: [32, 1, 1]
      });
      expect(scale.bindings).toHaveLength(2);
      expect(sources.every(source => typeof source.text === 'string')).toBe(true);
    } finally {
      compiled.destroy();
      inputs.destroy();
      device.destroy();
    }
  });

  test('validation accepts kernel facts, rejects malformed ones and keeps old records valid', () => {
    const node = {
      id: 'a',
      label: 'a',
      namespace: '',
      kind: 'compute',
      reads: [],
      writes: [],
      attrs: {}
    };
    const build = (extra: Record<string, unknown>) =>
      createGPUComputationRecord({
        id: 'validation',
        adapter: {backend: 'webgpu'},
        domains: ['gpu-core'],
        graphs: [
          {
            id: 'g',
            form: 'compiled',
            domain: 'gpu-core',
            nodes: [{...node, ...extra} as GPUComputationNode],
            edges: [],
            resources: []
          }
        ]
      });
    expect(validateGPUComputationRecord(build({})).valid).toBe(true);
    expect(
      validateGPUComputationRecord(
        build({workgroupSize: [64, 1, 1], constants: {N: 3}, dispatch: {workgroups: [2, 1, 1]}})
      ).valid
    ).toBe(true);
    const paths = (extra: Record<string, unknown>) =>
      validateGPUComputationRecord(build(extra)).issues.map(issue => issue.path);
    expect(paths({constants: {N: 'three'}})).toEqual(['graphs[0].nodes[0].constants.N']);
    expect(paths({workgroupSize: [64]})).toEqual(['graphs[0].nodes[0].workgroupSize']);
    expect(paths({dispatch: {workgroups: [1, -1, 1]}})).toEqual([
      'graphs[0].nodes[0].dispatch.workgroups'
    ]);
  });
});
