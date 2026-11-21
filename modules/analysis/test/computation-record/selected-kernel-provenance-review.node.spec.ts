// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createGPUComputeCommandNode,
  createTransientView
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createWGSLKernelNode,
  recordWGSLKernelNodes
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {describeGPUCommandGraph} from '../../src/computation-record/index';
import {GPUCellGeometry} from '../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-geometry';
import {GPUPointCloudFeatures} from '../../src/pointcloud-analysis/index';
import {GPUClassBreaks} from '../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {getGPUClassBreaksParameterLength} from '../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';

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

it.each([
  'before-graph',
  'after-graph',
  'after-description',
  'custom-before-graph'
] as const)('Review B2: standalone inspection cannot supply selected provenance (%s)', order => {
  const device = createDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const output = createTransientView(graph, 'output', 'uint32', 3);
  const selected = createWGSLKernelNode(graph, {
    id: 'step',
    operation: 'real',
    invocationCount: 3,
    workgroupSize: 2,
    bindings: [{name: 'output', view: output, type: 'u32', access: 'read_write'}],
    body: 'output[outputOffset + index] = 3u;'
  });
  const spare = createWGSLKernelNode(graph, {
    id: 'step',
    operation: 'other',
    invocationCount: 1,
    workgroupSize: 2,
    bindings: [{name: 'output', view: output, type: 'u32', access: 'read_write'}],
    body: 'output[outputOffset + index] = 7u;'
  });
  graph.add(
    order === 'custom-before-graph'
      ? createGPUComputeCommandNode({
          id: 'step',
          workload: {operation: 'realCustom', commandCount: 1},
          kernel: {
            source: '@compute @workgroup_size(1) fn main() {}',
            entryPoint: 'main',
            workgroupSize: [1, 1, 1],
            bindings: []
          },
          compile: () => ({encode: () => {}})
        })
      : selected
  );
  if (order === 'before-graph' || order === 'custom-before-graph')
    getGPUCommandNodeKernelSources([spare]);
  const compiled = graph.compile({kernelMetadata: 'source'});
  try {
    const options = {graph, sourceText: true};
    const previous =
      order === 'after-description' ? describeGPUCommandGraph(compiled, options) : undefined;
    if (order === 'after-graph' || order === 'after-description')
      getGPUCommandNodeKernelSources([spare]);
    const result = describeGPUCommandGraph(compiled, options);
    const node = result.graph.nodes[0];
    const source = result.sources.find(entry => entry.id === node.source);
    if (order === 'custom-before-graph') {
      expect(node.op).toBe('realCustom');
      expect(node.dispatch).toBeUndefined();
      expect(node.bindings).toBeUndefined();
      expect(source?.text).toBe('@compute @workgroup_size(1) fn main() {}');
      expect(node.declaredBindings).toEqual([]);
    } else {
      expect(source?.text).toContain('= 3u;');
      // Three invocations in two-lane workgroups require ceil(3/2)=2 groups.
      expect(node.dispatch?.workgroups).toEqual([2, 1, 1]);
      expect(
        node.bindings?.map(binding => [binding.name, binding.resource, binding.length])
      ).toEqual([['output', 'output', 3]]);
      if (previous) expect(result).toEqual(previous);
    }
  } finally {
    compiled.destroy();
    device.destroy();
  }
});

it.each([
  'geometry',
  'features',
  'class-breaks'
] as const)('Review B1: selected first-party factory retains complete recording facts (%s)', factory => {
  const device = createDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2' | 'float32x3'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const contributor =
    factory === 'geometry'
      ? new GPUCellGeometry({
          family: 's2',
          cells: view('cells', 'uint32x2', 11),
          output: {centers: view('centers', 'float32x2', 11)}
        })
      : factory === 'features'
        ? new GPUPointCloudFeatures({
            positions: view('positions', 'float32x3', 3),
            indices: view('indices', 'uint32', 4),
            counts: view('counts', 'uint32', 2),
            capacity: 2,
            output: {
              normals: view('normals', 'float32x3', 2),
              eigenvalues: view('eigenvalues', 'float32x3', 2),
              shapes: view('shapes', 'float32x3', 2),
              status: view('status', 'uint32', 2)
            }
          })
        : new GPUClassBreaks({
            values: view('values', 'float32', 4),
            parameters: view('parameters', 'float32', getGPUClassBreaksParameterLength(2)),
            methods: ['quantile'],
            maximumClassCount: 2,
            output: {
              breaks: view('breaks', 'float32', 3),
              classCount: view('class-count', 'uint32', 1),
              classCounts: view('class-counts', 'uint32', 2)
            }
          });
  const nodes = contributor.getCommandNodes(graph);
  graph.add(nodes);
  const compiled = graph.compile();
  try {
    const result = describeGPUCommandGraph(compiled, {graph, sourceText: true});
    const targets = nodes.filter(
      node =>
        node.type === 'compute' &&
        (factory !== 'class-breaks' || /quantile.*(histogram|select)/.test(node.id))
    );
    expect(targets).toHaveLength(factory === 'geometry' ? 1 : factory === 'features' ? 2 : 8);
    for (const target of targets) {
      const node = result.graph.nodes.find(entry => entry.id === target.id);
      expect(node?.source, target.id).toBeDefined();
      expect(node?.dispatch, target.id).toBeDefined();
      expect(node?.bindings?.length, target.id).toBeGreaterThan(0);
    }
  } finally {
    compiled.destroy();
    device.destroy();
  }
});
