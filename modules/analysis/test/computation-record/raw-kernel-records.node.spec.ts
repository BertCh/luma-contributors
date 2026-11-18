// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, test} from 'vitest';
import {createNullWebGPUDevice} from '../../src/testing/index';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createGraphChunkGroupKernelNode} from '../../src/analysis-core/kernels/graph-chunk-groups';
import {createClassificationKernelNode} from '../../src/dataframe-analysis/column-classification/classification-kernel-node';
import {GPUPointCloudFeatures} from '../../src/pointcloud-analysis/index';

test.each([
  'chunk-group',
  'classification-row',
  'classification-collective'
])('R1 %s retains its actual raw module and dispatch only on an opted-in graph', async route => {
  const device = createNullWebGPUDevice();
  const shaderSources: string[] = [];
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {shader: {source: string}}) => {
      shaderSources.push(props.shader.source);
      return {id: '', hash: '', destroy: () => {}};
    }
  });
  try {
    for (const enabled of [false, true]) {
      const graph = new GPUCommandGraph(device);
      if (enabled) recordWGSLKernelNodes(graph);
      const output = createTransientView(graph, 'output', 'float32', 256);
      const props = {
        id: route,
        operation: 'Review',
        bindings: [
          {name: 'output', view: output, type: 'f32' as const, access: 'read_write' as const}
        ],
        invocationCount: 256,
        workgroupSize: 256,
        body: 'output[outputOffset + index] = 1.0;'
      };
      const node =
        route === 'chunk-group'
          ? createGraphChunkGroupKernelNode(graph, props)
          : createClassificationKernelNode(graph, {
              ...props,
              workgroupCollective: route === 'classification-collective'
            });
      const record = getWGSLKernelNodeRecord(graph, route);
      if (!enabled) expect(record).toBeUndefined();
      else {
        expect(record).toMatchObject({
          id: route,
          entryPoint: 'main',
          workgroupSize: [256, 1, 1],
          workgroups: [1, 1, 1],
          invocationCount: 256,
          constants: {},
          bindings: [{name: 'output', view: output}]
        });
        expect(record!.source).toContain('var<uniform>');
        expect(record!.source).toContain(props.body);
        const compiled = await node.compile({device} as never);
        expect(shaderSources).toContain(record!.source);
        compiled.destroy?.();
      }
    }
  } finally {
    device.destroy();
  }
});

test('R1 actual point-cloud feature and shape chunks have source and storage facts', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const positions = createTransientView(graph, 'positions', 'float32x3', 3);
    const output = {
      normals: createTransientView(graph, 'normals', 'float32x3', 3),
      eigenvalues: createTransientView(graph, 'eigenvalues', 'float32x3', 3),
      shapes: createTransientView(graph, 'shapes', 'float32x3', 3),
      status: createTransientView(graph, 'status', 'uint32', 3)
    };
    const contributor = new GPUPointCloudFeatures({
      positions,
      indices: createTransientView(graph, 'indices', 'uint32', 9),
      counts: createTransientView(graph, 'counts', 'uint32', 3),
      capacity: 3,
      output
    });
    const nodes = contributor.getCommandNodes(graph);
    const chunks = nodes.filter(node => /features(?:-shapes)?-chunk-0$/.test(node.id));
    expect(chunks).toHaveLength(2);
    for (const node of chunks) {
      const record = getWGSLKernelNodeRecord(graph, node.id);
      expect(record).toMatchObject({
        id: node.id,
        invocationCount: 3,
        workgroupSize: [256, 1, 1],
        workgroups: [1, 1, 1]
      });
      expect(record!.source).toContain('chunkParameters');
      expect(record!.bindings.length).toBeGreaterThan(1);
    }
  } finally {
    device.destroy();
  }
});
