// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createGraphChunkGroupKernelNode} from '../../src/analysis-core/kernels/graph-chunk-groups';
import {createClassificationKernelNode} from '../../src/dataframe-analysis/column-classification/classification-kernel-node';
import {getPointCloudRuntimeKernelNodes} from '../../src/pointcloud-analysis/operations/pointcloud-chunk-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';

for (const factory of ['point-cloud', 'shared-rows', 'classification-collective'] as const) {
  it(`${factory} records the actual module, dispatch and writable binding`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    recordWGSLKernelNodes(graph);
    const output = createTransientView(graph, 'recorded-output', 'uint32', 256);
    const props = {
      id: 'recorded-kernel',
      operation: 'RecordingProbe',
      invocationCount: 256,
      bindings: [
        {name: 'result', view: output, type: 'u32' as const, access: 'read_write' as const}
      ],
      body: 'result[resultOffset + index] = index;'
    };
    const nodes =
      factory === 'point-cloud'
        ? getPointCloudRuntimeKernelNodes(graph, {
            id: props.id,
            length: 256,
            inputs: {},
            outputs: {result: output},
            body: 'writeResult(index, index);'
          })
        : [
            factory === 'shared-rows'
              ? createGraphChunkGroupKernelNode(graph, props)
              : createClassificationKernelNode(graph, {...props, workgroupCollective: true})
          ];
    const node = nodes.find(
      node => node.id === (factory === 'point-cloud' ? `${props.id}-chunk-0` : props.id)
    )!;
    const record = getWGSLKernelNodeRecord(graph, node.id);
    expect(record, node.id).toBeDefined();
    const [compiled] = getGPUCommandNodeKernelSources([node]);
    expect(record!.source).toBe(compiled.source);
    expect(record!.workgroups).toEqual([1, 1, 1]);
    expect(record!.invocationCount).toBe(256);
    expect(record!.bindings.find(binding => binding.name === 'result')).toMatchObject({
      view: output,
      access: 'read_write'
    });
    expect(record!.constants).toEqual(compiled.constants);
  });
}
