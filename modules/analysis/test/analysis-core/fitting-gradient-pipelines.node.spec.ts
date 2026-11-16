// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUFittingGradientSums} from '../../src/analysis-core/kernels/gpu-fitting-gradient-sums';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../src/testing/gpu-contributor-test-utils';
import {auditWGSLModule} from './kernel-audit/kernel-audit-wgsl';

it('uses one merge shader and pipeline across increasing reduction levels', () => {
  const device = createNullWebGPUDevice();
  const mergeKeys = new Set<string>();
  try {
    for (const rowCount of [513, 4097, 65537]) {
      const graph = new GPUCommandGraph(device);
      const gradients = createTransientView(graph, 'gradients', 'float32', rowCount * 2);
      const sums = createTransientView(graph, 'sums', 'float32', 2);
      const nodes = new GPUFittingGradientSums({
        rowCount,
        parameterCount: 2,
        gradients,
        output: {sums}
      }).getCommandNodes(graph);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      const distinct = new Set(
        kernels.map(kernel =>
          JSON.stringify([kernel.source, kernel.constants, kernel.shaderLayout])
        )
      );
      expect(distinct.size).toBe(2);
      for (const kernel of kernels.filter(kernel => kernel.id.includes('-merge-'))) {
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
        mergeKeys.add(JSON.stringify([kernel.source, kernel.constants, kernel.shaderLayout]));
      }
      for (const node of nodes.filter(node => node.id.includes('-merge-'))) {
        expect(node.workload?.maximumWorkgroupCount).toBe(
          Math.ceil((node.workload?.maximumInvocationCount ?? 0) / 256)
        );
      }
    }
    expect(mergeKeys.size).toBe(1);
  } finally {
    device.destroy();
  }
});
