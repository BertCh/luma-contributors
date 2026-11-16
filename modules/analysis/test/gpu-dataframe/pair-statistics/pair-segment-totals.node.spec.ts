// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {getPairStatisticsSegmentTotalNodes} from '../../../src/dataframe-analysis/pair-statistics/pair-statistics-cell-neighbors';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

it.each([
  'float32',
  'uint32'
] as const)('%s segment totals reuse one pipeline across reduction levels', format => {
  const build = (rows: number) => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const nodes = getPairStatisticsSegmentTotalNodes(graph, {
        id: 'totals',
        operation: 'PairSegmentTotals',
        rowCount: rows,
        segmentCount: 3,
        input: createTransientView(graph, 'input', format, 3 * rows),
        output: createTransientView(graph, 'output', format, format === 'uint32' ? 6 : 3),
        detectOverflow: format === 'uint32'
      });
      let entries = rows;
      for (const node of nodes) {
        entries = Math.ceil(entries / 256);
        expect(node.workload?.maximumInvocationCount).toBe(3 * entries);
      }
      for (const kernel of getGPUCommandNodeKernelSources(nodes))
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      const keys = getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true});
      expect(keys.size).toBe(1);
      return keys;
    } finally {
      device.destroy();
    }
  };
  expect(build(256)).toEqual(build(257));
  expect(build(257)).toEqual(build(65537));
});
