// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addCellFlowTopologyRecipe} from '../../../src/gpu-spatial-analysis/recipes/cell-flow-topology-recipe';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it.each([
  4096, 4097, 24577
])('RCP-105 bounds each cell and edge-slot scan for %s pairs', pairCount => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    addCellFlowTopologyRecipe(graph, {
      cells: createTransientView(graph, 'cells', 'uint32x2', 3),
      origins: createTransientView(graph, 'origins', 'uint32', pairCount),
      destinations: createTransientView(graph, 'destinations', 'uint32', pairCount),
      values: createTransientView(graph, 'values', 'float32', pairCount)
    });
    const kernels = getGPUCommandNodeKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    for (const suffix of ['cell-totals', 'edge-flows']) {
      const scans = kernels.filter(kernel => kernel.id.includes(suffix));
      expect(scans.length).toBe(Math.ceil(pairCount / 4096));
      expect(scans[0].source).toContain('const PAIR_CHUNK_COUNT: u32 = 4096u;');
      for (const scan of scans) {
        const loopConditions = [...scan.source.matchAll(/for\s*\([^;]+;\s*([^;]+);/g)].map(match =>
          match[1].replace(/\s/g, '')
        );
        expect(loopConditions).toEqual(['slot<PAIR_CHUNK_COUNT']);
      }
      const fullScans = scans.filter(scan =>
        scan.source.includes('const PAIR_CHUNK_COUNT: u32 = 4096u;')
      );
      expect(new Set(fullScans.map(scan => scan.source)).size).toBe(1);
      expect(new Set(fullScans.map(scan => scan.configurationKey)).size).toBe(1);
      if (pairCount > 4096) {
        expect(scans.at(-1)?.source).toContain('const PAIR_CHUNK_COUNT: u32 = 1u;');
      }
    }
  } finally {
    device.destroy();
  }
});
