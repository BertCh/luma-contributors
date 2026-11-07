// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it('S2 classification fits core shader budgets at maximum traversal options', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let sequence = 0;
    const view = <Format extends 'float32x3' | 'uint32' | 'uint32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `s2-classification-${sequence++}`, format, length);
    for (const mode of ['exterior', 'interior'] as const) {
      const contributor = new GPUS2PolygonCovering({
        positions: view('float32x3', 12),
        ringOffsets: view('uint32', 4),
        maximumLevel: 15,
        maximumVisitedCells: 1048576,
        mode,
        complement: true,
        capacity: 4096,
        output: {
          cells: view('uint32x2', 4096),
          counts: view('uint32x2', 1),
          status: view('uint32', 1),
          visitedCount: view('uint32', 1)
        }
      });
      for (const kernel of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)))
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});
