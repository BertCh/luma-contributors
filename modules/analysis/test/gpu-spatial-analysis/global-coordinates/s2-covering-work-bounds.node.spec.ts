// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('documents per-level GPU candidate work separately from the published DFS budget', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const covering = new GPUS2PolygonCovering({
      positions: createTransientView(graph, 'positions', 'float32x3', 3),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
      maximumLevel: 1,
      maximumVisitedCells: 6,
      capacity: 6,
      refinement: 'serial',
      output: {
        cells: createTransientView(graph, 'cells', 'uint32x2', 6),
        counts: createTransientView(graph, 'counts', 'uint32x2', 1),
        status: createTransientView(graph, 'status', 'uint32', 1)
      }
    });
    const classifiedSlots = covering
      .getCommandNodes(graph)
      .filter(node => /-round-\d+-batch-\d+-classify$/.test(node.id))
      .map(node => node.workload?.maximumInvocationCount);
    // Six candidate slots at each of two configured levels: not six total tests.
    expect(classifiedSlots).toEqual([6, 6]);
    const guide = readFileSync(
      new URL(
        '../../../../../docs/api/gpu-spatial-analysis/global-coordinates.md',
        import.meta.url
      ),
      'utf8'
    );
    const workBounds = guide.split('### Work bounds and reference routes')[1];
    expect(workBounds).toContain('published depth-first prefix');
    expect(workBounds).toContain('per configured level');
    expect(workBounds).not.toContain(
      'GPU polygons run at most `maximumVisitedCells` classified cells'
    );
    expect(workBounds).not.toContain('visit budget bounds classified cells only');
  } finally {
    device.destroy();
  }
});
