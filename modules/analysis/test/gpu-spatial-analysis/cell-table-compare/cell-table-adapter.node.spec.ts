// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {captureGraphCommandNodes} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {addCellTableColumnsNode} from '../../../src/gpu-spatial-analysis/recipes/recipe-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('the recipe table adapter observes optional sum loss without changing occupancy (UNFINISHED)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const counts = createTransientView(graph, 'counts', 'uint32', 4);
  const sumValues = createTransientView(graph, 'sums', 'float32', 4);
  const sumOverflow = createTransientView(graph, 'sum-overflow', 'uint32', 1);
  try {
    const table = {counts, sumValues, status: {sumOverflow}};
    const nodes = captureGraphCommandNodes(graph, () =>
      addCellTableColumnsNode(graph, 'adapter', table, {
        values: createTransientView(graph, 'values', 'float32', 4),
        mask: createTransientView(graph, 'mask', 'uint32', 4)
      })
    );
    const source = getGPUCommandNodeKernelSources(nodes).find(
      kernel => kernel.id === 'adapter-cell-table-columns'
    )?.source;
    expect(source).toContain('sumOverflow[sumOverflowOffset]');
    expect(source).toContain('getNaN()');
    expect(source).toContain('select(0u, 1u, rowCount > 0u)');
  } finally {
    device.destroy();
  }
});
