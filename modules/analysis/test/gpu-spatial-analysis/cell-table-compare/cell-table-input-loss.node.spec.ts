// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCellRollup} from '../../../src/gpu-spatial-analysis/cell-aggregation/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {compareCellTablesOnCPU} from './cell-table-compare-oracle';

it('comparison separates lost input cells from four-slot destination capacity', () => {
  const result = compareCellTablesOnCPU([{key: 1n, count: 1, sum: 0n}], [], {
    measure: 'count',
    sumScale: 65536,
    zScore: 'poisson',
    capacity: 4,
    inputOverflow: true
  });
  expect(result).toMatchObject({total: 1, overflow: false, candidateOverflow: true});
});

it('rollup publishes source loss separately from its destination capacity', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const table = (name: string, capacity: number) => ({
    cells: createTransientView(graph, `${name}-cells`, 'uint32x2', capacity),
    counts: createTransientView(graph, `${name}-counts`, 'uint32', capacity),
    status: {
      count: createTransientView(graph, `${name}-count`, 'uint32', 1),
      overflow: createTransientView(graph, `${name}-overflow`, 'uint32', 1),
      candidateOverflow: createTransientView(graph, `${name}-candidate-overflow`, 'uint32', 1),
      requiredCount: createTransientView(graph, `${name}-required`, 'uint32', 1)
    }
  });
  try {
    const nodes = new GPUCellRollup({
      id: 'retained-rollup',
      family: 'quadbin',
      sourceResolution: 1,
      resolution: 0,
      source: table('source', 1),
      output: table('output', 4)
    }).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    const publish = sources.find(kernel => kernel.id === 'retained-rollup-publish')!;
    expect(publish.source).not.toContain('overflowIn');
    const inherited = sources.find(kernel => kernel.id === 'retained-rollup-candidate-status');
    expect(inherited?.source).toContain('candidateOverflowOut');
    expect(inherited?.source).toContain('candidateIn1');
  } finally {
    device.destroy();
  }
});
