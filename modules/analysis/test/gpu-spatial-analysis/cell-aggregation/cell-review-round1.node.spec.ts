// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCellAggregation} from '../../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation';
import {GPUCellPyramid} from '../../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-pyramid';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

function createScene() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(format: Format, length = 1) =>
    createTransientView(graph, `round1-${serial++}`, format, length);
  const table = (flag: boolean) => ({
    cells: view('uint32x2', 4),
    counts: view('uint32', 4),
    sums: view('uint32x2', 4),
    status: {
      count: view('uint32'),
      overflow: view('uint32'),
      candidateOverflow: view('uint32'),
      ...(flag ? {sumOverflow: view('uint32')} : {})
    }
  });
  return {device, graph, view, table};
}

it('tiles retain full signed accumulators until final publication (B5)', () => {
  const {device, graph, view, table} = createScene();
  try {
    // Exact integer fixture: four 3*2^60 rows then two -3*2^60 rows -> 6*2^60.
    // Saturating the first tile gives 2^63-1 - 6*2^60 = 2*2^60-1 instead.
    const nodes = new GPUCellAggregation({
      id: 'wide-tiles',
      family: 'quadbin',
      resolution: 0,
      sumScale: 1,
      scratchTileSize: 4,
      cells: view('uint32x2', 8),
      values: view('float32', 8),
      output: table(true)
    }).getCommandNodes(graph);
    expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    for (const kernel of kernels)
      expect(kernel.source.match(/@binding\(/g)?.length ?? 0).toBeLessThanOrEqual(8);
    expect(kernels.some(kernel => /^wide-tiles-tile-\d+-saturate-sums$/.test(kernel.id))).toBe(
      false
    );
    expect(kernels.find(kernel => kernel.id === 'wide-tiles-sums-merge-0')?.source).toContain(
      'sourceGuards'
    );
  } finally {
    device.destroy();
  }
});

it('tiled row-clamp status is filtered by the retained union rank (earlier-fix note)', () => {
  const {device, graph, view, table} = createScene();
  try {
    const nodes = new GPUCellAggregation({
      id: 'retained-tiles',
      family: 'quadbin',
      resolution: 0,
      scratchTileSize: 4,
      cells: view('uint32x2', 8),
      values: view('float32', 8),
      output: table(true)
    }).getCommandNodes(graph);
    const flag = getGPUCommandNodeKernelSources(nodes).find(
      kernel => kernel.id === 'retained-tiles-sums-flag-0'
    );
    expect(flag?.source).toContain('destination < cellLevelField');
    expect(flag?.source).toContain('rowOverflow');
  } finally {
    device.destroy();
  }
});

it('pyramid carries internal saturation flags through intervening levels (B6)', () => {
  const {device, graph, view, table} = createScene();
  try {
    const nodes = new GPUCellPyramid({
      id: 'flag-pyramid',
      family: 'quadbin',
      resolutions: [2, 1, 0],
      cells: view('uint32x2', 1),
      values: view('float32', 1),
      output: {levels: [table(false), table(false), table(true)]}
    }).getCommandNodes(graph);
    expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    for (const kernel of kernels)
      expect(kernel.source.match(/@binding\(/g)?.length ?? 0).toBeLessThanOrEqual(8);
    expect(kernels.some(kernel => kernel.id === 'flag-pyramid-level-0-sum-status')).toBe(true);
    expect(
      kernels.find(kernel => kernel.id === 'flag-pyramid-level-1-sum-status')?.source
    ).toContain('sumOverflowIn0');
    expect(
      kernels.find(kernel => kernel.id === 'flag-pyramid-level-2-sum-status')?.source
    ).toContain('sumOverflowIn0');
  } finally {
    device.destroy();
  }
});
