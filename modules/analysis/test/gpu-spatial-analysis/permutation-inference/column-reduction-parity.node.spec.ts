// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {expect, it} from 'vitest';
import {
  GPUGlobalSpatialStatistics,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {
  BLOCK_ROWS,
  getColumnSumNodes
} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-inference-kernels';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function expectBatchBindings(
  graph: GPUCommandGraph,
  id: string,
  matrix: GraphDataView<'float32'>,
  totals: GraphDataView<'float32'>
): void {
  const blocks = getWGSLKernelNodeRecord(graph, `${id}-blocks`)!;
  const columns = getWGSLKernelNodeRecord(graph, `${id}-columns`)!;
  expect(blocks.bindings.find(binding => binding.name === 'segmentInput')!.view).toBe(matrix);
  expect(columns.bindings.find(binding => binding.name === 'segmentOutput')!.view).toBe(totals);
  expect(columns.bindings.find(binding => binding.name === 'segmentInput')!.view).toBe(
    blocks.bindings.find(binding => binding.name === 'segmentOutput')!.view
  );
}

it('S4 rejects incorrect input and destination resources even when generated kernels match', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const matrix = createTransientView(graph, 'correct-matrix', 'float32', 6),
    wrongMatrix = createTransientView(graph, 'wrong-matrix', 'float32', 6);
  const totals = createTransientView(graph, 'correct-totals', 'float32', 2),
    wrongTotals = createTransientView(graph, 'wrong-totals', 'float32', 2);
  const common = {
    operation: 'GPUGlobalSpatialStatistics',
    rows: 3,
    blockCount: 1,
    first: 0,
    count: 2
  };
  const correct = getGPUCommandNodeKernelSources(
    getColumnSumNodes(graph, {...common, id: 'correct', matrix, totals})
  );
  const wrong = getGPUCommandNodeKernelSources(
    getColumnSumNodes(graph, {...common, id: 'wrong', matrix: wrongMatrix, totals: wrongTotals})
  );
  expect(correct.map(node => node.source)).toEqual(wrong.map(node => node.source));
  expect(() => expectBatchBindings(graph, 'correct', matrix, totals)).not.toThrow();
  expect(() => expectBatchBindings(graph, 'wrong', matrix, totals)).toThrow();
  device.destroy();
});

it('global and permutation column reductions have the same layout, order and metadata (SSR-126)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const rows = 4099; // One 4096-row block and one partial three-row block.
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `column-parity-${serial++}`, format, length);
    const contributor = new GPUGlobalSpatialStatistics({
      id: 'column-parity',
      mode: 'reference',
      weights: {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', rows),
        weights: view('float32', rows)
      },
      values: view('float32', rows),
      statistics: ['moran'],
      output: {summary: view('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length)}
    });
    const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    const original = sources.filter(node => node.id.startsWith('column-parity-sum-d-'));
    for (const [batch, first, count] of [
      ['a', 0, 3],
      ['magnitudes', 0, 2],
      ['d', 25, 2],
      ['b', 3, 19],
      ['c', 22, 3]
    ] as const) {
      const offsets = sources.find(node => node.id === `column-parity-sum-${batch}-offsets`)!;
      expect(offsets.source).toContain(`const FIRST_COLUMN: u32 = ${first}u;`);
      expect(offsets.source).toContain(`const COLUMN_COUNT: u32 = ${count}u;`);
      expect(
        sources.filter(node => node.id.startsWith(`column-parity-sum-${batch}-`))
      ).toHaveLength(3);
      const input = getWGSLKernelNodeRecord(graph, 'column-parity-include')!.bindings.find(
        binding => binding.name === (batch === 'magnitudes' ? 'magnitudes' : 'matrix')
      )!.view as GraphDataView<'float32'>;
      const destination = getWGSLKernelNodeRecord(
        graph,
        batch === 'magnitudes' ? 'column-parity-value-scales' : 'column-parity-finalize'
      )!.bindings.find(
        binding =>
          binding.name ===
          (batch === 'magnitudes' ? 'magnitudeTotals' : `totals${batch.toUpperCase()}`)
      )!.view as GraphDataView<'float32'>;
      expectBatchBindings(graph, `column-parity-sum-${batch}`, input, destination);
    }
    const otherGraph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(otherGraph);
    const replacement = getGPUCommandNodeKernelSources(
      getColumnSumNodes(otherGraph, {
        id: 'column-parity-sum-d',
        operation: 'GPUGlobalSpatialStatistics',
        matrix: createTransientView(otherGraph, 'matrix', 'float32', 27 * rows),
        rows,
        blockCount: 2,
        first: 25,
        count: 2,
        totals: createTransientView(otherGraph, 'totals', 'float32', 2)
      })
    );
    expect(BLOCK_ROWS).toBe(4096);
    expect(original).toHaveLength(3);
    expect(replacement).toHaveLength(3);
    for (let index = 0; index < original.length; index++) {
      const existing = original[index];
      const shared = replacement[index];
      expect(existing.id).toBe(shared.id);
      expect(existing.operation).toBe('GPUGlobalSpatialStatistics');
      expect(shared.operation).toBe(existing.operation);
      expect(shared.constants).toEqual(existing.constants);
      expect(shared.source).toBe(existing.source);
    }
  } finally {
    device.destroy();
  }
});
