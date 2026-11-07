// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGlobalPermutationTest,
  GPU_GLOBAL_PERMUTATION_SUMMARY,
  GPU_PERMUTATION_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {getGPUGlobalPermutationBatchSize} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-global-permutation-test';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createChunkedTest(
  graph: GPUCommandGraph,
  chunkRowCounts: readonly number[],
  stopAfterExceedances?: number
) {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `weights7-view-${serial++}`, format, length);
  const rows = chunkRowCounts.reduce((total, rowCount) => total + rowCount, 0);
  let rowStart = 0;
  return new GPUGlobalPermutationTest({
    id: `weights7-global-${serial++}`,
    weights: {
      rowCount: rows,
      chunks: chunkRowCounts.map(rowCount => {
        const chunk = {
          rowStart,
          rowCount,
          weights: {
            offsets: view('uint32', rowCount + 1),
            neighbors: view('uint32', rowCount * 3),
            weights: view('float32', rowCount * 3)
          }
        };
        rowStart += rowCount;
        return chunk;
      })
    },
    values: view('float32', rows),
    parameters: view('uint32', GPU_PERMUTATION_PARAMETER_LENGTH),
    statistic: 'moran',
    maximumPermutations: 999,
    stopAfterExceedances,
    output: {summary: view('float32', GPU_GLOBAL_PERMUTATION_SUMMARY.length)}
  });
}

it('stopAfterExceedances accepts chunked weights with one gated pair dispatch per chunk and batch', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'weights7-stop-chunked'});
  for (const chunkCount of [1, 2, 4, 8]) {
    const chunkRowCounts = Array.from({length: chunkCount}, () => 5);
    const nodes = createChunkedTest(graph, chunkRowCounts, 10).getCommandNodes(graph);
    const batches = Math.ceil(999 / getGPUGlobalPermutationBatchSize(5 * chunkCount, 999));
    for (let batch = 0; batch < batches; batch++) {
      const pairNodes = nodes.filter(node =>
        new RegExp(`-batch-${batch}-pairs(-chunk\\d+)?$`).test(node.id)
      );
      expect(pairNodes, `batch ${batch} chunks=${chunkCount}`).toHaveLength(chunkCount);
    }
    // Scan, sums and finalize stay one kernel per batch: dispatch count is linear in chunks.
    expect(nodes.filter(node => /-batch-\d+-scan$/.test(node.id))).toHaveLength(batches);
    expect(nodes.filter(node => /-pairs-observed/.test(node.id))).toHaveLength(chunkCount);
  }
});
