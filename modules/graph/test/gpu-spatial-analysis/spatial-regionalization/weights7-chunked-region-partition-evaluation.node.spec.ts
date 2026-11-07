// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURegionPartitionEvaluation,
  GPU_REGION_PARTITION_EVALUATION_LAYOUT
} from '../../../src/gpu-spatial-analysis/spatial-regionalization';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createWeights7TransientChunks} from '../spatial-weights/weights7-chunk-harness';

const ROW_COUNT = 60;

function getIds(chunkRowCounts: number[] | undefined) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'weights7-partition'});
  const chunked = createWeights7TransientChunks(graph, chunkRowCounts ?? [ROW_COUNT]);
  const nodes = new GPURegionPartitionEvaluation({
    values: createTransientView(graph, 'values', 'float32', ROW_COUNT),
    labels: createTransientView(graph, 'labels', 'uint32', ROW_COUNT),
    labelCapacity: 6,
    weights: chunkRowCounts ? chunked : chunked.chunks[0].weights,
    output: {
      summary: createTransientView(
        graph,
        'summary',
        'float32',
        GPU_REGION_PARTITION_EVALUATION_LAYOUT.length
      )
    }
  }).getCommandNodes(graph);
  return getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.id);
}

it('sums link counts once per chunk and copies them into the row table', () => {
  const ids = getIds([35, 25]);
  expect(ids.filter(id => /-links-chunk\d$/.test(id))).toHaveLength(2);
  expect(ids.filter(id => /-link-counts-copy-chunk\d$/.test(id))).toHaveLength(2);
});

it('keeps the whole CSR as a single load-balanced link pass without a copy', () => {
  const ids = getIds(undefined);
  expect(ids.filter(id => id.endsWith('-links'))).toHaveLength(1);
  expect(ids.filter(id => id.includes('chunk') || id.includes('copy'))).toHaveLength(0);
});
