// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUGlobalPermutationTest,
  GPU_GLOBAL_PERMUTATION_SUMMARY
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

let serial = 0;
function createTest(
  graph: GPUCommandGraph,
  permutationBatchSize: number,
  maximumPermutations = 19,
  chunkCount = 2
) {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `materialized-stopping-${serial++}`, format, length);
  return new GPUGlobalPermutationTest({
    id: 'materialized-stopping',
    weights: {
      rowCount: chunkCount * 3,
      chunks: Array.from({length: chunkCount}, (_, chunkIndex) => chunkIndex * 3).map(rowStart => ({
        rowStart,
        rowCount: 3,
        weights: {
          offsets: view('uint32', 4),
          neighbors: view('uint32', 6),
          weights: view('float32', 6)
        }
      }))
    },
    values: view('float32', chunkCount * 3),
    secondValues: view('float32', chunkCount * 3),
    parameters: view('uint32', 4),
    statistic: 'leesL',
    maximumPermutations,
    stopAfterExceedances: 3,
    permutationBatchSize,
    output: {summary: view('float32', GPU_GLOBAL_PERMUTATION_SUMMARY.length)}
  });
}

it('materialized stopping gates materialization, every weight chunk, and sums before an ordered scan', () => {
  const device = createNullWebGPUDevice();
  for (const batchSize of [1, 4, 20]) {
    const graph = new GPUCommandGraph(device);
    const contributor = createTest(graph, batchSize);
    const nodes = contributor.getCommandNodes(graph);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(kernels.some(kernel => kernel.id.endsWith('-materialize-observed'))).toBe(true);
    const batchCount = Math.ceil(19 / batchSize);
    expect(kernels.filter(kernel => /-batch-\d+-scan$/.test(kernel.id))).toHaveLength(batchCount);
    for (const node of nodes.filter(node =>
      /-batch-\d+-(materialize|pair-sums|pairs-chunk\d+)$/.test(node.id)
    )) {
      expect(node.condition, node.id).toBeDefined();
    }
    for (const kernel of kernels.filter(kernel =>
      /-batch-\d+-(materialize|scan|pair-sums|pairs-chunk\d+)$/.test(kernel.id)
    )) {
      // wgsl_reflect tokenizes the existing local names `workgroup` and `block` as keywords.
      const parseable = kernel.source
        .replace(/\bworkgroup\b(?!>)/g, 'permutationWorkgroup')
        .replace(/\bblock\b/g, 'rowBlock');
      expect(() => new WgslReflect(parseable), kernel.id).not.toThrow();
    }
    const materialize = kernels.find(kernel => kernel.id.endsWith('-batch-0-materialize'))!;
    expect(materialize.source).toContain('+ 1u + slot');
    expect(kernels.find(kernel => kernel.id.endsWith('-sequential-init'))!.source).toContain(
      'stateOffset + 8u'
    );
  }
  device.destroy();
});

it('materialized stopping retains named batch, storage-element and unrolled-batch limits', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  for (const batchSize of [0, 1.5, 21]) {
    expect(() => createTest(graph, batchSize)).toThrow(
      /materialized-stopping permutationBatchSize.*\[1, 20\]/
    );
  }
  expect(() => createTest(graph, 1, 4097)).toThrow(
    /materialized-stopping permutation batches.*4096/
  );
  expect(() => createTest(graph, 1, 4096)).not.toThrow();
  expect(() => createTest(graph, 2, 7999, 8)).toThrow(/estimated command node count.*32768/);
  expect(() => createTest(graph, 2 ** 20, 2 ** 20, 12)).toThrow(
    /permutationBatchSize x rows.*33554432/
  );
  device.destroy();
});

it('materialization checks the actual device storage binding limit before adding nodes', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 16}
  });
  const graph = new GPUCommandGraph(device);
  const contributor = createTest(graph, 4);
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /materialized-stopping materialized permutation values.*limit 16/
  );
  device.destroy();
});
