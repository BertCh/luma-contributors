// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUEmergingHotSpotsParameterValues,
  GPUEmergingHotSpots
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {createWeights7TransientChunks} from '../spatial-weights/weights7-chunk-harness';

const CELL_COUNT = 60;
const SLICE_COUNT = 8;

function getKernels(chunkRowCounts: number[] | undefined) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'weights7-hot-spots'});
  const chunked = createWeights7TransientChunks(graph, chunkRowCounts ?? [CELL_COUNT]);
  const weights = chunkRowCounts ? chunked : chunked.chunks[0].weights;
  const view = <Format extends 'float32' | 'uint32' | 'sint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const binCount = CELL_COUNT * SLICE_COUNT;
  const nodes = new GPUEmergingHotSpots({
    values: view('values', 'float32', binCount),
    weights,
    sliceCount: SLICE_COUNT,
    parameters: view(
      'parameters',
      'float32',
      getGPUEmergingHotSpotsParameterValues({
        radius: 1,
        temporalWindow: 1,
        multipleTesting: 'benjamini-hochberg',
        confidenceLevel: 0.95,
        trendSignificanceLevel: 0.1,
        minimumTrendSliceCount: 4
      }).length
    ),
    output: {
      giZScores: view('gi', 'float32', binCount),
      trendZ: view('trendZ', 'float32', CELL_COUNT),
      trendP: view('trendP', 'float32', CELL_COUNT),
      trendS: view('trendS', 'sint32', CELL_COUNT),
      category: view('category', 'uint32', CELL_COUNT),
      hotSliceCount: view('hot', 'uint32', CELL_COUNT),
      coldSliceCount: view('cold', 'uint32', CELL_COUNT)
    }
  }).getCommandNodes(graph);
  return getGPUCommandNodeKernelSources(nodes);
}

it('dispatches one Gi* kernel per weights chunk and keeps the other nodes single', () => {
  const kernels = getKernels([25, 25, 10]);
  const giKernels = kernels.filter(kernel => kernel.id.includes('gi-star'));
  expect(giKernels.map(kernel => kernel.id.replace(/^.*gi-star/, ''))).toEqual([
    '-chunk0',
    '-chunk1',
    '-chunk2'
  ]);
  for (const [index, rowStart] of [0, 25, 50].entries()) {
    expect(giKernels[index].source).toContain(`const WEIGHTS_ROW_START: u32 = ${rowStart}u;`);
  }
  expect(kernels.filter(kernel => kernel.id.includes('mann-kendall'))).toHaveLength(1);
  expect(kernels.filter(kernel => kernel.id.includes('block-sums'))).toHaveLength(1);
  for (const {id, source} of kernels) {
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
  }
});

it('keeps the whole-CSR kernel as a single dispatch without chunk constants', () => {
  const kernels = getKernels(undefined);
  const giKernels = kernels.filter(kernel => kernel.id.includes('gi-star'));
  expect(giKernels).toHaveLength(1);
  expect(giKernels[0].id.endsWith('gi-star')).toBe(true);
  expect(giKernels[0].source).not.toContain('WEIGHTS_ROW_START');
});

it('sizes each chunk kernel from its own rows', () => {
  const kernels = getKernels([40, 20]);
  const giKernels = kernels.filter(kernel => kernel.id.includes('gi-star'));
  expect(giKernels[0].source).toContain('cell < WEIGHTS_ROW_START + 40u');
  expect(giKernels[1].source).toContain('cell < WEIGHTS_ROW_START + 20u');
});
