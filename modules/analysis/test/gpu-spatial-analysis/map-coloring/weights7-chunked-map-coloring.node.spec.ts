// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUMapColoringParameterValues,
  GPUMapColoring
} from '../../../src/gpu-spatial-analysis/map-coloring/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createWeights7TransientChunks} from '../spatial-weights/weights7-chunk-harness';

const ROW_COUNT = 50;
const MAXIMUM_ITERATIONS = 3;

function getKernels(
  chunkRowCounts: number[] | undefined,
  priority: 'largest-degree-first' | 'random'
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'weights7-coloring'});
  const chunked = createWeights7TransientChunks(graph, chunkRowCounts ?? [ROW_COUNT]);
  const view = <Format extends 'uint32'>(name: string, format: Format, length: number) =>
    createTransientView(graph, name, format, length);
  const nodes = new GPUMapColoring({
    weights: chunkRowCounts ? chunked : chunked.chunks[0].weights,
    priority,
    maximumIterations: MAXIMUM_ITERATIONS,
    parameters: view('parameters', 'uint32', getGPUMapColoringParameterValues({seed: 1}).length),
    output: {
      colors: view('colors', 'uint32', ROW_COUNT),
      conflictCount: view('conflicts', 'uint32', 1)
    }
  }).getCommandNodes(graph);
  return getGPUCommandNodeKernelSources(nodes);
}

it('dispatches one select kernel per chunk per round and one conflicts kernel per chunk', () => {
  const ids = getKernels([30, 20], 'largest-degree-first').map(kernel => kernel.id);
  for (let round = 0; round < MAXIMUM_ITERATIONS; round++) {
    expect(ids.filter(id => id.startsWith(`map-coloring-round-${round}-select`))).toHaveLength(2);
  }
  expect(ids.filter(id => id.includes('-conflicts-chunk'))).toHaveLength(2);
  expect(ids.filter(id => id.includes('-degrees-chunk'))).toHaveLength(2);
});

it('gathers neighbor degrees once for chunked degree priority only', () => {
  expect(
    getKernels([30, 20], 'random').filter(kernel => kernel.id.includes('degrees'))
  ).toHaveLength(0);
  const whole = getKernels(undefined, 'largest-degree-first');
  expect(whole.filter(kernel => kernel.id.includes('degrees'))).toHaveLength(0);
  for (const kernel of whole) {
    expect(kernel.source, kernel.id).not.toContain('WEIGHTS_ROW_START');
  }
  const select = whole.find(kernel => kernel.id.includes('round-0-select'));
  expect(select?.source).toContain('offsets[offsetsOffset + other + 1u]');
});

it('starts each chunk select kernel at its first row', () => {
  const selects = getKernels([30, 20], 'random').filter(kernel =>
    kernel.id.includes('round-0-select')
  );
  expect(selects[0].source).toContain('const WEIGHTS_ROW_START: u32 = 0u;');
  expect(selects[1].source).toContain('const WEIGHTS_ROW_START: u32 = 30u;');
});
