// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNeighborSearch} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('neighbor search refuses inherited status fields it cannot publish', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', 2);
  const parameters = createTransientView(graph, 'parameters', 'float32', 12);
  const offsets = createTransientView(graph, 'offsets', 'uint32', 3);
  const neighbors = createTransientView(graph, 'neighbors', 'uint32', 2);
  const weights = createTransientView(graph, 'weights', 'float32', 2);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
  const unsupported = createTransientView(graph, 'unsupported', 'uint32', 1);
  // Exact search has no uncertainty/iteration/approximation diagnostics. Clamped cardinality
  // is the final CSR offset. Refusal prevents a caller mistaking an untouched scalar for status.
  for (const field of [
    'count',
    'uncertainCount',
    'iterationCount',
    'converged',
    'iterationLimitReached',
    'approximation'
  ]) {
    expect(
      () =>
        new GPUNeighborSearch({
          id: 'oracle-search',
          mode: 'knn',
          neighborCount: 1,
          gridSize: [1, 1],
          positions,
          parameters,
          output: {offsets, neighbors, weights, status: {overflow, [field]: unsupported}}
        }),
      field
    ).toThrow(new RegExp(`oracle-search output.status.${field} .*use`));
  }
  device.destroy();
});
