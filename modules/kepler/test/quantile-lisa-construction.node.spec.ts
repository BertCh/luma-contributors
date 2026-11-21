// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {createNullWebGPUDevice} from '../../analysis/src/testing/gpu-contributor-test-utils';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {addWeightedQuantileLISA} from '../src/analysis/weighted-quantile-lisa';

test('all Quantile LISA alternatives construct with matching settings and provenance', () => {
  for (const choice of [
    {alternative: 'geoda-legacy' as const},
    {alternative: 'greater' as const},
    {alternative: 'two-sided' as const}
  ]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device, {id: 'legacy-quantile'});
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const source = {datasetId: 'cycle', revision: 1};
    try {
      const result = addWeightedQuantileLISA(graph, {
        source,
        population: 'all',
        values: view('values', 'float32', 8),
        mask: view('mask', 'uint32', 8),
        settings: {
          quantileCount: 2,
          selectedQuantile: 2,
          seed: 7,
          permutationCount: 9,
          significanceLevel: 0.05,
          multipleTesting: 'none',
          islands: 'exclude',
          quantileMethod: 'type-5',
          ...choice
        },
        topology: {
          source,
          maximumNeighbors: 2,
          mask: view('topology-mask', 'uint32', 8),
          weights: {
            offsets: view('offsets', 'uint32', 9),
            neighbors: view('neighbors', 'uint32', 16),
            weights: view('weights', 'float32', 16),
            information: {rowCount: 8, transform: 'binary', symmetric: true, hasSelfWeights: false}
          },
          status: {
            overflow: view('overflow', 'uint32', 1),
            requiredCount: view('required', 'uint32', 1),
            invalidCount: view('invalid', 'uint32', 1)
          }
        }
      });
      try {
        const {alternative} = choice;
        expect(result.settings.alternative).toBe(alternative);
        expect(result.inference.alternative).toBe(alternative);
        expect(result.inference.method).toBe(
          alternative === 'geoda-legacy' ? 'legacy-directed-pseudo' : alternative
        );
      } finally {
        result.destroyParameters();
      }
    } finally {
      device.destroy();
    }
  }
});
