// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {it, expect} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUMantelTest} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount,
  getPrivateMemoryBytes,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES
} from '../../utils/gpu-contributor-test-utils';
import {computeMantelTestOnCPU} from '../../../src/gpu-spatial-analysis/space-time-tests/cpu-mantel-test-operation';
import {expectResidualInference, getResidualInputs} from './mantel-residual-w5-fixture';

for (const [star, seed, stopping] of [
  [true, 0, true],
  [true, 84, false],
  [false, 8, false]
] as const) {
  it(`W5-06/W5-07 retains Pearson residuals: star=${star}, seed=${seed}, stopping=${stopping}`, () => {
    const inputs = getResidualInputs(star, seed);
    const result = computeMantelTestOnCPU(
      {
        offsets: inputs.offsets.values,
        neighbors: inputs.neighbors.values,
        weights: inputs.weights.values,
        distances: inputs.distances.values,
        times: inputs.times.values,
        parameters: inputs.parameters.values
      },
      {maximumPermutations: 2, ...(stopping ? {stopAfterExceedances: 1} : {})}
    );
    expectResidualInference(
      {...result, used: result.permutationCount, usable: result.usablePermutationCount},
      star,
      stopping
    );
  });
}

it('W5-06/W5-07 residual kernels assemble within the storage and private-memory limits', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    for (const stopping of [false, true]) {
      const view = <Format extends 'uint32' | 'float32'>(
        name: string,
        format: Format,
        length: number
      ) => createTransientView(graph, name + '-' + stopping, format, length);
      const contributor = new GPUMantelTest({
        id: 'residual-' + stopping,
        maximumPermutations: 2,
        ...(stopping ? {stopAfterExceedances: 1} : {}),
        pairs: {
          offsets: view('offsets', 'uint32', 4),
          neighbors: view('neighbors', 'uint32', 3),
          weights: view('weights', 'float32', 3),
          distances: view('distances', 'float32', 3)
        },
        times: view('times', 'float32', 3),
        parameters: view('parameters', 'uint32', 4),
        output: {
          statistics: view('statistics', 'float32', 3),
          summary: view('summary', 'float32', 12),
          permutationCount: view('used', 'uint32', 1),
          usablePermutationCount: view('usable', 'uint32', 1)
        }
      });
      const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      for (const kernel of kernels) {
        expect(getStorageBindingCount(kernel.source)).toBeLessThanOrEqual(8);
        expect(getPrivateMemoryBytes(kernel.source, kernel.constants)).toBeLessThanOrEqual(
          CORE_PRIVATE_MEMORY_LIMIT_BYTES
        );
      }
    }
  } finally {
    device.destroy();
  }
});
