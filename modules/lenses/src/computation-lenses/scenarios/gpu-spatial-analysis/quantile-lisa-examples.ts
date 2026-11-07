// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUQuantileLISA} from '@luma-contributors/graph/gpu-spatial-analysis';
import {getGPUPermutationParameterValues} from '@luma-contributors/graph/gpu-spatial-analysis';
import type {S2CaptureSpec, S2Uploads} from './s2-lens-run';

/** Two selected high-value rows next to one another on an eight-row cycle. */
export function buildQuantileLISAScenario(
  graph: GPUCommandGraph<undefined>,
  uploads: S2Uploads
): Record<string, S2CaptureSpec> {
  const captures: Record<string, S2CaptureSpec> = {};
  const input = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    values: number[] | Uint32Array
  ) => {
    const view = uploads.view(
      name,
      format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values),
      format
    );
    captures[name] = {view, role: 'input'};
    return view;
  };
  const output = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const view = uploads.view(
      name,
      format === 'uint32' ? new Uint32Array(length) : new Float32Array(length),
      format
    );
    captures[name] = {view, role: 'output'};
    return view;
  };
  graph.add(
    new GPUQuantileLISA({
      values: input('values', 'float32', [8, 9, 1, 2, 3, 4, 5, 6]),
      quantileParameters: input('quantile-parameters', 'uint32', [4, 4]),
      parameters: input(
        'permutation-parameters',
        'uint32',
        getGPUPermutationParameterValues({
          seed: 314159,
          permutationCount: 199,
          significanceLevel: 0.1
        })
      ),
      weights: {
        offsets: input('offsets', 'uint32', [0, 2, 4, 6, 8, 10, 12, 14, 16]),
        neighbors: input('neighbors', 'uint32', [1, 7, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 0, 6]),
        weights: input('weights', 'float32', new Array(16).fill(1))
      },
      maximumPermutations: 199,
      maximumNeighbors: 2,
      output: {
        membership: output('membership', 'uint32', 8),
        boundaries: output('boundaries', 'float32', 2),
        validCount: output('valid-count', 'uint32', 1),
        classes: output('classes', 'uint32', 8),
        invalidQuantile: output('invalid-quantile', 'uint32', 1),
        observed: output('local-join-counts', 'float32', 8),
        exceedances: output('exceedances', 'uint32', 8),
        pseudoPValues: output('pseudo-p-values', 'float32', 8),
        status: {overflow: output('overflow', 'uint32', 1)}
      }
    })
  );
  return captures;
}
