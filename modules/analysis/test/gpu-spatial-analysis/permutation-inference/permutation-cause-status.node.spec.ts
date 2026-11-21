// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPULocalPermutationTest,
  validateLocalPermutationSettings
} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-local-permutation-test';
import {GPUGlobalPermutationTest} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-global-permutation-test';
import {
  getGPUCommandNodeKernelSources,
  createNullWebGPUDevice,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

it('SSR-113 retains separately written cause ports without exceeding eight bindings', () => {
  for (const scope of ['local', 'global'] as const) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `status-${serial++}`, format, length);
    const status = {
      weightsOverflow: view('uint32', 1),
      candidateOverflow: view('uint32', 1),
      invalidInput: view('uint32', 1)
    };
    const props = {
      weights: {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 6),
        weights: view('float32', 6),
        status: {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)}
      },
      values: view('float32', 4),
      parameters: view('uint32', 4),
      maximumPermutations: 9
    };
    const contributor =
      scope === 'local'
        ? new GPULocalPermutationTest({
            ...props,
            statistic: 'localMoran',
            output: {
              exceedances: view('uint32', 4),
              pseudoPValues: view('float32', 4),
              permutationCounts: view('uint32', 4),
              status: {...status, neighborOverflow: view('uint32', 1), overflow: view('uint32', 1)}
            }
          })
        : new GPUGlobalPermutationTest({
            ...props,
            statistic: 'moran',
            output: {summary: view('float32', 12), status}
          });
    const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    if (scope === 'local')
      expect(sources.find(({id}) => id.endsWith('-cause-settings'))!.source).toContain(
        '!isPermutationCountValid() || !isPermutationSignificanceValid()'
      );
    for (const suffix of ['cause-weights-0', 'cause-candidates-0', 'cause-settings'])
      expect(sources.some(({id}) => id.endsWith(suffix))).toBe(true);
    if (scope === 'local')
      expect(sources.some(({id}) => id.endsWith('cause-neighbors-0'))).toBe(true);
    for (const {id, source} of sources)
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    device.destroy();
  }
});

it('VAL-08 exposes device-independent local permutation settings for composing parents', () => {
  expect(() =>
    validateLocalPermutationSettings('parent', {maximumPermutations: 9, maximumNeighbors: NaN})
  ).toThrow(/parent.*maximumNeighbors/);
  expect(() =>
    validateLocalPermutationSettings('parent', {maximumPermutations: 9, maximumNeighbors: 32})
  ).not.toThrow();
});
