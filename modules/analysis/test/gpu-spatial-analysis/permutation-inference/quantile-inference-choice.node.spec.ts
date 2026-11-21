// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUQuantileLISA,
  type GPUQuantileLISAProps
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-quantile-lisa';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {isPermutationRatioSignificant} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-significance';

it('W5-10 compares the exact ratio to the packed cutoff, including equality', () => {
  // .04 packs below 1/25; 1/32 is an exactly representable dyadic equality.
  expect(isPermutationRatioSignificant(1, 25, Math.fround(0.04))).toBe(false);
  expect(isPermutationRatioSignificant(1, 32, Math.fround(0.03125))).toBe(true);
});

it('DEF-01 requires an explicit Quantile LISA inference rule', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `choice-${serial++}`, format, length);
  const props = {
    values: view('float32', 4),
    weights: {
      offsets: view('uint32', 5),
      neighbors: view('uint32', 8),
      weights: view('float32', 8),
      status: {complete: view('uint32', 1)}
    },
    quantileParameters: view('uint32', 2),
    parameters: view('uint32', 4),
    maximumPermutations: 99,
    output: {
      membership: view('uint32', 4),
      boundaries: view('float32', 2),
      validCount: view('uint32', 1),
      classes: view('uint32', 4),
      invalidQuantile: view('uint32', 1),
      exceedances: view('uint32', 4),
      pseudoPValues: view('float32', 4),
      status: {overflow: view('uint32', 1)}
    }
  };
  expect(() => new GPUQuantileLISA(props as GPUQuantileLISAProps)).toThrow(
    /alternative.*greater.*two-sided.*geoda-legacy/
  );
  for (const alternative of ['greater', 'two-sided', 'geoda-legacy'] as const) {
    const contributor = new GPUQuantileLISA({
      ...props,
      id: alternative,
      alternative,
      output: {...props.output, tested: view('uint32', 4)}
    });
    const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(sources.find(({id}) => id.endsWith('-geoda-tail'))!.source).toContain(
      'isSignificant = select(0u, 1u, isPermutationRatioSignificant('
    );
    expect(sources.some(({source}) => source.includes('tested[testedOffset + index] ='))).toBe(
      true
    );
    for (const {id, source} of sources)
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
  }
  const rows = 5000000;
  const large = {
    ...props,
    id: 'large-row-bindings',
    alternative: 'greater' as const,
    values: view('float32', rows),
    weights: {...props.weights, offsets: view('uint32', rows + 1)},
    output: {
      ...props.output,
      membership: view('uint32', rows),
      classes: view('uint32', rows),
      exceedances: view('uint32', rows),
      pseudoPValues: view('float32', rows)
    }
  };
  // Five million rows need 4*(rows+1) bytes for the largest row-proportional binding,
  // rather than the former fictitious 32*rows matrix (which exceeds 128 MiB).
  expect(() => new GPUQuantileLISA(large).getCommandNodes(graph)).not.toThrow();
  device.destroy();
});
