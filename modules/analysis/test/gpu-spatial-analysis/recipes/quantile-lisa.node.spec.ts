// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUQuantileLISA,
  type GPUQuantileLISAProps
} from '../../../src/gpu-spatial-analysis/recipes/quantile-lisa-recipe';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  computeQuantileLISAReference,
  QUANTILE_LISA_REFERENCE_CASES
} from './quantile-lisa-reference';

it('quantile LISA matches pinned type-five and half-open class fixtures', () => {
  for (const fixture of QUANTILE_LISA_REFERENCE_CASES) {
    const result = computeQuantileLISAReference(
      Float32Array.from(fixture.values),
      fixture.quantileCount,
      fixture.selectedQuantile
    );
    expect(result.boundaries, fixture.name).toEqual(fixture.boundaries);
    expect(Array.from(result.membership), fixture.name).toEqual(fixture.membership);
  }
});

export function createQuantileLISAProps(graph: GPUCommandGraph): GPUQuantileLISAProps {
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `quantile-lisa-${serial++}`, format, length);
  return {
    values: view('float32', 8),
    quantileParameters: view('uint32', 2),
    parameters: view('uint32', 8),
    maximumPermutations: 99,
    weights: {
      offsets: view('uint32', 9),
      neighbors: view('uint32', 16),
      weights: view('float32', 16)
    },
    output: {
      membership: view('uint32', 8),
      boundaries: view('float32', 2),
      validCount: view('uint32', 1),
      classes: view('uint32', 8),
      invalidQuantile: view('uint32', 1),
      exceedances: view('uint32', 8),
      pseudoPValues: view('float32', 8),
      status: {overflow: view('uint32', 1)}
    }
  };
}

it('quantile LISA validates controls and ownership and emits only resident nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createQuantileLISAProps(graph);
    const nodes = new GPUQuantileLISA(props).getCommandNodes(graph);
    expect(nodes.length).toBeGreaterThan(20);
    for (const kernel of getGPUCommandNodeKernelSources(nodes))
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
    expect(nodes.map(node => node.id)).toContain('quantile-lisa-membership');
    expect(
      () =>
        new GPUQuantileLISA({
          ...props,
          quantileParameters: createTransientView(graph, 'wrong-control', 'uint32', 1)
        })
    ).toThrow();
    expect(
      () =>
        new GPUQuantileLISA({
          ...props,
          output: {...props.output, membership: props.weights.neighbors}
        })
    ).toThrow();
    expect(() =>
      new GPUQuantileLISA({
        ...props,
        output: {...props.output, membership: props.values as never}
      }).getCommandNodes(graph)
    ).toThrow();
  } finally {
    device.destroy();
  }
});
