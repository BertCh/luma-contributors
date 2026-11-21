// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {WgslReflect} from 'wgsl_reflect';

it('earlier-fix mean decode retains the integer count through final rounding', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `finite-${serial++}`, format, length);
  try {
    const nodes = new GPUKeyLookup({
      leftKeys: view('uint32', 1),
      rightKeys: view('uint32', 7),
      aggregates: [{operation: 'mean', column: view('float32', 7), output: view('float32', 1)}],
      output: {}
    }).getCommandNodes(graph);
    const source = getGPUCommandNodeKernelSources(nodes).find(kernel =>
      kernel.source.includes('let totalWords')
    )!;
    expect(source.source).toContain('getFiniteFixedPointMean(totalWords, unitExponent, n)');
    expect(() => new WgslReflect(source.source)).not.toThrow();
    expect(auditWGSLModule(source.source, source.constants)).toEqual([]);
  } finally {
    device.destroy();
  }
});

it.each([
  'sum',
  'mean'
] as const)('B1 %s modules independently resolve every WGSL call', operation => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `module-${serial++}`, format, length);
  try {
    const kernels = getGPUCommandNodeKernelSources(
      new GPUKeyLookup({
        leftKeys: view('uint32', 2),
        rightKeys: view('uint32', 3),
        aggregates: [{operation, column: view('float32', 3), output: view('float32', 2)}],
        output: {}
      }).getCommandNodes(graph)
    );
    for (const suffix of ['sum-scale', 'accumulate', 'finish']) {
      const kernel = kernels.find(candidate => candidate.id === `key-join-aggregate-0-${suffix}`)!;
      expect(kernel, suffix).toBeDefined();
      expect(auditWGSLModule(kernel.source, kernel.constants), suffix).toEqual([]);
      expect(kernel.source.includes('fn getFiniteFixedPointMean'), suffix).toBe(
        operation === 'mean' && suffix === 'finish'
      );
    }
  } finally {
    device.destroy();
  }
});
