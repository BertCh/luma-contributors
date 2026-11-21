// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGroupStatistics} from '../../../src/dataframe-analysis/group-statistics/gpu-group-statistics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {WgslReflect} from 'wgsl_reflect';

it('DPC-109 group published and moment means use exact integer division', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `finite-${serial++}`, format, length);
  try {
    const nodes = new GPUGroupStatistics({
      keys: view('uint32', 14),
      columns: [
        {
          values: view('float32', 14),
          statistics: ['mean', 'variance'],
          output: {means: view('float32', 3), variances: view('float32', 3)}
        }
      ],
      output: {
        keys: view('uint32', 3),
        counts: view('uint32', 3),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      }
    }).getCommandNodes(graph);
    const source = getGPUCommandNodeKernelSources(nodes).find(kernel =>
      kernel.source.includes('let unitExponent = getFixedPointUnitExponent')
    )!;
    expect(source.source).toContain('getFiniteFixedPointMean(words, unitExponent, count)');
    expect(source.source).toMatch(/meanScratch\[.*2u \* index\] = mean/);
    expect(() => new WgslReflect(source.source)).not.toThrow();
    expect(auditWGSLModule(source.source, source.constants)).toEqual([]);
  } finally {
    device.destroy();
  }
});
