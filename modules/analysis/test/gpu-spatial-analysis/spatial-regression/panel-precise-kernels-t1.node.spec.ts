// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {getPanelPreciseDiagnosticNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-precise-diagnostics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it('t1 precise panel diagnostics use valid WGSL names in row and multi-tile merge kernels', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `precise-t1-${serial++}`, format, length);
    const nodes = getPanelPreciseDiagnosticNodes(
      graph,
      'precise-t1',
      {
        periodCount: 2,
        predictorCount: 1,
        predictors: view('float32', 128),
        response: view('float32', 128),
        weights: {
          offsets: view('uint32', 65),
          neighbors: view('uint32', 64),
          weights: view('float32', 64)
        },
        output: {
          table: view('float32', 12),
          summary: view('float32', 32),
          status: {code: view('uint32', 1)}
        }
      },
      {lags: view('float32', 128), residuals: view('float32', 128), projected: view('float32', 128)}
    );
    const kernels = getGPUCommandNodeKernelSources(nodes);
    expect(kernels.some(kernel => kernel.id.includes('-pooled-merge-'))).toBe(true);
    for (const kernel of kernels)
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
  } finally {
    device.destroy();
  }
});
