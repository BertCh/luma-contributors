// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {
  auditWGSLModule,
  getWGSLModuleUsage
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('ordinary SUR stages fit core shader budgets with maximum shapes and optional inputs', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const [equations, columns] of [
      [2, 7],
      [3, 4],
      [4, 3]
    ]) {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `sur-${serial++}`, format, length);
      const size = equations * (columns + 1);
      const contributor = new GPUSeeminglyUnrelatedRegression({
        equationCount: equations,
        predictorCount: columns,
        rowCount: 65536,
        predictors: view('float32', equations * columns * 65536),
        responses: view('float32', equations * 65536),
        initialResiduals: view('float32', equations * 65536),
        mask: view('uint32', 65536),
        output: {
          coefficients: view('float32', size),
          covariance: view('float32', size * size),
          residualCovariance: view('float32', equations * equations),
          residuals: view('float32', equations * 65536),
          status: view('uint32', 1)
        }
      });
      for (const kernel of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
        const usage = getWGSLModuleUsage(kernel.source, kernel.constants);
        expect(usage.storageBuffers).toBeLessThanOrEqual(8);
        expect(usage.functionBytes).toBeLessThanOrEqual(8192);
        expect(usage.privateBytes).toBeLessThanOrEqual(8192);
        expect(usage.workgroupStorageBytes).toBeLessThanOrEqual(16384);
      }
    }
  } finally {
    device.destroy();
  }
});
