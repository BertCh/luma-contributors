// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUVariogramModelFit} from '../../../src/gpu-dataframe/pair-statistics/gpu-variogram-model-fit';
import {getParallelVariogramModelFitNodes} from '../../../src/gpu-dataframe/pair-statistics/variogram-model-fit-nodes';
import {
  auditWGSLModule,
  getWGSLModuleUsage
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('parallel variogram search fits core binding and shader budgets at maximum search options', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const weighting of ['pairs', 'none', 'cressie'] as const) {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'float32x4' | 'uint32'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `parallel-${serial++}`, format, length);
      const contributor = new GPUVariogramModelFit({
        distances: view('float32', 256),
        semivariances: view('float32', 256),
        pairCounts: view('uint32', 256),
        weighting,
        rangeSteps: 1024,
        maximumIterations: 64,
        output: {
          parameters: view('float32x4', 3),
          diagnostics: view('float32x4', 3),
          status: view('uint32', 3)
        }
      });
      const kernels = getGPUCommandNodeKernelSources(
        getParallelVariogramModelFitNodes(graph, contributor)
      );
      expect(kernels.length).toBeGreaterThan(1);
      for (const kernel of kernels) {
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
        const usage = getWGSLModuleUsage(kernel.source, kernel.constants);
        expect(usage.storageBuffers).toBeLessThanOrEqual(8);
        expect(usage.privateBytes).toBeLessThanOrEqual(8192);
        expect(usage.functionBytes).toBeLessThanOrEqual(8192);
        expect(usage.workgroupBytes).toBeLessThanOrEqual(16384);
      }
    }
  } finally {
    device.destroy();
  }
});
