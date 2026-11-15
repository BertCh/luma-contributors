// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {GPUNetworkRationalAccess} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

for (const originCount of [1, 17, 33, 64]) {
  for (const facilityCount of [1, 3, 32]) {
    it(`cooperatively evaluates ${originCount} origins and ${facilityCount} facilities in one workgroup`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let sequence = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `rational-parallel-${sequence++}`, format, length);
      try {
        const contributor = new GPUNetworkRationalAccess({
          costs: view('float32', originCount * facilityCount),
          demand: view('float32', originCount),
          supply: view('float32', facilityCount),
          parameters: view('float32', 6),
          maximumCycles: 4096,
          output: {
            assignments: view('float32', originCount * facilityCount),
            scores: view('float32', originCount),
            unservedDemand: view('float32', originCount),
            summary: view('uint32', 4)
          }
        });
        const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
        expect(kernels).toHaveLength(2);
        const solve = kernels[1];
        expect(solve.source).toContain('@workgroup_size(64)');
        expect(solve.constants['INVOCATION_COUNT']).toBe(64);
        expect(solve.constants['WGSL_KERNEL_DISPATCH_X']).toBe(1);
        expect(solve.constants['WGSL_KERNEL_DISPATCH_Y']).toBe(1);
        expect(solve.source).toContain('workgroupUniformLoad(&continueCycles)');
        expect(solve.source).not.toContain('if (index >= INVOCATION_COUNT)');
        for (const kernel of kernels) {
          expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
          expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
          expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
        }
      } finally {
        device.destroy();
      }
    });
  }
}
