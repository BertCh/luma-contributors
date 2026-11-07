// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPURasterWatershedSegmentation} from '../../src/raster-analysis/gpu-raster-watershed-segmentation';
import {makeGPURasterMetadata} from '../../src/raster-analysis/raster-grid/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {auditWGSLModule, getWGSLModuleUsage} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import {calculateWatershedReference, createWatershedWorkload} from './backlog-performance-fixtures';
import {calculateWatershedScanReference} from './watershed-scan-reference';

it('watershed scan and lazy-heap oracles agree on seeded plateau workloads', () => {
  for (let seed = 1; seed <= 32; seed++) {
    const workload = createWatershedWorkload(8, seed);
    for (const values of [workload.values, workload.updatedValues]) {
      expect(
        calculateWatershedScanReference({
          width: 8,
          height: 8,
          values,
          markers: workload.markers,
          connectivity: 4,
          maximumSteps: 64
        })
      ).toEqual(calculateWatershedReference(workload, values));
    }
  }
});

for (const storageLimit of [16384, 32768])
  for (const count of [1, 1023, 1024, 1025, 2047, 2048, 2049, 16384])
    it(`watershed ${count} cells fits ${storageLimit}-byte workgroup storage`, () => {
      const device = createNullWebGPUDevice();
      try {
        Object.assign(device.limits, {maxComputeWorkgroupStorageSize: storageLimit});
        const graph = new GPUCommandGraph(device);
        let serial = 0;
        const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
          createTransientView(graph, `watershed-${serial++}`, format, length);
        const contributor = new GPURasterWatershedSegmentation({
          grid: makeGPURasterMetadata({width: count, height: 1}),
          input: {
            id: 'maximum',
            format: 'float32',
            validity: view('uint32', count),
            storage: {kind: 'buffer', values: view('float32', count)}
          },
          markers: view('uint32', count),
          connectivity: 8,
          output: {labels: view('uint32', count), status: {converged: view('uint32', 1)}}
        });
        for (const kernel of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
          // The shared audit assumes core limits; check this device's workgroup limit below.
          expect(
            auditWGSLModule(kernel.source, kernel.constants).filter(
              finding => finding.check !== 'workgroup bytes'
            )
          ).toEqual([]);
          const usage = getWGSLModuleUsage(kernel.source, kernel.constants);
          expect(usage.storageBuffers).toBeLessThanOrEqual(8);
          expect(usage.privateBytes).toBeLessThanOrEqual(8192);
          expect(usage.functionBytes).toBeLessThanOrEqual(8192);
          expect(usage.workgroupBytes).toBeLessThanOrEqual(storageLimit);
          if (kernel.operation === 'GPURasterWatershedSegmentation') {
            const fitsWorkgroup = count * 16 <= storageLimit;
            expect(usage.workgroupBytes > 0).toBe(fitsWorkgroup);
            expect(usage.storageBuffers).toBe(fitsWorkgroup ? 5 : 7);
          }
        }
      } finally {
        device.destroy();
      }
    });
