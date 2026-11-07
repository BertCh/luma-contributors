// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {GPUBoundedFacilityAllocation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-bounded-facility-allocation';
import {GPUNetworkFacilityOptimization} from '../../../src/gpu-network/network-optimization/gpu-network-facility-optimization';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it('p-center accepts device-sized demand matrices and rejects oversized storage while capacity mode stays bounded', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `facility-${serial++}`, format, length);
  const facilityCount = 32;
  const maximumDemandCount = Math.floor(
    device.limits.maxStorageBufferBindingSize / (4 * facilityCount)
  );
  const make = (demandCount: number, operation: 'p-center' | 'capacitated-p-median' = 'p-center') =>
    new GPUBoundedFacilityAllocation({
      operation,
      facilityCount,
      selectCount: 2,
      costs: view('float32', demandCount * facilityCount),
      demand: view('float32', demandCount),
      capacities: view('float32', facilityCount),
      output: {
        selectedFacilities: view('uint32', facilityCount),
        assignments: view('uint32', demandCount),
        objective: view('float32', 1),
        status: view('uint32', 3)
      }
    });
  try {
    const nodes = make(maximumDemandCount).getCommandNodes(graph);
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
    expect(() => make(maximumDemandCount + 1).getCommandNodes(graph)).toThrow(/demandCount/);
    expect(() => make(4097, 'capacitated-p-median')).toThrow(/demandCount/);
  } finally {
    device.destroy();
  }
});

it('network candidate batches reuse pipelines and parse at the largest supported search', () => {
  const device = createNullWebGPUDevice();
  const makeNodes = (maximumCandidates: number) => {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `network-facility-${serial++}`, format, length);
    return new GPUNetworkFacilityOptimization({
      operation: 'capacitated-p-median',
      facilityCount: 4,
      selectCount: 2,
      maximumCandidates,
      costs: view('float32', 40),
      demand: view('float32', 10),
      capacities: view('float32', 4),
      threshold: view('float32', 1),
      output: {
        selectedFacilities: view('uint32', 4),
        assignments: view('uint32', 10),
        objective: view('float32', 1),
        status: view('uint32', 4)
      }
    }).getCommandNodes(graph);
  };
  try {
    const oneBatch = makeNodes(65536);
    const twoBatches = makeNodes(65537);
    const maximum = makeNodes(1048576);
    expect(twoBatches.length).toBeGreaterThan(oneBatch.length);
    expect(getGPUCommandNodePipelineKeys(twoBatches).size).toBe(
      getGPUCommandNodePipelineKeys(oneBatch).size
    );
    expect(getGPUCommandNodePipelineKeys(maximum).size).toBe(
      getGPUCommandNodePipelineKeys(oneBatch).size
    );
    for (const kernel of getGPUCommandNodeKernelSources(twoBatches)) {
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});
