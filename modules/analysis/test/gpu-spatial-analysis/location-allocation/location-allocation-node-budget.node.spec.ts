// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPULocationAllocation,
  type GPULocationAllocationProps
} from '../../../src/gpu-spatial-analysis/location-allocation/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let viewCount = 0;

type Configuration = {
  operation: GPULocationAllocationProps['operation'];
  facilityCount: number;
  demandCount: number;
  maximumFacilityCount: number;
  maximumSwapIterations?: number;
  isSparse?: boolean;
  hasWeights?: boolean;
  hasCostLimit?: boolean;
  hasCostsComplete?: boolean;
};

function createContributor(graph: GPUCommandGraph, configuration: Configuration) {
  const {facilityCount, demandCount} = configuration;
  const createView = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `allocation-budget-view-${viewCount++}`, format, length);
  const entryCount = 4 * demandCount;
  return new GPULocationAllocation({
    id: 'allocation',
    operation: configuration.operation,
    facilityCount,
    maximumFacilityCount: configuration.maximumFacilityCount,
    maximumSwapIterations: configuration.maximumSwapIterations,
    coverageDistance: configuration.operation === 'p-median' ? undefined : createView('float32', 1),
    costLimit: configuration.hasCostLimit ? createView('float32', 1) : undefined,
    costsComplete: configuration.hasCostsComplete ? createView('uint32', 1) : undefined,
    demandWeights: configuration.hasWeights ? createView('float32', demandCount) : undefined,
    costs: configuration.isSparse ? undefined : createView('float32', facilityCount * demandCount),
    sparseCosts: configuration.isSparse
      ? {
          offsets: createView('uint32', demandCount + 1),
          facilities: createView('uint32', entryCount),
          costs: createView('float32', entryCount)
        }
      : undefined,
    output: {
      selectedFacilities: createView('uint32', facilityCount),
      assignments: createView('uint32', demandCount),
      assignedCosts: createView('float32', demandCount),
      objective: createView('float32', 1),
      status: createView('uint32', 7)
    }
  } as GPULocationAllocationProps);
}

it('GPULocationAllocation estimates its dense command node count exactly', () => {
  for (const configuration of [
    {operation: 'p-median', facilityCount: 10, demandCount: 100, maximumFacilityCount: 3},
    {
      operation: 'p-median',
      facilityCount: 2,
      demandCount: 2,
      maximumFacilityCount: 1,
      maximumSwapIterations: 0,
      hasCostLimit: true
    },
    {
      operation: 'maximum-coverage',
      facilityCount: 2,
      demandCount: 2,
      maximumFacilityCount: 1,
      maximumSwapIterations: 0,
      hasCostLimit: true
    },
    {
      operation: 'maximum-coverage',
      facilityCount: 2,
      demandCount: 2,
      maximumFacilityCount: 1,
      maximumSwapIterations: 0,
      hasCostLimit: true,
      hasCostsComplete: true
    },
    {
      operation: 'p-median',
      facilityCount: 257,
      demandCount: 1,
      maximumFacilityCount: 1,
      maximumSwapIterations: 2
    },
    {
      operation: 'p-median',
      facilityCount: 65537,
      demandCount: 1,
      maximumFacilityCount: 1,
      maximumSwapIterations: 2
    },
    {
      operation: 'maximum-coverage',
      facilityCount: 20,
      demandCount: 50,
      maximumFacilityCount: 5,
      maximumSwapIterations: 7,
      hasWeights: true
    },
    {
      operation: 'set-covering',
      facilityCount: 12,
      demandCount: 30,
      maximumFacilityCount: 12,
      hasWeights: true
    },
    {
      operation: 'p-median',
      facilityCount: 9,
      demandCount: 20,
      maximumFacilityCount: 2,
      maximumSwapIterations: 0
    }
  ] satisfies Configuration[]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const contributor = createContributor(graph, configuration);
    expect(contributor.estimatedCommandNodeCount, JSON.stringify(configuration)).toBe(
      contributor.getCommandNodes(graph).length
    );
  }
});

it('GPULocationAllocation bounds its sparse command node count by the transpose sort allowance', () => {
  for (const configuration of [
    {
      operation: 'p-median',
      facilityCount: 10,
      demandCount: 100,
      maximumFacilityCount: 3,
      isSparse: true
    },
    {
      operation: 'maximum-coverage',
      facilityCount: 200,
      demandCount: 2000,
      maximumFacilityCount: 10,
      maximumSwapIterations: 5,
      isSparse: true,
      hasWeights: true
    },
    {
      operation: 'set-covering',
      facilityCount: 30,
      demandCount: 50000,
      maximumFacilityCount: 6,
      isSparse: true
    }
  ] satisfies Configuration[]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const contributor = createContributor(graph, configuration);
    const nodeCount = contributor.getCommandNodes(graph).length;
    expect(
      contributor.estimatedCommandNodeCount,
      JSON.stringify(configuration)
    ).toBeGreaterThanOrEqual(nodeCount);
    expect(
      contributor.estimatedCommandNodeCount - nodeCount,
      JSON.stringify(configuration)
    ).toBeLessThanOrEqual(64);
  }
});

it('GPULocationAllocation rejects configurations above the command node budget', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  // Three nodes per greedy step: 11,000 steps emit more than 33,000 nodes.
  expect(() =>
    createContributor(graph, {
      operation: 'p-median',
      facilityCount: 11000,
      demandCount: 2,
      maximumFacilityCount: 11000,
      maximumSwapIterations: 0
    })
  ).toThrow(/^allocation estimated command node count must be at most 32768$/);
  expect(() =>
    createContributor(graph, {
      operation: 'p-median',
      facilityCount: 1000,
      demandCount: 2,
      maximumFacilityCount: 1000,
      maximumSwapIterations: 0
    })
  ).not.toThrow();
});

it('counts a final completeness node when a matrix horizon is supplied alone', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const configuration = {
      operation: 'p-median',
      facilityCount: 2,
      demandCount: 2,
      maximumFacilityCount: 1,
      maximumSwapIterations: 0
    } as const;
    const ordinary = createContributor(graph, configuration);
    const withHorizon = createContributor(graph, {...configuration, hasCostLimit: true});
    expect(withHorizon.estimatedCommandNodeCount).toBe(ordinary.estimatedCommandNodeCount + 1);
    expect(withHorizon.getCommandNodes(graph).length).toBe(withHorizon.estimatedCommandNodeCount);
  } finally {
    device.destroy();
  }
});

it('rejects a dense p-median graph over the command-node ceiling even with 512 MiB bindings', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxStorageBufferBindingSize: 512 * 1024 * 1024,
      maxBufferSize: 512 * 1024 * 1024
    }
  });
  try {
    const graph = new GPUCommandGraph(device);
    // The swap pair table fits a 512 MiB storage binding: each pair uses 8 bytes.
    expect(7800 * 8000 * 8).toBeLessThan(device.limits.maxStorageBufferBindingSize);
    // Dense base (4), p-median setup (1), 7800 selections (3 each), swap setup (1),
    // and 1000 rounds (10 each). The two scan levels have 8000 and 32 entries.
    expect(() =>
      createContributor(graph, {
        operation: 'p-median',
        facilityCount: 8000,
        demandCount: 1,
        maximumFacilityCount: 7800,
        maximumSwapIterations: 1000
      })
    ).toThrow(
      expect.objectContaining({
        message: 'allocation estimated command node count must be at most 32768',
        received: 33406,
        expected: {maximum: 32768}
      })
    );
  } finally {
    device.destroy();
  }
});
