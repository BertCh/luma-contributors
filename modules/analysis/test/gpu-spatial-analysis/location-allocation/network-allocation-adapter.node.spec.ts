// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  GPUTranspose,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {addNetworkLocationAllocationAdapter} from '../../../src/gpu-spatial-analysis/location-allocation/network-location-allocation-adapter';
import {computeLocationAllocationOnCPU} from '../../../src/gpu-spatial-analysis/location-allocation/cpu-location-allocation-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function createReviewProps(graph: GPUCommandGraph, facilityAxis: 'rows' | 'columns' = 'columns') {
  const floats = (name: string, length: number) =>
    createTransientView(graph, name, 'float32', length);
  const integers = (name: string, length: number) =>
    createTransientView(graph, name, 'uint32', length);
  return {
    id: 'review-allocation',
    facilityAxis,
    trips: 'demand-to-facility' as const,
    matrix: {
      values: floats('review-costs', 6),
      rowCount: facilityAxis === 'rows' ? 2 : 3,
      columnCount: facilityAxis === 'rows' ? 3 : 2,
      startsPerRow: 1,
      startNodes: integers('review-starts', facilityAxis === 'rows' ? 2 : 3),
      targetNodes: integers('review-targets', facilityAxis === 'rows' ? 3 : 2),
      direction:
        facilityAxis === 'rows' ? ('target-to-start' as const) : ('start-to-target' as const),
      converged: integers('review-converged', 1),
      sourceComplete: integers('review-source', 1)
    },
    operation: 'maximum-coverage' as const,
    coverageDistance: floats('review-coverage', 1),
    maximumSwapIterations: 0,
    maximumFacilityCount: 1,
    output: {
      selectedFacilities: integers('review-selected', 2),
      assignments: integers('review-assignments', 3),
      objective: floats('review-objective', 1),
      status: integers('review-status', 7)
    }
  };
}

it.each([
  'rows',
  'columns'
] as const)('adds %s with source completeness through the real graph path', facilityAxis => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const add = vi.spyOn(graph, 'add');
  try {
    const props = createReviewProps(graph, facilityAxis);
    const {contributor} = addNetworkLocationAllocationAdapter(graph, props);
    const combined = add.mock.calls.find(
      ([primitive]) =>
        (primitive as {id?: string}).id === 'review-allocation-combine-costs-complete'
    )?.[0];
    expect(combined).toBeDefined();
    if (!combined) throw new Error('Missing combined evidence node');
    const memory = createCPUGraphMemory();
    memory.write(props.matrix.converged, Uint32Array.of(1));
    for (const complete of [0, 1]) {
      memory.write(props.matrix.sourceComplete, Uint32Array.of(complete));
      executeCommandNodesOnCPU([combined as GPUCommandNode], memory);
      expect(memory.readUint32(contributor.props.costsComplete!)[0]).toBe(complete);
    }
  } finally {
    device.destroy();
  }
});

it('can retry a rejected facility-row output shape without leaked adapter scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createReviewProps(graph, 'rows');
    const shortAssignments = createTransientView(graph, 'short-assignments', 'uint32', 2);
    expect(() =>
      addNetworkLocationAllocationAdapter(graph, {
        ...props,
        matrix: {...props.matrix, sourceComplete: undefined},
        output: {...props.output, assignments: shortAssignments}
      })
    ).toThrow(/assignments/);
    expect(() =>
      addNetworkLocationAllocationAdapter(graph, {
        ...props,
        matrix: {...props.matrix, sourceComplete: undefined}
      })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('rejects an unknown allocation operation even when a coverage view is supplied', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createReviewProps(graph);
    expect(() =>
      addNetworkLocationAllocationAdapter(graph, {
        ...props,
        matrix: {...props.matrix, sourceComplete: undefined},
        operation: 'typo' as 'maximum-coverage'
      })
    ).toThrow(/operation.*p-median.*maximum-coverage.*set-covering/);
  } finally {
    device.destroy();
  }
});

it('pins the directed demand-major expectation independently of square-matrix shape', () => {
  // Facility 0: 3+1=4; facility 1: 10+1=11. An unlabelled transpose gives 13 versus 2.
  const options = {
    facilityCount: 2,
    demandCount: 2,
    maximumFacilityCount: 1,
    maximumSwapIterations: 0,
    operation: 'p-median' as const
  };
  expect(
    Array.from(
      computeLocationAllocationOnCPU({costs: Float32Array.of(3, 10, 1, 1)}, options).objective
    )
  ).toEqual([4]);
  expect(
    Array.from(
      computeLocationAllocationOnCPU({costs: Float32Array.of(3, 1, 10, 1)}, options).objective
    )
  ).toEqual([2]);
});

it.each([
  'rows',
  'columns'
] as const)('preserves labelled %s, trips, target mapping and matrix evidence', facilityAxis => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const add = vi.spyOn(graph, 'add');
  const floats = (name: string, length: number) =>
    createTransientView(graph, name, 'float32', length);
  const integers = (name: string, length: number) =>
    createTransientView(graph, name, 'uint32', length);
  try {
    const matrix = {
      values: floats('costs', 4),
      rowCount: 2,
      columnCount: 2,
      startNodes: integers('starts', 2),
      targetNodes: integers('targets', 2),
      startsPerRow: 1,
      direction:
        facilityAxis === 'columns' ? ('start-to-target' as const) : ('target-to-start' as const),
      converged: integers('converged', 1),
      costLimit: floats('horizon', 1)
    };
    const props = {
      matrix,
      facilityAxis,
      trips: 'demand-to-facility' as const,
      maximumFacilityCount: 1,
      maximumSwapIterations: 0,
      operation: 'p-median' as const,
      assignmentNodes: integers('assignment-nodes', 2),
      output: {
        selectedFacilities: integers('selected', 2),
        assignments: integers('assignments', 2),
        objective: floats('objective', 1),
        status: integers('status', 7)
      }
    };
    const result = addNetworkLocationAllocationAdapter(graph, props);
    expect(result.facilityNodes).toBe(
      facilityAxis === 'columns' ? matrix.targetNodes : matrix.startNodes
    );
    expect(result.contributor.props.costsComplete).toBe(matrix.converged);
    expect(result.contributor.props.costLimit).toBe(matrix.costLimit);
    expect(add.mock.calls.some(([node]) => node instanceof GPUTranspose)).toBe(
      facilityAxis === 'rows'
    );
    expect(() =>
      addNetworkLocationAllocationAdapter(graph, {...props, trips: 'facility-to-demand'})
    ).toThrow(/trips must match/);
    expect(() =>
      addNetworkLocationAllocationAdapter(graph, {
        ...props,
        matrix: {...matrix, converged: undefined}
      })
    ).toThrow(/converged is required/);
  } finally {
    device.destroy();
  }
});
