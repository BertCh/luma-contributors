// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULocationAllocationGraphOperation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation-operation';
import {GPULocationAllocation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation';
import {computeLocationAllocationOnCPU} from '../../../src/gpu-spatial-analysis/location-allocation/cpu-location-allocation-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const sparse of [false, true]) {
  it(`rejects a ${sparse ? 'sparse' : 'dense'} operation that declares three rows but supplies two (B2)`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const operation = new GPULocationAllocationGraphOperation({
        id: 'review-topology',
        demandCount: 3,
        facilityCount: 2,
        operation: 'p-median',
        maximumFacilityCount: 1,
        maximumSwapIterations: 0,
        ...(sparse
          ? {sparseCosts: {offsets: 'offsets', facilities: 'facilities', costs: 'costs'}}
          : {costs: 'costs'}),
        output: {
          selectedFacilities: 'selected',
          assignments: 'assignments',
          objective: 'objective',
          status: 'status'
        }
      });
      const views: Record<string, GraphDataView> = {};
      for (const [name, output] of Object.entries(operation.outputs))
        views[name] = createTransientView(graph, name, output.format, output.length);
      if (sparse) {
        views['sparseOffsets'] = createTransientView(graph, 'offsets', 'uint32', 3);
        views['sparseFacilities'] = createTransientView(graph, 'facilities', 'uint32', 2);
        views['sparseCosts'] = createTransientView(graph, 'costs', 'float32', 2);
      } else views['costs'] = createTransientView(graph, 'costs', 'float32', 4);
      expect(() =>
        operation.createContributor(
          <Format extends GPUGraphVectorFormat>(name: string) =>
            views[name] as GraphDataView<Format>
        )
      ).toThrow(
        sparse
          ? /^review-topology sparseCosts.offsets length must equal demandCount \+ 1/
          : /^review-topology costs length must equal demandCount \* facilityCount/
      );
    } finally {
      device.destroy();
    }
  });
}

it('retains direct contributor padded-output support (B2)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const model = new GPULocationAllocation({
      costs: createTransientView(graph, 'costs', 'float32', 4),
      facilityCount: 2,
      operation: 'p-median',
      maximumFacilityCount: 1,
      maximumSwapIterations: 0,
      output: {
        selectedFacilities: createTransientView(graph, 'selected', 'uint32', 2),
        assignments: createTransientView(graph, 'assignments', 'uint32', 3),
        objective: createTransientView(graph, 'objective', 'float32', 1),
        status: createTransientView(graph, 'status', 'uint32', 7)
      }
    });
    expect(model.demandCount).toBe(2);
    expect(model.props.output.assignments.length).toBe(3);
    expect(() => model.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('keeps set-covering incomplete when an unreachable demand row remains (earlier-fix comment)', () => {
  const result = computeLocationAllocationOnCPU(
    {
      costs: Float32Array.of(1, Infinity),
      coverageDistance: Float32Array.of(1),
      costsComplete: Uint32Array.of(1)
    },
    {
      operation: 'set-covering',
      demandCount: 2,
      facilityCount: 1,
      maximumFacilityCount: 1,
      maximumSwapIterations: 0
    }
  );
  expect(result.status[1]).toBe(1);
  expect(result.status[2]).toBe(0);
});
