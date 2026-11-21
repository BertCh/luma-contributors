// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {computeLocationAllocationOnCPU} from '../../../src/gpu-spatial-analysis/location-allocation/cpu-location-allocation-operation';
import {GPULocationAllocation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation';
import {GPULocationAllocationGraphOperation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

for (const convergence of [undefined, 1]) {
  it(`checks the static horizon gate independently of convergence ${convergence} in CPU, direct and named forms`, () => {
    const options = {
      facilityCount: 1,
      demandCount: 2,
      maximumFacilityCount: 1,
      maximumSwapIterations: 0,
      operation: 'maximum-coverage' as const
    };
    const costsComplete = convergence === undefined ? undefined : Uint32Array.of(convergence);
    // True costs are [0,2]. At horizon 1 only one demand is visible; cutoff 2 promises both.
    const short = computeLocationAllocationOnCPU(
      {
        costs: Float32Array.of(0, Infinity),
        coverageDistance: Float32Array.of(2),
        costLimit: Float32Array.of(1),
        costsComplete
      },
      options
    );
    expect(Array.from(short.objective)).toEqual([1]);
    expect(short.status[2]).toBe(0);
    const full = computeLocationAllocationOnCPU(
      {
        costs: Float32Array.of(0, 2),
        coverageDistance: Float32Array.of(2),
        costLimit: Float32Array.of(2),
        costsComplete
      },
      options
    );
    expect(Array.from(full.objective)).toEqual([2]);
    expect(full.status[2]).toBe(1);
    for (const named of [false, true]) {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const memory = createCPUGraphMemory();
      const views = {
        costs: createTransientView(graph, 'costs', 'float32', 2),
        coverageDistance: createTransientView(graph, 'coverage', 'float32', 1),
        costLimit: createTransientView(graph, 'horizon', 'float32', 1),
        costsComplete: createTransientView(graph, 'convergence', 'uint32', 1),
        selectedFacilities: createTransientView(graph, 'selected', 'uint32', 1),
        assignments: createTransientView(graph, 'assignments', 'uint32', 2),
        objective: createTransientView(graph, 'objective', 'float32', 1),
        status: createTransientView(graph, 'status', 'uint32', 7)
      };
      try {
        const contributor = named
          ? (new GPULocationAllocationGraphOperation({
              ...options,
              costs: 'costs',
              coverageDistance: 'coverage',
              costLimit: 'horizon',
              costsComplete: convergence === undefined ? undefined : 'convergence',
              output: {
                selectedFacilities: 'selected',
                assignments: 'assignments',
                objective: 'objective',
                status: 'status'
              }
            }).createContributor(
              <Format extends GPUGraphVectorFormat>(port: string) =>
                views[port as keyof typeof views] as GraphDataView<Format>
            ) as GPULocationAllocation)
          : new GPULocationAllocation({
              ...options,
              costs: views.costs,
              coverageDistance: views.coverageDistance,
              costLimit: views.costLimit,
              costsComplete: convergence === undefined ? undefined : views.costsComplete,
              output: {
                selectedFacilities: views.selectedFacilities,
                assignments: views.assignments,
                objective: views.objective,
                status: views.status
              }
            });
        const nodes = contributor
          .getCommandNodes(graph)
          .filter(node => node.id.endsWith('-costs-complete'));
        expect(nodes).toHaveLength(1);
        memory.write(views.coverageDistance, Float32Array.of(2));
        memory.write(views.costsComplete, Uint32Array.of(1));
        for (const [horizon, expected] of [
          [1, 0],
          [2, 1]
        ]) {
          // Unit test only: seed the input to this clearing gate. Full-graph recovery without
          // any host status write is pinned in location-allocation-horizon.spec.ts.
          memory.write(views.status, Uint32Array.of(1, 0, 1, 0, 0, 0, 0));
          memory.write(views.costLimit, Float32Array.of(horizon));
          executeCommandNodesOnCPU(nodes, memory);
          expect(memory.readUint32(views.status)[2]).toBe(expected);
        }
      } finally {
        device.destroy();
      }
    }
  });
}
