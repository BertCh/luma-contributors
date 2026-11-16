// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {createFacilityScoreReductionNodes} from '../../../src/gpu-spatial-analysis/location-allocation/facility-score-reduction';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {GPUBoundedFacilityAllocation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-bounded-facility-allocation';
import {GPULocationAllocation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('rejects invalid bounded operations and checks aligned costs before scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const costStorage = createTransientView(graph, 'cost-storage', 'float32', 20);
    const props = {
      operation: 'p-center' as const,
      facilityCount: 2,
      selectCount: 1,
      costs: graph.createDataView(costStorage.buffer, {
        format: 'float32' as const,
        length: 4,
        byteOffset: 64
      }),
      demand: createTransientView(graph, 'demand', 'float32', 2),
      capacities: createTransientView(graph, 'capacities', 'float32', 2),
      output: {
        selectedFacilities: createTransientView(graph, 'selected', 'uint32', 2),
        assignments: createTransientView(graph, 'assignments', 'uint32', 2),
        objective: createTransientView(graph, 'objective', 'float32', 1),
        status: createTransientView(graph, 'status', 'uint32', 3)
      }
    };
    expect(
      () => new GPUBoundedFacilityAllocation({...props, operation: 'wrong' as 'p-center'})
    ).toThrow(/bounded-facility-allocation.*operation/);
    const contributor = new GPUBoundedFacilityAllocation(props);
    Object.assign(device.limits, {maxStorageBufferBindingSize: 64});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/costs.*80.*64/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('rejects overlapping location allocation outputs even when separate passes would accept them', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const output = createTransientView(graph, 'combined', 'uint32', 16);
    expect(
      () =>
        new GPULocationAllocation({
          operation: 'p-median',
          facilityCount: 2,
          maximumFacilityCount: 1,
          maximumSwapIterations: 0,
          costs: createTransientView(graph, 'costs', 'float32', 4),
          output: {
            selectedFacilities: graph.createDataView(output.buffer, {format: 'uint32', length: 2}),
            assignments: graph.createDataView(output.buffer, {format: 'uint32', length: 2}),
            objective: createTransientView(graph, 'objective', 'float32', 1),
            status: createTransientView(graph, 'status', 'uint32', 7)
          }
        })
    ).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});

it('shares one reduction shader and pipeline across levels and dispatches each level at its own size', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const candidates of [65, 4097, 262145]) {
      const graph = new GPUCommandGraph(device);
      const storage = createTransientView(graph, 'scores', 'uint32', candidates * 4 + 16);
      const scores = graph.createDataView(storage.buffer, {
        format: 'uint32',
        length: candidates * 4,
        byteOffset: 64
      });
      const {nodes} = createFacilityScoreReductionNodes(graph, {
        id: 'reduce',
        operation: 'FacilityScoreTest',
        scores,
        candidateCount: candidates
      });
      const reductions = nodes.filter(node => node.workload?.variant === 'candidate-reduction');
      expect(getGPUCommandNodePipelineKeys(reductions, {includeShaderLayout: true}).size).toBe(1);
      expect(
        new Set(getGPUCommandNodeKernelSources(reductions).map(kernel => kernel.source)).size
      ).toBe(1);
      let remaining = candidates;
      for (const node of reductions) {
        remaining = Math.ceil(remaining / 64);
        expect(node.workload?.maximumWorkgroupCount).toBe(remaining);
        expect(node.workload?.maximumInvocationCount).toBe(remaining * 64);
      }
      for (const kernel of getGPUCommandNodeKernelSources(reductions))
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

it('preflights nested swap dispatch storage before declaring location allocation scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const contributor = new GPULocationAllocation({
      operation: 'p-median',
      facilityCount: 2,
      maximumFacilityCount: 1,
      costs: createTransientView(graph, 'small-costs', 'float32', 4),
      output: {
        selectedFacilities: createTransientView(graph, 'small-selected', 'uint32', 2),
        assignments: createTransientView(graph, 'small-assignments', 'uint32', 2),
        objective: createTransientView(graph, 'small-objective', 'float32', 1),
        status: createTransientView(graph, 'small-status', 'uint32', 7)
      }
    });
    Object.assign(device.limits, {maxStorageBufferBindingSize: 48});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/swap dispatch.*60.*48/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('accepts a valid small sparse layout without preflighting an unused radix histogram', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const contributor = new GPULocationAllocation({
      operation: 'p-median',
      facilityCount: 2,
      maximumFacilityCount: 1,
      maximumSwapIterations: 0,
      sparseCosts: {
        offsets: createTransientView(graph, 'sparse-offsets', 'uint32', 3),
        facilities: createTransientView(graph, 'sparse-facilities', 'uint32', 4),
        costs: createTransientView(graph, 'sparse-costs', 'float32', 4)
      },
      output: {
        selectedFacilities: createTransientView(graph, 'sparse-selected', 'uint32', 2),
        assignments: createTransientView(graph, 'sparse-assignments', 'uint32', 2),
        objective: createTransientView(graph, 'sparse-objective', 'float32', 1),
        status: createTransientView(graph, 'sparse-status', 'uint32', 7)
      }
    });
    Object.assign(device.limits, {maxStorageBufferBindingSize: 32});
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});
