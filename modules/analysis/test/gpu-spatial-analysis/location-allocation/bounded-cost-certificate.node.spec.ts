// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUBoundedFacilityAllocation,
  decodeGPUBoundedFacilityAllocationStatus
} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-bounded-facility-allocation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('separates bounded enumeration from complete travel costs and recovers', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const costsComplete = createTransientView(graph, 'costs-complete', 'uint32', 1);
  const costLimit = createTransientView(graph, 'cost-limit', 'float32', 1);
  const complete = createTransientView(graph, 'complete', 'uint32', 1);
  const status = createTransientView(graph, 'status', 'uint32', 3);
  try {
    const model = new GPUBoundedFacilityAllocation({
      operation: 'p-center',
      facilityCount: 1,
      selectCount: 1,
      costs: createTransientView(graph, 'costs', 'float32', 1),
      demand: createTransientView(graph, 'demand', 'float32', 1),
      capacities: createTransientView(graph, 'capacities', 'float32', 1),
      costsComplete,
      costLimit,
      output: {
        selectedFacilities: createTransientView(graph, 'selected', 'uint32', 1),
        assignments: createTransientView(graph, 'assignments', 'uint32', 1),
        objective: createTransientView(graph, 'objective', 'float32', 1),
        status,
        ports: {status: {complete}}
      }
    });
    expect(model.status.complete).toBe(complete);
    expect(
      () =>
        new GPUBoundedFacilityAllocation({
          ...model.props,
          output: {
            ...model.props.output,
            // @ts-expect-error The direct scalar shorthand has been retired.
            complete
          }
        })
    ).toThrow(
      /^bounded-facility-allocation output.complete was removed; use output.ports.status.complete$/
    );
    const nodes = model
      .getCommandNodes(graph)
      .filter(node => node.id.endsWith('-cost-certificate'));
    expect(nodes).toHaveLength(1);
    // One enumerated subset is locally valid even if its sole cost is provisional infinity.
    // No finite horizon can prove an unbounded p-center optimum. Full costs at horizon infinity
    // recover the decision certificate without changing the local diagnostics.
    for (const [converged, horizon, expected] of [
      [0, Infinity, 0],
      [1, 1, 0],
      [1, Infinity, 1]
    ]) {
      memory.write(status, Uint32Array.of(1, 1, 1));
      memory.write(costsComplete, Uint32Array.of(converged));
      memory.write(costLimit, Float32Array.of(horizon));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(status))).toEqual([1, 1, expected]);
      expect(Array.from(memory.readUint32(complete))).toEqual([expected]);
    }
  } finally {
    device.destroy();
  }
});

it('decodes bounded diagnostics and decision evidence from bytes and refuses missing words', () => {
  const storage = new Uint32Array([99, 300, 1, 0, 99]);
  const bytes = new Uint8Array(storage.buffer, 4, 12);
  expect(decodeGPUBoundedFacilityAllocationStatus(bytes, Uint32Array.of(0))).toEqual({
    unassignedDemandCount: 300,
    validInput: 1,
    globalOptimum: 0,
    complete: 0
  });
  expect(() =>
    decodeGPUBoundedFacilityAllocationStatus(bytes.subarray(0, 8), Uint32Array.of(1))
  ).toThrow(/3/);
  expect(() => decodeGPUBoundedFacilityAllocationStatus(bytes, new Uint32Array())).toThrow(/1/);
  expect(() =>
    decodeGPUBoundedFacilityAllocationStatus(bytes.subarray(1), Uint32Array.of(1))
  ).toThrow(/whole/);
});
