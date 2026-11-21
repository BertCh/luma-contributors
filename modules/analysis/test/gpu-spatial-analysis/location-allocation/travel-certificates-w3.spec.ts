// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUBoundedFacilityAllocation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-bounded-facility-allocation';
import {addNetworkLocationAllocationAdapter} from '../../../src/gpu-spatial-analysis/location-allocation/network-location-allocation-adapter';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

it('bounded allocation refuses provisional or finite-horizon optimum and recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const costs = rig.input(Float32Array.of(Infinity), 'float32', 1);
  const costsComplete = rig.input(Uint32Array.of(0), 'uint32', 1);
  const costLimit = rig.input(Float32Array.of(1), 'float32', 1);
  const status = rig.output('uint32', 3),
    complete = rig.output('uint32', 1),
    objective = rig.output('float32', 1);
  rig.graph.add(
    new GPUBoundedFacilityAllocation({
      operation: 'p-center',
      facilityCount: 1,
      selectCount: 1,
      costs,
      costsComplete,
      costLimit,
      demand: rig.input(Float32Array.of(1), 'float32', 1),
      capacities: rig.input(Float32Array.of(1), 'float32', 1),
      output: {
        selectedFacilities: rig.output('uint32', 1).view,
        assignments: rig.output('uint32', 1).view,
        objective: objective.view,
        status: status.view,
        ports: {status: {complete: complete.view}}
      }
    })
  );
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(status.buffer, 3)).toEqual([1, 1, 0]);
    expect(await readUint32(complete.buffer, 1)).toEqual([0]);
    rig.bufferOf(costs).write(Float32Array.of(2));
    rig.bufferOf(costsComplete).write(Uint32Array.of(1));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(status.buffer, 3)).toEqual([0, 1, 0]);
    expect(await readUint32(complete.buffer, 1)).toEqual([0]);
    rig.bufferOf(costLimit).write(Float32Array.of(Infinity));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(status.buffer, 3)).toEqual([0, 1, 1]);
    expect(await readUint32(complete.buffer, 1)).toEqual([1]);
    expect(await readFloat32(objective.buffer, 1)).toEqual([2]);
  } finally {
    compiled.destroy();
    rig.destroy();
    device.destroy();
  }
});

for (const facilityAxis of ['columns', 'rows'] as const) {
  it(`labelled ${facilityAxis} preserve directed allocation and target-list assignments`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const selected = rig.output('uint32', 2),
      assignments = rig.output('uint32', 2),
      assignmentNodes = rig.output('uint32', 2),
      objective = rig.output('float32', 1),
      status = rig.output('uint32', 7);
    const sourceComplete = rig.input(Uint32Array.of(0), 'uint32', 1);
    // Directed trips: d0->f0=3, d0->f1=10, d1->f0=1, d1->f1=1.
    // Facility zero costs 3+1=4 versus facility one's 10+1=11.
    addNetworkLocationAllocationAdapter(rig.graph, {
      facilityAxis,
      trips: 'demand-to-facility',
      matrix: {
        values: rig.input(
          Float32Array.from(facilityAxis === 'columns' ? [3, 10, 1, 1] : [3, 1, 10, 1]),
          'float32',
          4
        ),
        rowCount: 2,
        columnCount: 2,
        startNodes: rig.input(
          Uint32Array.from(facilityAxis === 'columns' ? [0, 1] : [8, 13]),
          'uint32',
          2
        ),
        targetNodes: rig.input(
          Uint32Array.from(facilityAxis === 'columns' ? [8, 13] : [0, 1]),
          'uint32',
          2
        ),
        startsPerRow: 1,
        direction: facilityAxis === 'columns' ? 'start-to-target' : 'target-to-start',
        converged: rig.input(Uint32Array.of(1), 'uint32', 1),
        sourceComplete,
        costLimit: rig.input(Float32Array.of(Infinity), 'float32', 1)
      },
      operation: 'p-median',
      maximumFacilityCount: 1,
      maximumSwapIterations: 0,
      assignmentNodes: assignmentNodes.view,
      output: {
        selectedFacilities: selected.view,
        assignments: assignments.view,
        objective: objective.view,
        status: status.view
      }
    });
    const compiled = await rig.graph.compileAsync();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(selected.buffer, 2)).toEqual([1, 0]);
      expect(await readUint32(assignments.buffer, 2)).toEqual([0, 0]);
      expect(await readUint32(assignmentNodes.buffer, 2)).toEqual([8, 8]);
      expect(await readFloat32(objective.buffer, 1)).toEqual([4]);
      expect((await readUint32(status.buffer, 7))[2]).toBe(0);
      rig.bufferOf(sourceComplete).write(Uint32Array.of(1));
      submitGraph(device, compiled, undefined);
      expect(await readFloat32(objective.buffer, 1)).toEqual([4]);
      expect((await readUint32(status.buffer, 7))[2]).toBe(1);
    } finally {
      compiled.destroy();
      rig.destroy();
      device.destroy();
    }
  });
}
