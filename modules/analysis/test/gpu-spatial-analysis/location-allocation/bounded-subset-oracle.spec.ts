// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUBoundedFacilityAllocation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-bounded-facility-allocation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

import {BOUNDED_ALLOCATION_SCENES} from './bounded-subset-fixtures';

for (const scene of BOUNDED_ALLOCATION_SCENES) {
  it(`selects the unique best bounded ${scene.operation} subset, beyond feasibility`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const selected = rig.output('uint32', 3),
      assignments = rig.output('uint32', 3);
    const objective = rig.output('float32', 1),
      status = rig.output('uint32', 3);
    rig.graph.add(
      new GPUBoundedFacilityAllocation({
        operation: scene.operation,
        facilityCount: 3,
        selectCount: 2,
        costs: rig.input(Float32Array.from(scene.costs), 'float32', 9),
        demand: rig.input(Float32Array.from(scene.demand), 'float32', 3),
        capacities: rig.input(Float32Array.from(scene.capacities), 'float32', 3),
        output: {
          selectedFacilities: selected.view,
          assignments: assignments.view,
          objective: objective.view,
          status: status.view
        }
      })
    );
    const compiled = await rig.graph.compileAsync();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(selected.buffer, 3)).toEqual(scene.selected);
      expect(await readUint32(assignments.buffer, 3)).toEqual(scene.assignments);
      expect(await readUint32(status.buffer, 3)).toEqual(scene.status);
      const [score] = await readFloat32(objective.buffer, 1);
      expect(Number.isFinite(score)).toBe(true);
      expect(score).toBe(scene.objective);
      expect(score).toBeLessThan(scene.feasibleWorseObjective);
    } finally {
      compiled.destroy();
      rig.destroy();
      device.destroy();
    }
  });
}
