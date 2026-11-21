// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULocationAllocation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {FACILITY_SWAP_TIE_SCENES} from './facility-swap-tie-fixtures';

for (const scene of FACILITY_SWAP_TIE_SCENES) {
  for (const sparse of [false, true]) {
    it(`preserves ${scene.name} (${sparse ? 'sparse' : 'dense'})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new WeightsRig(device);
      const costs = rig.input(Float32Array.from(scene.costs), 'float32', scene.costs.length);
      const selected = rig.output('uint32', scene.facilityCount),
        assignments = rig.output('uint32', scene.demandCount),
        objective = rig.output('float32', 1),
        status = rig.output('uint32', 7);
      rig.graph.add(
        new GPULocationAllocation({
          operation: 'p-median',
          facilityCount: scene.facilityCount,
          maximumFacilityCount: scene.maximumFacilityCount,
          maximumSwapIterations: scene.maximumSwapIterations,
          ...(sparse
            ? {
                sparseCosts: {
                  costs,
                  offsets: rig.input(
                    Uint32Array.from(
                      {length: scene.demandCount + 1},
                      (_, row) => row * scene.facilityCount
                    ),
                    'uint32',
                    scene.demandCount + 1
                  ),
                  facilities: rig.input(
                    Uint32Array.from(
                      {length: scene.costs.length},
                      (_, index) => index % scene.facilityCount
                    ),
                    'uint32',
                    scene.costs.length
                  )
                }
              }
            : {costs}),
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
        expect(await readUint32(selected.buffer, scene.facilityCount)).toEqual(scene.selected);
        expect(await readUint32(assignments.buffer, scene.demandCount)).toEqual(scene.assignments);
        expect(await readFloat32(objective.buffer, 1)).toEqual([scene.objective]);
        expect(await readUint32(status.buffer, 7)).toEqual(scene.status);
      } finally {
        compiled.destroy();
        rig.destroy();
        device.destroy();
      }
    });
  }
}
