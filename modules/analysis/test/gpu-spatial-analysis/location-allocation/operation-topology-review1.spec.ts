// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULocationAllocationGraphOperation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';

for (const sparse of [false, true])
  it(`rejects mismatched ${sparse ? 'sparse' : 'dense'} topology during GPU operation lowering (B2)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const inputs: Record<string, OperationGraphInput> = sparse
      ? {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
          facilities: {format: 'uint32', values: Uint32Array.of(0, 1)},
          costs: {format: 'float32', values: Float32Array.of(1, 2)}
        }
      : {costs: {format: 'float32', values: Float32Array.of(1, 2, 3, 4)}};
    try {
      const operation = new GPULocationAllocationGraphOperation({
        id: 'gpu-review-topology',
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
      await expect(
        runOperationGraph(device, {inputs, operations: [operation], outputs: ['assignments']})
      ).rejects.toThrow(
        sparse
          ? /sparseCosts.offsets length must equal demandCount \+ 1/
          : /costs length must equal demandCount \* facilityCount/
      );
    } finally {
      device.destroy();
    }
  });
