// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPULocationAllocationOperation} from '../../../src/gpu-spatial-analysis/location-allocation/cpu-location-allocation-operation';
import {
  GPULocationAllocationGraphOperation,
  type LocationAllocationOperationProps
} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation-operation';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const OUTPUT = {
  selectedFacilities: 'selected',
  assignments: 'assignments',
  assignedCosts: 'assignedCosts',
  objective: 'objective',
  status: 'status'
};
const OUTPUTS = Object.values(OUTPUT);
const SCENES = [
  {
    operation: 'p-median' as const,
    costs: [5, 0, 100, 5, 1, 100, 5, 100, 0, 5, 100, 1],
    weights: [1, 2, 3, 4],
    budget: 2
  },
  {
    operation: 'maximum-coverage' as const,
    costs: [0, 0, 10, 0, 0, 10, 0, 10, 0, 0, 10, 0, 10, 0, 10, 10, 10, 0],
    weights: [1, 2, 1, 2, 1, 2],
    budget: 2
  },
  {
    operation: 'set-covering' as const,
    costs: [0, 0, 10, 0, 0, 10, 0, 10, 0, 0, 10, 0, 10, 0, 10, 10, 10, 0],
    weights: [0, 0, 0, 0, 0, 0],
    budget: 3
  },
  {
    operation: 'p-median' as const,
    costs: [Infinity, Infinity, Infinity, 9, 1, 2, 9, 1, 2],
    weights: [4, 2, 3],
    budget: 1
  },
  {operation: 'p-median' as const, costs: [1, NaN, 2, 3, 4, 5], weights: [1, 2], budget: 1},
  {operation: 'maximum-coverage' as const, costs: [4, 5, 6, 7, 8, 9], weights: [1, 2], budget: 2}
];
for (const [index, scene] of SCENES.entries())
  for (const sparse of [false, true])
    it(`allocation agrees on every output: scene ${index}, sparse ${sparse}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const demandCount = scene.weights.length;
      const props: LocationAllocationOperationProps = {
        demandCount,
        facilityCount: 3,
        maximumFacilityCount: scene.budget,
        maximumSwapIterations: scene.operation === 'set-covering' ? 0 : 10,
        operation: scene.operation,
        demandWeights: 'demandWeights',
        coverageDistance: scene.operation === 'p-median' ? undefined : 'coverageDistance',
        ...(sparse
          ? {sparseCosts: {offsets: 'offsets', facilities: 'facilities', costs: 'costs'}}
          : {costs: 'costs'}),
        output: OUTPUT
      };
      const inputs: Record<string, OperationGraphInput> = {
        costs: {format: 'float32', values: Float32Array.from(scene.costs)},
        demandWeights: {format: 'float32', values: Float32Array.from(scene.weights)},
        coverageDistance: {format: 'float32', values: Float32Array.of(0)}
      };
      if (sparse) {
        inputs['offsets'] = {
          format: 'uint32',
          values: Uint32Array.from({length: demandCount + 1}, (_, row) => row * 3)
        };
        inputs['facilities'] = {
          format: 'uint32',
          values: Uint32Array.from({length: demandCount * 3}, (_, slot) => slot % 3)
        };
      }
      const gpu = await runOperationGraph(device, {
        inputs,
        operations: [new GPULocationAllocationGraphOperation(props)],
        outputs: OUTPUTS
      });
      const cpu = await runOperationGraph(device, {
        inputs,
        operations: [new CPULocationAllocationOperation(props)],
        outputs: OUTPUTS
      });
      // Selected rows, assignments and all seven status words are exact. The small integer costs
      // and weights make the float outputs exact too (zero tolerance). Numeric equality permits -0.
      for (const output of OUTPUTS) {
        expect(gpu[output].length).toBe(cpu[output].length);
        for (let slot = 0; slot < cpu[output].length; slot++)
          expect(gpu[output][slot] === cpu[output][slot], `${output}[${slot}]`).toBe(true);
      }
    });
