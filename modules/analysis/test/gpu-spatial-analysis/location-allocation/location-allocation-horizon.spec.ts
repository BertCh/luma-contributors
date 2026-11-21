// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView, GPUGraphVectorFormat} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULocationAllocation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation';
import {GPULocationAllocationGraphOperation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

for (const operation of ['p-median', 'maximum-coverage'] as const) {
  for (const sparse of [false, true]) {
    it(`honors a horizon without a completeness word for ${operation}, sparse ${sparse}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new WeightsRig(device);
      const selected = rig.output('uint32', 1);
      const assignments = rig.output('uint32', 2);
      const objective = rig.output('float32', 1);
      const status = rig.output('uint32', 7);
      // True costs [0,2], truncated at 1, hide demand 1 despite a requested cutoff of 4.
      // The visible objective is 1 covered demand (or zero reachable p-median cost), but this
      // cannot certify complete costs. Neither representation supplies a costsComplete word.
      rig.graph.add(
        new GPULocationAllocation({
          operation,
          facilityCount: 1,
          maximumFacilityCount: 1,
          maximumSwapIterations: 0,
          ...(sparse
            ? {
                sparseCosts: {
                  offsets: rig.input(Uint32Array.of(0, 1, 1), 'uint32', 3),
                  facilities: rig.input(Uint32Array.of(0), 'uint32', 1),
                  costs: rig.input(Float32Array.of(0), 'float32', 1)
                }
              }
            : {costs: rig.input(Float32Array.of(0, Infinity), 'float32', 2)}),
          costLimit: rig.input(Float32Array.of(1), 'float32', 1),
          ...(operation === 'maximum-coverage'
            ? {coverageDistance: rig.input(Float32Array.of(4), 'float32', 1)}
            : {}),
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
        expect(await readUint32(selected.buffer, 1)).toEqual([1]);
        expect(await readUint32(assignments.buffer, 2)).toEqual([0, 0xffffffff]);
        expect(await readFloat32(objective.buffer, 1)).toEqual([
          operation === 'maximum-coverage' ? 1 : 0
        ]);
        expect(await readUint32(status.buffer, 7)).toEqual([1, 1, 0, 0, 0, 0, 0]);
      } finally {
        compiled.destroy();
        rig.destroy();
        device.destroy();
      }
    });
  }
}

for (const named of [false, true])
  for (const withConvergence of [false, true]) {
    it(`recovers the full allocation graph without status writes, named ${named}, convergence ${withConvergence}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new WeightsRig(device);
      const costs = rig.input(Float32Array.of(0, Infinity), 'float32', 2);
      const costLimit = rig.input(Float32Array.of(1), 'float32', 1);
      const coverageDistance = rig.input(Float32Array.of(2), 'float32', 1);
      const costsComplete = rig.input(Uint32Array.of(1), 'uint32', 1);
      const selected = rig.output('uint32', 1),
        assignments = rig.output('uint32', 2),
        objective = rig.output('float32', 1),
        status = rig.output('uint32', 7);
      const options = {
        operation: 'maximum-coverage' as const,
        facilityCount: 1,
        maximumFacilityCount: 1,
        maximumSwapIterations: 0
      };
      const views = {
        costs,
        costLimit,
        coverageDistance,
        costsComplete,
        selectedFacilities: selected.view,
        assignments: assignments.view,
        objective: objective.view,
        status: status.view
      };
      const contributor = named
        ? new GPULocationAllocationGraphOperation({
            ...options,
            demandCount: 2,
            costs: 'costs',
            costLimit: 'limit',
            coverageDistance: 'cutoff',
            costsComplete: withConvergence ? 'complete' : undefined,
            output: {
              selectedFacilities: 'selected',
              assignments: 'assigned',
              objective: 'objective',
              status: 'status'
            }
          }).createContributor(
            <Format extends GPUGraphVectorFormat>(port: string) =>
              views[port as keyof typeof views] as GraphDataView<Format>
          )
        : new GPULocationAllocation({
            ...options,
            costs,
            costLimit,
            coverageDistance,
            costsComplete: withConvergence ? costsComplete : undefined,
            output: {
              selectedFacilities: selected.view,
              assignments: assignments.view,
              objective: objective.view,
              status: status.view
            }
          });
      rig.graph.add(contributor);
      const compiled = await rig.graph.compileAsync();
      try {
        // True costs [0,2] cover two unit demands at cutoff 2; horizon 1 hides the second.
        submitGraph(device, compiled, undefined);
        expect(await readFloat32(objective.buffer, 1)).toEqual([1]);
        expect((await readUint32(status.buffer, 7))[2]).toBe(0);
        // Only input costs and horizon change. A finalizer retaining last frame's zero must fail.
        rig.bufferOf(costs).write(Float32Array.of(0, 2));
        rig.bufferOf(costLimit).write(Float32Array.of(2));
        submitGraph(device, compiled, undefined);
        expect(await readFloat32(objective.buffer, 1)).toEqual([2]);
        expect(await readUint32(assignments.buffer, 2)).toEqual([0, 0]);
        expect((await readUint32(status.buffer, 7))[2]).toBe(1);
      } finally {
        compiled.destroy();
        rig.destroy();
        device.destroy();
      }
    });
  }
