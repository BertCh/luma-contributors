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

for (const sparse of [false, true])
  for (const scene of [
    {
      name: 'finite greedy totals',
      costs: [7, 3, 6, 7, 5, 3],
      facilityCount: 3,
      demandCount: 2,
      scales: [1, 2 ** 125],
      swaps: 0,
      selected: [0, 1, 1],
      assignments: [1, 2],
      objective: 6
    },
    {
      name: 'finite swap tolerance',
      costs: [5, 13, 14, 4, 4, 8, 4, 5, 13, 15, 5, 2],
      facilityCount: 3,
      demandCount: 4,
      scales: [1, 2 ** 124],
      swaps: 1,
      selected: [1, 0, 1],
      assignments: [0, 0, 0, 2],
      objective: 15
    },
    {
      name: 'explicit one-swap budget',
      costs: [1, 7, 2, 0, 2, 1, 4, 8, 9, 5, 1, 0],
      facilityCount: 4,
      demandCount: 3,
      scales: [1, 1e30],
      swaps: 1,
      selected: [1, 0, 0, 1],
      assignments: [3, 0, 3],
      objective: 2
    }
  ])
    for (const scale of scene.scales)
      it(`review regression ${scene.name}, scale ${scale}, sparse ${sparse}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const props: LocationAllocationOperationProps = {
          operation: 'p-median',
          facilityCount: scene.facilityCount,
          demandCount: scene.demandCount,
          maximumFacilityCount: 2,
          maximumSwapIterations: scene.swaps,
          output: OUTPUT,
          ...(sparse
            ? {sparseCosts: {offsets: 'offsets', facilities: 'facilities', costs: 'costs'}}
            : {costs: 'costs'})
        };
        const inputs: Record<string, OperationGraphInput> = {
          costs: {
            format: 'float32',
            values: Float32Array.from(scene.costs, value => value * scale)
          },
          offsets: {
            format: 'uint32',
            values: Uint32Array.from(
              {length: scene.demandCount + 1},
              (_, row) => row * scene.facilityCount
            )
          },
          facilities: {
            format: 'uint32',
            values: Uint32Array.from(
              {length: scene.costs.length},
              (_, slot) => slot % scene.facilityCount
            )
          }
        };
        for (const Operation of [
          GPULocationAllocationGraphOperation,
          CPULocationAllocationOperation
        ]) {
          const result = await runOperationGraph(device, {
            inputs,
            operations: [new Operation(props)],
            outputs: OUTPUTS
          });
          expect(Array.from(result['selected'])).toEqual(scene.selected);
          expect(Array.from(result['assignments'])).toEqual(scene.assignments);
          expect(Array.from(result['assignedCosts'])).toEqual(
            Array.from(
              Float32Array.from(
                scene.assignments,
                (facility, demand) => scene.costs[demand * scene.facilityCount + facility] * scale
              )
            )
          );
          expect(Number.isFinite(result['objective'][0])).toBe(true);
          expect(result['objective'][0] / Math.fround(scene.objective * scale)).toBeCloseTo(1, 6);
          expect(Array.from(result['status'].slice(4))).toEqual(
            scene.swaps ? [1, 0, 1] : [0, 0, 0]
          );
        }
      });

for (const sparse of [false, true])
  for (const operation of ['p-median', 'maximum-coverage'] as const)
    for (const unusedCost of operation === 'p-median' ? [0, 2 ** 125] : [0])
      it(`weighted swap regression: ${operation}, sparse ${sparse}, unused cost ${unusedCost}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const scale = operation === 'p-median' ? 2 ** -40 : 2 ** 125;
        const costs =
          operation === 'p-median'
            ? Float32Array.from([
                2 ** 125,
                2 ** 125,
                2 ** 125,
                unusedCost,
                0,
                0,
                ...[5, 13, 14, 4, 4, 8, 4, 5, 13, 15, 5, 2].map(value => value * scale)
              ])
            : Float32Array.of(0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1);
        const weights =
          operation === 'p-median'
            ? Float32Array.of(0, 2 ** 125, 1, 1, 1, 1)
            : Float32Array.of(scale, scale, scale, scale, 2 ** 127);
        const demandCount = weights.length;
        const props: LocationAllocationOperationProps = {
          operation,
          facilityCount: 3,
          demandCount,
          maximumFacilityCount: 2,
          maximumSwapIterations: 1,
          demandWeights: 'weights',
          coverageDistance: operation === 'p-median' ? undefined : 'threshold',
          output: OUTPUT,
          ...(sparse
            ? {sparseCosts: {offsets: 'offsets', facilities: 'facilities', costs: 'costs'}}
            : {costs: 'costs'})
        };
        const inputs: Record<string, OperationGraphInput> = {
          costs: {format: 'float32', values: costs},
          weights: {format: 'float32', values: weights},
          threshold: {format: 'float32', values: Float32Array.of(0)},
          offsets: {
            format: 'uint32',
            values: Uint32Array.from({length: demandCount + 1}, (_, row) => row * 3)
          },
          facilities: {
            format: 'uint32',
            values: Uint32Array.from({length: costs.length}, (_, slot) => slot % 3)
          }
        };
        // The H4 variant makes facility 0 costly for row 1; facility 2 still costs zero.
        const assignments =
          operation === 'p-median' ? [0, unusedCost === 0 ? 0 : 2, 0, 0, 0, 2] : [1, 2, 1, 2, 1];
        for (const Operation of [
          GPULocationAllocationGraphOperation,
          CPULocationAllocationOperation
        ]) {
          const result = await runOperationGraph(device, {
            inputs,
            operations: [new Operation(props)],
            outputs: OUTPUTS
          });
          expect(Array.from(result['selected'])).toEqual(
            operation === 'p-median' ? [1, 0, 1] : [0, 1, 1]
          );
          expect(Array.from(result['assignments'])).toEqual(assignments);
          expect(Array.from(result['assignedCosts'])).toEqual(
            assignments.map((facility, row) => costs[row * 3 + facility])
          );
          expect(Number.isFinite(result['objective'][0])).toBe(true);
          expect(result['objective'][0]).toBe((operation === 'p-median' ? 15 : 4) * scale);
          expect(Array.from(result['status'].slice(4))).toEqual([1, 0, 1]);
        }
      });

for (const sparse of [false, true])
  for (const operation of ['p-median', 'set-covering'] as const)
    it(`preserves finite reachability and missing rows: ${operation}, sparse ${sparse}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const bits = new Uint32Array(Float32Array.of(1e30).buffer)[0];
      const costs =
        operation === 'p-median'
          ? new Float32Array(Uint32Array.of(bits - 1, bits, bits + 1, 0x7f800000).buffer)
          : Float32Array.of(1, 1, 1, Infinity);
      const props: LocationAllocationOperationProps = {
        demandCount: 4,
        facilityCount: 1,
        maximumFacilityCount: 1,
        maximumSwapIterations: 0,
        operation,
        coverageDistance: operation === 'set-covering' ? 'coverageDistance' : undefined,
        ...(sparse
          ? {sparseCosts: {offsets: 'offsets', facilities: 'facilities', costs: 'costs'}}
          : {costs: 'costs'}),
        output: OUTPUT
      };
      const inputs: Record<string, OperationGraphInput> = {
        costs: {format: 'float32', values: costs},
        coverageDistance: {format: 'float32', values: Float32Array.of(1e30)},
        offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 3, 4)},
        facilities: {format: 'uint32', values: new Uint32Array(4)}
      };
      for (const Operation of [
        GPULocationAllocationGraphOperation,
        CPULocationAllocationOperation
      ]) {
        const result = await runOperationGraph(device, {
          inputs,
          operations: [new Operation(props)],
          outputs: OUTPUTS
        });
        expect(Array.from(result['assignments'])).toEqual([0, 0, 0, 0xffffffff]);
        expect(Array.from(result['assignedCosts'])).toEqual(Array.from(costs));
        const expected = operation === 'p-median' ? Math.fround(costs[0] + costs[1] + costs[2]) : 1;
        expect(Number.isFinite(result['objective'][0])).toBe(true);
        expect(result['objective'][0] / expected).toBeCloseTo(1, 6);
        expect(Array.from(result['status'])).toEqual([
          1,
          1,
          operation === 'p-median' ? 1 : 0,
          0,
          0,
          0,
          0
        ]);
      }
    });

const SCENES = [
  {
    operation: 'p-median' as const,
    costs: [Math.fround(1e30), Math.fround(2e30), Infinity, Infinity, Infinity, Infinity],
    weights: [1, 1],
    budget: 1
  },
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
