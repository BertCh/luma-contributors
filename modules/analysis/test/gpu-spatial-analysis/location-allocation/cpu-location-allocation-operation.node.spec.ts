// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {
  computeLocationAllocationOnCPU,
  CPULocationAllocationOperation
} from '../../../src/gpu-spatial-analysis/location-allocation/cpu-location-allocation-operation';
import {
  GPULocationAllocationGraphOperation,
  type LocationAllocationOperationProps
} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation-operation';
import {GPU_LOCATION_ALLOCATION_STATUS} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const OUTPUT = {
  selectedFacilities: 'selected',
  assignments: 'assignments',
  assignedCosts: 'assignedCosts',
  objective: 'objective',
  status: 'status'
};
const COSTS = Float32Array.of(5, 0, 100, 5, 1, 100, 5, 100, 0, 5, 100, 1);
const WEIGHTS = Float32Array.of(1, 2, 3, 4);
const COVERAGE_COSTS = Float32Array.of(
  0,
  0,
  10,
  0,
  0,
  10,
  0,
  10,
  0,
  0,
  10,
  0,
  10,
  0,
  10,
  10,
  10,
  0
);

it('weighted p-median closes a greedy gap and equals exhaustive enumeration of all facility pairs', () => {
  let optimum = Infinity;
  for (let first = 0; first < 3; first++)
    for (let second = first + 1; second < 3; second++) {
      let objective = 0;
      for (let demand = 0; demand < 4; demand++)
        objective +=
          WEIGHTS[demand] * Math.min(COSTS[demand * 3 + first], COSTS[demand * 3 + second]);
      optimum = Math.min(optimum, objective);
    }
  const options = {
    facilityCount: 3,
    demandCount: 4,
    maximumFacilityCount: 2,
    operation: 'p-median' as const
  };
  const greedy = computeLocationAllocationOnCPU(
    {costs: COSTS, demandWeights: WEIGHTS},
    {...options, maximumSwapIterations: 0}
  );
  const result = computeLocationAllocationOnCPU(
    {costs: COSTS, demandWeights: WEIGHTS},
    {...options, maximumSwapIterations: 1}
  );
  expect(greedy.objective[0]).toBe(19);
  expect(result.objective[0]).toBe(optimum);
  expect(optimum).toBe(6);
  expect(Array.from(result.selectedFacilities)).toEqual([0, 1, 1]);
  expect(Array.from(result.assignments)).toEqual([1, 1, 2, 2]);
  expect(Array.from(result.assignedCosts)).toEqual([0, 1, 0, 1]);
  expect(Array.from(result.status)).toEqual([2, 0, 1, 0, 1, 1, 0]);
});

it('maximum coverage makes a real weighted swap; set covering counts zero-weight demand rows', () => {
  const result = computeLocationAllocationOnCPU(
    {
      costs: COVERAGE_COSTS,
      coverageDistance: Float32Array.of(0),
      demandWeights: Float32Array.of(1, 2, 1, 2, 1, 2)
    },
    {operation: 'maximum-coverage', facilityCount: 3, demandCount: 6, maximumFacilityCount: 2}
  );
  // Facilities 1 and 2 cover disjoint triples: their union is all six rows, total weight nine.
  expect(Array.from(result.selectedFacilities)).toEqual([0, 1, 1]);
  expect(result.objective[0]).toBe(9);
  expect(Array.from(result.status)).toEqual([2, 0, 1, 0, 1, 1, 0]);
  const covering = computeLocationAllocationOnCPU(
    {
      costs: COVERAGE_COSTS,
      coverageDistance: Float32Array.of(0),
      demandWeights: new Float32Array(6)
    },
    {operation: 'set-covering', facilityCount: 3, demandCount: 6, maximumFacilityCount: 3}
  );
  expect(Array.from(covering.selectedFacilities)).toEqual([1, 1, 1]);
  expect(covering.objective[0]).toBe(3);
  expect(Array.from(covering.status)).toEqual([3, 0, 1, 0, 0, 0, 0]);
});

it('unreachable rows do not swamp finite differences, and assignment ties select the lowest facility', () => {
  const result = computeLocationAllocationOnCPU(
    {
      costs: Float32Array.of(Infinity, Infinity, 9, 1, 9, 1),
      demandWeights: Float32Array.of(4, 2, 3)
    },
    {
      operation: 'p-median',
      facilityCount: 2,
      demandCount: 3,
      maximumFacilityCount: 1,
      maximumSwapIterations: 0
    }
  );
  expect(Array.from(result.selectedFacilities)).toEqual([0, 1]);
  expect(result.objective[0]).toBe(5);
  expect(Array.from(result.assignments)).toEqual([0xffffffff, 1, 1]);
  expect(result.assignedCosts[0]).toBe(Infinity);
  expect(Array.from(result.status)).toEqual([1, 1, 1, 0, 0, 0, 0]);
  const tied = computeLocationAllocationOnCPU(
    {costs: Float32Array.of(1, 1, 2, 2)},
    {
      operation: 'p-median',
      facilityCount: 2,
      demandCount: 2,
      maximumFacilityCount: 2,
      maximumSwapIterations: 0
    }
  );
  expect(Array.from(tied.assignments)).toEqual([0, 0]);
});

it('sparse costs produce the hand-computed result, with precise invalid status for valid offsets', () => {
  const sparseCosts = {
    offsets: Uint32Array.of(0, 3, 6, 9, 12),
    facilities: Uint32Array.of(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2),
    costs: COSTS
  };
  const options = {
    operation: 'p-median' as const,
    facilityCount: 3,
    demandCount: 4,
    maximumFacilityCount: 2
  };
  const result = computeLocationAllocationOnCPU({sparseCosts, demandWeights: WEIGHTS}, options);
  expect(Array.from(result.selectedFacilities)).toEqual([0, 1, 1]);
  expect(result.objective[0]).toBe(6);
  const bad = computeLocationAllocationOnCPU(
    {
      sparseCosts: {
        offsets: Uint32Array.of(0, 3),
        facilities: Uint32Array.of(1, 1, 5),
        costs: Float32Array.of(0, -1, NaN)
      },
      demandWeights: Float32Array.of(-1),
      coverageDistance: Float32Array.of(-1)
    },
    {operation: 'maximum-coverage', facilityCount: 3, demandCount: 1, maximumFacilityCount: 2}
  );
  expect(Array.from(bad.status)).toEqual([0, 1, 0, 6, 0, 0, 0]);
  expect(Array.from(bad.assignments)).toEqual([0xffffffff]);
  expect(bad.assignedCosts[0]).toBe(Infinity);
  expect(bad.objective[0]).toBe(0);
});

it('rejects inconsistent arrays and validates both classes at construction/lowering', () => {
  const props: LocationAllocationOperationProps = {
    id: 'allocation-test',
    demandCount: 4,
    facilityCount: 3,
    maximumFacilityCount: 2,
    costs: 'costs',
    operation: 'p-median',
    maximumSwapIterations: 0,
    output: OUTPUT
  };
  expect(new CPULocationAllocationOperation(props).contract.outputs).toEqual(
    new GPULocationAllocationGraphOperation(props).contract.outputs
  );
  expect(() => new CPULocationAllocationOperation({...props, maximumFacilityCount: 4})).toThrow(
    /^allocation-test maximumFacilityCount/
  );
  expect(() => new CPULocationAllocationOperation({...props, costs: undefined})).toThrow(
    /^allocation-test costs/
  );
  expect(
    () => new CPULocationAllocationOperation({...props, maximumSwapIterations: 1001})
  ).not.toThrow();
  expect(() => new GPULocationAllocationGraphOperation({...props, facilityCount: 0})).toThrow(
    /^allocation-test outputs.selectedFacilities.length/
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> =>
      createTransientView(
        graph,
        port,
        ['costs', 'assignedCosts', 'objective'].includes(port) ? 'float32' : 'uint32',
        port === 'costs'
          ? 12
          : port === 'status'
            ? GPU_LOCATION_ALLOCATION_STATUS.length
            : port === 'selectedFacilities'
              ? 3
              : port === 'objective'
                ? 1
                : 4
      ) as GraphDataView<Format>;
    expect(() =>
      new GPULocationAllocationGraphOperation({
        ...props,
        maximumFacilityCount: 4
      }).createContributor(getView)
    ).toThrow(/^allocation-test maximumFacilityCount/);
  } finally {
    device.destroy();
  }
  expect(() => computeLocationAllocationOnCPU({costs: Float32Array.of(1)}, props)).toThrow(
    'costs length'
  );
  expect(() =>
    computeLocationAllocationOnCPU({costs: COSTS, demandWeights: Float32Array.of(1)}, props)
  ).toThrow('demandWeights');
  expect(() =>
    computeLocationAllocationOnCPU(
      {
        sparseCosts: {
          offsets: Uint32Array.of(0, 1),
          facilities: Uint32Array.of(0),
          costs: Float32Array.of(1)
        }
      },
      props
    )
  ).toThrow('sparseCosts.offsets');
});
