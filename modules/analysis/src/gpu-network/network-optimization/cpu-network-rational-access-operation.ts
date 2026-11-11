// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUNetworkRationalAccessOperation,
  type NetworkRationalAccessOperationProps
} from './gpu-network-rational-access-operation';

/** Retained precision limit: integral assignments must be exactly representable in float32. */
const MAXIMUM_TOTAL_DEMAND = 16777216;

export type CPUNetworkRationalAccessInput = {
  costs: Float32Array;
  demand: Float32Array;
  supply: Float32Array;
  parameters: Float32Array;
};

export type CPUNetworkRationalAccessOptions = Pick<
  NetworkRationalAccessOperationProps,
  'maximumCycles' | 'initialLimitCycles' | 'demandMode' | 'stepMode'
>;

export type CPUNetworkRationalAccessResult = {
  assignments: Float32Array;
  scores: Float32Array;
  unservedDemand: Float32Array;
  /** [valid, gapWithinTolerance, completedCycles, float32 maximum-gap bits]. */
  status: Uint32Array;
};

/**
 * Runs simultaneous rational-agent equalization, retaining earliest-facility ties, integral
 * flooring and the early inflow cap's ties-to-even rounding. Unreachable or zero-demand origins
 * retain NaN scores. Invalid numeric data returns valid=0 and the partial initialization/results
 * reached before rejection, exactly in the contributor's order.
 *
 * What differs from the GPU contributor:
 * - Arithmetic uses float64 and rounds once per output. Near integer-shift or facility-cost ties,
 *   allocations and convergence cycles can differ. Well-conditioned scores/gaps use relative
 *   tolerance 2e-5; integral assignments compare exactly when the selected moves agree.
 * - GPU-only 64-origin, 32-facility and 4096-cycle caps are dropped. maximumCycles (default 150)
 *   and initialLimitCycles (default 20) remain algorithm controls, including nonconverged stalled
 *   states; the total-demand precision limit of 2^24 is retained for the float32 assignment layout.
 * - Float32 intermediate overflow/underflow does not invalidate otherwise finite float64 work;
 *   output overflow becomes Infinity. Subnormal values are retained.
 *
 * @throws If costs or parameters have inconsistent lengths, demand/supply are empty, or cycle
 * controls are not nonnegative integers. Invalid numeric rows instead return valid=0.
 */
export function computeNetworkRationalAccessOnCPU(
  input: CPUNetworkRationalAccessInput,
  options: CPUNetworkRationalAccessOptions = {}
): CPUNetworkRationalAccessResult {
  const {costs, demand, supply, parameters} = input;
  const originCount = demand.length;
  const facilityCount = supply.length;
  if (!originCount) throw new Error('demand must contain at least one origin');
  if (!facilityCount) throw new Error('supply must contain at least one facility');
  if (costs.length !== originCount * facilityCount)
    throw new Error('costs length must equal demand.length * supply.length');
  if (parameters.length !== 6) throw new Error('parameters must contain six rows');
  const maximumCycles = options.maximumCycles ?? 150;
  const initialLimitCycles = options.initialLimitCycles ?? 20;
  validateCapacity('computeNetworkRationalAccessOnCPU', 'maximumCycles', maximumCycles, 0);
  validateCapacity(
    'computeNetworkRationalAccessOnCPU',
    'initialLimitCycles',
    initialLimitCycles,
    0
  );
  const fractional = options.demandMode === 'fractional';
  const absolute = options.stepMode === 'absolute';
  const assignments = new Float64Array(costs.length);
  const scores = new Float64Array(originCount).fill(NaN);
  const unservedDemand = Float64Array.from(demand);
  const status = new Uint32Array(4);
  const finishResult = (): CPUNetworkRationalAccessResult => ({
    assignments: Float32Array.from(assignments),
    scores: Float32Array.from(scores),
    unservedDemand: Float32Array.from(unservedDemand),
    status
  });
  // The contributor has no exported parameter packer/layout. Its documented six-row layout:
  const [travelScale, supplyScale, initialStep, minimumStep, halfLife, tolerance] = parameters;
  if (
    !Number.isFinite(travelScale) ||
    travelScale <= 0 ||
    !Number.isFinite(supplyScale) ||
    supplyScale <= 0 ||
    !Number.isFinite(initialStep) ||
    initialStep < 0 ||
    (!absolute && initialStep > 1) ||
    !Number.isFinite(minimumStep) ||
    minimumStep < 0 ||
    (!absolute && minimumStep > 1) ||
    !Number.isFinite(halfLife) ||
    halfLife <= 0 ||
    !Number.isFinite(tolerance) ||
    tolerance < 0
  )
    return finishResult();
  const capacities = Float64Array.from(supply, value => value * supplyScale);
  if (capacities.some(value => !Number.isFinite(value) || value <= 0)) return finishResult();
  let totalDemand = 0;
  for (let origin = 0; origin < originCount; origin++) {
    const amount = demand[origin];
    if (
      !Number.isFinite(amount) ||
      amount < 0 ||
      (!fractional && Math.floor(amount) !== amount) ||
      amount > MAXIMUM_TOTAL_DEMAND - totalDemand
    )
      return finishResult();
    totalDemand += amount;
    let nearest = -1;
    let nearestCost = Infinity;
    for (let facility = 0; facility < facilityCount; facility++) {
      const cost = costs[origin * facilityCount + facility];
      if (cost < 0 || Number.isNaN(cost)) return finishResult();
      if (cost < nearestCost) {
        nearest = facility;
        nearestCost = cost;
      }
    }
    if (nearest >= 0) {
      assignments[origin * facilityCount + nearest] = amount;
      unservedDemand[origin] = 0;
    }
  }
  status[0] = 1;
  for (let cycle = 0; cycle <= maximumCycles; cycle++) {
    const loads = new Float64Array(facilityCount);
    const incoming = new Float64Array(facilityCount);
    const minima = new Uint32Array(originCount);
    const maxima = new Uint32Array(originCount);
    const shifts = new Float64Array(originCount);
    for (let origin = 0; origin < originCount; origin++)
      for (let facility = 0; facility < facilityCount; facility++)
        loads[facility] += assignments[origin * facilityCount + facility];
    let maximumGap = 0;
    const step = Math.max(minimumStep, initialStep * 2 ** (-cycle / halfLife));
    for (let origin = 0; origin < originCount; origin++) {
      let lowest = Infinity;
      let highest = -1;
      let minimum = 0;
      let maximum = 0;
      let weightedCost = 0;
      for (let facility = 0; facility < facilityCount; facility++) {
        const travel = costs[origin * facilityCount + facility];
        if (!Number.isFinite(travel)) continue;
        const cost = travel / travelScale + loads[facility] / capacities[facility];
        if (!Number.isFinite(cost)) {
          status[0] = 0;
          return finishResult();
        }
        const held = assignments[origin * facilityCount + facility];
        if (cost < lowest) {
          lowest = cost;
          minimum = facility;
        }
        if (held > 0 && cost > highest) {
          highest = cost;
          maximum = facility;
        }
        weightedCost += cost * held;
      }
      const amount = demand[origin];
      if (highest < 0 || amount === 0) continue;
      if (!Number.isFinite(weightedCost)) {
        status[0] = 0;
        return finishResult();
      }
      scores[origin] = weightedCost / amount;
      maximumGap = Math.max(maximumGap, highest - lowest);
      minima[origin] = minimum;
      maxima[origin] = maximum;
      if (minimum === maximum) continue;
      const held = assignments[origin * facilityCount + maximum];
      const equalizingShift =
        (highest - lowest) / (1 / capacities[minimum] + 1 / capacities[maximum]);
      const boundedShift = Math.max(
        0,
        Math.min(held, equalizingShift, absolute ? step : step * amount)
      );
      shifts[origin] = fractional ? boundedShift : Math.floor(boundedShift);
      incoming[minimum] += shifts[origin];
    }
    status[2] = cycle;
    status[3] = new Uint32Array(Float32Array.of(maximumGap).buffer)[0];
    if (maximumGap <= tolerance) {
      status[1] = 1;
      break;
    }
    if (cycle === maximumCycles) break;
    for (let origin = 0; origin < originCount; origin++) {
      let shift = shifts[origin];
      if (shift === 0) continue;
      if (cycle < initialLimitCycles) {
        const limited = shift / Math.max(1, incoming[minima[origin]] / capacities[minima[origin]]);
        shift = fractional ? limited : roundEven(limited);
      }
      assignments[origin * facilityCount + minima[origin]] += shift;
      assignments[origin * facilityCount + maxima[origin]] -= shift;
    }
  }
  return finishResult();
}

function roundEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  return lower + Number(fraction > 0.5 || (fraction === 0.5 && lower % 2 === 1));
}

/** CPU rational-access twin; GPU array-size and cycle-count caps are removed. */
export class CPUNetworkRationalAccessOperation extends GPUCPUOperation {
  constructor(props: NetworkRationalAccessOperationProps) {
    const id = props.id ?? 'network-rational-access';
    validateCapacity(id, 'originCount', props.originCount);
    validateCapacity(id, 'facilityCount', props.facilityCount);
    validateMaximumCount(
      id,
      'originCount * facilityCount',
      props.originCount * props.facilityCount,
      0xffffffff
    );
    validateCapacity(id, 'maximumCycles', props.maximumCycles ?? 150, 0);
    validateCapacity(id, 'initialLimitCycles', props.initialLimitCycles ?? 20, 0);
    const operation = new GPUNetworkRationalAccessOperation(props);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const input = {
          costs: getCPUOperationFloat32Rows(id, inputs, 'costs'),
          demand: getCPUOperationFloat32Rows(id, inputs, 'demand'),
          supply: getCPUOperationFloat32Rows(id, inputs, 'supply'),
          parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters')
        };
        if (input.demand.length !== props.originCount)
          throw new Error(`${id} demand length must equal originCount`);
        if (input.supply.length !== props.facilityCount)
          throw new Error(`${id} supply length must equal facilityCount`);
        const result = computeNetworkRationalAccessOnCPU(input, props);
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUNetworkRationalAccessResult]
          ])
        );
      }
    });
  }
}
