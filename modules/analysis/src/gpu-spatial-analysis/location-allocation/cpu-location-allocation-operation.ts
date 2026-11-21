// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {snapshotLocationAllocationOperationProps} from './location-allocation-construction';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPU_LOCATION_ALLOCATION_DEFAULT_MAXIMUM_SWAP_ITERATIONS,
  GPU_LOCATION_ALLOCATION_STATUS_LAYOUT,
  type GPULocationAllocationOperation as LocationAllocationObjective
} from './gpu-location-allocation';
import {
  GPULocationAllocationGraphOperation,
  type LocationAllocationOperationProps
} from './gpu-location-allocation-operation';

const NONE = 0xffffffff;

/** Input arrays; exactly one cost representation is required. */
export type CPULocationAllocationInput = {
  costs?: Float32Array;
  sparseCosts?: {offsets: Uint32Array; facilities: Uint32Array; costs: Float32Array};
  demandWeights?: Float32Array;
  coverageDistance?: Float32Array;
  /** Optional one-row matrix completeness word; see `GPULocationAllocationProps.costsComplete`. */
  costsComplete?: Uint32Array;
  /** Optional one-row matrix cost limit, evaluated independently of `costsComplete`. */
  costLimit?: Float32Array;
};
/** Output layout is GPULocationAllocation's seven-word status contract. */
export type CPULocationAllocationResult = {
  selectedFacilities: Uint32Array;
  assignments: Uint32Array;
  assignedCosts: Float32Array;
  objective: Float32Array;
  status: Uint32Array;
};
/** Dimensions, greedy objective and the maximum number of applied swap iterations. */
export type CPULocationAllocationOptions = {
  id?: string;
  facilityCount: number;
  demandCount: number;
  maximumFacilityCount: number;
  maximumSwapIterations?: number;
  operation: LocationAllocationObjective;
};

function validateOptions(id: string, options: CPULocationAllocationOptions): void {
  validateCapacity(id, 'facilityCount', options.facilityCount);
  validateCapacity(id, 'demandCount', options.demandCount);
  validateCapacity(
    id,
    'maximumFacilityCount',
    options.maximumFacilityCount,
    1,
    options.facilityCount
  );
  validateRequirement(
    id,
    ['p-median', 'maximum-coverage', 'set-covering'].includes(options.operation),
    'operation must be p-median, maximum-coverage or set-covering'
  );
  if (options.maximumSwapIterations !== undefined)
    validateCapacity(id, 'maximumSwapIterations', options.maximumSwapIterations, 0);
  validateRequirement(
    id,
    options.operation !== 'set-covering' || !options.maximumSwapIterations,
    'set-covering has no swap phase; omit maximumSwapIterations'
  );
}

/**
 * Greedy location-allocation followed, when enabled, by best improving single-facility swaps.
 * Reachable weight precedes reachable cost in p-median; ties select the lowest facility, and
 * swaps the lowest removed facility then lowest candidate. Set covering counts rows, including
 * zero-weight demand. Every finite nonnegative cost is reachable; positive infinity is unreachable.
 *
 * What differs from the GPU contributor:
 * - The explicit or default swap budget binds both forms; zero disables swaps. The CPU drops
 *   the GPU's 1000-iteration admission cap, command-node and swap-pair storage limits.
 * - Swaps accept every strictly positive lexicographic improvement, without the GPU's 1e-6
 *   noise tolerance. This can change selected facilities, swap counts and convergence evidence.
 * - Scores, gains and objectives use float64 with one float32 output rounding. Near ties may
 *   differ. Nonfinite demand weights use IEEE
 *   arithmetic and may produce NaN objectives.
 * - Inconsistent lengths throw. Invalid costs, weights, thresholds and sparse structure retain
 *   invalidCount status instead of throwing. Sparse row validation walks clamped row ranges;
 *   GPU merge-path traversal requires monotonic offsets, so its extra row-error count is not
 *   specified for decreasing offsets. The CPU reports each offending row entry once per range.
 *
 * @throws Error naming arrays whose lengths do not match the dimensions.
 */
export function computeLocationAllocationOnCPU(
  input: CPULocationAllocationInput,
  options: CPULocationAllocationOptions
): CPULocationAllocationResult {
  const id = options.id ?? 'location-allocation';
  validateOptions(id, options);
  const {facilityCount, demandCount, maximumFacilityCount, operation} = options;
  validateRequirement(
    id,
    !(Number(input.costs !== undefined) + Number(input.sparseCosts !== undefined) !== 1),
    'costs and sparseCosts require exactly one representation'
  );
  validateRequirement(
    id,
    !(input.costs && input.costs.length !== demandCount * facilityCount),
    'costs length must equal demandCount * facilityCount'
  );
  validateRequirement(
    id,
    !(input.demandWeights && input.demandWeights.length !== demandCount),
    'demandWeights length must equal demandCount'
  );
  validateRequirement(
    id,
    !(operation !== 'p-median' && !input.coverageDistance?.length),
    'coverageDistance must hold at least one value'
  );
  validateRequirement(
    id,
    !(input.costsComplete && input.costsComplete.length !== 1),
    'costsComplete must contain exactly one row'
  );
  validateRequirement(
    id,
    !(input.costLimit && input.costLimit.length !== 1),
    'costLimit must contain exactly one row'
  );
  const result: CPULocationAllocationResult = {
    selectedFacilities: new Uint32Array(facilityCount),
    assignments: new Uint32Array(demandCount).fill(NONE),
    assignedCosts: new Float32Array(demandCount).fill(Infinity),
    objective: new Float32Array(1),
    status: new Uint32Array(GPU_LOCATION_ALLOCATION_STATUS_LAYOUT.length)
  };
  const status = result.status;
  const fields = GPU_LOCATION_ALLOCATION_STATUS_LAYOUT;
  status[fields.uncoveredDemandCount] = demandCount;
  const isInvalid = (value: number) => Number.isNaN(value) || value < 0;
  const sparse = input.sparseCosts;
  if (sparse) {
    validateRequirement(
      id,
      !(sparse.offsets.length !== demandCount + 1),
      'sparseCosts.offsets length must equal demandCount + 1'
    );
    validateRequirement(
      id,
      !(!sparse.costs.length || sparse.facilities.length !== sparse.costs.length),
      'sparseCosts.costs and facilities must have equal positive lengths'
    );
    for (let row = 0; row <= demandCount; row++) {
      const offset = sparse.offsets[row];
      if (
        (row === 0 && offset !== 0) ||
        (row === demandCount && offset !== sparse.costs.length) ||
        (row < demandCount && offset > sparse.offsets[row + 1])
      )
        status[fields.invalidCount]++;
    }
    for (let row = 0; row < demandCount; row++)
      for (
        let entry = Math.min(sparse.offsets[row], sparse.costs.length);
        entry < Math.min(sparse.offsets[row + 1], sparse.costs.length);
        entry++
      ) {
        if (
          sparse.facilities[entry] >= facilityCount ||
          (entry > sparse.offsets[row] && sparse.facilities[entry - 1] >= sparse.facilities[entry])
        )
          status[fields.invalidCount]++;
      }
  }
  for (const cost of input.costs ?? sparse!.costs)
    if (isInvalid(cost)) status[fields.invalidCount]++;
  for (const weight of input.demandWeights ?? [])
    if (isInvalid(weight)) status[fields.invalidCount]++;
  const threshold = operation === 'p-median' ? 0 : input.coverageDistance![0];
  if (operation !== 'p-median' && isInvalid(threshold)) status[fields.invalidCount]++;
  if (status[fields.invalidCount]) return result;
  const getCost = (demand: number, facility: number): number => {
    if (input.costs) return input.costs[demand * facilityCount + facility];
    let lower = sparse!.offsets[demand];
    let upper = sparse!.offsets[demand + 1];
    const end = upper;
    while (lower < upper) {
      const middle = Math.floor((lower + upper) / 2);
      if (sparse!.facilities[middle] < facility) lower = middle + 1;
      else upper = middle;
    }
    return lower < end && sparse!.facilities[lower] === facility ? sparse!.costs[lower] : Infinity;
  };
  const getWeight = (demand: number) => input.demandWeights?.[demand] ?? 1;
  const nearest = new Float64Array(demandCount).fill(Infinity);
  const selected = result.selectedFacilities;
  for (let step = 0; step < maximumFacilityCount; step++) {
    let chosen = NONE;
    let bestPrimary = operation === 'p-median' ? Infinity : 0;
    let bestSecondary = Infinity;
    for (let facility = 0; facility < facilityCount; facility++) {
      if (selected[facility]) continue;
      let primary = 0;
      let secondary = 0;
      for (let demand = 0; demand < demandCount; demand++) {
        const cost = getCost(demand, facility);
        if (operation === 'p-median') {
          const next = Math.min(nearest[demand], cost);
          if (Number.isFinite(next)) secondary += getWeight(demand) * next;
          else primary += getWeight(demand);
        } else if (
          (!Number.isFinite(nearest[demand]) || nearest[demand] > threshold) &&
          Number.isFinite(cost) &&
          cost <= threshold
        )
          primary += operation === 'set-covering' ? 1 : getWeight(demand);
      }
      if (
        operation === 'p-median'
          ? primary < bestPrimary || (primary === bestPrimary && secondary < bestSecondary)
          : primary > bestPrimary
      ) {
        chosen = facility;
        bestPrimary = primary;
        bestSecondary = secondary;
      }
    }
    if (chosen === NONE) break;
    selected[chosen] = 1;
    status[fields.selectedFacilityCount]++;
    for (let demand = 0; demand < demandCount; demand++)
      nearest[demand] = Math.min(nearest[demand], getCost(demand, chosen));
  }
  const maximumSwapIterations =
    options.maximumSwapIterations ?? GPU_LOCATION_ALLOCATION_DEFAULT_MAXIMUM_SWAP_ITERATIONS;
  const hasSwaps = operation !== 'set-covering' && maximumSwapIterations > 0;
  if (hasSwaps)
    for (let iteration = 0; iteration < maximumSwapIterations; iteration++) {
      let bestPrimary = 0;
      let bestSecondary = 0;
      let removedFacility = NONE;
      let addedFacility = NONE;
      for (let removed = 0; removed < facilityCount; removed++) {
        if (!selected[removed]) continue;
        const fallback = new Float64Array(demandCount).fill(Infinity);
        for (let facility = 0; facility < facilityCount; facility++)
          if (selected[facility] && facility !== removed)
            for (let demand = 0; demand < demandCount; demand++)
              fallback[demand] = Math.min(fallback[demand], getCost(demand, facility));
        for (let candidate = 0; candidate < facilityCount; candidate++) {
          if (selected[candidate]) continue;
          let primary = 0;
          let secondary = 0;
          for (let demand = 0; demand < demandCount; demand++) {
            const before = nearest[demand];
            const after = Math.min(fallback[demand], getCost(demand, candidate));
            const weight = getWeight(demand);
            if (operation === 'p-median') {
              primary +=
                weight * (Number(Number.isFinite(after)) - Number(Number.isFinite(before)));
              secondary +=
                weight *
                ((Number.isFinite(before) ? before : 0) - (Number.isFinite(after) ? after : 0));
            } else
              primary +=
                weight *
                (Number(Number.isFinite(after) && after <= threshold) -
                  Number(Number.isFinite(before) && before <= threshold));
          }
          if (primary > bestPrimary || (primary === bestPrimary && secondary > bestSecondary)) {
            bestPrimary = primary;
            bestSecondary = secondary;
            removedFacility = removed;
            addedFacility = candidate;
          }
        }
      }
      if (removedFacility === NONE) {
        status[fields.converged] = 1;
        break;
      }
      selected[removedFacility] = 0;
      selected[addedFacility] = 1;
      status[fields.swapIterationCount]++;
      nearest.fill(Infinity);
      for (let facility = 0; facility < facilityCount; facility++)
        if (selected[facility])
          for (let demand = 0; demand < demandCount; demand++)
            nearest[demand] = Math.min(nearest[demand], getCost(demand, facility));
    }
  if (
    hasSwaps &&
    !status[fields.converged] &&
    status[fields.swapIterationCount] >= maximumSwapIterations
  )
    status[fields.iterationLimitReached] = 1;
  let objective = 0;
  status[fields.uncoveredDemandCount] = 0;
  for (let demand = 0; demand < demandCount; demand++) {
    let assignedCost = Infinity;
    for (let facility = 0; facility < facilityCount; facility++)
      if (selected[facility]) {
        const cost = getCost(demand, facility);
        if (
          Number.isFinite(cost) &&
          (cost < assignedCost || (cost === assignedCost && facility < result.assignments[demand]))
        ) {
          result.assignments[demand] = facility;
          assignedCost = cost;
        }
      }
    const assigned = result.assignments[demand] !== NONE;
    if (assigned) result.assignedCosts[demand] = assignedCost;
    const uncovered = !assigned || (operation !== 'p-median' && assignedCost > threshold);
    status[fields.uncoveredDemandCount] += Number(uncovered);
    if (!uncovered)
      objective += operation === 'p-median' ? getWeight(demand) * assignedCost : getWeight(demand);
  }
  result.objective[0] =
    operation === 'set-covering' ? status[fields.selectedFacilityCount] : objective;
  status[fields.complete] = Number(
    operation === 'set-covering'
      ? status[fields.uncoveredDemandCount] === 0
      : status[fields.selectedFacilityCount] === maximumFacilityCount
  );
  if (input.costsComplete || input.costLimit) {
    // The requested cutoff is coverageDistance, or +Infinity for p-median (NET-101).
    const requested = operation === 'p-median' ? Infinity : input.coverageDistance![0];
    const isWithinLimit = !input.costLimit || requested <= input.costLimit[0];
    const isComplete = !input.costsComplete || input.costsComplete[0] !== 0;
    if (!(isComplete && isWithinLimit)) status[fields.complete] = 0;
  }
  return result;
}

/** CPU twin with budgeted, zero-tolerance best-swap improvement when the phase is enabled. */
export class CPULocationAllocationOperation extends GPUCPUOperation {
  constructor(props: LocationAllocationOperationProps) {
    props = snapshotLocationAllocationOperationProps(props);
    const id = props.id ?? 'location-allocation';
    validateOptions(id, props);
    validateRequirement(
      id,
      Number(props.costs !== undefined) + Number(props.sparseCosts !== undefined) === 1,
      'costs and sparseCosts require exactly one representation'
    );
    validateRequirement(
      id,
      props.operation === 'p-median' || props.coverageDistance !== undefined,
      'coverageDistance is required for coverage operations'
    );
    const operation = new GPULocationAllocationGraphOperation(props);
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
        const result = computeLocationAllocationOnCPU(
          {
            costs:
              props.costs === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'costs'),
            sparseCosts:
              props.sparseCosts === undefined
                ? undefined
                : {
                    offsets: getCPUOperationUint32Rows(id, inputs, 'sparseOffsets'),
                    facilities: getCPUOperationUint32Rows(id, inputs, 'sparseFacilities'),
                    costs: getCPUOperationFloat32Rows(id, inputs, 'sparseCosts')
                  },
            demandWeights:
              props.demandWeights === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'demandWeights'),
            coverageDistance:
              props.coverageDistance === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'coverageDistance'),
            costsComplete:
              props.costsComplete === undefined
                ? undefined
                : getCPUOperationUint32Rows(id, inputs, 'costsComplete'),
            costLimit:
              props.costLimit === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'costLimit')
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPULocationAllocationResult]
          ])
        );
      }
    });
  }
}
