// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph, type GPUCPUResult} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getNetworkTimeRoutingOperationDescriptors,
  type NetworkTimeRoutingOperationProps
} from './gpu-network-time-routing-operation';

import {
  getNetworkOptimizationCPUStatus,
  type NetworkOptimizationCPUStatus
} from './network-optimization-status';

import {
  validateNetworkTimeRoutingOptions,
  snapshotNetworkTimeRoutingOperationProps
} from './network-time-routing-parameters';

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';

const NONE = 0xffffffff;

/** Packed routing inputs and cost layout, matching GPUNetworkTimeRouting. */
export type CPUNetworkTimeRoutingInput = Pick<
  NetworkTimeRoutingOperationProps,
  'costModel' | 'samplesPerTransition' | 'waitingPolicy' | 'maximumRelaxations'
> & {
  stateVertices: Uint32Array;
  transitions: Uint32Array;
  costs: Float32Array;
  endpoints: Uint32Array;
  departure: Float32Array;
};

/** Retained float64 labels for consumers that must round only their final output. @internal */
export type NetworkTimeRoutingFloat64Result = {
  arrivals: Float64Array;
  elapsedTimes: Float64Array;
  predecessorTransitions: Uint32Array;
  valid: number;
  converged: number;
  targetState: number;
  arrivalTime: number;
  iterationCount: number;
};

/** Validates compile-time routing shape/model props without GPU-only caps. @internal */
export function validateNetworkTimeRoutingOperationProps(
  id: string,
  props: Pick<
    NetworkTimeRoutingOperationProps,
    | 'stateCount'
    | 'transitionCount'
    | 'costModel'
    | 'samplesPerTransition'
    | 'waitingPolicy'
    | 'maximumRelaxations'
  >
): void {
  validateCapacity(id, 'stateCount', props.stateCount);
  validateCapacity(id, 'transitionCount', props.transitionCount, 0);
  validateNetworkTimeRoutingOptions(id, props);
  if (props.costModel === 'piecewise-arrival')
    validateCapacity(id, 'samplesPerTransition', props.samplesPerTransition as number, 2);
  if (props.maximumRelaxations !== undefined)
    validateCapacity(id, 'maximumRelaxations', props.maximumRelaxations, 0);
}

/**
 * Budgeted label correction with float64 labels. Scan order and strict improvements preserve the
 * GPU's predecessor tie rule. Used by the percentile twin without an intermediate float32 cast.
 * departureTime overrides the one-row departure for a float64 departure window. @internal
 */
export function computeNetworkTimeRoutingFloat64(
  input: CPUNetworkTimeRoutingInput,
  departureTime: number = input.departure[0]
): NetworkTimeRoutingFloat64Result {
  validateNetworkTimeRoutingOptions('network-time-routing', input);
  const {stateVertices, transitions, costs, endpoints, departure} = input;
  const stateCount = stateVertices.length;
  const maximumRelaxations = input.maximumRelaxations ?? stateCount;
  validateCapacity('network-time-routing', 'maximumRelaxations', maximumRelaxations, 0);
  if (transitions.length % 2) throw new Error('transitions must hold two components per row');
  const transitionCount = transitions.length / 2;
  const sampleCount = input.costModel === 'piecewise-arrival' ? input.samplesPerTransition! : 1;
  if (
    !Number.isInteger(sampleCount) ||
    sampleCount < 1 ||
    (input.costModel === 'piecewise-arrival' && sampleCount < 2)
  )
    throw new Error('samplesPerTransition must be at least two for piecewise costs');
  if (costs.length !== transitionCount * sampleCount * 2)
    throw new Error('costs length must match transitions and samplesPerTransition');
  if (endpoints.length !== 2)
    throw new Error('endpoints must hold initial state and destination vertex');
  if (departure.length !== 1) throw new Error('departure must hold one row');
  const arrivals = new Float64Array(stateCount).fill(Infinity);
  const elapsedTimes = new Float64Array(stateCount).fill(Infinity);
  const predecessorTransitions = new Uint32Array(stateCount).fill(NONE);
  const result: NetworkTimeRoutingFloat64Result = {
    arrivals,
    elapsedTimes,
    predecessorTransitions,
    valid: 0,
    converged: 0,
    targetState: NONE,
    arrivalTime: Infinity,
    iterationCount: 0
  };
  const initial = endpoints[0];
  if (initial >= stateCount || !Number.isFinite(departureTime) || departureTime < 0) return result;
  // Contents that the shader declares invalid produce its initialized missing labels/status.
  for (let edge = 0; edge < transitionCount; edge++) {
    if (transitions[edge * 2] >= stateCount || transitions[edge * 2 + 1] >= stateCount)
      return result;
    if (input.costModel === 'piecewise-arrival') {
      let previousDeparture = -1;
      let previousArrival = -1;
      for (let sample = 0; sample < sampleCount; sample++) {
        const offset = (edge * sampleCount + sample) * 2;
        const sampleDeparture = costs[offset];
        const sampleArrival = costs[offset + 1];
        if (
          !Number.isFinite(sampleDeparture) ||
          !Number.isFinite(sampleArrival) ||
          sampleDeparture < 0 ||
          sampleDeparture <= previousDeparture ||
          sampleArrival < sampleDeparture ||
          sampleArrival < previousArrival
        )
          return result;
        previousDeparture = sampleDeparture;
        previousArrival = sampleArrival;
      }
    } else {
      const base = costs[edge * 2];
      const slope = costs[edge * 2 + 1];
      if (!Number.isFinite(base) || !Number.isFinite(slope) || base < 0 || slope < 0) return result;
    }
  }
  result.valid = 1;
  arrivals[initial] = departureTime;
  elapsedTimes[initial] = 0;
  for (let round = 0; round < maximumRelaxations; round++) {
    result.iterationCount++;
    let changed = false;
    for (let edge = 0; edge < transitionCount; edge++) {
      const source = transitions[edge * 2];
      const destination = transitions[edge * 2 + 1];
      const time = elapsedTimes[source];
      if (!Number.isFinite(time)) continue;
      let candidate: number;
      if (input.costModel === 'piecewise-arrival') {
        const base = edge * sampleCount * 2;
        if (time <= costs[base] - departureTime) candidate = costs[base + 1] - departureTime;
        else {
          const last = base + (sampleCount - 1) * 2;
          candidate = time + Math.max(0, costs[last + 1] - costs[last]);
          for (let sample = 1; sample < sampleCount; sample++) {
            const right = base + sample * 2;
            if (time <= costs[right] - departureTime) {
              const fraction =
                (time - (costs[right - 2] - departureTime)) / (costs[right] - costs[right - 2]);
              candidate = Math.max(
                time,
                time +
                  (costs[right - 1] - costs[right - 2]) +
                  (costs[right + 1] - costs[right] - (costs[right - 1] - costs[right - 2])) *
                    fraction
              );
              break;
            }
          }
        }
      } else
        candidate =
          time + costs[edge * 2] + costs[edge * 2 + 1] * departureTime + costs[edge * 2 + 1] * time;
      if (!Number.isFinite(candidate)) {
        result.valid = 0;
        return result;
      }
      if (candidate < elapsedTimes[destination]) {
        elapsedTimes[destination] = candidate;
        arrivals[destination] = departureTime + candidate;
        predecessorTransitions[destination] = edge;
        changed = true;
      }
    }
    if (!changed) {
      result.converged = 1;
      break;
    }
  }
  let bestElapsed = Infinity;
  for (let state = 0; state < stateCount; state++) {
    if (stateVertices[state] === endpoints[1] && elapsedTimes[state] < bestElapsed) {
      bestElapsed = elapsedTimes[state];
      result.targetState = state;
      result.arrivalTime = arrivals[state];
    }
  }
  return result;
}

/** CPU routing payload and row-scoped float32 representability evidence. */
export type CPUNetworkTimeRoutingResult = {
  arrivals: Float32Array;
  /** Relative seconds since departure, one per expanded state. */
  elapsedTimes?: Float32Array;
  predecessorTransitions: Uint32Array;
  summary: Uint32Array;
  status: NetworkOptimizationCPUStatus;
  /** One per state: finite float64 arrival rounded to a nonfinite float32 value. */
  unrepresentable: Uint32Array;
};

/**
 * Computes FIFO earliest arrivals and predecessor transitions in float64, then rounds once.
 * Summary is [valid, converged, targetState, float32 arrival bits]; missing keys are 0xffffffff.
 *
 * What differs from the GPU contributor:
 * - GPU state, transition and sample caps are dropped. maximumRelaxations bounds transition
 *   sweeps (default state count), including the final unchanged sweep that proves convergence.
 * - Float64 arithmetic can distinguish routes tied in float32 and accept arrivals that overflow
 *   shader arithmetic. Float32 output overflow becomes Infinity without clearing valid.
 * - Strict improvement in transition scan order and lowest-state target ties match the GPU.
 *
 * @throws If transitions, costs, endpoints or departure have inconsistent array lengths.
 */
export function computeNetworkTimeRoutingOnCPU(
  input: CPUNetworkTimeRoutingInput
): CPUNetworkTimeRoutingResult {
  const result = computeNetworkTimeRoutingFloat64(input);
  const arrivalBits = new Uint32Array(new Float32Array([result.arrivalTime]).buffer)[0];
  const summary = new Uint32Array([
    result.valid,
    result.converged,
    result.targetState,
    arrivalBits
  ]);
  return {
    arrivals: Float32Array.from(result.arrivals),
    elapsedTimes: Float32Array.from(result.elapsedTimes),
    unrepresentable: Uint32Array.from(result.arrivals, value =>
      Number.isFinite(value) && !Number.isFinite(Math.fround(value)) ? 1 : 0
    ),
    predecessorTransitions: result.predecessorTransitions,
    summary,
    status: getNetworkOptimizationCPUStatus(summary, result.iterationCount)
  };
}

/** CPU twin, preferred for exact reference routing. GPU shape caps are dropped; budgets bind. */
export class CPUNetworkTimeRoutingOperation extends GPUCPUOperation {
  constructor(originalProps: NetworkTimeRoutingOperationProps) {
    const props = snapshotNetworkTimeRoutingOperationProps(originalProps);
    const id = props.id ?? 'network-time-routing';
    validateNetworkTimeRoutingOperationProps(id, props);
    // Validate graph names and output aliases independently of the GPU routing limits.
    const operation = new GPUContributorOperation({
      ...getNetworkTimeRoutingOperationDescriptors(props),
      createContributor: () => {
        throw new Error(`${id}: CPU descriptors do not lower a contributor`);
      }
    });
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
        const stateVertices = getCPUOperationUint32Rows(id, inputs, 'stateVertices');
        const transitions = getCPUOperationUint32Rows(id, inputs, 'transitions', 'uint32x2');
        if (stateVertices.length !== props.stateCount)
          throw new Error(`${id} stateVertices length must equal stateCount`);
        if (transitions.length !== props.transitionCount * 2)
          throw new Error(`${id} transitions length must equal transitionCount * 2`);
        const result = computeNetworkTimeRoutingOnCPU({
          stateVertices,
          transitions,
          costs: getCPUOperationFloat32Rows(id, inputs, 'costs', 'float32x2'),
          endpoints: getCPUOperationUint32Rows(id, inputs, 'endpoints'),
          departure: getCPUOperationFloat32Rows(id, inputs, 'departure'),
          costModel: props.costModel,
          maximumRelaxations: props.maximumRelaxations,
          samplesPerTransition: props.samplesPerTransition,
          waitingPolicy: props.waitingPolicy
        });
        const {status, ...primaryRows} = result;
        const outputRows = {...primaryRows, ...status};
        return Object.fromEntries<GPUCPUResult>(
          Object.entries(operation.outputs).map(([port, output]) => {
            const rows = outputRows[port as keyof typeof outputRows];
            if (rows === undefined) {
              throw new Error(`${id} output.${port} has no CPU result`);
            }
            return [output.value, rows] as const;
          })
        );
      }
    });
  }
}
