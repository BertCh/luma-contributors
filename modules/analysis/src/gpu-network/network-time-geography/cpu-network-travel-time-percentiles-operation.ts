// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  computeNetworkTimeRoutingFloat64,
  validateNetworkTimeRoutingOperationProps,
  type CPUNetworkTimeRoutingInput
} from '../network-optimization/cpu-network-time-routing-operation';
import {
  GPUNetworkTravelTimePercentilesOperation,
  type NetworkTravelTimePercentilesOperationProps
} from './gpu-network-travel-time-percentiles-operation';
import {getNetworkTravelTimePercentileRank} from './network-travel-time-percentile-rank';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Packed inputs for regular-window FIFO travel-time percentiles. */
export type CPUNetworkTravelTimePercentilesInput = Omit<
  CPUNetworkTimeRoutingInput,
  'endpoints' | 'departure'
> & {
  vertexCount: number;
  departureCount: number;
  percentiles: readonly number[];
  initialState: Uint32Array;
  window: Float32Array;
};

function validatePercentileOptions(
  id: string,
  props: Pick<
    CPUNetworkTravelTimePercentilesInput,
    'vertexCount' | 'departureCount' | 'percentiles'
  >
): void {
  validateCapacity(id, 'vertexCount', props.vertexCount);
  validateCapacity(id, 'departureCount', props.departureCount);
  validateCapacity(id, 'percentiles.length', props.percentiles.length);
  validateRequirement(
    id,
    Array.from(props.percentiles).every(
      (percentile, index, percentiles) =>
        Number.isFinite(percentile) &&
        percentile > 0 &&
        percentile <= 100 &&
        (index === 0 || percentile >= percentiles[index - 1])
    ),
    'percentiles must be ascending numbers in (0, 100]'
  );
}

/**
 * Routes every window departure within its budget, then selects vertex-major nearest-rank travel
 * times. Unreachable or invalid departures count as Infinity. Optional graph outputs have the
 * same layouts as these always-returned reachedCounts and per-departure [valid, converged] rows.
 * Ranks are ceil(percentile * departureCount / 100), evaluated exactly for the supplied
 * binary64 percentile without an epsilon, using the same host helper as the GPU contributor.
 *
 * What differs from the GPU contributor:
 * - Routing state, transition, sample and round caps are dropped, as are the 60-departure and
 *   16-percentile caps. Explicit maximumRelaxations still bounds each departure's sweeps.
 * - Departures, routing labels and travel-time subtraction stay float64 until final float32
 *   output; the GPU rounds at its intermediate buffers and shader operations.
 * - Float32 output overflow does not clear valid. A budget exit reports [1, 0]; invalid input
 *   contents retain the GPU's [0, 0] routing status.
 *
 * @throws If initialState, window, transitions or costs have inconsistent array lengths.
 */
export function computeNetworkTravelTimePercentilesOnCPU(
  input: CPUNetworkTravelTimePercentilesInput
): {
  travelTimes: Float32Array;
  reachedCounts: Uint32Array;
  status: Uint32Array;
} {
  validatePercentileOptions('network-travel-time-percentiles', input);
  if (input.initialState.length !== 1) throw new Error('initialState must hold one row');
  if (input.window.length !== 2) throw new Error('window must hold first departure and step');
  const {vertexCount, departureCount, percentiles, window} = input;
  const ranks = percentiles.map(percentile =>
    getNetworkTravelTimePercentileRank(percentile, departureCount)
  );
  const values = Array.from({length: vertexCount}, () =>
    new Float64Array(departureCount).fill(Infinity)
  );
  const status = new Uint32Array(departureCount * 2);
  const endpoints = new Uint32Array([input.initialState[0], 0]);
  for (let departure = 0; departure < departureCount; departure++) {
    const time = window[0] + departure * window[1];
    const result = computeNetworkTimeRoutingFloat64(
      {
        stateVertices: input.stateVertices,
        transitions: input.transitions,
        costs: input.costs,
        costModel: input.costModel,
        maximumRelaxations: input.maximumRelaxations,
        samplesPerTransition: input.samplesPerTransition,
        waitingPolicy: input.waitingPolicy,
        endpoints,
        departure: window.subarray(0, 1)
      },
      time
    );
    status[departure * 2] = result.valid;
    status[departure * 2 + 1] = result.converged;
    if (result.valid)
      for (let state = 0; state < input.stateVertices.length; state++) {
        const vertex = input.stateVertices[state];
        if (vertex < vertexCount)
          values[vertex][departure] = Math.min(
            values[vertex][departure],
            result.arrivals[state] - time
          );
      }
  }
  const travelTimes = new Float32Array(vertexCount * percentiles.length);
  const reachedCounts = new Uint32Array(vertexCount);
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    values[vertex].sort();
    reachedCounts[vertex] = values[vertex].reduce(
      (count, value) => count + (value < Infinity ? 1 : 0),
      0
    );
    for (let column = 0; column < percentiles.length; column++) {
      travelTimes[vertex * percentiles.length + column] = values[vertex][ranks[column] - 1];
    }
  }
  return {travelTimes, reachedCounts, status};
}

/** CPU reference twin with no GPU routing, window or percentile caps. */
export class CPUNetworkTravelTimePercentilesOperation extends GPUCPUOperation {
  constructor(originalProps: NetworkTravelTimePercentilesOperationProps) {
    const settings = snapshotConstructionFields(originalProps, [
      'id',
      'stateCount',
      'transitionCount',
      'stateVertices',
      'transitions',
      'costs',
      'initialState',
      'window',
      'vertexCount',
      'departureCount',
      'percentiles',
      'costModel',
      'maximumRelaxations',
      'samplesPerTransition',
      'waitingPolicy',
      'output'
    ]);
    const props = {
      ...settings,
      percentiles: Object.freeze(Array.from(settings.percentiles)),
      output: snapshotConstructionFields(settings.output, [
        'travelTimes',
        'reachedCounts',
        'status'
      ])
    };
    const id = props.id ?? 'network-travel-time-percentiles';
    validatePercentileOptions(id, props);
    validateNetworkTimeRoutingOperationProps(id, props);
    const operation = new GPUNetworkTravelTimePercentilesOperation(props);
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
        const result = computeNetworkTravelTimePercentilesOnCPU({
          stateVertices,
          transitions,
          costs: getCPUOperationFloat32Rows(id, inputs, 'costs', 'float32x2'),
          initialState: getCPUOperationUint32Rows(id, inputs, 'initialState'),
          window: getCPUOperationFloat32Rows(id, inputs, 'window'),
          vertexCount: props.vertexCount,
          departureCount: props.departureCount,
          percentiles: props.percentiles,
          costModel: props.costModel,
          maximumRelaxations: props.maximumRelaxations,
          samplesPerTransition: props.samplesPerTransition,
          waitingPolicy: props.waitingPolicy
        });
        const outputs: Record<string, Float32Array | Uint32Array> = {
          [props.output.travelTimes]: result.travelTimes
        };
        if (props.output.reachedCounts !== undefined)
          outputs[props.output.reachedCounts] = result.reachedCounts;
        if (props.output.status !== undefined) outputs[props.output.status] = result.status;
        return outputs;
      }
    });
  }
}
