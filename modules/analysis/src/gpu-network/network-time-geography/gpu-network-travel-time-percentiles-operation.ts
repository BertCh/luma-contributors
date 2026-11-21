// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationProps,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import type {NetworkTimeRoutingOperationProps} from '../network-optimization/gpu-network-time-routing-operation';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGPUNetworkTimeRoutingDeclarations,
  validateNetworkTimeRoutingResolvedRows
} from '../network-optimization/network-time-routing-parameters';
import {
  GPUNetworkTravelTimePercentiles,
  GPU_NETWORK_TRAVEL_TIME_PERCENTILES_MAXIMUM_DEPARTURE_COUNT,
  GPU_NETWORK_TRAVEL_TIME_PERCENTILES_MAXIMUM_PERCENTILE_COUNT
} from './gpu-network-travel-time-percentiles';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Shared graph-value names and compile-time shape for travel-time percentile twins. */
export type NetworkTravelTimePercentilesOperationProps = Omit<
  NetworkTimeRoutingOperationProps,
  'endpoints' | 'departure' | 'output'
> & {
  vertexCount: number;
  departureCount: number;
  /** Percentages in (0, 100], not fractions: probability .95 is percentile 95. */
  percentiles: readonly number[];
  /** uint32, one row. */
  initialState: string;
  /** float32, two rows: first relative departure and step. */
  window: string;
  output: {
    travelTimes: string;
    reachedCounts?: string;
    status?: string;
    /** CPU operation only: uint32 mask in the vertex-major selected percentile row domain. */
    unrepresentable?: string;
  };
};

/** Shared descriptors without GPU-only limits. @internal */
export function getNetworkTravelTimePercentilesOperationDescriptors(
  props: NetworkTravelTimePercentilesOperationProps
): Pick<GPUContributorOperationProps, 'id' | 'inputs' | 'outputs'> {
  const outputs: Record<string, GPUContributorOperationOutput> = {
    travelTimes: {
      value: props.output.travelTimes,
      format: 'float32',
      length: props.vertexCount * props.percentiles.length
    }
  };
  if (props.output.unrepresentable !== undefined)
    outputs['unrepresentable'] = {
      value: props.output.unrepresentable,
      format: 'uint32',
      length: props.vertexCount * props.percentiles.length
    };
  if (props.output.reachedCounts !== undefined)
    outputs['reachedCounts'] = {
      value: props.output.reachedCounts,
      format: 'uint32',
      length: props.vertexCount
    };
  if (props.output.status !== undefined)
    outputs['status'] = {
      value: props.output.status,
      format: 'uint32',
      length: props.departureCount * 2
    };
  return {
    id: props.id ?? 'network-travel-time-percentiles',
    inputs: {
      stateVertices: props.stateVertices,
      transitions: props.transitions,
      costs: props.costs,
      initialState: props.initialState,
      window: props.window
    },
    outputs
  };
}

/**
 * GPU operation form. Graph contents are per-frame; all properties are compile-time.
 * Declarations require positive vertexCount, 1–60 departures and 1–16 ascending finite
 * percentiles in (0,100]. Each departure allows 1–256 states, 0–4096 transitions,
 * 0–65536 sweeps and 2–256 samples per piecewise transition. Resolved state/transition
 * rows must match declarations before the contributor is constructed.
 */
export class GPUNetworkTravelTimePercentilesOperation extends GPUContributorOperation {
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
        'status',
        'unrepresentable'
      ])
    };
    const id = props.id ?? 'network-travel-time-percentiles';
    validateGPUNetworkTimeRoutingDeclarations(id, props);
    validateRequirement(
      id,
      props.output.unrepresentable === undefined,
      'output.unrepresentable requires CPUNetworkTravelTimePercentilesOperation'
    );
    validateCapacity(id, 'vertexCount', props.vertexCount);
    validateCapacity(
      id,
      'departureCount',
      props.departureCount,
      1,
      GPU_NETWORK_TRAVEL_TIME_PERCENTILES_MAXIMUM_DEPARTURE_COUNT
    );
    validateCapacity(
      id,
      'percentiles rows',
      props.percentiles.length,
      1,
      GPU_NETWORK_TRAVEL_TIME_PERCENTILES_MAXIMUM_PERCENTILE_COUNT
    );
    validateRequirement(
      id,
      props.percentiles.every(
        (percentile, index) =>
          Number.isFinite(percentile) &&
          percentile > 0 &&
          percentile <= 100 &&
          (index === 0 || percentile >= props.percentiles[index - 1])
      ),
      'percentiles must be ascending numbers in (0, 100]'
    );
    super({
      ...getNetworkTravelTimePercentilesOperationDescriptors(props),
      createContributor: getView => {
        const stateVertices = getView<'uint32'>('stateVertices');
        const transitions = getView<'uint32x2'>('transitions');
        validateNetworkTimeRoutingResolvedRows(id, props, stateVertices.length, transitions.length);
        return new GPUNetworkTravelTimePercentiles({
          id: props.id,
          vertexCount: props.vertexCount,
          departureCount: props.departureCount,
          percentiles: props.percentiles,
          costModel: props.costModel,
          samplesPerTransition: props.samplesPerTransition,
          waitingPolicy: props.waitingPolicy,
          maximumRelaxations: props.maximumRelaxations,
          stateVertices,
          transitions,
          costs: getView<'float32x2'>('costs'),
          initialState: getView<'uint32'>('initialState'),
          window: getView<'float32'>('window'),
          output: {
            travelTimes: getView<'float32'>('travelTimes'),
            reachedCounts:
              props.output.reachedCounts === undefined
                ? undefined
                : getView<'uint32'>('reachedCounts'),
            status: props.output.status === undefined ? undefined : getView<'uint32'>('status')
          }
        });
      }
    });
  }
}
