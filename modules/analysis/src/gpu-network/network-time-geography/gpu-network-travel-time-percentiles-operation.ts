// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import type {NetworkTimeRoutingOperationProps} from '../network-optimization/gpu-network-time-routing-operation';
import {GPUNetworkTravelTimePercentiles} from './gpu-network-travel-time-percentiles';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Shared graph-value names and compile-time shape for travel-time percentile twins. */
export type NetworkTravelTimePercentilesOperationProps = Omit<
  NetworkTimeRoutingOperationProps,
  'endpoints' | 'departure' | 'output'
> & {
  vertexCount: number;
  departureCount: number;
  percentiles: readonly number[];
  /** uint32, one row. */
  initialState: string;
  /** float32, two rows: first relative departure and step. */
  window: string;
  output: {travelTimes: string; reachedCounts?: string; status?: string};
};

/** GPU operation form. Graph contents are per-frame; all properties are compile-time. */
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
        'status'
      ])
    };
    const outputs: Record<string, GPUContributorOperationOutput> = {
      travelTimes: {
        value: props.output.travelTimes,
        format: 'float32',
        length: props.vertexCount * props.percentiles.length
      }
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
    super({
      id: props.id ?? 'network-travel-time-percentiles',
      inputs: {
        stateVertices: props.stateVertices,
        transitions: props.transitions,
        costs: props.costs,
        initialState: props.initialState,
        window: props.window
      },
      outputs,
      createContributor: getView =>
        new GPUNetworkTravelTimePercentiles({
          id: props.id,
          vertexCount: props.vertexCount,
          departureCount: props.departureCount,
          percentiles: props.percentiles,
          costModel: props.costModel,
          samplesPerTransition: props.samplesPerTransition,
          waitingPolicy: props.waitingPolicy,
          maximumRelaxations: props.maximumRelaxations,
          stateVertices: getView<'uint32'>('stateVertices'),
          transitions: getView<'uint32x2'>('transitions'),
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
        })
    });
  }
}
