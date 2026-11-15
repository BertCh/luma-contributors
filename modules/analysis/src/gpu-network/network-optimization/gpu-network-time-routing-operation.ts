// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getNetworkOptimizationStatusOutputs,
  type NetworkOptimizationStatusValues
} from './network-optimization-status';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNetworkTimeRouting, type GPUNetworkTimeRoutingProps} from './gpu-network-time-routing';

/** Shared shapes, model choices and graph-value names for FIFO routing operations. */
export type NetworkTimeRoutingOperationProps = Pick<
  GPUNetworkTimeRoutingProps,
  'id' | 'costModel' | 'samplesPerTransition' | 'waitingPolicy' | 'maximumRelaxations'
> & {
  stateCount: number;
  transitionCount: number;
  /** uint32 physical vertex per expanded state. */
  stateVertices: string;
  /** uint32x2 expanded-state endpoints per transition. */
  transitions: string;
  /** float32x2 affine costs, or transition-major piecewise samples. */
  costs: string;
  /** uint32, two rows: initial state, physical destination. */
  endpoints: string;
  /** float32, one relative departure. */
  departure: string;
  output: {
    arrivals: string;
    predecessorTransitions: string;
    summary: string;
    status?: NetworkOptimizationStatusValues;
  };
};

/** GPU operation form. Every property is compile-time; graph contents are per-frame. */
export class GPUNetworkTimeRoutingOperation extends GPUContributorOperation {
  constructor(props: NetworkTimeRoutingOperationProps) {
    super({
      id: props.id ?? 'network-time-routing',
      inputs: {
        stateVertices: props.stateVertices,
        transitions: props.transitions,
        costs: props.costs,
        endpoints: props.endpoints,
        departure: props.departure
      },
      outputs: {
        arrivals: {value: props.output.arrivals, format: 'float32', length: props.stateCount},
        predecessorTransitions: {
          value: props.output.predecessorTransitions,
          format: 'uint32',
          length: props.stateCount
        },
        summary: {value: props.output.summary, format: 'uint32', length: 4},
        ...getNetworkOptimizationStatusOutputs(
          props.id ?? 'network-time-routing',
          props.output.status
        )
      },
      createContributor: getView =>
        new GPUNetworkTimeRouting({
          id: props.id,
          costModel: props.costModel,
          samplesPerTransition: props.samplesPerTransition,
          waitingPolicy: props.waitingPolicy,
          maximumRelaxations: props.maximumRelaxations,
          stateVertices: getView<'uint32'>('stateVertices'),
          transitions: getView<'uint32x2'>('transitions'),
          costs: getView<'float32x2'>('costs'),
          endpoints: getView<'uint32'>('endpoints'),
          departure: getView<'float32'>('departure'),
          output: {
            arrivals: getView<'float32'>('arrivals'),
            predecessorTransitions: getView<'uint32'>('predecessorTransitions'),
            summary: getView<'uint32'>('summary'),
            status: Object.fromEntries(
              Object.entries(props.output.status ?? {}).flatMap(([name, value]) =>
                value === undefined ? [] : [[name, getView<'uint32'>(name)]]
              )
            )
          }
        })
    });
  }
}
