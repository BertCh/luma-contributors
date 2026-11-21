// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getNetworkOptimizationStatusOutputs,
  type NetworkOptimizationStatusValues
} from './network-optimization-status';
import {
  snapshotNetworkTimeRoutingOperationProps,
  validateGPUNetworkTimeRoutingDeclarations,
  validateNetworkTimeRoutingResolvedRows
} from './network-time-routing-parameters';
import {
  GPUContributorOperation,
  type GPUContributorOperationProps
} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
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
    /** Relative seconds since departure, one per expanded state. */
    elapsedTimes?: string;
    predecessorTransitions: string;
    summary: string;
    status?: NetworkOptimizationStatusValues;
    /** CPU operation only: one representability word per state. */
    unrepresentable?: string;
  };
};

/** Shared descriptors without GPU-only validation. @internal */
export function getNetworkTimeRoutingOperationDescriptors(
  props: NetworkTimeRoutingOperationProps
): Pick<GPUContributorOperationProps, 'id' | 'inputs' | 'outputs'> {
  return {
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
      ...(props.output.elapsedTimes !== undefined
        ? {
            elapsedTimes: {
              value: props.output.elapsedTimes,
              format: 'float32' as const,
              length: props.stateCount
            }
          }
        : {}),
      predecessorTransitions: {
        value: props.output.predecessorTransitions,
        format: 'uint32',
        length: props.stateCount
      },
      summary: {value: props.output.summary, format: 'uint32', length: 4},
      ...(props.output.unrepresentable !== undefined
        ? {
            unrepresentable: {
              value: props.output.unrepresentable,
              format: 'uint32' as const,
              length: props.stateCount
            }
          }
        : {}),
      ...getNetworkOptimizationStatusOutputs(
        props.id ?? 'network-time-routing',
        props.output.status
      )
    }
  };
}

/**
 * GPU operation form. Every property is compile-time; graph contents are per-frame.
 * Declarations require 1–256 states, 0–4096 transitions, 0–65536 sweeps per solve and
 * 2–256 samples per piecewise transition. Resolved state/transition rows must match declarations.
 */
export class GPUNetworkTimeRoutingOperation extends GPUContributorOperation {
  constructor(originalProps: NetworkTimeRoutingOperationProps) {
    const props = snapshotNetworkTimeRoutingOperationProps(originalProps);
    const id = props.id ?? 'network-time-routing';
    validateGPUNetworkTimeRoutingDeclarations(id, props);
    validateRequirement(
      id,
      props.output.unrepresentable === undefined,
      'output.unrepresentable requires CPUNetworkTimeRoutingOperation'
    );
    super({
      ...getNetworkTimeRoutingOperationDescriptors(props),
      createContributor: getView => {
        const stateVertices = getView<'uint32'>('stateVertices');
        const transitions = getView<'uint32x2'>('transitions');
        validateNetworkTimeRoutingResolvedRows(id, props, stateVertices.length, transitions.length);
        return new GPUNetworkTimeRouting({
          id: props.id,
          costModel: props.costModel,
          samplesPerTransition: props.samplesPerTransition,
          waitingPolicy: props.waitingPolicy,
          maximumRelaxations: props.maximumRelaxations,
          stateVertices,
          transitions,
          costs: getView<'float32x2'>('costs'),
          endpoints: getView<'uint32'>('endpoints'),
          departure: getView<'float32'>('departure'),
          output: {
            arrivals: getView<'float32'>('arrivals'),
            ...(props.output.elapsedTimes !== undefined
              ? {elapsedTimes: getView<'float32'>('elapsedTimes')}
              : {}),
            predecessorTransitions: getView<'uint32'>('predecessorTransitions'),
            summary: getView<'uint32'>('summary'),
            status: Object.fromEntries(
              Object.entries(props.output.status ?? {}).flatMap(([name, value]) =>
                value === undefined ? [] : [[name, getView<'uint32'>(name)]]
              )
            )
          }
        });
      }
    });
  }
}
