// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {NetworkTimeRoutingOperationProps} from './gpu-network-time-routing-operation';

/** Single-invocation GPU routing bounds; CPU twins retain their independent size contract. */
export const MAXIMUM_STATE_COUNT = 256;
export const MAXIMUM_TRANSITION_COUNT = 4096;
export const MAXIMUM_SAMPLES_PER_TRANSITION = 256;
export const MAXIMUM_RELAXATIONS = 65536;

type RoutingOptions = Pick<
  NetworkTimeRoutingOperationProps,
  'costModel' | 'samplesPerTransition' | 'waitingPolicy' | 'maximumRelaxations'
>;

/** Model and waiting-policy semantics; each form applies its own sample and sweep bounds. */
export function validateNetworkTimeRoutingOptions(id: string, props: RoutingOptions): void {
  validateRequirement(
    id,
    props.costModel === undefined ||
      props.costModel === 'affine-duration' ||
      props.costModel === 'piecewise-arrival',
    'costModel must be affine-duration or piecewise-arrival'
  );
  validateRequirement(
    id,
    props.waitingPolicy === undefined || props.waitingPolicy === 'wait-until-first-sample',
    'waitingPolicy must be wait-until-first-sample'
  );
  if (props.costModel !== 'piecewise-arrival')
    validateRequirement(
      id,
      props.samplesPerTransition === undefined && props.waitingPolicy === undefined,
      'samplesPerTransition and waitingPolicy require costModel piecewise-arrival'
    );
}

/** Validate declarations without requiring bound graph views or constructing a contributor. */
export function validateGPUNetworkTimeRoutingDeclarations(
  id: string,
  props: RoutingOptions & {stateCount: number; transitionCount: number}
): void {
  validateNetworkTimeRoutingOptions(id, props);
  validateCapacity(id, 'stateCount', props.stateCount, 1, MAXIMUM_STATE_COUNT);
  validateCapacity(id, 'transitionCount', props.transitionCount, 0, MAXIMUM_TRANSITION_COUNT);
  if (props.costModel === 'piecewise-arrival')
    validateCapacity(
      id,
      'samplesPerTransition',
      props.samplesPerTransition as number,
      2,
      MAXIMUM_SAMPLES_PER_TRANSITION
    );
  validateCapacity(
    id,
    'maximumRelaxations',
    props.maximumRelaxations ?? props.stateCount,
    0,
    MAXIMUM_RELAXATIONS
  );
}

/** Resolved view lengths are row counts, including the two-component transition format. */
export function validateNetworkTimeRoutingResolvedRows(
  id: string,
  props: {stateCount: number; transitionCount: number},
  stateRows: number,
  transitionRows: number
): void {
  validateRequirement(
    id,
    stateRows === props.stateCount,
    'stateVertices rows must equal stateCount'
  );
  validateRequirement(
    id,
    transitionRows === props.transitionCount,
    'transitions rows must equal transitionCount'
  );
}

/** Capture declarations, model choices and nested graph-value maps once. @internal */
export function snapshotNetworkTimeRoutingOperationProps(
  originalProps: NetworkTimeRoutingOperationProps
): NetworkTimeRoutingOperationProps {
  const settings = snapshotConstructionFields(originalProps, [
    'id',
    'stateCount',
    'transitionCount',
    'stateVertices',
    'transitions',
    'costs',
    'endpoints',
    'departure',
    'costModel',
    'samplesPerTransition',
    'waitingPolicy',
    'maximumRelaxations',
    'output'
  ]);
  const output = snapshotConstructionFields(settings.output, [
    'arrivals',
    'elapsedTimes',
    'predecessorTransitions',
    'summary',
    'status',
    'unrepresentable'
  ]);
  return Object.freeze({
    ...settings,
    output: Object.freeze({
      ...output,
      ...(output.status === undefined ? {} : {status: Object.freeze({...output.status})})
    })
  });
}
