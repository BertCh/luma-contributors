// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUTrajectoryLocations,
  GPU_TRAJECTORY_LOCATIONS_MAXIMUM_ITERATIONS,
  type GPUTrajectoryLocationsProps
} from './gpu-trajectory-locations';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';

/** Graph values in place of views, with the staypoint table capacity explicit. */
export type TrajectoryLocationsOperationProps = Pick<
  GPUTrajectoryLocationsProps,
  'id' | 'spatialContext' | 'maximumIterations'
> & {
  centroids: string;
  tracks: string;
  count: string;
  staypointStatus?: {overflow?: string; candidateOverflow?: string};
  parameters: string;
  capacity: number;
  output: {
    locationIds: string;
    centers?: string;
    status: {
      count: string;
      iterationCount?: string;
      converged?: string;
      complete?: string;
      iterationLimitReached: string;
    };
  };
};

/** @internal Snapshot graph value names without retaining caller-owned port maps. */
export function snapshotTrajectoryLocationsOperationProps(
  props: TrajectoryLocationsOperationProps
): TrajectoryLocationsOperationProps {
  props = snapshotConstructionFields(props, [
    'id',
    'spatialContext',
    'maximumIterations',
    'centroids',
    'tracks',
    'count',
    'staypointStatus',
    'parameters',
    'capacity',
    'output'
  ]);
  const output = snapshotConstructionFields(props.output, ['locationIds', 'centers', 'status']);
  return Object.freeze({
    ...props,
    staypointStatus: props.staypointStatus
      ? snapshotConstructionFields(props.staypointStatus, ['overflow', 'candidateOverflow'])
      : undefined,
    spatialContext: props.spatialContext
      ? snapshotConstructionSpatialContext(props.spatialContext)
      : undefined,
    output: Object.freeze({
      ...output,
      status: snapshotConstructionFields(output.status ?? ({} as typeof output.status), [
        'count',
        'iterationCount',
        'converged',
        'complete',
        'iterationLimitReached'
      ])
    })
  });
}

/**
 * GPU locations operation. Bind `output.status.iterationLimitReached` to a one-row uint32
 * value for solver exhaustion. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 */
export class GPUTrajectoryLocationsOperation extends GPUContributorOperation {
  constructor(props: TrajectoryLocationsOperationProps) {
    props = snapshotTrajectoryLocationsOperationProps(props);
    const id = props.id ?? 'trajectory-locations';
    validateRequirement(
      id,
      typeof props.output.status?.iterationLimitReached === 'string' &&
        props.output.status.iterationLimitReached.length > 0,
      `output.status.iterationLimitReached is required; pass the name of a one-row uint32 graph value, received ${props.output.status?.iterationLimitReached}`
    );
    if (props.maximumIterations !== undefined)
      validateCapacity(
        id,
        'maximumIterations',
        props.maximumIterations,
        0,
        GPU_TRAJECTORY_LOCATIONS_MAXIMUM_ITERATIONS
      );
    const {output, capacity} = props;
    super({
      id,
      inputs: {
        centroids: props.centroids,
        tracks: props.tracks,
        count: props.count,
        ...(props.staypointStatus?.overflow !== undefined
          ? {stayOverflow: props.staypointStatus.overflow}
          : {}),
        ...(props.staypointStatus?.candidateOverflow !== undefined
          ? {stayCandidateOverflow: props.staypointStatus.candidateOverflow}
          : {}),
        parameters: props.parameters
      },
      outputs: {
        ...(output.status.complete !== undefined
          ? {complete: {value: output.status.complete, format: 'uint32' as const, length: 1}}
          : {}),
        locationIds: {value: output.locationIds, format: 'uint32', length: capacity},
        locationCount: {value: output.status.count, format: 'uint32', length: 1},
        ...(output.status.iterationCount !== undefined
          ? {iterationCount: {value: output.status.iterationCount, format: 'uint32', length: 1}}
          : {}),
        ...(output.centers !== undefined
          ? {centers: {value: output.centers, format: 'float32x2', length: capacity}}
          : {}),
        ...(output.status.converged !== undefined
          ? {converged: {value: output.status.converged, format: 'uint32', length: 1}}
          : {}),
        iterationLimitReached: {
          value: output.status.iterationLimitReached,
          format: 'uint32',
          length: 1
        }
      },
      createContributor: getView =>
        new GPUTrajectoryLocations({
          id,
          spatialContext: props.spatialContext,
          maximumIterations: props.maximumIterations,
          centroids: getView<'float32x2'>('centroids'),
          tracks: getView<'uint32'>('tracks'),
          count: getView<'uint32'>('count'),
          staypointStatus: {
            overflow:
              props.staypointStatus?.overflow !== undefined
                ? getView<'uint32'>('stayOverflow')
                : undefined,
            candidateOverflow:
              props.staypointStatus?.candidateOverflow !== undefined
                ? getView<'uint32'>('stayCandidateOverflow')
                : undefined
          },
          parameters: getView<'float32'>('parameters'),
          output: {
            locationIds: getView<'uint32'>('locationIds'),
            centers: output.centers !== undefined ? getView<'float32x2'>('centers') : undefined,
            status: {
              count: getView<'uint32'>('locationCount'),
              complete:
                output.status.complete !== undefined ? getView<'uint32'>('complete') : undefined,
              iterationCount:
                output.status.iterationCount !== undefined
                  ? getView<'uint32'>('iterationCount')
                  : undefined,
              converged:
                output.status.converged !== undefined ? getView<'uint32'>('converged') : undefined,
              iterationLimitReached: getView<'uint32'>('iterationLimitReached')
            }
          }
        })
    });
  }
}
