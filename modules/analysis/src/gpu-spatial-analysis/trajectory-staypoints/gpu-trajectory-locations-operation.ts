// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUTrajectoryLocations, type GPUTrajectoryLocationsProps} from './gpu-trajectory-locations';

/** Graph values in place of views, with the staypoint table capacity explicit. */
export type TrajectoryLocationsOperationProps = Pick<
  GPUTrajectoryLocationsProps,
  'id' | 'spatialContext' | 'maximumIterations'
> & {
  centroids: string;
  tracks: string;
  count: string;
  parameters: string;
  capacity: number;
  output: {
    locationIds: string;
    centers?: string;
    status: {count: string; converged?: string; iterationLimitReached?: string};
  };
};

/** GPU locations operation. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUTrajectoryLocationsOperation extends GPUContributorOperation {
  constructor(props: TrajectoryLocationsOperationProps) {
    const id = props.id ?? 'trajectory-locations';
    const {output, capacity} = props;
    super({
      id,
      inputs: {
        centroids: props.centroids,
        tracks: props.tracks,
        count: props.count,
        parameters: props.parameters
      },
      outputs: {
        locationIds: {value: output.locationIds, format: 'uint32', length: capacity},
        locationCount: {value: output.status.count, format: 'uint32', length: 1},
        ...(output.centers !== undefined
          ? {centers: {value: output.centers, format: 'float32x2', length: capacity}}
          : {}),
        ...(output.status.converged !== undefined
          ? {converged: {value: output.status.converged, format: 'uint32', length: 1}}
          : {}),
        ...(output.status.iterationLimitReached !== undefined
          ? {
              iterationLimitReached: {
                value: output.status.iterationLimitReached,
                format: 'uint32',
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUTrajectoryLocations({
          id,
          spatialContext: props.spatialContext,
          maximumIterations: props.maximumIterations,
          centroids: getView<'float32x2'>('centroids'),
          tracks: getView<'uint32'>('tracks'),
          count: getView<'uint32'>('count'),
          parameters: getView<'float32'>('parameters'),
          output: {
            locationIds: getView<'uint32'>('locationIds'),
            centers: output.centers !== undefined ? getView<'float32x2'>('centers') : undefined,
            status: {
              count: getView<'uint32'>('locationCount'),
              converged:
                output.status.converged !== undefined ? getView<'uint32'>('converged') : undefined,
              iterationLimitReached:
                output.status.iterationLimitReached !== undefined
                  ? getView<'uint32'>('iterationLimitReached')
                  : undefined
            }
          }
        })
    });
  }
}
