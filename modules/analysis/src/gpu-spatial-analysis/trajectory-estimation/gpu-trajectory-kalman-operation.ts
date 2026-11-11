// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUTrajectoryKalman, type GPUTrajectoryKalmanProps} from './gpu-trajectory-kalman';

/** Graph-value names replacing Kalman views. Output lengths are explicit observation counts. */
export type TrajectoryKalmanOperationProps = Omit<
  GPUTrajectoryKalmanProps,
  'positions' | 'times' | 'offsets' | 'timestamps' | 'trackOffsets' | 'parameters' | 'output'
> & {
  positions: string;
  parameters: string;
  rowCount: number;
  timestampFormat?: 'float32' | 'uint32x2';
  output: {
    filteredState: string;
    filteredCovariance: string;
    status: string;
    smoothedState?: string;
    smoothedCovariance?: string;
  };
} & (
    | {timestamps: string; trackOffsets: string; times?: never; offsets?: never}
    | {times: string; offsets: string; timestamps?: never; trackOffsets?: never}
  );

/**
 * Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. Inputs are per-frame
 * vectors; all other props are compile-time. CPUTrajectoryKalmanOperation is its float64 twin.
 * Contributor validation runs when the graph resolves its views.
 */
export class GPUTrajectoryKalmanOperation extends GPUContributorOperation {
  constructor(props: TrajectoryKalmanOperationProps) {
    const covarianceLength = props.rowCount * (props.covarianceMode === 'full' ? 4 : 1);
    super({
      id: props.id ?? 'trajectory-kalman',
      inputs: {
        positions: props.positions,
        timestamps: (props.timestamps ?? props.times)!,
        trackOffsets: (props.trackOffsets ?? props.offsets)!,
        parameters: props.parameters
      },
      outputs: {
        filteredState: {
          value: props.output.filteredState,
          format: 'float32x4',
          length: props.rowCount
        },
        filteredCovariance: {
          value: props.output.filteredCovariance,
          format: 'float32x4',
          length: covarianceLength
        },
        status: {value: props.output.status, format: 'uint32', length: props.rowCount},
        ...(props.output.smoothedState === undefined
          ? {}
          : {
              smoothedState: {
                value: props.output.smoothedState,
                format: 'float32x4' as const,
                length: props.rowCount
              }
            }),
        ...(props.output.smoothedCovariance === undefined
          ? {}
          : {
              smoothedCovariance: {
                value: props.output.smoothedCovariance,
                format: 'float32x4' as const,
                length: covarianceLength
              }
            })
      },
      createContributor: getView =>
        new GPUTrajectoryKalman({
          id: props.id,
          spatialContext: props.spatialContext,
          temporalContext: props.temporalContext,
          timeUnitsPerSecond: props.timeUnitsPerSecond,
          covarianceMode: props.covarianceMode,
          positions: getView<'float32x2'>('positions'),
          timestamps:
            props.timestampFormat === 'uint32x2'
              ? getView<'uint32x2'>('timestamps')
              : getView<'float32'>('timestamps'),
          trackOffsets: getView<'uint32'>('trackOffsets'),
          parameters: getView<'float32'>('parameters'),
          output: {
            filteredState: getView<'float32x4'>('filteredState'),
            filteredCovariance: getView<'float32x4'>('filteredCovariance'),
            status: getView<'uint32'>('status'),
            smoothedState:
              props.output.smoothedState === undefined
                ? undefined
                : getView<'float32x4'>('smoothedState'),
            smoothedCovariance:
              props.output.smoothedCovariance === undefined
                ? undefined
                : getView<'float32x4'>('smoothedCovariance')
          }
        })
    });
  }
}
