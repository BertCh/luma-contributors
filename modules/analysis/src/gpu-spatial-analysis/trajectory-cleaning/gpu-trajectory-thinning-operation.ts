// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUTrajectoryThinning, type GPUTrajectoryThinningProps} from './gpu-trajectory-thinning';

/** Same graph values for the GPU contributor and uncapped CPU sequential walk. */
export type TrajectoryThinningOperationProps = Pick<
  GPUTrajectoryThinningProps,
  'id' | 'mode' | 'spatialContext' | 'temporalContext' | 'maximumSearchRows'
> & {
  positions: string;
  timestamps: string;
  trackOffsets: string;
  parameters: string;
  timestampFormat?: 'float32' | 'uint32x2';
  rowCount: number;
  trackCount: number;
  capacity: number;
  output: {
    rowIds: string;
    trackOffsets: string;
    status: {count: string; overflow: string; requiredCount?: string; candidateOverflow?: string};
    validity?: string;
    positions?: string;
    timestamps?: string;
  };
};

/** GPU thinning operation. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUTrajectoryThinningOperation extends GPUContributorOperation {
  constructor(props: TrajectoryThinningOperationProps) {
    const id = props.id ?? 'trajectory-thinning';
    const {output, capacity} = props;
    const timestampFormat = props.timestampFormat ?? 'float32';
    super({
      id,
      inputs: {
        positions: props.positions,
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
        parameters: props.parameters
      },
      outputs: {
        rowIds: {value: output.rowIds, format: 'uint32', length: capacity},
        outputTrackOffsets: {
          value: output.trackOffsets,
          format: 'uint32',
          length: props.trackCount + 1
        },
        count: {value: output.status.count, format: 'uint32', length: 1},
        overflow: {value: output.status.overflow, format: 'uint32', length: 1},
        ...(output.status.requiredCount !== undefined
          ? {requiredCount: {value: output.status.requiredCount, format: 'uint32', length: 1}}
          : {}),
        ...(output.status.candidateOverflow !== undefined
          ? {
              candidateOverflow: {
                value: output.status.candidateOverflow,
                format: 'uint32',
                length: 1
              }
            }
          : {}),
        ...(output.validity !== undefined
          ? {validity: {value: output.validity, format: 'uint32', length: props.rowCount}}
          : {}),
        ...(output.positions !== undefined
          ? {outputPositions: {value: output.positions, format: 'float32x2', length: capacity}}
          : {}),
        ...(output.timestamps !== undefined
          ? {
              outputTimestamps: {
                value: output.timestamps,
                format: timestampFormat,
                length: capacity
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUTrajectoryThinning({
          id,
          mode: props.mode,
          spatialContext: props.spatialContext,
          temporalContext: props.temporalContext,
          maximumSearchRows: props.maximumSearchRows,
          positions: getView<'float32x2'>('positions'),
          timestamps:
            timestampFormat === 'uint32x2'
              ? getView<'uint32x2'>('timestamps')
              : getView<'float32'>('timestamps'),
          trackOffsets: getView<'uint32'>('trackOffsets'),
          parameters: getView<'float32'>('parameters'),
          output: {
            rowIds: getView<'uint32'>('rowIds'),
            trackOffsets: getView<'uint32'>('outputTrackOffsets'),
            status: {
              count: getView<'uint32'>('count'),
              overflow: getView<'uint32'>('overflow'),
              requiredCount:
                output.status.requiredCount !== undefined
                  ? getView<'uint32'>('requiredCount')
                  : undefined,
              candidateOverflow:
                output.status.candidateOverflow !== undefined
                  ? getView<'uint32'>('candidateOverflow')
                  : undefined
            },
            validity: output.validity !== undefined ? getView<'uint32'>('validity') : undefined,
            positions:
              output.positions !== undefined ? getView<'float32x2'>('outputPositions') : undefined,
            timestamps:
              output.timestamps !== undefined
                ? timestampFormat === 'uint32x2'
                  ? getView<'uint32x2'>('outputTimestamps')
                  : getView<'float32'>('outputTimestamps')
                : undefined
          }
        })
    });
  }
}
