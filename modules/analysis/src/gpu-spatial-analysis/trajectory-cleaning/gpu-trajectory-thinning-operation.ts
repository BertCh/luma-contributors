// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotThinningOutput} from './thinning-output-aliases';
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
    sourceRows: string;
    sourceSelectionMask?: string;
    trackOffsets: string;
    status: {count: string; overflow: string; requiredCount?: string; candidateOverflow: string};
    positions?: string;
    timestamps?: string;
  };
};

/**
 * GPU thinning operation. Bind `output.status.candidateOverflow` to a one-row uint32 value
 * for the search-loss flag. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 */
export class GPUTrajectoryThinningOperation extends GPUContributorOperation {
  constructor(props: TrajectoryThinningOperationProps) {
    const id = props.id ?? 'trajectory-thinning';
    const output = snapshotThinningOutput(id, props.output);
    validateRequirement(
      id,
      typeof output.status?.candidateOverflow === 'string' &&
        output.status.candidateOverflow.length > 0,
      `output.status.candidateOverflow is required; pass the name of a one-row uint32 graph value, received ${output.status?.candidateOverflow}`
    );
    props = {...props, output};
    const {capacity} = props;
    const timestampFormat = props.timestampFormat ?? 'float32';
    super({
      id,
      emptyOutputPorts: ['sourceSelectionMask'],
      inputs: {
        positions: props.positions,
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
        parameters: props.parameters
      },
      outputs: {
        sourceRows: {value: output.sourceRows, format: 'uint32', length: capacity},
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
        candidateOverflow: {value: output.status.candidateOverflow, format: 'uint32', length: 1},
        ...(output.sourceSelectionMask !== undefined
          ? {
              sourceSelectionMask: {
                value: output.sourceSelectionMask,
                format: 'uint32',
                length: props.rowCount
              }
            }
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
            sourceRows: getView<'uint32'>('sourceRows'),
            trackOffsets: getView<'uint32'>('outputTrackOffsets'),
            status: {
              count: getView<'uint32'>('count'),
              overflow: getView<'uint32'>('overflow'),
              requiredCount:
                output.status.requiredCount !== undefined
                  ? getView<'uint32'>('requiredCount')
                  : undefined,
              candidateOverflow: getView<'uint32'>('candidateOverflow')
            },
            sourceSelectionMask:
              output.sourceSelectionMask !== undefined
                ? getView<'uint32'>('sourceSelectionMask')
                : undefined,
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
