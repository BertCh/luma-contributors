// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUTrajectorySplit, type GPUTrajectorySplitProps} from './gpu-trajectory-split';

/** Graph names replace views; capacities and input row count are explicit. */
export type TrajectorySplitOperationProps = Pick<
  GPUTrajectorySplitProps,
  'id' | 'mode' | 'temporalContext'
> & {
  positions: string;
  timestamps: string;
  timestampFormat?: 'float32' | 'uint32x2';
  trackOffsets: string;
  parameters?: string;
  staypointIds?: string;
  staypointStatus?: {overflow?: string; candidateOverflow?: string};
  values?: string;
  rowCount: number;
  segmentCapacity: number;
  rowCapacity: number;
  output: {
    segmentOffsets: string;
    segmentTracks: string;
    rowIds: string;
    status: {count: string; overflow: string; requiredCount?: string; candidateOverflow?: string};
    positions?: string;
    timestamps?: string;
  };
};

/** GPU split operation. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUTrajectorySplitOperation extends GPUContributorOperation {
  constructor(props: TrajectorySplitOperationProps) {
    const {staypointStatus, output: destinations} = props;
    props = {
      ...props,
      staypointStatus:
        staypointStatus &&
        snapshotConstructionFields(staypointStatus, ['overflow', 'candidateOverflow']),
      output: {
        ...destinations,
        status: snapshotConstructionFields(destinations.status, [
          'count',
          'overflow',
          'requiredCount',
          'candidateOverflow'
        ])
      }
    };
    const id = props.id ?? 'trajectory-split';
    const {output, segmentCapacity, rowCapacity} = props;
    const {overflow, candidateOverflow} = props.staypointStatus ?? {};
    const hasStaypointStatus = overflow !== undefined || candidateOverflow !== undefined;
    validateRequirement(
      id,
      !hasStaypointStatus || props.mode === 'stop',
      "staypointStatus is supported in mode 'stop'"
    );
    validateRequirement(
      id,
      !hasStaypointStatus || Boolean(output.status.candidateOverflow),
      'output.status.candidateOverflow is required with staypointStatus; pass the name of a one-row uint32 graph value'
    );
    const timestampFormat = props.timestampFormat ?? 'float32';
    super({
      id,
      inputs: {
        positions: props.positions,
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
        ...(props.parameters !== undefined ? {parameters: props.parameters} : {}),
        ...(props.staypointIds !== undefined ? {staypointIds: props.staypointIds} : {}),
        ...(overflow !== undefined ? {stayOverflow: overflow} : {}),
        ...(candidateOverflow !== undefined ? {stayCandidateOverflow: candidateOverflow} : {}),
        ...(props.values !== undefined ? {values: props.values} : {})
      },
      outputs: {
        segmentOffsets: {
          value: output.segmentOffsets,
          format: 'uint32',
          length: segmentCapacity + 1
        },
        segmentTracks: {value: output.segmentTracks, format: 'uint32', length: segmentCapacity},
        rowIds: {value: output.rowIds, format: 'uint32', length: rowCapacity},
        count: {value: output.status.count, format: 'uint32', length: 1},
        overflow: {value: output.status.overflow, format: 'uint32', length: 1},
        ...(output.status.candidateOverflow !== undefined
          ? {
              candidateOverflow: {
                value: output.status.candidateOverflow,
                format: 'uint32',
                length: 1
              }
            }
          : {}),
        ...(output.status.requiredCount !== undefined
          ? {requiredCount: {value: output.status.requiredCount, format: 'uint32', length: 1}}
          : {}),
        ...(output.positions !== undefined
          ? {outputPositions: {value: output.positions, format: 'float32x2', length: rowCapacity}}
          : {}),
        ...(output.timestamps !== undefined
          ? {
              outputTimestamps: {
                value: output.timestamps,
                format: timestampFormat,
                length: rowCapacity
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUTrajectorySplit({
          id,
          mode: props.mode,
          temporalContext: props.temporalContext,
          positions: getView<'float32x2'>('positions'),
          timestamps:
            timestampFormat === 'uint32x2'
              ? getView<'uint32x2'>('timestamps')
              : getView<'float32'>('timestamps'),
          trackOffsets: getView<'uint32'>('trackOffsets'),
          parameters: props.parameters !== undefined ? getView<'float32'>('parameters') : undefined,
          staypointIds:
            props.staypointIds !== undefined ? getView<'uint32'>('staypointIds') : undefined,
          staypointStatus: {
            overflow: overflow !== undefined ? getView<'uint32'>('stayOverflow') : undefined,
            candidateOverflow:
              candidateOverflow !== undefined
                ? getView<'uint32'>('stayCandidateOverflow')
                : undefined
          },
          values: props.values !== undefined ? getView<'uint32'>('values') : undefined,
          output: {
            segmentOffsets: getView<'uint32'>('segmentOffsets'),
            segmentTracks: getView<'uint32'>('segmentTracks'),
            rowIds: getView<'uint32'>('rowIds'),
            status: {
              count: getView<'uint32'>('count'),
              overflow: getView<'uint32'>('overflow'),
              candidateOverflow:
                output.status.candidateOverflow !== undefined
                  ? getView<'uint32'>('candidateOverflow')
                  : undefined,
              requiredCount:
                output.status.requiredCount !== undefined
                  ? getView<'uint32'>('requiredCount')
                  : undefined
            },
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
