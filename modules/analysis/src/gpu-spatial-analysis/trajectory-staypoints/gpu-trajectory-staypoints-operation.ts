// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUTrajectoryStaypoints,
  type GPUTrajectoryStaypointsProps
} from './gpu-trajectory-staypoints';

/** Graph-value names in place of the contributor's packed views. */
export type TrajectoryStaypointsOperationProps = Pick<
  GPUTrajectoryStaypointsProps,
  'id' | 'spatialContext' | 'temporalContext' | 'maximumSearchRows'
> & {
  positions: string;
  timestamps: string;
  /** Encoding of the timestamp graph value; defaults to float32. */
  timestampFormat?: 'float32' | 'uint32x2';
  trackOffsets: string;
  parameters: string;
  /** Explicit lengths formerly inferred from views. */
  rowCount: number;
  capacity: number;
  output: {
    ids: string;
    status: {count: string; overflow: string; requiredCount?: string; candidateOverflow?: string};
    drawInstanceCount?: string;
    startRows?: string;
    endRows?: string;
    finishRows?: string;
    centroids?: string;
    durations?: string;
    staypointIds?: string;
  };
};

/** GPU staypoints as a graph operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUTrajectoryStaypointsOperation extends GPUContributorOperation {
  constructor(props: TrajectoryStaypointsOperationProps) {
    const id = props.id ?? 'trajectory-staypoints';
    const {output, capacity, rowCount} = props;
    super({
      id,
      inputs: {
        positions: props.positions,
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
        parameters: props.parameters
      },
      outputs: {
        ids: {value: output.ids, format: 'uint32', length: capacity},
        count: {value: output.status.count, format: 'uint32', length: 1},
        overflow: {value: output.status.overflow, format: 'uint32', length: 1},
        ...Object.fromEntries(
          (['requiredCount', 'candidateOverflow'] as const).flatMap(port =>
            output.status[port]
              ? [[port, {value: output.status[port]!, format: 'uint32' as const, length: 1}]]
              : []
          )
        ),
        ...Object.fromEntries(
          (
            ['drawInstanceCount', 'startRows', 'endRows', 'finishRows', 'staypointIds'] as const
          ).flatMap(port =>
            output[port]
              ? [
                  [
                    port,
                    {
                      value: output[port]!,
                      format: 'uint32' as const,
                      length:
                        port === 'drawInstanceCount'
                          ? 1
                          : port === 'staypointIds'
                            ? rowCount
                            : capacity
                    }
                  ]
                ]
              : []
          )
        ),
        ...(output.centroids
          ? {centroids: {value: output.centroids, format: 'float32x2', length: capacity}}
          : {}),
        ...(output.durations
          ? {durations: {value: output.durations, format: 'float32', length: capacity}}
          : {})
      },
      createContributor: getView =>
        new GPUTrajectoryStaypoints({
          id,
          spatialContext: props.spatialContext,
          temporalContext: props.temporalContext,
          maximumSearchRows: props.maximumSearchRows,
          positions: getView<'float32x2'>('positions'),
          timestamps:
            props.timestampFormat === 'uint32x2'
              ? getView<'uint32x2'>('timestamps')
              : getView<'float32'>('timestamps'),
          trackOffsets: getView<'uint32'>('trackOffsets'),
          parameters: getView<'float32'>('parameters'),
          output: {
            ids: getView<'uint32'>('ids'),
            status: {
              count: getView<'uint32'>('count'),
              overflow: getView<'uint32'>('overflow'),
              requiredCount: output.status.requiredCount
                ? getView<'uint32'>('requiredCount')
                : undefined,
              candidateOverflow: output.status.candidateOverflow
                ? getView<'uint32'>('candidateOverflow')
                : undefined
            },
            drawInstanceCount: output.drawInstanceCount
              ? getView<'uint32'>('drawInstanceCount')
              : undefined,
            startRows: output.startRows ? getView<'uint32'>('startRows') : undefined,
            endRows: output.endRows ? getView<'uint32'>('endRows') : undefined,
            finishRows: output.finishRows ? getView<'uint32'>('finishRows') : undefined,
            centroids: output.centroids ? getView<'float32x2'>('centroids') : undefined,
            durations: output.durations ? getView<'float32'>('durations') : undefined,
            staypointIds: output.staypointIds ? getView<'uint32'>('staypointIds') : undefined
          }
        })
    });
  }
}
