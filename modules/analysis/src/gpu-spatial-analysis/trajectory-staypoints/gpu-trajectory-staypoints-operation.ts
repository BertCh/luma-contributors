// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUTrajectoryStaypoints,
  type GPUTrajectoryStaypointsProps
} from './gpu-trajectory-staypoints';

/** Graph-value names in place of the contributor's packed views. */
export type TrajectoryStaypointsOperationProps = Pick<
  GPUTrajectoryStaypointsProps,
  'id' | 'spatialContext' | 'temporalContext' | 'maximumSearchRows' | 'missingFixPolicy'
> & {
  positions: string;
  mask?: string;
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
    status: {
      count: string;
      overflow: string;
      requiredCount?: string;
      candidateOverflow: string;
      missingFixCount?: string;
    };
    drawInstanceCount?: string;
    startRows?: string;
    endRows?: string;
    finishRows?: string;
    centroids?: string;
    durations?: string;
    staypointIds?: string;
  };
};

/** @internal Capture declared ports, including prototype getters, exactly once. */
export function snapshotTrajectoryStaypointsOperationProps(
  props: TrajectoryStaypointsOperationProps
): TrajectoryStaypointsOperationProps {
  props = snapshotConstructionFields(props, [
    'id',
    'spatialContext',
    'temporalContext',
    'maximumSearchRows',
    'missingFixPolicy',
    'positions',
    'mask',
    'timestamps',
    'timestampFormat',
    'trackOffsets',
    'parameters',
    'rowCount',
    'capacity',
    'output'
  ]);
  const output = snapshotConstructionFields(props.output, [
    'ids',
    'status',
    'drawInstanceCount',
    'startRows',
    'endRows',
    'finishRows',
    'centroids',
    'durations',
    'staypointIds'
  ]);
  return {
    ...props,
    spatialContext: props.spatialContext
      ? snapshotConstructionSpatialContext(props.spatialContext)
      : undefined,
    output: {
      ...output,
      status: snapshotConstructionFields(output.status ?? ({} as typeof output.status), [
        'count',
        'overflow',
        'requiredCount',
        'candidateOverflow',
        'missingFixCount'
      ])
    }
  };
}

/**
 * GPU staypoints operation. Bind `output.status.candidateOverflow` to a one-row uint32 value
 * for the search-loss flag. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 */
export class GPUTrajectoryStaypointsOperation extends GPUContributorOperation {
  constructor(props: TrajectoryStaypointsOperationProps) {
    props = snapshotTrajectoryStaypointsOperationProps(props);
    const id = props.id ?? 'trajectory-staypoints';
    validateRequirement(
      id,
      typeof props.output.status?.candidateOverflow === 'string' &&
        props.output.status.candidateOverflow.length > 0,
      `output.status.candidateOverflow is required; pass the name of a one-row uint32 graph value, received ${props.output.status?.candidateOverflow}`
    );
    validateRequirement(
      id,
      props.missingFixPolicy === undefined || props.missingFixPolicy === 'break',
      `missingFixPolicy must be break; received ${props.missingFixPolicy}`
    );
    const {output, capacity, rowCount} = props;
    super({
      id,
      emptyOutputPorts: ['staypointIds'],
      inputs: {
        positions: props.positions,
        ...(props.mask !== undefined ? {mask: props.mask} : {}),
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
        parameters: props.parameters
      },
      outputs: {
        ids: {value: output.ids, format: 'uint32', length: capacity},
        count: {value: output.status.count, format: 'uint32', length: 1},
        overflow: {value: output.status.overflow, format: 'uint32', length: 1},
        candidateOverflow: {value: output.status.candidateOverflow, format: 'uint32', length: 1},
        ...Object.fromEntries(
          (['requiredCount', 'missingFixCount'] as const).flatMap(port =>
            output.status[port] !== undefined
              ? [[port, {value: output.status[port]!, format: 'uint32' as const, length: 1}]]
              : []
          )
        ),
        ...Object.fromEntries(
          (
            ['drawInstanceCount', 'startRows', 'endRows', 'finishRows', 'staypointIds'] as const
          ).flatMap(port =>
            output[port] !== undefined
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
          missingFixPolicy: props.missingFixPolicy,
          mask: props.mask !== undefined ? getView<'uint32'>('mask') : undefined,
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
              missingFixCount: output.status.missingFixCount
                ? getView<'uint32'>('missingFixCount')
                : undefined,
              candidateOverflow: getView<'uint32'>('candidateOverflow')
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
