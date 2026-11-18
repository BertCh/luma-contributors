// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUTrajectorySegments, type GPUTrajectorySegmentsProps} from './gpu-trajectory-segments';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Graph-value form of tripleg and optional trip construction. */
export type TrajectorySegmentsOperationProps = Pick<
  GPUTrajectorySegmentsProps,
  'id' | 'temporalContext'
> & {
  positions: string;
  timestamps: string;
  timestampFormat?: 'float32' | 'uint32x2';
  trackOffsets: string;
  staypointIds: string;
  parameters: string;
  rowCount: number;
  triplegCapacity: number;
  /** Required when output.trips is present. */
  tripCapacity?: number;
  staypoints?: {
    tracks: string;
    startRows: string;
    finishRows: string;
    durations: string;
    status: {count: string; overflow?: string; candidateOverflow?: string};
  };
  output: {
    triplegs: {
      ids: string;
      status: {count: string; overflow: string; requiredCount?: string; candidateOverflow?: string};
      startRows?: string;
      endRows?: string;
      rowTriplegIds?: string;
    };
    trips?: {
      ids: string;
      status: {count: string; overflow: string; requiredCount?: string; candidateOverflow?: string};
      startRows?: string;
      finishRows?: string;
      originStaypoints?: string;
      destinationStaypoints?: string;
      triplegTripIds?: string;
    };
  };
};

/** @internal Freeze validated port maps and settings shared by both operation forms. */
export function snapshotTrajectorySegmentsOperationProps(
  props: TrajectorySegmentsOperationProps
): TrajectorySegmentsOperationProps {
  props = snapshotConstructionFields(props, [
    'id',
    'temporalContext',
    'positions',
    'timestamps',
    'timestampFormat',
    'trackOffsets',
    'staypointIds',
    'parameters',
    'rowCount',
    'triplegCapacity',
    'tripCapacity',
    'staypoints',
    'output'
  ]);
  const table = <Table extends TrajectorySegmentsOperationProps['output']['triplegs']>(
    value: Table
  ): Table => {
    const captured = snapshotConstructionFields(value, [
      'ids',
      'status',
      'startRows',
      'endRows',
      'rowTriplegIds',
      'finishRows',
      'originStaypoints',
      'destinationStaypoints',
      'triplegTripIds'
    ] as (keyof Table)[]);
    return {
      ...captured,
      status: snapshotConstructionFields(captured.status, [
        'count',
        'overflow',
        'requiredCount',
        'candidateOverflow'
      ])
    };
  };
  const output = snapshotConstructionFields(props.output, ['triplegs', 'trips']);
  const staypoints =
    props.staypoints &&
    snapshotConstructionFields(props.staypoints, [
      'tracks',
      'startRows',
      'finishRows',
      'durations',
      'status'
    ]);
  const captured = {
    ...props,
    output: {triplegs: table(output.triplegs), trips: output.trips && table(output.trips)},
    staypoints: staypoints && {
      ...staypoints,
      status: snapshotConstructionFields(staypoints.status, [
        'count',
        'overflow',
        'candidateOverflow'
      ])
    }
  };
  const id = captured.id ?? 'trajectory-segments';
  validateRequirement(
    id,
    !captured.output.trips || captured.output.trips.status.candidateOverflow !== undefined,
    'output.trips.status.candidateOverflow is required because triplegs can be truncated'
  );
  validateRequirement(
    id,
    !(
      captured.staypoints?.status.overflow !== undefined ||
      captured.staypoints?.status.candidateOverflow !== undefined
    ) || captured.output.triplegs.status.candidateOverflow !== undefined,
    'output.triplegs.status.candidateOverflow is required with supplied staypoint loss'
  );
  return captured;
}

/** GPU segments operation; all settings except parameter contents are compile-time. */
export class GPUTrajectorySegmentsOperation extends GPUContributorOperation {
  constructor(props: TrajectorySegmentsOperationProps) {
    props = snapshotTrajectorySegmentsOperationProps(props);
    const id = props.id ?? 'trajectory-segments';
    const {triplegs, trips} = props.output;
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    const addOutput = (port: string, value: string | undefined, length: number) => {
      if (value !== undefined) outputs[port] = {value, format: 'uint32', length};
    };
    addOutput('triplegIds', triplegs.ids, props.triplegCapacity);
    addOutput('triplegCount', triplegs.status.count, 1);
    addOutput('triplegOverflow', triplegs.status.overflow, 1);
    addOutput('triplegCandidateOverflow', triplegs.status.candidateOverflow, 1);
    addOutput('triplegRequiredCount', triplegs.status.requiredCount, 1);
    addOutput('triplegStartRows', triplegs.startRows, props.triplegCapacity);
    addOutput('triplegEndRows', triplegs.endRows, props.triplegCapacity);
    addOutput('rowTriplegIds', triplegs.rowTriplegIds, props.rowCount);
    if (trips) {
      const capacity = props.tripCapacity!;
      addOutput('tripIds', trips.ids, capacity);
      addOutput('tripCount', trips.status.count, 1);
      addOutput('tripOverflow', trips.status.overflow, 1);
      addOutput('tripCandidateOverflow', trips.status.candidateOverflow, 1);
      addOutput('tripRequiredCount', trips.status.requiredCount, 1);
      addOutput('tripStartRows', trips.startRows, capacity);
      addOutput('tripFinishRows', trips.finishRows, capacity);
      addOutput('originStaypoints', trips.originStaypoints, capacity);
      addOutput('destinationStaypoints', trips.destinationStaypoints, capacity);
      addOutput('triplegTripIds', trips.triplegTripIds, props.triplegCapacity);
    }
    super({
      id,
      inputs: {
        positions: props.positions,
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
        staypointIds: props.staypointIds,
        parameters: props.parameters,
        ...(props.staypoints
          ? {
              stayTracks: props.staypoints.tracks,
              stayStartRows: props.staypoints.startRows,
              stayFinishRows: props.staypoints.finishRows,
              stayDurations: props.staypoints.durations,
              stayCount: props.staypoints.status.count,
              ...(props.staypoints.status.overflow
                ? {stayOverflow: props.staypoints.status.overflow}
                : {}),
              ...(props.staypoints.status.candidateOverflow
                ? {stayCandidateOverflow: props.staypoints.status.candidateOverflow}
                : {})
            }
          : {})
      },
      outputs,
      createContributor: getView => {
        const optionalView = (port: string) =>
          outputs[port] ? getView<'uint32'>(port) : undefined;
        return new GPUTrajectorySegments({
          id,
          temporalContext: props.temporalContext,
          positions: getView<'float32x2'>('positions'),
          timestamps:
            props.timestampFormat === 'uint32x2'
              ? getView<'uint32x2'>('timestamps')
              : getView<'float32'>('timestamps'),
          trackOffsets: getView<'uint32'>('trackOffsets'),
          staypointIds: getView<'uint32'>('staypointIds'),
          parameters: getView<'float32'>('parameters'),
          staypoints: props.staypoints
            ? {
                tracks: getView<'uint32'>('stayTracks'),
                startRows: getView<'uint32'>('stayStartRows'),
                finishRows: getView<'uint32'>('stayFinishRows'),
                durations: getView<'float32'>('stayDurations'),
                status: {
                  count: getView<'uint32'>('stayCount'),
                  overflow: props.staypoints.status.overflow
                    ? getView<'uint32'>('stayOverflow')
                    : undefined,
                  candidateOverflow: props.staypoints.status.candidateOverflow
                    ? getView<'uint32'>('stayCandidateOverflow')
                    : undefined
                }
              }
            : undefined,
          output: {
            triplegs: {
              ids: getView<'uint32'>('triplegIds'),
              status: {
                count: getView<'uint32'>('triplegCount'),
                overflow: getView<'uint32'>('triplegOverflow'),
                candidateOverflow: optionalView('triplegCandidateOverflow'),
                requiredCount: optionalView('triplegRequiredCount')
              },
              startRows: optionalView('triplegStartRows'),
              endRows: optionalView('triplegEndRows'),
              rowTriplegIds: optionalView('rowTriplegIds')
            },
            trips: trips
              ? {
                  ids: getView<'uint32'>('tripIds'),
                  status: {
                    count: getView<'uint32'>('tripCount'),
                    overflow: getView<'uint32'>('tripOverflow'),
                    candidateOverflow: optionalView('tripCandidateOverflow'),
                    requiredCount: optionalView('tripRequiredCount')
                  },
                  startRows: optionalView('tripStartRows'),
                  finishRows: optionalView('tripFinishRows'),
                  originStaypoints: optionalView('originStaypoints'),
                  destinationStaypoints: optionalView('destinationStaypoints'),
                  triplegTripIds: optionalView('triplegTripIds')
                }
              : undefined
          }
        });
      }
    });
  }
}
