// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUTrajectorySegments, type GPUTrajectorySegmentsProps} from './gpu-trajectory-segments';

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
    status: {count: string};
  };
  output: {
    triplegs: {
      ids: string;
      status: {count: string; overflow: string; requiredCount?: string};
      startRows?: string;
      endRows?: string;
      rowTriplegIds?: string;
    };
    trips?: {
      ids: string;
      status: {count: string; overflow: string; requiredCount?: string};
      startRows?: string;
      finishRows?: string;
      originStaypoints?: string;
      destinationStaypoints?: string;
      triplegTripIds?: string;
    };
  };
};

/** GPU segments operation; all settings except parameter contents are compile-time. */
export class GPUTrajectorySegmentsOperation extends GPUContributorOperation {
  constructor(props: TrajectorySegmentsOperationProps) {
    const id = props.id ?? 'trajectory-segments';
    const {triplegs, trips} = props.output;
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    const addOutput = (port: string, value: string | undefined, length: number) => {
      if (value !== undefined) outputs[port] = {value, format: 'uint32', length};
    };
    addOutput('triplegIds', triplegs.ids, props.triplegCapacity);
    addOutput('triplegCount', triplegs.status.count, 1);
    addOutput('triplegOverflow', triplegs.status.overflow, 1);
    addOutput('triplegRequiredCount', triplegs.status.requiredCount, 1);
    addOutput('triplegStartRows', triplegs.startRows, props.triplegCapacity);
    addOutput('triplegEndRows', triplegs.endRows, props.triplegCapacity);
    addOutput('rowTriplegIds', triplegs.rowTriplegIds, props.rowCount);
    if (trips) {
      const capacity = props.tripCapacity!;
      addOutput('tripIds', trips.ids, capacity);
      addOutput('tripCount', trips.status.count, 1);
      addOutput('tripOverflow', trips.status.overflow, 1);
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
              stayCount: props.staypoints.status.count
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
                status: {count: getView<'uint32'>('stayCount')}
              }
            : undefined,
          output: {
            triplegs: {
              ids: getView<'uint32'>('triplegIds'),
              status: {
                count: getView<'uint32'>('triplegCount'),
                overflow: getView<'uint32'>('triplegOverflow'),
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
