// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUMobilityIndicators, type GPUMobilityIndicatorsProps} from './gpu-mobility-indicators';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Graph-value names and explicit input lengths shared by both operation classes. */
export type MobilityIndicatorsOperationProps = Pick<
  GPUMobilityIndicatorsProps,
  'id' | 'spatialContext' | 'trackCount' | 'maximumEntropySequenceRows'
> & {
  timestamps: string;
  timestampFormat?: 'float32' | 'uint32x2';
  rowCount: number;
  staypointCapacity: number;
  staypoints: {
    tracks: string;
    centroids: string;
    startRows: string;
    durations: string;
    locationIds: string;
    status: {count: string; overflow?: string; candidateOverflow?: string};
  };
  triplegCapacity?: number;
  triplegs?: {
    tracks: string;
    startRows: string;
    endRows: string;
    status: {count: string; overflow?: string; candidateOverflow?: string};
    staypointFinishRows: string;
  };
  output: {
    radiusOfGyration?: string;
    jumpLengths?: string;
    waitingTimes?: string;
    randomEntropy?: string;
    uncorrelatedEntropy?: string;
    realEntropy?: string;
    homeLocations?: string;
    workLocations?: string;
    trackingQuality?: string;
    status?: {candidateOverflow?: string};
  };
};

/** @internal Capture settings and nested maps shared by both operation forms. */
export function snapshotMobilityIndicatorsOperationProps(
  props: MobilityIndicatorsOperationProps
): MobilityIndicatorsOperationProps {
  props = snapshotConstructionFields(props, [
    'id',
    'spatialContext',
    'trackCount',
    'maximumEntropySequenceRows',
    'timestamps',
    'timestampFormat',
    'rowCount',
    'staypointCapacity',
    'staypoints',
    'triplegCapacity',
    'triplegs',
    'output'
  ]);
  const stays = snapshotConstructionFields(props.staypoints, [
    'tracks',
    'centroids',
    'startRows',
    'durations',
    'locationIds',
    'status'
  ]);
  const legs =
    props.triplegs &&
    snapshotConstructionFields(props.triplegs, [
      'tracks',
      'startRows',
      'endRows',
      'status',
      'staypointFinishRows'
    ]);
  const output = snapshotConstructionFields(props.output, [
    'radiusOfGyration',
    'jumpLengths',
    'waitingTimes',
    'randomEntropy',
    'uncorrelatedEntropy',
    'realEntropy',
    'homeLocations',
    'workLocations',
    'trackingQuality',
    'status'
  ]);
  return {
    ...props,
    staypoints: {
      ...stays,
      status: snapshotConstructionFields(stays.status, ['count', 'overflow', 'candidateOverflow'])
    },
    triplegs: legs && {
      ...legs,
      status: snapshotConstructionFields(legs.status, ['count', 'overflow', 'candidateOverflow'])
    },
    output: {
      ...output,
      status: output.status && snapshotConstructionFields(output.status, ['candidateOverflow'])
    }
  };
}

export class GPUMobilityIndicatorsOperation extends GPUContributorOperation {
  constructor(props: MobilityIndicatorsOperationProps) {
    props = snapshotMobilityIndicatorsOperationProps(props);
    const id = props.id ?? 'mobility-indicators';
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    for (const port of [
      'radiusOfGyration',
      'jumpLengths',
      'waitingTimes',
      'randomEntropy',
      'uncorrelatedEntropy',
      'realEntropy',
      'homeLocations',
      'workLocations',
      'trackingQuality'
    ] as const) {
      const value = props.output[port];
      if (value !== undefined)
        outputs[port] = {
          value,
          format: port === 'homeLocations' || port === 'workLocations' ? 'uint32' : 'float32',
          length:
            port === 'jumpLengths' || port === 'waitingTimes'
              ? props.staypointCapacity
              : props.trackCount
        };
    }
    if (props.output.status?.candidateOverflow !== undefined)
      outputs['candidateOverflow'] = {
        value: props.output.status.candidateOverflow,
        format: 'uint32',
        length: 1
      };
    super({
      id,
      inputs: {
        timestamps: props.timestamps,
        stayTracks: props.staypoints.tracks,
        centroids: props.staypoints.centroids,
        stayStartRows: props.staypoints.startRows,
        durations: props.staypoints.durations,
        locationIds: props.staypoints.locationIds,
        stayCount: props.staypoints.status.count,
        ...(props.staypoints.status.overflow !== undefined
          ? {stayOverflow: props.staypoints.status.overflow}
          : {}),
        ...(props.staypoints.status.candidateOverflow !== undefined
          ? {stayCandidateOverflow: props.staypoints.status.candidateOverflow}
          : {}),
        ...(props.triplegs
          ? {
              triplegTracks: props.triplegs.tracks,
              triplegStartRows: props.triplegs.startRows,
              triplegEndRows: props.triplegs.endRows,
              triplegCount: props.triplegs.status.count,
              ...(props.triplegs.status.overflow !== undefined
                ? {triplegOverflow: props.triplegs.status.overflow}
                : {}),
              ...(props.triplegs.status.candidateOverflow !== undefined
                ? {triplegCandidateOverflow: props.triplegs.status.candidateOverflow}
                : {}),
              stayFinishRows: props.triplegs.staypointFinishRows
            }
          : {})
      },
      outputs,
      createContributor: getView => {
        const optionalFloatView = (port: string) =>
          outputs[port] ? getView<'float32'>(port) : undefined;
        const optionalIntegerView = (port: string) =>
          outputs[port] ? getView<'uint32'>(port) : undefined;
        return new GPUMobilityIndicators({
          id,
          spatialContext: props.spatialContext,
          trackCount: props.trackCount,
          maximumEntropySequenceRows: props.maximumEntropySequenceRows,
          timestamps:
            props.timestampFormat === 'uint32x2'
              ? getView<'uint32x2'>('timestamps')
              : getView<'float32'>('timestamps'),
          staypoints: {
            tracks: getView<'uint32'>('stayTracks'),
            centroids: getView<'float32x2'>('centroids'),
            startRows: getView<'uint32'>('stayStartRows'),
            durations: getView<'float32'>('durations'),
            locationIds: getView<'uint32'>('locationIds'),
            status: {
              count: getView<'uint32'>('stayCount'),
              overflow:
                props.staypoints.status.overflow !== undefined
                  ? getView<'uint32'>('stayOverflow')
                  : undefined,
              candidateOverflow:
                props.staypoints.status.candidateOverflow !== undefined
                  ? getView<'uint32'>('stayCandidateOverflow')
                  : undefined
            }
          },
          triplegs: props.triplegs
            ? {
                tracks: getView<'uint32'>('triplegTracks'),
                startRows: getView<'uint32'>('triplegStartRows'),
                endRows: getView<'uint32'>('triplegEndRows'),
                status: {
                  count: getView<'uint32'>('triplegCount'),
                  overflow:
                    props.triplegs.status.overflow !== undefined
                      ? getView<'uint32'>('triplegOverflow')
                      : undefined,
                  candidateOverflow:
                    props.triplegs.status.candidateOverflow !== undefined
                      ? getView<'uint32'>('triplegCandidateOverflow')
                      : undefined
                },
                staypointFinishRows: getView<'uint32'>('stayFinishRows')
              }
            : undefined,
          output: {
            radiusOfGyration: optionalFloatView('radiusOfGyration'),
            jumpLengths: optionalFloatView('jumpLengths'),
            waitingTimes: optionalFloatView('waitingTimes'),
            randomEntropy: optionalFloatView('randomEntropy'),
            uncorrelatedEntropy: optionalFloatView('uncorrelatedEntropy'),
            realEntropy: optionalFloatView('realEntropy'),
            homeLocations: optionalIntegerView('homeLocations'),
            workLocations: optionalIntegerView('workLocations'),
            trackingQuality: optionalFloatView('trackingQuality'),
            status: {candidateOverflow: optionalIntegerView('candidateOverflow')}
          }
        });
      }
    });
  }
}
