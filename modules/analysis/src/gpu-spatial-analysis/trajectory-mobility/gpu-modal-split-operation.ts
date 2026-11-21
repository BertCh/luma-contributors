// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUModalSplit, type GPUModalSplitProps} from './gpu-modal-split';
import {resolveModalSpeedOptions, validateModalSpeedOptionName} from './modal-split-options';
import {
  snapshotConstructionSpatialContext,
  snapshotConstructionFields
} from '../../analysis-core/kernels/construction-snapshot';

/**
 * Graph-value form. In prediction mode the optional input modes port is omitted: the contributor
 * would write it in place. Its writable modes table is instead the output or transient scratch.
 */
export type ModalSplitOperationProps = Pick<
  GPUModalSplitProps,
  'id' | 'metric' | 'modeCount' | 'spatialContext' | 'temporalContext' | 'unknownModePolicy'
> & {
  positions: string;
  timestamps: string;
  trackOffsets: string;
  timestampFormat?: 'float32' | 'uint32x2';
  rowCount: number;
  trackCount: number;
  triplegCapacity: number;
  triplegs: {
    tracks: string;
    startRows: string;
    endRows: string;
    status: {count: string; overflow?: string; candidateOverflow?: string};
  };
  output: {modalSplit: string; predictedModes?: string};
} & (
    | {modes: string; modeSpeedUpperBoundsFloat32?: never}
    | {modes?: never; modeSpeedUpperBoundsFloat32: readonly number[]}
  );

/** @internal Capture compile-time settings and nested port maps for both operation forms. */
export function snapshotModalSplitOperationProps(
  input: ModalSplitOperationProps
): ModalSplitOperationProps {
  const props = snapshotConstructionFields(input, [
    'id',
    'metric',
    'modeCount',
    'spatialContext',
    'temporalContext',
    'positions',
    'timestamps',
    'trackOffsets',
    'timestampFormat',
    'rowCount',
    'trackCount',
    'triplegCapacity',
    'triplegs',
    'modes',
    'modeSpeedUpperBoundsFloat32',
    'unknownModePolicy',
    'output'
  ]);
  validateModalSpeedOptionName(props.id ?? 'modal-split', input);
  const triplegs = snapshotConstructionFields(props.triplegs, [
    'tracks',
    'startRows',
    'endRows',
    'status'
  ]);
  const captured = {
    ...props,
    spatialContext: props.spatialContext
      ? snapshotConstructionSpatialContext(props.spatialContext)
      : undefined,
    triplegs: {
      ...triplegs,
      status: snapshotConstructionFields(triplegs.status, [
        'count',
        'overflow',
        'candidateOverflow'
      ])
    },
    output: snapshotConstructionFields(props.output, ['modalSplit', 'predictedModes'])
  };
  const bounds = resolveModalSpeedOptions(props.id ?? 'modal-split', props);
  return bounds
    ? {...captured, modes: undefined, modeSpeedUpperBoundsFloat32: bounds}
    : {
        ...captured,
        modes: props.modes!,
        modeSpeedUpperBoundsFloat32: undefined
      };
}

/** GPU modal aggregation operation. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUModalSplitOperation extends GPUContributorOperation {
  constructor(props: ModalSplitOperationProps) {
    props = snapshotModalSplitOperationProps(props);
    const id = props.id ?? 'modal-split';
    const predictedModes = props.output.predictedModes;
    super({
      id,
      emptyOutputPorts: ['modalSplit', 'predictedModes'],
      inputs: {
        positions: props.positions,
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
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
        ...(!props.modeSpeedUpperBoundsFloat32 && props.modes !== undefined
          ? {modes: props.modes}
          : {})
      },
      outputs: {
        modalSplit: {
          value: props.output.modalSplit,
          format: 'float32',
          length: props.trackCount * props.modeCount
        },
        ...(predictedModes !== undefined
          ? {
              predictedModes: {
                value: predictedModes,
                format: 'uint32',
                length: props.triplegCapacity
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUModalSplit({
          id,
          spatialContext: props.spatialContext,
          temporalContext: props.temporalContext,
          metric: props.metric,
          modeCount: props.modeCount,
          modeSpeedUpperBoundsFloat32: props.modeSpeedUpperBoundsFloat32,
          unknownModePolicy: props.unknownModePolicy,
          positions: getView<'float32x2'>('positions'),
          timestamps:
            props.timestampFormat === 'uint32x2'
              ? getView<'uint32x2'>('timestamps')
              : getView<'float32'>('timestamps'),
          trackOffsets: getView<'uint32'>('trackOffsets'),
          triplegs: {
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
            }
          },
          modes:
            !props.modeSpeedUpperBoundsFloat32 && props.modes !== undefined
              ? getView<'uint32'>('modes')
              : undefined,
          output: {
            modalSplit: getView<'float32'>('modalSplit'),
            predictedModes:
              predictedModes !== undefined ? getView<'uint32'>('predictedModes') : undefined
          }
        })
    });
  }
}
