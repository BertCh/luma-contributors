// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUModalSplit, type GPUModalSplitProps} from './gpu-modal-split';

/**
 * Graph-value form. In prediction mode the optional input modes port is omitted: the contributor
 * would write it in place. Its writable modes table is instead the output or transient scratch.
 */
export type ModalSplitOperationProps = Pick<
  GPUModalSplitProps,
  'id' | 'metric' | 'modeCount' | 'spatialContext' | 'temporalContext'
> & {
  positions: string;
  timestamps: string;
  trackOffsets: string;
  timestampFormat?: 'float32' | 'uint32x2';
  rowCount: number;
  trackCount: number;
  triplegCapacity: number;
  triplegs: {tracks: string; startRows: string; endRows: string; status: {count: string}};
  output: {modalSplit: string; predictedModes?: string};
} & (
    | {modes: string; modeSpeedBreaks?: never}
    | {modes?: never; modeSpeedBreaks: readonly number[]}
  );

/** GPU modal aggregation operation. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUModalSplitOperation extends GPUContributorOperation {
  constructor(props: ModalSplitOperationProps) {
    const id = props.id ?? 'modal-split';
    const predictedModes = props.modeSpeedBreaks ? props.output.predictedModes : undefined;
    super({
      id,
      inputs: {
        positions: props.positions,
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
        triplegTracks: props.triplegs.tracks,
        triplegStartRows: props.triplegs.startRows,
        triplegEndRows: props.triplegs.endRows,
        triplegCount: props.triplegs.status.count,
        ...(!props.modeSpeedBreaks && props.modes !== undefined ? {modes: props.modes} : {})
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
          modeSpeedBreaks: props.modeSpeedBreaks,
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
            status: {count: getView<'uint32'>('triplegCount')}
          },
          modes:
            !props.modeSpeedBreaks && props.modes !== undefined
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
