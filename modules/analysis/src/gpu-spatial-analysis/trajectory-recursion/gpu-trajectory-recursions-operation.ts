// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUTrajectoryRecursions,
  type GPUTrajectoryRecursionsProps
} from './gpu-trajectory-recursions';

/** Graph-value names replacing contributor views, with explicit lengths. */
export type TrajectoryRecursionsOperationProps = Pick<
  GPUTrajectoryRecursionsProps,
  'id' | 'spatialContext' | 'temporalContext' | 'maximumTrackSegments'
> & {
  positions: string;
  timestamps: string;
  timestampFormat?: 'float32' | 'uint32x2';
  trackOffsets: string;
  radii: string;
  rowCount: number;
  radiusCount: number;
  output: {
    revisits?: string;
    residenceTimes?: string;
    firstPassageTimes?: string;
    firstPassageCensored?: string;
    status?: {candidateOverflow?: string};
  };
};

/** GPU recursion analysis as a graph operation; use GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUTrajectoryRecursionsOperation extends GPUContributorOperation {
  constructor(props: TrajectoryRecursionsOperationProps) {
    const id = props.id ?? 'trajectory-recursions';
    const {output} = props;
    const length = props.rowCount * props.radiusCount;
    super({
      id,
      inputs: {
        positions: props.positions,
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
        radii: props.radii
      },
      outputs: {
        ...Object.fromEntries(
          (['revisits', 'firstPassageCensored'] as const).flatMap(port =>
            output[port] ? [[port, {value: output[port]!, format: 'uint32' as const, length}]] : []
          )
        ),
        ...Object.fromEntries(
          (['residenceTimes', 'firstPassageTimes'] as const).flatMap(port =>
            output[port] ? [[port, {value: output[port]!, format: 'float32' as const, length}]] : []
          )
        ),
        ...(output.status?.candidateOverflow
          ? {
              candidateOverflow: {
                value: output.status.candidateOverflow,
                format: 'uint32',
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUTrajectoryRecursions({
          id,
          spatialContext: props.spatialContext,
          temporalContext: props.temporalContext,
          maximumTrackSegments: props.maximumTrackSegments,
          positions: getView<'float32x2'>('positions'),
          timestamps:
            props.timestampFormat === 'uint32x2'
              ? getView<'uint32x2'>('timestamps')
              : getView<'float32'>('timestamps'),
          trackOffsets: getView<'uint32'>('trackOffsets'),
          radii: getView<'float32'>('radii'),
          output: {
            revisits: output.revisits ? getView<'uint32'>('revisits') : undefined,
            residenceTimes: output.residenceTimes
              ? getView<'float32'>('residenceTimes')
              : undefined,
            firstPassageTimes: output.firstPassageTimes
              ? getView<'float32'>('firstPassageTimes')
              : undefined,
            firstPassageCensored: output.firstPassageCensored
              ? getView<'uint32'>('firstPassageCensored')
              : undefined,
            status: output.status?.candidateOverflow
              ? {candidateOverflow: getView<'uint32'>('candidateOverflow')}
              : undefined
          }
        })
    });
  }
}
