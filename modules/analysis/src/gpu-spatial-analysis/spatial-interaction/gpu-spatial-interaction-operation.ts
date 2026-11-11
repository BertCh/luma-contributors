// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUSpatialInteraction, type GPUSpatialInteractionProps} from './gpu-spatial-interaction';

/** Graph value names replace views; pairCount explicitly supplies their common length. */
export type SpatialInteractionOperationProps = Pick<
  GPUSpatialInteractionProps,
  | 'id'
  | 'originCount'
  | 'destinationCount'
  | 'constraint'
  | 'coefficients'
  | 'calibration'
  | 'maximumCalibrationIterations'
  | 'coefficientTolerance'
  | 'maximumIterations'
  | 'tolerance'
> & {
  pairCount: number;
  origins: string;
  destinations: string;
  flows: string;
  covariate: string;
  pairState?: string;
  output: {
    predictions: string;
    summary: string;
    status: string;
    candidates: string;
    calibration?: string;
    originResiduals?: string;
    destinationResiduals?: string;
  };
};

/** GPU gravity operation. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING; all props compile-time. */
export class GPUSpatialInteractionOperation extends GPUContributorOperation {
  constructor(props: SpatialInteractionOperationProps) {
    const id = props.id ?? 'spatial-interaction';
    super({
      id,
      inputs: {
        origins: props.origins,
        destinations: props.destinations,
        flows: props.flows,
        covariate: props.covariate,
        ...(props.pairState ? {pairState: props.pairState} : {})
      },
      outputs: {
        predictions: {value: props.output.predictions, format: 'float32', length: props.pairCount},
        summary: {value: props.output.summary, format: 'float32', length: 4},
        status: {value: props.output.status, format: 'uint32', length: 1},
        candidates: {
          value: props.output.candidates,
          format: 'float32',
          length: props.coefficients.length * 4
        },
        ...(props.output.calibration
          ? {calibration: {value: props.output.calibration, format: 'float32' as const, length: 4}}
          : {}),
        ...(props.output.originResiduals
          ? {
              originResiduals: {
                value: props.output.originResiduals,
                format: 'float32' as const,
                length: props.originCount
              }
            }
          : {}),
        ...(props.output.destinationResiduals
          ? {
              destinationResiduals: {
                value: props.output.destinationResiduals,
                format: 'float32' as const,
                length: props.destinationCount
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUSpatialInteraction({
          ...props,
          id,
          origins: getView<'uint32'>('origins'),
          destinations: getView<'uint32'>('destinations'),
          flows: getView<'float32'>('flows'),
          covariate: getView<'float32'>('covariate'),
          pairState: props.pairState ? getView<'uint32'>('pairState') : undefined,
          output: {
            predictions: getView<'float32'>('predictions'),
            summary: getView<'float32'>('summary'),
            status: getView<'uint32'>('status'),
            candidates: getView<'float32'>('candidates'),
            calibration: props.output.calibration ? getView<'float32'>('calibration') : undefined,
            originResiduals: props.output.originResiduals
              ? getView<'float32'>('originResiduals')
              : undefined,
            destinationResiduals: props.output.destinationResiduals
              ? getView<'float32'>('destinationResiduals')
              : undefined
          }
        })
    });
  }
}
