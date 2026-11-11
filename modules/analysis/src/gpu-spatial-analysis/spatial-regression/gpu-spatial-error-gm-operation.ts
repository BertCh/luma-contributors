// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUSpatialErrorGM, type GPUSpatialErrorGMProps} from './gpu-spatial-error-gm';
import {
  GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH,
  GPU_SPATIAL_ERROR_GM_TABLE_STRIDE
} from './spatial-error-gm-parameters';

/** Graph names replace views; rowCount supplies the response and whole CSR row count. */
export type SpatialErrorGMOperationProps = Pick<
  GPUSpatialErrorGMProps,
  'id' | 'predictorCount' | 'estimator' | 'traceMode' | 'traceProbeCount' | 'tileRowCount'
> & {
  rowCount: number;
  weights: {offsets: string; neighbors: string; weights: string};
  predictors: string;
  response: string;
  traceParameters?: string;
  output: {status: {code: string}; table: string; summary: string; residuals?: string};
};

/**
 * Whole-CSR GM operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Compile-time: all props. Per-frame: contents of the input graph values, including trace seed.
 * Chunked weights are outside this operation's packed-vector contract.
 */
export class GPUSpatialErrorGMOperation extends GPUContributorOperation {
  constructor(props: SpatialErrorGMOperationProps) {
    const id = props.id ?? 'spatial-error-gm';
    super({
      id,
      inputs: {
        ...props.weights,
        predictors: props.predictors,
        response: props.response,
        ...(props.traceParameters ? {traceParameters: props.traceParameters} : {})
      },
      outputs: {
        code: {value: props.output.status.code, format: 'uint32', length: 1},
        table: {
          value: props.output.table,
          format: 'float32',
          length: (props.predictorCount + 2) * GPU_SPATIAL_ERROR_GM_TABLE_STRIDE
        },
        summary: {
          value: props.output.summary,
          format: 'float32',
          length: GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH
        },
        ...(props.output.residuals
          ? {
              residuals: {
                value: props.output.residuals,
                format: 'float32' as const,
                length: props.rowCount
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUSpatialErrorGM({
          ...props,
          id,
          weights: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights')
          },
          predictors: getView<'float32'>('predictors'),
          response: getView<'float32'>('response'),
          traceParameters: props.traceParameters ? getView<'uint32'>('traceParameters') : undefined,
          output: {
            status: {code: getView<'uint32'>('code')},
            table: getView<'float32'>('table'),
            summary: getView<'float32'>('summary'),
            residuals: props.output.residuals ? getView<'float32'>('residuals') : undefined
          }
        })
    });
  }
}
