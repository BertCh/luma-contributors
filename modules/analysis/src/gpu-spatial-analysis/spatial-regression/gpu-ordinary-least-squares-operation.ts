// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUOrdinaryLeastSquares,
  type GPUOrdinaryLeastSquaresProps
} from './gpu-ordinary-least-squares';
import {GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH} from './ordinary-least-squares-parameters';

/** Graph names replace views; rowCount explicitly supplies the response length. */
export type OrdinaryLeastSquaresOperationProps = Pick<
  GPUOrdinaryLeastSquaresProps,
  'id' | 'predictorCount' | 'tileRowCount'
> & {
  rowCount: number;
  predictors: string;
  response: string;
  mask?: string;
  parameters?: string;
  output: {
    status: {code: string};
    coefficients: string;
    standardErrors: string;
    tStatistics: string;
    summary: string;
    residuals?: string;
    fitted?: string;
  };
};

/** GPU OLS as a graph operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUOrdinaryLeastSquaresOperation extends GPUContributorOperation {
  constructor(props: OrdinaryLeastSquaresOperationProps) {
    const id = props.id ?? 'ordinary-least-squares';
    super({
      id,
      inputs: {
        predictors: props.predictors,
        response: props.response,
        ...(props.mask ? {mask: props.mask} : {}),
        ...(props.parameters ? {parameters: props.parameters} : {})
      },
      outputs: {
        code: {value: props.output.status.code, format: 'uint32', length: 1},
        coefficients: {
          value: props.output.coefficients,
          format: 'float32',
          length: props.predictorCount + 1
        },
        standardErrors: {
          value: props.output.standardErrors,
          format: 'float32',
          length: props.predictorCount + 1
        },
        tStatistics: {
          value: props.output.tStatistics,
          format: 'float32',
          length: props.predictorCount + 1
        },
        summary: {
          value: props.output.summary,
          format: 'float32',
          length: GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
        },
        ...(props.output.residuals
          ? {
              residuals: {
                value: props.output.residuals,
                format: 'float32' as const,
                length: props.rowCount
              }
            }
          : {}),
        ...(props.output.fitted
          ? {
              fitted: {
                value: props.output.fitted,
                format: 'float32' as const,
                length: props.rowCount
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUOrdinaryLeastSquares({
          id,
          predictorCount: props.predictorCount,
          tileRowCount: props.tileRowCount,
          predictors: getView<'float32'>('predictors'),
          response: getView<'float32'>('response'),
          mask: props.mask ? getView<'uint32'>('mask') : undefined,
          parameters: props.parameters ? getView<'float32'>('parameters') : undefined,
          output: {
            status: {code: getView<'uint32'>('code')},
            coefficients: getView<'float32'>('coefficients'),
            standardErrors: getView<'float32'>('standardErrors'),
            tStatistics: getView<'float32'>('tStatistics'),
            summary: getView<'float32'>('summary'),
            residuals: props.output.residuals ? getView<'float32'>('residuals') : undefined,
            fitted: props.output.fitted ? getView<'float32'>('fitted') : undefined
          }
        })
    });
  }
}
