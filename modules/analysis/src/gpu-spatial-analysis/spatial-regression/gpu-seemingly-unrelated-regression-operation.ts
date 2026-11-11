// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUSeeminglyUnrelatedRegression,
  type GPUSeeminglyUnrelatedRegressionProps
} from './gpu-seemingly-unrelated-regression';

/** Whole-CSR spatial GM or nonspatial one-step SUR, with graph names replacing every view. */
export type SeeminglyUnrelatedRegressionOperationProps = Pick<
  GPUSeeminglyUnrelatedRegressionProps,
  'id' | 'equationCount' | 'predictorCount' | 'rowCount'
> & {
  predictors: string;
  responses: string;
  mask?: string;
  initialResiduals?: string;
  spatial?: {
    weights: {offsets: string; neighbors: string; weights: string};
    output: {parameters: string; status: string};
  };
  output: {
    coefficients: string;
    covariance: string;
    residualCovariance: string;
    residuals: string;
    status: string;
  };
};

/** GPU SUR as an operation, including optional spatial GM parameters and equation statuses. */
export class GPUSeeminglyUnrelatedRegressionOperation extends GPUContributorOperation {
  constructor(props: SeeminglyUnrelatedRegressionOperationProps) {
    const id = props.id ?? 'seemingly-unrelated-regression';
    const coefficientCount = props.equationCount * (props.predictorCount + 1);
    super({
      id,
      inputs: {
        predictors: props.predictors,
        responses: props.responses,
        ...(props.mask ? {mask: props.mask} : {}),
        ...(props.initialResiduals ? {initialResiduals: props.initialResiduals} : {}),
        ...(props.spatial
          ? {
              spatialOffsets: props.spatial.weights.offsets,
              spatialNeighbors: props.spatial.weights.neighbors,
              spatialWeights: props.spatial.weights.weights
            }
          : {})
      },
      outputs: {
        coefficients: {
          value: props.output.coefficients,
          format: 'float32',
          length: coefficientCount
        },
        covariance: {
          value: props.output.covariance,
          format: 'float32',
          length: coefficientCount ** 2
        },
        residualCovariance: {
          value: props.output.residualCovariance,
          format: 'float32',
          length: props.equationCount ** 2
        },
        residuals: {
          value: props.output.residuals,
          format: 'float32',
          length: props.equationCount * props.rowCount
        },
        status: {value: props.output.status, format: 'uint32', length: 1},
        ...(props.spatial
          ? {
              spatialParameters: {
                value: props.spatial.output.parameters,
                format: 'float32' as const,
                length: props.equationCount
              },
              spatialStatus: {
                value: props.spatial.output.status,
                format: 'uint32' as const,
                length: props.equationCount
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUSeeminglyUnrelatedRegression({
          id,
          equationCount: props.equationCount,
          predictorCount: props.predictorCount,
          rowCount: props.rowCount,
          predictors: getView<'float32'>('predictors'),
          responses: getView<'float32'>('responses'),
          mask: props.mask ? getView<'uint32'>('mask') : undefined,
          initialResiduals: props.initialResiduals
            ? getView<'float32'>('initialResiduals')
            : undefined,
          spatial: props.spatial
            ? {
                weights: {
                  offsets: getView<'uint32'>('spatialOffsets'),
                  neighbors: getView<'uint32'>('spatialNeighbors'),
                  weights: getView<'float32'>('spatialWeights')
                },
                output: {
                  parameters: getView<'float32'>('spatialParameters'),
                  status: getView<'uint32'>('spatialStatus')
                }
              }
            : undefined,
          output: {
            coefficients: getView<'float32'>('coefficients'),
            covariance: getView<'float32'>('covariance'),
            residualCovariance: getView<'float32'>('residualCovariance'),
            residuals: getView<'float32'>('residuals'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
