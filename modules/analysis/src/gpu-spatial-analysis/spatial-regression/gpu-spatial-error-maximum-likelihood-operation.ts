// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotRegressionOperationProps} from './regression-construction-snapshot';

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateCompleteRegressionWeights} from './cpu-regression-weights';
import {
  GPUSpatialErrorMaximumLikelihood,
  type GPUSpatialErrorMaximumLikelihoodProps
} from './gpu-spatial-error-maximum-likelihood';
import {
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE
} from './spatial-maximum-likelihood-parameters';

/** Graph values replace views; rowCount is the explicit response/whole-CSR row count. */
export type SpatialErrorMaximumLikelihoodOperationProps = Pick<
  GPUSpatialErrorMaximumLikelihoodProps,
  | 'id'
  | 'predictorCount'
  | 'maximumPower'
  | 'traceMode'
  | 'traceProbeCount'
  | 'tailExtrapolation'
  | 'logDeterminant'
  | 'tileRowCount'
> & {
  rowCount: number;
  weights: {offsets: string; neighbors: string; weights: string};
  predictors: string;
  response: string;
  eigenvalues?: string;
  traceParameters?: string;
  output: {
    status: {code: string; converged?: string};
    table: string;
    summary: string;
    residuals?: string;
    powerTraceStandardErrors?: string;
  };
};

/**
 * Whole-CSR spatial error likelihood operation. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Compile-time: all props. Per-frame: contents of every input, including the packed Philox seed.
 * Chunked weights are omitted because the mixed executor requires one packed vector per value.
 */
export class GPUSpatialErrorMaximumLikelihoodOperation extends GPUContributorOperation {
  constructor(props: SpatialErrorMaximumLikelihoodOperationProps) {
    props = snapshotRegressionOperationProps(props);
    const id = props.id ?? 'spatial-error-maximum-likelihood';
    validateCompleteRegressionWeights(id, props.weights);
    super({
      id,
      inputs: {
        ...props.weights,
        predictors: props.predictors,
        response: props.response,
        ...(props.eigenvalues !== undefined ? {eigenvalues: props.eigenvalues} : {}),
        ...(props.traceParameters !== undefined ? {traceParameters: props.traceParameters} : {})
      },
      outputs: {
        code: {value: props.output.status.code, format: 'uint32', length: 1},
        ...(props.output.status.converged !== undefined
          ? {
              converged: {
                value: props.output.status.converged,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {}),
        table: {
          value: props.output.table,
          format: 'float32',
          length: (props.predictorCount + 2) * GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE
        },
        summary: {
          value: props.output.summary,
          format: 'float32',
          length: GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH
        },
        ...(props.output.residuals !== undefined
          ? {
              residuals: {
                value: props.output.residuals,
                format: 'float32' as const,
                length: props.rowCount
              }
            }
          : {}),
        ...(props.output.powerTraceStandardErrors !== undefined
          ? {
              powerTraceStandardErrors: {
                value: props.output.powerTraceStandardErrors,
                format: 'float32' as const,
                length: props.maximumPower ?? GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUSpatialErrorMaximumLikelihood({
          ...props,
          id,
          weights: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights')
          },
          predictors: getView<'float32'>('predictors'),
          response: getView<'float32'>('response'),
          eigenvalues:
            props.eigenvalues !== undefined ? getView<'float32'>('eigenvalues') : undefined,
          traceParameters:
            props.traceParameters !== undefined ? getView<'uint32'>('traceParameters') : undefined,
          output: {
            status: {
              code: getView<'uint32'>('code'),
              converged:
                props.output.status.converged !== undefined
                  ? getView<'uint32'>('converged')
                  : undefined
            },
            table: getView<'float32'>('table'),
            summary: getView<'float32'>('summary'),
            residuals:
              props.output.residuals !== undefined ? getView<'float32'>('residuals') : undefined,
            powerTraceStandardErrors:
              props.output.powerTraceStandardErrors !== undefined
                ? getView<'float32'>('powerTraceStandardErrors')
                : undefined
          }
        })
    });
  }
}
