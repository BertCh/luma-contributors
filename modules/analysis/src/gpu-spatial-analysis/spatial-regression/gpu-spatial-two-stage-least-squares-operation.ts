// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotRegressionOperationProps} from './regression-construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateCompleteRegressionWeights} from './cpu-regression-weights';
import type {GPUInstrumentalVariablesCovarianceMethod} from './instrumental-variables-covariance-parameters';
import {
  GPUSpatialTwoStageLeastSquares,
  type GPUSpatialTwoStageLeastSquaresProps
} from './gpu-spatial-two-stage-least-squares';
import {
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH
} from './spatial-two-stage-least-squares-parameters';

/** Graph names replace every view; rowCount explicitly sizes the residual output. */
export type SpatialTwoStageLeastSquaresOperationProps = Pick<
  GPUSpatialTwoStageLeastSquaresProps,
  'id' | 'predictorCount' | 'tileRowCount' | 'instrumentOrder'
> & {
  rowCount: number;
  weights: {offsets: string; neighbors: string; weights: string};
  predictors: string;
  response: string;
  robust?: {
    method: GPUInstrumentalVariablesCovarianceMethod;
    clusters?: string;
    clusterCount?: number;
    kernel?: {
      offsets: string;
      neighbors: string;
      weights: string;
      status?: {
        overflow?: string;
        candidateOverflow?: string;
        complete?: string;
        count?: string;
        requiredCount?: string;
      };
    };
    output: {covariance: string; status: {code: string}};
  };
  output: {status: {code: string}; table: string; summary: string; residuals?: string};
};

/** Whole-CSR GPU IV operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Compile-time: all props. Per-frame: contents of the input graph values.
 * Unused model weights metadata is omitted; HAC overflow and candidateOverflow are separate optional inputs.
 */
export class GPUSpatialTwoStageLeastSquaresOperation extends GPUContributorOperation {
  constructor(props: SpatialTwoStageLeastSquaresOperationProps) {
    props = snapshotRegressionOperationProps(props);
    const id = props.id ?? 'spatial-two-stage-least-squares';
    validateCompleteRegressionWeights(id, props.weights);
    const robust = props.robust;
    if (robust)
      validateRequirement(
        id,
        Boolean(robust.output.status?.code),
        'robust.output.status.code is required'
      );
    for (const name of [
      'complete',
      'count',
      'requiredCount',
      'overflow',
      'candidateOverflow'
    ] as const) {
      const value = robust?.kernel?.status?.[name];
      if (value !== undefined)
        validateRequirement(
          id,
          typeof value === 'string' && value.length > 0,
          `robust.kernel.status.${name} must name a non-empty graph value; received ${String(value)}`
        );
    }
    const coefficientCount = props.predictorCount + 2;
    super({
      id,
      inputs: {
        ...props.weights,
        predictors: props.predictors,
        response: props.response,
        ...(robust?.clusters ? {clusters: robust.clusters} : {}),
        ...(robust?.kernel
          ? {
              kernelOffsets: robust.kernel.offsets,
              kernelNeighbors: robust.kernel.neighbors,
              kernelWeights: robust.kernel.weights,
              ...(robust.kernel.status?.complete
                ? {kernelComplete: robust.kernel.status.complete}
                : {}),
              ...(robust.kernel.status?.count ? {kernelCount: robust.kernel.status?.count} : {}),
              ...(robust.kernel.status?.requiredCount
                ? {kernelRequiredCount: robust.kernel.status?.requiredCount}
                : {}),
              ...(robust.kernel.status?.overflow
                ? {kernelOverflow: robust.kernel.status?.overflow}
                : {}),
              ...(robust.kernel.status?.candidateOverflow
                ? {kernelCandidateOverflow: robust.kernel.status?.candidateOverflow}
                : {})
            }
          : {})
      },
      outputs: {
        code: {value: props.output.status.code, format: 'uint32', length: 1},
        table: {
          value: props.output.table,
          format: 'float32',
          length: coefficientCount * GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE
        },
        summary: {
          value: props.output.summary,
          format: 'float32',
          length: GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH
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
        ...(robust
          ? {
              robustCovariance: {
                value: robust.output.covariance,
                format: 'float32' as const,
                length: coefficientCount ** 2
              },
              robustStatus: {value: robust.output.status.code, format: 'uint32' as const, length: 1}
            }
          : {})
      },
      createContributor: getView => {
        validateRequirement(
          id,
          getView('response').length === props.rowCount,
          'response length must equal rowCount'
        );
        return new GPUSpatialTwoStageLeastSquares({
          ...props,
          id,
          weights: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights')
          },
          predictors: getView<'float32'>('predictors'),
          response: getView<'float32'>('response'),
          robust: robust
            ? {
                method: robust.method,
                clusters: robust.clusters ? getView<'uint32'>('clusters') : undefined,
                clusterCount: robust.clusterCount,
                kernel: robust.kernel
                  ? {
                      offsets: getView<'uint32'>('kernelOffsets'),
                      neighbors: getView<'uint32'>('kernelNeighbors'),
                      weights: getView<'float32'>('kernelWeights'),
                      status: {
                        complete: robust.kernel.status?.complete
                          ? getView<'uint32'>('kernelComplete')
                          : undefined,
                        count: robust.kernel.status?.count
                          ? getView<'uint32'>('kernelCount')
                          : undefined,
                        requiredCount: robust.kernel.status?.requiredCount
                          ? getView<'uint32'>('kernelRequiredCount')
                          : undefined,
                        ...(robust.kernel.status?.overflow
                          ? {overflow: getView<'uint32'>('kernelOverflow')}
                          : {}),
                        ...(robust.kernel.status?.candidateOverflow
                          ? {candidateOverflow: getView<'uint32'>('kernelCandidateOverflow')}
                          : {})
                      }
                    }
                  : undefined,
                output: {
                  covariance: getView<'float32'>('robustCovariance'),
                  status: {code: getView<'uint32'>('robustStatus')}
                }
              }
            : undefined,
          output: {
            status: {code: getView<'uint32'>('code')},
            table: getView<'float32'>('table'),
            summary: getView<'float32'>('summary'),
            residuals: props.output.residuals ? getView<'float32'>('residuals') : undefined
          }
        });
      }
    });
  }
}
