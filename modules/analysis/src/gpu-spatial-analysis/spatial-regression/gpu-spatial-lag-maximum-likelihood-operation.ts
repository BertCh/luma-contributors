// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUSpatialLagMaximumLikelihood,
  type GPUSpatialLagMaximumLikelihoodProps
} from './gpu-spatial-lag-maximum-likelihood';
import {
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE
} from './spatial-maximum-likelihood-parameters';

/** Graph values replace views; rowCount is the explicit response/whole-CSR row count. */
export type SpatialLagMaximumLikelihoodOperationProps = Pick<
  GPUSpatialLagMaximumLikelihoodProps,
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
    powerTraces?: string;
    powerSums?: string;
    infinityNorm?: string;
    powerTraceStandardErrors?: string;
  };
};

/**
 * Whole-CSR spatial lag likelihood operation. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Compile-time: all props. Per-frame: contents of every input, including the packed Philox seed.
 * Chunked weights are omitted because the mixed executor requires one packed vector per value.
 */
export class GPUSpatialLagMaximumLikelihoodOperation extends GPUContributorOperation {
  constructor(props: SpatialLagMaximumLikelihoodOperationProps) {
    const id = props.id ?? 'spatial-lag-maximum-likelihood';
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
        ...Object.fromEntries(
          (['powerTraces', 'powerSums', 'infinityNorm'] as const)
            .filter(port => props.output[port] !== undefined)
            .map(port => [
              port,
              {
                value: props.output[port]!,
                format: 'float32' as const,
                length:
                  port === 'infinityNorm'
                    ? 1
                    : (props.maximumPower ?? GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER)
              }
            ])
        ),
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
        new GPUSpatialLagMaximumLikelihood({
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
            powerTraces:
              props.output.powerTraces !== undefined
                ? getView<'float32'>('powerTraces')
                : undefined,
            powerSums:
              props.output.powerSums !== undefined ? getView<'float32'>('powerSums') : undefined,
            infinityNorm:
              props.output.infinityNorm !== undefined
                ? getView<'float32'>('infinityNorm')
                : undefined,
            powerTraceStandardErrors:
              props.output.powerTraceStandardErrors !== undefined
                ? getView<'float32'>('powerTraceStandardErrors')
                : undefined
          }
        })
    });
  }
}
