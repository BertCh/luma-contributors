// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUMultiscaleGeographicallyWeightedRegression,
  type GPUMultiscaleGeographicallyWeightedRegressionProps
} from './gpu-multiscale-geographically-weighted-regression';

/**
 * Graph names replace views, with explicit rowCount. Both known-variance conditional errors
 * and full bandwidth-history-conditioned coupled-smoother inference are supported.
 */
export type MultiscaleGeographicallyWeightedRegressionOperationProps = Pick<
  GPUMultiscaleGeographicallyWeightedRegressionProps,
  | 'id'
  | 'predictorCount'
  | 'kernel'
  | 'maximumIterations'
  | 'spatialContext'
  | 'indexGridSize'
  | 'gaussianCutoff'
  | 'bandwidthCandidateCount'
  | 'bandwidthSelection'
> & {
  rowCount: number;
  positions: string;
  predictors: string;
  response: string;
  parameters: string;
  mask?: string;
  bandwidthCandidates?: string;
  noiseVariance?: string;
  inference?: {
    varianceConvention?: 'residual-df' | 'smoother-df';
    output: {standardErrors: string; effectiveParameters: string; summary: string; status: string};
  };
  output: {
    coefficients: string;
    fitted: string;
    status: {codes: string; iterationCount: string};
    selectedBandwidths?: string;
    conditionalStandardErrors?: string;
  };
};

/**
 * MGWR graph operation. All props are compile-time; packed
 * tolerance/bandwidth parameters, candidates, noise variance and data contents change per frame.
 * Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 */
export class GPUMultiscaleGeographicallyWeightedRegressionOperation extends GPUContributorOperation {
  constructor(props: MultiscaleGeographicallyWeightedRegressionOperationProps) {
    const id = props.id ?? 'multiscale-geographically-weighted-regression';
    const coefficientCount = props.rowCount * (props.predictorCount + 1);
    super({
      id,
      inputs: {
        positions: props.positions,
        predictors: props.predictors,
        response: props.response,
        parameters: props.parameters,
        ...(props.mask ? {mask: props.mask} : {}),
        ...(props.bandwidthCandidates ? {bandwidthCandidates: props.bandwidthCandidates} : {}),
        ...(props.noiseVariance ? {noiseVariance: props.noiseVariance} : {})
      },
      outputs: {
        coefficients: {
          value: props.output.coefficients,
          format: 'float32',
          length: coefficientCount
        },
        fitted: {value: props.output.fitted, format: 'float32', length: props.rowCount},
        codes: {value: props.output.status.codes, format: 'uint32', length: props.rowCount},
        iterationCount: {value: props.output.status.iterationCount, format: 'uint32', length: 1},
        ...(props.output.selectedBandwidths
          ? {
              selectedBandwidths: {
                value: props.output.selectedBandwidths,
                format: 'float32' as const,
                length: props.predictorCount + 1
              }
            }
          : {}),
        ...(props.output.conditionalStandardErrors
          ? {
              conditionalStandardErrors: {
                value: props.output.conditionalStandardErrors,
                format: 'float32' as const,
                length: coefficientCount
              }
            }
          : {}),
        ...(props.inference
          ? {
              inferenceStandardErrors: {
                value: props.inference.output.standardErrors,
                format: 'float32' as const,
                length: coefficientCount
              },
              inferenceEffectiveParameters: {
                value: props.inference.output.effectiveParameters,
                format: 'float32' as const,
                length: props.predictorCount + 1
              },
              inferenceSummary: {
                value: props.inference.output.summary,
                format: 'float32' as const,
                length: 4
              },
              inferenceStatus: {
                value: props.inference.output.status,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUMultiscaleGeographicallyWeightedRegression({
          ...props,
          id,
          positions: getView<'float32x2'>('positions'),
          predictors: getView<'float32'>('predictors'),
          response: getView<'float32'>('response'),
          parameters: getView<'float32'>('parameters'),
          mask: props.mask ? getView<'uint32'>('mask') : undefined,
          bandwidthCandidates: props.bandwidthCandidates
            ? getView<'float32'>('bandwidthCandidates')
            : undefined,
          noiseVariance: props.noiseVariance ? getView<'float32'>('noiseVariance') : undefined,
          inference: props.inference
            ? {
                varianceConvention: props.inference.varianceConvention,
                output: {
                  standardErrors: getView<'float32'>('inferenceStandardErrors'),
                  effectiveParameters: getView<'float32'>('inferenceEffectiveParameters'),
                  summary: getView<'float32'>('inferenceSummary'),
                  status: getView<'uint32'>('inferenceStatus')
                }
              }
            : undefined,
          output: {
            coefficients: getView<'float32'>('coefficients'),
            fitted: getView<'float32'>('fitted'),
            status: {
              codes: getView<'uint32'>('codes'),
              iterationCount: getView<'uint32'>('iterationCount')
            },
            selectedBandwidths: props.output.selectedBandwidths
              ? getView<'float32'>('selectedBandwidths')
              : undefined,
            conditionalStandardErrors: props.output.conditionalStandardErrors
              ? getView<'float32'>('conditionalStandardErrors')
              : undefined
          }
        })
    });
  }
}
