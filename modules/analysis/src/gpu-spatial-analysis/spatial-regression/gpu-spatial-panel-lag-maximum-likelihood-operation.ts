// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUSpatialPanelLagMaximumLikelihood,
  type GPUSpatialPanelLagMaximumLikelihoodProps
} from './gpu-spatial-panel-lag-maximum-likelihood';

/** Graph names in place of views. entityCount explicitly sizes the balanced panel outputs. */
export type SpatialPanelLagMaximumLikelihoodOperationProps = Pick<
  GPUSpatialPanelLagMaximumLikelihoodProps,
  | 'id'
  | 'periodCount'
  | 'predictorCount'
  | 'effects'
  | 'maximumIterations'
  | 'maximumSearchIterations'
  | 'tolerance'
> & {
  entityCount: number;
  weights: Omit<
    GPUSpatialPanelLagMaximumLikelihoodProps['weights'],
    'offsets' | 'neighbors' | 'weights' | 'distances' | 'status'
  > & {
    offsets: string;
    neighbors: string;
    weights: string;
    distances?: string;
    status?: Partial<
      Record<
        keyof NonNullable<GPUSpatialPanelLagMaximumLikelihoodProps['weights']['status']>,
        string
      >
    >;
    /** Compatibility alias for status.overflow, used only when that field is absent. */
    overflow?: string;
  };
  predictors: string;
  response: string;
  output: {
    coefficients: string;
    covariance: string;
    summary: string;
    status: string;
    fitted: string;
    residuals: string;
  };
  hausman?: {tests: string; status: string};
};

/** GPU panel lag ML operation. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Compile-time: all props. Per-frame: input graph values. Optional weights views are graph ports;
 * information and spatialContext are forwarded unchanged, including contributor validation.
 */
export class GPUSpatialPanelLagMaximumLikelihoodOperation extends GPUContributorOperation {
  constructor(props: SpatialPanelLagMaximumLikelihoodOperationProps) {
    const id = props.id ?? 'spatial-panel-lag-maximum-likelihood';
    const rowCount = props.entityCount * props.periodCount;
    super({
      id,
      inputs: {
        predictors: props.predictors,
        response: props.response,
        offsets: props.weights.offsets,
        neighbors: props.weights.neighbors,
        weights: props.weights.weights,
        ...(props.weights.distances !== undefined ? {distances: props.weights.distances} : {}),
        ...Object.fromEntries(
          Object.entries(props.weights.status ?? {})
            .filter(([, value]) => value !== undefined)
            .map(([field, value]) => [`weightsStatus_${field}`, value!])
        ),
        ...(props.weights.overflow && !props.weights.status?.overflow
          ? {overflow: props.weights.overflow}
          : {})
      },
      outputs: {
        coefficients: {
          value: props.output.coefficients,
          format: 'float32',
          length: props.predictorCount + 3
        },
        covariance: {
          value: props.output.covariance,
          format: 'float32',
          length: (props.predictorCount + 2) ** 2
        },
        summary: {value: props.output.summary, format: 'float32', length: 6},
        status: {value: props.output.status, format: 'uint32', length: 1},
        fitted: {value: props.output.fitted, format: 'float32', length: rowCount},
        residuals: {value: props.output.residuals, format: 'float32', length: rowCount},
        ...(props.hausman
          ? {
              hausmanTests: {value: props.hausman.tests, format: 'float32' as const, length: 3},
              hausmanStatus: {value: props.hausman.status, format: 'uint32' as const, length: 1}
            }
          : {})
      },
      createContributor: getView =>
        new GPUSpatialPanelLagMaximumLikelihood({
          ...props,
          id,
          predictors: getView<'float32'>('predictors'),
          response: getView<'float32'>('response'),
          weights: {
            ...props.weights,
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            distances:
              props.weights.distances !== undefined ? getView<'float32'>('distances') : undefined,
            status:
              props.weights.status || props.weights.overflow
                ? {
                    ...Object.fromEntries(
                      Object.entries(props.weights.status ?? {})
                        .filter(([, value]) => value !== undefined)
                        .map(([field]) => [field, getView<'uint32'>(`weightsStatus_${field}`)])
                    ),
                    ...(props.weights.overflow && !props.weights.status?.overflow
                      ? {overflow: getView<'uint32'>('overflow')}
                      : {})
                  }
                : undefined
          },
          output: {
            coefficients: getView<'float32'>('coefficients'),
            covariance: getView<'float32'>('covariance'),
            summary: getView<'float32'>('summary'),
            status: getView<'uint32'>('status'),
            fitted: getView<'float32'>('fitted'),
            residuals: getView<'float32'>('residuals')
          },
          hausman: props.hausman
            ? {
                tests: getView<'float32'>('hausmanTests'),
                status: getView<'uint32'>('hausmanStatus')
              }
            : undefined
        })
    });
  }
}
