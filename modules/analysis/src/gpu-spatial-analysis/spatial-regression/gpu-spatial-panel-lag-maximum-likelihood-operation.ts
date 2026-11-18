// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  GPUSpatialPanelLagMaximumLikelihood,
  snapshotPanelLagWeights,
  type GPUSpatialPanelLagMaximumLikelihoodProps
} from './gpu-spatial-panel-lag-maximum-likelihood';

/** Graph names in place of views. entityCount explicitly sizes the balanced panel outputs. */
export type SpatialPanelLagMaximumLikelihoodOperationProps = Pick<
  GPUSpatialPanelLagMaximumLikelihoodProps,
  | 'id'
  | 'fitMode'
  | 'periodCount'
  | 'predictorCount'
  | 'effects'
  | 'maximumIterations'
  | 'maximumSearchIterations'
  | 'tolerance'
> & {
  entityCount: number;
  eigenvalues?: string;
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

/** Copies compile-time settings and port maps; graph values are still read on every run. @internal */
export function snapshotSpatialPanelLagOperationProps(
  props: SpatialPanelLagMaximumLikelihoodOperationProps
): SpatialPanelLagMaximumLikelihoodOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'fitMode',
    'entityCount',
    'eigenvalues',
    'periodCount',
    'predictorCount',
    'effects',
    'maximumIterations',
    'maximumSearchIterations',
    'tolerance',
    'weights',
    'predictors',
    'response',
    'output',
    'hausman'
  ]);
  return Object.freeze({
    ...snapshot,
    weights: snapshotPanelLagWeights(snapshot.weights),
    output: snapshotConstructionFields(snapshot.output, [
      'coefficients',
      'covariance',
      'summary',
      'status',
      'fitted',
      'residuals'
    ]),
    hausman: snapshot.hausman
      ? snapshotConstructionFields(snapshot.hausman, ['tests', 'status'])
      : undefined
  });
}

/** GPU panel lag ML operation. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Compile-time: all props. Per-frame: input graph values. Optional weights views are graph ports;
 * Port maps, information and spatialContext metadata are captured at construction, including
 * contributor validation; graph value contents remain per-frame inputs.
 */
export class GPUSpatialPanelLagMaximumLikelihoodOperation extends GPUContributorOperation {
  constructor(props: SpatialPanelLagMaximumLikelihoodOperationProps) {
    props = snapshotSpatialPanelLagOperationProps(props);
    const id = props.id ?? 'spatial-panel-lag-maximum-likelihood';
    const rowCount = props.entityCount * props.periodCount;
    super({
      id,
      inputs: {
        ...(props.eigenvalues !== undefined ? {eigenvalues: props.eigenvalues} : {}),
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
          eigenvalues:
            props.eigenvalues !== undefined ? getView<'float32'>('eigenvalues') : undefined,
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
