// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {GPUVariogramModelFit} from './gpu-variogram-model-fit';
import type {VariogramModelType, VariogramModelWeighting} from './variogram-model';
import {validateVariogramModelFitChoices} from './variogram-model-fit-requirements';

/** Model families fitted when `models` is omitted, in output row order. */
export const VARIOGRAM_MODEL_FIT_DEFAULT_MODELS: readonly VariogramModelType[] = [
  'spherical',
  'exponential',
  'gaussian'
];

/**
 * Properties shared by {@link GPUVariogramModelFitOperation} and `CPUVariogramModelFitOperation`:
 * names of `GPUGraph` values in place of views. Either class built from the same properties reads
 * and writes the same graph values.
 */
export type VariogramModelFitOperationProps = {
  id?: string;
  /** `float32` graph value: representative lag per bin. */
  distances: string;
  /** `float32` graph value: empirical semivariance per bin. */
  semivariances: string;
  /** `uint32` graph value: pair count per bin. */
  pairCounts: string;
  /** Model families; output row order follows it. Defaults to spherical, exponential, Gaussian. */
  models?: readonly VariogramModelType[];
  /** Least-squares weighting. Defaults to `'cressie'`. */
  weighting?: VariogramModelWeighting;
  /** Log-range ladder length in both forms, 3–1024. Defaults to 200. */
  rangeSteps?: number;
  /** Golden-section budget per weighting pass in both forms, 0–64. Defaults to GPU 32 / CPU 60. */
  maximumIterations?: number;
  /** Graph values the fit writes, one row per model. */
  output: {
    /** `float32x4`: [nugget, partial sill, effective range, weighted residual]. */
    parameters: string;
    /** `float32x4`: [usable bins, unweighted squared residual, AIC, relative log-range bracket]. */
    diagnostics: string;
    /** `uint32`: 0 interior solution; 1 insufficient bins; 2 numerical failure; 3 boundary/budget. */
    status: string;
  };
};

/** @internal Capture options, model order and port names before either operation declares them. */
export function snapshotVariogramModelFitOperationProps(
  props: VariogramModelFitOperationProps
): VariogramModelFitOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'distances',
    'semivariances',
    'pairCounts',
    'models',
    'weighting',
    'rangeSteps',
    'maximumIterations',
    'output'
  ]);
  return Object.freeze({
    ...snapshot,
    models: Object.freeze([...(snapshot.models ?? VARIOGRAM_MODEL_FIT_DEFAULT_MODELS)]),
    output: snapshotConstructionFields(snapshot.output, ['parameters', 'diagnostics', 'status'])
  });
}

/**
 * `GPUVariogramModelFit` as a `GPUGraph` operation. Compile the graph with
 * `GPU_CONTRIBUTOR_OPERATION_LOWERING`. `CPUVariogramModelFitOperation` is its CPU twin.
 *
 * Compile-time: every property. Per-frame: none (the bins are graph values).
 */
export class GPUVariogramModelFitOperation extends GPUContributorOperation {
  constructor(props: VariogramModelFitOperationProps) {
    props = snapshotVariogramModelFitOperationProps(props);
    const id = props.id ?? 'variogram-model-fit';
    validateVariogramModelFitChoices(
      id,
      props.models ?? VARIOGRAM_MODEL_FIT_DEFAULT_MODELS,
      props.weighting ?? 'cressie'
    );
    const length = (props.models ?? VARIOGRAM_MODEL_FIT_DEFAULT_MODELS).length;
    super({
      id,
      inputs: {
        distances: props.distances,
        semivariances: props.semivariances,
        pairCounts: props.pairCounts
      },
      outputs: {
        parameters: {value: props.output.parameters, format: 'float32x4', length},
        diagnostics: {value: props.output.diagnostics, format: 'float32x4', length},
        status: {value: props.output.status, format: 'uint32', length}
      },
      createContributor: getView =>
        new GPUVariogramModelFit({
          id,
          distances: getView<'float32'>('distances'),
          semivariances: getView<'float32'>('semivariances'),
          pairCounts: getView<'uint32'>('pairCounts'),
          models: props.models,
          weighting: props.weighting,
          rangeSteps: props.rangeSteps,
          maximumIterations: props.maximumIterations,
          output: {
            parameters: getView<'float32x4'>('parameters'),
            diagnostics: getView<'float32x4'>('diagnostics'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
