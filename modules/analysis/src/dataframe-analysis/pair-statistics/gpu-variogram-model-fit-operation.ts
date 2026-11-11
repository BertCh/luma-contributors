// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUVariogramModelFit} from './gpu-variogram-model-fit';
import type {VariogramModelType, VariogramModelWeighting} from './variogram-model';

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
  /** Range ladder length of the GPU search. The CPU fit uses its own fixed ladder. */
  rangeSteps?: number;
  /** Golden-section budget of the GPU search. The CPU fit uses its own fixed budget. */
  maximumIterations?: number;
  /** Graph values the fit writes, one row per model. */
  output: {
    /** `float32x4`: [nugget, partial sill, effective range, weighted residual]. */
    parameters: string;
    /** `float32x4`: [usable bins, unweighted squared residual, AIC, relative log-range bracket]. */
    diagnostics: string;
    /** `uint32`: 0 interior solution; 1 insufficient bins; 2 numerical failure; 3 boundary. */
    status: string;
  };
};

/**
 * `GPUVariogramModelFit` as a `GPUGraph` operation. Compile the graph with
 * `GPU_CONTRIBUTOR_OPERATION_LOWERING`. `CPUVariogramModelFitOperation` is its CPU twin.
 *
 * Compile-time: every property. Per-frame: none (the bins are graph values).
 */
export class GPUVariogramModelFitOperation extends GPUContributorOperation {
  constructor(props: VariogramModelFitOperationProps) {
    const id = props.id ?? 'variogram-model-fit';
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
