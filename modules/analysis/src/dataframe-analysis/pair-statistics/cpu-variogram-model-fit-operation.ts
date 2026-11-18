// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUVariogramModelFitOperation,
  VARIOGRAM_MODEL_FIT_DEFAULT_MODELS,
  snapshotVariogramModelFitOperationProps,
  type VariogramModelFitOperationProps
} from './gpu-variogram-model-fit-operation';
import {
  evaluateVariogramModel,
  fitVariogramModelWithDiagnostics,
  getVariogramModelSearchOptions,
  getVariogramModelShape,
  type VariogramModelInput,
  type VariogramModelType,
  type VariogramModelWeighting
} from './variogram-model';

function validateVariogramModelFitBins(id: string, input: VariogramModelInput): void {
  if (
    input.semivariances.length !== input.distances.length ||
    input.pairCounts.length !== input.distances.length
  ) {
    throw new Error(`${id} invalid model or bin topology`);
  }
}

/** Options of {@link computeVariogramModelFitOnCPU}. */
export type CPUVariogramModelFitOptions = {
  /** Model families; output row order follows it. Defaults to spherical, exponential, Gaussian. */
  models?: readonly VariogramModelType[];
  /** Least-squares weighting. Defaults to `'cressie'`. */
  weighting?: VariogramModelWeighting;
  /** Log-range ladder length, 3–1024. Defaults to 200. */
  rangeSteps?: number;
  /** Golden-section refinements per weighting pass, 0–64. Defaults to 60 on CPU. */
  maximumIterations?: number;
};

/** Result of {@link computeVariogramModelFitOnCPU}, in the layout of `GPUVariogramModelFit`. */
export type CPUVariogramModelFitResult = {
  /** Per model: [nugget, partial sill, effective range, weighted residual]. */
  parameters: Float32Array;
  /** Per model: [usable bins, unweighted squared residual, AIC, relative log-range bracket]. */
  diagnostics: Float32Array;
  /** Per model: 0 interior solution; 1 insufficient bins; 2 numerical failure; 3 boundary/budget. */
  status: Uint32Array;
};

/**
 * Fits each model family to empirical variogram bins with `fitVariogramModel` in float64 and
 * returns the rows `GPUVariogramModelFit` writes, rounded to float32.
 *
 * What differs from the GPU fit:
 * - Status 2 is set only when a fitted value is not finite. The GPU fit also refuses inputs whose
 *   float32 objective cannot be evaluated; float64 fits those.
 * - CPU defaults to 60 refinement steps, GPU to 32. Explicit rangeSteps and maximumIterations
 *   bind both searches. CPU refines up to its budget in float64; GPU may stop at bracket 1e-5.
 * - The bracket diagnostic is the measured final log-range bracket. Both forms set status 3
 *   for an endpoint ladder optimum or a final bracket wider than 1e-5.
 * - Cressie weights have no floor other than the smallest float64.
 *
 * @param input Empirical bins. Bins with no pairs, a non-finite value or a non-positive lag are
 *   skipped.
 * @param options Model families, weighting and search budgets.
 * @returns One row per model.
 * @throws If bin arrays have unequal lengths, a model family or the weighting is unknown.
 */
export function computeVariogramModelFitOnCPU(
  input: VariogramModelInput,
  options: CPUVariogramModelFitOptions = {}
): CPUVariogramModelFitResult {
  validateVariogramModelFitBins('variogram-model-fit', input);
  const searchOptions = getVariogramModelSearchOptions(options, 'variogram-model-fit');
  const models = options.models ?? VARIOGRAM_MODEL_FIT_DEFAULT_MODELS;
  const weighting = options.weighting ?? 'cressie';
  const distances: number[] = [];
  const semivariances: number[] = [];
  const pairCounts: number[] = [];
  const length = input.distances.length;
  for (let bin = 0; bin < length; bin++) {
    const distance = input.distances[bin];
    const semivariance = input.semivariances[bin];
    const pairCount = input.pairCounts[bin];
    if (
      Number.isFinite(distance) &&
      Number.isFinite(semivariance) &&
      distance > 0 &&
      pairCount > 0
    ) {
      distances.push(distance);
      semivariances.push(semivariance);
      pairCounts.push(pairCount);
    }
  }
  const binCount = distances.length;
  let minimumLag = Number.POSITIVE_INFINITY;
  let maximumLag = Number.NEGATIVE_INFINITY;
  for (const distance of distances) {
    minimumLag = Math.min(minimumLag, distance);
    maximumLag = Math.max(maximumLag, distance);
  }
  // The rule of the GPU fit: two usable bins at two different lags.
  const isFittable = binCount >= 2 && maximumLag > minimumLag;

  const parameters = new Float32Array(models.length * 4).fill(Number.NaN);
  const diagnostics = new Float32Array(models.length * 4).fill(Number.NaN);
  const status = new Uint32Array(models.length);
  models.forEach((model, modelIndex) => {
    // Throws for an unknown family before any row is written.
    getVariogramModelShape(model, 0);
    if (!isFittable) {
      status[modelIndex] = 1;
      return;
    }
    const search = fitVariogramModelWithDiagnostics(
      {distances, semivariances, pairCounts},
      {model, weighting, ...searchOptions}
    );
    const fit = search.model;
    let unweightedResidual = 0;
    for (let bin = 0; bin < binCount; bin++) {
      const difference = semivariances[bin] - evaluateVariogramModel(fit, distances[bin]);
      unweightedResidual += difference * difference;
    }
    const row = [fit.nugget, fit.sill, fit.range, fit.residual];
    if (
      !row.every(value => Number.isFinite(Math.fround(value))) ||
      !Number.isFinite(Math.fround(unweightedResidual))
    ) {
      status[modelIndex] = 2;
      return;
    }
    parameters.set(row, modelIndex * 4);
    diagnostics.set(
      [
        binCount,
        unweightedResidual,
        binCount * Math.log(Math.max(unweightedResidual / binCount, 1e-30)) + 6,
        search.logRangeBracket
      ],
      modelIndex * 4
    );
    status[modelIndex] = search.atBoundary || search.logRangeBracket > 1e-5 ? 3 : 0;
  });
  return {parameters, diagnostics, status};
}

/**
 * CPU twin of `GPUVariogramModelFitOperation`: the same graph values, fitted by
 * {@link computeVariogramModelFitOnCPU}. The executor reads the bins back before it and uploads
 * the three small outputs after it. Prefer it over the GPU fit: the bins are a few hundred rows
 * at most, and the float64 fit has no representability limit (measured in one graph on the test
 * GPU: 0.5 ms against 1.6 to 2.5 ms).
 *
 * Explicit `rangeSteps` and `maximumIterations` bind every weighting pass just as on GPU.
 * CPU defaults remain 200 ladder points and 60 refinements when those props are omitted.
 * Unequal bin lengths throw the GPU contributor's `invalid model or bin topology` error,
 * prefixed by this operation's ID.
 */
export class CPUVariogramModelFitOperation extends GPUCPUOperation {
  constructor(props: VariogramModelFitOperationProps) {
    props = snapshotVariogramModelFitOperationProps(props);
    const id = props.id ?? 'variogram-model-fit';
    const models = props.models ?? VARIOGRAM_MODEL_FIT_DEFAULT_MODELS;
    const weighting = props.weighting ?? 'cressie';
    const searchOptions = getVariogramModelSearchOptions(props, id);
    validateRequirement(
      id,
      models.length >= 1 && models.length <= 5,
      'models must name one to five model families'
    );
    for (const model of models) {
      getVariogramModelShape(model, 0);
    }
    validateRequirement(
      id,
      ['pairs', 'none', 'cressie'].includes(weighting),
      "weighting must be 'cressie', 'pairs' or 'none'"
    );
    const inputs = {
      distances: props.distances,
      semivariances: props.semivariances,
      pairCounts: props.pairCounts
    };
    for (const [port, value] of Object.entries(inputs)) {
      validateRequirement(
        id,
        typeof value === 'string' && value.length > 0,
        `inputs.${port} must be a non-empty graph value name`
      );
    }
    // Validate before computed keys coerce non-string names into strings.
    for (const port of ['parameters', 'diagnostics', 'status'] as const) {
      const value = props.output[port];
      validateRequirement(
        id,
        typeof value === 'string' && value.length > 0,
        `outputs.${port}.value must be a non-empty graph value name`
      );
    }
    // Apply the shared contributor-operation name rules before output keys can collapse.
    new GPUVariogramModelFitOperation(props);
    const shape = {length: models.length};
    super({
      id,
      inputs,
      outputs: {
        [props.output.parameters]: gpuGraph.vector('float32x4', shape),
        [props.output.diagnostics]: gpuGraph.vector('float32x4', shape),
        [props.output.status]: gpuGraph.vector('uint32', shape)
      },
      execute({inputs}) {
        const bins = {
          distances: getCPUOperationFloat32Rows(id, inputs, 'distances'),
          semivariances: getCPUOperationFloat32Rows(id, inputs, 'semivariances'),
          pairCounts: getCPUOperationUint32Rows(id, inputs, 'pairCounts')
        };
        validateVariogramModelFitBins(id, bins);
        const result = computeVariogramModelFitOnCPU(bins, {models, weighting, ...searchOptions});
        return {
          [props.output.parameters]: result.parameters,
          [props.output.diagnostics]: result.diagnostics,
          [props.output.status]: result.status
        };
      }
    });
  }
}
