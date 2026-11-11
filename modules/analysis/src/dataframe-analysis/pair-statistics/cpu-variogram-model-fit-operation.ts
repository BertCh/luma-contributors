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
  VARIOGRAM_MODEL_FIT_DEFAULT_MODELS,
  type VariogramModelFitOperationProps
} from './gpu-variogram-model-fit-operation';
import {
  evaluateVariogramModel,
  fitVariogramModel,
  getVariogramModelShape,
  type VariogramModelInput,
  type VariogramModelType,
  type VariogramModelWeighting
} from './variogram-model';

/** Range ladder length of `fitVariogramModel`, which sets the boundary intervals below. */
const RANGE_LADDER_LENGTH = 200;
/** Golden-section steps `fitVariogramModel` always runs. */
const GOLDEN_SECTION_ITERATIONS = 60;
const GOLDEN_SECTION_RATIO = (Math.sqrt(5) - 1) / 2;

/** Options of {@link computeVariogramModelFitOnCPU}. */
export type CPUVariogramModelFitOptions = {
  /** Model families; output row order follows it. Defaults to spherical, exponential, Gaussian. */
  models?: readonly VariogramModelType[];
  /** Least-squares weighting. Defaults to `'cressie'`. */
  weighting?: VariogramModelWeighting;
};

/** Result of {@link computeVariogramModelFitOnCPU}, in the layout of `GPUVariogramModelFit`. */
export type CPUVariogramModelFitResult = {
  /** Per model: [nugget, partial sill, effective range, weighted residual]. */
  parameters: Float32Array;
  /** Per model: [usable bins, unweighted squared residual, AIC, relative log-range bracket]. */
  diagnostics: Float32Array;
  /** Per model: 0 interior solution; 1 insufficient bins; 2 numerical failure; 3 boundary. */
  status: Uint32Array;
};

/**
 * Fits each model family to empirical variogram bins with `fitVariogramModel` in float64 and
 * returns the rows `GPUVariogramModelFit` writes, rounded to float32.
 *
 * What differs from the GPU fit:
 * - Status 2 is set only when a fitted value is not finite. The GPU fit also refuses inputs whose
 *   float32 objective cannot be evaluated; float64 fits those.
 * - Status 3 is decided from the fitted range: it lies in the first or last interval of the range
 *   ladder over `[minimum lag / 10, 3 * maximum lag]`. The GPU fit decides it from the best ladder
 *   step and from its own search budget.
 * - The bracket diagnostic is an upper bound of the final golden-section bracket, which is below
 *   float32 resolution of the log range after the fixed 60 steps.
 * - Cressie weights have no floor other than the smallest float64.
 *
 * @param input Empirical bins. Bins with no pairs, a non-finite value or a non-positive lag are
 *   skipped.
 * @param options Model families and weighting.
 * @returns One row per model.
 * @throws If a model family or the weighting is unknown.
 */
export function computeVariogramModelFitOnCPU(
  input: VariogramModelInput,
  options: CPUVariogramModelFitOptions = {}
): CPUVariogramModelFitResult {
  const models = options.models ?? VARIOGRAM_MODEL_FIT_DEFAULT_MODELS;
  const weighting = options.weighting ?? 'cressie';
  const distances: number[] = [];
  const semivariances: number[] = [];
  const pairCounts: number[] = [];
  const length = Math.min(
    input.distances.length,
    input.semivariances.length,
    input.pairCounts.length
  );
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
  const lowestLogRange = Math.log(minimumLag / 10);
  const highestLogRange = Math.log(3 * maximumLag);
  const ladderStep = (highestLogRange - lowestLogRange) / (RANGE_LADDER_LENGTH - 1);

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
    const fit = fitVariogramModel({distances, semivariances, pairCounts}, {model, weighting});
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
    const logRange = Math.log(fit.range);
    const isBoundary =
      logRange <= lowestLogRange + ladderStep || logRange >= highestLogRange - ladderStep;
    parameters.set(row, modelIndex * 4);
    diagnostics.set(
      [
        binCount,
        unweightedResidual,
        binCount * Math.log(Math.max(unweightedResidual / binCount, 1e-30)) + 6,
        2 * ladderStep * GOLDEN_SECTION_RATIO ** GOLDEN_SECTION_ITERATIONS
      ],
      modelIndex * 4
    );
    status[modelIndex] = isBoundary ? 3 : 0;
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
 * `rangeSteps` and `maximumIterations` are budgets of the GPU search and are not used.
 */
export class CPUVariogramModelFitOperation extends GPUCPUOperation {
  constructor(props: VariogramModelFitOperationProps) {
    const id = props.id ?? 'variogram-model-fit';
    const models = props.models ?? VARIOGRAM_MODEL_FIT_DEFAULT_MODELS;
    const weighting = props.weighting ?? 'cressie';
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
    const shape = {length: models.length};
    super({
      id,
      inputs: {
        distances: props.distances,
        semivariances: props.semivariances,
        pairCounts: props.pairCounts
      },
      outputs: {
        [props.output.parameters]: gpuGraph.vector('float32x4', shape),
        [props.output.diagnostics]: gpuGraph.vector('float32x4', shape),
        [props.output.status]: gpuGraph.vector('uint32', shape)
      },
      execute({inputs}) {
        const result = computeVariogramModelFitOnCPU(
          {
            distances: getCPUOperationFloat32Rows(id, inputs, 'distances'),
            semivariances: getCPUOperationFloat32Rows(id, inputs, 'semivariances'),
            pairCounts: getCPUOperationUint32Rows(id, inputs, 'pairCounts')
          },
          {models, weighting}
        );
        return {
          [props.output.parameters]: result.parameters,
          [props.output.diagnostics]: result.diagnostics,
          [props.output.status]: result.status
        };
      }
    });
  }
}
