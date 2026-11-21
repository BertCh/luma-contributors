// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConditionalGaussianSimulationOperationProps} from './interpolation-operation-snapshots';

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getPhilox4x32,
  getPhiloxOpenUnitFloat,
  PHILOX_PURPOSES
} from '../../analysis-core/kernels/philox-random';
import {getCPUVecchiaCorrelation} from './cpu-vecchia-evaluation';
import {
  getPredictionCovarianceModelCode,
  type GPUPredictionCovariance
} from './prediction-covariance';
import {
  GPUConditionalGaussianSimulationOperation,
  type ConditionalGaussianSimulationOperationProps
} from './gpu-conditional-gaussian-simulation-operation';

/** Packed inputs in the contributor's layouts. Observation validity must be strictly > 0; zero, negative and NaN omit the sample. */
export type CPUConditionalGaussianSimulationInput = {
  positions: Float32Array;
  observations: Float32Array;
  queries: Float32Array;
};
/** Known-mean Gaussian model and stable Philox draw identities. */
export type CPUConditionalGaussianSimulationOptions = {
  covariance: GPUPredictionCovariance;
  mean?: number;
  seed: number;
  replicateCount: number;
  replicateOffset?: number;
};
/** Replicate-major values, query-major [mean, variance] rows and per-replicate statuses. */
export type CPUConditionalGaussianSimulationResult = {
  values: Float32Array;
  moments: Float32Array;
  status: Uint32Array;
};

function validateSimulationOptions(
  id: string,
  options: CPUConditionalGaussianSimulationOptions
): void {
  validateCapacity(id, 'seed', options.seed, 0);
  validateCapacity(id, 'replicateCount', options.replicateCount);
  validateCapacity(id, 'replicateOffset', options.replicateOffset ?? 0, 0);
  validateMaximumCount(
    id,
    'replicateOffset + replicateCount',
    (options.replicateOffset ?? 0) + options.replicateCount,
    0x100000000
  );
  validateRequirement(
    id,
    ['spherical', 'exponential', 'gaussian', 'matern-3/2', 'matern-5/2', 'matern'].includes(
      options.covariance.model
    ),
    'covariance.model must name a supported family'
  );
  validateRequirement(
    id,
    options.covariance.model !== 'matern' ||
      [0.5, 1.5, 2.5].includes(options.covariance.smoothness ?? 0),
    'covariance.smoothness must be 0.5, 1.5 or 2.5 for matern'
  );
  // Plain numeric covariance and mean props retain their float64 precision on CPU.
  validateRequirement(id, Number.isFinite(options.mean ?? 0), 'mean must be finite');
  for (const [name, value, minimum] of [
    ['range', options.covariance.range, Number.MIN_VALUE],
    ['sill', options.covariance.sill, 0],
    ['nugget', options.covariance.nugget ?? 0, 0]
  ] as const) {
    validateRequirement(
      id,
      Number.isFinite(value) && value >= minimum,
      `covariance.${name} must be finite and ${minimum === 0 ? 'nonnegative' : 'positive'}`
    );
  }
  validateRequirement(
    id,
    options.covariance.sill + (options.covariance.nugget ?? 0) > 0 &&
      Number.isFinite(options.covariance.sill + (options.covariance.nugget ?? 0)),
    'covariance sill + nugget must be positive and finite'
  );
}

/**
 * Conditions a joint Gaussian distribution and draws its conditional query vector using the
 * contributor's Philox purpose and (replicateOffset + replicate, latent query index, 0, 0) counters.
 * All calculations use float64, with a single rounding into each returned float32 array.
 *
 * What differs from the GPU contributor:
 * - The 4096 joint-row and GPU storage-binding limits are dropped; this still allocates a dense
 *   quadratic factor. Uint32 draw identities and output lengths remain limited by their layouts.
 * - Positive observation pivots are accepted below the GPU's 1e-6 covariance-scale cutoff.
 *   Query roundoff is tested at 64 * Number.EPSILON * joint size * covariance scale instead.
 * - Exact coordinate aliases have exactly zero new variance; noiseless observed queries copy
 *   the observation. The cooperative GPU path can retain positive cancellation residue.
 * - Full closed-form Matern correlations replace truncated series; the matern smoothness alias
 *   is accepted. Plain numeric covariance and mean props keep their float64 values.
 * - Philox words and their order match. Box-Muller and its half-bin angle are evaluated in
 *   float64; GPU transcendental and angle rounding differ. Compare ordinary moments at relative
 *   1e-4 and realizations at 2e-4 of max(conditional standard deviation, absolute mean).
 * - The pure function and CPU operation recompute preparation on each call.
 * - Invalid included observations or queries give status 1; singular observations give status 2.
 *   Subnormal outputs are retained; finite float64 results outside float32 range round to infinity
 *   without setting status 2. These extremes are outside GPU comparisons.
 *
 * @throws If positions, observations or queries have inconsistent lengths or options are invalid.
 */
export function computeConditionalGaussianSimulationOnCPU(
  input: CPUConditionalGaussianSimulationInput,
  options: CPUConditionalGaussianSimulationOptions
): CPUConditionalGaussianSimulationResult {
  const id = 'computeConditionalGaussianSimulationOnCPU';
  validateSimulationOptions(id, options);
  validateRequirement(
    'computeConditionalGaussianSimulationOnCPU',
    !(input.positions.length % 2 !== 0),
    'positions length must be a multiple of two'
  );
  validateRequirement(
    'computeConditionalGaussianSimulationOnCPU',
    !(input.observations.length !== input.positions.length * 2),
    'observations length must equal positions row count * four'
  );
  validateRequirement(
    'computeConditionalGaussianSimulationOnCPU',
    !(input.queries.length % 2 !== 0 || input.queries.length === 0),
    'queries must contain one or more two-component rows'
  );
  const queryCount = input.queries.length / 2;
  validateCapacity(id, 'queryCount * replicateCount', queryCount * options.replicateCount);
  const result: CPUConditionalGaussianSimulationResult = {
    values: new Float32Array(queryCount * options.replicateCount).fill(NaN),
    moments: new Float32Array(queryCount * 2).fill(NaN),
    status: new Uint32Array(options.replicateCount).fill(1)
  };
  const observed: number[] = [];
  for (let row = 0; row < input.positions.length / 2; row++) {
    const [value, error, , validity] = input.observations.subarray(row * 4, row * 4 + 4);
    if (!(validity > 0)) continue;
    if (
      !Number.isFinite(value) ||
      !Number.isFinite(error) ||
      error < 0 ||
      !Number.isFinite(input.positions[row * 2]) ||
      !Number.isFinite(input.positions[row * 2 + 1])
    )
      return result;
    observed.push(row);
  }
  if (!input.queries.every(Number.isFinite)) return result;
  const count = observed.length;
  const size = count + queryCount;
  const positions = new Float64Array(size * 2);
  const errors = new Float64Array(size);
  const centered = new Float64Array(count);
  observed.forEach((sample, row) => {
    positions.set(input.positions.subarray(sample * 2, sample * 2 + 2), row * 2);
    errors[row] = input.observations[sample * 4 + 1];
    centered[row] = input.observations[sample * 4] - (options.mean ?? 0);
  });
  positions.set(input.queries, count * 2);
  result.status.fill(2);
  if (!centered.every(Number.isFinite)) return result;
  const modelCode = getPredictionCovarianceModelCode(options.covariance);
  const covarianceScale = options.covariance.sill + (options.covariance.nugget ?? 0);
  const covariance = (first: number, second: number): number => {
    const differenceX = positions[first * 2] - positions[second * 2];
    const differenceY = positions[first * 2 + 1] - positions[second * 2 + 1];
    return (
      options.covariance.sill *
        getCPUVecchiaCorrelation(
          modelCode,
          Math.sqrt(differenceX * differenceX + differenceY * differenceY) /
            options.covariance.range
        ) +
      (differenceX === 0 && differenceY === 0 ? (options.covariance.nugget ?? 0) : 0)
    );
  };
  const lower = new Float64Array(size * size);
  const aliases = new Float64Array(queryCount).fill(-1);
  const tolerance = 64 * Number.EPSILON * size * covarianceScale;
  for (let row = 0; row < size; row++) {
    if (row < count && errors[row] === 0) {
      // Two noiseless observations of the same latent variable make the observation block
      // singular, even if a floating-point diagonal subtraction leaves a tiny positive residue.
      for (let previous = 0; previous < row; previous++) {
        if (
          errors[previous] === 0 &&
          positions[row * 2] === positions[previous * 2] &&
          positions[row * 2 + 1] === positions[previous * 2 + 1]
        )
          return result;
      }
    }
    if (row >= count) {
      for (let candidate = 0; candidate < row; candidate++) {
        if (
          positions[row * 2] === positions[candidate * 2] &&
          positions[row * 2 + 1] === positions[candidate * 2 + 1] &&
          (candidate >= count || errors[candidate] === 0)
        ) {
          aliases[row - count] = candidate;
          for (let column = 0; column <= candidate; column++)
            lower[row * size + column] = lower[candidate * size + column];
          break;
        }
      }
      if (aliases[row - count] >= 0) continue;
    }
    for (let column = 0; column <= row; column++) {
      let value = covariance(row, column) + (row === column ? errors[row] : 0);
      for (let component = 0; component < column; component++)
        value -= lower[row * size + component] * lower[column * size + component];
      if (!Number.isFinite(value)) return result;
      if (row === column) {
        if (row < count ? !(value > 0) : value < -tolerance) return result;
        lower[row * size + column] = Math.sqrt(Math.max(value, 0));
      } else if (lower[column * size + column] === 0) {
        if (Math.abs(value) > tolerance) return result;
      } else {
        lower[row * size + column] = value / lower[column * size + column];
        if (!Number.isFinite(lower[row * size + column])) return result;
      }
    }
  }
  const whitened = centered.slice();
  for (let row = 0; row < count; row++) {
    for (let column = 0; column < row; column++)
      whitened[row] -= lower[row * size + column] * whitened[column];
    whitened[row] /= lower[row * size + row];
    if (!Number.isFinite(whitened[row])) return result;
  }
  const moments = new Float64Array(queryCount * 2);
  for (let query = 0; query < queryCount; query++) {
    const row = count + query;
    let expectation = options.mean ?? 0;
    let variance = 0;
    for (let column = 0; column < count; column++)
      expectation += lower[row * size + column] * whitened[column];
    for (let column = count; column <= row; column++) variance += lower[row * size + column] ** 2;
    const alias = aliases[query];
    if (alias >= 0 && alias < count) {
      expectation = input.observations[observed[alias] * 4];
      variance = 0;
    } else if (alias >= count) {
      expectation = moments[(alias - count) * 2];
      variance = moments[(alias - count) * 2 + 1];
    }
    if (!Number.isFinite(expectation) || !Number.isFinite(variance)) return result;
    moments.set([expectation, variance], query * 2);
  }
  result.moments.set(moments);
  for (let replicate = 0; replicate < options.replicateCount; replicate++) {
    const normal = Float64Array.from({length: queryCount}, (_, query) => {
      const [radiusWord, angleWord] = getPhilox4x32(
        [replicate + (options.replicateOffset ?? 0), query, 0, 0],
        [options.seed, PHILOX_PURPOSES.conditionalGaussianSimulation.id]
      );
      return (
        Math.sqrt(-2 * Math.log(getPhiloxOpenUnitFloat(radiusWord))) *
        Math.cos((2 * Math.PI * ((angleWord >>> 8) + 0.5)) / 16777216)
      );
    });
    const values = new Float64Array(queryCount);
    for (let query = 0; query < queryCount; query++) {
      let value = moments[query * 2];
      for (let component = 0; component <= query; component++)
        value += lower[(count + query) * size + count + component] * normal[component];
      const alias = aliases[query];
      if (alias >= count) value = values[alias - count];
      values[query] = value;
    }
    if (!values.every(Number.isFinite)) continue;
    result.values.set(values, replicate * queryCount);
    result.status[replicate] = 0;
  }
  return result;
}

/** CPU simulation without the GPU joint-row cap or retained preparation state. */
export class CPUConditionalGaussianSimulationOperation extends GPUCPUOperation {
  constructor(props: ConditionalGaussianSimulationOperationProps) {
    props = snapshotConditionalGaussianSimulationOperationProps(props);
    const id = props.id ?? 'conditional-gaussian-simulation';
    validateCapacity(id, 'observationCount', props.observationCount, 0);
    validateCapacity(id, 'queryCount', props.queryCount);
    validateCapacity(id, 'queryCount * replicateCount', props.queryCount * props.replicateCount);
    validateSimulationOptions(id, props);
    resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    const operation = new GPUConditionalGaussianSimulationOperation(props);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const positions = getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2');
        const queries = getCPUOperationFloat32Rows(id, inputs, 'queries', 'float32x2');
        const observations = getCPUOperationFloat32Rows(id, inputs, 'observations', 'float32x4');
        validateRequirement(
          id,
          observations.length === props.observationCount * 4,
          'observations length must equal observationCount * 4'
        );
        validateRequirement(
          id,
          !(positions.length !== props.observationCount * 2),
          `positions length must equal observationCount * 2`
        );
        validateRequirement(
          id,
          !(queries.length !== props.queryCount * 2),
          `queries length must equal queryCount * 2`
        );
        const result = computeConditionalGaussianSimulationOnCPU(
          {
            positions,
            queries,
            observations
          },
          props
        );
        return {
          [props.output.values]: result.values,
          [props.output.moments]: result.moments,
          [props.output.status]: result.status
        };
      }
    });
  }
}
