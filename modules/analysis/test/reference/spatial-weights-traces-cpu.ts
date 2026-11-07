// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Float64 CPU reference oracle for GPUSpatialWeightsTraces. Test-only; not part of the package API.

import {
  GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT,
  getSpatialWeightsTracesProbeValue,
  type GPUSpatialWeightsTracesMode,
  type SpatialWeightsCSR
} from '../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';

/** Options of {@link getSpatialWeightsTracesOnCPU}. */
export type SpatialWeightsTracesCPUOptions = {
  /** `'exact'` (default) uses unit probes; `'stochastic'` uses Rademacher probes. */
  mode?: GPUSpatialWeightsTracesMode;
  /** Probes of the stochastic mode. Defaults to 64. */
  probeCount?: number;
  /** Philox seed of the stochastic probes. Defaults to 0. */
  seed?: number;
  /** Also compute the Gram traces. */
  includeGramTraces?: boolean;
};

/** Result of {@link getSpatialWeightsTracesOnCPU}. */
export type SpatialWeightsTracesCPUResult = {
  /** `tr((W / s)^q)` for `q = 1..maximumPower`, `s = infinityNorm`. */
  powerTraces: Float64Array;
  /** Standard error of the mean of `powerTraces` across probes (0 in exact mode). */
  powerTraceStandardErrors: Float64Array;
  /** `1' (W / s)^q 1` for `q = 1..maximumPower` (always exact). */
  powerSums: Float64Array;
  /** `[tr(W'W), tr(W'W W), tr((W'W)^2), sum c_i^2]` of the unscaled `W`, when requested. */
  gramTraces?: Float64Array;
  /** `s = ||W||_inf`, the largest absolute row sum (1 when `W` has no links). */
  infinityNorm: number;
};

/**
 * `||W||_inf`, the largest absolute row sum over slots whose neighbor id is `< rows`; 1 when it
 * is zero or not finite (the scale `GPUSpatialWeightsTraces` divides by).
 */
export function getSpatialWeightsTracesInfinityNorm(weights: SpatialWeightsCSR): number {
  const rows = weights.offsets.length - 1;
  let largest = 0;
  for (let row = 0; row < rows; row++) {
    let sum = 0;
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      if (weights.neighbors[slot] < rows) {
        sum += Math.abs(weights.weights[slot]);
      }
    }
    largest = Math.max(largest, sum);
  }
  return largest > 0 && Number.isFinite(largest) ? largest : 1;
}

/**
 * Float64 mirror of `GPUSpatialWeightsTraces` on sparse CSR weights. Like the GPU node it runs the
 * chains with `W / s`, `s = ||W||_inf` (returned as `infinityNorm`), so `powerTraces` and
 * `powerSums` are those of the scaled matrix; the Gram traces are of the unscaled `W`.
 *
 * `'exact'` (default) runs one unit-vector chain per row (`O(rows * maximumPower * nnz)`);
 * `'stochastic'` runs `probeCount` seeded Rademacher chains and averages, using the same probes
 * as the GPU. `powerSums` is always one chain from the ones vector. Slots whose neighbor id is
 * `>= rows` are dropped, as the GPU kernels do; self-loops are allowed.
 */
export function getSpatialWeightsTracesOnCPU(
  weights: SpatialWeightsCSR,
  maximumPower: number,
  options: SpatialWeightsTracesCPUOptions = {}
): SpatialWeightsTracesCPUResult {
  const rows = weights.offsets.length - 1;
  const stochastic = options.mode === 'stochastic';
  const probeTotal = stochastic
    ? (options.probeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT)
    : rows;
  const seed = (options.seed ?? 0) >>> 0;
  const includeGram = options.includeGramTraces === true;
  const infinityNorm = getSpatialWeightsTracesInfinityNorm(weights);
  const stepCount = includeGram ? Math.max(maximumPower, 2) : maximumPower;
  const multiply = (vector: Float64Array): Float64Array => {
    const result = new Float64Array(rows);
    for (let row = 0; row < rows; row++) {
      let sum = 0;
      for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
        const neighbor = weights.neighbors[slot];
        if (neighbor < rows) {
          sum += (weights.weights[slot] / infinityNorm) * vector[neighbor];
        }
      }
      result[row] = sum;
    }
    return result;
  };
  const multiplyTranspose = (vector: Float64Array): Float64Array => {
    const result = new Float64Array(rows);
    for (let row = 0; row < rows; row++) {
      for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
        const neighbor = weights.neighbors[slot];
        if (neighbor < rows) {
          result[neighbor] += weights.weights[slot] * vector[row];
        }
      }
    }
    return result;
  };
  const powerTraces = new Float64Array(maximumPower);
  const dotSquares = new Float64Array(maximumPower);
  let tracePairs = 0;
  let traceSquare = 0;
  for (let probe = 0; probe < probeTotal; probe++) {
    const z = new Float64Array(rows);
    if (stochastic) {
      for (let row = 0; row < rows; row++) {
        z[row] = getSpatialWeightsTracesProbeValue(seed, probe, row);
      }
    } else {
      z[probe] = 1;
    }
    let vector: Float64Array = z;
    let first: Float64Array = z;
    for (let step = 1; step <= stepCount; step++) {
      const next = multiply(vector);
      if (step <= maximumPower) {
        let dot = 0;
        for (let row = 0; row < rows; row++) {
          dot += z[row] * next[row];
        }
        powerTraces[step - 1] += dot;
        dotSquares[step - 1] += dot * dot;
      }
      if (step === 1) {
        first = next;
      } else if (includeGram && step === 2) {
        for (let row = 0; row < rows; row++) {
          tracePairs += first[row] * next[row];
        }
        const transposed = multiplyTranspose(first);
        for (let row = 0; row < rows; row++) {
          traceSquare += transposed[row] * transposed[row];
        }
      }
      vector = next;
    }
  }
  const scale = stochastic ? 1 / probeTotal : 1;
  const powerTraceStandardErrors = new Float64Array(maximumPower);
  for (let index = 0; index < maximumPower; index++) {
    powerTraces[index] *= scale;
    if (stochastic) {
      const variance =
        (dotSquares[index] - probeTotal * powerTraces[index] ** 2) / (probeTotal - 1);
      powerTraceStandardErrors[index] = Math.sqrt(Math.max(variance, 0) / probeTotal);
    }
  }
  const powerSums = new Float64Array(maximumPower);
  let ones: Float64Array = new Float64Array(rows).fill(1);
  for (let step = 0; step < maximumPower; step++) {
    ones = multiply(ones);
    let sum = 0;
    for (let row = 0; row < rows; row++) {
      sum += ones[row];
    }
    powerSums[step] = sum;
  }
  if (!includeGram) {
    return {powerTraces, powerTraceStandardErrors, powerSums, infinityNorm};
  }
  let traceGram = 0;
  const columnSquares = new Float64Array(rows);
  for (let row = 0; row < rows; row++) {
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      const neighbor = weights.neighbors[slot];
      if (neighbor < rows) {
        const square = weights.weights[slot] * weights.weights[slot];
        traceGram += square;
        columnSquares[neighbor] += square;
      }
    }
  }
  let sumColumnSquares = 0;
  for (let column = 0; column < rows; column++) {
    sumColumnSquares += columnSquares[column] * columnSquares[column];
  }
  return {
    powerTraces,
    powerTraceStandardErrors,
    powerSums,
    infinityNorm,
    // The chains hold W / s: tr(W'W W) = s^3 E[v1' v2] and tr((W'W)^2) = s^2 E[|W' v1|^2].
    gramTraces: Float64Array.of(
      traceGram,
      tracePairs * scale * infinityNorm ** 3,
      traceSquare * scale * infinityNorm ** 2,
      sumColumnSquares
    )
  };
}
