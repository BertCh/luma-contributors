// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {GPU_VECCHIA_FISHER_CELL_COUNT} from './gpu-vecchia-fisher-scoring';
import {
  GPUVecchiaFisherScoringOperation,
  type VecchiaFisherScoringOperationProps
} from './gpu-vecchia-fisher-scoring-operation';
import {MATERN_3_2_RANGE_FACTOR} from './prediction-covariance';
import {
  getCPUVecchiaCovariance,
  getCPUVecchiaCorrelation,
  getCPUVecchiaNeighbors,
  isCPUVecchiaObservationUsable,
  isCPUVecchiaPositionUsable,
  solveCPUVecchiaConditioning,
  sumCPUVecchiaCells,
  validateCPUVecchiaProps,
  validateCPUVecchiaRows,
  type CPUVecchiaRows
} from './cpu-vecchia-evaluation';

/** Packed point rows, observations, CSR conditioning sets and per-frame covariance parameters. */
export type CPUVecchiaFisherScoringInput = CPUVecchiaRows;
/** Same flat ten-cell per-site and twenty-word total layouts as GPUVecchiaFisherScoring. */
export type CPUVecchiaFisherScoringResult = {
  siteTerms: Float32Array;
  siteStatus: Uint32Array;
  totals: Float32Array;
  status: Uint32Array;
};

/**
 * Evaluates the Gaussian conditional likelihood, log-parameter score and expected Fisher
 * information. Parameter order is log sill, log effective range, log nugget; information is
 * I00, I01, I02, I11, I12, I22. Measurement-error variances are fixed, not differentiated.
 *
 * What differs from the GPU contributor:
 * - All usable CSR slots are used, without neighborCount truncation or the 32-neighbor GPU limit.
 * - Float64 arithmetic replaces intermediate float32; equilibrated reliability thresholds are
 *   retained, so near-threshold statuses can differ. Compare ordinary outputs at relative 1e-4.
 * - Matern correlations use the full closed form instead of the GPU's short-lag Taylor series.
 * - Totals sum unrounded float64 site cells and split into high/low float32 words. The GPU sums
 *   rounded site cells with a fixed-point accumulator. Compare decoded totals at relative 1e-4.
 * - Malformed CSR spans throw an Error naming neighborhoodOffsets; the GPU has no explicit check.
 * - Nonfinite site coordinates fail with status 2. Float64-finite results beyond float32 range
 *   publish infinities on rounding, and subnormal outputs are retained; these are not compared.
 *
 * @throws If positions, observations, neighborhoodOffsets or parameters have inconsistent lengths.
 */
export function computeVecchiaFisherScoringOnCPU(
  input: CPUVecchiaFisherScoringInput
): CPUVecchiaFisherScoringResult {
  const rowCount = validateCPUVecchiaRows(input);
  const model = getCPUVecchiaCovariance(input);
  const cellCount = GPU_VECCHIA_FISHER_CELL_COUNT;
  const siteTerms = new Float64Array(rowCount * cellCount).fill(NaN);
  const siteStatus = new Uint32Array(rowCount);
  const status = new Uint32Array(3);
  const derivative = (parameter: number, first: number, second: number): number => {
    const differenceX = input.positions[first * 2] - input.positions[second * 2];
    const differenceY = input.positions[first * 2 + 1] - input.positions[second * 2 + 1];
    const scaled = Math.sqrt(differenceX * differenceX + differenceY * differenceY) / model.range;
    if (parameter === 0) return model.sill * getCPUVecchiaCorrelation(model.modelCode, scaled);
    if (parameter === 2) return differenceX === 0 && differenceY === 0 ? model.nugget : 0;
    const modelCode = Math.trunc(model.modelCode);
    if (modelCode !== 1 && modelCode !== 3) return NaN;
    const argument = (modelCode === 1 ? 3 : MATERN_3_2_RANGE_FACTOR) * scaled;
    return argument === Infinity
      ? 0
      : model.sill * (modelCode === 1 ? argument : argument * argument) * Math.exp(-argument);
  };
  for (let site = 0; site < rowCount; site++) {
    if (!isCPUVecchiaObservationUsable(input.observations, site)) {
      siteStatus[site] = 1;
      status[1]++;
      continue;
    }
    siteStatus[site] = 2;
    if (!model.valid || !isCPUVecchiaPositionUsable(input.positions, site)) {
      status[0]++;
      continue;
    }
    const rows = getCPUVecchiaNeighbors(input, site);
    const cross = Float64Array.from(rows, row => model.covariance(row, site));
    const solved = solveCPUVecchiaConditioning(input, rows, model.covariance, model.scale, cross);
    if (!solved.solution) {
      status[0]++;
      continue;
    }
    const weights = solved.solution;
    let variance = model.covariance(site, site) + input.observations[site * 4 + 1];
    let prediction = 0;
    rows.forEach((row, index) => {
      variance -= weights[index] * cross[index];
      prediction += weights[index] * input.observations[row * 4];
    });
    if (!(variance > 0) || !Number.isFinite(variance)) {
      siteStatus[site] = 3;
      status[0]++;
      continue;
    }
    const residual = input.observations[site * 4] - prediction;
    const varianceDerivatives = new Float64Array(3);
    const residualDerivatives = new Float64Array(3);
    const vectors: Float64Array[] = [];
    const solutions: Float64Array[] = [];
    let failed = false;
    for (let parameter = 0; parameter < 3; parameter++) {
      let varianceDerivative = derivative(parameter, site, site);
      const vector = Float64Array.from(rows, (row, index) => {
        const crossDerivative = derivative(parameter, row, site);
        let value = crossDerivative;
        rows.forEach((other, column) => {
          value -= derivative(parameter, row, other) * weights[column];
        });
        varianceDerivative -= (crossDerivative + value) * weights[index];
        return value;
      });
      varianceDerivatives[parameter] = varianceDerivative;
      const derivativeSolve = solveCPUVecchiaConditioning(
        input,
        rows,
        model.covariance,
        model.scale,
        vector
      );
      if (!derivativeSolve.solution) {
        failed = true;
        break;
      }
      vectors.push(vector);
      solutions.push(derivativeSolve.solution);
      rows.forEach((row, index) => {
        residualDerivatives[parameter] -=
          derivativeSolve.solution![index] * input.observations[row * 4];
      });
    }
    if (failed) {
      status[0]++;
      continue;
    }
    const standardized = (residual * residual) / variance;
    const output = new Float64Array(cellCount);
    output[0] = -0.5 * (Math.log(2 * Math.PI) + Math.log(variance) + standardized);
    for (let parameter = 0; parameter < 3; parameter++) {
      output[1 + parameter] =
        -0.5 * (varianceDerivatives[parameter] / variance) * (1 - standardized) -
        (residual * residualDerivatives[parameter]) / variance;
    }
    let cell = 4;
    for (let first = 0; first < 3; first++) {
      for (let second = first; second < 3; second++) {
        let quadratic = 0;
        for (let row = 0; row < rows.length; row++)
          quadratic += vectors[first][row] * solutions[second][row];
        // Divide before multiplying to avoid overflowing an otherwise finite information value.
        output[cell++] =
          0.5 * (varianceDerivatives[first] / variance) * (varianceDerivatives[second] / variance) +
          quadratic / variance;
      }
    }
    if (!output.every(Number.isFinite)) {
      siteStatus[site] = 3;
      status[0]++;
      continue;
    }
    siteTerms.set(output, site * cellCount);
    siteStatus[site] = 0;
    status[2]++;
  }
  return {
    siteTerms: Float32Array.from(siteTerms),
    siteStatus,
    totals: sumCPUVecchiaCells(siteTerms, cellCount, status),
    status
  };
}

/** CPU Fisher evaluation without the GPU-only conditioning capacity. */
export class CPUVecchiaFisherScoringOperation extends GPUCPUOperation {
  constructor(props: VecchiaFisherScoringOperationProps) {
    const id = props.id ?? 'vecchia-fisher-scoring';
    validateCPUVecchiaProps(id, props);
    const operation = new GPUVecchiaFisherScoringOperation(props);
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
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions length must equal rowCount * 2`);
        const result = computeVecchiaFisherScoringOnCPU({
          positions,
          observations: getCPUOperationFloat32Rows(id, inputs, 'observations', 'float32x4'),
          neighborhoodOffsets: getCPUOperationUint32Rows(id, inputs, 'neighborhoodOffsets'),
          neighborhoods: getCPUOperationUint32Rows(id, inputs, 'neighborhoods'),
          parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters')
        });
        return Object.fromEntries(
          Object.entries(props.output).map(([port, value]) => [
            value,
            result[port as keyof CPUVecchiaFisherScoringResult]
          ])
        );
      }
    });
  }
}
