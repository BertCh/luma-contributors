// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUVecchiaLogLikelihoodOperation,
  type VecchiaLogLikelihoodOperationProps
} from './gpu-vecchia-log-likelihood-operation';
import {
  evaluateCPUVecchiaLikelihood,
  validateCPUVecchiaProps,
  type CPUVecchiaRows,
  type CPUVecchiaLikelihoodRows
} from './cpu-vecchia-evaluation';

/** Packed input in the contributor's view layouts, including its per-frame parameter buffer. */
export type CPUVecchiaLogLikelihoodInput = CPUVecchiaRows;
/** Every output port, in the same row order and formats as the contributor. */
export type CPUVecchiaLogLikelihoodResult = CPUVecchiaLikelihoodRows;

/**
 * Evaluates zero-mean Gaussian conditional predictions and likelihood in float64, rounding each
 * published array once. Conditioning errors affect the solve diagonal; the site's error is added
 * only to its likelihood variance. Excluded sites still receive conditional prediction rows.
 *
 * What differs from the GPU contributor:
 * - All usable conditioning slots are solved, without the GPU's neighborCount truncation or
 *   32-neighbor private-array limit; stable distance order is preserved.
 * - Float64 solves keep the GPU's equilibrated pivot diagnostic and reliability thresholds.
 *   Near a threshold, rounding can change a status. Compare ordinary outputs at relative 1e-4.
 *   For conditional means, cancellation can amplify relative error: comparisons also allow an
 *   absolute error of 2 * 2^-24 * sum(abs(weight * observation)), using independent oracle weights.
 * - Matern correlations use the full closed form instead of a short-lag truncated series.
 * - Totals sum unrounded float64 site terms, then split into float32 high/low words; the GPU sums
 *   float32 site terms with a fixed-point accumulator. Compare decoded totals at relative 1e-4.
 * - Nonfinite coordinates are rejected by the conditional solve; an empty CSR site's likelihood
 *   still follows the GPU's marginal branch. Float64-finite results outside float32 range publish
 *   infinities on rounding; subnormal outputs are retained. These are outside GPU comparisons.
 * - Malformed CSR spans retain the spatial kernel's conditionalStatus 1 result.
 *
 * @throws If positions, observations, parameters or neighborhoodOffsets have inconsistent lengths.
 */
export function computeVecchiaLogLikelihoodOnCPU(
  input: CPUVecchiaLogLikelihoodInput
): CPUVecchiaLogLikelihoodResult {
  return evaluateCPUVecchiaLikelihood(input);
}

/** CPU twin; GPU-only neighborhood capacity is dropped. Counts and contexts are compile-time. */
export class CPUVecchiaLogLikelihoodOperation extends GPUCPUOperation {
  constructor(props: VecchiaLogLikelihoodOperationProps) {
    const id = props.id ?? 'vecchia-log-likelihood';
    validateCPUVecchiaProps(id, props);
    const operation = new GPUVecchiaLogLikelihoodOperation(props);
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
        const result = computeVecchiaLogLikelihoodOnCPU({
          positions,
          observations: getCPUOperationFloat32Rows(id, inputs, 'observations', 'float32x4'),
          neighborhoodOffsets: getCPUOperationUint32Rows(id, inputs, 'neighborhoodOffsets'),
          neighborhoods: getCPUOperationUint32Rows(id, inputs, 'neighborhoods'),
          parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters')
        });
        return Object.fromEntries(
          Object.entries(props.output).map(([port, value]) => [
            value,
            result[port as keyof CPUVecchiaLogLikelihoodResult]
          ])
        );
      }
    });
  }
}
