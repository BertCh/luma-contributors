// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {resolveGPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
import {
  GPUVecchiaSpaceTimeLogLikelihoodOperation,
  type VecchiaSpaceTimeLogLikelihoodOperationProps
} from './gpu-vecchia-space-time-log-likelihood-operation';
import {
  evaluateCPUVecchiaLikelihood,
  validateCPUVecchiaProps,
  type CPUVecchiaRows,
  type CPUVecchiaLikelihoodRows
} from './cpu-vecchia-evaluation';

/** Packed input in the contributor's view layouts, including its per-frame parameter buffer. */
export type CPUVecchiaSpaceTimeLogLikelihoodInput = CPUVecchiaRows & {times: Float32Array};
/** Every output port, in the same row order and formats as the contributor. */
export type CPUVecchiaSpaceTimeLogLikelihoodResult = CPUVecchiaLikelihoodRows;

/**
 * Evaluates zero-mean Gaussian conditional predictions and likelihood in float64, rounding each
 * published array once. Conditioning errors affect the solve diagonal; the site's error is added
 * only to its likelihood variance. Excluded sites still receive conditional prediction rows.
 * Finite negative structure codes in hand-written buffers select spherical covariance, following
 * the GPU's clamped u32 conversion; the normal parameter packer does not produce these codes.
 *
 * What differs from the GPU contributor:
 * - All usable conditioning slots are solved, without the GPU's neighborCount truncation or
 *   32-neighbor private-array limit; CSR visit order is preserved.
 * - Float64 solves keep the GPU's equilibrated pivot diagnostic and reliability thresholds.
 *   Near a threshold, rounding can change a status. Compare ordinary outputs at relative 1e-4.
 * - Matern correlations use the full closed form instead of a short-lag truncated series.
 * - Totals sum unrounded float64 site terms, then split into float32 high/low words; the GPU sums
 *   float32 site terms with a fixed-point accumulator. Compare decoded totals at relative 1e-4.
 * - Nonfinite coordinates are rejected by the conditional solve; an empty CSR site's likelihood
 *   still follows the GPU's marginal branch. Float64-finite results outside float32 range publish
 *   infinities on rounding; subnormal outputs are retained. These are outside GPU comparisons.
 * - Malformed CSR spans throw an Error naming neighborhoodOffsets (the GPU has no explicit check).
 * - A NaN structure code selects the Matern 5/2 default branch on CPU; WGSL's NaN-to-u32 result
 *   is indeterminate, so such parameter buffers are outside GPU comparisons.
 *
 * @throws If positions, observations, times, parameters or neighborhoodOffsets have inconsistent lengths.
 */
export function computeVecchiaSpaceTimeLogLikelihoodOnCPU(
  input: CPUVecchiaSpaceTimeLogLikelihoodInput
): CPUVecchiaSpaceTimeLogLikelihoodResult {
  return evaluateCPUVecchiaLikelihood(input, input.times);
}

/** CPU twin; GPU-only neighborhood capacity is dropped. Counts and contexts are compile-time. */
export class CPUVecchiaSpaceTimeLogLikelihoodOperation extends GPUCPUOperation {
  constructor(props: VecchiaSpaceTimeLogLikelihoodOperationProps) {
    const id = props.id ?? 'vecchia-space-time-log-likelihood';
    validateCPUVecchiaProps(id, props);
    resolveGPUTemporalContext(id, props.temporalContext, {encodings: ['float32-relative']});
    const operation = new GPUVecchiaSpaceTimeLogLikelihoodOperation(props);
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
        const result = computeVecchiaSpaceTimeLogLikelihoodOnCPU({
          positions,
          times: getCPUOperationFloat32Rows(id, inputs, 'times'),
          observations: getCPUOperationFloat32Rows(id, inputs, 'observations', 'float32x4'),
          neighborhoodOffsets: getCPUOperationUint32Rows(id, inputs, 'neighborhoodOffsets'),
          neighborhoods: getCPUOperationUint32Rows(id, inputs, 'neighborhoods'),
          parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters')
        });
        return Object.fromEntries(
          Object.entries(props.output).map(([port, value]) => [
            value,
            result[port as keyof CPUVecchiaSpaceTimeLogLikelihoodResult]
          ])
        );
      }
    });
  }
}
