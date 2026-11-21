// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getGPUColumnQuantilesParameterLength,
  getGPUColumnQuantilesRationalProbabilityLength,
  GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH,
  GPU_COLUMN_QUANTILE_INTERPOLATION_CODES
} from './column-quantiles-parameters';
import {
  GPUColumnQuantilesOperation,
  snapshotColumnQuantilesOperationProps,
  type ColumnQuantilesOperationProps
} from './gpu-column-quantiles-operation';

export type CPUColumnQuantilesInput = {
  values: Float32Array;
  mask?: Uint32Array;
  parameters: Float32Array;
  rationalProbabilities?: Uint32Array;
};

export type CPUColumnQuantilesResult = {
  quantiles: Float32Array;
  validCount: Uint32Array;
  filterBounds: Float32Array;
  outputMask: Uint32Array;
};

/**
 * Sorts the finite, unmasked float32 sample and computes order statistics in float64.
 * The parameter header and probability pairs have the contributor packers' layouts.
 *
 * What differs from the GPU contributor:
 * - At most 16,777,216 valid rows: packed float ranks and filter products round to float32,
 *   as on GPU. Above that GPU row limit, CPU rank products retain float64 precision.
 *   Interpolation uses float64 with one float32 output rounding; rational ranks use exact integers.
 * - The GPU's 64-quantile and float32 row-count limits are dropped; uint32 output counts remain.
 * - Non-finite values are missing and zero is canonicalized to +0, as on the GPU. Subnormal values
 *   are retained; devices may flush them to zero. Unknown interpolation codes use lower, as the kernel does.
 *
 * @throws If mask, parameters or rationalProbabilities have inconsistent lengths.
 */
export function computeColumnQuantilesOnCPU(
  input: CPUColumnQuantilesInput,
  quantileCount: number
): CPUColumnQuantilesResult {
  validateCapacity('computeColumnQuantilesOnCPU', 'quantileCount', quantileCount, 1);
  const {values, mask, parameters, rationalProbabilities} = input;
  validateCapacity('computeColumnQuantilesOnCPU', 'values.length', values.length, 0);
  if (mask && mask.length !== values.length)
    throw new Error('mask length must equal values length');
  if (parameters.length < getGPUColumnQuantilesParameterLength(quantileCount)) {
    throw new Error('parameters must hold a header and quantileCount probabilities');
  }
  if (
    rationalProbabilities &&
    rationalProbabilities.length < getGPUColumnQuantilesRationalProbabilityLength(quantileCount)
  ) {
    throw new Error('rationalProbabilities must hold quantileCount numerator/denominator pairs');
  }
  const sorted: number[] = [];
  for (let row = 0; row < values.length; row++) {
    if (Number.isFinite(values[row]) && (!mask || mask[row] !== 0))
      sorted.push(values[row] === 0 ? 0 : values[row]);
  }
  sorted.sort((first, second) => first - second);
  const count = sorted.length;
  // GPU float-rank/filter routes support at most 2^24 rows. Larger CPU samples retain
  // wide products so even endpoint ranks cannot round down or beyond the sample.
  const usePackedFloatRanks = count <= 2 ** 24;
  const quantiles = new Float32Array(quantileCount).fill(NaN);
  const filterBounds = new Float32Array([NaN, NaN]);
  const outputMask = new Uint32Array(values.length);
  const [rawCode, rawLower, rawUpper] = parameters;
  // WGSL converts the nonnegative integer codes supplied by the packer to u32.
  const code = Math.min(0xffffffff, Math.max(0, Math.trunc(rawCode))) || 0;
  for (let index = 0; index < quantileCount && count > 0; index++) {
    let lower: number;
    let fraction: number;
    if (rationalProbabilities) {
      const [numerator, denominator] = rationalProbabilities.subarray(2 * index, 2 * index + 2);
      if (denominator === 0 || numerator > denominator) continue;
      const product = BigInt(count - 1) * BigInt(numerator);
      lower = Number(product / BigInt(denominator));
      fraction = Number(product % BigInt(denominator)) / denominator;
    } else {
      const probability = parameters[GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH + index];
      if (!Number.isFinite(probability) || probability < 0 || probability > 1) continue;
      const product = (count - 1) * probability;
      const height = usePackedFloatRanks ? Math.fround(product) : product;
      lower = Math.floor(height);
      fraction = height - lower;
    }
    const upper = Math.min(lower + (fraction > 0 ? 1 : 0), count - 1);
    let result = sorted[lower];
    if (code === GPU_COLUMN_QUANTILE_INTERPOLATION_CODES.higher) result = sorted[upper];
    else if (code === GPU_COLUMN_QUANTILE_INTERPOLATION_CODES.nearest) {
      result = sorted[lower + (fraction > 0.5 || (fraction === 0.5 && lower % 2 === 1) ? 1 : 0)];
    } else if (code === GPU_COLUMN_QUANTILE_INTERPOLATION_CODES.linear) {
      result += (sorted[Math.min(lower + 1, count - 1)] - result) * fraction;
    } else if (code === GPU_COLUMN_QUANTILE_INTERPOLATION_CODES.midpoint)
      result = (result + sorted[upper]) / 2;
    quantiles[index] = result;
  }
  if (count > 0) {
    const lowerFraction = Number.isNaN(rawLower) ? 0 : Math.max(0, Math.min(1, rawLower));
    const upperFraction = Number.isNaN(rawUpper) ? 1 : Math.max(0, Math.min(1, rawUpper));
    const lowerProduct = count * lowerFraction;
    const upperProduct = count * upperFraction;
    const lowerIndex = Math.min(
      count - 1,
      Math.max(0, Math.floor(usePackedFloatRanks ? Math.fround(lowerProduct) : lowerProduct))
    );
    const upperIndex = Math.min(
      count - 1,
      Math.max(0, Math.ceil(usePackedFloatRanks ? Math.fround(upperProduct) : upperProduct) - 1)
    );
    filterBounds.set([sorted[lowerIndex], sorted[upperIndex]]);
    for (let row = 0; row < values.length; row++) {
      outputMask[row] = Number(
        Number.isFinite(values[row]) &&
          (!mask || mask[row] !== 0) &&
          values[row] >= filterBounds[0] &&
          values[row] <= filterBounds[1]
      );
    }
  }
  return {quantiles, validCount: Uint32Array.of(count), filterBounds, outputMask};
}

/** CPU twin; drops the GPU radix slot and float32 rank limits. */
export class CPUColumnQuantilesOperation extends GPUCPUOperation {
  constructor(inputProps: ColumnQuantilesOperationProps) {
    const props = snapshotColumnQuantilesOperationProps(inputProps);
    const id = props.id ?? 'column-quantiles';
    validateCapacity(id, 'rowCount', props.rowCount, 1);
    validateCapacity(id, 'quantileCount', props.quantileCount, 1);
    new GPUColumnQuantilesOperation(props);
    super({
      id,
      inputs: {
        values: props.values,
        parameters: props.parameters,
        ...(props.mask !== undefined ? {mask: props.mask} : {}),
        ...(props.rationalProbabilities !== undefined
          ? {rationalProbabilities: props.rationalProbabilities}
          : {})
      },
      outputs: {
        [props.output.quantiles]: gpuGraph.vector('float32', {length: props.quantileCount}),
        [props.output.validCount]: gpuGraph.vector('uint32', {length: 1}),
        ...(props.output.outputMask !== undefined
          ? {[props.output.outputMask]: gpuGraph.vector('uint32', {length: props.rowCount})}
          : {}),
        ...(props.output.filterBounds !== undefined
          ? {[props.output.filterBounds]: gpuGraph.vector('float32', {length: 2})}
          : {})
      },
      execute({inputs}) {
        const values = getCPUOperationFloat32Rows(id, inputs, 'values');
        if (values.length !== props.rowCount)
          throw new Error(`${id} values length must equal rowCount`);
        const result = computeColumnQuantilesOnCPU(
          {
            values,
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters'),
            mask:
              props.mask !== undefined ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined,
            rationalProbabilities:
              props.rationalProbabilities !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'rationalProbabilities')
                : undefined
          },
          props.quantileCount
        );
        return {
          [props.output.quantiles]: result.quantiles,
          [props.output.validCount]: result.validCount,
          ...(props.output.outputMask !== undefined
            ? {[props.output.outputMask]: result.outputMask}
            : {}),
          ...(props.output.filterBounds !== undefined
            ? {[props.output.filterBounds]: result.filterBounds}
            : {})
        };
      }
    });
  }
}
