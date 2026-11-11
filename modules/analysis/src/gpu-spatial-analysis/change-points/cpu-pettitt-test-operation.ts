// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {GPU_PETTITT_TEST_NO_CHANGE_SLICE} from './gpu-pettitt-test';
import {
  GPUPettittTestOperation,
  type PettittTestOperationProps
} from './gpu-pettitt-test-operation';

/** Largest series for which floor(n² / 4), the largest possible K, fits a uint32 output. */
export const CPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT = 131071;

export type CPUPettittTestOptions = Pick<
  PettittTestOperationProps,
  'cellCount' | 'sliceCount' | 'layout'
>;

export type CPUPettittTestResult = {
  statistic: Uint32Array;
  changeSlice: Uint32Array;
  pValues: Float32Array;
};

/**
 * Computes Pettitt's statistic from average ranks, with the contributor's first-maximum rule.
 * Non-finite values are skipped. A constant series of length >= 2 returns its second finite slice.
 *
 * What differs from the GPU contributor:
 * - Uses float64 ranks and the p-value formula, rounding p-values once; compare within 2e-6 absolute.
 * - Replaces quadratic pair comparisons by sorting and rank prefixes.
 * - Accepts up to 131071 slices instead of 512; the retained limit protects the uint32 statistic.
 */
export function computePettittTestOnCPU(
  values: Float32Array,
  options: CPUPettittTestOptions
): CPUPettittTestResult {
  const {cellCount, sliceCount, layout} = options;
  validateCapacity('computePettittTestOnCPU', 'cellCount', cellCount);
  validateCapacity(
    'computePettittTestOnCPU',
    'sliceCount',
    sliceCount,
    1,
    CPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT
  );
  if (values.length !== cellCount * sliceCount) {
    throw new Error('computePettittTestOnCPU values length must equal cellCount * sliceCount');
  }
  const statistic = new Uint32Array(cellCount);
  const changeSlice = new Uint32Array(cellCount).fill(GPU_PETTITT_TEST_NO_CHANGE_SLICE);
  const pValues = new Float32Array(cellCount).fill(1);
  for (let cell = 0; cell < cellCount; cell++) {
    const positions: number[] = [];
    const series: number[] = [];
    for (let slice = 0; slice < sliceCount; slice++) {
      const value =
        values[layout === 'cell-major' ? cell * sliceCount + slice : slice * cellCount + cell];
      if (Number.isFinite(value)) {
        positions.push(slice);
        series.push(value);
      }
    }
    const length = series.length;
    if (length < 2) continue;
    const order = Array.from({length}, (_, row) => row).sort(
      (first, second) => series[first] - series[second]
    );
    const ranks = new Float64Array(length);
    for (let start = 0; start < length; ) {
      let end = start + 1;
      while (end < length && series[order[end]] === series[order[start]]) end++;
      const averageRank = (start + 1 + end) / 2;
      for (let index = start; index < end; index++) ranks[order[index]] = averageRank;
      start = end;
    }
    let rankTotal = 0;
    let largest = 0;
    let largestRow = 0;
    for (let row = 0; row + 1 < length; row++) {
      rankTotal += ranks[row];
      const magnitude = Math.abs(2 * rankTotal - (row + 1) * (length + 1));
      if (magnitude > largest) {
        largest = magnitude;
        largestRow = row;
      }
    }
    statistic[cell] = largest;
    changeSlice[cell] = positions[largestRow + 1];
    pValues[cell] = Math.min(
      1,
      2 * Math.exp((-6 * largest * largest) / (length ** 3 + length ** 2))
    );
  }
  return {statistic, changeSlice, pValues};
}

/** CPU twin; the GPU's 512-slice private-array limit is replaced by the uint32 output limit. */
export class CPUPettittTestOperation extends GPUCPUOperation {
  constructor(props: PettittTestOperationProps) {
    const id = props.id ?? 'pettitt-test';
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(id, 'sliceCount', props.sliceCount, 1, CPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT);
    const shape = {length: props.cellCount};
    // Reuse the graph-port shape and disjointness checks; this allocates no GPU resources.
    new GPUPettittTestOperation(props);
    super({
      id,
      inputs: {values: props.values},
      outputs: {
        [props.output.statistic]: gpuGraph.vector('uint32', shape),
        [props.output.changeSlice]: gpuGraph.vector('uint32', shape),
        [props.output.pValues]: gpuGraph.vector('float32', shape)
      },
      execute({inputs}) {
        const result = computePettittTestOnCPU(
          getCPUOperationFloat32Rows(id, inputs, 'values'),
          props
        );
        return {
          [props.output.statistic]: result.statistic,
          [props.output.changeSlice]: result.changeSlice,
          [props.output.pValues]: result.pValues
        };
      }
    });
  }
}
