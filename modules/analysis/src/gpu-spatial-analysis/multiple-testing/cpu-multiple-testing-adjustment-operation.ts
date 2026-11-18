// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import type {GPUMultipleTestingMethod} from '../../analysis-core/contracts/index';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {getMultipleTestingAdjustedPValues} from './gpu-multiple-testing-adjustment';
import type {MultipleTestingAdjustmentOperationProps} from './gpu-multiple-testing-adjustment-operation';

/**
 * Calls the public float64 adjustment and rounds only the output. testedCount is the family size,
 * including minimumFamilySize, even for method none.
 *
 * What differs from the GPU contributor:
 * - Float64 arithmetic replaces float32 intermediate arithmetic; compare with relative tolerance 2e-6.
 * - Benjamini-Yekutieli uses a binary64 Euler-Maclaurin expansion above 64 terms.
 * - The GPU sort row and minimumFamilySize limits at 2^31 are dropped; the uint32 family-size
 *   layout limit remains. The logical family size does not allocate virtual rows: only supplied
 *   p-values are stored and sorted, and the harmonic factor takes bounded work.
 * - Subnormal p-values may be flushed to zero on the GPU and are kept by the twin.
 */
export function computeMultipleTestingAdjustmentOnCPU(
  pValues: Float32Array,
  method: GPUMultipleTestingMethod,
  minimumFamilySize = 0
): {adjustedPValues: Float32Array; testedCount: Uint32Array} {
  validateCapacity(
    'computeMultipleTestingAdjustmentOnCPU',
    'minimumFamilySize',
    minimumFamilySize,
    0
  );
  let tested = 0;
  for (const value of pValues) {
    if (Number.isFinite(value) && value >= 0 && value <= 1) {
      tested++;
    }
  }
  const familySize = Math.max(tested, minimumFamilySize);
  const adjusted = getMultipleTestingAdjustedPValues(pValues, method, minimumFamilySize);
  const adjustedPValues = Float32Array.from(adjusted, value => (value === 0 ? 0 : value));
  return {adjustedPValues, testedCount: Uint32Array.of(familySize)};
}

/** CPU twin. The GPU sort row-count cap is dropped; uint32 output sizes remain. */
export class CPUMultipleTestingAdjustmentOperation extends GPUCPUOperation {
  constructor(props: MultipleTestingAdjustmentOperationProps) {
    const id = props.id ?? 'multiple-testing-adjustment';
    validateCapacity(id, 'rowCount', props.rowCount);
    validateCapacity(id, 'minimumFamilySize', props.minimumFamilySize ?? 0, 0);
    validateRequirement(
      id,
      ['none', 'bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli'].includes(
        props.method
      ),
      'method must name a supported multiple-testing method'
    );
    super({
      id,
      inputs: {pValues: props.pValues},
      outputs: {
        [props.output.adjustedPValues]: gpuGraph.vector('float32', {length: props.rowCount}),
        ...(props.output.testedCount
          ? {[props.output.testedCount]: gpuGraph.vector('uint32', {length: 1})}
          : {})
      },
      execute({inputs}) {
        const pValues = getCPUOperationFloat32Rows(id, inputs, 'pValues');
        if (pValues.length !== props.rowCount) {
          throw new Error(`${id} pValues length must equal rowCount`);
        }
        const result = computeMultipleTestingAdjustmentOnCPU(
          pValues,
          props.method,
          props.minimumFamilySize
        );
        return {
          [props.output.adjustedPValues]: result.adjustedPValues,
          ...(props.output.testedCount ? {[props.output.testedCount]: result.testedCount} : {})
        };
      }
    });
  }
}
