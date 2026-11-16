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
import {
  getMultipleTestingAdjustedPValues,
  GPU_MULTIPLE_TESTING_MAXIMUM_ROWS
} from './gpu-multiple-testing-adjustment';
import type {MultipleTestingAdjustmentOperationProps} from './gpu-multiple-testing-adjustment-operation';

/**
 * Calls the public float64 adjustment and rounds only the output. testedCount is the family size,
 * including minimumFamilySize, even for method none.
 *
 * What differs from the GPU contributor:
 * - Float64 arithmetic replaces float32 intermediate arithmetic; compare with relative tolerance 2e-6.
 * - Benjamini-Yekutieli sums the full harmonic series instead of the GPU approximation above 64.
 * - The GPU sort row and minimumFamilySize limits at 2^31 are dropped; the uint32 family-size
 *   layout limit remains. Above the public helper's minimumFamilySize cap, virtual p=1 rows pin
 *   the family instead. That path allocates and sorts the padded family inside the public helper.
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
  let adjusted: Float64Array;
  if (method === 'none') {
    adjusted = getMultipleTestingAdjustedPValues(pValues, method);
  } else if (minimumFamilySize < GPU_MULTIPLE_TESTING_MAXIMUM_ROWS) {
    adjusted = getMultipleTestingAdjustedPValues(pValues, method, minimumFamilySize);
  } else {
    // The public helper accepts larger row counts but caps its minimumFamilySize argument.
    // Its documented equivalent is to append p=1 rows, then discard their outputs. A read-only
    // virtual input avoids allocating another input column; the helper still owns its full result.
    const paddedValues: ArrayLike<number> = new Proxy(
      {length: pValues.length + familySize - tested},
      {
        get(target, property) {
          if (property === 'length') return target.length;
          const row = Number(property);
          return row < pValues.length ? pValues[row] : 1;
        }
      }
    );
    adjusted = getMultipleTestingAdjustedPValues(paddedValues, method).subarray(0, pValues.length);
  }
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
