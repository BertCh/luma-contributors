// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import type {GPUMultipleTestingMethod} from '../../analysis-core/contracts/index';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {validateOperationGraphNames} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  getMultipleTestingAdjustedPValues,
  resolveGPUMultipleTestingMethod
} from './gpu-multiple-testing-adjustment';
import type {MultipleTestingAdjustmentOperationProps} from './gpu-multiple-testing-adjustment-operation';
import {
  MULTIPLE_TESTING_COUNT_OUTPUT_NAMES,
  validateMultipleTestingCountPorts
} from './multiple-testing-counts';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Scalar float32 adjusted rows and independent one-word uint32 family counts. */
export type CPUMultipleTestingAdjustmentResult = {
  adjustedPValues: Float32Array;
  /** Number of finite p-values in [0, 1], before padding. */
  finiteTestCount: Uint32Array;
  /** max(finiteTestCount, minimumFamilySize), the adjustment denominator. */
  effectiveFamilySize: Uint32Array;
};

/**
 * Calls the public float64 adjustment and rounds only the output. effectiveFamilySize is the family size,
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
): CPUMultipleTestingAdjustmentResult {
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
  return {
    adjustedPValues,
    finiteTestCount: Uint32Array.of(tested),
    effectiveFamilySize: Uint32Array.of(familySize)
  };
}

/** CPU twin. The GPU sort row-count cap is dropped; uint32 output sizes remain. */
export class CPUMultipleTestingAdjustmentOperation extends GPUCPUOperation {
  constructor(props: MultipleTestingAdjustmentOperationProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'pValues',
      'rowCount',
      'method',
      'minimumFamilySize',
      'output'
    ]);
    const id = props.id ?? 'multiple-testing-adjustment';
    validateMultipleTestingCountPorts(id, props.output);
    props = Object.freeze({
      ...props,
      output: snapshotConstructionFields(props.output, [
        'adjustedPValues',
        ...MULTIPLE_TESTING_COUNT_OUTPUT_NAMES
      ])
    });
    const countOutputs = MULTIPLE_TESTING_COUNT_OUTPUT_NAMES.flatMap(name =>
      props.output[name] !== undefined ? [[name, props.output[name]!] as const] : []
    );
    validateOperationGraphNames(
      id,
      {pValues: props.pValues},
      {
        adjustedPValues: props.output.adjustedPValues,
        ...Object.fromEntries(countOutputs)
      }
    );
    validateCapacity(id, 'rowCount', props.rowCount);
    validateCapacity(id, 'minimumFamilySize', props.minimumFamilySize ?? 0, 0);
    const method = resolveGPUMultipleTestingMethod(id, props.method, true);
    super({
      id,
      inputs: {pValues: props.pValues},
      outputs: {
        [props.output.adjustedPValues]: gpuGraph.vector('float32', {length: props.rowCount}),
        ...Object.fromEntries(
          countOutputs.map(([, destination]) => [
            destination,
            gpuGraph.vector('uint32', {length: 1})
          ])
        )
      },
      execute({inputs}) {
        const pValues = getCPUOperationFloat32Rows(id, inputs, 'pValues');
        if (pValues.length !== props.rowCount) {
          throw new Error(`${id} pValues length must equal rowCount`);
        }
        const result = computeMultipleTestingAdjustmentOnCPU(
          pValues,
          method,
          props.minimumFamilySize
        );
        return {
          [props.output.adjustedPValues]: result.adjustedPValues,
          ...Object.fromEntries(
            countOutputs.map(([name, destination]) => [destination, result[name]])
          )
        };
      }
    });
  }
}
