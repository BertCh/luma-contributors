// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUVariableStandardization,
  type GPUVariableStandardizationMethod
} from '../../../src/gpu-dataframe/variable-standardization';
import type {KernelAuditCase} from './kernel-audit-context';

const METHODS: GPUVariableStandardizationMethod[] = [
  'deviation-from-mean',
  'z-score',
  'mad',
  'range-adjust',
  'range-standardize'
];

/** Maximum-size masked recipe variants; all reduction passes use shared core kernels. */
export const VARIABLE_STANDARDIZATION_AUDIT_CASES: readonly KernelAuditCase[] = METHODS.flatMap(
  method =>
    (['sample', 'population'] as const).map(variance => ({
      exportName: 'GPUVariableStandardization',
      name: `${method}-${variance}-maximum-masked`,
      build: ({graph, view}) => {
        const length = 2 ** 24;
        graph.add(
          new GPUVariableStandardization({
            values: view('float32', length),
            mask: view('uint32', length),
            method,
            variance,
            output: view('float32', length)
          })
        );
      }
    }))
);
