// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Core WebGPU permits at most 128 MiB in each storage-buffer binding. */
const CORE_STORAGE_BINDING_BYTE_LIMIT = 134217728;

/** Expected declaration refusals of the statistics cases. */
export const STATISTICS_REJECTIONS: readonly KernelAuditRejection[] = [
  ...[false, true].flatMap(includeGramTraces => [
    {
      exportName: 'GPUSpatialWeightsTraces',
      name: `exact gram ${includeGramTraces} next batch above the command limit`,
      kind: 'dispatch' as const,
      limit: 32768,
      rejects: /^Error: spatial-weights-traces estimated command node count must be at most 32768$/
    },
    {
      exportName: 'GPUSpatialWeightsTraces',
      name: `exact gram ${includeGramTraces} binding-limited rows exceed the command limit`,
      kind: 'dispatch' as const,
      limit: 32768,
      rejects: /^Error: spatial-weights-traces estimated command node count must be at most 32768$/
    }
  ]),
  ...['none', 'bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli'].map(method => ({
    exportName: 'GPUMultipleTestingAdjustment',
    name: `${method} one row above the binding limit`,
    kind: 'binding' as const,
    limit: CORE_STORAGE_BINDING_BYTE_LIMIT,
    rejects:
      /^Error: multiple-testing-adjustment binding 0 chunk 0 needs 134217732 bytes in one binding \(limit 134217728\); reduce the chunk row count or output capacity$/
  })),
  {
    exportName: 'GPUGlobalPermutationTest',
    name: 'moran permutations x row blocks above the maximum',
    kind: 'dispatch',
    limit: 16777215,
    rejects: /^Error: global-moran permutations x row blocks must be at most 16777215$/
  },
  {
    exportName: 'GPUGlobalPermutationTest',
    name: 'geary permutations x row blocks above the maximum',
    kind: 'dispatch',
    limit: 16777215,
    rejects: /^Error: global-geary permutations x row blocks must be at most 16777215$/
  },
  {
    exportName: 'GPUGlobalPermutationTest',
    name: 'getisOrdG permutations x row blocks above the maximum',
    kind: 'dispatch',
    limit: 16777215,
    rejects: /^Error: global-getisOrdG permutations x row blocks must be at most 16777215$/
  },
  {
    exportName: 'GPUGlobalPermutationTest',
    name: 'bivariateMoran permutations x row blocks above the maximum',
    kind: 'dispatch',
    limit: 16777215,
    rejects: /^Error: global-bivariateMoran permutations x row blocks must be at most 16777215$/
  },
  {
    exportName: 'GPUGlobalPermutationTest',
    name: 'leesL permutations x row blocks above the maximum',
    kind: 'dispatch',
    limit: 16777215,
    rejects: /^Error: global-leesL permutations x row blocks must be at most 16777215$/
  },
  {
    exportName: 'GPUSpatialRegressionDiagnostics',
    name: 'restandardize false rows above the maximum',
    kind: 'binding',
    limit: 33554432,
    rejects:
      /^Error: spatial-regression-diagnostics rows x \(predictorCount \+ 3\) must be at most 33554432$/
  },
  {
    exportName: 'GPUSpatialRegressionDiagnostics',
    name: 'restandardize true rows above the maximum',
    kind: 'binding',
    limit: 33554432,
    rejects:
      /^Error: spatial-regression-diagnostics rows x \(predictorCount \+ 3\) must be at most 33554432$/
  }
];
