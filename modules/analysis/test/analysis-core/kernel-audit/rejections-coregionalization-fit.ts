// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Reference-array and dispatch-budget refusal companions of the maximal LMC fitter. */
export const COREGIONALIZATION_FIT_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUCoregionalizationFit',
    name: 'variableCount-above-maximum',
    kind: 'binding',
    limit: 4,
    rejects: /^Error: coregionalization-fit variableCount must be an integer in \[2, 4\]$/
  },
  {
    exportName: 'GPUCoregionalizationFit',
    name: 'structureCount-above-maximum',
    kind: 'binding',
    limit: 4,
    rejects: /^Error: coregionalization-fit structureCount must be an integer in \[1, 4\]$/
  },
  {
    exportName: 'GPUCoregionalizationFit',
    name: 'maximumIterations-above-maximum',
    kind: 'dispatch',
    limit: 1024,
    rejects: /^Error: coregionalization-fit maximumIterations must be an integer in \[0, 1024\]$/
  },
  {
    exportName: 'GPUCoregionalizationFit',
    name: 'distances.length-above-maximum',
    kind: 'binding',
    limit: 256,
    rejects: /^Error: coregionalization-fit distances.length must be an integer in \[0, 256\]$/
  }
];
