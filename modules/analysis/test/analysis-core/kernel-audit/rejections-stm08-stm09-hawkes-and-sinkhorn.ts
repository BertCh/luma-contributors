// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the STM-08 and STM-09 cases. */
export const STM08_STM09_HAWKES_AND_SINKHORN_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUSpaceTimeHawkes',
    name: 'one event above the row limit',
    kind: 'binding',
    limit: 8388608,
    rejects: /^Error: space-time-hawkes positions length must be at most 8388608$/
  },
  {
    exportName: 'GPUSinkhornDivergence',
    name: 'one cell above the grid side limit',
    kind: 'dispatch',
    limit: 2048,
    rejects: /^Error: sinkhorn-divergence grid\.width and grid\.height must be at most 2048$/
  }
];
