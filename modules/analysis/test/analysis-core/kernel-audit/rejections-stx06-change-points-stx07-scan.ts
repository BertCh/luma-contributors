// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

const BINDING_BYTES = 134217728;

/** Expected declaration refusals of the STX-06 change point and STX-07 scan cases. */
export const STX06_CHANGE_POINTS_STX07_SCAN_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPULocationChangePoints',
    name: 'one cell above the cube binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: location-change-points values chunk 0 needs 134219776 bytes in one binding \(limit 134217728\); reduce the chunk row count or output capacity$/
  },
  {
    exportName: 'GPUPettittTest',
    name: 'one cell above the cube binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: pettitt-test values chunk 0 needs 134219776 bytes in one binding \(limit 134217728\); reduce the chunk row count or output capacity$/
  },
  {
    exportName: 'GPUExpectationBasedSpaceTimeScan',
    name: 'one zone above the count table limit',
    kind: 'binding',
    limit: 16777216,
    rejects:
      /^Error: expectation-based-scan zones \* maximumWindowLength received 16777248; must be at most 16777216; reduce zones or maximumWindowLength$/
  }
];
