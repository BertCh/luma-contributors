// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** The default `maxStorageBufferBindingSize` in bytes, which the message of the byte-length guard names. */
const BINDING_BYTES = 134217728;

/** Expected declaration refusals of the value cube cases. */
export const VALUE_CUBE_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUValueCube',
    name: 'one bin above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: value-cube fixed-point sums needs 134217732 bytes in one binding \(limit 134217728\); reduce cellCount \* periodCount$/
  },
  {
    exportName: 'GPUCellPeriodKeys',
    name: 'one row above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: cell-period-keys cells needs 134217736 bytes in one binding \(limit 134217728\); split the rows over several graphs$/
  }
];
