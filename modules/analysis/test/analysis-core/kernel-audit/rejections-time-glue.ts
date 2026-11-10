// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** The default `maxStorageBufferBindingSize` in bytes, which the message of the byte-length guard names. */
const BINDING_BYTES = 134217728;

/** Expected declaration refusals of the time glue cases. */
export const TIME_GLUE_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUPeriodIndex',
    name: 'calendar hour step, one row above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: period-index output.periods needs 134217732 bytes in one binding \(limit 134217728\); split the rows over several graphs$/
  },
  {
    exportName: 'GPUPeriodIndex',
    name: 'fixed width, one row above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: period-index timestamps needs 134217736 bytes in one binding \(limit 134217728\); split the rows over several graphs$/
  },
  {
    exportName: 'GPUCubeLayoutTranspose',
    name: 'cube one period above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: cube-layout-transpose cube needs 134234112 bytes in one binding \(limit 134217728\); reduce cellCount \* periodCount$/
  },
  {
    exportName: 'GPUTimeWordsRelative',
    name: 'one row above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: time-words-relative timestamps needs 134217736 bytes in one binding \(limit 134217728\); convert fewer rows per graph$/
  }
];
