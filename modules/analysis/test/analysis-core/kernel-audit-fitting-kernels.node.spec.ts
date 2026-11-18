// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {FITTING_KERNEL_AUDIT_CASES} from './kernel-audit/cases-fitting-kernels';
import {auditKernelCase, formatKernelAuditFindings} from './kernel-audit/kernel-audit-runner';

// The library-wide discovery includes core; retain this focused suite for fitting-kernel changes.
describe('kernel audit: shared fitting kernels', () => {
  for (const auditCase of FITTING_KERNEL_AUDIT_CASES) {
    it(`${auditCase.exportName} ${auditCase.name}`, () => {
      const result = auditKernelCase(auditCase);
      expect(result.findings, formatKernelAuditFindings(result.findings)).toEqual([]);
      expect(result.nodeCount).toBeGreaterThan(0);
    });
  }
});
