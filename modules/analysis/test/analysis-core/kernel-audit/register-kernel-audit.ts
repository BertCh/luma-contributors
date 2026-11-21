// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import type {KernelAuditCase} from './kernel-audit-context';
import {selectAnalysisCases} from './analysis-contributors';
import {
  auditKernelCase,
  formatKernelAuditFindings,
  getKernelAuditResultFailures
} from './kernel-audit-runner';

/** Registers only one area's builds, so Vitest can schedule audit work across spec files. */
export function registerKernelAudit(
  area: string,
  cases: readonly KernelAuditCase[],
  registration: {
    describe: (name: string, define: () => void) => unknown;
    it: (name: string, run: () => void) => unknown;
  } = {describe, it}
): void {
  registration.describe(`kernel audit: ${area}`, () => {
    for (const auditCase of selectAnalysisCases(cases)) {
      registration.it(`${auditCase.exportName} ${auditCase.name}`, () => {
        const result = auditKernelCase(auditCase);
        const failures = getKernelAuditResultFailures(result);
        expect(failures, formatKernelAuditFindings(failures)).toEqual([]);
        if (!result.rejected) {
          expect(result.nodeCount, 'declared command nodes').toBeGreaterThan(0);
        }
      });
    }
  });
}
