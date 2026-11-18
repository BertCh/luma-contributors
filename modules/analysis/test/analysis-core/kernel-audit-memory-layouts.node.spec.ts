// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it, vi} from 'vitest';
import {KERNEL_AUDIT_CASES} from './kernel-audit/kernel-audit-cases';
import {auditKernelCase, formatKernelAuditFindings} from './kernel-audit/kernel-audit-runner';

vi.setConfig({testTimeout: 120_000});

describe('registered kernel memory layouts', () => {
  it('has a nonempty case registry in every area', () => {
    expect(Object.keys(KERNEL_AUDIT_CASES).length).toBeGreaterThan(0);
    for (const [area, cases] of Object.entries(KERNEL_AUDIT_CASES)) {
      expect(cases.length, area).toBeGreaterThan(0);
    }
  });

  // Enumerate the registry directly: a separately maintained area-spec list or contributor
  // filter must not silently remove a registered configuration from memory-layout coverage.
  for (const [area, cases] of Object.entries(KERNEL_AUDIT_CASES)) {
    for (const auditCase of cases) {
      it(`${area}: ${auditCase.exportName} ${auditCase.name}`, () => {
        const result = auditKernelCase(auditCase);
        const blocking = result.findings.filter(finding =>
          [
            'unresolved memory layout',
            'build throws',
            'compile throws',
            'expected rejection',
            'unexpected refusal',
            'refusal names no documented limit',
            'generic refusal'
          ].includes(finding.check)
        );
        expect(blocking, formatKernelAuditFindings(blocking)).toEqual([]);
        if (!result.rejected) {
          expect(result.moduleCount, 'recorded shader modules').toBeGreaterThan(0);
        }
      });
    }
  }
});
