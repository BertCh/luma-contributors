// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {SCORING_RULES_AUDIT_CASES} from './kernel-audit/cases-scoring-rules';
import {auditKernelCase, formatKernelAuditFindings} from './kernel-audit/kernel-audit-runner';

// `registerKernelAudit` keeps only cases of contributors the analysis barrels export, and the
// scoring contributors are not in the `gpu-spatial-analysis` barrel yet, so these cases are
// audited directly. Move them to `KERNEL_AUDIT_CASES` once `export * from './scoring-rules/index'`
// is added to `src/gpu-spatial-analysis/index.ts`.
describe('kernel audit: scoring-rules', () => {
  for (const auditCase of SCORING_RULES_AUDIT_CASES) {
    it(`${auditCase.exportName} ${auditCase.name}`, () => {
      const result = auditKernelCase(auditCase);
      expect(result.findings, formatKernelAuditFindings(result.findings)).toEqual([]);
      expect(result.nodeCount, 'declared command nodes').toBeGreaterThan(0);
    });
  }
});
