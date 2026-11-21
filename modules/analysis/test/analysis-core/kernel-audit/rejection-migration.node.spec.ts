// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {KERNEL_AUDIT_CASES} from './kernel-audit-cases';
import {KERNEL_AUDIT_REJECTIONS} from './kernel-audit-rejections';
import {KERNEL_AUDIT_REJECTION_CONTRACTS} from './kernel-audit-rejection-contracts';
import {auditKernelCase, formatKernelAuditFindings} from './kernel-audit-runner';

it('VAL-12/13 has an exact reviewed phase and structured contract for every rejection', () => {
  const labels = KERNEL_AUDIT_REJECTIONS.map(
    rejection => `${rejection.exportName} ${rejection.name}`
  );
  expect(Object.keys(KERNEL_AUDIT_REJECTION_CONTRACTS).sort()).toEqual(labels.sort());
  // The four exact-trace command-budget refusals were reviewed after the original migration.
  const reviewedTraceLabels = [
    'GPUSpatialWeightsTraces exact gram false next batch above the command limit',
    'GPUSpatialWeightsTraces exact gram false binding-limited rows exceed the command limit',
    'GPUSpatialWeightsTraces exact gram true next batch above the command limit',
    'GPUSpatialWeightsTraces exact gram true binding-limited rows exceed the command limit'
  ];
  const structuredLabels = KERNEL_AUDIT_REJECTIONS.filter(rejection => rejection.validation).map(
    rejection => `${rejection.exportName} ${rejection.name}`
  );
  expect(structuredLabels).toEqual(expect.arrayContaining(reviewedTraceLabels));
  expect(structuredLabels.filter(label => !reviewedTraceLabels.includes(label))).toHaveLength(205);
});

const cases = Object.values(KERNEL_AUDIT_CASES).flat();
for (const rejection of KERNEL_AUDIT_REJECTIONS) {
  it(`VAL-12/13 ${rejection.exportName} ${rejection.name}`, () => {
    const auditCase = cases.find(
      entry => entry.exportName === rejection.exportName && entry.name === rejection.name
    )!;
    const result = auditKernelCase(auditCase);
    expect(result.rejectionObservation?.phase, formatKernelAuditFindings(result.findings)).toBe(
      rejection.phase
    );
    expect(result.findings).toEqual([]);
    expect(result.rejected).toBe(true);
    expect(result.rejectionObservation!.after).toEqual(result.rejectionObservation!.before);
  });
}
