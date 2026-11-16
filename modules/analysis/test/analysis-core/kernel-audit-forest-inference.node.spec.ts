// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  FOREST_INFERENCE_AUDIT_CASES,
  FOREST_INFERENCE_EMPTY_AUDIT_CASE
} from './kernel-audit/cases-forest-inference';
import {auditKernelCase} from './kernel-audit/kernel-audit-runner';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('forest-inference', FOREST_INFERENCE_AUDIT_CASES);

it('audits empty forest input as a valid no-op without nodes or shaders', () => {
  expect(auditKernelCase(FOREST_INFERENCE_EMPTY_AUDIT_CASE)).toEqual({
    findings: [],
    nodeCount: 0,
    moduleCount: 0,
    rejected: false
  });
});
