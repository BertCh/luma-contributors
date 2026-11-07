// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {COVERAGE_STATISTICS_CASES} from './cases-coverage-statistics';
import {
  createKernelAuditContext,
  getDeclaredNodes,
  recordShaderModules
} from './kernel-audit-context';
import {auditWGSLModule} from './kernel-audit-wgsl';

it.each(COVERAGE_STATISTICS_CASES)('$exportName $name compiles maximal options', entry => {
  const context = createKernelAuditContext(`${entry.exportName}-${entry.name}`);
  try {
    entry.build(context);
    let modules = 0;
    recordShaderModules(getDeclaredNodes(context), module => {
      modules++;
      expect(auditWGSLModule(module.source, module.constants), module.nodeId).toEqual([]);
    });
    expect(modules).toBeGreaterThan(0);
  } finally {
    context.device.destroy();
  }
}, 120000);
