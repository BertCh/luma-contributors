// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {DATAFRAME_CASES} from './kernel-audit/cases-dataframe';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('dataframe', DATAFRAME_CASES);

// W3-03: source capture and parsing alone cannot establish device module validity.
// Audit every generated module; the lead also runs representative joins on a real device.
import {expect, it} from 'vitest';
import {
  createKernelAuditContext,
  getDeclaredNodes,
  recordShaderModules
} from './kernel-audit/kernel-audit-context';
import {auditWGSLModule} from './kernel-audit/kernel-audit-wgsl';

const joinVariants = DATAFRAME_CASES.filter(
  auditCase =>
    auditCase.exportName === 'GPUKeyLookup' &&
    /every option|rows at the binding limit/.test(auditCase.name)
);
it('W3-03 retains all six join module variants', () => {
  expect(joinVariants).toHaveLength(6);
});
for (const auditCase of joinVariants) {
  it(`W3-03 resolves every scale, accumulation and publication call: ${auditCase.name}`, () => {
    const context = createKernelAuditContext(auditCase.exportName);
    try {
      auditCase.build(context);
      const stages = new Set<string>();
      let moduleCount = 0;
      recordShaderModules(getDeclaredNodes(context), module => {
        moduleCount++;
        for (const stage of ['sum-scale', 'accumulate', 'finish'])
          if (module.nodeId.endsWith(stage)) stages.add(stage);
        expect(
          auditWGSLModule(module.source, module.constants, module.declaredBindings).filter(
            finding => finding.check === 'undeclared call'
          ),
          module.nodeId
        ).toEqual([]);
      });
      expect(moduleCount).toBeGreaterThan(0);
      expect([...stages].sort()).toEqual(['accumulate', 'finish', 'sum-scale']);
    } finally {
      context.device.destroy();
    }
  });
}
