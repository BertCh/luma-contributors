// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {KERNEL_AUDIT_CASES} from './kernel-audit/kernel-audit-cases';
import {NETWORK_CENTRALITY_SCHEDULE_KERNEL_AUDIT_CASES} from '../gpu-network/network-centrality/network-centrality-kernel-audit-cases';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

it('includes the supplied public schedule-step cases in the shared registry', () => {
  for (const auditCase of NETWORK_CENTRALITY_SCHEDULE_KERNEL_AUDIT_CASES) {
    expect(KERNEL_AUDIT_CASES['network']).toContain(auditCase);
  }
});

registerKernelAudit('network-centrality-schedule', NETWORK_CENTRALITY_SCHEDULE_KERNEL_AUDIT_CASES);
