// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {validateGPUComputationRecord} from '@luma-contributors/analysis/computation-record';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {
  describeNetworkOptimizationCoverage,
  type NetworkOptimizationKind
} from '../../src/computation-lenses/scenarios/gpu-network/optimization-coverage.lens';
import {COVERAGE_NETWORK_MOVEMENT_CASES} from '@luma-contributors/analysis/test/analysis-core/kernel-audit/cases-coverage-network-movement';
import {auditKernelCase} from '@luma-contributors/analysis/test/analysis-core/kernel-audit/kernel-audit-runner';

for (const auditCase of COVERAGE_NETWORK_MOVEMENT_CASES) {
  it(`bounded coverage maximal shader audit: ${auditCase.exportName} ${auditCase.name}`, () => {
    const result = auditKernelCase(auditCase);
    expect(result.findings).toEqual([]);
    expect(result.nodeCount).toBeGreaterThan(0);
  });
}
for (const kind of [
  'point-routing',
  'flow',
  'louvain',
  'facility-optimization',
  'time-routing',
  'rational-access'
] as const satisfies readonly NetworkOptimizationKind[]) {
  it(`network ${kind} lens builds a valid executable graph description`, async () => {
    const device = createNullWebGPUDevice();
    try {
      const record = await describeNetworkOptimizationCoverage(device, kind);
      expect(validateGPUComputationRecord(record)).toEqual({valid: true, issues: []});
      expect(record.graphs.length).toBeGreaterThan(0);
    } finally {
      device.destroy();
    }
  });
}
